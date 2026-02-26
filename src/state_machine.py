"""!
State machine for the RX200 armlab.

States:
    idle              — waiting for input
    initialize_rxarm  — homes the arm and enables torque
    manual            — direct slider control
    estop             — emergency stop (disables torque)
    execute           — plays back the default waypoint list
    calibrate         — AprilTag-based camera extrinsic calibration
    click_to_grab     — click a block to pick, click again to place
    teach_play        — replay taught waypoints (see record_waypoints)
    challenge_one     — sort blocks by size (L1: spread, L2: stack rainbow)
    challenge_two     — arrange blocks in rainbow-order horizontal lines
    challenge_three   — stack blocks as high as possible
    auto_L3           — alternative auto sort/stack (level 3)
"""
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot, QTimer
import time
import numpy as np
import cv2
from kinematics import IK_geometric, FK_dh, IK_geometric_stack, find_feasible_ik, check_joint_limits, compute_best_psi, compute_paired_psi
class StateMachine():
    """!
    @brief      State machine implementing arm control logic for all three challenges,
                click-to-grab, calibration, and autonomous sort/stack operations.
    """

    def __init__(self, rxarm, camera):
        """!
        @brief      Constructs a new instance.

        @param      rxarm   The rxarm
        @param      planner  The planner
        @param      camera   The camera
        """
        self.rxarm = rxarm
        self.camera = camera
        self.status_message = "State: Idle"
        self.current_state = "idle"
        self.next_state = "idle"
        self.waypoints = [
            [-np.pi/2,       -0.5,      -0.3,          0.0,        0.0],
            [0.75*-np.pi/2,   0.5,       0.3,     -np.pi/3,    np.pi/2],
            [0.5*-np.pi/2,   -0.5,      -0.3,      np.pi/2,        0.0],
            [0.25*-np.pi/2,   0.5,       0.3,     -np.pi/3,    np.pi/2],
            [0.0,             0.0,       0.0,          0.0,        0.0],
            [0.25*np.pi/2,   -0.5,      -0.3,          0.0,    np.pi/2],
            [0.5*np.pi/2,     0.5,       0.3,     -np.pi/3,        0.0],
            [0.75*np.pi/2,   -0.5,      -0.3,          0.0,    np.pi/2],
            [np.pi/2,         0.5,       0.3,     -np.pi/3,        0.0],
            [0.0,             0.0,       0.0,          0.0,        0.0]]
        self.current_gripper_state = 'open'

        # ── Teach & Play (record waypoints, then replay them) ──
        # Usage: Wire buttons in control_station.py to record_waypoints() / clear_waypoints(),
        #        then set next_state to "teach_play" to replay. See control_station.py for example.
        self.taught_waypoints = []

        # ── Click-to-grab state ──
        self.holding_object = False          # True after a successful pick, False after drop
        self.last_pick_mm = None             # stores last picked [x,y,z] in mm
        self.z_grab_threshold = 50
        self.gripper_opened = True
        self.drop_height = None
        # ── Auto sort/stack parameters ──
        self.COLOR_ORDER = "roygbv"
        self.BLOCK_H = {"l": 21, "s": 10}     # mm
        self.NUM_TOWER_BLOCKS = 3                 # L2/L3

        # drop zones: negative y (required). small left, large right
        self.DROP_XY = {
            "s": np.array([-160.0, 100.0], dtype=float),
            "l": np.array([ 160.0, 100.0], dtype=float),
        }
        self.DROP_Z0 = 15.0  # base z at table (tune)
        self.z_psi_threshold = 270 

        self.tower_h = {"s": 0.0, "l": 0.0}   # running stacking height per size
        self.placed_set = set()                 # tracks (size, color) already placed


    def set_next_state(self, state):
        """!
        @brief      Sets the next state.

            This is in a different thread than run so we do nothing here and let run handle it on the next iteration.

        @param      state  a string representing the next state.
        """
        self.next_state = state

    def run(self):
        """!
        @brief      Run the logic for the next state (called in a loop by StateMachineThread).
        """
        if self.next_state == "initialize_rxarm":
            self.initialize_rxarm()
        if self.next_state == "idle":
            self.idle()
        if self.next_state == "estop":
            self.estop()
        if self.next_state == "execute":
            self.execute()
        if self.next_state == "calibrate":
            self.calibrate()
        if self.next_state == "manual":
            self.manual()
        if self.next_state == "click_to_grab":
            self.click_to_grab()
        if self.next_state == "teach_play":
            self.teach_play()
        if self.next_state == "challenge_one":
            self.challenge_one()
        if self.next_state == "auto_L3":
            self.auto_sort_stack(level=3)
        if self.next_state == "challenge_two":
            self.challenge_two()
        if self.next_state == "challenge_three":
            self.challenge_three()

    def _move_xyz(self, xyz_mm, block_angle_deg=0.0, wait=1.5, slow=True):
        """Move the end-effector to a world-frame position using IK."""
        if slow:
            self.rxarm.moving_time = 3.5
            self.rxarm.accel_time = 3.0

        xyz_mm = np.asarray(xyz_mm, dtype=float).reshape(3)
        q = None

        try:
            q_try, psi = find_feasible_ik(xyz_mm, float(block_angle_deg), -np.pi/2)
            if q_try is not None:
                q = np.asarray(q_try, dtype=float).reshape(-1)
        except Exception:
            pass

        if q is None:
            q_try = IK_geometric(xyz_mm, float(block_angle_deg))
            if q_try is None:
                return False
            q = np.asarray(q_try, dtype=float).reshape(-1)

        self.rxarm.set_positions(q)
        time.sleep(wait)
        return True

    def goto_observe_pose(self, wait=1.2):
        # raise arm to avoid blocking camera; tune (x,y,z) if needed
        q= np.array([0.0, 0.0, -1.57, 0.0, 0.0])
        self.rxarm.set_positions(q)
        time.sleep(wait)

    # ---------------------------
    # perception helpers
    # ---------------------------


    def _color_to_letter(self, color_name: str) -> str:
        m = {"red":"r","orange":"o","yellow":"y","green":"g","blue":"b","purple":"v","violet":"v"}
        return m.get((color_name or "").lower(), "?")

    def _classify_size_key(self, det: dict) -> str:
        area = float(det.get("rect_area", 0.0))
        return "l" if area >= 1200.0 else "s"

    def _build_block_dict(self, level: int):
        self.camera.blockDetector()
        dets = getattr(self.camera, "block_detections", []) or []

        block_dict = {
            "l": {c: [] for c in "roygbv"},
            "s": {c: [] for c in "roygbv"},
        }

        for d in dets:
            # 1) color
            c_letter = self._color_to_letter(d.get("color", "unknown"))
            if c_letter not in "roygbv":
                continue

            # 2) size
            sz = self._classify_size_key(d)

            # 3) pixel -> world
            cx, cy = d.get("center_px", (None, None))
            if cx is None or cy is None:
                continue
            xyz = self.camera.pixel_to_world(int(cx), int(cy))
            if xyz is None:
                continue
            xyz = np.asarray(xyz[:3], dtype=float)

            angle_deg = float(d.get("angle_deg", 0.0))

            block_dict[sz][c_letter].append({
                "xyz": xyz,
                "angle_deg": angle_deg,
                "det": d
            })
        return block_dict

    def _pickup_targets(self, xyz, sz):
        """
        returns (hover_xyz, pick_xyz)
        hover: above block
        pick: near surface
        """
        xyz = np.asarray(xyz, dtype=float).reshape(3)

        hover_xyz = xyz.copy()
        hover_xyz[2] += 300.0          # hover above

        pick_xyz = xyz.copy()
        pick_xyz[2] -=9.0          # grab offset (tune)
        return hover_xyz, pick_xyz

    def _dropoff_targets(self, sz, level, place_idx):
        """
        returns (hover_xyz, drop_xyz)
        - level 1: spread blocks (no stacking)
        - level 2/3: stack at same xy, increase z using tower_h
        """
        base_xy = self.DROP_XY[sz].copy()

        if level == 1:

            sign = -1.0 if sz == "s" else 1.0
            base_xy[0] = base_xy[0] + sign * 80.0 * place_idx
            z = self.DROP_Z0
        else:

            z = self.DROP_Z0 + self.tower_h[sz]

        drop_xyz = np.array([base_xy[0], base_xy[1], z], dtype=float)
        hover_xyz = drop_xyz.copy()
        hover_xyz[2] += 450
        return hover_xyz, drop_xyz



    # ---------------------------
    # pick & place primitives
    # ---------------------------
    def _select_next_block_like_them(self, block_dict):
        """
        returns (sz, c, cand) or (None,None,None)
        sz in {'s','l'}, c in 'roygbv', cand has xyz + angle_deg
        """
        for sz in ["s", "l"]:  # you can swap order if you want large first
            for c in self.COLOR_ORDER:
                key = (sz, c)
                if key in self.placed_set:
                    continue
                lst = block_dict.get(sz, {}).get(c, [])
                if not lst:
                    continue
                return sz, c, lst[0]
        return None, None, None

    def auto_sort_stack(self, level=3):
        """Autonomous sort and stack: L1 spread, L2 stack by size, L3 unstack-then-stack."""
        self.current_state = f"auto_L{level}"
        t0 = time.time()
        self.goto_observe_pose(wait=2.0)

        self.tower_h = {"s": 0.0, "l": 0.0}
        self.placed_set = set()
        place_idx = {"s": 0, "l": 0}
        do_stack = (level >= 2)
        cached_block_dict = None

        Z_TWO_HIGH = 55.0
        dump_i = 0

        # Level 3: unstack any blocks sitting on top of others first
        if level >= 3:
            cached_block_dict = self._build_block_dict(level)
            highs = []
            for sz in ["s", "l"]:
                for c in self.COLOR_ORDER:
                    for cand in cached_block_dict[sz][c]:
                        zt = float(cand.get("z_top", cand["xyz"][2]))
                        if zt > Z_TWO_HIGH:
                            highs.append((zt, cand, sz))

            for i in range(len(highs)):
                highs.sort(key=lambda t: -t[0])
                cand = highs[i][1]
                sz = highs[i][2]
                xyz = cand["xyz"]
                ang = cand["angle_deg"]

                ho_pick, pick = self._pickup_targets(xyz, sz)
                self._move_xyz(ho_pick, ang, wait=1.0, slow=True)
                self._move_xyz(pick, ang, wait=2.0, slow=True)
                time.sleep(2.0)
                self.rxarm.gripper.grasp()
                time.sleep(1.0)
                self._move_xyz(ho_pick, ang, wait=2.0, slow=True)

                dump_xyz = np.array([0.0, 200.0, 5.0], dtype=float)
                dump_xyz[0] += 70.0 * dump_i
                dump_i += 1

                ho_drop = dump_xyz.copy()
                ho_drop[2] += 80.0
                self._move_xyz(ho_drop, 0.0, wait=1.0, slow=True)
                self._move_xyz(dump_xyz, 0.0, wait=2.0, slow=True)
                self.rxarm.gripper.release()
                time.sleep(1.0)
                self._move_xyz(ho_drop, 0.0, wait=2.0, slow=True)

        cached_block_dict = self._build_block_dict(level)

        while True:
            if time.time() - t0 > 175:
                break

            # refresh detection every loop (L3 needs this; L1/L2 also fine)
            block_dict = cached_block_dict if cached_block_dict is not None else self._build_block_dict(level)
            sz, c, cand = self._select_next_block_like_them(block_dict)
            if cand is None:
                break

            xyz = cand["xyz"]
            angle_deg = cand["angle_deg"]

            # ---------- PICK ----------
            ho_pick, pick = self._pickup_targets(xyz, sz)
            if not self._move_xyz(ho_pick, angle_deg, wait=1.0, slow=True):
                continue
            if not self._move_xyz(pick, angle_deg, wait=2.0, slow=True):
                continue
            time.sleep(1.0)
            self.rxarm.gripper.grasp()
            time.sleep(1.0)

            self._move_xyz(ho_pick, angle_deg, wait=1.0, slow=True)
            time.sleep(0.5)

            # ---------- PLACE ----------
            ho_drop, drop = self._dropoff_targets(sz, level, place_idx[sz])

            if not self._move_xyz(ho_drop, 0, wait=1.0, slow=True):
                self.rxarm.gripper.release()
                time.sleep(1.0)
                continue
            if not self._move_xyz(drop, 0, wait=2.0, slow=True):
                self.rxarm.gripper.release()
                time.sleep(1.0)
                continue
            time.sleep(1.0)
            self.rxarm.gripper.release()
            place_idx[sz] += 1
            if level >= 2:
                self.tower_h[sz] += self.BLOCK_H[sz]
            time.sleep(0.25)

            self._move_xyz(ho_drop, angle_deg, wait=1.0, slow=True)

            self.placed_set.add((sz, c))
            if do_stack:
                self.tower_h[sz] += self.BLOCK_H[sz]

            # done condition for L2/L3
            if level >= 2:
                cnt_s = sum(1 for (s, _) in self.placed_set if s == "s")
                cnt_l = sum(1 for (s, _) in self.placed_set if s == "l")
                if cnt_s >= self.NUM_TOWER_BLOCKS and cnt_l >= self.NUM_TOWER_BLOCKS:
                    break

        self.status_message = f"Auto L{level}: done in {time.time()-t0:.1f}s"
        self.next_state = "idle"

    # ─── Core state functions ───────────────────────────────────────────

    def manual(self):
        """!
        @brief      Manually control the rxarm
        """
        self.status_message = "State: Manual - Use sliders to control arm"
        self.current_state = "manual"

    def idle(self):
        """!
        @brief      Do nothing
        """
        self.status_message = "State: Idle - Waiting for input"
        self.current_state = "idle"

    def estop(self):
        """!
        @brief      Emergency stop disable torque.
        """
        self.status_message = "EMERGENCY STOP - Check rxarm and restart program"
        self.current_state = "estop"
        self.rxarm.disable_torque()

    def execute(self):
        """!
        @brief      Go through all waypoints
        TODO: Implement this function to execute a waypoint plan
              Make sure you respect estop signal
        """

        self.current_state = "execute"
        self.status_message = "State: Execute - Executing motion plan"
        for waypoint in self.waypoints:
            if self.current_state != "execute":
                break
            self.rxarm.set_positions(waypoint)
            time.sleep(1)  # Wait for 1 seconds
        self.status_message = "State: Execute - Completed motion plan"

        self.next_state = "idle"

    def initialize_rxarm(self):
        """!
        @brief      Initializes the rxarm.
        """
        self.current_state = "initialize_rxarm"
        self.status_message = "RXArm Initialized!"
        if not self.rxarm.initialize():
            print('Failed to initialize the rxarm')
            self.status_message = "State: Failed to initialize the rxarm!"
            time.sleep(5)
        self.next_state = "idle"

    def set_gripper_state(self, state):
        """Set logical gripper state recorded by GUI ('open' or 'closed')."""
        if state not in ('open', 'closed'):
            raise ValueError("gripper state must be 'open' or 'closed'")
        self.current_gripper_state = state
        self.status_message = f"Gripper marked: {state}"

    # ─── Teach & Play ───────────────────────────────────────────────────
    #
    # How to use:
    #   1. Wire GUI buttons in control_station.py (see commented-out examples there):
    #        btnUser4 → self.sm.record_waypoints()    (records current joint pose + gripper)
    #        btnUser5 → self.sm.clear_waypoints()     (resets the list)
    #        btnUser6 → nxt_if_arm_init('teach_play') (replays all recorded waypoints)
    #   2. With torque OFF, physically move the arm to each desired pose.
    #   3. Click "Record WP" at each pose (toggle gripper open/closed between picks).
    #   4. Click "Teach Play" to replay the sequence 10 times.

    def record_waypoints(self):
        """Record the current joint positions and gripper state as a waypoint."""
        joints = None
        try:
            joints = self.rxarm.get_positions()
        except Exception:
            joints = None
        if joints is None:
            joints = getattr(self.rxarm, 'position_fb', None) or getattr(self.rxarm, 'position', None)
        if joints is None:
            self.status_message = "Record failed: no joint feedback"
            return
        joints = list(np.array(joints)[0:self.rxarm.num_joints])
        gripper_state = getattr(self, 'current_gripper_state', 'open')
        self.taught_waypoints.append({'joints': joints, 'gripper': gripper_state})
        self.status_message = f"Recorded WP #{len(self.taught_waypoints)}"

    def clear_waypoints(self):
        """Clear all recorded waypoints."""
        self.taught_waypoints = []
        self.status_message = "Cleared taught waypoints"

    def teach_play(self):
        """Replay all taught waypoints for 10 cycles, respecting gripper state at each."""
        num_cycles = 10
        for _ in range(num_cycles):
            self.current_state = "teach_play"
            self.status_message = "State: Teach & Play - Executing taught waypoints"
            for idx, wp in enumerate(self.taught_waypoints):
                if self.current_state != "teach_play":
                    break
                self.rxarm.set_positions(wp['joints'])
                time.sleep(1)
                if wp['gripper'] == 'closed':
                    self.rxarm.gripper.grasp()
                else:
                    self.rxarm.gripper.release()
                time.sleep(1)
                self.status_message = f"State: Teach & Play - WP #{idx+1}/{len(self.taught_waypoints)}"
            self.status_message = "State: Teach & Play - Cycle complete"
        self.next_state = "idle"

    # ─── Interactive modes ──────────────────────────────────────────────

    def click_to_grab(self):
        """!
        @brief      Click-to-grab mode: alternates between picking (click on block)
                    and placing (click on destination). Stays active for repeated clicks.
        """
        self.current_state = "click_to_grab"

        if not getattr(self.camera, "new_click", False):
            self.status_message = "State: Click to Grab - Click on a target"
            return

        u = int(self.camera.last_click[0])
        v = int(self.camera.last_click[1])
        self.camera.new_click = False
        self.status_message = f"State: Click to Grab - Processing ({u}, {v})"

        try:
            pose_world = self.camera.pixel_to_world(u, v)
            if pose_world is None:
                raise ValueError("pixel_to_world returned None")

            pose_world = np.asarray(pose_world, dtype=float).reshape(-1)
            if pose_world.size < 3:
                raise ValueError(f"pixel_to_world returned shape {pose_world.shape}")

            pose_world[2] += self.z_grab_threshold

            q = IK_geometric(pose_world[:3], 0.0)
            q = np.asarray(q, dtype=float).reshape(-1)

            if self.current_gripper_state == 'open':
                # PICK: approach, descend, grasp, retreat
                T_target = FK_dh(None, q, link=5)
                T_target[2, 3] += self.z_grab_threshold
                q_approach = IK_geometric(T_target[:3, 3], 0.0)
                self.rxarm.set_positions(q_approach)
                time.sleep(3.0)

                pose_world[2] += 20
                q_descend = IK_geometric(pose_world[:3], 0.0)
                q_descend = np.asarray(q_descend, dtype=float).reshape(-1)
                self.rxarm.set_positions(q_descend)
                time.sleep(3.0)

                self.rxarm.gripper.grasp()
                self.current_gripper_state = 'closed'
                time.sleep(0.5)

                pose_world[2] += self.z_grab_threshold + 40
                self.drop_height = pose_world[2]

                q_retreat = IK_geometric(pose_world[:3], 0.0)
                q_retreat = np.asarray(q_retreat, dtype=float).reshape(-1)
                self.rxarm.set_positions(q_retreat)
                time.sleep(1.0)

                self.status_message = "State: Click to Grab - Picked. Click drop location."

            else:
                # PLACE: approach at drop height, descend, release
                T_target = FK_dh(None, q, link=5)
                T_target[2, 3] = self.drop_height
                q_drop = IK_geometric(T_target[:3, 3], 0.0)
                q_drop = np.asarray(q_drop, dtype=float).reshape(-1)

                self.rxarm.set_positions(q_drop)
                time.sleep(2.0)

                pose_world[2] += 20
                q_descend = IK_geometric(pose_world[:3], 0.0)
                q_descend = np.asarray(q_descend, dtype=float).reshape(-1)
                self.rxarm.set_positions(q_descend)
                time.sleep(2.0)

                self.rxarm.gripper.release()
                self.current_gripper_state = 'open'
                time.sleep(0.5)
                self.status_message = "State: Click to Grab - Dropped. Click pickup location."

        except Exception as e:
            self.status_message = f"Click to Grab error: {type(e).__name__}: {e}"

        return

    # ─── Calibration ──────────────────────────────────────────────────────

    def calibrate(self):
        """
        Compute the camera extrinsic matrix (world → camera) using 4 AprilTag
        detections and cv2.solvePnP, then compute the homography for board view.
        """
        self.current_state = "calibrate"

        world_points = np.array([
            [-250, -25, 0],   # Tag 1
            [ 250, -25, 0],   # Tag 2
            [ 250, 275, 0],   # Tag 3
            [-250, 275, 0],   # Tag 4
        ], dtype=np.float32)

        obj_pts = []
        img_pts = []

        for detection in self.camera.tag_detections.detections:
            tag_id = detection.id

            # Only use expected tag IDs
            if tag_id < 1 or tag_id > 4:
                continue

            # 2D image point (pixels)
            img_pts.append([
                detection.centre.x,
                detection.centre.y
            ])

            # Corresponding 3D world point
            obj_pts.append(world_points[tag_id - 1])

        # Convert to NumPy arrays
        obj_pts = np.array(obj_pts, dtype=np.float32)
        img_pts = np.array(img_pts, dtype=np.float32)

        # Sanity checks
        if len(obj_pts) < 4:
            self.status_message = "Calibration failed: missing tags"
            self.next_state = "idle"
            return

        [success, rvec, tvec] = cv2.solvePnP(obj_pts,
                                    img_pts,
                                    self.camera.intrinsic_matrix,
                                    self.camera.distortion,
                                    flags=cv2.SOLVEPNP_ITERATIVE)


        if not success:
            self.status_message = "Calibration failed"
            self.next_state = "idle"
            return

        R, _ = cv2.Rodrigues(rvec)

        extrinsic = np.eye(4, dtype=np.float32)
        extrinsic[:3, :3] = R
        extrinsic[:3, 3] = tvec.flatten()
        self.camera.extrinsic_matrix = extrinsic

        self.camera._calculateH()
        self.status_message = "Calibration - Completed"
        self.next_state = "idle"
        time.sleep(5)

    # ─── Challenge 2: Rainbow line arrangement ───────────────────────────

    def challenge_two(self):
        """Arrange blocks in two horizontal lines (large/small) in rainbow color order."""
        self.current_state = "challenge_two"
        self.status_message = "Challenge 2: Starting..."

        COLORS = ["red", "orange", "yellow", "green", "blue", "purple"]

        LARGE_LINE_Y = 175.0
        SMALL_LINE_Y = 275.0
        LINE_X_START = -137.5
        LINE_SPACING = 55.0
        PLACE_Z = 20.0

        SIZE_THRESH = 1200.0

        PICK_CLEARANCE = 60.0
        GRIP_OFFSET = 15.0

        MT_FAST = 2.0;  AT_FAST = 0.5
        MT_SLOW = 3.0;  AT_SLOW = 1.0

        TIMEOUT = 580.0
        SETTLE = 1.5
        MAX_PASSES = 6
        VERIFY_RADIUS = 45.0
        SCAN_Q = np.array([0.0, 0.0, -1.57, 0.0, 0.0])

        t0 = time.time()
        placed_targets = []

        def mv(q, mt=MT_FAST, at=AT_FAST):
            self.rxarm.set_moving_time(mt)
            self.rxarm.set_accel_time(at)
            self.rxarm.set_positions(np.asarray(q, dtype=float).reshape(-1))
            time.sleep(mt + 0.4)

        def go_auto(xyz, ba=0.0, mt=MT_FAST, at=AT_FAST):
            q, psi = find_feasible_ik(xyz, ba, -np.pi / 2)
            if q is None:
                print(f"[C2] UNREACHABLE: {xyz}")
                return False
            mv(q, mt, at)
            return True

        def go_place(xyz, mt=MT_FAST, at=AT_FAST):
            q = IK_geometric(np.asarray(xyz, dtype=float), 0.0)
            if q is None:
                print(f"[C2] UNREACHABLE place: {xyz}")
                return False
            mv(q, mt, at)
            return True

        def fresh_detections():
            """Move to scan pose, trigger detection, return fresh block list."""
            mv(SCAN_Q)
            time.sleep(SETTLE)
            self.camera.blockDetector()
            time.sleep(0.5)
            dets = getattr(self.camera, 'block_detections', None)
            if not dets or len(dets) == 0:
                time.sleep(1.0)
                self.camera.blockDetector()
                time.sleep(0.5)
                dets = getattr(self.camera, 'block_detections', None)
            return dets or []

        def verify_placements():
            """Re-scan and return indices of blocks missing from their target positions."""
            dets = fresh_detections()
            missing_indices = []
            verified_targets = []

            for idx in range(len(schedule)):
                if idx in remaining:
                    continue

                size_label, target_color, target_x, target_y = schedule[idx]
                found = False
                for b in dets:
                    if b.get('color', '').lower() != target_color:
                        continue
                    rect_area = float(b.get('rect_area', 0.0))
                    is_large = rect_area >= SIZE_THRESH
                    if size_label == "large" and not is_large:
                        continue
                    if size_label == "small" and is_large:
                        continue
                    cx, cy = b['center_px']
                    w = self.camera.pixel_to_world(int(cx), int(cy))
                    if w is None:
                        continue
                    w = np.asarray(w, dtype=float).ravel()
                    if np.hypot(w[0] - target_x, w[1] - target_y) < VERIFY_RADIUS:
                        found = True
                        break

                if found:
                    verified_targets.append((target_x, target_y))
                else:
                    missing_indices.append(idx)
                    tag = "*** ORANGE MISSING ***" if target_color == "orange" else "MISSING"
                    print(f"[C2-VERIFY] {tag}: {target_color} {size_label} "
                          f"at ({target_x:.0f}, {target_y:.0f}) — will retry")

            return missing_indices, verified_targets

        schedule = []
        for i, color in enumerate(COLORS):
            x = LINE_X_START + i * LINE_SPACING
            schedule.append(("large", color, x, LARGE_LINE_Y))
        for i, color in enumerate(COLORS):
            x = LINE_X_START + i * LINE_SPACING
            schedule.append(("small", color, x, SMALL_LINE_Y))

        remaining = set(range(len(schedule)))
        n_placed = 0
        passes_without_progress = 0

        while remaining and (time.time() - t0) < TIMEOUT and passes_without_progress < MAX_PASSES:
            progress_this_pass = False

            for idx in sorted(remaining):
                if (time.time() - t0) > TIMEOUT:
                    break

                size_label, target_color, target_x, target_y = schedule[idx]
                self.status_message = f"C2: {n_placed} placed — {target_color} {size_label}"

                mv(SCAN_Q)
                time.sleep(SETTLE)

                dets = getattr(self.camera, 'block_detections', None)
                if not dets or len(dets) == 0:
                    time.sleep(1.0)
                    dets = getattr(self.camera, 'block_detections', None)
                if not dets or len(dets) == 0:
                    continue

                best = None
                best_dist = 1e9
                for b in dets:
                    if b.get('color', '').lower() != target_color:
                        continue
                    rect_area = float(b.get('rect_area', 0.0))
                    is_large = rect_area >= SIZE_THRESH
                    if size_label == "large" and not is_large:
                        continue
                    if size_label == "small" and is_large:
                        continue

                    cx, cy = b['center_px']
                    w = self.camera.pixel_to_world(int(cx), int(cy))
                    if w is None:
                        continue
                    w = np.asarray(w, dtype=float).ravel()

                    near_placed = False
                    for px, py in placed_targets:
                        if np.hypot(w[0] - px, w[1] - py) < 40.0:
                            near_placed = True
                            break
                    if near_placed:
                        continue

                    d = np.hypot(w[0], w[1])
                    if d < best_dist:
                        best_dist = d
                        best = (b, w)

                if best is None:
                    print(f"[C2] No {target_color} {size_label} found (will retry)")
                    continue

                blk, pw = best
                ang = float(blk.get('angle_deg', 0.0))
                print(f"[C2] Picking {target_color} {size_label} at "
                      f"({pw[0]:.1f}, {pw[1]:.1f}, {pw[2]:.1f}) ang={ang:.1f}")

                # ── PICK ──
                self.rxarm.gripper.release()
                time.sleep(0.3)

                if not go_auto([pw[0], pw[1], pw[2] + PICK_CLEARANCE], ang):
                    continue
                if not go_auto([pw[0], pw[1], pw[2] - GRIP_OFFSET], ang, MT_SLOW, AT_SLOW):
                    continue

                self.rxarm.gripper.grasp()
                time.sleep(0.5)

                go_auto([pw[0], pw[1], pw[2] + PICK_CLEARANCE], ang)

                # ── PLACE ──
                mv(SCAN_Q)

                if not go_place([target_x, target_y, PLACE_Z + PICK_CLEARANCE]):
                    self.rxarm.gripper.release()
                    time.sleep(0.3)
                    continue
                if not go_place([target_x, target_y, PLACE_Z], MT_SLOW, AT_SLOW):
                    self.rxarm.gripper.release()
                    time.sleep(0.3)
                    continue

                self.rxarm.gripper.release()
                time.sleep(0.5)

                go_place([target_x, target_y, PLACE_Z + PICK_CLEARANCE])

                placed_targets.append((target_x, target_y))
                remaining.discard(idx)
                n_placed += 1
                progress_this_pass = True
                print(f"[C2] Placed {target_color} {size_label} at "
                      f"({target_x:.0f}, {target_y:.0f}). #{n_placed} done. "
                      f"Elapsed: {time.time()-t0:.1f}s")

            # ── VERIFY: check all supposedly-placed blocks are actually there ──
            if (time.time() - t0) < TIMEOUT:
                missing_indices, verified_targets = verify_placements()
                if missing_indices:
                    for mi in missing_indices:
                        remaining.add(mi)
                    placed_targets = verified_targets
                    n_placed = len(schedule) - len(remaining)
                    print(f"[C2-VERIFY] Re-queued {len(missing_indices)} missing block(s). "
                          f"{n_placed}/{len(schedule)} confirmed.")

            if progress_this_pass:
                passes_without_progress = 0
            else:
                passes_without_progress += 1
                print(f"[C2] No progress this pass ({passes_without_progress}/{MAX_PASSES})")

        mv(SCAN_Q)
        if remaining:
            missed = [(schedule[i][1], schedule[i][0]) for i in sorted(remaining)]
            print(f"[C2] Could not find: {missed}")
        self.status_message = f"C2: Done! {n_placed} blocks in {time.time()-t0:.1f}s"
        print(self.status_message)
        self.next_state = "idle"

    # ─── Challenge 3: Stack as high as possible ──────────────────────────

    def challenge_three(self):
        """Stack all reachable blocks as high as possible at a fixed world XY."""
        self.current_state = "challenge_three"
        self.status_message = "Challenge 3: Starting..."

        BLOCK_HEIGHT    = 35.0
        STACK_X         = 0.0
        STACK_Y         = 225.0
        PICK_CLEARANCE  = 60.0
        GRIP_OFFSET     = 15.0
        STACK_PROXIMITY = 60.0

        SIDE_THR        = 6
        SIDE_Y_BACK     = 80.0

        MT_FAST  = 2.0;  AT_FAST  = 0.5
        MT_PLACE = 2.5;  AT_PLACE = 0.8

        SETTLE   = 1.0
        MAX_BLK  = 20
        TIMEOUT  = 580.0
        SCAN_Q   = np.array([0.0, 0.0, -1.57, 0.0, 0.0])
        z_psi_threshold = 265

        n = 0
        t0 = time.time()

        def mv(q, mt=MT_FAST, at=AT_FAST):
            self.rxarm.set_moving_time(mt)
            self.rxarm.set_accel_time(at)
            self.rxarm.set_positions(np.asarray(q, dtype=float).reshape(-1))
            time.sleep(mt + 0.4)

        def go_psi(xyz, psi, ba=0.0, mt=MT_FAST, at=AT_FAST):
            for eu in [True, False]:
                q = IK_geometric_stack(np.asarray(xyz, dtype=float), psi, ba, eu)
                if q is not None and check_joint_limits(q):
                    mv(q, mt, at)
                    return True
            print(f"[C3] UNREACHABLE psi={np.rad2deg(psi):.1f}: {xyz}")
            return False

        def go_auto(xyz, ba=0.0, mt=MT_FAST, at=AT_FAST):
            q, psi = find_feasible_ik(xyz, ba, -np.pi / 2)
            if q is None:
                print(f"[C3] UNREACHABLE auto: {xyz}")
                return False, None
            mv(q, mt, at)
            return True, psi

        while n < MAX_BLK and (time.time() - t0) < TIMEOUT:
            self.status_message = f"C3: {n} stacked - scanning..."

            mv(SCAN_Q)
            time.sleep(SETTLE)

            dets = getattr(self.camera, 'block_detections', None)
            if not dets or len(dets) == 0:
                time.sleep(1.0)
                dets = getattr(self.camera, 'block_detections', None)
                if not dets or len(dets) == 0:
                    break

            cands = []
            for b in dets:
                cx, cy = b['center_px']
                w = self.camera.pixel_to_world(cx, cy)
                if w is None:
                    continue
                w = np.asarray(w, dtype=float).ravel()
                if np.hypot(w[0] - STACK_X, w[1] - STACK_Y) < STACK_PROXIMITY:
                    continue
                cands.append((b, w))
            if not cands:
                break

            cands.sort(key=lambda c: np.hypot(c[1][0], c[1][1]))
            blk, pw = cands[0]
            ang = blk.get('angle_deg', 0.0)

            print(f"[C3] Pick ({pw[0]:.1f}, {pw[1]:.1f}, {pw[2]:.1f}) ang={ang:.1f}")

            self.rxarm.gripper.release()
            time.sleep(0.3)

            ok, _ = go_auto([pw[0], pw[1], pw[2] + PICK_CLEARANCE], ang)
            if not ok:
                continue
            ok, _ = go_auto([pw[0], pw[1], pw[2] - GRIP_OFFSET], ang, MT_PLACE, AT_PLACE)
            if not ok:
                continue

            self.rxarm.gripper.grasp()
            time.sleep(0.5)
            go_auto([pw[0], pw[1], pw[2] + PICK_CLEARANCE], ang)

            # ── PLACE ──
            stack_z   = n * BLOCK_HEIGHT
            rel_off   = 20.0 + n * 2.0
            release_z = stack_z + rel_off

            mv(SCAN_Q)

            if n < SIDE_THR:
                clearance  = PICK_CLEARANCE
                approach_z = release_z + clearance
                app_pt = np.array([STACK_X, STACK_Y, approach_z])
                rel_pt = np.array([STACK_X, STACK_Y, release_z])

                psi = compute_paired_psi(app_pt, rel_pt)
                if psi is None:
                    psi = compute_best_psi(rel_pt)
                if psi is None:
                    print(f"[C3] #{n+1} No feasible psi. Stopping.")
                    self.rxarm.gripper.release()
                    break

                print(f"[C3] #{n+1} ABOVE stack_z={stack_z:.0f} rel_z={release_z:.0f} "
                      f"app_z={approach_z:.0f} psi={np.rad2deg(psi):.1f}")


                if release_z > z_psi_threshold:
                    psi = 0.0

                if not go_psi(app_pt, psi):
                    self.rxarm.gripper.release(); break
                if not go_psi(rel_pt, psi, 0.0, MT_PLACE, AT_PLACE):
                    self.rxarm.gripper.release(); break

                self.rxarm.gripper.release()
                time.sleep(0.15)
                go_psi(app_pt, psi)

            else:
                hover_above = 20.0
                hover_z  = release_z + hover_above
                over_pt  = np.array([STACK_X, STACK_Y, hover_z])
                rel_pt   = np.array([STACK_X, STACK_Y, release_z])
                side_pt  = np.array([STACK_X, STACK_Y - SIDE_Y_BACK, hover_z])

                psi = compute_paired_psi(over_pt, rel_pt)
                if psi is None:
                    psi = compute_best_psi(rel_pt)
                if psi is None:
                    print(f"[C3] #{n+1} No feasible psi (side). Stopping.")
                    self.rxarm.gripper.release()
                    break

                print(f"[C3] #{n+1} SIDE stack_z={stack_z:.0f} rel_z={release_z:.0f} "
                      f"hover_z={hover_z:.0f} psi={np.rad2deg(psi):.1f}")

                if release_z > z_psi_threshold:
                    psi = 0.0
                    
                if not go_psi(side_pt, psi):
                    self.rxarm.gripper.release(); break
                if not go_psi(over_pt, psi, 0.0, MT_PLACE, AT_PLACE):
                    self.rxarm.gripper.release(); break
                if not go_psi(rel_pt, psi, 0.0, MT_PLACE, AT_PLACE):
                    self.rxarm.gripper.release(); break

                self.rxarm.gripper.release()
                time.sleep(0.15)
                go_psi(over_pt, psi)
                go_psi(side_pt, psi)

            n += 1
            print(f"[C3] Block #{n} placed. Elapsed: {time.time()-t0:.1f}s")

        mv(SCAN_Q)
        self.status_message = f"C3: Done! {n} blocks in {time.time()-t0:.1f}s"
        print(self.status_message)
        self.next_state = "idle"

    # ─── Challenge 1: Sort by size ────────────────────────────────────────

    def challenge_one(self):
        """
        Sort blocks by size: small → left stack, large → right stack.
        Auto-detects level from block counts:
          L1 (large only): spread on right side, no stacking
          L2 (small + large): stack each group in rainbow order (red on bottom)
        Handles pre-stacked input blocks by unstacking to a temp area first.
        """
        self.current_state = "challenge_one"
        self.status_message = "Challenge 1: Starting..."

        RAINBOW = ["red", "orange", "yellow", "green", "blue", "purple"]
        SIZE_THRESH = 1200.0
        Z_STACKED = 55.0

        SMALL_STACK_XY = np.array([-350.0, -25.0])
        LARGE_STACK_XY = np.array([ 350.0, -25.0])
        LARGE_SPREAD_Y = -25.0
        LARGE_SPREAD_X0 = 100.0
        LARGE_SPREAD_DX = 75.0
        BASE_Z = 15.0
        LARGE_BH = 39.0
        SMALL_BH = 25.0

        PICK_CLR = 60.0
        GRIP_OFF = 15.0

        MT_FAST = 2.0;  AT_FAST = 0.5
        MT_SLOW = 3.0;  AT_SLOW = 1.0
        SETTLE = 1.5
        TIMEOUT = 175.0

        SCAN_Q = np.array([0.0, 0.0, -1.57, 0.0, 0.0])

        t0 = time.time()

        def elapsed():
            return time.time() - t0

        def mv(q, mt=MT_FAST, at=AT_FAST):
            self.rxarm.set_moving_time(mt)
            self.rxarm.set_accel_time(at)
            self.rxarm.set_positions(np.asarray(q, dtype=float).reshape(-1))
            time.sleep(mt + 0.4)

        def go(xyz, ba=0.0, mt=MT_FAST, at=AT_FAST):
            q, psi = find_feasible_ik(xyz, ba, -np.pi / 2)
            if q is None:
                print(f"[C1] UNREACHABLE: {xyz}")
                return False
            mv(q, mt, at)
            return True

        def scan_blocks():
            mv(SCAN_Q)
            time.sleep(SETTLE)
            self.camera.blockDetector()
            time.sleep(0.5)
            dets = getattr(self.camera, 'block_detections', None)
            if not dets or len(dets) == 0:
                time.sleep(1.0)
                self.camera.blockDetector()
                time.sleep(0.5)
                dets = getattr(self.camera, 'block_detections', None)
            result = {"large": [], "small": []}
            for d in (dets or []):
                color = (d.get('color', '') or '').lower()
                if color not in RAINBOW:
                    continue
                rect_area = float(d.get('rect_area', 0.0))
                sz = "large" if rect_area >= SIZE_THRESH else "small"
                cx, cy = d.get('center_px', (None, None))
                if cx is None or cy is None:
                    continue
                w = self.camera.pixel_to_world(int(cx), int(cy))
                if w is None:
                    continue
                w = np.asarray(w, dtype=float).ravel()
                if w[1] < -50.0:
                    continue
                result[sz].append({
                    'color': color,
                    'xyz': w[:3].copy(),
                    'angle_deg': float(d.get('angle_deg', 0.0)),
                })
            return result

        def rainbow_rank(color):
            try:
                return RAINBOW.index(color)
            except ValueError:
                return 999

        def pick_block(xyz, angle):
            self.rxarm.gripper.release()
            time.sleep(0.3)
            if not go([xyz[0], xyz[1], xyz[2] + PICK_CLR], angle):
                return False
            if not go([xyz[0], xyz[1], xyz[2] - GRIP_OFF], angle, MT_SLOW, AT_SLOW):
                return False
            self.rxarm.gripper.grasp()
            time.sleep(0.5)
            go([xyz[0], xyz[1], xyz[2] + PICK_CLR], angle)
            return True

        def place_block(dst):
            mv(SCAN_Q)
            if not go([dst[0], dst[1], dst[2] + PICK_CLR], 0.0):
                self.rxarm.gripper.release()
                time.sleep(0.3)
                return False
            if not go([dst[0], dst[1], dst[2]], 0.0, MT_SLOW, AT_SLOW):
                self.rxarm.gripper.release()
                time.sleep(0.3)
                return False
            self.rxarm.gripper.release()
            time.sleep(0.5)
            go([dst[0], dst[1], dst[2] + PICK_CLR], 0.0)
            return True

        def find_fresh(color, sz, blocks_dict):
            for fb in blocks_dict[sz]:
                if fb['color'] == color:
                    return fb
            return None

        # ─── Phase 1: unstack any blocks sitting on top of others ───
        print("[C1] Phase 1: checking for stacked blocks...")
        blocks = scan_blocks()
        n_large = len(blocks["large"])
        n_small = len(blocks["small"])
        level = 2 if n_small > 0 else 1
        print(f"[C1] Detected {n_large} large, {n_small} small → Level {level}")

        all_blks = blocks["large"] + blocks["small"]
        stacked = sorted(
            [b for b in all_blks if b['xyz'][2] > Z_STACKED],
            key=lambda b: -b['xyz'][2],
        )
        if stacked:
            print(f"[C1] {len(stacked)} stacked block(s) found, moving aside...")
            for i, blk in enumerate(stacked):
                if elapsed() > TIMEOUT:
                    break
                self.status_message = f"C1: unstacking {blk['color']}..."
                tmp = np.array([-100.0 + 70.0 * i, 225.0, BASE_Z])
                if pick_block(blk['xyz'], blk['angle_deg']):
                    place_block(tmp)

        # ─── Phase 2: fresh scan, sort by rainbow order ───
        if elapsed() < TIMEOUT:
            print("[C1] Phase 2: re-scanning & sorting...")
            time.sleep(0.5)
            blocks = scan_blocks()
            for sz in ["large", "small"]:
                blocks[sz].sort(key=lambda b: rainbow_rank(b['color']))
                for b in blocks[sz]:
                    print(f"[C1]   {sz}: {b['color']} @ "
                        f"({b['xyz'][0]:.0f}, {b['xyz'][1]:.0f}, {b['xyz'][2]:.0f})")

        # ─── Phase 3: pick & place ───
        n_placed = 0

        if level == 1:
            for i, blk in enumerate(blocks["large"]):
                if elapsed() > TIMEOUT:
                    break
                fresh = scan_blocks()
                target = find_fresh(blk['color'], "large", fresh)
                if target is None:
                    print(f"[C1] Lost {blk['color']} large, skipping")
                    continue
                drop = np.array([
                    LARGE_SPREAD_X0 + LARGE_SPREAD_DX * i,
                    LARGE_SPREAD_Y,
                    BASE_Z,
                ])
                self.status_message = f"C1 L1: placing {target['color']} large #{i+1}"
                if pick_block(target['xyz'], target['angle_deg']):
                    if place_block(drop):
                        n_placed += 1
                        print(f"[C1] Placed #{n_placed}. {elapsed():.1f}s")
        else:
            stack_z = {"large": BASE_Z, "small": BASE_Z}
            bh = {"large": LARGE_BH, "small": SMALL_BH}
            sxy = {"large": LARGE_STACK_XY, "small": SMALL_STACK_XY}

            for sz in ["large", "small"]:
                for blk in blocks[sz]:
                    if elapsed() > TIMEOUT:
                        break
                    fresh = scan_blocks()
                    target = find_fresh(blk['color'], sz, fresh)
                    if target is None:
                        print(f"[C1] Lost {blk['color']} {sz}, skipping")
                        continue
                    drop = np.array([sxy[sz][0], sxy[sz][1], stack_z[sz]])
                    self.status_message = (f"C1 L2: {target['color']} {sz} "
                                        f"→ z={stack_z[sz]:.0f}")
                    if pick_block(target['xyz'], target['angle_deg']):
                        if place_block(drop):
                            stack_z[sz] += bh[sz]
                            n_placed += 1
                            print(f"[C1] Placed #{n_placed}. {elapsed():.1f}s")

        mv(SCAN_Q)
        self.status_message = f"C1 L{level}: done! {n_placed} blocks in {elapsed():.1f}s"
        print(self.status_message)
        self.next_state = "idle"


class StateMachineThread(QThread):
    """!
    @brief      Runs the state machine
    """
    updateStatusMessage = pyqtSignal(str)
    
    def __init__(self, state_machine, parent=None):
        """!
        @brief      Constructs a new instance.

        @param      state_machine  The state machine
        @param      parent         The parent
        """
        QThread.__init__(self, parent=parent)
        self.sm=state_machine

    def run(self):
        """!
        @brief      Update the state machine at a set rate
        """
        while True:
            self.sm.run()
            self.updateStatusMessage.emit(self.sm.status_message)
            time.sleep(0.05)