"""!
The state machine that implements the logic.
"""
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot, QTimer
import time
import numpy as np
import rclpy
import cv2
from cv_bridge import CvBridge
from kinematics import IK_geometric, FK_dh, compute_wrist_roll, IK_geometric_stack, find_feasible_ik, check_joint_limits
class StateMachine():
    """!
    @brief      This class describes a state machine.

                TODO: Add states and state functions to this class to implement all of the required logic for the armlab
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
        #changes   
        self.taught_waypoints = []  # List to store taught waypoints
        self.current_gripper_state = 'open'
        # For click/grab task
        self.holding_object = False          # True after a successful pick, False after drop
        self.last_pick_mm = None             # stores last picked [x,y,z] in mm
        self.z_grab_threshold = 15
        self.gripper_opened = True
        self.drop_height = None
        # ---------- Auto sort/stack parameters ----------
        # ---------- Sort/Stack memory (their idea) ----------
        self.COLOR_ORDER = "roygbv"
        self.BLOCK_H = {"l": 21, "s": 10}     # mm
        self.NUM_TOWER_BLOCKS = 3                 # L2/L3

        # drop zones: negative y (required). small left, large right
        self.DROP_XY = {
            "s": np.array([-160.0, 100.0], dtype=float),
            "l": np.array([ 160.0, 100.0], dtype=float),
        }
        self.DROP_Z0 = 15.0  # base z at table (tune)
        

        # stacking height counters
        self.tower_h = {"s": 0.0, "l": 0.0}

        # remember placed (size,color) like them
        self.placed_set = set()

        

    def set_next_state(self, state):
        """!
        @brief      Sets the next state.

            This is in a different thread than run so we do nothing here and let run handle it on the next iteration.

        @param      state  a string representing the next state.
        """
        self.next_state = state

    def run(self):
        """!
        @brief      Run the logic for the next state

                    This is run in its own thread.

                    TODO: Add states and functions as needed.
        """

        # IMPORTANT: This function runs in a loop. If you make a new state, it will be run every iteration.
        #            The function (and the state functions within) will continuously be called until the state changes.

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

        if self.next_state == "detect":
            self.detect()

        if self.next_state == "manual":
            self.manual()

        if self.next_state == "teach_play":
            self.teach_play()

        if self.next_state == "click_to_grab":
            self.click_to_grab()
        
        if self.next_state == "test_wrist_align":
            self.test_wrist_align()

        if self.next_state == "auto_L1":
            print("Starting auto sort/stack level 1...")
            self.auto_sort_stack(level=1)
        if self.next_state == "auto_L2":
            print("check!!!!")
            self.auto_sort_stack(level=2)
        if self.next_state == "auto_L3":
            self.auto_sort_stack(level=3)
        if self.next_state == "event_2":
            self.event_2()


        if self.next_state =="challenge_three":
            self.challenge_three()
                
# # ####################################test

    def _move_xyz(self, xyz_mm, block_angle_deg=0.0, wait=1.5, slow=True, place=False):
        if slow:
            self.rxarm.moving_time = 3.5
            self.rxarm.accel_time  = 3.0

        xyz_mm = np.asarray(xyz_mm, dtype=float).reshape(3)

        q = IK_geometric(xyz_mm, float(block_angle_deg))
        if q is None:
            return False

        q = np.asarray(q, dtype=float).reshape(-1)

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
        print(f"Detected {len(dets)} blocks:")

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

            # 4) optional: only pick blocks in front side (if needed)
            # if xyz[1] < 50.0: continue

            # 5) attach angle for IK
            angle_deg = float(d.get("angle_deg", 0.0))
            # print(f"  - Block: size={sz}, color={c_letter}, xyz=({xyz[0]:.1f},{xyz[1]:.1f},{xyz[2]:.1f}), angle={angle_deg:.1f}°")

            block_dict[sz][c_letter].append({
                "xyz": xyz,
                "angle_deg": angle_deg,
                "det": d
            })
        cnt_s = sum(len(lst) for lst in block_dict["s"].values())
        cnt_l = sum(len(lst) for lst in block_dict["l"].values())
        print(f"Total: {cnt_s} small blocks, {cnt_l} large blocks")
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
        self.current_state = f"auto_L{level}"
        t0 = time.time()
        self.goto_observe_pose(wait=2.0)

        # reset memory
        self.tower_h = {"s": 0.0, "l": 0.0}
        self.placed_set = set()
        place_idx = {"s": 0, "l": 0}   # for level 1 spread placement
        do_stack = (level >= 2)
        cached_block_dict = None
       

        Z_TWO_HIGH = 55.0
        dump_i = 0

        if level >= 3:
            cached_block_dict = self._build_block_dict(level)
            highs = []
            for sz in ["s", "l"]:
                for c in self.COLOR_ORDER:
                    for cand in cached_block_dict [sz][c]:
                        zt = float(cand.get("z_top", cand["xyz"][2]))
                        if zt > Z_TWO_HIGH:
                            highs.append((zt, cand, sz))

            k = len(highs)
                
            for i in range(k):

                highs.sort(key=lambda t: -t[0])
                cand = highs[i][1]
                sz = highs[i][2]
                xyz = cand["xyz"]
                ang = cand["angle_deg"]

                # pick
                ho_pick, pick = self._pickup_targets(xyz, sz) 
                self._move_xyz(ho_pick, ang, wait=1.0, slow=True)
                self._move_xyz(pick,   ang, wait=2.0, slow=True)
                time.sleep(2.0)
                self.rxarm.gripper.grasp()
                time.sleep(1.0)
                self._move_xyz(ho_pick, ang, wait=2.0, slow=True)

                # dump (spread along x)
                dump_xyz = np.array([0.0, 200.0, 5.0], dtype=float)
                dump_xyz[0] += 70.0 * dump_i
                dump_i += 1

                ho_drop = dump_xyz.copy(); ho_drop[2] += 80.0
                self._move_xyz(ho_drop, 0.0, wait=1.0, slow=True)   # place angle fixed ok
                self._move_xyz(dump_xyz, 0.0, wait=2.0, slow=True)
                self.rxarm.gripper.release()
                time.sleep(1.0)
                self._move_xyz(ho_drop, 0.0, wait=2.0, slow=True)




        cached_block_dict = self._build_block_dict(level)





        # print("block_dict", cached_block_dict)
        while True:
            if time.time() - t0 > 175:
                break

            # refresh detection every loop (L3 needs this; L1/L2 also fine)
            block_dict = cached_block_dict if cached_block_dict is not None else self._build_block_dict(level)

            sz, c, cand = self._select_next_block_like_them(block_dict)
            # print(f"Next target: size={sz}, color={c}, candidate={cand}")
            if cand is None:
                break

            xyz = cand["xyz"]
            angle_deg = cand["angle_deg"]

            # ---------- PICK ----------
            ho_pick, pick = self._pickup_targets(xyz, sz)
            # print(f"Pick targets: hover=({ho_pick[0]:.1f},{ho_pick[1]:.1f},{ho_pick[2]:.1f}), pick=({pick[0]:.1f},{pick[1]:.1f},{pick[2]:.1f}), angle={angle_deg:.1f}°")

            if not self._move_xyz(ho_pick, angle_deg, wait=1.0, slow=True,place=False):
                continue
            if not self._move_xyz(pick, angle_deg, wait=2.0, slow=True,place =False):
                continue
            time.sleep(1.0)
            print(f"Grasping...")
            self.rxarm.gripper.grasp()
            time.sleep(1.0)

            self._move_xyz(ho_pick, angle_deg, wait=1.0, slow=True)
            time.sleep(0.5)

            # ---------- PLACE ----------
            ho_drop, drop = self._dropoff_targets(sz, level, place_idx[sz])

            if not self._move_xyz(ho_drop, 0, wait=1.0, slow=True,place = True):
                self.rxarm.gripper.release()
                time.sleep(1.0)
                continue
            if not self._move_xyz(drop, 0, wait=2.0, slow=True,place = True):
                self.rxarm.gripper.release()
                time.sleep(1.0)
                continue
            time.sleep(1.0)
            self.rxarm.gripper.release()
            place_idx[sz] += 1
            if level >= 2:
                print(f"Placed {sz} block. Updating tower height.")
                self.tower_h[sz] += self.BLOCK_H[sz]
            time.sleep(0.25)

            self._move_xyz(ho_drop, angle_deg, wait=1.0, slow=True)

            # ---------- UPDATE MEMORY (their idea) ----------
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








###############################################
    """Functions run for each state"""

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

    def detect(self):
        """!
        @brief      Detect the blocks
        """
        time.sleep(1)

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
        #Set logical gripper state recorded by GUI ('open' or 'closed').
        if state not in ('open', 'closed'):
            raise ValueError("gripper state must be 'open' or 'closed'")
        self.current_gripper_state = state
        self.status_message = f"Gripper marked: {state}"
    

    def record_waypoints(self):
        # read current joint positions (safe fallbacks)
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
        print(f"Recorded waypoint: joints={joints}, gripper={gripper_state}")
        self.status_message = f"Recorded WP #{len(self.taught_waypoints)}"

    def clear_waypoints(self):
        self.taught_waypoints = []
        self.status_message = "Cleared taught waypoints"

    def teach_play(self):
        """!
        @brief      Play back taught waypoints
        """
        num_cycles = 10
        for _ in range(num_cycles):
            self.current_state = "teach_play"
            self.status_message = "State: Teach & Play - Executing taught waypoints"
            for idx, wp in enumerate(self.taught_waypoints):
                if self.current_state != "teach_play":
                    break
                self.rxarm.set_positions(wp['joints'])
                time.sleep(1)  # Wait for 1 second
                # Control gripper
                if wp['gripper'] == 'closed':
                    self.rxarm.gripper.grasp()
                else:
                    self.rxarm.gripper.release()
                time.sleep(1)  # Wait for 1 second
                self.status_message = f"State: Teach & Play - Completed WP #{idx+1}"
            self.status_message = "State: Teach & Play - Completed all taught waypoints"

        self.next_state = "idle"    

    def click_to_grab(self):
        """!
        @jbrief Click-to-grab mode: wait for clicks, process each click once. Stays active so you can click repeatedly.
        """
        self.current_state = "click_to_grab"

        # Called every 0.05s; if no click yet, just keep waiting
        if not getattr(self.camera, "new_click", False):
            self.status_message = "State: Click to Grab - Click on a target"
            return

        # Latch click (consume exactly once)
        u = int(self.camera.last_click[0])
        v = int(self.camera.last_click[1])
        self.camera.new_click = False
        self.status_message = f"State: Click to Grab - Processing ({u}, {v})"
        print(f"Received click at pixel coords: (u={u}, v={v})")

        try:
            pose_world = self.camera.pixel_to_world(u, v)
            if pose_world is None:
                raise ValueError("pixel_to_world returned None")

            pose_world = np.asarray(pose_world, dtype=float).reshape(-1)
            if pose_world.size < 3:
                raise ValueError(f"pixel_to_world returned shape {pose_world.shape}")

            pose_world[2] += self.z_grab_threshold  # Adjust Z for grasping height 
            print(f"Target world pose from click: {pose_world}")

            params = np.array([
                [0, 1.570796327, 103.91, 0],
                [205.73, 0, 0, 1.3342],
                [200, 0, 0, -1.3342],
                [0, 1.570796327, 0, 1.570796327],
                [0, 0, 174.15, 0]], dtype=float)

            # IK/FK checker
            q = IK_geometric(pose_world[:3])
            q = np.asarray(q, dtype=float).reshape(-1)
            print(f"IK solution: {q}")
            T = FK_dh(params, q, link=5)
            p_fk = np.array([T[0,3], T[1,3], T[2,3]])
            xyz = pose_world[:3]
            print("target xyz:", xyz)
            print("FK xyz:", p_fk)
            print("pos err (mm):", np.linalg.norm(p_fk - xyz))

        
            # 1) Move to approach position (with z offset applied above)
     
            if self.current_gripper_state == 'open':
                # PICK: descend to surface, grasp, retreat

                params = np.array([
                    [0, 1.570796327, 103.91, 0],
                    [205.73, 0, 0, 1.3342],
                    [200, 0, 0, -1.3342],
                    [0, 1.570796327, 0, 1.570796327],
                    [0, 0, 174.15, 0]], dtype=float)
                T_target = FK_dh(params, q, link=5)
                T_target[2,3] += self.z_grab_threshold
                q = IK_geometric(T_target[:3,3])
                self.rxarm.set_positions(q)

                time.sleep(3.0)
                pose_world[2] -= (self.z_grab_threshold + 10)  # descend to just above surface
                q_descend = IK_geometric(pose_world[:3])
                q_descend = np.asarray(q_descend, dtype=float).reshape(-1)
                self.rxarm.set_positions(q_descend)
                time.sleep(3.0)
                self.rxarm.gripper.grasp()
                self.current_gripper_state = 'closed'
                time.sleep(0.5)




                # pose_world[2] += self.z_grab_threshold
                # q_descend = IK_geometric(pose_world[:3])
                # q_descend = np.asarray(q_descend, dtype=float).reshape(-1)
                # self.rxarm.set_positions(q_descend)
                # time.sleep(1.0)


                pose_world[2] += self.z_grab_threshold + 40
                self.drop_height = pose_world[2]

                q_retreat = IK_geometric(pose_world[:3])
                q_retreat = np.asarray(q_retreat, dtype=float).reshape(-1)
                self.rxarm.set_positions(q_retreat)
                time.sleep(1.0)

                self.status_message = "State: Click to Grab - Picked. Click drop location."

            else:
                # DROP: already at offset height, just release
                # print(drop_height)

                params = np.array([
                    [0, 1.570796327, 103.91, 0],
                    [205.73, 0, 0, 1.3342],
                    [200, 0, 0, -1.3342],
                    [0, 1.570796327, 0, 1.570796327],
                    [0, 0, 174.15, 0]], dtype=float)
                T_target = FK_dh(params, q, link=5)
                T_target[2,3] = self.drop_height
                q_drop = IK_geometric(T_target[:3,3])
                q_drop = np.asarray(q_drop, dtype=float).reshape(-1)

                self.rxarm.set_positions(q_drop)
                time.sleep(2.0)
                pose_world[2] += 20
                q_descend = IK_geometric(pose_world[:3])
                q_descend = np.asarray(q_descend, dtype=float).reshape(-1)
                self.rxarm.set_positions(q_descend)
                time.sleep(2.0)




                # pose_world[2] = self.drop_height+10 
                # q_drop = IK_geometric(pose_world[:3])
                # q_drop = np.asarray(q_drop, dtype=float).reshape(-1)
                # self.rxarm.set_positions(q_drop)
                self.rxarm.gripper.release()
                self.current_gripper_state = 'open'
                time.sleep(0.5)
                self.status_message = "State: Click to Grab - Dropped. Click pickup location."

        except Exception as e:
            # Report error but keep click_to_grab alive
            self.status_message = f"Click to Grab error: {type(e).__name__}: {e}"
            print(self.status_message)

        # Do NOT change next_state: staying in click_to_grab allows repeated clicks
        return




    def test_wrist_align(self):
        self.current_state = "test_wrist_align"

        # ---- PHASE 1: AUTO-DETECT BLOCK ----
        if self.camera.block_detections is None or len(self.camera.block_detections) == 0:
            self.status_message = "No blocks detected"
            self.next_state = "idle"
            return

        block = self.camera.block_detections[0]
        u, v = block["center_px"]
        block_angle_deg = block["angle_deg"]
        self.status_message = f"Picking {block['color']} block at {block_angle_deg:.0f}°"
        print(f"\n[AUTO PICK] ===== PICK PHASE =====")
        print(f"[AUTO PICK] Block: color={block['color']}, px=({u},{v}), angle={block_angle_deg:.1f}°")

        try:
            pose_world = np.asarray(self.camera.pixel_to_world(u, v), dtype=float)
            print(f"[AUTO PICK] World XYZ: x={pose_world[0]:.1f}, y={pose_world[1]:.1f}, z={pose_world[2]:.1f}")

            self.rxarm.gripper.release()
            time.sleep(0.3)

            # 2a) Approach: +z_grab_threshold on top of initial +z_grab_threshold (same as click_to_grab)
            pose_world[2] += self.z_grab_threshold
            approach = pose_world.copy()
            approach[2] += self.z_grab_threshold
            q_app = IK_geometric(approach[:3], block_angle_deg)
            if q_app is None:
                self.status_message = "IK unreachable (approach)"
                self.next_state = "idle"
                return
            q_app = np.asarray(q_app, dtype=float)
            #q_app[4] = compute_wrist_roll(q_app[0], block_angle_deg)
            print(f"[AUTO PICK] Approach Z={approach[2]:.1f}")
            self.rxarm.set_positions(q_app)
            time.sleep(3.0)

            # 2b) Descend: -(z_grab_threshold + 10) from current pose_world
            descend = pose_world.copy()
            descend[2] -= (self.z_grab_threshold + 10)
            q_desc = IK_geometric(descend[:3], block_angle_deg)
            if q_desc is None:
                self.status_message = "IK unreachable (descend)"
                self.next_state = "idle"
                return
            q_desc = np.asarray(q_desc, dtype=float)
            #q_desc[4] = compute_wrist_roll(q_desc[0], block_angle_deg)
            print(f"[AUTO PICK] Descend Z={descend[2]:.1f}")
            self.rxarm.set_positions(q_desc)
            time.sleep(3.0)

            # 2c) Grasp
            self.rxarm.gripper.grasp()
            time.sleep(0.5)
            print(f"[AUTO PICK] Grasped!")

            # 2d) Retreat: +(z_grab_threshold + 40) from descend
            retreat = descend.copy()
            retreat[2] += self.z_grab_threshold + 40
            drop_height = retreat[2]
            q_ret = IK_geometric(retreat[:3], block_angle_deg)
            if q_ret is None:
                self.status_message = "IK unreachable (retreat)"
                self.next_state = "idle"
                return
            q_ret = np.asarray(q_ret, dtype=float)
            #q_ret[4] = compute_wrist_roll(q_ret[0], block_angle_deg)
            print(f"[AUTO PICK] Retreat Z={retreat[2]:.1f}")
            self.rxarm.set_positions(q_ret)
            time.sleep(1.0)

            # ---- PHASE 3: WAIT FOR DROP CLICK ----
            self.status_message = "Block picked! Click drop location."
            self.camera.new_click = False
            print(f"[AUTO PICK] Waiting for drop click...")

            while not self.camera.new_click:
                if self.next_state == "estop":
                    return
                time.sleep(0.05)

            drop_u = int(self.camera.last_click[0])
            drop_v = int(self.camera.last_click[1])
            self.camera.new_click = False

            print(f"\n[AUTO PLACE] ===== PLACE PHASE =====")
            print(f"[AUTO PLACE] Click pixel: ({drop_u}, {drop_v})")

            drop_world = np.asarray(self.camera.pixel_to_world(drop_u, drop_v), dtype=float)
            print(f"[AUTO PLACE] Drop XYZ: x={drop_world[0]:.1f}, y={drop_world[1]:.1f}, z={drop_world[2]:.1f}")

            PLACE_ANGLE = 90.0

            # 4a) Approach at drop_height
            drop_app = drop_world.copy()
            drop_app[2] = drop_height
            q_dapp = IK_geometric(drop_app[:3], PLACE_ANGLE)
            if q_dapp is None:
                self.status_message = "IK unreachable (drop approach)"
                self.next_state = "idle"
                return
            q_dapp = np.asarray(q_dapp, dtype=float)
            #q_dapp[4] = compute_wrist_roll(q_dapp[0], PLACE_ANGLE)
            print(f"[AUTO PLACE] Drop approach Z={drop_app[2]:.1f}")
            self.rxarm.set_positions(q_dapp)
            time.sleep(2.0)

            # 4b) Descend: +20 above clicked point
            drop_desc = drop_world.copy()
            drop_desc[2] += 20
            q_ddesc = IK_geometric(drop_desc[:3], PLACE_ANGLE)
            if q_ddesc is None:
                self.status_message = "IK unreachable (drop descend)"
                self.next_state = "idle"
                return
            q_ddesc = np.asarray(q_ddesc, dtype=float)
            #q_ddesc[4] = compute_wrist_roll(q_ddesc[0], PLACE_ANGLE)
            print(f"[AUTO PLACE] Drop descend Z={drop_desc[2]:.1f}")
            self.rxarm.set_positions(q_ddesc)
            time.sleep(2.0)

            # 4c) Release
            self.rxarm.gripper.release()
            time.sleep(0.5)
            print(f"[AUTO PLACE] Released!")

            # 4d) Retreat
            self.rxarm.set_positions(q_dapp)
            time.sleep(1.0)

            self.status_message = "Pick & place done!"
            print(f"[AUTO PLACE] Complete!")

        except Exception as e:
            self.status_message = f"Auto pick/place error: {e}"
            print(f"[AUTO] Error: {e}")
            import traceback
            traceback.print_exc()

        self.next_state = "idle"


    def click_to_grab2(self):
        """
        First click = PICK, second click = DROP (toggles using self.holding_object)

        Assumptions:
        - camera.world_cood(u,v) returns [x,y,z] in mm.
        - IK_geometric expects [x,y,z] in mm.
        - FK_dh(dh_params, q, link=5) returns mm translation (since dh_params are mm).
        """

        # decide action based on flag
        action = "DROP" if getattr(self, "holding_object", False) else "PICK"

        self.current_state = "click_to_grab"
        if action == "PICK":
            self.status_message = "Click location in workspace for block pickup"
        else:
            self.status_message = "Click location in workspace to drop block"

        # read mouse click
        # Latch click (consume exactly once)
        u = int(self.camera.last_click[0])
        v = int(self.camera.last_click[1])
        p_mm = np.array(self.camera.pixel_to_world(u, v), dtype=float)  # [x,y,z] in mm
        
        #-------
        # IK/FK checker - can delete once working
        PARAMS = np.array(
            [
                [0.0,    1.570796327, 103.91,       0.0],
                [205.73, 0.0,         0.0,          1.3342],
                [200.0,  0.0,         0.0,         -1.3342],
                [0.0,    1.570796327, 0.0,          1.570796327],
                [0.0,    0.0,         174.15,       0.0],
            ],
            dtype=float,
        )

        print("\n---- Real click target FK/IK check ----")
        print(f"\n mouse click (u,v)=({u}, {v})")
        print(f"  action: {action}")
        print(f"  target pose: {p_mm}")

        q_hat = IK_geometric(p_mm)
        if q_hat is None:
            print("  Unreachable")
            self.next_state = "idle"
            return

        T_fk = FK_dh(PARAMS, q_hat, link=5)
        p_hat_mm = T_fk[:3, 3].astype(float)

        err_vec = p_hat_mm - p_mm
        err = float(np.linalg.norm(err_vec))

        print(f"  IK: {q_hat} ")
        print(f"  FK: {p_hat_mm} ")
        print(f"  error: {err_vec} ")
        print(f"  ||error||: {err:.3f} ")

        # motion function
        def goto_xyz_mm(xyz_mm, label=""):
            q = IK_geometric(xyz_mm)
            if q is None:
                print(f"Unreachable for {label} xyz_mm={xyz_mm}")
                return False
            self.rxarm.set_positions(q)
            return True

        # motion
        approach_mm = p_mm.copy()
        approach_mm[2] += 75.0  # 75 mm above - change accordingly
        descend_mm = p_mm.copy()
        # safer default: don't go below table unless you intentionally want it
        descend_mm[2] = p_mm[2]
        if not goto_xyz_mm(approach_mm, label="approach(+75mm)"):
            self.next_state = "idle"
            return
        time.sleep(1.2)

        if not goto_xyz_mm(descend_mm, label="descend"):
            self.next_state = "idle"
            return
        time.sleep(1.2)

        if action == "PICK":
            self.rxarm.gripper.grasp()
            time.sleep(0.8)
            self.holding_object = True
            self.last_pick_mm = p_mm.copy()
            self.status_message = "Picked. Click a drop location."
        else:
            self.rxarm.gripper.release()
            time.sleep(0.8)
            self.holding_object = False
            self.last_pick_mm = None
            self.status_message = "Dropped. Click a pickup location."

        goto_xyz_mm(approach_mm, label="retreat(+75mm)")
        time.sleep(1.2)

        self.next_state = "idle"

    def calibrate(self):
        """
        Perform camera extrinsic calibration using AprilTag detections
        """
        self.current_state = "calibrate"

        world_points = np.array([
            [-250, -25, 0],   # Tag 1
            [ 250, -25, 0],   # Tag 2
            [ 250, 275, 0],   # Tag 3
            [-250, 275, 0]   # Tag 4
            # [300, 125, 156],#Tag 5
            # [-350, 325, 86]
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
            print("Calibration failed: not all 4 tags detected")
            print(f"Detected {len(obj_pts)} tags")
            self.status_message = "Calibration failed: missing tags"
            self.next_state = "idle"
            return

        print("World points:\n", obj_pts)
        print("Image points:\n", img_pts)
        print("camera intrinsics:\n", self.camera.intrinsic_matrix)
        print("distortion coeffs:\n", self.camera.distortion)

        # success,  rvec, tvec , _= cv2.solvePnPRansac(
        #     obj_pts,
        #     img_pts,
        #     self.camera.intrinsic_matrix,
        #     self.camera.distortion,
        #     iterationsCount = 2000, 
        #     reprojectionError = 2.0,
            
        # )

        [success, rvec, tvec] = cv2.solvePnP(obj_pts,
                                    img_pts,
                                    self.camera.intrinsic_matrix,
                                    self.camera.distortion,
                                    flags=cv2.SOLVEPNP_ITERATIVE)


        if not success:
            print("solvePnP failed")
            self.status_message = "Calibration failed"
            self.next_state = "idle"
            return

        R, _ = cv2.Rodrigues(rvec)

        print("Rotation Matrix:\n", R)
        print("Translation Vector:\n", tvec)

        extrinsic = np.eye(4, dtype=np.float32)
        extrinsic[:3, :3] = R
        extrinsic[:3, 3] = tvec.flatten()
        self.camera.extrinsic_matrix = extrinsic

        print("Extrinsic Matrix:\n", extrinsic)
        self.camera._calculateH()
        self.status_message = "Calibration - Completed"
        self.next_state = "idle"
        time.sleep(5)

    def challenge_three(self):

        """Stack large blocks as high as possible at a fixed world location."""

        self.current_state = "challenge_three"

        self.status_message = "Challenge 3: Starting..."

        # ── Tunable constants ──

        BLOCK_HEIGHT     = 38.0     # large block height in mm (measure and adjust)

        STACK_X          = 0.0      # world X of stack center (mm)

        STACK_Y          = 225.0    # world Y of stack center (mm)

        APPROACH_CLEARANCE = 80.0   # mm above target for approach waypoint

        GRIP_OFFSET      = 15.0     # mm below z_top for pick grip center

        PLACE_OFFSET     = 5.0      # mm above stack top when releasing

        STACK_PROXIMITY  = 60.0     # mm — ignore detected blocks this close to stack

        MOVE_TIME_FAST   = 1.5      # seconds for large moves

        MOVE_TIME_SLOW   = 1.2      # seconds for short approach/descend

        ACCEL_TIME       = 0.4

        SETTLE_TIME      = 1.0      # seconds to wait after moving to scan pose

        MAX_BLOCKS       = 20       # safety cap

        TIMEOUT          = 580.0    # seconds (keep 20s buffer from 600s limit)

        # scan pose: arm pointing up, out of camera view

        SCAN_POSE = np.array([0.0, 0.0, -1.57, 0.0, 0.0])

        blocks_stacked = 0

        start_time = time.time()

        def elapsed():

            return time.time() - start_time

        def move(q, mt=MOVE_TIME_FAST):

            self.rxarm.set_moving_time(mt)

            self.rxarm.set_accel_time(ACCEL_TIME)

            self.rxarm.set_positions(np.asarray(q, dtype=float).reshape(-1))

            time.sleep(mt + 0.3)

        def goto_xyz(xyz, block_angle=0.0, pref_psi=-np.pi/2, mt=MOVE_TIME_FAST):

            q, psi = find_feasible_ik(xyz, block_angle, pref_psi)

            if q is None:

                print(f"[C3] UNREACHABLE: {xyz}, pref_psi={pref_psi:.2f}")

                return False, None

            move(q, mt)

            return True, psi

        while blocks_stacked < MAX_BLOCKS and elapsed() < TIMEOUT:

            self.status_message = f"C3: Stacked {blocks_stacked} — scanning..."

            # ─── 1. Move to scan position ───

            move(SCAN_POSE, MOVE_TIME_FAST)

            time.sleep(SETTLE_TIME)

            # ─── 2. Read detected blocks ───

            detections = getattr(self.camera, 'block_detections', None)

            if not detections or len(detections) == 0:

                print("[C3] No blocks detected, retrying...")

                time.sleep(1.0)

                detections = getattr(self.camera, 'block_detections', None)

                if not detections or len(detections) == 0:

                    print("[C3] Still no blocks. Done.")

                    break

            # ─── 3. Filter out blocks near the stack ───

            candidates = []

            for blk in detections:

                cx, cy = blk['center_px']

                w_pos = self.camera.pixel_to_world(cx, cy)

                if w_pos is None:

                    continue

                w_pos = np.asarray(w_pos, dtype=float).ravel()

                dist_to_stack = np.hypot(w_pos[0] - STACK_X, w_pos[1] - STACK_Y)

                if dist_to_stack < STACK_PROXIMITY:

                    continue

                candidates.append((blk, w_pos))

            if not candidates:

                print("[C3] No pickable blocks (all near stack or none). Done.")

                break

            # pick the block closest to the arm base (easiest reach)

            candidates.sort(key=lambda c: np.hypot(c[1][0], c[1][1]))

            chosen_blk, pick_world = candidates[0]

            angle_deg = chosen_blk.get('angle_deg', 0.0)

            pick_z = pick_world[2]

            print(f"[C3] Picking block at world ({pick_world[0]:.1f}, {pick_world[1]:.1f}, {pick_z:.1f}), angle={angle_deg:.1f}°")

            self.status_message = f"C3: Picking block #{blocks_stacked+1}..."

            # ─── 4. PICK: approach → descend → grasp → retreat ───

            self.rxarm.gripper.release()

            time.sleep(0.3)

            # approach above block (tool-down)

            approach_pick = np.array([pick_world[0], pick_world[1], pick_z + APPROACH_CLEARANCE])

            ok, _ = goto_xyz(approach_pick, angle_deg, -np.pi/2, MOVE_TIME_FAST)

            if not ok:

                print("[C3] Can't reach pick approach, skipping block")

                continue

            # descend to grip

            descend_pick = np.array([pick_world[0], pick_world[1], pick_z - GRIP_OFFSET])

            ok, _ = goto_xyz(descend_pick, angle_deg, -np.pi/2, MOVE_TIME_SLOW)

            if not ok:

                print("[C3] Can't reach pick descend, skipping block")

                continue

            self.rxarm.gripper.grasp()

            time.sleep(0.5)

            # retreat

            goto_xyz(approach_pick, angle_deg, -np.pi/2, MOVE_TIME_SLOW)

            # ─── 5. PLACE: transit → approach stack → descend → release → retreat ───

            stack_z = blocks_stacked * BLOCK_HEIGHT

            place_z = stack_z + PLACE_OFFSET

            self.status_message = f"C3: Placing block #{blocks_stacked+1} at z={place_z:.0f}mm..."

            # transit to scan pose first to avoid sweeping over the workspace

            move(SCAN_POSE, MOVE_TIME_FAST)

            # approach above stack

            approach_stack = np.array([STACK_X, STACK_Y, place_z + APPROACH_CLEARANCE])

            ok, used_psi = goto_xyz(approach_stack, 0.0, -np.pi/2, MOVE_TIME_FAST)

            if not ok:

                print(f"[C3] Can't reach stack approach at z={place_z + APPROACH_CLEARANCE:.0f}. Dropping block and stopping.")

                self.rxarm.gripper.release()

                break

            # descend to place (use adaptive psi)

            place_target = np.array([STACK_X, STACK_Y, place_z])

            ok, used_psi = goto_xyz(place_target, 0.0, -np.pi/2, MOVE_TIME_SLOW)

            if not ok:

                print(f"[C3] Can't reach place target z={place_z:.0f}. Dropping block and stopping.")

                self.rxarm.gripper.release()

                break

            print(f"[C3] Placing with psi={np.rad2deg(used_psi):.1f}° at z={place_z:.0f}mm")

            self.rxarm.gripper.release()

            time.sleep(0.4)

            # retreat above stack

            goto_xyz(approach_stack, 0.0, -np.pi/2, MOVE_TIME_SLOW)

            blocks_stacked += 1

            print(f"[C3] ✓ Block #{blocks_stacked} placed. Elapsed: {elapsed():.1f}s")

        # ─── Done ───

        move(SCAN_POSE, MOVE_TIME_FAST)

        self.status_message = f"C3: Done! Stacked {blocks_stacked} blocks in {elapsed():.1f}s"

        print(self.status_message)

        self.next_state = "idle"


    def event_2(self):
            block_detections = self.camera.block_detections

            # Catch for no block detections
            if block_detections is None or block_detections == 0:
                self.status_message = "No blocks detected"
                self.next_state = "idle"
                return

            # set allowable colours and create empty lists for small and large blocks
            colours = ["red","orange","yellow","green","blue","purple"]
            small_blocks, large_blocks = [], []

            # assign block sizes
            for block in block_detections:
                # ignore any block not in color scope
                if block["color"] not in colours:
                    continue
                # convert block center point to world point
                # convert block orientation to rad
                world = self.camera.pixel_to_world(block["center_px"])
                block["xyz"] = np.array(world[:3])
                block["psi"] = np.deg2rad(block["angle_deg"])

                # sort into large and small
                # PHYSICAL TUNING REQUIRED
                if block["area"] <= 800:
                    small_blocks.append(block)
                else:
                    large_blocks.append(block)

            # sort small and large blocks by colour
            small_blocks = sorted(small_blocks, key=lambda x: colours.index(x["color"]))
            large_blocks = sorted(large_blocks, key=lambda x: colours.index(x["color"]))

            def pick_block(pick_position, block_orientation):
                # SAFETY OFFSETS
                approach_height = 50
                grip_offset = 15
                approach_target = pick_position[2] + approach_height
                grip_target = pick_position[2] - grip_offset
                IK_geometric(approach_target, block_orientation)
                time.sleep(1)
                IK_geometric(grip_target, block_orientation)
                self.rxarm.gripper.grasp()

            def move_block(from_position, to_position):
                # SAFETY OFFSETS
                move_height = 150
                IK_geometric(from_position[2] + move_height, 0.0)
                time.sleep(1)
                IK_geometric(to_position[2] + move_height, 0.0)

            def place_block(place_position, block_orientation):
                # SAFETY OFFSETS
                approach_height = 50
                drop_offset = 20
                approach_target = place_position[2] + approach_height
                drop_target = place_position[2] + drop_offset
                IK_geometric(approach_target, block_orientation)
                time.sleep(1)
                IK_geometric(drop_target, block_orientation)
                self.rxarm.gripper.release()

            # ---- PLACE LARGE LINE ----
            for i, block in enumerate(large_blocks):
                target = np.array([-175 + i*50, 250, 0])
                pick_block(block["xyz"], block["psi"])
                move_block(block["xyz"], target)
                place_block(target, 0.0)

            # ---- PLACE SMALL LINE ----
            for i, block in enumerate(small_blocks):
                target = np.array([-175 + i*50, 175, 0])
                pick_block(block["xyz"], block["psi"])
                move_block(block["xyz"], target)
                place_block(target, 0.0)

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