"""!
The state machine that implements the logic.
"""
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot, QTimer
import time
import numpy as np
import rclpy
import cv2
from cv_bridge import CvBridge
from kinematics import IK_geometric, FK_dh
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

            self.rxarm.set_positions(q)
            self.status_message = "State: Click to Grab - Done. Click again."

        except Exception as e:
            # Report error but keep click_to_grab alive
            self.status_message = f"Click to Grab error: {type(e).__name__}: {e}"
            print(self.status_message)

        # Do NOT change next_state: staying in click_to_grab allows repeated clicks
        return


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