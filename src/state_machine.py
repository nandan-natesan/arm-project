"""!
The state machine that implements the logic.
"""
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot, QTimer
import time
import numpy as np
import rclpy
import cv2
from cv_bridge import CvBridge

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

    # def calibrate(self):
    #     self.current_state = "calibrate"
    #     self.status_message = "Calibration - Running"

    #     # map known tag ids -> world coordinates (same units as you will use downstream)
    #     world_map = {
    #         1: [-250.0, -25.0, 0.0],
    #         2: [250.0, -25.0, 0.0],
    #         3: [250.0, 275.0, 0.0],
    #         4: [-250.0, 275.0, 0.0],
    #     }

    #     detections_msg = getattr(self.camera, "tag_detections", None)
    #     if not detections_msg or not getattr(detections_msg, "detections", None):
    #         self.status_message = "Calibration failed: no tag detections"
    #         return

    #     obj_pts = []
    #     img_pts = []

    #     for det in detections_msg.detections:
    #         tag_id = getattr(det, "id", None) or getattr(det, "tag_id", None)
    #         if tag_id is None:
    #             continue

    #         centre = getattr(det, "centre", None) or getattr(det, "center", None)
    #         if centre is None:
    #             continue

    #         cx = getattr(centre, "x", None) or getattr(centre, "u", None)
    #         cy = getattr(centre, "y", None) or getattr(centre, "v", None)
    #         if cx is None or cy is None:
    #             continue

    #         if int(tag_id) in world_map:
    #             obj_pts.append(world_map[int(tag_id)])
    #             img_pts.append([float(cx), float(cy)])

    #     if len(obj_pts) < 4:
    #         self.status_message = f"Calibration failed: need >=4 correspondences, got {len(obj_pts)}"
    #         return

    #     k = getattr(self.camera, "intrinsic_matrix", None)
    #     d = getattr(self.camera, "distortion", None)
    #     if k is None:
    #         self.status_message = "Calibration failed: missing camera intrinsics"
    #         return

    #     obj_pts_np = np.asarray(obj_pts, dtype=np.float64)
    #     img_pts_np = np.asarray(img_pts, dtype=np.float64)
    #     cam_mtx = np.asarray(k, dtype=np.float64)
    #     dist_coef = None if d is None else np.asarray(d, dtype=np.float64)

    #     # Debug print of inputs (comment out later)
    #     print("Calibration inputs: obj_pts shape", obj_pts_np.shape, "img_pts shape", img_pts_np.shape)
    #     print("Camera matrix:\n", cam_mtx)
    #     if dist_coef is not None:
    #         print("Distortion coeffs:", dist_coef)

    #     try:
    #         ret, rvec, tvec = cv2.solvePnP(obj_pts_np, img_pts_np, cam_mtx, dist_coef)
    #     except Exception as e:
    #         self.status_message = f"Calibration error: {e}"
    #         return

    #     if not ret:
    #         self.status_message = "Calibration failed: solvePnP returned False"
    #         return

    #     R, _ = cv2.Rodrigues(rvec)
    #     extrinsic = np.eye(4, dtype=np.float64)
    #     extrinsic[:3, :3] = R
    #     extrinsic[:3, 3] = tvec.flatten()

    #     self.camera.extrinsic_matrix = extrinsic
    #     print("Extrinsic matrix:\n", extrinsic)

    #     self.status_message = "Calibration - Completed Calibration"
    #     self.next_state = "idle"
    #     time.sleep(1)

    # def calibrate(self):
    #     """!
    #     @brief      Gets the user input to perform the calibration
    #     """
    #     self.current_state = "calibrate"
              
    #     """TODO Perform camera calibration routine here"""
    #     world_points = np.array([[-250,-25,0],
    #                              [250,-25,0],
    #                              [250,275,0],
    #                              [-250,275,0]
    #                              ])
    #     image_points = np.zeros((4,2))
    #     for detection in self.camera.tag_detections.detections:            
    #         center_x = int(detection.centre.x)
    #         center_y = int(detection.centre.y)
    #         tag_id = detection.id            
    #         image_points[tag_id-1,:] = np.array([center_x, center_y])        
        
    #     k = self.camera.intrinsic_matrix
    #     d = self.camera.distortion        
    #     [_, R_exp, t] = cv2.solvePnP(world_points,
    #                                 image_points,
    #                                 k,
    #                                 d,
    #                                 flags=cv2.SOLVEPNP_ITERATIVE)
    #     R, _ = cv2.Rodrigues(R_exp)  
    #     print(f"Rotation Matrix:\n{R}\nTranslation Vector:\n{t}")      
    #     extrinsic = np.zeros((4,4))
    #     # print(f"EXTRINSIC INITIAL:\n{extrinsic}")
    #     extrinsic[:3,:3] = R
    #     extrinsic[:-1,3] = t.flatten()
    #     extrinsic[-1,-1] = 1        
        
        
    #     self.camera.extrinsic_matrix = extrinsic
    #     # print(f"Extrinsic matrix:{self.camera.extrinsic_matrix}")
        
    #     self.status_message = "Calibration - Completed Calibration"
    #     self.next_state = "idle"  
    #     time.sleep(5)
    
    def calibrate(self):
        """
        Perform camera extrinsic calibration using AprilTag detections
        """
        self.current_state = "calibrate"

        world_points = np.array([
            [-250, -25, 0],   # Tag 1
            [ 250, -25, 0],   # Tag 2
            [ 250, 275, 0],   # Tag 3
            [-250, 275, 0],   # Tag 4
            [300, 125, 156],#Tag 5
            [-350, 325, 86]
        ], dtype=np.float32)

        obj_pts = []
        img_pts = []

        for detection in self.camera.tag_detections.detections:
            tag_id = detection.id

            # Only use expected tag IDs
            if tag_id < 1 or tag_id > 6:
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
        if len(obj_pts) < 6:
            print("Calibration failed: not all 6 tags detected")
            print(f"Detected {len(obj_pts)} tags")
            self.status_message = "Calibration failed: missing tags"
            self.next_state = "idle"
            return

        print("World points:\n", obj_pts)
        print("Image points:\n", img_pts)
        print("camera intrinsics:\n", self.camera.intrinsic_matrix)
        print("distortion coeffs:\n", self.camera.distortion)

        success,  rvec, tvec , _= cv2.solvePnPRansac(
            obj_pts,
            img_pts,
            self.camera.intrinsic_matrix,
            self.camera.distortion,
            iterationsCount = 2000, 
            reprojectionError = 2.0,
            
        )

        # [success, rvec, tvec] = cv2.solvePnP(obj_pts,
        #                             img_pts,
        #                             self.camera.intrinsic_matrix,
        #                             self.camera.distortion,
        #                             flags=cv2.SOLVEPNP_ITERATIVE)


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
        # self.camera._calculateH()

        print("Extrinsic Matrix:\n", extrinsic)

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