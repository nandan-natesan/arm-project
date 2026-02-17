#!/usr/bin/env python3

"""!
Class to represent the camera.
"""
 
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor, MultiThreadedExecutor

import cv2
import time
import numpy as np
from PyQt5.QtGui import QImage
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
from apriltag_msgs.msg import *
from cv_bridge import CvBridge, CvBridgeError


class Camera():
    """!
    @brief      This class describes a camera.
    """

    def __init__(self):
        """!
        @brief      Construcfalsets a new instance.
        """
        self.VideoFrame = np.zeros((720,1280, 3)).astype(np.uint8)
        self.VideoFrameWarped = np.zeros((720,1280, 3)).astype(np.uint8)
        self.VideoFrameWarpedDrawn = np.zeros((720,1280, 3)).astype(np.uint8)
        self.GridFrame = np.zeros((720,1280, 3)).astype(np.uint8)
        self.TagImageFrame = np.zeros((720,1280, 3)).astype(np.uint8)
        self.DepthFrameRaw = np.zeros((720,1280)).astype(np.uint16)
        """ Extra arrays for colormaping the depth image"""
        self.DepthFrameHSV = np.zeros((720,1280, 3)).astype(np.uint8)
        self.DepthFrameRGB = np.zeros((720,1280, 3)).astype(np.uint8)


        # mouse clicks & calibration variables
        self.camera_calibrated = False
        self.distortion = np.array(
            [
            0.15564486384391785,
            0.48568257689476013,
            0.0019681642297655344,
            0.0007267732871696353,
            0.44230175018310547
            ]
        )
        self.intrinsic_matrix = np.eye(3)
        self.extrinsic_matrix = np.eye(4)
        self.last_click = np.array([0, 0]) # This contains the last clicked position
        self.new_click = False # This is automatically set to True whenever a click is received. Set it to False yourself after processing a click
        self.rgb_click_points = np.zeros((5, 2), int)
        self.depth_click_points = np.zeros((5, 2), int)
        self.grid_x_points = np.arange(-450, 500, 50)
        self.grid_y_points = np.arange(-175, 525, 50)
        self.grid_points = np.array(np.meshgrid(self.grid_x_points, self.grid_y_points))
        self.tag_detections = None
        self.tag_locations = [[-250, -25], [250, -25], [250, 275], [-250, 275]]
        """ block info """
        self.block_contours = np.array([])
        self.block_detections = np.array([])

        self.H = np.eye(3)
        self.hasHcalculate = False
        self.DepthFrameWarpedRaw = None
        self.DepthFrameWarpedRGB = None

    def processVideoFrame(self):
        """!
        @brief      Process a video frame
        """
        # cv2.drawContours(self.VideoFrame, self.block_contours, -1,
        #                  (255, 0, 255), 3)
        if not hasattr(self,'VideoFrameWarped') or self.VideoFrameWarped is None:
            return
        out = self.VideoFrameWarped.copy()
        cv2.drawContours(out, self.block_contours, -1,(255, 0, 255), 3)
        self.VideoFrameWarpedDrawn = out

    def ColorizeDepthFrame(self):
        """!
        @brief Converts frame to colormaped formats in HSV and RGB
        """
        self.DepthFrameHSV[..., 0] = self.DepthFrameRaw >> 1
        self.DepthFrameHSV[..., 1] = 0xFF
        self.DepthFrameHSV[..., 2] = 0x9F
        self.DepthFrameRGB = cv2.cvtColor(self.DepthFrameHSV,
                                          cv2.COLOR_HSV2RGB)
        if self.hasHcalculate:
            self.warpDepthImage()
            self.generateWorldHeightMap()
            self.blockDetector()


    def loadVideoFrame(self):
        """!
        @brief      Loads a video frame.
        """
        self.VideoFrame = cv2.cvtColor(
            cv2.imread("data/rgb_image.png", cv2.IMREAD_UNCHANGED),
            cv2.COLOR_BGR2RGB)

    def loadDepthFrame(self):
        """!
        @brief      Loads a depth frame.
        """
        self.DepthFrameRaw = cv2.imread("data/raw_depth.png",
                                        0).astype(np.uint16)

    def convertQtVideoFrame(self):
        """!
        @brief      Converts frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.VideoFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtGridFrame(self):
        """!
        @brief      Converts frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.VideoFrameWarped, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtDepthFrame(self):
        """!
       @brief      Converts colormaped depth frame to format suitable for Qt

       @return     QImage
       """
        try:
            img = QImage(self.DepthFrameRGB, self.DepthFrameRGB.shape[1],
                         self.DepthFrameRGB.shape[0], QImage.Format_RGB888)
            return img
        except:
            return None


    def convertQtDepthFrame(self):
            """!
            @brief      Returns the QImage for the GUI. 
                        Switches between Original and Warped based on state.
            """
            if self.DepthFrameRaw is None:
                return None

            # Decide which image to show
            if self.hasHcalculate and hasattr(self, 'DepthFrameWarpedRGB'):
                display_frame = self.DepthFrameWarpedRGB
            else:
                display_frame = self.DepthFrameRGB

            if display_frame is None:
                return None

            # Convert to QImage
            height, width, channel = display_frame.shape
            bytes_per_line = 3 * width
            return QImage(display_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)


    def convertQtTagImageFrame(self):
        """!
        @brief      Converts tag image frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.TagImageFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def getAffineTransform(self, coord1, coord2):
        """!
        @brief      Find the affine matrix transform between 2 sets of corresponding coordinates.

        @param      coord1  Points in coordinate frame 1
        @param      coord2  Points in coordinate frame 2

        @return     Affine transform between coordinates.
        """
        pts1 = coord1[0:3].astype(np.float32)
        pts2 = coord2[0:3].astype(np.float32)
        # print(cv2.getAffineTransform(pts1, pts2))
        return cv2.getAffineTransform(pts1, pts2)

    def loadCameraCalibration(self, file):
        """!
        @brief      Load camera intrinsic matrix from file.

                    TODO: use this to load in any calibration files you need to

        @param      file  The file
        """
        pass

    def classifyBlockColorBoard(self, bgr_bd, contour_bd):
        """
        Color detection by matching mean BGR in contour to reference colors.
        Input:
        - bgr_bd: board-view BGR image (OpenCV default)
        - contour_bd: contour on bgr_bd
        Output: red/orange/yellow/green/blue/purple/unknown
        """

        # Reference colors (assumed BGR here!)
        color_refs = [
            {'id': 'red',    'color': (120, 30, 35)},
            {'id': 'orange', 'color': (212, 96, 44)},
            {'id': 'yellow', 'color': (180, 170, 50)},
            {'id': 'green', 'color': (46, 113, 85)},
            {'id': 'blue', 'color': (11, 78, 120)},
            {'id': 'purple', 'color': (55, 60, 70)}
    ]

        # Mask inside contour
        mask = np.zeros(bgr_bd.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [contour_bd], 255)
        mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=2)

        if int(np.count_nonzero(mask)) < 30:
            return "unknown"

        # Mean BGR in mask
        mean_bgr = np.array(cv2.mean(bgr_bd, mask=mask)[:3], dtype=np.float32)

        # Optional: reject low-color / dark regions using HSV
        hsv = cv2.cvtColor(bgr_bd, cv2.COLOR_BGR2HSV)
        mean_h, mean_s, mean_v, _ = cv2.mean(hsv, mask=mask)
        if mean_s < 50 or mean_v < 40:
            return "unknown"

        # Nearest reference color in BGR space
        best_color = "unknown"
        best_dist = 1e9

        for ref in color_refs:
            ref_bgr = np.array(ref["color"], dtype=np.float32) # BGR
            d = float(np.linalg.norm(ref_bgr - mean_bgr))
            if d < best_dist:
                best_dist = d
                best_color = ref["id"]

        # Distance gate: too far -> unknown (tune if needed)
        if best_dist > 60:
            return "unknown"

        return best_color


    def blockDetector(self):
        """!
        @brief      Hybrid Detector: Finds blocks using RGB (masked to table area) 
                    and verifies them using Depth.
        """
        if self.VideoFrame is None or self.WorldHeightMap is None:
            return

        # 1. Prepare RGB Image (Warping)
        h, w = self.VideoFrame.shape[:2]
        rgb_bd = cv2.warpPerspective(self.VideoFrame, self.H, (w, h))
        hsv_bd = cv2.cvtColor(rgb_bd, cv2.COLOR_BGR2HSV)
        
        # --- 2. Create Workspace Mask (Same logic as depth function) ---
        workspace_mask = np.zeros((h, w), dtype=np.uint8)
        # Outer bounds (Table) - White
        cv2.rectangle(workspace_mask, (225, 0), (1170, 670), 255, cv2.FILLED)
        # Inner "hole" (Robot Base) - Black
        cv2.rectangle(workspace_mask, (620, 340), (820, 670), 0, cv2.FILLED)

        # Define Color Ranges
        HSV_RANGES = {
            "red_low":   (np.array([0, 100, 80]),   np.array([10, 255, 255])),
            "red_high":  (np.array([170, 100, 80]), np.array([180, 255, 255])),
            "orange":    (np.array([10, 100, 80]),  np.array([22, 255, 255])),
            "yellow":    (np.array([22, 100, 80]),  np.array([35, 255, 255])),
            "green":     (np.array([35, 100, 80]),  np.array([85, 255, 255])),
            "blue":      (np.array([85, 100, 80]),  np.array([130, 255, 255])),
            "purple":    (np.array([130, 100, 80]), np.array([170, 255, 255]))
        }

        block_rows = []
        contours_out = []
        detected_centers = []

        # 3. Iterate through colors
        for color_name, (lower, upper) in HSV_RANGES.items():
            
            # A. Find Color
            color_mask = cv2.inRange(hsv_bd, lower, upper)
            
            # B. Apply Workspace Mask (Only keep color inside valid table area)
            mask = cv2.bitwise_and(color_mask, workspace_mask)
            
            # Morphological Cleanup
            kernel = np.ones((3,3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                # --- C. GEOMETRY FILTER ---
                rect = cv2.minAreaRect(contour)
                (center_x, center_y), (width, height), angle = rect
                
                if width < 1 or height < 1: continue
                
                aspect_ratio = max(width, height) / min(width, height)
                if aspect_ratio > 3: continue 
                
                rect_area = width * height
                # if rect_area < 50 or rect_area > 3000: continue

                # --- D. DEPTH VERIFICATION ---
                cX, cY = int(center_x), int(center_y)
                
                # Bounds check
                if cY >= self.WorldHeightMap.shape[0] or cX >= self.WorldHeightMap.shape[1]:
                    continue
                    
                z_val = self.WorldHeightMap[cY, cX]

                # Threshold: Block must have real physical height (ignore flat stickers)
                if z_val < 5:
                    continue

                # --- E. DUPLICATE CHECK ---
                is_duplicate = False
                for existing_c in detected_centers:
                    dist = np.sqrt((cX - existing_c[0])**2 + (cY - existing_c[1])**2)
                    if dist < 10: 
                        is_duplicate = True
                        break
                if is_duplicate:
                    continue
                
                # --- F. COLOR CLASSIFICATION (MODIFIED) ---
                # We use your function to confirm the color, instead of trusting the HSV loop variable
                detected_color = self.classifyBlockColorBoard(rgb_bd, contour)

                if detected_color == "unknown":
                    continue

                detected_centers.append((cX, cY))
                
                # Draw
                cv2.drawContours(rgb_bd, [contour], -1, (0, 0, 0), 2)
                cv2.circle(rgb_bd, (cX, cY), 3, (0, 0, 255), -1)
                cv2.putText(rgb_bd, f"{detected_color} {int(z_val)}mm", (cX-30, cY-20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

                # Angle Correction
                if width < height:
                    angle = (90 + angle) % 180
                else:
                    angle = angle % 180
                
                world = self.pixel_to_world(cX, cY)
                
                block_rows.append((detected_color, world[0], world[1], z_val, angle, rect_area))
                contours_out.append(contour)

        self.block_contours = contours_out
        self.block_detections = np.array(block_rows, dtype=object)
        self.VideoFrameWarped = rgb_bd

    def detectBlocksInDepthImage(self):
        """!
        @brief      Detect blocks from depth
        """
        assert self.DepthFrameRaw is not None, "Depth frame not loaded"
        
        # Use WorldHeightMap as the source (Do not modify self.WorldHeightMap directly)
        height_map = self.WorldHeightMap
        
        # --- 1. Workspace Masking ---
        workspace_mask = np.zeros_like(height_map, dtype=np.uint8)
        # Outer bounds (Table)
        cv2.rectangle(workspace_mask, (225, 0), (1170, 670), 255, cv2.FILLED)
        # Inner "hole" (Robot Base)
        cv2.rectangle(workspace_mask, (620, 340), (820, 670), 0, cv2.FILLED)

        # --- 2. Threshold by Height ---
        zmin, zmax = 10, 200
        depth_threshold = cv2.inRange(height_map.astype(np.uint16), zmin, zmax)
        
        # Apply workspace mask to the threshold
        # This ensures we ignore any 'valid height' noise outside the table or near the robot base
        depth_threshold = cv2.bitwise_and(depth_threshold, workspace_mask)

        # Clean up noise
        kernel = np.ones((4,4), np.uint8)
        depth_threshold = cv2.morphologyEx(depth_threshold, cv2.MORPH_CLOSE, kernel)

        # --- 3. Find and Filter Contours ---
        contours, _ = cv2.findContours(depth_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Visualization: Create a BGR image to draw contours on
        self.vis_image = cv2.cvtColor(depth_threshold, cv2.COLOR_GRAY2BGR)
        
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 320 or area > 3000:
                continue
            
            rect = cv2.minAreaRect(contour)
            (center_x, center_y), (width, height), angle = rect
            
            if width < 1 or height < 1:
                continue
                
            aspect_ratio = max(width, height) / min(width, height)
            if aspect_ratio > 3:
                continue
                
            rect_area = width * height
            if rect_area < 50 or rect_area > 3000:
                continue

            if width < height:
                angle = (90 + angle) % 180
            else:
                angle = angle % 180

            # Draw filtered contours on the viz image
            cv2.drawContours(self.vis_image, [contour], -1, (0, 255, 0), 2)

            candidates.append({
                "contour": contour,
                "center": (int(center_x), int(center_y)),
                "angle": angle,
                "rect_area": rect_area
            })

        # CRITICAL FIX: Do NOT overwrite self.WorldHeightMap with the binary threshold.
        # We need self.WorldHeightMap to remain as real depth data (mm) for blockDetector to use.
        # self.WorldHeightMap = depth_threshold  <-- REMOVED THIS LINE

        # Save the thresholded/masked image to a DIFFERENT variable if you need to see it
        self.DepthFrameWarped = self.vis_image

        return candidates

    def detectBlocksInDepthImage(self):
        """!
        @brief      Detect blocks from depth
        """
        assert self.DepthFrameRaw is not None, "Depth frame not loaded"
        
        # Use WorldHeightMap as the source
        height_map = self.WorldHeightMap
        
        # --- 1. Workspace Masking (Logic from detect_blocks) ---
        workspace_mask = np.zeros_like(height_map, dtype=np.uint8)
        # Outer bounds
        cv2.rectangle(workspace_mask, (225, 0), (1170, 670), 255, cv2.FILLED)
        # Inner "hole" (robot base)
        cv2.rectangle(workspace_mask, (620, 340), (820, 670), 0, cv2.FILLED)

        # --- 2. Threshold by Height ---
        zmin, zmax = 10, 200
        depth_threshold = cv2.inRange(height_map.astype(np.uint16), zmin, zmax)
        
        # Apply workspace mask
        depth_threshold = cv2.bitwise_and(depth_threshold, workspace_mask)

        # Clean up noise
        kernel = np.ones((4,4), np.uint8)
        depth_threshold = cv2.morphologyEx(depth_threshold, cv2.MORPH_CLOSE, kernel)

        # --- 3. Find and Filter Contours ---
        contours, _ = cv2.findContours(depth_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.vis_image = cv2.cvtColor(depth_threshold, cv2.COLOR_GRAY2BGR)
        candidates = []
        for contour in contours:
            # Area filter
            area = cv2.contourArea(contour)
            if area < 320 or area > 3000:
                continue
            
            # Geometry filter (Aspect Ratio & Rect Area)
            rect = cv2.minAreaRect(contour)
            (center_x, center_y), (width, height), angle = rect
            
            # Safety check for zero dimensions
            if width < 1 or height < 1:
                continue
                
            aspect_ratio = max(width, height) / min(width, height)
            if aspect_ratio > 3: # Ignore long skinny objects
                continue
                
            rect_area = width * height
            if rect_area < 50 or rect_area > 3000:
                continue

            # Correct the angle for block orientation
            if width < height:
                angle = (90 + angle) % 180
            else:
                angle = angle % 180

            candidates.append({
                "contour": contour,
                "center": (int(center_x), int(center_y)),
                "angle": angle,
                "rect_area": rect_area
            })

        # Optional: Update the map for visualization if needed, otherwise this line can be removed
        self.WorldHeightMap = depth_threshold 

        return candidates

    def projectGridInRGBImage(self):
        """!
        @brief      projects

                    TODO: Use the intrinsic and extrinsic matricies to project the gridpoints 
                    on the board into pixel coordinates. copy self.VideoFrame to self.GridFrame
                    and draw on self.GridFrame the grid intersection points from self.grid_points
                    (hint: use the cv2.circle function to draw circles on the image)
        """
        modified_image = self.VideoFrame.copy()
        if self.hasHcalculate is False:
            return
        modified_image = self.wrap(modified_image)
        all_points = self.grid_points.reshape(2, -1).T  # shape (N, 2)
        for pt in all_points:
            if self.DepthFrameRaw.any()!=0:
                pixel_pos = self.world_to_pixel(np.hstack([pt,[0.0]]))
                # cv2.circle(modified_image, (int(pixel_pos[0]), int(pixel_pos[1])), 3, (0, 255, 0), -1)


        
        # Write your code here

        self.GridFrame = modified_image

    def world_to_pixel(self,Xw):


            K = self.intrinsic_matrix.astype(float)
            D = self.distortion.astype(float)

            T_wc = self.extrinsic_matrix.astype(float)
            R_wc = T_wc[:3, :3]
            t_wc = T_wc[:3, 3:4]

            Xw = np.array(Xw, dtype=float).reshape(3,1)
            Xc = R_wc @ Xw + t_wc
            Zc = float(Xc[2,0])
            uv, _ = cv2.projectPoints(Xc.reshape(1,1,3),
                                    rvec=np.zeros((3,1)),
                                    tvec=np.zeros((3,1)),
                                    cameraMatrix=K,
                                    distCoeffs=D)
            u_raw, v_raw = uv.reshape(-1,2)[0]

            pts = np.array([[[u_raw, v_raw]]], dtype=float)
            if self.hasHcalculate is False:
                self._calculateH()
            pts_H = cv2.perspectiveTransform(pts, self.H)  # note: H, not inv(H)
            u, v = float(pts_H[0,0,0]), float(pts_H[0,0,1])

            return u, v, Zc


    def pixel_to_world(self, u, v):
            K = self.intrinsic_matrix.astype(float)
            D = self.distortion.astype(float)
            T_wc = self.extrinsic_matrix.astype(float)
            R_wc = T_wc[:3,:3]
            t_wc = T_wc[:3,3:4]

            pts = np.array([[[u, v]]], dtype=float)
            # if not self.camera.hasHcalculate:
            #     self.camera._calculateH()
            pts = cv2.perspectiveTransform(pts,np.linalg.inv(self.H))
            p1 = int(pts[0,0,1])
            p2 = int(pts[0,0,0])
            Zc = self.DepthFrameRaw[p1][p2]
            # undistortPoints with P=I gives normalized points (x,y,1)
            undist = cv2.undistortPoints(pts, K, D, P=np.eye(3))
            x_norm, y_norm = undist[0,0,0], undist[0,0,1]

            Xc = np.array([[x_norm * Zc, y_norm * Zc, Zc]], dtype=float).T  # 3x1

            Xw = R_wc.T @ (Xc - t_wc)
            return Xw.ravel()


    def generateWorldHeightMap(self):
        """!
        @brief      Converts the Warped Depth Image (Camera Distance) into a 
                    World Height Map (World Z coordinate for every pixel).
        
        @return     height_map (float32 image where value = World Z in mm/meters)
        """
        if not self.hasHcalculate or self.DepthFrameWarpedRaw is None:
            print("Error: Depth or Homography not ready.")
            return None

        # 1. Get Image Dimensions and Grid
        h, w = self.DepthFrameWarpedRaw.shape
        # Create a grid of (u, v) coordinates for every pixel in the image
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        
        # Stack into (N, 1, 2) format for OpenCV functions
        # This represents every pixel coordinate in the WARPED image
        pixel_coords_warped = np.dstack((u, v)).reshape(-1, 1, 2).astype(float)

        # 2. Un-Warp: Convert Warped Grid (u,v) -> Raw Camera Pixels (u_raw, v_raw)
        # We use the Inverse Homography
        H_inv = np.linalg.inv(self.H)
        pixel_coords_raw = cv2.perspectiveTransform(pixel_coords_warped, H_inv)
        # 3. Un-Distort: Raw Pixels -> Normalized Camera Rays (xn, yn)
        # Reverses the K matrix and D distortion coefficients
        # Output is (x, y) where z=1 in Camera Frame
        normalized_rays = cv2.undistortPoints(
            pixel_coords_raw, 
            self.intrinsic_matrix, 
            self.distortion, 
            P=None 
        )
        
        # 4. Get Depth Values (Zc)
        # Flatten the depth image to match the list of points
        # Ensure we convert to Float (and meters if needed, here keeping original scale)
        # NOTE: If your depth is in mm, the result Z will be in mm.
        zc_values = self.DepthFrameWarpedRaw.flatten().astype(float)
        
        # Filter out 0 depth (invalid) to avoid garbage data
        # We'll mask them out later, but for now let's keep the math simple
        
        # 5. Reconstruct 3D Points in Camera Frame (Xc)
        # Xc = [xn * Zc,  yn * Zc,  Zc]
        # normalized_rays shape is (N, 1, 2) -> split into xn and yn
        xn = normalized_rays[:, 0, 0]
        yn = normalized_rays[:, 0, 1]
        
        # Vectorized multiplication
        Xc_x = xn * zc_values
        Xc_y = yn * zc_values
        Xc_z = zc_values
        
        # Stack into (3, N) matrix for rotation
        Points_C = np.vstack((Xc_x, Xc_y, Xc_z))

        # 6. Transform Camera Frame -> World Frame (Xw)
        # Xw = R_wc.T @ (Xc - t_wc)
        
        T_wc = self.extrinsic_matrix.astype(float)
        R_wc = T_wc[:3, :3]
        t_wc = T_wc[:3, 3:4] # Shape (3, 1)

        # Subtract translation from every point
        Points_C_shifted = Points_C - t_wc
        
        # Apply Inverse Rotation
        Points_W = R_wc.T @ Points_C_shifted
        
        # 7. Reshape back to Image
        # Points_W is (3, N). Row 2 is the Z-coordinate (World Height)
        world_z_flat = Points_W[2, :] 
        
        # Reshape to original (H, W)
        height_map = world_z_flat.reshape(h, w)

        # Optional: Mask out invalid depth pixels (where original was 0)
        height_map[self.DepthFrameWarpedRaw == 0] = 0

        self.WorldHeightMap = height_map
        height_map_visual = cv2.normalize(height_map, None, 0, 255, cv2.NORM_MINMAX)

        # 2. Convert to 8-bit integer (Essential for imwrite/cvtColor)
        height_map_visual = height_map_visual.astype(np.uint8)
        # 3. Now you can convert to RGB and save
        # (Note: Grayscale images are already valid PNGs, but if you specifically need RGB format:)
        height_map_rgb = cv2.cvtColor(height_map_visual, cv2.COLOR_GRAY2RGB)
        # 4. Save
        # cv2.imwrite("/home/rob550-student-am/world_height_map_visual.png", height_map_rgb)


        return height_map

	

    def drawTagsInRGBImage(self, msg):
        """
        @brief      Draw tags from the tag detection                    TODO: Use the tag detections output, to draw the corners/center/tagID of
                    the apriltags on the copy of the RGB image. And output the video to self.TagImageFrame.
                    Message type can be found here: /opt/ros/humble/share/apriltag_msgs/msg                    center of the tag: (detection.centre.x, detection.centre.y) they are floats
                    id of the tag: detection.id
        """
        modified_image = self.VideoFrame.copy()
        # Write your code here
        for detection in msg.detections:                
            corners = detection.corners
            center_x = int(detection.centre.x)
            center_y = int(detection.centre.y)
            tag_id = detection.id                # Convert corner coordinates to a NumPy array of integers for drawing
            corner_points = np.array([
                [int(corners[0].x), int(corners[0].y)],
                [int(corners[1].x), int(corners[1].y)],
                [int(corners[2].x), int(corners[2].y)],
                [int(corners[3].x), int(corners[3].y)]
            ], dtype=np.int32)                # Draw the tag's outline using a polygon
            cv2.polylines(modified_image, [corner_points], isClosed=True, color=(0, 0, 255), thickness=2)                
            # Draw a circle at the center of the tag
            cv2.circle(modified_image, (center_x, center_y), 3, (0, 255, 0), -1)                
            text = f"ID: {tag_id}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 1
            text_color = (255, 0, 0)                
            text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
            text_x = center_x - 20
            text_y = center_y - 20             
            
            cv2.putText(modified_image, text, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)        
    
        
        modified_image = self.wrap(modified_image)
        self.TagImageFrame = modified_image

    def _calculateH(self):
        print("----- Calculating H called -----")
        if self.tag_detections is None:
            return
        src = np.zeros((4,2))
        for detection in self.tag_detections.detections:
            center_x = int(detection.centre.x)
            center_y = int(detection.centre.y)
            src[detection.id - 1, :] = [center_x, center_y]
        
        x_dim = 1280
        y_dim = 720
        # scale = 1000
        scale =0.4
        x_off = 0.35*x_dim
        y_off = 0.3*y_dim
        dst = np.array([x_off                , y_off+0.6*scale*x_dim, 
                        x_off + scale * x_dim, y_off+0.6*scale*x_dim,
                        x_off +scale * x_dim , y_off,
                        x_off                , y_off]).reshape((4,2))

 
        self.H = cv2.findHomography(src, dst)[0]
        print(f"H:\n{self.H}")
        self.hasHcalculate = True

        
    def wrap(self,image):
        # if not self.hasHcalculate:
        #     self._calculateH()
        new_img = cv2.warpPerspective(image, self.H, (image.shape[1], image.shape[0]))

        return new_img


    def warpDepthImage(self):
        """!
        @brief      Warps both the raw and colorized depth images to the grid.
        """
        if self.DepthFrameRaw is None or not hasattr(self, 'H'):
            return

        h, w = self.DepthFrameRaw.shape[:2]

        # WARP RAW DATA (Use INTER_NEAREST to preserve actual distance values)
        self.DepthFrameWarpedRaw = cv2.warpPerspective(
            self.DepthFrameRaw, 
            self.H, 
            (w, h), 
            flags=cv2.INTER_NEAREST
        )

        # WARP VISUALIZATION (Use INTER_LINEAR or NEAREST)
        self.DepthFrameWarpedRGB = cv2.warpPerspective(
            self.DepthFrameRGB, 
            self.H, 
            (w, h), 
            flags=cv2.INTER_LINEAR # Linear is okay for visualization smoothing
        )



class ImageListener(Node):
    def __init__(self, topic, camera):
        super().__init__('image_listener')
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, topic, self.callback, 10)
        self.camera = camera

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
        except CvBridgeError as e:
            print(e)
        self.camera.VideoFrame = cv_image


class TagDetectionListener(Node):
    def __init__(self, topic, camera):
        super().__init__('tag_detection_listener')
        self.topic = topic
        self.tag_sub = self.create_subscription(
            AprilTagDetectionArray,
            topic,
            self.callback,
            10
        )
        self.camera = camera

    def callback(self, msg):
        self.camera.tag_detections = msg
        if np.any(self.camera.VideoFrame != 0):
            self.camera.drawTagsInRGBImage(msg)


class CameraInfoListener(Node):
    def __init__(self, topic, camera):
        super().__init__('camera_info_listener')  
        self.topic = topic
        self.tag_sub = self.create_subscription(CameraInfo, topic, self.callback, 10)
        self.camera = camera

    def callback(self, data):
        self.camera.intrinsic_matrix = np.reshape(data.k, (3, 3))


        self.camera.distortion = np.reshape(data.d, (5,1))
        # print(self.camera.intrinsic_matrix)


class DepthListener(Node):
    def __init__(self, topic, camera):
        super().__init__('depth_listener')
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, topic, self.callback, 10)
        self.camera = camera

    def callback(self, data):
        try:
            cv_depth = self.bridge.imgmsg_to_cv2(data, data.encoding)
            # cv_depth = cv2.rotate(cv_depth, cv2.ROTATE_180)
        except CvBridgeError as e:
            print(e)
        self.camera.DepthFrameRaw = cv_depth
        # self.camera.DepthFrameRaw = self.camera.DepthFrameRaw / 2
        self.camera.ColorizeDepthFrame()


class VideoThread(QThread):
    updateFrame = pyqtSignal(QImage, QImage, QImage, QImage)

    def __init__(self, camera, parent=None):
        QThread.__init__(self, parent=parent)
        self.camera = camera
        image_topic = "/camera/color/image_raw"
        depth_topic = "/camera/aligned_depth_to_color/image_raw"
        camera_info_topic = "/camera/color/camera_info"
        tag_detection_topic = "/detections"
        image_listener = ImageListener(image_topic, self.camera)
        depth_listener = DepthListener(depth_topic, self.camera)
        camera_info_listener = CameraInfoListener(camera_info_topic,
                                                  self.camera)
        tag_detection_listener = TagDetectionListener(tag_detection_topic,
                                                      self.camera)
        
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(image_listener)
        self.executor.add_node(depth_listener)
        self.executor.add_node(camera_info_listener)
        self.executor.add_node(tag_detection_listener)

    def run(self):
        if __name__ == '__main__':
            cv2.namedWindow("Image window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Depth window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Tag window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Grid window", cv2.WINDOW_NORMAL)
            time.sleep(0.5)
        self.has_saved_grid = False
        try:
            while rclpy.ok():
                start_time = time.time()
                rgb_frame = self.camera.convertQtVideoFrame()
                depth_frame = self.camera.convertQtDepthFrame()
                tag_frame = self.camera.convertQtTagImageFrame()
                self.camera.projectGridInRGBImage()
                grid_frame = self.camera.convertQtGridFrame()

                                        # Only save if H is calculated and we haven't saved yet
                # if self.camera.hasHcalculate and not self.has_saved_grid:
                #     save_path = "/home/rob550-student-am/block_detection/Grid.png"
                #     cv2.imwrite(save_path, cv2.cvtColor(self.camera.GridFrame, cv2.COLOR_RGB2BGR))
                #     print(f"Grid image saved to {save_path}")
                #     self.has_saved_grid = True # Set flag to True so it never

                if ((rgb_frame != None) & (depth_frame != None)):
                    self.updateFrame.emit(
                        rgb_frame, depth_frame, tag_frame, grid_frame)
                self.executor.spin_once() # comment this out when run this file alone.
                self.camera.processVideoFrame()
                elapsed_time = time.time() - start_time
                sleep_time = max(0.03 - elapsed_time, 0)
                time.sleep(sleep_time)

                if __name__ == '__main__':
                    cv2.imshow(
                        "Image window",
                        cv2.cvtColor(self.camera.VideoFrame, cv2.COLOR_RGB2BGR))
                    cv2.imshow("Depth window", self.camera.DepthFrameRGB)
                    cv2.imshow(
                        "Tag window",
                        cv2.cvtColor(self.camera.TagImageFrame, cv2.COLOR_RGB2BGR))
                    cv2.imshow("Grid window",
                        cv2.cvtColor(self.camera.GridFrame, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(3)


                    time.sleep(0.03)
        except KeyboardInterrupt:
            pass
        
        self.executor.shutdown()
        

def main(args=None):
    rclpy.init(args=args)
    try:
        camera = Camera()
        videoThread = VideoThread(camera)
        videoThread.start()
        try:
            videoThread.executor.spin()
        finally:
            videoThread.executor.shutdown()
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()