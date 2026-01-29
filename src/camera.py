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
        self.tag_detections = np.array([])
        self.tag_locations = [[-250, -25], [250, -25], [250, 275], [-250, 275]]
        """ block info """
        self.block_contours = np.array([])
        self.block_detections = np.array([])

        self.H = np.eye(3)
        self.hasHcalculate = False

    def processVideoFrame(self):
        """!
        @brief      Process a video frame
        """
        cv2.drawContours(self.VideoFrame, self.block_contours, -1,
                         (255, 0, 255), 3)

    def ColorizeDepthFrame(self):
        """!
        @brief Converts frame to colormaped formats in HSV and RGB
        """
        self.DepthFrameHSV[..., 0] = self.DepthFrameRaw >> 1
        self.DepthFrameHSV[..., 1] = 0xFF
        self.DepthFrameHSV[..., 2] = 0x9F
        self.DepthFrameRGB = cv2.cvtColor(self.DepthFrameHSV,
                                          cv2.COLOR_HSV2RGB)

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
            frame = cv2.resize(self.GridFrame, (1280, 720))
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

    def blockDetector(self):
        """!
        @brief      Detect blocks from rgb

                    TODO: Implement your block detector here. You will need to locate blocks in 3D space and put their XYZ
                    locations in self.block_detections
        """
        pass

    def detectBlocksInDepthImage(self):
        """!
        @brief      Detect blocks from depth

                    TODO: Implement a blob detector to find blocks in the depth image
        """
        pass

    def projectGridInRGBImage(self):
        """!
        @brief      projects

                    TODO: Use the intrinsic and extrinsic matricies to project the gridpoints 
                    on the board into pixel coordinates. copy self.VideoFrame to self.GridFrame
                    and draw on self.GridFrame the grid intersection points from self.grid_points
                    (hint: use the cv2.circle function to draw circles on the image)
        """
        modified_image = self.VideoFrame.copy()
        # Write your code here

        self.GridFrame = modified_image
     
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
    
        
        # modified_image = self.wrap(modified_image)
        self.TagImageFrame = modified_image

    # def _calculateH(self):
    #     print("----- Calculating H called -----")
    #     if self.tag_detections is None:
    #         return
    #     src = np.zeros((4,2))
    #     for detection in self.tag_detections.detections:
    #         center_x = int(detection.centre.x)
    #         center_y = int(detection.centre.y)
    #         src[detection.id - 1, :] = [center_x, center_y]
        
    #     x_dim = 1280
    #     y_dim = 720
    #     # scale = 1000
    #     scale =0.4
    #     x_off = 0.35*x_dim
    #     y_off = 0.3*y_dim
    #     dst = np.array([x_off                , y_off+0.6*scale*x_dim, 
    #                     x_off + scale * x_dim, y_off+0.6*scale*x_dim,
    #                     x_off +scale * x_dim , y_off,
    #                     x_off                , y_off]).reshape((4,2))

 
    #     self.H = cv2.findHomography(src, dst)[0]
    #     # print(f"H:\n{self.H}")
    #     self.hasHcalculate = True

        
    # def wrap(self,image):
    #     # if not self.hasHcalculate:
    #     #     self._calculateH()
    #     new_img = cv2.warpPerspective(image, self.H, (image.shape[1], image.shape[0]))
    #     cv2.imwrite("test.jpg",new_img)
    #     return new_img

    def pixel_to_camera(self, u: float, v: float, d_mm: float):
        """
        Back-project pixel (u,v) with depth d_mm (mm) into camera coordinates (mm)
        using explicit K^{-1}.
        """
        if d_mm is None or d_mm <= 0:
            return None

        K = self.intrinsic_matrix.astype(float)  # 3x3
        K_inv = np.linalg.inv(K)

        uv1 = np.array([float(u), float(v), 1.0], dtype=float)  # homogeneous pixel
        ray = K_inv @ uv1  # normalized ray: [x/z, y/z, 1]

        z = float(d_mm)  # mm
        Xc = ray * z     # [x, y, z] in mm
        return Xc


    def camera_to_world(self, p_cam_mm: np.ndarray):
        """Map camera-frame point (mm) to world-frame (mm) using T_wc."""
        p_h = np.ones((4, 1), dtype=float)
        p_h[:3, 0] = p_cam_mm.reshape(3)
        p_w = (np.linalg.inv(self.extrinsic_matrix) @ p_h)[:3, 0]
        return p_w

    def pixel_to_world(self, u: int, v: int):
        """!
        @brief      Map a pixel (u,v) to world coordinates using depth + calibration.

        Depth image from the ROS driver is a uint16 in millimeters.
        Returns (3,) in mm.
        """
        if self.DepthFrameRaw is None or self.DepthFrameRaw.size == 0:
            return None
        if v < 0 or v >= self.DepthFrameRaw.shape[0] or u < 0 or u >= self.DepthFrameRaw.shape[1]:
            return None
        d = float(self.DepthFrameRaw[v, u])
        p_cam = self.pixel_to_camera(u, v, d)
        if p_cam is None:
            return None
        return self.camera_to_world(p_cam)

    # def world_to_pixel(self, p_world_mm: np.ndarray):
    #     """!
    #     @brief      Project a world point (mm) into pixel coordinates (u,v).

    #     Uses T_cw = inv(T_wc) and pinhole intrinsics.
    #     """
    #     try:
    #         T_cw = np.linalg.inv(self.extrinsic_matrix)
    #     except np.linalg.LinAlgError:
    #         return None

    #     p_h = np.ones((4, 1), dtype=float)
    #     p_h[:3, 0] = p_world_mm.reshape(3)
    #     p_c = (T_cw @ p_h)[:3, 0]
    #     Xc, Yc, Zc = float(p_c[0]), float(p_c[1]), float(p_c[2])
    #     if Zc <= 1e-6:
    #         return None

    #     K = self.intrinsic_matrix
    #     fx, fy = float(K[0, 0]), float(K[1, 1])
    #     cx, cy = float(K[0, 2]), float(K[1, 2])
    #     u = int(round((Xc / Zc) * fx + cx))
    #     v = int(round((Yc / Zc) * fy + cy))
    #     return (u, v)




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
        try:
            while rclpy.ok():
                start_time = time.time()
                rgb_frame = self.camera.convertQtVideoFrame()
                depth_frame = self.camera.convertQtDepthFrame()
                tag_frame = self.camera.convertQtTagImageFrame()
                self.camera.projectGridInRGBImage()
                grid_frame = self.camera.convertQtGridFrame()
                if ((rgb_frame != None) & (depth_frame != None)):
                    self.updateFrame.emit(
                        rgb_frame, depth_frame, tag_frame, grid_frame)
                self.executor.spin_once() # comment this out when run this file alone.
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