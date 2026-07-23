#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
完整图生文节点：深度立牌 + RGB白色过滤 + YOLO障碍物排除 + 地图坐标核对 + 千问。

已知立牌物理尺寸：宽 0.15m，高 0.20m。
默认只搜索 RGB/深度图的下半部分，并允许立牌有角度倾斜。

新增过滤：
  1. 排除“高白色占比且几乎没有纹理”的全白候选；
  2. 订阅 /racing_obstacle_detection；
  3. YOLO当前没有障碍物时，不执行障碍物排除；
  4. YOLO来自下相机时，只把“当前存在障碍物”作为门控，再用深度候选的0.20m×0.30m尺寸和三角轮廓排除；不跨相机比较像素框。

订阅：
  /aurora/rgb/image_raw       sensor_msgs/msg/Image
  /aurora/depth/image_raw     sensor_msgs/msg/Image
  /aurora/rgb/camera_info     sensor_msgs/msg/CameraInfo
  /racing_obstacle_detection   ai_msgs/msg/PerceptionTargets

发布：
  /vision_board_depth_debug/compressed  sensor_msgs/msg/CompressedImage
  /vision_board_depth_result            std_msgs/msg/String

检测思想：
  1. 在下半图按深度直方图寻找多个可能深度层；
  2. 在每个深度层内找连通区域；
  3. 对区域拟合 z=a*x+b*y+c，允许立牌倾斜；
  4. 用深度和相机内参估算区域实际长短边；
  5. 与 0.15m × 0.20m 比较；
  6. 连续多帧命中后确认；
  7. 用RGB排除全白且无纹理的候选；
  8. 有YOLO障碍物时，判断候选是否为0.20m×0.30m三角障碍物；
  9. 把候选和排除原因画到压缩调试图；只有YOLO与RGBD来自同一相机时才绘制其像素框。
"""

import os
import sys
import json
import math
import time
import base64
import bisect
import threading
from collections import deque
from typing import Dict, List, Optional, Tuple

import cv2
import message_filters
import numpy as np
import rclpy

from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSDurabilityPolicy,
    QoSReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from ai_msgs.msg import PerceptionTargets
from openai import OpenAI


Candidate = Dict[str, object]


class VisionBoardDepthTestNode(Node):
    def __init__(self):
        super().__init__('vision_board_llm_node')

        # ------------------------- 话题 -------------------------
        self.declare_parameter('rgb_topic', '/aurora/rgb/image_raw')
        self.declare_parameter('depth_topic', '/aurora/depth/image_raw')
        self.declare_parameter('camera_info_topic', '/aurora/rgb/camera_info')
        self.declare_parameter(
            'debug_topic',
            '/vision_board_depth_debug/compressed',
        )
        self.declare_parameter(
            'result_topic',
            '/vision_board_depth_result',
        )
        self.declare_parameter(
            'obstacle_topic',
            '/racing_obstacle_detection',
        )

        # ---------------------- 立牌与搜索区 ----------------------
        self.declare_parameter('board_width_m', 0.15)
        self.declare_parameter('board_height_m', 0.20)

        # OpenCV 图像坐标原点在左上角，y 向下增大。
        # 0.45 表示从高度 45% 的位置开始，一直搜索到图像底部。
        self.declare_parameter('search_y_start_ratio', 0.45)
        self.declare_parameter('search_x_min_ratio', 0.05)
        self.declare_parameter('search_x_max_ratio', 0.95)

        self.declare_parameter('min_depth_m', 0.30)
        self.declare_parameter('max_depth_m', 3.00)

        # ---------------------- 深度分层参数 ----------------------
        self.declare_parameter('depth_hist_bin_m', 0.05)
        self.declare_parameter('depth_window_half_m', 0.12)
        self.declare_parameter('max_depth_layers', 12)
        self.declare_parameter('morph_kernel_size', 5)

        self.declare_parameter('min_component_area_px', 160)
        self.declare_parameter('max_component_area_ratio', 0.35)
        self.declare_parameter('min_rectangularity', 0.25)
        self.declare_parameter('min_valid_ratio', 0.45)

        # 倾斜后投影尺寸会变小，因此尺寸下限需要放宽。
        self.declare_parameter('dimension_min_scale', 0.42)
        self.declare_parameter('dimension_max_scale', 1.70)
        self.declare_parameter('area_ratio_min', 0.12)
        self.declare_parameter('area_ratio_max', 2.80)

        # 平面拟合残差。立牌有角度不影响，因为拟合允许斜率。
        self.declare_parameter('max_plane_rmse_m', 0.040)
        self.declare_parameter('max_plane_mad_m', 0.030)
        self.declare_parameter('max_candidate_score', 2.80)

        # ------------------------- 多帧确认 -------------------------
        self.declare_parameter('confirm_frames', 3)
        self.declare_parameter('lost_frames', 3)
        self.declare_parameter('track_center_tolerance_px', 110.0)
        self.declare_parameter('track_depth_tolerance_m', 0.40)

        self.declare_parameter('jpeg_quality', 70)

        # -------------------- RGB全白候选排除 --------------------
        # 白色像素：HSV饱和度低、亮度高。
        self.declare_parameter('white_saturation_max', 38)
        self.declare_parameter('white_value_min', 210)

        # 只有同时满足“白色占比高 + 灰度变化小 + 边缘少”才判为空白，
        # 避免误删白底但有文字/图案的立牌。
        self.declare_parameter('all_white_ratio_threshold', 0.90)
        self.declare_parameter('all_white_gray_std_max', 22.0)
        self.declare_parameter('all_white_edge_ratio_max', 0.055)

        # -------------------- YOLO障碍物排除 --------------------
        self.declare_parameter('obstacle_confidence_threshold', 0.60)
        self.declare_parameter('obstacle_timeout_sec', 0.60)

        # 障碍物真实尺寸：0.20m × 0.30m。
        self.declare_parameter('obstacle_width_m', 0.20)
        self.declare_parameter('obstacle_height_m', 0.30)
        self.declare_parameter('obstacle_dimension_min_scale', 0.55)
        self.declare_parameter('obstacle_dimension_max_scale', 1.55)

        # 深度候选框与YOLO障碍物框的空间匹配阈值。
        self.declare_parameter('obstacle_iou_threshold', 0.12)
        self.declare_parameter('obstacle_candidate_coverage_threshold', 0.28)
        self.declare_parameter('obstacle_strong_coverage_threshold', 0.55)
        self.declare_parameter('obstacle_box_expand_ratio', 1.12)

        # 当前 /racing_obstacle_detection 来自下相机，而本节点使用 Aurora RGBD。
        # 两个相机的像素坐标不能直接比较。默认 False：
        #   YOLO只负责说明“当前确实看到了障碍物”；
        #   是否排除当前深度候选，继续由物理尺寸和三角轮廓共同决定。
        # 只有将YOLO切换到与RGBD完全相同且已对齐的图像源时才设为 True。
        self.declare_parameter('obstacle_boxes_same_camera', False)

        # 不规则三角形轮廓通常顶点较少且矩形填充率低。
        self.declare_parameter('obstacle_triangle_max_vertices', 7)
        self.declare_parameter('obstacle_triangle_rectangularity_max', 0.76)

        # ------------------------- 读取参数 -------------------------
        self.rgb_topic = str(self.get_parameter('rgb_topic').value)
        self.depth_topic = str(self.get_parameter('depth_topic').value)
        self.camera_info_topic = str(
            self.get_parameter('camera_info_topic').value
        )
        self.debug_topic = str(self.get_parameter('debug_topic').value)
        self.result_topic = str(self.get_parameter('result_topic').value)
        self.obstacle_topic = str(
            self.get_parameter('obstacle_topic').value
        )

        self.board_width_m = float(
            self.get_parameter('board_width_m').value
        )
        self.board_height_m = float(
            self.get_parameter('board_height_m').value
        )

        self.search_y_start_ratio = self._clamp(
            float(self.get_parameter('search_y_start_ratio').value),
            0.0,
            0.95,
        )
        self.search_x_min_ratio = self._clamp(
            float(self.get_parameter('search_x_min_ratio').value),
            0.0,
            0.95,
        )
        self.search_x_max_ratio = self._clamp(
            float(self.get_parameter('search_x_max_ratio').value),
            self.search_x_min_ratio + 0.01,
            1.0,
        )

        self.min_depth_m = max(
            0.05,
            float(self.get_parameter('min_depth_m').value),
        )
        self.max_depth_m = max(
            self.min_depth_m + 0.05,
            float(self.get_parameter('max_depth_m').value),
        )

        self.depth_hist_bin_m = max(
            0.01,
            float(self.get_parameter('depth_hist_bin_m').value),
        )
        self.depth_window_half_m = max(
            0.02,
            float(self.get_parameter('depth_window_half_m').value),
        )
        self.max_depth_layers = max(
            1,
            int(self.get_parameter('max_depth_layers').value),
        )

        self.morph_kernel_size = max(
            1,
            int(self.get_parameter('morph_kernel_size').value),
        )
        if self.morph_kernel_size % 2 == 0:
            self.morph_kernel_size += 1

        self.min_component_area_px = max(
            20,
            int(self.get_parameter('min_component_area_px').value),
        )
        self.max_component_area_ratio = self._clamp(
            float(self.get_parameter('max_component_area_ratio').value),
            0.01,
            1.0,
        )
        self.min_rectangularity = self._clamp(
            float(self.get_parameter('min_rectangularity').value),
            0.01,
            1.0,
        )
        self.min_valid_ratio = self._clamp(
            float(self.get_parameter('min_valid_ratio').value),
            0.01,
            1.0,
        )

        self.dimension_min_scale = max(
            0.05,
            float(self.get_parameter('dimension_min_scale').value),
        )
        self.dimension_max_scale = max(
            self.dimension_min_scale + 0.05,
            float(self.get_parameter('dimension_max_scale').value),
        )
        self.area_ratio_min = max(
            0.01,
            float(self.get_parameter('area_ratio_min').value),
        )
        self.area_ratio_max = max(
            self.area_ratio_min + 0.01,
            float(self.get_parameter('area_ratio_max').value),
        )

        self.max_plane_rmse_m = max(
            0.001,
            float(self.get_parameter('max_plane_rmse_m').value),
        )
        self.max_plane_mad_m = max(
            0.001,
            float(self.get_parameter('max_plane_mad_m').value),
        )
        self.max_candidate_score = max(
            0.1,
            float(self.get_parameter('max_candidate_score').value),
        )

        self.confirm_frames = max(
            1,
            int(self.get_parameter('confirm_frames').value),
        )
        self.lost_frames = max(
            1,
            int(self.get_parameter('lost_frames').value),
        )
        self.track_center_tolerance_px = max(
            1.0,
            float(
                self.get_parameter('track_center_tolerance_px').value
            ),
        )
        self.track_depth_tolerance_m = max(
            0.01,
            float(
                self.get_parameter('track_depth_tolerance_m').value
            ),
        )
        self.jpeg_quality = int(
            self._clamp(
                int(self.get_parameter('jpeg_quality').value),
                1,
                100,
            )
        )

        self.white_saturation_max = int(
            self._clamp(
                int(self.get_parameter('white_saturation_max').value),
                0,
                255,
            )
        )
        self.white_value_min = int(
            self._clamp(
                int(self.get_parameter('white_value_min').value),
                0,
                255,
            )
        )
        self.all_white_ratio_threshold = self._clamp(
            float(
                self.get_parameter('all_white_ratio_threshold').value
            ),
            0.0,
            1.0,
        )
        self.all_white_gray_std_max = max(
            0.0,
            float(
                self.get_parameter('all_white_gray_std_max').value
            ),
        )
        self.all_white_edge_ratio_max = self._clamp(
            float(
                self.get_parameter('all_white_edge_ratio_max').value
            ),
            0.0,
            1.0,
        )

        self.obstacle_confidence_threshold = self._clamp(
            float(
                self.get_parameter(
                    'obstacle_confidence_threshold'
                ).value
            ),
            0.0,
            1.0,
        )
        self.obstacle_timeout_sec = max(
            0.05,
            float(
                self.get_parameter('obstacle_timeout_sec').value
            ),
        )
        self.obstacle_width_m = max(
            0.01,
            float(self.get_parameter('obstacle_width_m').value),
        )
        self.obstacle_height_m = max(
            0.01,
            float(self.get_parameter('obstacle_height_m').value),
        )
        self.obstacle_dimension_min_scale = max(
            0.05,
            float(
                self.get_parameter(
                    'obstacle_dimension_min_scale'
                ).value
            ),
        )
        self.obstacle_dimension_max_scale = max(
            self.obstacle_dimension_min_scale + 0.05,
            float(
                self.get_parameter(
                    'obstacle_dimension_max_scale'
                ).value
            ),
        )
        self.obstacle_iou_threshold = self._clamp(
            float(
                self.get_parameter('obstacle_iou_threshold').value
            ),
            0.0,
            1.0,
        )
        self.obstacle_candidate_coverage_threshold = self._clamp(
            float(
                self.get_parameter(
                    'obstacle_candidate_coverage_threshold'
                ).value
            ),
            0.0,
            1.0,
        )
        self.obstacle_strong_coverage_threshold = self._clamp(
            float(
                self.get_parameter(
                    'obstacle_strong_coverage_threshold'
                ).value
            ),
            0.0,
            1.0,
        )
        self.obstacle_box_expand_ratio = max(
            1.0,
            float(
                self.get_parameter('obstacle_box_expand_ratio').value
            ),
        )
        self.obstacle_boxes_same_camera = bool(
            self.get_parameter('obstacle_boxes_same_camera').value
        )
        self.obstacle_triangle_max_vertices = max(
            3,
            int(
                self.get_parameter(
                    'obstacle_triangle_max_vertices'
                ).value
            ),
        )
        self.obstacle_triangle_rectangularity_max = self._clamp(
            float(
                self.get_parameter(
                    'obstacle_triangle_rectangularity_max'
                ).value
            ),
            0.05,
            1.0,
        )

        # ------------------------- 状态 -------------------------
        self.bridge = CvBridge()
        self.state_lock = threading.RLock()

        self.fx = 0.0
        self.fy = 0.0
        self.cx = 0.0
        self.cy = 0.0
        self.intrinsics_ready = False

        self.frame_count = 0
        self.consecutive_detect_count = 0
        self.consecutive_lost_count = 0
        self.board_confirmed = False
        self.last_candidate: Optional[Candidate] = None
        self.last_log_time = 0.0

        # 最新YOLO障碍物框：(x1, y1, x2, y2, confidence)。
        self.latest_obstacle_boxes: List[Tuple[float, float, float, float, float]] = []
        self.latest_obstacle_time = 0.0
        self.has_received_obstacle_message = False

        self.white_reject_count = 0
        self.obstacle_reject_count = 0

        # ------------------------- ROS 通信 -------------------------
        self.info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            qos_profile_sensor_data,
        )

        self.obstacle_sub = self.create_subscription(
            PerceptionTargets,
            self.obstacle_topic,
            self.obstacle_callback,
            qos_profile_sensor_data,
        )

        self.rgb_sub = message_filters.Subscriber(
            self,
            Image,
            self.rgb_topic,
            qos_profile=qos_profile_sensor_data,
        )
        self.depth_sub = message_filters.Subscriber(
            self,
            Image,
            self.depth_topic,
            qos_profile=qos_profile_sensor_data,
        )

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=12,
            slop=0.08,
        )
        self.sync.registerCallback(self.synced_callback)

        self.debug_pub = self.create_publisher(
            CompressedImage,
            self.debug_topic,
            2,
        )
        self.result_pub = self.create_publisher(
            String,
            self.result_topic,
            10,
        )

        self.get_logger().info('=' * 74)
        self.get_logger().info('图生文立牌深度测试节点已启动')
        self.get_logger().info(f'RGB: {self.rgb_topic}')
        self.get_logger().info(f'Depth: {self.depth_topic}')
        self.get_logger().info(f'CameraInfo: {self.camera_info_topic}')
        self.get_logger().info(f'YOLO障碍物: {self.obstacle_topic}')
        self.get_logger().info(
            f'立牌尺寸: {self.board_width_m:.3f}m × '
            f'{self.board_height_m:.3f}m'
        )
        self.get_logger().info(
            f'搜索区域: X={self.search_x_min_ratio:.2f}~'
            f'{self.search_x_max_ratio:.2f}, '
            f'Y={self.search_y_start_ratio:.2f}~1.00'
        )
        self.get_logger().info(f'调试图: {self.debug_topic}')
        self.get_logger().info(f'结果JSON: {self.result_topic}')
        self.get_logger().info(
            '过滤：全白无纹理候选 + YOLO空间重叠/障碍物尺寸/三角轮廓'
        )
        self.get_logger().info('=' * 74)

    @staticmethod
    def _clamp(value, lower, upper):
        return max(lower, min(value, upper))

    def camera_info_callback(self, msg: CameraInfo):
        fx = float(msg.k[0])
        fy = float(msg.k[4])
        cx = float(msg.k[2])
        cy = float(msg.k[5])

        if fx <= 0.0 or fy <= 0.0:
            self.get_logger().warning(
                'CameraInfo 中 fx/fy 无效',
                throttle_duration_sec=2.0,
            )
            return

        with self.state_lock:
            first = not self.intrinsics_ready
            self.fx = fx
            self.fy = fy
            self.cx = cx
            self.cy = cy
            self.intrinsics_ready = True

        if first:
            self.get_logger().info(
                f'相机内参已收到: fx={fx:.2f}, fy={fy:.2f}, '
                f'cx={cx:.2f}, cy={cy:.2f}'
            )

    def obstacle_callback(self, msg: PerceptionTargets):
        boxes: List[Tuple[float, float, float, float, float]] = []

        for target in msg.targets:
            if str(target.type).strip() != 'construction_cone':
                continue

            for roi in target.rois:
                confidence = float(roi.confidence)
                if confidence < self.obstacle_confidence_threshold:
                    continue

                rect = roi.rect
                x1 = float(rect.x_offset)
                y1 = float(rect.y_offset)
                x2 = x1 + float(rect.width)
                y2 = y1 + float(rect.height)

                if x2 <= x1 or y2 <= y1:
                    continue

                boxes.append((x1, y1, x2, y2, confidence))

        with self.state_lock:
            self.latest_obstacle_boxes = boxes
            self.latest_obstacle_time = time.monotonic()
            self.has_received_obstacle_message = True

    def obstacle_snapshot(self):
        with self.state_lock:
            boxes = list(self.latest_obstacle_boxes)
            last_time = self.latest_obstacle_time
            received = self.has_received_obstacle_message

        if not received:
            return [], 'not_received', float('inf')

        age = time.monotonic() - last_time
        if age > self.obstacle_timeout_sec:
            return [], 'stale', age

        if not boxes:
            return [], 'no_obstacle', age

        return boxes, 'active', age

    @staticmethod
    def rectangle_intersection_metrics(box_a, box_b):
        ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
        bx1, by1, bx2, by2 = [float(v) for v in box_b]

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        inter_w = max(0.0, ix2 - ix1)
        inter_h = max(0.0, iy2 - iy1)
        intersection = inter_w * inter_h

        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - intersection

        iou = intersection / union if union > 1e-9 else 0.0
        coverage_a = intersection / area_a if area_a > 1e-9 else 0.0
        return iou, coverage_a

    def expand_box(self, box):
        x1, y1, x2, y2 = [float(v) for v in box]
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        half_w = 0.5 * (x2 - x1) * self.obstacle_box_expand_ratio
        half_h = 0.5 * (y2 - y1) * self.obstacle_box_expand_ratio
        return (
            cx - half_w,
            cy - half_h,
            cx + half_w,
            cy + half_h,
        )

    def depth_to_meters(self, msg: Image) -> Optional[np.ndarray]:
        try:
            depth = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='passthrough',
            )
        except Exception as exc:
            self.get_logger().error(
                f'深度图转换失败: {exc}',
                throttle_duration_sec=1.0,
            )
            return None

        depth = np.asarray(depth)
        if depth.ndim != 2:
            self.get_logger().error(
                f'深度图不是单通道: shape={depth.shape}',
                throttle_duration_sec=1.0,
            )
            return None

        encoding = str(getattr(msg, 'encoding', '')).lower()

        if encoding in ('16uc1', 'mono16', '16sc1') or depth.dtype == np.uint16:
            depth_m = depth.astype(np.float32) * 0.001
        elif encoding in ('32fc1', '32fc') or depth.dtype in (np.float32, np.float64):
            depth_m = depth.astype(np.float32)
        else:
            depth_m = depth.astype(np.float32)
            positive = depth_m[np.isfinite(depth_m) & (depth_m > 0)]
            if positive.size and float(np.median(positive)) > 20.0:
                depth_m *= 0.001

        depth_m[~np.isfinite(depth_m)] = 0.0
        return depth_m

    def depth_layer_centers(self, roi_depth: np.ndarray) -> List[float]:
        valid = roi_depth[
            np.isfinite(roi_depth)
            & (roi_depth >= self.min_depth_m)
            & (roi_depth <= self.max_depth_m)
        ]
        if valid.size < self.min_component_area_px:
            return []

        count = max(
            1,
            int(math.ceil((self.max_depth_m - self.min_depth_m) / self.depth_hist_bin_m)),
        )
        histogram, edges = np.histogram(
            valid,
            bins=count,
            range=(self.min_depth_m, self.max_depth_m),
        )

        nonzero = np.flatnonzero(histogram > 0)
        if nonzero.size == 0:
            return []

        order = nonzero[np.argsort(histogram[nonzero])[::-1]]
        centers: List[float] = []
        for index in order:
            center = float((edges[index] + edges[index + 1]) * 0.5)
            if any(
                abs(center - old) < self.depth_hist_bin_m * 1.5
                for old in centers
            ):
                continue
            centers.append(center)
            if len(centers) >= self.max_depth_layers:
                break
        return centers

    @staticmethod
    def fit_depth_plane(
        xs: np.ndarray,
        ys: np.ndarray,
        zs: np.ndarray,
    ) -> Tuple[float, float]:
        """拟合 z=a*x+b*y+c，允许立牌本身有角度。"""
        if zs.size < 12:
            return float('inf'), float('inf')

        if zs.size > 1600:
            step = int(math.ceil(zs.size / 1600.0))
            xs = xs[::step]
            ys = ys[::step]
            zs = zs[::step]

        design = np.column_stack(
            [
                xs.astype(np.float64),
                ys.astype(np.float64),
                np.ones(xs.size, dtype=np.float64),
            ]
        )
        values = zs.astype(np.float64)

        try:
            coeffs, _, _, _ = np.linalg.lstsq(
                design,
                values,
                rcond=None,
            )
            residual = values - design @ coeffs
            rmse = float(np.sqrt(np.mean(residual * residual)))
            mad = float(
                np.median(np.abs(residual - np.median(residual)))
            )
            return rmse, mad
        except np.linalg.LinAlgError:
            return float('inf'), float('inf')

    def evaluate_contour(
        self,
        contour: np.ndarray,
        depth_roi: np.ndarray,
        origin_x: int,
        origin_y: int,
        search_area_px: int,
        fx: float,
        fy: float,
    ) -> Optional[Candidate]:
        contour_area = float(cv2.contourArea(contour))
        if contour_area < self.min_component_area_px:
            return None
        if contour_area > search_area_px * self.max_component_area_ratio:
            return None

        x, y, width, height = cv2.boundingRect(contour)
        if width < 8 or height < 8:
            return None

        rotated_rect = cv2.minAreaRect(contour)
        side_a_px = float(rotated_rect[1][0])
        side_b_px = float(rotated_rect[1][1])
        if side_a_px < 5.0 or side_b_px < 5.0:
            return None

        rotated_area = side_a_px * side_b_px
        if rotated_area <= 1.0:
            return None

        rectangularity = contour_area / rotated_area
        if rectangularity < self.min_rectangularity:
            return None

        local_mask = np.zeros((height, width), dtype=np.uint8)
        shifted = contour.copy()
        shifted[:, 0, 0] -= x
        shifted[:, 0, 1] -= y
        cv2.drawContours(local_mask, [shifted], -1, 255, -1)

        local_depth = depth_roi[y:y + height, x:x + width]
        valid_mask = (
            (local_mask > 0)
            & np.isfinite(local_depth)
            & (local_depth >= self.min_depth_m)
            & (local_depth <= self.max_depth_m)
        )

        contour_pixels = int(np.count_nonzero(local_mask))
        valid_pixels = int(np.count_nonzero(valid_mask))
        if contour_pixels <= 0:
            return None

        valid_ratio = valid_pixels / float(contour_pixels)
        if valid_ratio < self.min_valid_ratio:
            return None

        ys, xs = np.nonzero(valid_mask)
        zs = local_depth[valid_mask]
        if zs.size < self.min_component_area_px:
            return None

        median_depth = float(np.median(zs))
        plane_rmse, plane_mad = self.fit_depth_plane(
            xs + x,
            ys + y,
            zs,
        )
        if plane_rmse > self.max_plane_rmse_m:
            return None
        if plane_mad > self.max_plane_mad_m:
            return None

        short_px = min(side_a_px, side_b_px)
        long_px = max(side_a_px, side_b_px)
        focal = max(1.0, 0.5 * (fx + fy))

        estimated_short_m = short_px * median_depth / focal
        estimated_long_m = long_px * median_depth / focal

        expected_short_m = min(self.board_width_m, self.board_height_m)
        expected_long_m = max(self.board_width_m, self.board_height_m)

        if not (
            expected_short_m * self.dimension_min_scale
            <= estimated_short_m
            <= expected_short_m * self.dimension_max_scale
        ):
            return None
        if not (
            expected_long_m * self.dimension_min_scale
            <= estimated_long_m
            <= expected_long_m * self.dimension_max_scale
        ):
            return None

        expected_area_px = (
            fx
            * fy
            * self.board_width_m
            * self.board_height_m
            / max(median_depth * median_depth, 1e-6)
        )
        area_ratio = contour_area / max(expected_area_px, 1.0)
        if not (self.area_ratio_min <= area_ratio <= self.area_ratio_max):
            return None

        short_error = abs(
            math.log(max(estimated_short_m, 1e-5) / expected_short_m)
        )
        long_error = abs(
            math.log(max(estimated_long_m, 1e-5) / expected_long_m)
        )
        plane_score = 0.5 * (
            plane_rmse / self.max_plane_rmse_m
            + plane_mad / self.max_plane_mad_m
        )
        score = (
            short_error
            + long_error
            + plane_score
            + (1.0 - min(rectangularity, 1.0)) * 0.60
            + (1.0 - min(valid_ratio, 1.0)) * 0.40
            + abs(math.log(max(area_ratio, 1e-5))) * 0.25
        )
        if score > self.max_candidate_score:
            return None

        perimeter = float(cv2.arcLength(contour, True))
        if perimeter > 1e-6:
            approximate = cv2.approxPolyDP(
                contour,
                0.04 * perimeter,
                True,
            )
            approx_vertices = int(len(approximate))
        else:
            approx_vertices = 0

        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        solidity = (
            contour_area / hull_area
            if hull_area > 1e-6
            else 0.0
        )

        box_points = cv2.boxPoints(rotated_rect)
        box_points[:, 0] += origin_x
        box_points[:, 1] += origin_y
        box_points = np.round(box_points).astype(np.int32)

        return {
            'score': float(score),
            'bbox': [
                int(x + origin_x),
                int(y + origin_y),
                int(x + width + origin_x),
                int(y + height + origin_y),
            ],
            'box_points': box_points,
            'center_x': float(rotated_rect[0][0] + origin_x),
            'center_y': float(rotated_rect[0][1] + origin_y),
            'distance_m': median_depth,
            'estimated_short_m': estimated_short_m,
            'estimated_long_m': estimated_long_m,
            'plane_rmse_m': plane_rmse,
            'plane_mad_m': plane_mad,
            'rectangularity': rectangularity,
            'valid_ratio': valid_ratio,
            'area_ratio': area_ratio,
            'approx_vertices': approx_vertices,
            'solidity': solidity,
        }

    def detect_candidate(
        self,
        depth_m: np.ndarray,
        fx: float,
        fy: float,
    ) -> Tuple[Optional[Candidate], Tuple[int, int, int, int]]:
        image_h, image_w = depth_m.shape[:2]

        x1 = int(round(image_w * self.search_x_min_ratio))
        x2 = int(round(image_w * self.search_x_max_ratio))
        y1 = int(round(image_h * self.search_y_start_ratio))
        y2 = image_h

        x1 = int(self._clamp(x1, 0, max(0, image_w - 1)))
        x2 = int(self._clamp(x2, x1 + 1, image_w))
        y1 = int(self._clamp(y1, 0, max(0, image_h - 1)))

        roi = depth_m[y1:y2, x1:x2]
        search_box = (x1, y1, x2, y2)
        if roi.size == 0:
            return None, search_box

        filtered = cv2.medianBlur(roi.astype(np.float32), 5)
        valid_global = (
            np.isfinite(filtered)
            & (filtered >= self.min_depth_m)
            & (filtered <= self.max_depth_m)
        )

        centers = self.depth_layer_centers(filtered)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (self.morph_kernel_size, self.morph_kernel_size),
        )

        search_area_px = int(roi.shape[0] * roi.shape[1])
        best: Optional[Candidate] = None
        seen = set()

        for center in centers:
            mask = (
                valid_global
                & (np.abs(filtered - center) <= self.depth_window_half_m)
            ).astype(np.uint8) * 255

            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_CLOSE,
                kernel,
                iterations=2,
            )

            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            for contour in contours:
                bx, by, bw, bh = cv2.boundingRect(contour)
                key = (int(bx / 8), int(by / 8), int(bw / 8), int(bh / 8))
                if key in seen:
                    continue
                seen.add(key)

                candidate = self.evaluate_contour(
                    contour,
                    roi,
                    x1,
                    y1,
                    search_area_px,
                    fx,
                    fy,
                )
                if candidate is None:
                    continue

                candidate['layer_center_m'] = float(center)
                if best is None or float(candidate['score']) < float(best['score']):
                    best = candidate

        return best, search_box

    def analyze_candidate_appearance(
        self,
        rgb: np.ndarray,
        candidate: Candidate,
    ) -> Dict[str, object]:
        height, width = rgb.shape[:2]
        points = np.asarray(candidate['box_points'], dtype=np.int32).copy()
        points[:, 0] = np.clip(points[:, 0], 0, max(0, width - 1))
        points[:, 1] = np.clip(points[:, 1], 0, max(0, height - 1))

        x, y, box_w, box_h = cv2.boundingRect(points)
        x2 = min(width, x + box_w)
        y2 = min(height, y + box_h)

        if x2 <= x or y2 <= y:
            return {
                'valid': False,
                'white_ratio': 1.0,
                'gray_std': 0.0,
                'edge_ratio': 0.0,
                'all_white': True,
            }

        crop = rgb[y:y2, x:x2]
        mask = np.zeros((y2 - y, x2 - x), dtype=np.uint8)
        local_points = points.copy()
        local_points[:, 0] -= x
        local_points[:, 1] -= y
        cv2.fillConvexPoly(mask, local_points, 255)

        selected_count = int(np.count_nonzero(mask))
        if selected_count < 20:
            return {
                'valid': False,
                'white_ratio': 1.0,
                'gray_std': 0.0,
                'edge_ratio': 0.0,
                'all_white': True,
            }

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        white_mask = (
            (hsv[:, :, 1] <= self.white_saturation_max)
            & (hsv[:, :, 2] >= self.white_value_min)
            & (mask > 0)
        )
        white_ratio = float(np.count_nonzero(white_mask)) / selected_count

        gray_values = gray[mask > 0]
        gray_std = float(np.std(gray_values)) if gray_values.size else 0.0

        edges = cv2.Canny(gray, 60, 150)
        edge_ratio = float(
            np.count_nonzero((edges > 0) & (mask > 0))
        ) / selected_count

        all_white = (
            white_ratio >= self.all_white_ratio_threshold
            and gray_std <= self.all_white_gray_std_max
            and edge_ratio <= self.all_white_edge_ratio_max
        )

        return {
            'valid': True,
            'white_ratio': white_ratio,
            'gray_std': gray_std,
            'edge_ratio': edge_ratio,
            'all_white': bool(all_white),
        }

    def analyze_obstacle_match(
        self,
        candidate: Candidate,
        obstacle_boxes,
        obstacle_status: str,
    ) -> Dict[str, object]:
        """
        判断深度候选是否应作为障碍物排除。

        当前工程中：
            深度候选来自 Aurora RGBD；
            /racing_obstacle_detection 的框来自下相机 /image。

        因此默认 obstacle_boxes_same_camera=False 时，禁止跨相机比较
        bbox、中心点、IoU和覆盖率。YOLO只作为“当前确实存在障碍物”的
        门控，最终必须同时满足：
            1. 深度估算尺寸接近 0.20m × 0.30m；
            2. 深度轮廓呈不规则三角形。
        """
        short_m = float(candidate['estimated_short_m'])
        long_m = float(candidate['estimated_long_m'])
        obstacle_short = min(
            self.obstacle_width_m,
            self.obstacle_height_m,
        )
        obstacle_long = max(
            self.obstacle_width_m,
            self.obstacle_height_m,
        )

        size_match = (
            obstacle_short * self.obstacle_dimension_min_scale
            <= short_m
            <= obstacle_short * self.obstacle_dimension_max_scale
            and obstacle_long * self.obstacle_dimension_min_scale
            <= long_m
            <= obstacle_long * self.obstacle_dimension_max_scale
        )

        approx_vertices = int(candidate.get('approx_vertices', 0))
        rectangularity = float(candidate['rectangularity'])
        triangle_like = (
            3 <= approx_vertices <= self.obstacle_triangle_max_vertices
            and rectangularity
            <= self.obstacle_triangle_rectangularity_max
        )

        best_iou = 0.0
        best_coverage = 0.0
        center_inside = False
        best_box = None
        best_confidence = 0.0

        if (
            self.obstacle_boxes_same_camera
            and obstacle_status == 'active'
        ):
            candidate_box = candidate['bbox']
            candidate_center_x = float(candidate['center_x'])
            candidate_center_y = float(candidate['center_y'])

            for obstacle_box in obstacle_boxes:
                ox1, oy1, ox2, oy2, confidence = obstacle_box
                expanded = self.expand_box(
                    (ox1, oy1, ox2, oy2)
                )
                iou, coverage = (
                    self.rectangle_intersection_metrics(
                        candidate_box,
                        expanded,
                    )
                )

                inside = (
                    expanded[0]
                    <= candidate_center_x
                    <= expanded[2]
                    and expanded[1]
                    <= candidate_center_y
                    <= expanded[3]
                )

                if (
                    coverage > best_coverage
                    or (
                        abs(coverage - best_coverage) < 1e-9
                        and iou > best_iou
                    )
                ):
                    best_iou = iou
                    best_coverage = coverage
                    center_inside = inside
                    best_box = [ox1, oy1, ox2, oy2]
                    best_confidence = confidence

            spatial_match = (
                center_inside
                or best_iou >= self.obstacle_iou_threshold
                or best_coverage
                >= self.obstacle_candidate_coverage_threshold
            )

            obstacle_reject = (
                spatial_match
                and (
                    size_match
                    or triangle_like
                    or best_coverage
                    >= self.obstacle_strong_coverage_threshold
                )
            )
            comparison_mode = 'same_camera_bbox'

        else:
            # 下相机YOLO与上方RGBD不共享像素坐标。只在YOLO当前确实
            # 检测到障碍物时，才启动深度尺寸+三角轮廓双重排除。
            spatial_match = obstacle_status == 'active'
            obstacle_reject = (
                spatial_match
                and size_match
                and triangle_like
            )
            comparison_mode = 'cross_camera_presence_gate'

        return {
            'status': obstacle_status,
            'comparison_mode': comparison_mode,
            'boxes_same_camera': bool(
                self.obstacle_boxes_same_camera
            ),
            'active_box_count': len(obstacle_boxes),
            'size_match': bool(size_match),
            'triangle_like': bool(triangle_like),
            'approx_vertices': approx_vertices,
            'rectangularity': rectangularity,
            'best_iou': float(best_iou),
            'candidate_coverage': float(best_coverage),
            'center_inside': bool(center_inside),
            'spatial_or_presence_match': bool(spatial_match),
            'matched_box': best_box,
            'matched_confidence': float(best_confidence),
            'obstacle_reject': bool(obstacle_reject),
        }

    def apply_candidate_filters(
        self,
        rgb: np.ndarray,
        candidate: Optional[Candidate],
        obstacle_boxes,
        obstacle_status: str,
    ):
        if candidate is None:
            return None, '', {}, {}

        appearance = self.analyze_candidate_appearance(rgb, candidate)
        candidate['appearance'] = appearance

        if bool(appearance.get('all_white', False)):
            self.white_reject_count += 1
            return None, 'all_white', appearance, {}

        obstacle_analysis = self.analyze_obstacle_match(
            candidate,
            obstacle_boxes,
            obstacle_status,
        )
        candidate['obstacle_analysis'] = obstacle_analysis

        if bool(obstacle_analysis.get('obstacle_reject', False)):
            self.obstacle_reject_count += 1
            return None, 'yolo_obstacle', appearance, obstacle_analysis

        return candidate, '', appearance, obstacle_analysis

    def clear_confirmation_immediately(self):
        with self.state_lock:
            self.consecutive_detect_count = 0
            self.consecutive_lost_count = 0
            self.board_confirmed = False
            self.last_candidate = None

    def candidate_matches_last(
        self,
        candidate: Candidate,
        last: Candidate,
    ) -> bool:
        dx = float(candidate['center_x']) - float(last['center_x'])
        dy = float(candidate['center_y']) - float(last['center_y'])
        center_gap = math.hypot(dx, dy)
        depth_gap = abs(
            float(candidate['distance_m']) - float(last['distance_m'])
        )
        return (
            center_gap <= self.track_center_tolerance_px
            and depth_gap <= self.track_depth_tolerance_m
        )

    def update_confirmation(self, candidate: Optional[Candidate]) -> bool:
        with self.state_lock:
            if candidate is None:
                self.consecutive_lost_count += 1
                self.consecutive_detect_count = 0
                if self.consecutive_lost_count >= self.lost_frames:
                    self.board_confirmed = False
                    self.last_candidate = None
                return self.board_confirmed

            self.consecutive_lost_count = 0
            if (
                self.last_candidate is not None
                and self.candidate_matches_last(candidate, self.last_candidate)
            ):
                self.consecutive_detect_count += 1
            else:
                self.consecutive_detect_count = 1

            self.last_candidate = candidate
            if self.consecutive_detect_count >= self.confirm_frames:
                self.board_confirmed = True

            return self.board_confirmed


    # ---------------------------------------------------------------
    # 子类扩展钩子
    # ---------------------------------------------------------------
    def on_depth_frame_processed(
        self,
        rgb_msg: Image,
        depth_msg: Image,
        rgb: np.ndarray,
        depth_m: np.ndarray,
        raw_candidate: Optional[Candidate],
        accepted_candidate: Optional[Candidate],
        reject_reason: str,
        appearance: Dict[str, object],
        obstacle_analysis: Dict[str, object],
        obstacle_boxes,
        obstacle_status: str,
        obstacle_age: float,
        confirmed: bool,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
    ):
        """供完整图生文子类接入；纯调试基类默认不处理。"""
        return

    def decorate_debug_image(
        self,
        debug: np.ndarray,
        rgb_msg: Image,
    ) -> np.ndarray:
        """供完整图生文子类在调试图上叠加地图核对状态。"""
        return debug

    def synced_callback(self, rgb_msg: Image, depth_msg: Image):
        with self.state_lock:
            if not self.intrinsics_ready:
                self.get_logger().info(
                    '等待相机内参...',
                    throttle_duration_sec=1.0,
                )
                return
            fx = self.fx
            fy = self.fy
            cx = self.cx
            cy = self.cy

        try:
            rgb = self.bridge.imgmsg_to_cv2(
                rgb_msg,
                desired_encoding='bgr8',
            )
        except Exception as exc:
            self.get_logger().error(
                f'RGB图像转换失败: {exc}',
                throttle_duration_sec=1.0,
            )
            return

        depth_m = self.depth_to_meters(depth_msg)
        if depth_m is None:
            return

        self.frame_count += 1

        if rgb.shape[:2] != depth_m.shape[:2]:
            self.get_logger().error(
                'RGB与深度图尺寸不一致；本脚本要求深度已对齐到RGB。'
                f' RGB={rgb.shape[:2]}, Depth={depth_m.shape[:2]}',
                throttle_duration_sec=1.0,
            )
            self.publish_debug(
                rgb,
                rgb_msg,
                None,
                None,
                '',
                {},
                {},
                [],
                'unavailable',
                (
                    0,
                    int(rgb.shape[0] * self.search_y_start_ratio),
                    rgb.shape[1],
                    rgb.shape[0],
                ),
                False,
                'RGB/DEPTH SIZE MISMATCH',
            )
            return

        raw_candidate, search_box = self.detect_candidate(
            depth_m,
            fx,
            fy,
        )

        obstacle_boxes, obstacle_status, obstacle_age = (
            self.obstacle_snapshot()
        )

        accepted_candidate, reject_reason, appearance, obstacle_analysis = (
            self.apply_candidate_filters(
                rgb,
                raw_candidate,
                obstacle_boxes,
                obstacle_status,
            )
        )

        if reject_reason:
            self.clear_confirmation_immediately()
            confirmed = False
        else:
            confirmed = self.update_confirmation(accepted_candidate)

        try:
            self.on_depth_frame_processed(
                rgb_msg,
                depth_msg,
                rgb,
                depth_m,
                raw_candidate,
                accepted_candidate,
                reject_reason,
                appearance,
                obstacle_analysis,
                obstacle_boxes,
                obstacle_status,
                obstacle_age,
                confirmed,
                fx,
                fy,
                cx,
                cy,
            )
        except Exception as exc:
            self.get_logger().error(
                f'图生文帧处理异常: {exc}',
                throttle_duration_sec=1.0,
            )

        self.publish_result(
            rgb_msg,
            raw_candidate,
            accepted_candidate,
            reject_reason,
            appearance,
            obstacle_analysis,
            obstacle_status,
            obstacle_age,
            obstacle_boxes,
            confirmed,
            fx,
            fy,
            cx,
            cy,
        )
        self.publish_debug(
            rgb,
            rgb_msg,
            raw_candidate,
            accepted_candidate,
            reject_reason,
            appearance,
            obstacle_analysis,
            obstacle_boxes,
            obstacle_status,
            search_box,
            confirmed,
        )
        self.periodic_log(
            raw_candidate,
            accepted_candidate,
            reject_reason,
            obstacle_status,
            confirmed,
        )

    def publish_result(
        self,
        rgb_msg: Image,
        raw_candidate: Optional[Candidate],
        accepted_candidate: Optional[Candidate],
        reject_reason: str,
        appearance: Dict[str, object],
        obstacle_analysis: Dict[str, object],
        obstacle_status: str,
        obstacle_age: float,
        obstacle_boxes,
        confirmed: bool,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
    ):
        payload = {
            'header': {
                'stamp': {
                    'sec': int(rgb_msg.header.stamp.sec),
                    'nanosec': int(rgb_msg.header.stamp.nanosec),
                },
                'frame_id': str(rgb_msg.header.frame_id),
            },
            'depth_candidate_found': raw_candidate is not None,
            'accepted_this_frame': accepted_candidate is not None,
            'confirmed': bool(confirmed),
            'reject_reason': reject_reason,
            'consecutive_detect_count': int(self.consecutive_detect_count),
            'confirm_frames': int(self.confirm_frames),
            'board_size_m': {
                'width': self.board_width_m,
                'height': self.board_height_m,
            },
            'obstacle_size_m': {
                'width': self.obstacle_width_m,
                'height': self.obstacle_height_m,
            },
            'obstacle_yolo': {
                'topic': self.obstacle_topic,
                'status': obstacle_status,
                'age_sec': (
                    None
                    if not math.isfinite(obstacle_age)
                    else round(float(obstacle_age), 4)
                ),
                'box_count': len(obstacle_boxes),
            },
            'appearance': appearance or None,
            'obstacle_analysis': obstacle_analysis or None,
            'camera_intrinsics': {
                'fx': fx,
                'fy': fy,
                'cx': cx,
                'cy': cy,
            },
            'candidate': None,
        }

        if raw_candidate is not None:
            payload['candidate'] = {
                'bbox': [int(v) for v in raw_candidate['bbox']],
                'center': {
                    'x': round(float(raw_candidate['center_x']), 2),
                    'y': round(float(raw_candidate['center_y']), 2),
                },
                'distance_m': round(float(raw_candidate['distance_m']), 4),
                'estimated_short_side_m': round(
                    float(raw_candidate['estimated_short_m']),
                    4,
                ),
                'estimated_long_side_m': round(
                    float(raw_candidate['estimated_long_m']),
                    4,
                ),
                'plane_rmse_m': round(
                    float(raw_candidate['plane_rmse_m']),
                    5,
                ),
                'plane_mad_m': round(
                    float(raw_candidate['plane_mad_m']),
                    5,
                ),
                'rectangularity': round(
                    float(raw_candidate['rectangularity']),
                    4,
                ),
                'approx_vertices': int(
                    raw_candidate.get('approx_vertices', 0)
                ),
                'solidity': round(
                    float(raw_candidate.get('solidity', 0.0)),
                    4,
                ),
                'valid_depth_ratio': round(
                    float(raw_candidate['valid_ratio']),
                    4,
                ),
                'area_ratio': round(
                    float(raw_candidate['area_ratio']),
                    4,
                ),
                'score': round(float(raw_candidate['score']), 4),
            }

        msg = String()
        msg.data = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(',', ':'),
        )
        self.result_pub.publish(msg)

    def publish_debug(
        self,
        rgb: np.ndarray,
        rgb_msg: Image,
        raw_candidate: Optional[Candidate],
        accepted_candidate: Optional[Candidate],
        reject_reason: str,
        appearance: Dict[str, object],
        obstacle_analysis: Dict[str, object],
        obstacle_boxes,
        obstacle_status: str,
        search_box: Tuple[int, int, int, int],
        confirmed: bool,
        error_text: str = '',
    ):
        debug = rgb.copy()
        sx1, sy1, sx2, sy2 = search_box

        # 蓝色：深度搜索区域。
        cv2.rectangle(
            debug,
            (sx1, sy1),
            (sx2 - 1, sy2 - 1),
            (255, 128, 0),
            2,
        )
        cv2.putText(
            debug,
            'DEPTH SEARCH AREA',
            (sx1 + 6, max(24, sy1 + 24)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 128, 0),
            2,
            cv2.LINE_AA,
        )

        # 只有YOLO框和当前RGBD图像来自同一相机时，像素框才可绘制。
        if (
            obstacle_status == 'active'
            and self.obstacle_boxes_same_camera
        ):
            for x1, y1, x2, y2, confidence in obstacle_boxes:
                cv2.rectangle(
                    debug,
                    (int(round(x1)), int(round(y1))),
                    (int(round(x2)), int(round(y2))),
                    (255, 0, 255),
                    2,
                )
                cv2.putText(
                    debug,
                    f'YOLO OBSTACLE {confidence:.2f}',
                    (int(round(x1)), max(18, int(round(y1)) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (255, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

        if error_text:
            cv2.putText(
                debug,
                error_text,
                (12, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.70,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        elif raw_candidate is None:
            cv2.putText(
                debug,
                'BOARD: NOT FOUND',
                (12, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.78,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        else:
            if reject_reason == 'all_white':
                color = (0, 0, 255)
                status = 'REJECT: ALL WHITE'
            elif reject_reason == 'yolo_obstacle':
                color = (0, 0, 255)
                status = 'REJECT: YOLO OBSTACLE'
            elif confirmed:
                color = (0, 255, 0)
                status = 'CONFIRMED'
            else:
                color = (0, 215, 255)
                status = (
                    f'CANDIDATE {self.consecutive_detect_count}/'
                    f'{self.confirm_frames}'
                )

            points = np.asarray(raw_candidate['box_points'], dtype=np.int32)
            cv2.polylines(debug, [points], True, color, 3, cv2.LINE_AA)

            x1, y1, _, _ = [int(v) for v in raw_candidate['bbox']]
            lines = [
                f'BOARD: {status}',
                (
                    f'Z={float(raw_candidate["distance_m"]):.2f}m '
                    f'SIZE={float(raw_candidate["estimated_short_m"]):.2f}x'
                    f'{float(raw_candidate["estimated_long_m"]):.2f}m'
                ),
            ]

            if appearance:
                lines.append(
                    f'WHITE={float(appearance.get("white_ratio", 0.0)):.2f} '
                    f'STD={float(appearance.get("gray_std", 0.0)):.1f} '
                    f'EDGE={float(appearance.get("edge_ratio", 0.0)):.3f}'
                )

            if obstacle_analysis:
                if self.obstacle_boxes_same_camera:
                    lines.append(
                        f'OBS IOU={float(obstacle_analysis.get("best_iou", 0.0)):.2f} '
                        f'COV={float(obstacle_analysis.get("candidate_coverage", 0.0)):.2f} '
                        f'SIZE={int(bool(obstacle_analysis.get("size_match", False)))} '
                        f'TRI={int(bool(obstacle_analysis.get("triangle_like", False)))}'
                    )
                else:
                    lines.append(
                        'OBS LOWER-CAM GATE '
                        f'SIZE={int(bool(obstacle_analysis.get("size_match", False)))} '
                        f'TRI={int(bool(obstacle_analysis.get("triangle_like", False)))}'
                    )

            lines.append(
                f'VERT={int(raw_candidate.get("approx_vertices", 0))} '
                f'RECT={float(raw_candidate["rectangularity"]):.2f} '
                f'SCORE={float(raw_candidate["score"]):.2f}'
            )

            text_y = max(26, y1 - (len(lines) * 21 + 5))
            for index, line in enumerate(lines):
                cv2.putText(
                    debug,
                    line,
                    (max(4, x1), text_y + index * 21),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    color,
                    2,
                    cv2.LINE_AA,
                )

        cv2.putText(
            debug,
            f'YOLO={obstacle_status} '
            f'MODE={"SAME_CAM" if self.obstacle_boxes_same_camera else "LOWER_CAM_GATE"} '
            f'WHITE_REJ={self.white_reject_count} '
            f'OBS_REJ={self.obstacle_reject_count}',
            (12, max(24, debug.shape[0] - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        try:
            debug = self.decorate_debug_image(debug, rgb_msg)
        except Exception as exc:
            self.get_logger().warning(
                f'调试图地图状态叠加失败: {exc}',
                throttle_duration_sec=1.0,
            )

        ok, encoded = cv2.imencode(
            '.jpg',
            debug,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            self.get_logger().warning(
                '调试图像JPEG编码失败',
                throttle_duration_sec=1.0,
            )
            return

        msg = CompressedImage()
        msg.header = rgb_msg.header
        msg.format = 'jpeg'
        msg.data = encoded.tobytes()
        self.debug_pub.publish(msg)

    def periodic_log(
        self,
        raw_candidate: Optional[Candidate],
        accepted_candidate: Optional[Candidate],
        reject_reason: str,
        obstacle_status: str,
        confirmed: bool,
    ):
        now = time.monotonic()
        if now - self.last_log_time < 1.0:
            return
        self.last_log_time = now

        if raw_candidate is None:
            self.get_logger().info(
                '深度立牌检测 | '
                f'帧={self.frame_count} | 当前无深度候选 | '
                f'YOLO={obstacle_status} | '
                f'连续丢失={self.consecutive_lost_count}/{self.lost_frames} | '
                f'确认={confirmed}'
            )
            return

        appearance = raw_candidate.get('appearance', {})
        obstacle = raw_candidate.get('obstacle_analysis', {})

        self.get_logger().info(
            '深度立牌检测 | '
            f'帧={self.frame_count} | '
            f'结果={"接受" if accepted_candidate is not None else "排除"} | '
            f'原因={reject_reason or "none"} | '
            f'距离={float(raw_candidate["distance_m"]):.3f}m | '
            f'尺寸={float(raw_candidate["estimated_short_m"]):.3f}x'
            f'{float(raw_candidate["estimated_long_m"]):.3f}m | '
            f'白色={float(appearance.get("white_ratio", 0.0)):.3f} | '
            f'灰度STD={float(appearance.get("gray_std", 0.0)):.1f} | '
            f'YOLO={obstacle_status} | '
            f'覆盖={float(obstacle.get("candidate_coverage", 0.0)):.3f} | '
            f'障碍尺寸={bool(obstacle.get("size_match", False))} | '
            f'三角={bool(obstacle.get("triangle_like", False))} | '
            f'确认={confirmed}'
        )



class VisionBoardLLMNode(VisionBoardDepthTestNode):
    """
    完整比赛版图生文节点。

    在同一节点内完成：
        RGB + 深度同步；
        固定尺寸倾斜立牌检测；
        全白候选排除；
        YOLO锥桶排除；
        带header.stamp的Odometry按相机帧时间对齐；
        odom坐标加地图偏置；
        深度候选反投影并转换到地图坐标；
        按二维码方向核对预期立牌地图位置；
        多帧候选中选择最清晰的一张；
        调用千问 qwen3.7-plus 视觉模型并流式发布结果。

    本节点直接订阅带 header.stamp 的 nav_msgs/msg/Odometry。
    对每个 RGB+深度同步帧，按图像时间戳在里程计缓存中插值或选最近位姿。
    """

    def __init__(self):
        # 初始化完整深度检测基类，它会创建：
        # RGB/Depth同步、CameraInfo、YOLO障碍物、深度结果和压缩调试图。
        super().__init__()

        # ============================================================
        # 千问与结果话题
        # ============================================================
        self.declare_parameter('model_name', 'qwen3.7-plus')

        # 阿里云百炼北京地域支持两种配置方式：
        # 1. DASHSCOPE_BASE_URL 保存完整兼容接口地址；
        # 2. DASHSCOPE_WORKSPACE_ID 只保存业务空间ID，由代码拼接地址。
        self.declare_parameter(
            'workspace_id',
            os.getenv('DASHSCOPE_WORKSPACE_ID', ''),
        )
        self.declare_parameter(
            'dashscope_base_url',
            os.getenv('DASHSCOPE_BASE_URL', ''),
        )

        self.declare_parameter(
            'prompt_text',
            '20字以内立刻描述立牌上的画面。',
        )
        self.declare_parameter('max_output_chars', 20)
        self.declare_parameter('max_output_tokens', 64)

        # qwen3.7-plus 默认会思考；简单图生文任务显式关闭以降低首字延迟。
        self.declare_parameter('enable_thinking', False)

        # 发给千问的立牌裁剪图使用JPEG，减小上传数据量。
        self.declare_parameter('llm_jpeg_quality', 70)

        # 控制视觉输入的最大像素预算。
        self.declare_parameter('max_pixels', 262144)

        self.declare_parameter('llm_result_topic', '/vision_llm_result')
        self.declare_parameter(
            'stream_status_topic',
            '/vision_llm_stream_status',
        )

        # 0表示收到模型字符后立即发布，不人为增加OLED显示延迟。
        self.declare_parameter('stream_char_interval_sec', 0.0)

        # ============================================================
        # 二维码方向与坐标
        # ============================================================
        self.declare_parameter('qr_topic', '/qr_direction_result')
        self.declare_parameter('odom_topic', '/odom')

        # odom_pose=(0,0) 到地图坐标的偏置。
        self.declare_parameter('odom_offset_x', 0.55)
        self.declare_parameter('odom_offset_y', 0.25)

        # 图生文立牌的大致地图位置。
        self.declare_parameter('counterclockwise_board_map_x', 0.35)
        self.declare_parameter('counterclockwise_board_map_y', 4.40)
        self.declare_parameter('clockwise_board_map_x', 4.65)
        self.declare_parameter('clockwise_board_map_y', 0.44)

        # “大体位置正确”的容差，同时使用轴向和欧氏距离限制。
        self.declare_parameter('board_map_tolerance_x_m', 0.80)
        self.declare_parameter('board_map_tolerance_y_m', 0.80)
        self.declare_parameter('board_map_tolerance_radius_m', 1.00)

        # ============================================================
        # SD/RGBD相机相对车辆坐标的安装偏置
        # ============================================================
        # 默认假设相机光轴与车辆前进方向一致，安装点位于车辆坐标原点。
        # 现场若已知相机安装偏置，可通过ROS参数修正，无需改代码。
        self.declare_parameter('camera_forward_offset_m', 0.0)
        self.declare_parameter('camera_left_offset_m', 0.0)
        self.declare_parameter('camera_yaw_offset_deg', 0.0)

        # 深度相机若向下俯视，光学Z不能直接当作水平前向距离。
        # 正值表示相机光轴向下俯视。
        self.declare_parameter('camera_pitch_deg', 0.0)

        # 当前底盘代码默认 g_yaw 已经是弧度、逆时针为正、0指向地图+X。
        # 若实车定义不同，可用符号和固定零偏校正。
        self.declare_parameter('odom_yaw_sign', 1.0)
        self.declare_parameter('odom_yaw_offset_deg', 0.0)

        # ============================================================
        # 位姿时间对齐
        # ============================================================
        self.declare_parameter('pose_cache_duration_sec', 4.0)
        self.declare_parameter('pose_frame_tolerance_sec', 0.25)
        self.declare_parameter('odom_timeout_sec', 0.60)

        # 图像header与节点ROS时钟相差太大时，说明时钟源不一致；
        # 此时自动改用图像到达时刻与odom到达时刻匹配。
        self.declare_parameter('image_header_clock_max_skew_sec', 5.0)

        # ============================================================
        # 正式候选采集与选图
        # ============================================================
        self.declare_parameter('capture_exit_confirm_frames', 4)
        self.declare_parameter('capture_max_duration_sec', 10.0)
        self.declare_parameter('capture_max_valid_frames', 120)
        self.declare_parameter('candidate_interval_sec', 0.0)

        # 发给大模型时，围绕深度候选框扩大后裁剪。
        self.declare_parameter('board_crop_expand_ratio', 1.35)
        self.declare_parameter('board_crop_min_size_px', 48)
        self.declare_parameter('sharpness_max_width', 640)

        # 读取参数。
        self.model_name = str(
            self.get_parameter('model_name').value
        ).strip()

        self.workspace_id = str(
            self.get_parameter('workspace_id').value
        ).strip()
        configured_base_url = str(
            self.get_parameter('dashscope_base_url').value
        ).strip()

        if configured_base_url:
            self.dashscope_base_url = configured_base_url.rstrip('/')
        elif self.workspace_id:
            self.dashscope_base_url = (
                f'https://{self.workspace_id}.'
                'cn-beijing.maas.aliyuncs.com/'
                'compatible-mode/v1'
            )
        else:
            raise RuntimeError(
                '未配置阿里云百炼接口地址。请设置 '
                'DASHSCOPE_BASE_URL、DASHSCOPE_WORKSPACE_ID，'
                '或ROS参数 dashscope_base_url/workspace_id'
            )

        if (
            '{WorkspaceId}' in self.dashscope_base_url
            or '你的WorkspaceId' in self.dashscope_base_url
        ):
            raise RuntimeError(
                '百炼接口地址中仍包含WorkspaceId占位符，'
                '请替换为真实业务空间ID'
            )

        self.prompt_text = str(self.get_parameter('prompt_text').value)
        self.max_output_chars = max(
            1,
            int(self.get_parameter('max_output_chars').value),
        )
        self.max_output_tokens = max(
            16,
            int(self.get_parameter('max_output_tokens').value),
        )
        self.enable_thinking = bool(
            self.get_parameter('enable_thinking').value
        )
        self.llm_jpeg_quality = int(
            self._clamp(
                int(self.get_parameter('llm_jpeg_quality').value),
                30,
                95,
            )
        )
        self.max_pixels = max(
            4096,
            int(self.get_parameter('max_pixels').value),
        )

        self.llm_result_topic = str(
            self.get_parameter('llm_result_topic').value
        )
        self.stream_status_topic = str(
            self.get_parameter('stream_status_topic').value
        )
        self.stream_char_interval_sec = max(
            0.0,
            float(self.get_parameter('stream_char_interval_sec').value),
        )

        self.qr_topic = str(self.get_parameter('qr_topic').value)
        self.odom_topic = str(self.get_parameter('odom_topic').value)

        self.odom_offset_x = float(
            self.get_parameter('odom_offset_x').value
        )
        self.odom_offset_y = float(
            self.get_parameter('odom_offset_y').value
        )

        self.counterclockwise_board_map_x = float(
            self.get_parameter('counterclockwise_board_map_x').value
        )
        self.counterclockwise_board_map_y = float(
            self.get_parameter('counterclockwise_board_map_y').value
        )
        self.clockwise_board_map_x = float(
            self.get_parameter('clockwise_board_map_x').value
        )
        self.clockwise_board_map_y = float(
            self.get_parameter('clockwise_board_map_y').value
        )

        self.board_map_tolerance_x_m = max(
            0.01,
            float(self.get_parameter('board_map_tolerance_x_m').value),
        )
        self.board_map_tolerance_y_m = max(
            0.01,
            float(self.get_parameter('board_map_tolerance_y_m').value),
        )
        self.board_map_tolerance_radius_m = max(
            0.01,
            float(
                self.get_parameter('board_map_tolerance_radius_m').value
            ),
        )

        self.camera_forward_offset_m = float(
            self.get_parameter('camera_forward_offset_m').value
        )
        self.camera_left_offset_m = float(
            self.get_parameter('camera_left_offset_m').value
        )
        self.camera_yaw_offset_rad = math.radians(
            float(self.get_parameter('camera_yaw_offset_deg').value)
        )
        self.camera_pitch_rad = math.radians(
            float(self.get_parameter('camera_pitch_deg').value)
        )
        self.odom_yaw_sign = float(
            self.get_parameter('odom_yaw_sign').value
        )
        self.odom_yaw_offset_rad = math.radians(
            float(self.get_parameter('odom_yaw_offset_deg').value)
        )

        self.pose_cache_duration_sec = max(
            0.5,
            float(self.get_parameter('pose_cache_duration_sec').value),
        )
        self.pose_frame_tolerance_sec = max(
            0.01,
            float(self.get_parameter('pose_frame_tolerance_sec').value),
        )
        self.odom_timeout_sec = max(
            0.05,
            float(self.get_parameter('odom_timeout_sec').value),
        )
        self.image_header_clock_max_skew_sec = max(
            0.1,
            float(
                self.get_parameter(
                    'image_header_clock_max_skew_sec'
                ).value
            ),
        )

        self.capture_exit_confirm_frames = max(
            1,
            int(
                self.get_parameter('capture_exit_confirm_frames').value
            ),
        )
        self.capture_max_duration_sec = max(
            0.5,
            float(self.get_parameter('capture_max_duration_sec').value),
        )
        self.capture_max_valid_frames = max(
            1,
            int(
                self.get_parameter('capture_max_valid_frames').value
            ),
        )
        self.candidate_interval_sec = max(
            0.0,
            float(self.get_parameter('candidate_interval_sec').value),
        )
        self.board_crop_expand_ratio = max(
            1.0,
            float(self.get_parameter('board_crop_expand_ratio').value),
        )
        self.board_crop_min_size_px = max(
            8,
            int(self.get_parameter('board_crop_min_size_px').value),
        )
        self.sharpness_max_width = max(
            64,
            int(self.get_parameter('sharpness_max_width').value),
        )

        # 阿里云百炼 OpenAI 兼容客户端。
        self.api_key = os.getenv('DASHSCOPE_API_KEY')
        if not self.api_key:
            raise RuntimeError('环境变量 DASHSCOPE_API_KEY 未设置')

        self.client = OpenAI(
            base_url=self.dashscope_base_url,
            api_key=self.api_key,
            timeout=60.0,
            max_retries=0,
        )

        # 图生文状态使用独立锁，避免与深度检测基类的锁互相干扰。
        self.llm_lock = threading.RLock()

        self.qr_direction: Optional[str] = None
        self.qr_direction_locked = False

        # (odom_header_time, monotonic_receive_time, x, y, theta)
        self.pose_cache = deque(maxlen=600)
        self.latest_pose_monotonic = 0.0

        self.llm_capture_started = False
        self.llm_capture_start_monotonic = 0.0
        self.llm_invalid_frame_count = 0
        self.llm_valid_frame_count = 0
        self.llm_last_candidate_monotonic = 0.0
        self.llm_best_candidates: Dict[
            Tuple[int, bool], Dict[str, object]
        ] = {}

        self.llm_has_triggered = False
        self.llm_thread: Optional[threading.Thread] = None

        # 当前帧地图核对信息，供调试图叠加。
        self.latest_llm_overlay: Dict[str, object] = {
            'status': 'WAIT_QR',
        }

        result_qos = QoSProfile(depth=1)
        result_qos.reliability = QoSReliabilityPolicy.RELIABLE
        result_qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL

        self.llm_result_pub = self.create_publisher(
            String,
            self.llm_result_topic,
            result_qos,
        )
        self.stream_status_pub = self.create_publisher(
            String,
            self.stream_status_topic,
            result_qos,
        )

        self.qr_sub = self.create_subscription(
            String,
            self.qr_topic,
            self.qr_callback,
            10,
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_callback,
            qos_profile_sensor_data,
        )

        self.llm_status_timer = self.create_timer(
            1.0,
            self.llm_status_callback,
        )

        self.get_logger().info('=' * 78)
        self.get_logger().info('完整深度地图核对图生文节点已启动')
        self.get_logger().info(
            f'odom偏置=({self.odom_offset_x:+.3f},'
            f'{self.odom_offset_y:+.3f})m'
        )
        self.get_logger().info(
            '逆时针立牌地图位置='
            f'({self.counterclockwise_board_map_x:.3f},'
            f'{self.counterclockwise_board_map_y:.3f})m'
        )
        self.get_logger().info(
            '顺时针立牌地图位置='
            f'({self.clockwise_board_map_x:.3f},'
            f'{self.clockwise_board_map_y:.3f})m'
        )
        self.get_logger().info(
            '地图核对容差='
            f'X±{self.board_map_tolerance_x_m:.2f}m，'
            f'Y±{self.board_map_tolerance_y_m:.2f}m，'
            f'半径≤{self.board_map_tolerance_radius_m:.2f}m'
        )
        self.get_logger().info(
            '最终候选条件：深度尺寸通过 + 非全白 + 非YOLO障碍物 + '
            '相机帧位姿对齐 + 估计地图位置正确'
        )
        self.get_logger().info(
            f'里程计输入={self.odom_topic} (nav_msgs/Odometry，使用header.stamp)'
        )
        self.get_logger().info(
            '障碍物YOLO接口模式='
            f'{"同相机像素框" if self.obstacle_boxes_same_camera else "下相机存在门控；不跨相机比较bbox"}'
        )
        self.get_logger().info(
            '最终选图：confirmed优先 > 无锥桶优先 > 清晰度最高'
        )
        self.get_logger().info(
            f'图生文结果={self.llm_result_topic} | '
            f'流式状态={self.stream_status_topic}'
        )
        self.get_logger().info(
            f'千问模型={self.model_name} | '
            f'思考模式={"开启" if self.enable_thinking else "关闭"}'
        )
        self.get_logger().info(
            f'百炼地址={self.dashscope_base_url}'
        )
        self.get_logger().info(
            f'LLM图像=JPEG质量{self.llm_jpeg_quality} | '
            f'max_pixels={self.max_pixels} | '
            f'max_output_tokens={self.max_output_tokens}'
        )
        self.get_logger().info('=' * 78)

    # ============================================================
    # 二维码与位姿缓存
    # ============================================================
    @staticmethod
    def normalize_angle(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    @staticmethod
    def header_stamp_sec(header) -> float:
        return (
            float(header.stamp.sec)
            + float(header.stamp.nanosec) * 1e-9
        )

    def qr_callback(self, msg: String):
        text = str(msg.data).strip()
        if text not in ('顺时针', '逆时针'):
            self.get_logger().warning(
                f'二维码方向无效，继续等待: {text!r}',
                throttle_duration_sec=1.0,
            )
            return

        with self.llm_lock:
            if self.qr_direction_locked:
                return
            self.qr_direction = text
            self.qr_direction_locked = True
            subscription = self.qr_sub
            self.qr_sub = None

        if subscription is not None:
            try:
                self.destroy_subscription(subscription)
            except Exception as exc:
                self.get_logger().warning(
                    f'二维码方向已锁定，但销毁订阅失败: {exc}'
                )

        target = self.expected_board_position(text)
        self.get_logger().warning(
            f'二维码方向已锁定: {text} | '
            f'预期立牌地图位置=({target[0]:.3f},{target[1]:.3f})'
        )

    @staticmethod
    def quaternion_to_yaw(quaternion) -> float:
        x = float(quaternion.x)
        y = float(quaternion.y)
        z = float(quaternion.z)
        w = float(quaternion.w)

        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    def odom_callback(self, msg: Odometry):
        stamp = self.header_stamp_sec(msg.header)
        if stamp <= 0.0:
            stamp = self.get_clock().now().nanoseconds * 1e-9

        mono_time = time.monotonic()
        raw_yaw = self.quaternion_to_yaw(
            msg.pose.pose.orientation
        )
        corrected_yaw = self.normalize_angle(
            self.odom_yaw_sign * raw_yaw
            + self.odom_yaw_offset_rad
        )

        pose = (
            stamp,
            mono_time,
            float(msg.pose.pose.position.x),
            float(msg.pose.pose.position.y),
            corrected_yaw,
        )

        with self.llm_lock:
            self.pose_cache.append(pose)
            self.latest_pose_monotonic = mono_time

            min_stamp = stamp - self.pose_cache_duration_sec
            while (
                self.pose_cache
                and self.pose_cache[0][0] < min_stamp
            ):
                self.pose_cache.popleft()

    def resolve_pose_for_frame(self, rgb_msg: Image):
        now_ros = self.get_clock().now().nanoseconds * 1e-9
        now_mono = time.monotonic()
        image_stamp = self.header_stamp_sec(rgb_msg.header)

        use_header = (
            image_stamp > 0.0
            and abs(now_ros - image_stamp)
            <= self.image_header_clock_max_skew_sec
        )
        target_time = image_stamp if use_header else now_ros

        with self.llm_lock:
            cache = list(self.pose_cache)
            latest_pose_mono = self.latest_pose_monotonic

        if not cache:
            return None, 'no_pose', float('inf'), use_header

        if now_mono - latest_pose_mono > self.odom_timeout_sec:
            return None, 'odom_stale', now_mono - latest_pose_mono, use_header

        times = [item[0] for item in cache]
        index = bisect.bisect_left(times, target_time)

        # 若目标时刻被两条位姿夹住，则做线性插值。
        if 0 < index < len(cache):
            before = cache[index - 1]
            after = cache[index]
            t0 = before[0]
            t1 = after[0]

            if t1 - t0 > 1e-6:
                ratio = self._clamp(
                    (target_time - t0) / (t1 - t0),
                    0.0,
                    1.0,
                )
                x = before[2] + ratio * (after[2] - before[2])
                y = before[3] + ratio * (after[3] - before[3])
                dtheta = self.normalize_angle(after[4] - before[4])
                theta = self.normalize_angle(before[4] + ratio * dtheta)
                gap = max(abs(target_time - t0), abs(t1 - target_time))

                if gap <= self.pose_frame_tolerance_sec:
                    return (
                        (x, y, theta),
                        'interpolated',
                        gap,
                        use_header,
                    )

        candidates = []
        if index < len(cache):
            candidates.append(cache[index])
        if index > 0:
            candidates.append(cache[index - 1])

        if not candidates:
            return None, 'no_near_pose', float('inf'), use_header

        nearest = min(
            candidates,
            key=lambda item: abs(item[0] - target_time),
        )
        gap = abs(nearest[0] - target_time)

        if gap > self.pose_frame_tolerance_sec:
            return None, 'pose_gap_too_large', gap, use_header

        return (
            (nearest[2], nearest[3], nearest[4]),
            'nearest',
            gap,
            use_header,
        )

    # ============================================================
    # 深度相对位置 -> 地图位置
    # ============================================================
    def expected_board_position(self, direction: str) -> Tuple[float, float]:
        if direction == '逆时针':
            return (
                self.counterclockwise_board_map_x,
                self.counterclockwise_board_map_y,
            )
        return (
            self.clockwise_board_map_x,
            self.clockwise_board_map_y,
        )

    def estimate_board_map_position(
        self,
        candidate: Candidate,
        odom_pose: Tuple[float, float, float],
        fx: float,
        fy: float,
        cx: float,
        cy: float,
    ) -> Dict[str, float]:
        odom_x, odom_y, vehicle_yaw = odom_pose

        vehicle_map_x = odom_x + self.odom_offset_x
        vehicle_map_y = odom_y + self.odom_offset_y

        distance_z = float(candidate['distance_m'])
        pixel_u = float(candidate['center_x'])
        pixel_v = float(candidate['center_y'])

        camera_right_m = (
            (pixel_u - cx)
            * distance_z
            / max(fx, 1e-6)
        )
        camera_down_m = (
            (pixel_v - cy)
            * distance_z
            / max(fy, 1e-6)
        )

        # 相机光学坐标：Z向前、X向右、Y向下。
        # 车辆平面坐标使用前、左。相机有俯角时，将光学Z/Y共同
        # 投影到车辆水平前向，避免把斜向下的Z全部当作水平距离。
        relative_forward_m = (
            math.cos(self.camera_pitch_rad) * distance_z
            + math.sin(self.camera_pitch_rad) * camera_down_m
        )
        relative_left_m = -camera_right_m

        camera_yaw = self.normalize_angle(
            vehicle_yaw + self.camera_yaw_offset_rad
        )
        cos_yaw = math.cos(camera_yaw)
        sin_yaw = math.sin(camera_yaw)

        camera_map_x = (
            vehicle_map_x
            + cos_yaw * self.camera_forward_offset_m
            - sin_yaw * self.camera_left_offset_m
        )
        camera_map_y = (
            vehicle_map_y
            + sin_yaw * self.camera_forward_offset_m
            + cos_yaw * self.camera_left_offset_m
        )

        board_map_x = (
            camera_map_x
            + cos_yaw * relative_forward_m
            - sin_yaw * relative_left_m
        )
        board_map_y = (
            camera_map_y
            + sin_yaw * relative_forward_m
            + cos_yaw * relative_left_m
        )

        return {
            'vehicle_map_x': vehicle_map_x,
            'vehicle_map_y': vehicle_map_y,
            'camera_map_x': camera_map_x,
            'camera_map_y': camera_map_y,
            'camera_yaw': camera_yaw,
            'camera_pitch': self.camera_pitch_rad,
            'relative_forward_m': relative_forward_m,
            'relative_left_m': relative_left_m,
            'relative_right_m': camera_right_m,
            'relative_down_m': camera_down_m,
            'board_map_x': board_map_x,
            'board_map_y': board_map_y,
        }

    def validate_board_map_position(
        self,
        direction: str,
        board_position: Dict[str, float],
    ) -> Dict[str, object]:
        target_x, target_y = self.expected_board_position(direction)
        board_x = float(board_position['board_map_x'])
        board_y = float(board_position['board_map_y'])

        error_x = board_x - target_x
        error_y = board_y - target_y
        error_radius = math.hypot(error_x, error_y)

        matched = (
            abs(error_x) <= self.board_map_tolerance_x_m
            and abs(error_y) <= self.board_map_tolerance_y_m
            and error_radius <= self.board_map_tolerance_radius_m
        )

        return {
            'matched': matched,
            'target_x': target_x,
            'target_y': target_y,
            'error_x': error_x,
            'error_y': error_y,
            'error_radius': error_radius,
        }

    # ============================================================
    # 清晰度、裁剪与候选选择
    # ============================================================
    def expand_bbox(
        self,
        bbox,
        image_w: int,
        image_h: int,
    ) -> List[int]:
        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        center_x = 0.5 * (x1 + x2)
        center_y = 0.5 * (y1 + y2)
        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)

        width *= self.board_crop_expand_ratio
        height *= self.board_crop_expand_ratio

        return [
            max(0, int(round(center_x - width * 0.5))),
            max(0, int(round(center_y - height * 0.5))),
            min(image_w, int(round(center_x + width * 0.5))),
            min(image_h, int(round(center_y + height * 0.5))),
        ]

    def crop_board_image(
        self,
        rgb: np.ndarray,
        candidate: Candidate,
    ) -> Tuple[np.ndarray, List[int]]:
        image_h, image_w = rgb.shape[:2]
        crop_box = self.expand_bbox(
            candidate['bbox'],
            image_w,
            image_h,
        )
        x1, y1, x2, y2 = crop_box

        if (
            x2 - x1 < self.board_crop_min_size_px
            or y2 - y1 < self.board_crop_min_size_px
        ):
            return rgb.copy(), [0, 0, image_w, image_h]

        crop = rgb[y1:y2, x1:x2].copy()
        if crop.size == 0:
            return rgb.copy(), [0, 0, image_w, image_h]

        return crop, crop_box

    def calculate_board_sharpness(
        self,
        board_image: np.ndarray,
    ) -> Tuple[float, float, float]:
        image = board_image
        _, width = image.shape[:2]

        if width > self.sharpness_max_width:
            scale = self.sharpness_max_width / float(width)
            image = cv2.resize(
                image,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA,
            )

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        laplacian = float(
            cv2.Laplacian(gray, cv2.CV_64F, ksize=3).var()
        )
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        tenengrad = float((grad_x * grad_x + grad_y * grad_y).mean())

        score = (
            0.60 * math.log1p(max(laplacian, 0.0))
            + 0.40 * math.log1p(max(tenengrad, 0.0))
        )
        return score, laplacian, tenengrad

    def update_llm_candidate(
        self,
        rgb: np.ndarray,
        rgb_msg: Image,
        candidate: Candidate,
        confirmed: bool,
        direction: str,
        odom_pose: Tuple[float, float, float],
        pose_gap: float,
        board_position: Dict[str, float],
        map_check: Dict[str, object],
        obstacle_boxes,
    ):
        crop, crop_box = self.crop_board_image(rgb, candidate)
        sharpness, laplacian, tenengrad = (
            self.calculate_board_sharpness(crop)
        )

        cone_free = len(obstacle_boxes) == 0
        level = 2 if confirmed else 1
        category = (level, cone_free)

        record = {
            'image': crop,
            'full_image': rgb.copy(),
            'crop_box': crop_box,
            'image_stamp': self.header_stamp_sec(rgb_msg.header),
            'direction': direction,
            'odom_pose': odom_pose,
            'pose_gap_sec': pose_gap,
            'board_position': dict(board_position),
            'map_check': dict(map_check),
            'confirmed': bool(confirmed),
            'cone_free': cone_free,
            'cone_count': len(obstacle_boxes),
            'sharpness': sharpness,
            'laplacian': laplacian,
            'tenengrad': tenengrad,
            'depth_score': float(candidate.get('score', float('inf'))),
            'distance_m': float(candidate['distance_m']),
        }

        replaced = False
        with self.llm_lock:
            current = self.llm_best_candidates.get(category)
            if current is None:
                self.llm_best_candidates[category] = record
                replaced = True
            else:
                current_key = (
                    float(current['sharpness']),
                    -float(current['map_check']['error_radius']),
                )
                new_key = (
                    sharpness,
                    -float(map_check['error_radius']),
                )
                if new_key > current_key:
                    self.llm_best_candidates[category] = record
                    replaced = True

            self.llm_valid_frame_count += 1
            valid_count = self.llm_valid_frame_count
            self.llm_last_candidate_monotonic = time.monotonic()

        if replaced:
            self.get_logger().info(
                '图生文候选更新 | '
                f'方向={direction} | '
                f'confirmed={confirmed} | '
                f'无锥桶={cone_free} | '
                f'立牌地图=('
                f'{board_position["board_map_x"]:.3f},'
                f'{board_position["board_map_y"]:.3f}) | '
                f'地图误差={map_check["error_radius"]:.3f}m | '
                f'位姿帧差={pose_gap:.3f}s | '
                f'清晰度={sharpness:.4f} | '
                f'Laplacian={laplacian:.1f} | '
                f'Tenengrad={tenengrad:.1f} | '
                f'有效帧={valid_count}'
            )

        return crop_box, sharpness, laplacian, tenengrad

    def choose_final_llm_candidate(self):
        with self.llm_lock:
            candidates = list(self.llm_best_candidates.values())

        if not candidates:
            return None, '没有通过地图核对的立牌候选图片'

        confirmed_items = [item for item in candidates if item['confirmed']]
        if confirmed_items:
            stage_one = confirmed_items
            confirmed_reason = '使用连续深度确认候选'
        else:
            stage_one = candidates
            confirmed_reason = '没有confirmed，回退到单帧通过候选'

        cone_free_items = [item for item in stage_one if item['cone_free']]
        if cone_free_items:
            stage_two = cone_free_items
            cone_reason = '优先使用画面无YOLO锥桶候选'
        else:
            stage_two = stage_one
            cone_reason = '没有无锥桶候选，保留当前候选'

        best = max(
            stage_two,
            key=lambda item: (
                float(item['sharpness']),
                -float(item['map_check']['error_radius']),
            ),
        )

        return best, (
            f'{confirmed_reason}；{cone_reason}；'
            '最终选择立牌裁剪区域清晰度最高的一张'
        )

    def reset_llm_capture(self):
        with self.llm_lock:
            self.llm_capture_started = False
            self.llm_capture_start_monotonic = 0.0
            self.llm_invalid_frame_count = 0
            self.llm_valid_frame_count = 0
            self.llm_last_candidate_monotonic = 0.0
            self.llm_best_candidates.clear()

    def finalize_llm_selection(self, reason: str):
        with self.llm_lock:
            if self.llm_has_triggered:
                return

        best, select_reason = self.choose_final_llm_candidate()
        if best is None:
            self.get_logger().warning(
                f'图生文采集结束但没有可用图片: {reason}'
            )
            self.reset_llm_capture()
            return

        with self.llm_lock:
            if self.llm_has_triggered:
                return
            self.llm_has_triggered = True

        pose = best['odom_pose']
        position = best['board_position']
        check = best['map_check']

        self.get_logger().warning(
            '图生文最优帧确定，准备调用大模型 | '
            f'结束原因={reason} | '
            f'筛选={select_reason} | '
            f'方向={best["direction"]} | '
            f'帧时odom=({pose[0]:.3f},{pose[1]:.3f},'
            f'{math.degrees(pose[2]):.1f}deg) | '
            f'加偏置车辆地图=('
            f'{position["vehicle_map_x"]:.3f},'
            f'{position["vehicle_map_y"]:.3f}) | '
            f'估计立牌地图=('
            f'{position["board_map_x"]:.3f},'
            f'{position["board_map_y"]:.3f}) | '
            f'目标=({check["target_x"]:.3f},'
            f'{check["target_y"]:.3f}) | '
            f'地图误差={check["error_radius"]:.3f}m | '
            f'清晰度={best["sharpness"]:.4f} | '
            f'深度距离={best["distance_m"]:.3f}m'
        )

        self.llm_thread = threading.Thread(
            target=self.run_llm_once,
            args=(best['image'],),
            name='vision_llm_once',
            daemon=True,
        )
        self.llm_thread.start()

    # ============================================================
    # 深度基类处理完每个同步帧后，执行地图核对和选图
    # ============================================================
    def on_depth_frame_processed(
        self,
        rgb_msg: Image,
        depth_msg: Image,
        rgb: np.ndarray,
        depth_m: np.ndarray,
        raw_candidate: Optional[Candidate],
        accepted_candidate: Optional[Candidate],
        reject_reason: str,
        appearance: Dict[str, object],
        obstacle_analysis: Dict[str, object],
        obstacle_boxes,
        obstacle_status: str,
        obstacle_age: float,
        confirmed: bool,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
    ):
        now = time.monotonic()

        with self.llm_lock:
            direction = self.qr_direction
            already_triggered = self.llm_has_triggered
            capture_started = self.llm_capture_started
            capture_start = self.llm_capture_start_monotonic
            last_candidate_time = self.llm_last_candidate_monotonic

        overlay: Dict[str, object] = {
            'direction': direction or 'WAIT',
            'status': 'WAIT_QR',
            'reject_reason': reject_reason,
            'obstacle_status': obstacle_status,
            'map_matched': False,
            'crop_box': None,
        }

        if already_triggered:
            overlay['status'] = 'LLM_TRIGGERED'
            with self.llm_lock:
                self.latest_llm_overlay = overlay
            return

        if direction not in ('顺时针', '逆时针'):
            with self.llm_lock:
                self.latest_llm_overlay = overlay
            return

        pose, pose_mode, pose_gap, used_header = self.resolve_pose_for_frame(
            rgb_msg
        )
        overlay['pose_mode'] = pose_mode
        overlay['pose_gap_sec'] = pose_gap
        overlay['used_header_stamp'] = used_header

        if pose is None:
            overlay['status'] = 'WAIT_FRAME_POSE'
            self._handle_invalid_llm_frame(now)
            with self.llm_lock:
                self.latest_llm_overlay = overlay
            return

        overlay['odom_pose'] = pose
        overlay['vehicle_map'] = (
            pose[0] + self.odom_offset_x,
            pose[1] + self.odom_offset_y,
        )

        if accepted_candidate is None:
            if raw_candidate is None:
                overlay['status'] = 'NO_DEPTH_BOARD'
            elif reject_reason == 'all_white':
                overlay['status'] = 'REJECT_WHITE'
            elif reject_reason == 'yolo_obstacle':
                overlay['status'] = 'REJECT_OBSTACLE'
            else:
                overlay['status'] = 'REJECT_DEPTH'

            self._handle_invalid_llm_frame(now)
            with self.llm_lock:
                self.latest_llm_overlay = overlay
            return

        board_position = self.estimate_board_map_position(
            accepted_candidate,
            pose,
            fx,
            fy,
            cx,
            cy,
        )
        map_check = self.validate_board_map_position(
            direction,
            board_position,
        )

        overlay['board_position'] = board_position
        overlay['map_check'] = map_check
        overlay['map_matched'] = bool(map_check['matched'])

        if not map_check['matched']:
            overlay['status'] = 'MAP_REJECT'
            self._handle_invalid_llm_frame(now)
            with self.llm_lock:
                self.latest_llm_overlay = overlay
            return

        overlay['status'] = 'VALID_BOARD'

        with self.llm_lock:
            if not self.llm_capture_started:
                self.llm_capture_started = True
                self.llm_capture_start_monotonic = now
                self.llm_invalid_frame_count = 0
                self.llm_valid_frame_count = 0
                self.llm_last_candidate_monotonic = 0.0
                self.llm_best_candidates.clear()
                capture_start = now

                self.get_logger().warning(
                    '地图位置核对成功，开始采集图生文候选 | '
                    f'方向={direction} | '
                    f'估计立牌=('
                    f'{board_position["board_map_x"]:.3f},'
                    f'{board_position["board_map_y"]:.3f}) | '
                    f'目标=({map_check["target_x"]:.3f},'
                    f'{map_check["target_y"]:.3f}) | '
                    f'误差={map_check["error_radius"]:.3f}m'
                )

            self.llm_invalid_frame_count = 0
            capture_start = self.llm_capture_start_monotonic
            last_candidate_time = self.llm_last_candidate_monotonic

        if (
            self.candidate_interval_sec <= 0.0
            or now - last_candidate_time >= self.candidate_interval_sec
        ):
            crop_box, sharpness, laplacian, tenengrad = (
                self.update_llm_candidate(
                    rgb,
                    rgb_msg,
                    accepted_candidate,
                    confirmed,
                    direction,
                    pose,
                    pose_gap,
                    board_position,
                    map_check,
                    obstacle_boxes,
                )
            )
            overlay['crop_box'] = crop_box
            overlay['sharpness'] = sharpness
            overlay['laplacian'] = laplacian
            overlay['tenengrad'] = tenengrad

        with self.llm_lock:
            valid_count = self.llm_valid_frame_count

        elapsed = now - capture_start
        if valid_count >= self.capture_max_valid_frames:
            self.finalize_llm_selection(
                f'达到最大有效候选帧数 {self.capture_max_valid_frames}'
            )
        elif elapsed >= self.capture_max_duration_sec:
            self.finalize_llm_selection(
                f'达到最长采集时间 {self.capture_max_duration_sec:.1f}s'
            )

        with self.llm_lock:
            self.latest_llm_overlay = overlay

    def _handle_invalid_llm_frame(self, now: float):
        with self.llm_lock:
            if not self.llm_capture_started or self.llm_has_triggered:
                return

            self.llm_invalid_frame_count += 1
            invalid_count = self.llm_invalid_frame_count
            capture_start = self.llm_capture_start_monotonic
            valid_count = self.llm_valid_frame_count

        if (
            invalid_count >= self.capture_exit_confirm_frames
            and valid_count > 0
        ):
            self.finalize_llm_selection(
                f'连续 {invalid_count} 帧未通过最终立牌条件'
            )
            return

        if now - capture_start >= self.capture_max_duration_sec:
            self.finalize_llm_selection(
                f'达到最长采集时间 {self.capture_max_duration_sec:.1f}s'
            )

    # ============================================================
    # 调试图地图状态叠加
    # ============================================================
    def decorate_debug_image(
        self,
        debug: np.ndarray,
        rgb_msg: Image,
    ) -> np.ndarray:
        with self.llm_lock:
            overlay = dict(self.latest_llm_overlay)
            capture_started = self.llm_capture_started
            valid_count = self.llm_valid_frame_count
            invalid_count = self.llm_invalid_frame_count
            triggered = self.llm_has_triggered

        crop_box = overlay.get('crop_box')
        if isinstance(crop_box, list) and len(crop_box) >= 4:
            x1, y1, x2, y2 = [int(v) for v in crop_box[:4]]
            cv2.rectangle(
                debug,
                (x1, y1),
                (x2, y2),
                (255, 255, 0),
                2,
            )
            cv2.putText(
                debug,
                'LLM CROP',
                (x1 + 4, max(20, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

        status = str(overlay.get('status', 'WAIT'))
        direction = str(overlay.get('direction', 'WAIT'))
        status_color = (
            (0, 255, 0)
            if status in ('VALID_BOARD', 'LLM_TRIGGERED')
            else (0, 0, 255)
        )

        lines = [
            f'LLM={status} DIR={direction}',
        ]

        pose = overlay.get('odom_pose')
        vehicle_map = overlay.get('vehicle_map')
        if pose is not None and vehicle_map is not None:
            lines.append(
                f'ODOM=({pose[0]:.2f},{pose[1]:.2f}) '
                f'MAP=({vehicle_map[0]:.2f},{vehicle_map[1]:.2f})'
            )

        gap = overlay.get('pose_gap_sec')
        mode = overlay.get('pose_mode')
        if isinstance(gap, (float, int)) and math.isfinite(float(gap)):
            lines.append(f'POSE_SYNC={mode} GAP={float(gap):.3f}s')

        position = overlay.get('board_position')
        check = overlay.get('map_check')
        if isinstance(position, dict) and isinstance(check, dict):
            lines.append(
                f'BOARD_MAP=({position["board_map_x"]:.2f},'
                f'{position["board_map_y"]:.2f}) '
                f'TARGET=({check["target_x"]:.2f},'
                f'{check["target_y"]:.2f})'
            )
            lines.append(
                f'MAP_ERR=({check["error_x"]:+.2f},'
                f'{check["error_y"]:+.2f}) '
                f'R={check["error_radius"]:.2f}m '
                f'MATCH={check["matched"]}'
            )

        lines.append(
            f'CAPTURE={capture_started} VALID={valid_count} '
            f'LOST={invalid_count} TRIGGERED={triggered}'
        )

        start_y = 28
        for index, line in enumerate(lines):
            y = start_y + index * 22
            cv2.putText(
                debug,
                line,
                (12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                status_color,
                2,
                cv2.LINE_AA,
            )

        return debug

    # ============================================================
    # 状态日志
    # ============================================================
    def llm_status_callback(self):
        with self.llm_lock:
            direction = self.qr_direction
            cache_count = len(self.pose_cache)
            capture = self.llm_capture_started
            valid_count = self.llm_valid_frame_count
            invalid_count = self.llm_invalid_frame_count
            categories = len(self.llm_best_candidates)
            triggered = self.llm_has_triggered
            overlay = dict(self.latest_llm_overlay)

        if triggered:
            return

        self.get_logger().info(
            '图生文状态 | '
            f'方向={direction or "等待"} | '
            f'位姿缓存={cache_count} | '
            f'帧状态={overlay.get("status", "WAIT")} | '
            f'采集中={capture} | '
            f'有效帧={valid_count} | '
            f'连续无效={invalid_count}/'
            f'{self.capture_exit_confirm_frames} | '
            f'候选类别={categories}/4'
        )

    # ============================================================
    # 千问调用和OLED兼容流式发布
    # ============================================================
    @staticmethod
    def extract_chat_delta_text(delta) -> str:
        """从OpenAI兼容Chat Completions流式delta中提取回复文本。"""
        if delta is None:
            return ''

        content = getattr(delta, 'content', None)
        if isinstance(content, str):
            return content

        # 兼容部分SDK把多模态文本内容表示为列表的情况。
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                    continue

                if isinstance(item, dict):
                    text_value = item.get('text')
                else:
                    text_value = getattr(item, 'text', None)

                if text_value:
                    parts.append(str(text_value))

            return ''.join(parts)

        return ''

    def publish_llm_result(self, text: str):
        msg = String()
        msg.data = text
        self.llm_result_pub.publish(msg)

    def publish_stream_status(self, status: str):
        msg = String()
        msg.data = status
        self.stream_status_pub.publish(msg)

    def publish_text_as_stream(
        self,
        text: str,
        current_text: str = '',
    ) -> str:
        if len(current_text) >= self.max_output_chars:
            return current_text[:self.max_output_chars]

        remaining = self.max_output_chars - len(current_text)
        for char in str(text)[:remaining]:
            current_text += char
            print(char, end='', flush=True)
            self.publish_llm_result(current_text)

            if self.stream_char_interval_sec > 0.0:
                time.sleep(self.stream_char_interval_sec)

        return current_text

    def run_llm_once(self, cv_img: np.ndarray):
        """
        将现有筛选流程选出的最优立牌裁剪图发送给qwen3.7-plus。

        不改变前面的深度尺寸、全白过滤、YOLO排除、地图坐标核对、
        多帧采集和清晰度选图逻辑。
        """
        stream_started = False
        total_start = time.monotonic()

        try:
            height, width = cv_img.shape[:2]
            encode_start = time.monotonic()

            ok, encoded = cv2.imencode(
                '.jpg',
                cv_img,
                [
                    int(cv2.IMWRITE_JPEG_QUALITY),
                    self.llm_jpeg_quality,
                ],
            )
            if not ok:
                raise RuntimeError('立牌裁剪图JPEG编码失败')

            encoded_bytes = encoded.tobytes()
            base64_image = base64.b64encode(
                encoded_bytes
            ).decode('ascii')
            image_data_url = (
                f'data:image/jpeg;base64,{base64_image}'
            )

            encode_elapsed_ms = (
                time.monotonic() - encode_start
            ) * 1000.0

            self.get_logger().info(
                '准备发送千问立牌最优裁剪图 | '
                f'尺寸={width}x{height} | '
                f'JPEG质量={self.llm_jpeg_quality} | '
                f'JPEG={len(encoded_bytes) / 1024.0:.1f}KB | '
                f'编码耗时={encode_elapsed_ms:.1f}ms'
            )

            self.publish_stream_status('start')
            stream_started = True

            request_start = time.monotonic()
            first_text_received = False

            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        'role': 'user',
                        'content': [
                            {
                                'type': 'image_url',
                                'image_url': {
                                    'url': image_data_url,
                                },
                                'max_pixels': self.max_pixels,
                            },
                            {
                                'type': 'text',
                                'text': self.prompt_text,
                            },
                        ],
                    }
                ],
                max_tokens=self.max_output_tokens,
                stream=True,
                extra_body={
                    # qwen3.7-plus默认开启思考，简单图生文显式关闭。
                    'enable_thinking': self.enable_thinking,
                },
            )

            create_elapsed_ms = (
                time.monotonic() - request_start
            ) * 1000.0
            self.get_logger().info(
                '千问流式连接已建立 | '
                f'chat.completions.create耗时={create_elapsed_ms:.1f}ms'
            )
            self.get_logger().info(
                '================ 千问流式识别结果 ================'
            )

            full_text = ''

            for chunk in stream:
                choices = getattr(chunk, 'choices', None) or []
                if not choices:
                    continue

                delta = getattr(choices[0], 'delta', None)
                delta_text = self.extract_chat_delta_text(delta)
                if not delta_text:
                    continue

                if not first_text_received:
                    first_text_received = True
                    first_text_ms = (
                        time.monotonic() - request_start
                    ) * 1000.0
                    self.get_logger().warning(
                        f'收到千问首字 | API首字延迟={first_text_ms:.1f}ms'
                    )

                full_text = self.publish_text_as_stream(
                    delta_text,
                    full_text,
                )

            full_text = full_text.strip()[:self.max_output_chars]
            if not full_text:
                raise RuntimeError(
                    '千问流式响应结束，但未提取到图生文文本'
                )

            self.publish_llm_result(full_text)
            print('', flush=True)

            total_elapsed_ms = (
                time.monotonic() - total_start
            ) * 1000.0

            self.get_logger().warning(
                f'千问图生文完成 | 话题={self.llm_result_topic} | '
                f'字符数={len(full_text)} | '
                f'总耗时={total_elapsed_ms:.1f}ms'
            )
            self.publish_stream_status('done')

        except Exception as exc:
            if stream_started:
                self.publish_stream_status('error')
            self.get_logger().error(
                f'阿里云百炼千问图生文调用失败: {exc}'
            )


def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = VisionBoardLLMNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f'节点启动失败: {exc}', file=sys.stderr)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
