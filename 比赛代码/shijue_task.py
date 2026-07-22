#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
任务1巡航 + 锥桶避障 + 地图边界（纯 PD 融合版）

角速度约定：
    w > 0：左转
    w < 0：右转

正常运动只保留三项控制：
    w_total = w_target + w_boundary + w_cone

1. 目标点角度 PD：
    e_yaw = normalize(target_yaw - current_yaw)
    w_target = target_kp * e_yaw + target_kd * de_yaw/dt

2. 地图边界 PD：
    上边界：仅 x < 3.9 时生效，e = y_actual - y_upper_boundary
    下边界：e = y_actual - y_lower_boundary
    右边界：物理边界 x=5.0，距边界 0.20m 开始保护，
              e = x_right_protect_start - x_actual（负值），输出左转角速度
    w_boundary = -(boundary_kp * e + boundary_kd * de/dt)

3. 锥桶 PD：
    d1 = (bottom_y - cone_trigger_y) * cone_near_k
    e_x = cone_center_x - image_center_x
    d2 = cone_kp * e_x + cone_kd * de_x/dt
    w_cone = d1 * d2

当前版本新增：
- 图像左右边缘忽略区域：根据实测点轻度平滑后分段线性插值；
- 右边界关于图像中心自动对称；
- 边界向图像内部增加少量安全余量，降低测量误差影响。

仍未使用：
- 280/300 双阈值方向预判；
- 图像中心强制选择避障方向；
- 锥桶 D 项防反向处理；
- 上下地图区域强制修改锥桶方向；
- 中央近距离角速度额外增强；
- “y + 航向角”紧急强制右转；
- 任何固定角速度边界接管。

保留任务必需逻辑：目标点停车锁存、二维码结果交接、ACK 重发、
QoS、线程锁及输出限幅。
"""

import math
import os
import threading
import time

import rclpy
from ai_msgs.msg import PerceptionTargets
from geometry_msgs.msg import Pose2D, Twist
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from std_msgs.msg import String


class Task1NavAvoidNode(Node):
    def __init__(self):
        super().__init__('task1_nav_avoid_node')

        # ================================================================
        # [配置区 1]：目标点、坐标偏移和目标角度 PD
        # ================================================================
        self.target_map_x = 5.0
        self.target_map_y = 2.0

        # odom_pose=(0,0) 对应地图坐标 (0.55,0.22)。
        self.start_offset_x = 0.55
        self.start_offset_y = 0.22

        # 线速度：v = kp_linear * 目标距离。
        self.kp_linear = 1.0
        self.max_v = 1.0

        # 目标角度 PD。
        self.target_kp = 3.0
        self.target_kd = 0.20
        self.target_d_filter_alpha = 0.35
        self.target_d_limit = 6.0
        self.target_w_limit = 2.0

        # 总角速度限幅。
        self.max_w = 2.0
        self.arrival_tolerance = 0.5

        # ================================================================
        # [配置区 2]：锥桶 PD
        # ================================================================
        self.conf_thresh = 0.6

        # 只使用一个固定触发阈值，不再使用 280/300 双阈值。
        self.cone_trigger_y = 290.0

        # d1 = (bottom_y - cone_trigger_y) * cone_near_k
        self.cone_near_k = 0.125 #0.125
        self.cone_near_max = 1.50

        # d2 = cone_kp * e_x + cone_kd * de_x/dt
        self.cone_kp = 0.32 #0.3
        self.cone_kd = 0.0010
        self.cone_d_filter_alpha = 0.35
        self.cone_d_limit = 800.0
        self.cone_w_limit = 1.80

        self.image_width = 640.0
        self.image_height = 480.0

        # ================================================================
        # [配置区 2.1]：摄像头画面边缘忽略
        # ================================================================
        # 用户实测数据的坐标顺序为：
        #     (bottom_y, 左侧边界 x)
        # 即检测框底部位于 bottom_y 时，中心点若落在该左边界之外，
        # 就认为目标处于画面边缘区域，不参与锥桶 PD。
        self.edge_measure_points = [
            (290.0, 155.0),
            (295.0, 144.0),
            (300.0, 120.0),
            (305.0, 101.0),
            (310.0, 92.0),
            (315.0, 70.0),
            (320.0, 66.0),
            (325.0, 60.0),
            (330.0, 56.0),
            (335.0, 54.0),
            (340.0, 49.0),
            (345.0, 30.0),
        ]

        # 对实测 x 做一次 1:2:1 邻域平滑，降低单点测量误差。
        self.edge_smoothing_enabled = True

        # 将左右有效区域各向图像内部收缩 5 px。
        # 数值增大：忽略边缘范围更宽；数值减小：保留更多边缘检测。
        self.edge_ignore_margin_px = 5.0

        # 根据上述点建立分段线性边界；右边界关于 x=320 对称。
        self._build_edge_boundaries()

        # ================================================================
        # [配置区 3]：地图边界 PD
        # ================================================================
        self.y_lower_boundary = 0.20
        self.y_upper_boundary = 1.60

        # 上边界保护只在 x < 3.9 区域启用；3.9~5.0 为上边界豁免区。
        self.upper_boundary_disable_x_start = 3.90
        self.upper_boundary_disable_x_end = 5.00

        # 右侧物理边界为 x=5.0。提前 0.20m 开始左转保护，避免真正越界后才动作。
        self.x_right_boundary = 5.00
        self.x_right_protect_distance = 0.20
        self.x_right_protect_start = (
            self.x_right_boundary - self.x_right_protect_distance
        )

        self.boundary_kp = 76.0 #60
        self.boundary_kd = 2.0 #2.0
        self.boundary_d_filter_alpha = 0.35
        self.boundary_d_limit = 2.0
        self.boundary_w_limit = 18.0

        # ================================================================
        # [配置区 4]：目标点停车和二维码交接
        # ================================================================
        self.qr_result_topic = '/qr_direction_result'
        self.qr_success_topic = '/qr_success'
        self.channel_ack_topic = '/channel_navigation_ack'

        self.fast_stop_burst_count = 4
        self.fast_stop_hold_sec = 0.35
        self.fast_stop_timer_period = 0.005

        self.handoff_repeat_period = 0.5
        self.handoff_repeat_limit = 20

        # ================================================================
        # [状态变量]
        # ================================================================
        self.cur_pose = [0.0, 0.0, 0.0]
        self.pose_received = False
        self.navigation_arrived = False
        self.is_finished = False

        self.last_cmd_v = 0.0
        self.last_cmd_w = 0.0

        # 目标点 PD 状态，只在收到新位姿时更新。
        self.target_dist = float('inf')
        self.target_v = 0.0
        self.target_w = 0.0
        self.target_angle_error = 0.0
        self.target_angle_d = 0.0
        self.target_prev_error = None
        self.target_prev_time = None
        self.target_d_filtered = 0.0

        # 边界 PD 状态，只在收到新位姿时更新。
        self.boundary_active = False
        self.boundary_mode = '安全区'
        self.boundary_error = 0.0
        self.boundary_error_d = 0.0
        self.boundary_w = 0.0
        self.boundary_prev_error = None
        self.boundary_prev_time = None
        self.boundary_d_filtered = 0.0

        # 锥桶 PD 状态，只在收到新检测帧时更新。
        # 两个检测帧之间持续使用最近一次计算结果。
        self.cone_control_active = False
        self.current_cone_w = 0.0
        self.cone_prev_error = None
        self.cone_prev_time = None
        self.cone_d_filtered = 0.0
        self.last_cone_center_x = 0.0
        self.last_cone_bottom_y = 0.0

        # 二维码交接状态。
        self.qr_result_received = False
        self.handoff_complete = False
        self.qr_result = ''
        self.handoff_publish_count = 0
        self.channel_ack_received = False
        self.channel_ack_data = ''

        # 停车和运动发布状态。
        self.motion_lock = threading.RLock()
        self.stop_latched = False
        self.fast_stop_until = 0.0

        # ================================================================
        # [回调组与 ROS 通信]
        # ================================================================
        self.qr_callback_group = MutuallyExclusiveCallbackGroup()
        self.stop_callback_group = MutuallyExclusiveCallbackGroup()
        self.control_callback_group = MutuallyExclusiveCallbackGroup()
        self.obstacle_callback_group = MutuallyExclusiveCallbackGroup()
        self.pose_callback_group = MutuallyExclusiveCallbackGroup()

        self.pose_sub = self.create_subscription(
            Pose2D,
            'odom_pose',
            self.pose_cb,
            10,
            callback_group=self.pose_callback_group,
        )

        obstacle_qos = QoSProfile(depth=1)
        obstacle_qos.history = HistoryPolicy.KEEP_LAST
        obstacle_qos.reliability = ReliabilityPolicy.BEST_EFFORT
        obstacle_qos.durability = DurabilityPolicy.VOLATILE
        self.obs_sub = self.create_subscription(
            PerceptionTargets,
            'racing_obstacle_detection',
            self.obs_cb,
            obstacle_qos,
            callback_group=self.obstacle_callback_group,
        )

        self.qr_result_sub = self.create_subscription(
            String,
            self.qr_result_topic,
            self.qr_result_cb,
            10,
            callback_group=self.qr_callback_group,
        )

        # 兼容节点运行在 namespace 中的情况。
        self.qr_result_sub_relative = None
        relative_qr_topic = self.resolve_topic_name('qr_direction_result')
        if relative_qr_topic != self.qr_result_topic:
            self.qr_result_sub_relative = self.create_subscription(
                String,
                'qr_direction_result',
                self.qr_result_cb,
                10,
                callback_group=self.qr_callback_group,
            )

        handoff_qos = QoSProfile(depth=1)
        handoff_qos.reliability = ReliabilityPolicy.RELIABLE
        handoff_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.qr_success_pub = self.create_publisher(
            String,
            self.qr_success_topic,
            handoff_qos,
        )
        self.channel_ack_sub = self.create_subscription(
            String,
            self.channel_ack_topic,
            self.channel_ack_cb,
            handoff_qos,
        )

        cmd_qos = QoSProfile(depth=1)
        cmd_qos.history = HistoryPolicy.KEEP_LAST
        cmd_qos.reliability = ReliabilityPolicy.RELIABLE
        cmd_qos.durability = DurabilityPolicy.VOLATILE
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', cmd_qos)

        # 40 Hz 持续发布最近一次融合结果。
        self.timer = self.create_timer(
            0.025,
            self.control_loop,
            callback_group=self.control_callback_group,
        )

        # 到达目标点后的 200 Hz 快速零速度覆盖。
        self.fast_stop_timer = self.create_timer(
            self.fast_stop_timer_period,
            self.fast_stop_loop,
            callback_group=self.stop_callback_group,
        )
        self.fast_stop_timer.cancel()

        self.handoff_repeat_timer = self.create_timer(
            self.handoff_repeat_period,
            self.handoff_repeat_loop,
        )
        self.handoff_repeat_timer.cancel()

        self.get_logger().info(
            '🚀 纯PD融合巡航节点启动：'
            'W总=W目标+W边界+W锥桶；w>0左转，w<0右转'
        )
        self.get_logger().info(
            f'目标PD=({self.target_kp:.3f},{self.target_kd:.3f})，'
            f'边界PD=({self.boundary_kp:.3f},{self.boundary_kd:.3f})，'
            f'锥桶PD=({self.cone_kp:.4f},{self.cone_kd:.4f})，'
            f'锥桶阈值={self.cone_trigger_y:.0f}，Wmax={self.max_w:.2f}'
        )
        self.get_logger().info(
            f'🧱 边界规则：上边界在 x=['
            f'{self.upper_boundary_disable_x_start:.2f},'
            f'{self.upper_boundary_disable_x_end:.2f}] 禁用；'
            f'右边界 x={self.x_right_boundary:.2f}，'
            f'x>={self.x_right_protect_start:.2f} 开始左转保护'
        )

    # ------------------------------------------------------------------
    # 通用辅助函数
    # ------------------------------------------------------------------
    @staticmethod
    def clamp(value, lower, upper):
        return max(lower, min(value, upper))

    @staticmethod
    def normalize_angle(angle):
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    # ------------------------------------------------------------------
    # 位姿更新：目标点 PD + 边界 PD
    # ------------------------------------------------------------------
    def pose_cb(self, msg):
        now = time.monotonic()

        with self.motion_lock:
            first_pose = not self.pose_received
            self.cur_pose = [float(msg.x), float(msg.y), float(msg.theta)]
            self.pose_received = True

            self._update_target_pd_from_pose_locked(now)
            self._update_boundary_pd_from_pose_locked(now)

            actual_x = self.cur_pose[0] + self.start_offset_x
            actual_y = self.cur_pose[1] + self.start_offset_y

        if first_pose:
            self.get_logger().info(
                f'📍 位姿已连接，actual=({actual_x:.3f},{actual_y:.3f})'
            )

    def _update_target_pd_from_pose_locked(self, now):
        actual_x = self.cur_pose[0] + self.start_offset_x
        actual_y = self.cur_pose[1] + self.start_offset_y
        actual_yaw = self.cur_pose[2]

        dx = self.target_map_x - actual_x
        dy = self.target_map_y - actual_y
        dist = math.hypot(dx, dy)
        target_yaw = math.atan2(dy, dx)
        angle_error = self.normalize_angle(target_yaw - actual_yaw)

        angle_d = 0.0
        if self.target_prev_error is not None and self.target_prev_time is not None:
            dt = now - self.target_prev_time
            if dt >= 1e-3:
                delta_error = self.normalize_angle(
                    angle_error - self.target_prev_error
                )
                raw_d = delta_error / dt
                raw_d = self.clamp(
                    raw_d,
                    -self.target_d_limit,
                    self.target_d_limit,
                )
                alpha = self.target_d_filter_alpha
                self.target_d_filtered = (
                    alpha * raw_d
                    + (1.0 - alpha) * self.target_d_filtered
                )
                angle_d = self.target_d_filtered

        self.target_prev_error = angle_error
        self.target_prev_time = now

        target_w_raw = (
            self.target_kp * angle_error
            + self.target_kd * angle_d
        )

        self.target_dist = dist
        self.target_angle_error = angle_error
        self.target_angle_d = angle_d
        self.target_v = self.clamp(
            self.kp_linear * dist,
            0.0,
            self.max_v,
        )
        self.target_w = self.clamp(
            target_w_raw,
            -self.target_w_limit,
            self.target_w_limit,
        )

    def _reset_boundary_pd_locked(self):
        self.boundary_active = False
        self.boundary_mode = '安全区'
        self.boundary_error = 0.0
        self.boundary_error_d = 0.0
        self.boundary_w = 0.0
        self.boundary_prev_error = None
        self.boundary_prev_time = None
        self.boundary_d_filtered = 0.0

    def _update_boundary_pd_from_pose_locked(self, now):
        actual_x = self.cur_pose[0] + self.start_offset_x
        actual_y = self.cur_pose[1] + self.start_offset_y

        # 右边界优先级最高：接近 x=5.0 时必须向左转。
        # error 在保护区内为负值，因此沿用统一公式
        # w_boundary = -(Kp*error + Kd*error_d) 后得到正角速度（左转）。
        if actual_x >= self.x_right_protect_start:
            mode = '右边界PD'
            error = self.x_right_protect_start - actual_x

        # x 位于 3.9~5.0 时，明确取消上边界保护。
        elif (
            actual_y >= self.y_upper_boundary
            and not (
                self.upper_boundary_disable_x_start
                <= actual_x
                <= self.upper_boundary_disable_x_end
            )
        ):
            mode = '上边界PD'
            error = actual_y - self.y_upper_boundary

        elif actual_y <= self.y_lower_boundary:
            mode = '下边界PD'
            error = actual_y - self.y_lower_boundary
        else:
            self._reset_boundary_pd_locked()
            return

        error_d = 0.0
        same_boundary = self.boundary_active and self.boundary_mode == mode

        if (
            same_boundary
            and self.boundary_prev_error is not None
            and self.boundary_prev_time is not None
        ):
            dt = now - self.boundary_prev_time
            if dt >= 1e-3:
                raw_d = (error - self.boundary_prev_error) / dt
                raw_d = self.clamp(
                    raw_d,
                    -self.boundary_d_limit,
                    self.boundary_d_limit,
                )
                alpha = self.boundary_d_filter_alpha
                self.boundary_d_filtered = (
                    alpha * raw_d
                    + (1.0 - alpha) * self.boundary_d_filtered
                )
                error_d = self.boundary_d_filtered
        else:
            # 刚进入边界区时先不使用 D，避免首帧微分尖峰。
            self.boundary_d_filtered = 0.0

        self.boundary_active = True
        self.boundary_mode = mode
        self.boundary_error = error
        self.boundary_error_d = error_d
        self.boundary_prev_error = error
        self.boundary_prev_time = now

        boundary_w_raw = -(
            self.boundary_kp * error
            + self.boundary_kd * error_d
        )
        self.boundary_w = self.clamp(
            boundary_w_raw,
            -self.boundary_w_limit,
            self.boundary_w_limit,
        )

    # ------------------------------------------------------------------
    # 摄像头画面边缘忽略
    # ------------------------------------------------------------------
    def _build_edge_boundaries(self):
        """
        根据实测点建立左右边缘函数。

        处理步骤：
        1. 将点按 bottom_y 排序；
        2. 对左边界 x 使用轻度 1:2:1 平滑，降低单点测量误差；
        3. 在相邻点之间进行分段线性插值；
        4. 左边界向内增加 edge_ignore_margin_px；
        5. 右边界关于图像中心自动对称。

        超出实测 y 范围时使用端点值，避免曲线外推产生异常边界。
        """
        points = sorted(
            [(float(y), float(x)) for y, x in self.edge_measure_points],
            key=lambda item: item[0],
        )

        if len(points) < 2:
            raise ValueError('edge_measure_points 至少需要两个点')

        y_values = [item[0] for item in points]
        raw_x_values = [item[1] for item in points]

        if self.edge_smoothing_enabled and len(raw_x_values) >= 3:
            smoothed_x_values = []
            last_index = len(raw_x_values) - 1

            for index, current_x in enumerate(raw_x_values):
                if index == 0:
                    # 首点采用 3:1 平滑，避免端点移动过大。
                    smooth_x = (
                        0.75 * current_x
                        + 0.25 * raw_x_values[index + 1]
                    )
                elif index == last_index:
                    smooth_x = (
                        0.25 * raw_x_values[index - 1]
                        + 0.75 * current_x
                    )
                else:
                    smooth_x = (
                        0.25 * raw_x_values[index - 1]
                        + 0.50 * current_x
                        + 0.25 * raw_x_values[index + 1]
                    )

                smoothed_x_values.append(smooth_x)
        else:
            smoothed_x_values = raw_x_values[:]

        self.edge_y_values = y_values
        self.edge_left_x_values = smoothed_x_values

        self.get_logger().info(
            f'✅ 画面边缘忽略已构建：实测点={len(points)}，'
            f'bottom_y范围=[{y_values[0]:.0f},{y_values[-1]:.0f}]，'
            f'平滑={self.edge_smoothing_enabled}，'
            f'内缩余量={self.edge_ignore_margin_px:.1f}px'
        )

    def _interpolate_left_edge_x(self, bottom_y):
        """按照 bottom_y 分段线性插值左边界 x。"""
        y = float(bottom_y)
        y_values = self.edge_y_values
        x_values = self.edge_left_x_values

        # 不做多项式外推：实测范围外直接保持端点边界。
        if y <= y_values[0]:
            return x_values[0]
        if y >= y_values[-1]:
            return x_values[-1]

        # 点数量很少，顺序查找足够快，并且不增加额外依赖。
        for index in range(1, len(y_values)):
            y1 = y_values[index]
            if y <= y1:
                y0 = y_values[index - 1]
                x0 = x_values[index - 1]
                x1 = x_values[index]
                ratio = (y - y0) / (y1 - y0)
                return x0 + ratio * (x1 - x0)

        return x_values[-1]

    def get_edge_boundaries(self, bottom_y):
        """返回指定 bottom_y 下，加入安全余量后的左右有效边界。"""
        center_x = self.image_width / 2.0

        left_x = (
            self._interpolate_left_edge_x(bottom_y)
            + self.edge_ignore_margin_px
        )

        # 防止错误参数让左右边界交叉。
        left_x = self.clamp(left_x, 0.0, center_x - 1.0)
        right_x = 2.0 * center_x - left_x
        return left_x, right_x

    def is_in_edge_ignore_zone(self, center_x, bottom_y):
        """检测框中心落在左右有效边界之外时返回 True。"""
        left_x, right_x = self.get_edge_boundaries(bottom_y)
        return center_x < left_x or center_x > right_x

    # ------------------------------------------------------------------
    # 锥桶 PD
    # ------------------------------------------------------------------
    def _clear_cone_control_locked(self):
        self.cone_control_active = False
        self.current_cone_w = 0.0
        self.cone_prev_error = None
        self.cone_prev_time = None
        self.cone_d_filtered = 0.0
        self.last_cone_center_x = 0.0
        self.last_cone_bottom_y = 0.0

    def collect_valid_obstacle_candidates(self, msg):
        """
        保留满足以下条件的锥桶候选：
        1. 置信度大于阈值；
        2. bottom_y 超过固定触发阈值；
        3. 检测框中心位于左右拟合边界之间。

        返回元素：(bottom_y, center_x, roi)
        """
        candidates = []

        for target in msg.targets:
            for roi in target.rois:
                if roi.confidence <= self.conf_thresh:
                    continue

                rect = roi.rect
                center_x = rect.x_offset + rect.width / 2.0
                bottom_y = rect.y_offset + rect.height

                if bottom_y <= self.cone_trigger_y:
                    continue

                if self.is_in_edge_ignore_zone(center_x, bottom_y):
                    left_x, right_x = self.get_edge_boundaries(bottom_y)
                    self.get_logger().info(
                        f'🟡 画面边缘忽略：center_x={center_x:.0f}，'
                        f'bottom_y={bottom_y:.0f}，'
                        f'有效范围=[{left_x:.0f},{right_x:.0f}]，'
                        f'conf={roi.confidence:.2f}',
                        throttle_duration_sec=0.5,
                    )
                    continue

                candidates.append((bottom_y, center_x, roi))

        return candidates

    def _update_cone_derivative_locked(self, error_x, now):
        error_d = 0.0

        if self.cone_prev_error is not None and self.cone_prev_time is not None:
            dt = now - self.cone_prev_time
            if dt >= 1e-3:
                raw_d = (error_x - self.cone_prev_error) / dt
                raw_d = self.clamp(
                    raw_d,
                    -self.cone_d_limit,
                    self.cone_d_limit,
                )
                alpha = self.cone_d_filter_alpha
                self.cone_d_filtered = (
                    alpha * raw_d
                    + (1.0 - alpha) * self.cone_d_filtered
                )
                error_d = self.cone_d_filtered

        self.cone_prev_error = error_x
        self.cone_prev_time = now
        return error_d

    def update_cone_control_locked(self, msg):
        candidates = self.collect_valid_obstacle_candidates(msg)

        if not candidates:
            was_active = self.cone_control_active
            old_w = self.current_cone_w
            self._clear_cone_control_locked()

            if was_active:
                self.get_logger().info(
                    f'✅ 当前检测帧无触发锥桶，锥桶PD清零：旧W={old_w:+.3f}',
                    throttle_duration_sec=0.5,
                )
            return False

        # 选择 bottom_y 最大的锥桶，即图像中最靠近车辆的锥桶。
        bottom_y, center_x, roi = max(candidates, key=lambda item: item[0])

        image_center_x = self.image_width / 2.0
        error_x = center_x - image_center_x
        now = time.monotonic()
        error_x_d = self._update_cone_derivative_locked(error_x, now)

        d1 = (bottom_y - self.cone_trigger_y) * self.cone_near_k
        d1 = self.clamp(d1, 0.0, self.cone_near_max)

        d2_p = self.cone_kp * error_x
        d2_d = self.cone_kd * error_x_d
        d2 = d2_p + d2_d

        cone_w_raw = d1 * d2
        cone_w = self.clamp(
            cone_w_raw,
            -self.cone_w_limit,
            self.cone_w_limit,
        )

        self.cone_control_active = True
        self.current_cone_w = cone_w
        self.last_cone_center_x = center_x
        self.last_cone_bottom_y = bottom_y

        self.get_logger().info(
            f'🟠 锥桶PD：bottom_y={bottom_y:.0f}，'
            f'd1={d1:.3f}，e_x={error_x:+.1f}px，'
            f'de_x={error_x_d:+.1f}px/s，'
            f'd2=P({d2_p:+.3f})+D({d2_d:+.3f})={d2:+.3f}，'
            f'W锥桶={cone_w:+.3f}，conf={roi.confidence:.2f}',
            throttle_duration_sec=0.5,
        )
        return True

    # ------------------------------------------------------------------
    # 停车及二维码交接
    # ------------------------------------------------------------------
    def _publish_zero_locked(self):
        zero = Twist()
        self.last_cmd_v = 0.0
        self.last_cmd_w = 0.0
        self.cmd_pub.publish(zero)

    def _activate_target_stop_locked(self):
        self.stop_latched = True
        self.fast_stop_until = time.monotonic() + self.fast_stop_hold_sec
        self.fast_stop_timer.reset()

        self._clear_cone_control_locked()
        self._reset_boundary_pd_locked()

        for _ in range(self.fast_stop_burst_count):
            self._publish_zero_locked()

    def fast_stop_loop(self):
        with self.motion_lock:
            if self.handoff_complete or self.is_finished:
                self.fast_stop_timer.cancel()
                return

            if not self.stop_latched:
                self.fast_stop_timer.cancel()
                return

            if time.monotonic() <= self.fast_stop_until:
                self._publish_zero_locked()
            else:
                self.fast_stop_timer.cancel()

    def qr_result_cb(self, msg):
        result = str(msg.data).strip()
        if not result:
            return
        if self.handoff_complete or self.qr_result_received:
            return

        self.qr_result = result
        self.get_logger().warn(
            f'📩 收到二维码结果："{result}"，立即向通道导航节点交接'
        )
        self._complete_qr_handoff()

    def _publish_qr_success(self, reason):
        if not self.qr_result:
            self.get_logger().error('无法发布 qr_success：qr_result 为空')
            return

        success_msg = String()
        success_msg.data = self.qr_result
        self.qr_success_pub.publish(success_msg)
        self.handoff_publish_count += 1

        self.get_logger().warn(
            f'📤 /qr_success 第{self.handoff_publish_count}次：'
            f'"{self.qr_result}" ({reason})'
        )

    def _complete_qr_handoff(self):
        if self.handoff_complete or self.qr_result_received:
            return
        if not self.qr_result:
            self.get_logger().error('无法完成交接：二维码结果为空')
            return

        result = self.qr_result

        with self.motion_lock:
            self.qr_result_received = True
            self.handoff_complete = True
            self.is_finished = True
            self._clear_cone_control_locked()
            self._reset_boundary_pd_locked()
            self.fast_stop_timer.cancel()

        self._publish_qr_success('收到二维码解析结果，立即正式交接')
        self.handoff_repeat_timer.reset()

        self.get_logger().warn(
            f'🚦 已交接二维码结果“{result}”：巡航节点停止发布 /cmd_vel；'
            f'等待通道节点通过 {self.channel_ack_topic} 返回 ACK'
        )

    def handoff_repeat_loop(self):
        if not self.handoff_complete or self.channel_ack_received:
            self.handoff_repeat_timer.cancel()
            return

        if self.handoff_publish_count >= self.handoff_repeat_limit:
            self.get_logger().error(
                f'❌ 已重发 {self.qr_success_topic} '
                f'{self.handoff_publish_count} 次仍未收到 ACK；'
                f'请检查 channel_navigation、话题和 ROS_DOMAIN_ID'
            )
            self.handoff_repeat_timer.cancel()
            return

        self._publish_qr_success('等待通道 ACK，定时重发')

    def channel_ack_cb(self, msg):
        ack = str(msg.data).strip()
        self.channel_ack_received = True
        self.channel_ack_data = ack
        self.handoff_repeat_timer.cancel()

        self.get_logger().warn(
            f'✅ 收到通道节点 ACK："{ack}"；'
            f'交接消息共发布 {self.handoff_publish_count} 次，'
            f'巡航节点确认不再发布 /cmd_vel'
        )

    # ------------------------------------------------------------------
    # 统一 PD 融合控制
    # ------------------------------------------------------------------
    def obs_cb(self, msg):
        with self.motion_lock:
            if self.is_finished or self.handoff_complete:
                return
            if self.stop_latched or self.navigation_arrived:
                return

            self.update_cone_control_locked(msg)
            self._publish_fused_control_locked('检测帧更新')

    def control_loop(self):
        with self.motion_lock:
            self._publish_fused_control_locked('40Hz保持')

    def _publish_fused_control_locked(self, source):
        if self.is_finished or self.handoff_complete:
            return

        if self.stop_latched or self.navigation_arrived:
            self.execute_drive(
                0.0,
                0.0,
                '🏁 已到达第一目标点，停车保持',
            )
            return

        # 第一帧位姿到达之前不发布巡航速度。
        if not self.pose_received:
            return

        if self.target_dist < self.arrival_tolerance:
            self.navigation_arrived = True
            self._activate_target_stop_locked()
            self.get_logger().warn(
                f'🏁 到达第一目标点({self.target_map_x:.1f},'
                f'{self.target_map_y:.1f})，距离={self.target_dist:.3f}m；'
                f'已建立停车锁存'
            )
            return

        w_raw = self.target_w + self.boundary_w + self.current_cone_w
        w_total = self.clamp(w_raw, -self.max_w, self.max_w)
        limited = abs(w_total - w_raw) > 1e-9

        active_items = []
        if self.boundary_active:
            active_items.append(self.boundary_mode)
        if self.cone_control_active:
            active_items.append('锥桶PD')
        mode = '+'.join(active_items) if active_items else '目标PD'

        limit_text = '，总角速度已限幅' if limited else ''
        log_tag = (
            f'🧮 纯PD融合[{source}/{mode}]：'
            f'W目标={self.target_w:+.3f}，'
            f'W边界={self.boundary_w:+.3f}，'
            f'W锥桶={self.current_cone_w:+.3f}，'
            f'W原始={w_raw:+.3f}，W输出={w_total:+.3f}'
            f'{limit_text}'
        )

        self.execute_drive(self.target_v, w_total, log_tag)

    # ------------------------------------------------------------------
    # 最终速度发布
    # ------------------------------------------------------------------
    def execute_drive(self, v, w, log_tag):
        requested_nonzero = abs(v) > 1e-6 or abs(w) > 1e-6

        with self.motion_lock:
            if self.handoff_complete or self.is_finished:
                return
            if self.stop_latched and requested_nonzero:
                return

            v = self.clamp(v, 0.0, self.max_v)
            w = self.clamp(w, -self.max_w, self.max_w)

            self.last_cmd_v = float(v)
            self.last_cmd_w = float(w)

            twist = Twist()
            twist.linear.x = float(v)
            twist.angular.z = float(w)
            self.cmd_pub.publish(twist)

        if v == 0.0 and w == 0.0:
            self.get_logger().info(
                f'{log_tag} | V:0.00 W:0.00',
                throttle_duration_sec=1.0,
            )
        elif 'PD融合' in log_tag:
            self.get_logger().info(
                f'{log_tag} | V:{v:.2f} W:{w:.2f}',
                throttle_duration_sec=0.5,
            )


# ----------------------------------------------------------------------
# 主函数
# ----------------------------------------------------------------------
def main():
    rclpy.init()
    node = Task1NavAvoidNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        if node.handoff_complete:
            node.get_logger().warn(
                '巡航节点已交接给通道节点，退出时不再发布停车命令，'
                '避免抢占通道节点 /cmd_vel'
            )
        else:
            node.get_logger().warn('🛑 节点退出，发布停车命令')
            for _ in range(5):
                node.cmd_pub.publish(Twist())
                time.sleep(0.02)
            os.system(
                "ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist '{}' "
                '> /dev/null 2>&1'
            )
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
