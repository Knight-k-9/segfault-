#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
任务1巡航 + 锥桶避障 + 边界保护（PD 融合版）

角速度约定：
    w > 0：左转
    w < 0：右转

正常控制时：
    w_total = w_boundary + w_cone + w_target

其中：
1. 边界：
    e_upper = y_actual - y_upper_boundary，e_upper >= 0 时生效
    e_lower = y_actual - y_lower_boundary，e_lower <= 0 时生效
    w_boundary = -(Kp * e + Kd * de/dt)

2. 锥桶：
    d1 = (bottom_y - trigger_y) * cone_near_k
    e_x = cone_center_x - image_center_x
    d2 = cone_kp * e_x + cone_kd * de_x/dt
    w_cone = d1 * d2

3. 目标点：
    e_yaw = normalize(target_yaw - current_yaw)
    w_target = target_kp * e_yaw + target_kd * de_yaw/dt

停车锁存、二维码结果交接、曲线边缘过滤、双阈值预判、上下区域
避障方向保护，以及“y + 航向角”紧急右转保护均保留。
"""

import math
import os
import threading
import time

import numpy as np
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
        # [配置区 1]：目标点及目标角度 PD
        # ================================================================
        self.target_map_x = 4.7
        self.target_map_y = 2.0
        self.start_offset_x = 0.55
        self.start_offset_y = 0.22

        # 线速度仍按原逻辑：v = kp_linear * 目标距离。
        self.kp_linear = 1.0

        # 目标角度 PD：w_target = Kp*角度误差 + Kd*角度误差微分。
        self.target_kp = 3.0
        self.target_kd = 0.20
        self.target_d_filter_alpha = 0.35
        self.target_d_limit = 6.0          # rad/s，限制角误差微分尖峰
        self.target_w_limit = 2.0

        self.max_v = 1.0
        self.max_w = 2.0
        self.arrival_tolerance = 0.5

        # ================================================================
        # [配置区 1.1]：二维码结果直接交接与目标点停车
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
        # [配置区 2]：锥桶 PD 融合参数
        # ================================================================
        self.conf_thresh = 0.6

        # 当前转向与避障方向一致/直行时，用 300；方向相反时，用 280。
        self.dist_thresh_y = 300
        self.early_dist_thresh_y = 280

        # d1 = (bottom_y - trigger_threshold) * cone_near_k
        self.cone_near_k = 0.0125
        self.cone_near_max = 1.50

        # d2 = cone_kp * e_x + cone_kd * de_x/dt
        # e_x 单位为像素，所以 Kp/Kd 数值会比角度 PD 小很多。
        self.cone_kp = 0.0150
        self.cone_kd = 0.0010
        self.cone_d_filter_alpha = 0.35
        self.cone_d_limit = 800.0          # pixel/s
        self.cone_w_limit = 1.80

        # 锥桶接近图像正中心时，避免 e_x=0 导致完全不转弯。
        self.cone_center_deadband = 20.0
        self.cone_center_min_offset = 35.0

        # 上侧区域只允许锥桶修正向右，下侧区域只允许锥桶修正向左。
        self.upper_avoid_y_threshold = 1.5
        self.lower_avoid_y_threshold = 0.5
        self.lower_avoid_x_min = 1.0

        # 判断当前角速度方向的死区。
        self.turn_w_deadband = 0.02

        # 保留原“中央近距离增强”。
        self.center_close_bottom_y_threshold = 320
        self.center_close_x_half_range = 40
        self.center_close_w_bonus = 0.2

        self.image_width = 640
        self.image_height = 480

        # ================================================================
        # [配置区 2.1]：曲线边缘忽略区域
        # ================================================================
        self.left_edge_points = [
            (145, 300),
            (136, 305),
            (127, 310),
            (114, 315),
            (103, 320),
        ]
        self._build_edge_boundaries()

        # ================================================================
        # [配置区 3]：地图上下边界 PD
        # ================================================================
        # 安全区域为 y_lower_boundary < y < y_upper_boundary。
        self.y_lower_boundary = 0.20
        self.y_upper_boundary = 1.80

        # w_boundary = -(Kp*e + Kd*de/dt)
        # 上边界 e>0，因此得到负角速度（右转）；
        # 下边界 e<0，因此得到正角速度（左转）。
        self.boundary_kp = 6.0
        self.boundary_kd = 0.80
        self.boundary_d_filter_alpha = 0.35
        self.boundary_d_limit = 2.0        # m/s
        self.boundary_w_limit = 1.80

        # ================================================================
        # [配置区 3.1]：原上侧“航向角+y”紧急保护
        # ================================================================
        # 该逻辑作为安全兜底，优先级高于普通三项融合。
        self.upper_heading_y_threshold = 1.60
        self.upper_heading_yaw_threshold = math.radians(30.0)
        self.upper_heading_force_v = 0.60
        self.upper_heading_force_w = 2.00

        # ================================================================
        # [状态变量]
        # ================================================================
        self.cur_pose = [0.0, 0.0, 0.0]
        self.pose_received = False
        self.navigation_arrived = False
        self.is_finished = False

        self.last_cmd_v = 0.0
        self.last_cmd_w = 0.0

        # 目标点 PD 缓存，只在收到新位姿时更新微分。
        self.target_dist = float('inf')
        self.target_angle_error = 0.0
        self.target_angle_d = 0.0
        self.target_w = 0.0
        self.target_v = 0.0
        self.target_prev_error = None
        self.target_prev_time = None
        self.target_d_filtered = 0.0

        # 边界 PD 缓存，只在收到新位姿时更新微分。
        self.boundary_active = False
        self.boundary_mode = '安全区'
        self.boundary_error = 0.0
        self.boundary_error_d = 0.0
        self.boundary_w = 0.0
        self.boundary_prev_error = None
        self.boundary_prev_time = None
        self.boundary_d_filtered = 0.0

        # 锥桶 PD 缓存，只在收到新检测帧时更新微分。
        # 当前结果会一直保持到下一条检测消息。
        self.cone_control_active = False
        self.current_cone_w = 0.0
        self.cone_prev_error = None
        self.cone_prev_time = None
        self.cone_d_filtered = 0.0
        self.cone_direction_sign = 0
        self.cone_prev_direction_sign = 0
        self.last_cone_center_x = 0.0
        self.last_cone_bottom_y = 0.0
        self.last_cone_trigger_threshold = self.dist_thresh_y

        self.upper_heading_force_active = False

        # 二维码交接状态。
        self.qr_result_received = False
        self.handoff_complete = False
        self.qr_result = ''

        # 发布和状态更新共用可重入锁。
        self.motion_lock = threading.RLock()
        self.stop_latched = False
        self.fast_stop_until = 0.0

        self.handoff_publish_count = 0
        self.channel_ack_received = False
        self.channel_ack_data = ''

        # ================================================================
        # [回调组与通信接口]
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

        self.qr_result_sub_relative = None
        relative_qr_result_topic = self.resolve_topic_name('qr_direction_result')
        if relative_qr_result_topic != self.qr_result_topic:
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

        self.timer = self.create_timer(
            0.025,
            self.control_loop,
            callback_group=self.control_callback_group,
        )

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
            '🚀 PD融合巡航节点启动：w>0左转，w<0右转；'
            'W总=W边界+W锥桶+W目标；目标点=(5.0,2.0)'
        )
        self.get_logger().info(
            f'参数：边界PD=({self.boundary_kp:.3f},{self.boundary_kd:.3f})，'
            f'锥桶PD=({self.cone_kp:.4f},{self.cone_kd:.4f})，'
            f'目标PD=({self.target_kp:.3f},{self.target_kd:.3f})，'
            f'Wmax={self.max_w:.2f}'
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

    def angular_direction_sign(self, w):
        """+1 左转，-1 右转，0 近似直行。"""
        if w > self.turn_w_deadband:
            return 1
        if w < -self.turn_w_deadband:
            return -1
        return 0

    @staticmethod
    def direction_name(direction_sign):
        if direction_sign > 0:
            return '左转'
        if direction_sign < 0:
            return '右转'
        return '直行'

    # ------------------------------------------------------------------
    # 边缘曲线构建
    # ------------------------------------------------------------------
    def _build_edge_boundaries(self):
        center_x = self.image_width / 2.0
        extended_left_points = []

        min_y_point = min(self.left_edge_points, key=lambda p: p[1])
        min_y = min_y_point[1]
        min_x = min_y_point[0]

        if min_y > 100:
            extended_left_points.append((min_x, 100))
            extended_left_points.append((min_x, min_y))

        sorted_points = sorted(self.left_edge_points, key=lambda p: p[1])
        extended_left_points.extend(sorted_points)

        max_y_point = max(sorted_points, key=lambda p: p[1])
        max_y = max_y_point[1]
        max_x = max_y_point[0]
        if max_y < self.image_height - 1:
            extended_left_points.append((max_x, self.image_height - 1))

        unique_by_y = {}
        for x, y in extended_left_points:
            unique_by_y[y] = x
        extended_left_points = sorted(
            [(x, y) for y, x in unique_by_y.items()],
            key=lambda p: p[1],
        )

        left_y_coords = np.array(
            [p[1] for p in extended_left_points], dtype=float
        )
        left_x_coords = np.array(
            [p[0] for p in extended_left_points], dtype=float
        )

        def left_boundary_func(y):
            if y < left_y_coords[0]:
                return float(left_x_coords[0])
            if y > left_y_coords[-1]:
                return float(left_x_coords[-1])
            return float(np.interp(y, left_y_coords, left_x_coords))

        right_y_coords = left_y_coords
        right_x_coords = 2.0 * center_x - left_x_coords

        def right_boundary_func(y):
            if y < right_y_coords[0]:
                return float(right_x_coords[0])
            if y > right_y_coords[-1]:
                return float(right_x_coords[-1])
            return float(np.interp(y, right_y_coords, right_x_coords))

        self.left_boundary_func = left_boundary_func
        self.right_boundary_func = right_boundary_func

        self.get_logger().info(
            f'✅ 曲线边缘已构建：左侧 {len(extended_left_points)} 点，'
            f'y范围[{left_y_coords[0]:.0f},{left_y_coords[-1]:.0f}]'
        )

    def is_in_edge_ignore_zone(self, center_x, bottom_y):
        left_x = self.left_boundary_func(bottom_y)
        right_x = self.right_boundary_func(bottom_y)
        return center_x < left_x or center_x > right_x

    # ------------------------------------------------------------------
    # 位姿回调及两组位姿相关 PD
    # ------------------------------------------------------------------
    def pose_cb(self, msg):
        now = time.monotonic()
        first_pose = False

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
        mx = self.cur_pose[0] + self.start_offset_x
        my = self.cur_pose[1] + self.start_offset_y
        m_yaw = self.cur_pose[2]

        dx = self.target_map_x - mx
        dy = self.target_map_y - my
        dist = math.hypot(dx, dy)
        target_yaw = math.atan2(dy, dx)
        angle_error = self.normalize_angle(target_yaw - m_yaw)

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
                a = self.target_d_filter_alpha
                self.target_d_filtered = (
                    a * raw_d + (1.0 - a) * self.target_d_filtered
                )
                angle_d = self.target_d_filtered

        self.target_prev_error = angle_error
        self.target_prev_time = now

        w_raw = self.target_kp * angle_error + self.target_kd * angle_d
        self.target_w = self.clamp(
            w_raw,
            -self.target_w_limit,
            self.target_w_limit,
        )
        self.target_v = self.clamp(
            dist * self.kp_linear,
            0.0,
            self.max_v,
        )
        self.target_dist = dist
        self.target_angle_error = angle_error
        self.target_angle_d = angle_d

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
        actual_y = self.cur_pose[1] + self.start_offset_y

        if actual_y >= self.y_upper_boundary:
            mode = '上边界'
            error = actual_y - self.y_upper_boundary
        elif actual_y <= self.y_lower_boundary:
            mode = '下边界'
            error = actual_y - self.y_lower_boundary
        else:
            self._reset_boundary_pd_locked()
            return

        error_d = 0.0
        same_zone = self.boundary_active and self.boundary_mode == mode
        if (
            same_zone
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
                a = self.boundary_d_filter_alpha
                self.boundary_d_filtered = (
                    a * raw_d + (1.0 - a) * self.boundary_d_filtered
                )
                error_d = self.boundary_d_filtered
        else:
            # 刚进入边界区时不使用微分，避免 derivative kick。
            self.boundary_d_filtered = 0.0

        self.boundary_active = True
        self.boundary_mode = mode
        self.boundary_prev_error = error
        self.boundary_prev_time = now
        self.boundary_error = error
        self.boundary_error_d = error_d

        # 负号用于匹配：w>0左转、w<0右转。
        w_raw = -(
            self.boundary_kp * error
            + self.boundary_kd * error_d
        )

        # D项只能调整力度，不能让边界修正反向。
        if mode == '上边界':
            w_raw = min(w_raw, 0.0)
        else:
            w_raw = max(w_raw, 0.0)

        self.boundary_w = self.clamp(
            w_raw,
            -self.boundary_w_limit,
            self.boundary_w_limit,
        )

    # ------------------------------------------------------------------
    # 停车及二维码交接
    # ------------------------------------------------------------------
    def _publish_zero_locked(self):
        zero = Twist()
        self.last_cmd_v = 0.0
        self.last_cmd_w = 0.0
        self.cmd_pub.publish(zero)

    def _clear_cone_control_locked(self, reset_direction=False):
        self.cone_control_active = False
        self.current_cone_w = 0.0
        self.cone_prev_error = None
        self.cone_prev_time = None
        self.cone_d_filtered = 0.0
        self.cone_prev_direction_sign = 0
        self.last_cone_center_x = 0.0
        self.last_cone_bottom_y = 0.0
        if reset_direction:
            self.cone_direction_sign = 0

    def _activate_target_stop_locked(self):
        self.stop_latched = True
        self.fast_stop_until = time.monotonic() + self.fast_stop_hold_sec
        self.fast_stop_timer.reset()

        self._clear_cone_control_locked(reset_direction=True)
        self._reset_boundary_pd_locked()
        self.upper_heading_force_active = False

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
            self._clear_cone_control_locked(reset_direction=True)
            self._reset_boundary_pd_locked()
            self.upper_heading_force_active = False
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
    # 上侧“航向角+y”紧急保护
    # ------------------------------------------------------------------
    def upper_heading_force_condition(self):
        if not self.pose_received:
            return False

        actual_y = self.cur_pose[1] + self.start_offset_y
        yaw = self.normalize_angle(self.cur_pose[2])
        return (
            actual_y > self.upper_heading_y_threshold
            and yaw >= self.upper_heading_yaw_threshold
        )

    def update_upper_heading_force_state(self):
        active_now = self.upper_heading_force_condition()

        if active_now and not self.upper_heading_force_active:
            self.upper_heading_force_active = True
            self._clear_cone_control_locked(reset_direction=False)

            actual_y = self.cur_pose[1] + self.start_offset_y
            yaw_deg = math.degrees(self.normalize_angle(self.cur_pose[2]))
            self.get_logger().warn(
                f'🧭 进入上侧航向保护：actual_y={actual_y:.3f}>'
                f'{self.upper_heading_y_threshold:.2f}，'
                f'yaw={yaw_deg:.1f}°>='
                f'{math.degrees(self.upper_heading_yaw_threshold):.1f}°，'
                f'强制右转',
                throttle_duration_sec=0.5,
            )

        elif not active_now and self.upper_heading_force_active:
            self.upper_heading_force_active = False
            actual_y = self.cur_pose[1] + self.start_offset_y
            yaw_deg = math.degrees(self.normalize_angle(self.cur_pose[2]))
            self.get_logger().info(
                f'✅ 退出上侧航向保护：actual_y={actual_y:.3f}，'
                f'yaw={yaw_deg:.1f}°',
                throttle_duration_sec=0.5,
            )

        return self.upper_heading_force_active

    def upper_heading_force_log(self, mode):
        actual_y = self.cur_pose[1] + self.start_offset_y
        yaw_deg = math.degrees(self.normalize_angle(self.cur_pose[2]))
        return (
            f'🧭 上侧航向强制右转[{mode}]：'
            f'actual_y={actual_y:.3f}, yaw={yaw_deg:.1f}°'
        )

    # ------------------------------------------------------------------
    # 锥桶候选框筛选
    # ------------------------------------------------------------------
    def collect_valid_obstacle_candidates(self, msg):
        candidates = []

        for target in msg.targets:
            for roi in target.rois:
                if roi.confidence <= self.conf_thresh:
                    continue

                rect = roi.rect
                center_x = rect.x_offset + rect.width / 2.0
                bottom_y = rect.y_offset + rect.height

                # 280以下连提前判定区域都没进入。
                if bottom_y <= self.early_dist_thresh_y:
                    continue

                if self.is_in_edge_ignore_zone(center_x, bottom_y):
                    left_x = self.left_boundary_func(bottom_y)
                    right_x = self.right_boundary_func(bottom_y)
                    self.get_logger().info(
                        f'🟡 曲线边缘忽略：center_x={center_x:.0f}, '
                        f'bottom_y={bottom_y:.0f} | '
                        f'有效范围[{left_x:.0f},{right_x:.0f}]',
                        throttle_duration_sec=0.5,
                    )
                    continue

                candidates.append((bottom_y, center_x, roi))

        return candidates

    def _choose_center_cone_direction_locked(self, actual_x, actual_y):
        """锥桶位于图像中心死区时选择稳定避障方向。"""
        if self.upper_heading_force_active:
            return -1
        if actual_y > self.upper_avoid_y_threshold:
            return -1
        if (
            actual_y < self.lower_avoid_y_threshold
            and actual_x > self.lower_avoid_x_min
        ):
            return 1
        if self.cone_direction_sign != 0:
            return self.cone_direction_sign

        current_sign = self.angular_direction_sign(self.last_cmd_w)
        if current_sign != 0:
            return current_sign

        # 没有历史方向时默认左转，和原代码 center_x==320 时的分支一致。
        return 1

    def _update_cone_derivative_locked(self, error_x, direction_sign, now):
        error_d = 0.0

        same_direction = (
            self.cone_prev_direction_sign == direction_sign
            and self.cone_prev_direction_sign != 0
        )
        if (
            same_direction
            and self.cone_prev_error is not None
            and self.cone_prev_time is not None
        ):
            dt = now - self.cone_prev_time
            if dt >= 1e-3:
                raw_d = (error_x - self.cone_prev_error) / dt
                raw_d = self.clamp(
                    raw_d,
                    -self.cone_d_limit,
                    self.cone_d_limit,
                )
                a = self.cone_d_filter_alpha
                self.cone_d_filtered = (
                    a * raw_d + (1.0 - a) * self.cone_d_filtered
                )
                error_d = self.cone_d_filtered
        else:
            # 避障方向变化时，清零D项，防止符号翻转产生尖峰。
            self.cone_d_filtered = 0.0

        self.cone_prev_error = error_x
        self.cone_prev_time = now
        self.cone_prev_direction_sign = direction_sign
        return error_d

    def update_cone_control_locked(self, msg):
        """
        根据最新检测帧计算并保存 w_cone。

        没有新检测帧时，40Hz控制循环继续使用当前保存值；
        下一帧无危险时立即清零。
        """
        candidates = self.collect_valid_obstacle_candidates(msg)
        if not candidates:
            was_active = self.cone_control_active
            old_w = self.current_cone_w
            self._clear_cone_control_locked(reset_direction=False)
            if was_active:
                self.get_logger().info(
                    f'✅ 当前检测帧无有效锥桶，解除锥桶PD：旧W={old_w:.2f}',
                    throttle_duration_sec=0.5,
                )
            return False

        bottom_y, center_x, roi = max(candidates, key=lambda item: item[0])
        image_center_x = self.image_width / 2.0
        raw_error_x = center_x - image_center_x

        actual_x = self.cur_pose[0] + self.start_offset_x
        actual_y = self.cur_pose[1] + self.start_offset_y

        # 先得到视觉所需避让方向：右侧锥桶 -> 左转(+)，左侧 -> 右转(-)。
        if raw_error_x > self.cone_center_deadband:
            desired_sign = 1
            error_x = raw_error_x
        elif raw_error_x < -self.cone_center_deadband:
            desired_sign = -1
            error_x = raw_error_x
        else:
            desired_sign = self._choose_center_cone_direction_locked(
                actual_x,
                actual_y,
            )
            error_x = desired_sign * self.cone_center_min_offset

        self.cone_direction_sign = desired_sign

        # 保留原双阈值逻辑。
        if self.upper_heading_force_active:
            current_sign = -1
            current_w_for_log = -abs(self.upper_heading_force_w)
        else:
            current_sign = self.angular_direction_sign(self.last_cmd_w)
            current_w_for_log = self.last_cmd_w

        directions_opposite = (
            current_sign != 0
            and desired_sign != 0
            and current_sign != desired_sign
        )

        if directions_opposite:
            trigger_threshold = self.early_dist_thresh_y
            trigger_mode = (
                f'方向相反，使用{self.early_dist_thresh_y}提前触发'
            )
        else:
            trigger_threshold = self.dist_thresh_y
            if current_sign == desired_sign and current_sign != 0:
                trigger_mode = (
                    f'方向一致，忽略{self.early_dist_thresh_y}，'
                    f'按{self.dist_thresh_y}'
                )
            else:
                trigger_mode = (
                    f'当前直行，忽略{self.early_dist_thresh_y}，'
                    f'按{self.dist_thresh_y}'
                )

        self.get_logger().info(
            f'👀 锥桶预判：当前={self.direction_name(current_sign)}'
            f'(W={current_w_for_log:.2f})，'
            f'需要={self.direction_name(desired_sign)}，'
            f'bottom_y={bottom_y:.0f}，阈值={trigger_threshold}，'
            f'{trigger_mode}',
            throttle_duration_sec=0.5,
        )

        if bottom_y <= trigger_threshold:
            was_active = self.cone_control_active
            old_w = self.current_cone_w
            self._clear_cone_control_locked(reset_direction=False)
            if was_active:
                self.get_logger().info(
                    f'✅ 锥桶尚未达到本帧触发阈值，解除锥桶PD：'
                    f'bottom_y={bottom_y:.0f}<={trigger_threshold}，'
                    f'旧W={old_w:.2f}',
                    throttle_duration_sec=0.5,
                )
            return False

        now = time.monotonic()
        error_x_d = self._update_cone_derivative_locked(
            error_x,
            desired_sign,
            now,
        )

        d1 = (bottom_y - trigger_threshold) * self.cone_near_k
        d1 = self.clamp(d1, 0.0, self.cone_near_max)

        d2_p = self.cone_kp * error_x
        d2_d = self.cone_kd * error_x_d
        d2 = d2_p + d2_d

        # D项可以增强或减弱，但不允许把避障方向翻转成朝向锥桶。
        if desired_sign * d2 < 0.0:
            d2 = desired_sign * abs(d2_p)

        cone_w = d1 * d2
        direction_protect_mode = ''

        # 保留上下区域避障方向保护，但强度仍来自锥桶PD。
        if self.upper_heading_force_active or actual_y > self.upper_avoid_y_threshold:
            if cone_w > 0.0:
                cone_w = -abs(cone_w)
                desired_sign = -1
                self.cone_direction_sign = -1
                direction_protect_mode = (
                    f' | 上侧区域y={actual_y:.2f}，锥桶修正改为右转'
                )
        elif (
            actual_y < self.lower_avoid_y_threshold
            and actual_x > self.lower_avoid_x_min
        ):
            if cone_w < 0.0:
                cone_w = abs(cone_w)
                desired_sign = 1
                self.cone_direction_sign = 1
                direction_protect_mode = (
                    f' | 下侧区域y={actual_y:.2f},x={actual_x:.2f}，'
                    f'锥桶修正改为左转'
                )

        cone_w = self.clamp(
            cone_w,
            -self.cone_w_limit,
            self.cone_w_limit,
        )

        # 保留原中央近距离 +0.2 增强，不改变最终方向。
        center_close_mode = ''
        center_offset_x = abs(center_x - image_center_x)
        if (
            abs(cone_w) > 1e-6
            and bottom_y > self.center_close_bottom_y_threshold
            and center_offset_x <= self.center_close_x_half_range
        ):
            old_w = cone_w
            enhanced_abs_w = min(
                abs(cone_w) + self.center_close_w_bonus,
                self.cone_w_limit,
            )
            cone_w = math.copysign(enhanced_abs_w, cone_w)
            center_close_mode = (
                f' | 中央近距离增强W:{old_w:.2f}->{cone_w:.2f}'
            )

        self.cone_control_active = abs(cone_w) > 1e-6
        self.current_cone_w = cone_w
        self.last_cone_center_x = center_x
        self.last_cone_bottom_y = bottom_y
        self.last_cone_trigger_threshold = trigger_threshold

        self.get_logger().info(
            f'🟠 锥桶PD：d1={d1:.3f}，'
            f'e_x={error_x:.1f}px，de_x={error_x_d:.1f}px/s，'
            f'd2=P({d2_p:.3f})+D({d2_d:.3f})={d2:.3f}，'
            f'W锥桶={cone_w:.3f}，conf={roi.confidence:.2f}'
            f'{direction_protect_mode}{center_close_mode}',
            throttle_duration_sec=0.5,
        )
        return self.cone_control_active

    # ------------------------------------------------------------------
    # 统一融合控制
    # ------------------------------------------------------------------
    def obs_cb(self, msg):
        with self.motion_lock:
            if self.is_finished or self.handoff_complete or self.stop_latched:
                return
            if self.navigation_arrived:
                return

            self.update_upper_heading_force_state()
            self.update_cone_control_locked(msg)
            self._publish_fused_control_locked('检测帧立即更新')

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
                '🏁 已到达第一目标点(5.0,2.0)，停车保持',
            )
            return

        # 没有位姿前不启动巡航，避免以默认(0,0,0)误控制。
        if not self.pose_received:
            return

        if self.target_dist < self.arrival_tolerance:
            self.navigation_arrived = True
            self._activate_target_stop_locked()
            self.get_logger().warn(
                f'🏁 到达第一目标点({self.target_map_x:.1f},'
                f'{self.target_map_y:.1f})，距离={self.target_dist:.3f}m；'
                f'已锁存停车并继续等待二维码解析结果'
            )
            return

        # 原“上侧航向角+y”保护保持最高优先级。
        if self.update_upper_heading_force_state():
            self.execute_drive(
                self.upper_heading_force_v,
                -abs(self.upper_heading_force_w),
                self.upper_heading_force_log(source),
            )
            return

        w_raw = self.target_w + self.boundary_w + self.current_cone_w
        w_total = self.clamp(w_raw, -self.max_w, self.max_w)
        limited = abs(w_total - w_raw) > 1e-9

        active_items = []
        if self.boundary_active:
            active_items.append(self.boundary_mode)
        if self.cone_control_active:
            active_items.append('锥桶')
        mode = '+'.join(active_items) if active_items else '目标循迹'

        limit_text = '，已触发W总限幅' if limited else ''
        log_tag = (
            f'🧮 PD融合[{source}/{mode}]：'
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
        elif 'PD融合' in log_tag or '航向强制右转' in log_tag:
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
            node.get_logger().warn('🛑 紧急强制停车')
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
