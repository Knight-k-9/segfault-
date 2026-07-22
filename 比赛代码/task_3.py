#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
返回 P 点 + 地图坐标锥桶避障 + 地图边界（纯 PD 融合版）

最终目标点：
    P = (start_offset_x, start_offset_y)
      = (0.55, 0.22)

角速度约定：
    w > 0：左转
    w < 0：右转

正常运动只保留三项控制：
    w_total = w_target + w_boundary + w_cone

1. 目标点角度 PD：

    e_yaw = normalize(target_yaw - current_yaw)

    w_target = (
        target_kp * e_yaw
        + target_kd * de_yaw/dt
    )

2. 地图边界 PD：

    上边界：
        e = y_actual - y_upper_boundary
        e >= 0 时生效

    下边界：
        e = y_actual - y_lower_boundary
        e <= 0 时生效

    w_boundary = -(
        boundary_kp * e
        + boundary_kd * de/dt
    )

3. 锥桶坐标 PD：

    订阅 /cone_coordinates（std_msgs/msg/String），
    消息为 JSON 数组：

        [{"x": 1.2345, "y": 0.6789}, ...]

    将地图中的锥桶坐标转换到小车坐标系：

        longitudinal：
            小车前方为正，后方为负

        lateral_left：
            小车左侧为正，右侧为负

        lateral_right = -lateral_left
            小车右侧为正，用于保持原图像 x 的符号

    触发条件：

        cone_min_longitudinal
            < longitudinal
            < cone_trigger_longitudinal

        abs(lateral_right)
            <= cone_lateral_limit

    d1 = (
        cone_trigger_longitudinal
        - longitudinal
    ) * cone_near_k

    d2 = (
        cone_kp * lateral_right
        + cone_kd * d(lateral_right)/dt
    )

    w_cone = d1 * d2

    因此：

        锥桶在小车右侧：
            lateral_right > 0
            w_cone > 0
            向左避障

        锥桶在小车左侧：
            lateral_right < 0
            w_cone < 0
            向右避障

当前版本：

- 使用 /cone_coordinates 中的地图坐标进行锥桶避障；
- 使用小车到锥桶的纵向距离代替原图像 y；
- 使用小车到锥桶的横向距离代替原图像 x；
- 横向距离绝对值大于 cone_lateral_limit 时忽略该锥桶；
- 选择小车前方、横向有效范围内纵向距离最近的锥桶；
- 每次位姿更新时重新计算锥桶相对位置；
- 最终返回地图起始点 P=(0.55, 0.22)；
- 不再订阅或处理二维码；
- 不再发布 /qr_success；
- 不再等待通道导航 ACK；
- 到达 P 点后建立停车锁存并持续发布零速度。

保留：
    目标点停车锁存、快速停车覆盖、QoS、
    线程锁、控制融合及输出限幅。
"""

import json
import math
import os
import threading
import time

import rclpy
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
        # [配置区 1]：P 点、坐标偏移和目标角度 PD
        # ================================================================

        # odom_pose=(0,0) 对应地图坐标 P=(0.55,0.22)。
        self.start_offset_x = 0.55
        self.start_offset_y = 0.22

        # 最终返回起始地图点 P。
        self.target_map_x = self.start_offset_x
        self.target_map_y = self.start_offset_y

        # 线速度：
        # v = kp_linear * 目标距离
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

        # 到达 P 点的距离阈值。
        # 按照原代码保持为 0.5 m。
        self.arrival_tolerance = 0.5

        # ================================================================
        # [配置区 2]：基于地图坐标的锥桶 PD
        # ================================================================

        self.cone_coordinates_topic = '/cone_coordinates'

        # 只处理小车前方该距离以内的锥桶。
        #
        # /cone_coordinates 的建图节点当前也会过滤距离
        # 大于约 1.6 m 的新锥桶，因此这里同样设为 1.60 m。
        self.cone_trigger_longitudinal = 1.60

        # 小车后轴/位姿点前方过近或已经越过的锥桶，
        # 不再参与控制。
        self.cone_min_longitudinal = 0.03

        # 锥桶与小车中心线的横向距离绝对值大于该值时，
        # 不触发锥桶避障。
        self.cone_lateral_limit = 0.40

        # 距离接近系数：
        #
        # d1 = (
        #     cone_trigger_longitudinal
        #     - longitudinal
        # ) * cone_near_k
        self.cone_near_k = 1.00
        self.cone_near_max = 1.50

        # 横向距离 PD。
        #
        # lateral_right > 0：
        #     锥桶位于小车右侧
        #
        # 生成正角速度，使小车向左避障。
        self.cone_kp = 8.00
        self.cone_kd = 0.15
        self.cone_d_filter_alpha = 0.35
        self.cone_d_limit = 3.0
        self.cone_w_limit = 1.80

        # ================================================================
        # [配置区 3]：地图上下边界 PD
        # ================================================================

        self.y_lower_boundary = 0.20
        self.y_upper_boundary = 1.60

        self.boundary_kp = 76.0
        self.boundary_kd = 2.0
        self.boundary_d_filter_alpha = 0.35
        self.boundary_d_limit = 2.0
        self.boundary_w_limit = 18.0

        # ================================================================
        # [配置区 4]：目标点停车
        # ================================================================

        # 到达目标点时立即连续发送若干次零速度。
        self.fast_stop_burst_count = 4

        # 到达目标点后，以 200 Hz 快速覆盖零速度的持续时间。
        self.fast_stop_hold_sec = 0.35

        # 200 Hz 快速停车定时器。
        self.fast_stop_timer_period = 0.005

        # ================================================================
        # [状态变量]
        # ================================================================

        self.cur_pose = [0.0, 0.0, 0.0]
        self.pose_received = False

        # 到达 P 点后设置为 True。
        self.navigation_arrived = False

        self.last_cmd_v = 0.0
        self.last_cmd_w = 0.0

        # ---------------------------------------------------------------
        # 目标点 PD 状态
        # ---------------------------------------------------------------

        # 目标点 PD 只在收到新位姿时更新。
        self.target_dist = float('inf')
        self.target_v = 0.0
        self.target_w = 0.0

        self.target_angle_error = 0.0
        self.target_angle_d = 0.0

        self.target_prev_error = None
        self.target_prev_time = None
        self.target_d_filtered = 0.0

        # ---------------------------------------------------------------
        # 边界 PD 状态
        # ---------------------------------------------------------------

        # 边界 PD 只在收到新位姿时更新。
        self.boundary_active = False
        self.boundary_mode = '安全区'

        self.boundary_error = 0.0
        self.boundary_error_d = 0.0
        self.boundary_w = 0.0

        self.boundary_prev_error = None
        self.boundary_prev_time = None
        self.boundary_d_filtered = 0.0

        # ---------------------------------------------------------------
        # 锥桶坐标及 PD 状态
        # ---------------------------------------------------------------

        # 地图坐标列表由 /cone_coordinates 更新。
        #
        # 锥桶相对于车辆的位置在每次位姿更新时重新计算。
        self.latest_cone_coordinates = []
        self.cone_coordinates_received = False

        self.cone_control_active = False
        self.current_cone_w = 0.0

        self.cone_prev_error = None
        self.cone_prev_time = None
        self.cone_d_filtered = 0.0

        self.last_cone_map_x = 0.0
        self.last_cone_map_y = 0.0
        self.last_cone_longitudinal = 0.0
        self.last_cone_lateral_right = 0.0

        # ---------------------------------------------------------------
        # 停车和运动发布状态
        # ---------------------------------------------------------------

        self.motion_lock = threading.RLock()

        # 一旦到达 P 点，该锁存永远保持 True。
        # 建立锁存后不再允许发布非零速度。
        self.stop_latched = False

        # 快速零速度覆盖结束时间。
        self.fast_stop_until = 0.0

        # ================================================================
        # [回调组与 ROS 通信]
        # ================================================================

        self.stop_callback_group = MutuallyExclusiveCallbackGroup()
        self.control_callback_group = MutuallyExclusiveCallbackGroup()
        self.cone_callback_group = MutuallyExclusiveCallbackGroup()
        self.pose_callback_group = MutuallyExclusiveCallbackGroup()

        # ---------------------------------------------------------------
        # 位姿订阅
        # ---------------------------------------------------------------

        self.pose_sub = self.create_subscription(
            Pose2D,
            'odom_pose',
            self.pose_cb,
            10,
            callback_group=self.pose_callback_group,
        )

        # ---------------------------------------------------------------
        # 锥桶地图坐标订阅
        # ---------------------------------------------------------------

        cone_qos = QoSProfile(depth=1)
        cone_qos.history = HistoryPolicy.KEEP_LAST
        cone_qos.reliability = ReliabilityPolicy.BEST_EFFORT
        cone_qos.durability = DurabilityPolicy.VOLATILE

        self.cone_coordinates_sub = self.create_subscription(
            String,
            self.cone_coordinates_topic,
            self.cone_coordinates_cb,
            cone_qos,
            callback_group=self.cone_callback_group,
        )

        # ---------------------------------------------------------------
        # 速度发布
        # ---------------------------------------------------------------

        cmd_qos = QoSProfile(depth=1)
        cmd_qos.history = HistoryPolicy.KEEP_LAST
        cmd_qos.reliability = ReliabilityPolicy.RELIABLE
        cmd_qos.durability = DurabilityPolicy.VOLATILE

        self.cmd_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            cmd_qos,
        )

        # ---------------------------------------------------------------
        # 控制定时器
        # ---------------------------------------------------------------

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

        # ================================================================
        # 启动日志
        # ================================================================

        self.get_logger().info(
            '🚀 返回P点纯PD融合巡航节点启动：'
            'W总=W目标+W边界+W锥桶；'
            'w>0左转，w<0右转'
        )

        self.get_logger().info(
            f'最终目标点P='
            f'({self.target_map_x:.2f},{self.target_map_y:.2f})，'
            f'到达阈值={self.arrival_tolerance:.2f}m'
        )

        self.get_logger().info(
            f'目标PD=({self.target_kp:.3f},{self.target_kd:.3f})，'
            f'边界PD=({self.boundary_kp:.3f},{self.boundary_kd:.3f})，'
            f'锥桶PD=({self.cone_kp:.4f},{self.cone_kd:.4f})，'
            f'锥桶纵向阈值={self.cone_trigger_longitudinal:.2f}m，'
            f'横向有效范围=±{self.cone_lateral_limit:.2f}m，'
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
        return (
            (angle + math.pi)
            % (2.0 * math.pi)
            - math.pi
        )

    # ------------------------------------------------------------------
    # 位姿更新：目标点 PD + 边界 PD + 锥桶 PD
    # ------------------------------------------------------------------

    def pose_cb(self, msg):
        now = time.monotonic()

        with self.motion_lock:
            first_pose = not self.pose_received

            self.cur_pose = [
                float(msg.x),
                float(msg.y),
                float(msg.theta),
            ]
            self.pose_received = True

            # 停车锁存建立后不再更新控制器输出，
            # 但仍然保存最新位姿用于日志和状态观察。
            if not self.stop_latched:
                self._update_target_pd_from_pose_locked(now)
                self._update_boundary_pd_from_pose_locked(now)
                self._update_cone_control_from_coordinates_locked(now)

            actual_x = (
                self.cur_pose[0]
                + self.start_offset_x
            )
            actual_y = (
                self.cur_pose[1]
                + self.start_offset_y
            )

        if first_pose:
            self.get_logger().info(
                f'📍 位姿已连接，'
                f'actual=({actual_x:.3f},{actual_y:.3f})'
            )

    def _update_target_pd_from_pose_locked(self, now):
        actual_x = (
            self.cur_pose[0]
            + self.start_offset_x
        )
        actual_y = (
            self.cur_pose[1]
            + self.start_offset_y
        )
        actual_yaw = self.cur_pose[2]

        dx = self.target_map_x - actual_x
        dy = self.target_map_y - actual_y

        dist = math.hypot(dx, dy)

        target_yaw = math.atan2(dy, dx)

        angle_error = self.normalize_angle(
            target_yaw - actual_yaw
        )

        angle_d = 0.0

        if (
            self.target_prev_error is not None
            and self.target_prev_time is not None
        ):
            dt = now - self.target_prev_time

            if dt >= 1e-3:
                delta_error = self.normalize_angle(
                    angle_error
                    - self.target_prev_error
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
                    + (1.0 - alpha)
                    * self.target_d_filtered
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

    # ------------------------------------------------------------------
    # 地图边界 PD
    # ------------------------------------------------------------------

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
        actual_y = (
            self.cur_pose[1]
            + self.start_offset_y
        )

        if actual_y >= self.y_upper_boundary:
            mode = '上边界PD'
            error = (
                actual_y
                - self.y_upper_boundary
            )

        elif actual_y <= self.y_lower_boundary:
            mode = '下边界PD'
            error = (
                actual_y
                - self.y_lower_boundary
            )

        else:
            self._reset_boundary_pd_locked()
            return

        error_d = 0.0

        same_boundary = (
            self.boundary_active
            and self.boundary_mode == mode
        )

        if (
            same_boundary
            and self.boundary_prev_error is not None
            and self.boundary_prev_time is not None
        ):
            dt = now - self.boundary_prev_time

            if dt >= 1e-3:
                raw_d = (
                    error
                    - self.boundary_prev_error
                ) / dt

                raw_d = self.clamp(
                    raw_d,
                    -self.boundary_d_limit,
                    self.boundary_d_limit,
                )

                alpha = self.boundary_d_filter_alpha

                self.boundary_d_filtered = (
                    alpha * raw_d
                    + (1.0 - alpha)
                    * self.boundary_d_filtered
                )

                error_d = self.boundary_d_filtered

        else:
            # 刚进入边界区时先不使用 D，
            # 避免首帧微分尖峰。
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
    # 地图坐标锥桶避障
    # ------------------------------------------------------------------

    def _clear_cone_control_locked(self):
        self.cone_control_active = False
        self.current_cone_w = 0.0

        self.cone_prev_error = None
        self.cone_prev_time = None
        self.cone_d_filtered = 0.0

        self.last_cone_map_x = 0.0
        self.last_cone_map_y = 0.0
        self.last_cone_longitudinal = 0.0
        self.last_cone_lateral_right = 0.0

    def _parse_cone_coordinates(self, raw_text):
        """
        解析 /cone_coordinates 的 JSON 数组。

        返回：
            [(x, y), ...]
        """
        data = json.loads(raw_text)

        if not isinstance(data, list):
            raise ValueError(
                'JSON 顶层必须是数组'
            )

        coordinates = []

        for index, item in enumerate(data):
            if not isinstance(item, dict):
                self.get_logger().warn(
                    f'忽略第 {index} 个锥桶：'
                    f'元素不是对象',
                    throttle_duration_sec=1.0,
                )
                continue

            if 'x' not in item or 'y' not in item:
                self.get_logger().warn(
                    f'忽略第 {index} 个锥桶：'
                    f'缺少 x 或 y',
                    throttle_duration_sec=1.0,
                )
                continue

            try:
                cone_x = float(item['x'])
                cone_y = float(item['y'])

            except (TypeError, ValueError):
                self.get_logger().warn(
                    f'忽略第 {index} 个锥桶：'
                    f'x/y 不是有效数字',
                    throttle_duration_sec=1.0,
                )
                continue

            if (
                not math.isfinite(cone_x)
                or not math.isfinite(cone_y)
            ):
                self.get_logger().warn(
                    f'忽略第 {index} 个锥桶：'
                    f'x/y 非有限值',
                    throttle_duration_sec=1.0,
                )
                continue

            coordinates.append(
                (cone_x, cone_y)
            )

        return coordinates

    def cone_coordinates_cb(self, msg):
        raw_text = str(msg.data).strip()

        if not raw_text:
            coordinates = []

        else:
            try:
                coordinates = (
                    self._parse_cone_coordinates(
                        raw_text
                    )
                )

            except (
                json.JSONDecodeError,
                ValueError,
            ) as exc:
                self.get_logger().error(
                    f'❌ /cone_coordinates JSON 解析失败：'
                    f'{exc}；'
                    f'原始内容={raw_text[:200]!r}',
                    throttle_duration_sec=1.0,
                )
                return

        now = time.monotonic()

        with self.motion_lock:
            # 即使已经停车，也保存最新锥桶列表，
            # 但不再产生避障控制。
            self.latest_cone_coordinates = coordinates
            self.cone_coordinates_received = True

            if (
                self.stop_latched
                or self.navigation_arrived
            ):
                self._clear_cone_control_locked()
                return

            self._update_cone_control_from_coordinates_locked(
                now
            )

            self._publish_fused_control_locked(
                '锥桶坐标更新'
            )

    def _map_cone_to_vehicle_frame_locked(
        self,
        cone_x,
        cone_y,
    ):
        """
        将地图坐标中的锥桶转换到小车坐标系。

        返回：
            longitudinal：
                前方为正，单位 m

            lateral_right：
                右侧为正，单位 m
        """
        actual_x = (
            self.cur_pose[0]
            + self.start_offset_x
        )
        actual_y = (
            self.cur_pose[1]
            + self.start_offset_y
        )
        yaw = self.cur_pose[2]

        dx = cone_x - actual_x
        dy = cone_y - actual_y

        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        # 标准车体坐标：
        #
        # 前方为 +x
        # 左侧为 +y
        longitudinal = (
            cos_yaw * dx
            + sin_yaw * dy
        )

        lateral_left = (
            -sin_yaw * dx
            + cos_yaw * dy
        )

        # 为保持原图像 x 的符号：
        # 画面/车体右侧为正。
        lateral_right = -lateral_left

        return longitudinal, lateral_right

    def collect_valid_cone_candidates_locked(self):
        """
        筛选锥桶候选并转换到车体坐标。

        条件：
        1. 已收到位姿；
        2. 锥桶位于小车前方；
        3. 纵向距离小于触发距离；
        4. 横向距离绝对值不大于 cone_lateral_limit。

        返回元素：
            (
                longitudinal,
                lateral_right,
                cone_map_x,
                cone_map_y,
            )
        """
        if not self.pose_received:
            return []

        candidates = []

        for cone_x, cone_y in self.latest_cone_coordinates:
            (
                longitudinal,
                lateral_right,
            ) = self._map_cone_to_vehicle_frame_locked(
                cone_x,
                cone_y,
            )

            if (
                longitudinal
                <= self.cone_min_longitudinal
            ):
                continue

            if (
                longitudinal
                >= self.cone_trigger_longitudinal
            ):
                continue

            if (
                abs(lateral_right)
                > self.cone_lateral_limit
            ):
                self.get_logger().info(
                    f'🟡 横向边缘忽略：'
                    f'锥桶地图=({cone_x:.3f},{cone_y:.3f})，'
                    f'纵向={longitudinal:.3f}m，'
                    f'横向={lateral_right:+.3f}m，'
                    f'允许范围='
                    f'±{self.cone_lateral_limit:.3f}m',
                    throttle_duration_sec=0.5,
                )
                continue

            candidates.append(
                (
                    longitudinal,
                    lateral_right,
                    cone_x,
                    cone_y,
                )
            )

        return candidates

    def _update_cone_derivative_locked(
        self,
        error_lateral,
        now,
    ):
        error_d = 0.0

        if (
            self.cone_prev_error is not None
            and self.cone_prev_time is not None
        ):
            dt = now - self.cone_prev_time

            if dt >= 1e-3:
                raw_d = (
                    error_lateral
                    - self.cone_prev_error
                ) / dt

                raw_d = self.clamp(
                    raw_d,
                    -self.cone_d_limit,
                    self.cone_d_limit,
                )

                alpha = self.cone_d_filter_alpha

                self.cone_d_filtered = (
                    alpha * raw_d
                    + (1.0 - alpha)
                    * self.cone_d_filtered
                )

                error_d = self.cone_d_filtered

        self.cone_prev_error = error_lateral
        self.cone_prev_time = now

        return error_d

    def _update_cone_control_from_coordinates_locked(
        self,
        now,
    ):
        if (
            not self.cone_coordinates_received
            or not self.pose_received
        ):
            self._clear_cone_control_locked()
            return False

        candidates = (
            self.collect_valid_cone_candidates_locked()
        )

        if not candidates:
            was_active = self.cone_control_active
            old_w = self.current_cone_w

            self._clear_cone_control_locked()

            if was_active:
                self.get_logger().info(
                    f'✅ 当前无有效锥桶，'
                    f'锥桶PD清零：旧W={old_w:+.3f}',
                    throttle_duration_sec=0.5,
                )

            return False

        # 选择小车前方纵向距离最近的锥桶。
        (
            longitudinal,
            lateral_right,
            cone_x,
            cone_y,
        ) = min(
            candidates,
            key=lambda item: item[0],
        )

        lateral_d = (
            self._update_cone_derivative_locked(
                lateral_right,
                now,
            )
        )

        d1 = (
            self.cone_trigger_longitudinal
            - longitudinal
        ) * self.cone_near_k

        d1 = self.clamp(
            d1,
            0.0,
            self.cone_near_max,
        )

        d2_p = (
            self.cone_kp
            * lateral_right
        )

        d2_d = (
            self.cone_kd
            * lateral_d
        )

        d2 = d2_p + d2_d

        cone_w_raw = d1 * d2

        cone_w = self.clamp(
            cone_w_raw,
            -self.cone_w_limit,
            self.cone_w_limit,
        )

        self.cone_control_active = True
        self.current_cone_w = cone_w

        self.last_cone_map_x = cone_x
        self.last_cone_map_y = cone_y
        self.last_cone_longitudinal = longitudinal
        self.last_cone_lateral_right = lateral_right

        side_text = (
            '右侧'
            if lateral_right > 0.0
            else '左侧'
        )

        if abs(lateral_right) < 1e-4:
            side_text = '正前方'

        self.get_logger().info(
            f'🟠 锥桶坐标PD：'
            f'地图=({cone_x:.3f},{cone_y:.3f})，'
            f'车体纵向={longitudinal:.3f}m，'
            f'车体横向={lateral_right:+.3f}m'
            f'({side_text})，'
            f'd1={d1:.3f}，'
            f'd横向={lateral_d:+.3f}m/s，'
            f'd2=P({d2_p:+.3f})'
            f'+D({d2_d:+.3f})'
            f'={d2:+.3f}，'
            f'W锥桶={cone_w:+.3f}',
            throttle_duration_sec=0.5,
        )

        return True

    # ------------------------------------------------------------------
    # 目标点停车
    # ------------------------------------------------------------------

    def _publish_zero_locked(self):
        zero = Twist()

        self.last_cmd_v = 0.0
        self.last_cmd_w = 0.0

        self.cmd_pub.publish(zero)

    def _activate_target_stop_locked(self):
        """
        建立永久停车锁存。

        1. stop_latched 设置为 True；
        2. 清除锥桶和边界控制；
        3. 立即连续发布零速度；
        4. 开启 200 Hz 快速零速度覆盖；
        5. 后续 40 Hz 控制循环持续发布零速度。
        """
        self.stop_latched = True

        self.fast_stop_until = (
            time.monotonic()
            + self.fast_stop_hold_sec
        )

        self._clear_cone_control_locked()
        self._reset_boundary_pd_locked()

        self.target_v = 0.0
        self.target_w = 0.0

        for _ in range(
            self.fast_stop_burst_count
        ):
            self._publish_zero_locked()

        self.fast_stop_timer.reset()

    def fast_stop_loop(self):
        with self.motion_lock:
            if not self.stop_latched:
                self.fast_stop_timer.cancel()
                return

            if (
                time.monotonic()
                <= self.fast_stop_until
            ):
                self._publish_zero_locked()

            else:
                # 快速停车阶段结束。
                #
                # 之后仍由 40 Hz control_loop 持续发布零速度。
                self.fast_stop_timer.cancel()

    # ------------------------------------------------------------------
    # 统一 PD 融合控制
    # ------------------------------------------------------------------

    def control_loop(self):
        with self.motion_lock:
            self._publish_fused_control_locked(
                '40Hz保持'
            )

    def _publish_fused_control_locked(self, source):
        # 到达 P 点后永久保持停车。
        if (
            self.stop_latched
            or self.navigation_arrived
        ):
            self.execute_drive(
                0.0,
                0.0,
                '🏁 已返回P点，停车保持',
            )
            return

        # 第一帧位姿到达之前不发布巡航速度。
        if not self.pose_received:
            return

        # 到达 P 点附近后建立停车锁存。
        if (
            self.target_dist
            < self.arrival_tolerance
        ):
            self.navigation_arrived = True

            self._activate_target_stop_locked()

            actual_x = (
                self.cur_pose[0]
                + self.start_offset_x
            )
            actual_y = (
                self.cur_pose[1]
                + self.start_offset_y
            )

            self.get_logger().warn(
                f'🏁 已返回P点'
                f'({self.target_map_x:.2f},'
                f'{self.target_map_y:.2f})，'
                f'当前位置='
                f'({actual_x:.3f},{actual_y:.3f})，'
                f'距离={self.target_dist:.3f}m；'
                f'已建立停车锁存'
            )
            return

        w_raw = (
            self.target_w
            + self.boundary_w
            + self.current_cone_w
        )

        w_total = self.clamp(
            w_raw,
            -self.max_w,
            self.max_w,
        )

        limited = (
            abs(w_total - w_raw)
            > 1e-9
        )

        active_items = []

        if self.boundary_active:
            active_items.append(
                self.boundary_mode
            )

        if self.cone_control_active:
            active_items.append(
                '锥桶PD'
            )

        mode = (
            '+'.join(active_items)
            if active_items
            else '目标PD'
        )

        limit_text = (
            '，总角速度已限幅'
            if limited
            else ''
        )

        log_tag = (
            f'🧮 纯PD融合[{source}/{mode}]：'
            f'W目标={self.target_w:+.3f}，'
            f'W边界={self.boundary_w:+.3f}，'
            f'W锥桶={self.current_cone_w:+.3f}，'
            f'W原始={w_raw:+.3f}，'
            f'W输出={w_total:+.3f}'
            f'{limit_text}'
        )

        self.execute_drive(
            self.target_v,
            w_total,
            log_tag,
        )

    # ------------------------------------------------------------------
    # 最终速度发布
    # ------------------------------------------------------------------

    def execute_drive(
        self,
        v,
        w,
        log_tag,
    ):
        requested_nonzero = (
            abs(v) > 1e-6
            or abs(w) > 1e-6
        )

        with self.motion_lock:
            # 停车锁存建立后，拒绝所有非零速度。
            if (
                self.stop_latched
                and requested_nonzero
            ):
                return

            v = self.clamp(
                float(v),
                0.0,
                self.max_v,
            )

            w = self.clamp(
                float(w),
                -self.max_w,
                self.max_w,
            )

            self.last_cmd_v = v
            self.last_cmd_w = w

            twist = Twist()
            twist.linear.x = v
            twist.angular.z = w

            self.cmd_pub.publish(twist)

        if v == 0.0 and w == 0.0:
            self.get_logger().info(
                f'{log_tag} | '
                f'V:0.00 W:0.00',
                throttle_duration_sec=1.0,
            )

        elif 'PD融合' in log_tag:
            self.get_logger().info(
                f'{log_tag} | '
                f'V:{v:.2f} W:{w:.2f}',
                throttle_duration_sec=0.5,
            )

# ----------------------------------------------------------------------
# 主函数
# ----------------------------------------------------------------------

def main():
    rclpy.init()

    node = Task1NavAvoidNode()

    executor = MultiThreadedExecutor(
        num_threads=4
    )
    executor.add_node(node)

    try:
        executor.spin()

    except KeyboardInterrupt:
        node.get_logger().warn(
            '🛑 节点退出，发布停车命令'
        )

        # 直接通过当前发布器连续发送零速度。
        for _ in range(5):
            node.cmd_pub.publish(Twist())
            time.sleep(0.02)

        # 额外使用一次 ros2 topic pub，
        # 尽量确保退出时速度被清零。
        os.system(
            "ros2 topic pub --once "
            "/cmd_vel geometry_msgs/msg/Twist '{}' "
            "> /dev/null 2>&1"
        )

    finally:
        executor.shutdown()
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
