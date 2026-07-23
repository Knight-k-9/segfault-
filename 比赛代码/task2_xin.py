#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
任务二：根据二维码顺逆时针方向跟踪 merged_lines.txt。

工作流程：
1. 节点启动后订阅 /qr_direction_result，缓存顺时针或逆时针；
2. 节点启动后订阅 /task2_start；
3. 只有同时满足：
       已收到有效二维码方向；
       已收到 /task2_start，内容为 start；
   才开始发布非零 /cmd_vel；
4. 根据二维码方向动态拼接 merged_lines.txt：
       顺时针：D0 + reverse(L) + reverse(T) + R + reverse(D1)
       逆时针：D1 + reverse(R) + T + L + reverse(D0)
5. 到达轨迹终点后：
       连续发布零速度；
       发布 /task3_start，内容为 start；
       停止任务二控制，不再抢占 /cmd_vel。

避障角速度：
    使用任务一同款 d1 × d2 动态幅值；
    避障激活时完全接管，不与 Stanley W 融合。

订阅：
    /odom_pose
    /imu_data
    /racing_obstacle_detection
    /qr_direction_result
    /task2_start

发布：
    /cmd_vel
    /task3_start
"""

import math
import os
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

import rclpy
from ai_msgs.msg import PerceptionTargets
from geometry_msgs.msg import Pose2D, Twist
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import Imu
from std_msgs.msg import String


PathPoint = Tuple[float, float]


class FileTrackFollowerNode(Node):

    def __init__(self):
        super().__init__('task2_file_track_follower_node')

        # ==============================================================
        # 1. 轨迹和任务交接话题
        # ==============================================================
        self.track_file = 'merged_lines.txt'

        self.qr_direction_topic = '/qr_direction_result'
        self.task2_start_topic = '/task2_start'
        self.task3_start_topic = '/task3_start'

        # 当前方向：
        # None：还没收到二维码；
        # True：顺时针；
        # False：逆时针。
        self.is_clockwise: Optional[bool] = None
        self.qr_direction_text = ''

        # 去通道是否已经允许任务二开始。
        self.task2_start_received = False

        # 任务二是否已经正式激活。
        self.task2_active = False

        # 是否已经完成任务二。
        self.is_finished = False

        # 是否已经发布过任务三启动消息。
        self.task3_start_published = False

        # 路线只在收到二维码方向后加载。
        self.dense_path: List[PathPoint] = []
        self.current_path_index = 0

        self.is_first_run = True
        self.has_reached_middle = False

        # ==============================================================
        # 2. 地图坐标与阿克曼参数
        # ==============================================================
        self.start_offset_x = 0.55
        self.start_offset_y = 0.20

        self.wheelbase = 0.144
        self.imu_to_front = 0.11

        self.target_v = 1.0
        self.k_stanley = 2.1

        # 舵机响应时延预测。
        self.servo_delay = 0.15

        self.max_v = 1.0
        self.max_w = 5.0

        # Stanley前视。
        self.look_ahead_steps = 5
        self.max_steer_angle = math.radians(35.0)

        # ==============================================================
        # 3. RGB锥桶避障
        # ==============================================================
        self.conf_thresh = 0.60
        self.dist_thresh_y = 300.0

        self.avoid_v = 1.0

        # 只移植任务一的 d1 × d2 动态避障角速度，不做任何 W 融合：
        #
        # d1：锥桶在图像中越靠下，说明越接近，避障强度越大。
        # d2：锥桶越靠近图像中心，横向避障强度越大，并加入 D 项抑制突变。
        # W_avoid = clamp(d1 * d2, -avoid_w_limit, avoid_w_limit)
        #
        # 避障激活时仍然完全使用 W_avoid，不与 Stanley 的 W 相加或加权。
        self.avoid_near_k = 0.035
        self.avoid_near_max = 1.6

        self.avoid_center_effect_width_px = 285.0
        self.avoid_center_direction_deadband_px = 8.0
        self.avoid_default_turn_sign = 1.0

        self.avoid_kp = 0.006
        self.avoid_kd = 0.0007
        self.avoid_d_filter_alpha = 0.35
        self.avoid_d_limit = 800.0

        self.avoid_w_limit = 3.00
        self.avoid_min_abs_w = 0.80

        self.image_width = 640.0
        self.image_height = 480.0

        # ==============================================================
        # 摄像头画面边缘忽略
        # 每个元素为：(bottom_y, 左侧有效边界x)。
        # 右侧边界以图像中心为轴对称生成。
        # ==============================================================
        self.edge_measure_points = [
            (285.0, 158.0),
            (290.0, 145.0),
            (295.0, 134.0),
            (300.0, 110.0),
            (305.0, 91.0),
            (310.0, 82.0),
            (315.0, 60.0),
            (320.0, 56.0),
            (325.0, 50.0),
            (330.0, 46.0),
            (335.0, 44.0),
            (340.0, 39.0),
            (345.0, 20.0),
        ]
        self.edge_smoothing_enabled = True
        self.edge_ignore_margin_px = 5.0
        self.edge_y_values: List[float] = []
        self.edge_left_x_values: List[float] = []
        self._build_edge_boundaries()

        # 避障状态只允许由真实障碍物检测帧更新。
        # 两个检测帧之间持续保持上一帧的状态，避免避障/Stanley来回切换。
        self.rgb_avoid_active = False
        self.current_avoid_w = 0.0

        # d1 × d2 中 d2 的 PD 状态。
        self.avoid_prev_error = None
        self.avoid_prev_time = None
        self.avoid_d_filtered = 0.0
        self.avoid_turn_sign = 0.0
        self.last_avoid_center_x = 0.0
        self.last_avoid_bottom_y = 0.0

        # ==============================================================
        # 4. 终点与停车参数
        # ==============================================================
        self.goal_index_ratio = 0.95
        self.middle_index_ratio = 0.50
        self.goal_distance_tolerance = 0.20

        self.goal_slow_index_ratio = 0.90
        self.goal_slow_distance = 0.60
        self.goal_slow_kp = 1.50

        # 到达终点时先连续发布零速度，再通知任务三。
        self.finish_stop_burst_count = 8
        self.finish_stop_interval = 0.02

        # ==============================================================
        # 5. 实时数据状态
        # ==============================================================
        self.cur_pose = [0.0, 0.0, 0.0]
        self.pose_received = False

        self.imu_w_z = 0.0
        self.imu_received = False

        # /racing_obstacle_detection回调只缓存最新真实检测帧。
        # 40Hz控制循环每帧最多消费一次；没有新帧时保持上一帧避障状态。
        self.latest_obs: Optional[PerceptionTargets] = None
        self.obstacle_frame_sequence = 0
        self.obstacle_frame_consumed_sequence = 0

        self.state_lock = threading.RLock()

        # 统一以40Hz发布底层速度。
        self.dt = 0.025

        self.cmd_publish_count = 0

        # ==============================================================
        # 6. ROS订阅与发布
        # ==============================================================

        self.pose_sub = self.create_subscription(
            Pose2D,
            '/odom_pose',
            self.pose_cb,
            10,
        )

        self.imu_sub = self.create_subscription(
            Imu,
            '/imu_data',
            self.imu_cb,
            10,
        )

        obstacle_qos = QoSProfile(depth=1)
        obstacle_qos.history = HistoryPolicy.KEEP_LAST
        obstacle_qos.reliability = ReliabilityPolicy.BEST_EFFORT
        obstacle_qos.durability = DurabilityPolicy.VOLATILE

        self.obs_sub = self.create_subscription(
            PerceptionTargets,
            '/racing_obstacle_detection',
            self.obs_cb,
            obstacle_qos,
        )

        # 二维码节点使用普通可靠QoS发布，因此这里同样使用depth=10。
        self.qr_direction_sub = self.create_subscription(
            String,
            self.qr_direction_topic,
            self.qr_direction_cb,
            10,
        )

        # 任务二应提前启动，因此这里使用普通可靠QoS即可。
        # 去通道完成时发布 start。
        start_qos = QoSProfile(depth=1)
        start_qos.history = HistoryPolicy.KEEP_LAST
        start_qos.reliability = ReliabilityPolicy.RELIABLE
        start_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

        self.task2_start_sub = self.create_subscription(
            String,
            self.task2_start_topic,
            self.task2_start_cb,
            start_qos,
        )

        cmd_qos = QoSProfile(depth=1)
        cmd_qos.history = HistoryPolicy.KEEP_LAST
        cmd_qos.reliability = ReliabilityPolicy.RELIABLE
        cmd_qos.durability = DurabilityPolicy.VOLATILE

        self.cmd_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            cmd_qos,
        )

        # TRANSIENT_LOCAL保证任务三稍晚启动仍可收到start。
        self.task3_start_pub = self.create_publisher(
            String,
            self.task3_start_topic,
            start_qos,
        )

        self.timer = self.create_timer(
            self.dt,
            self.control_loop,
        )

        self.status_timer = self.create_timer(
            1.0,
            self.status_loop,
        )

        self.get_logger().info('=' * 72)
        self.get_logger().info('🚀 任务二轨迹跟踪节点启动')
        self.get_logger().info(
            f'方向订阅：{self.qr_direction_topic}'
        )
        self.get_logger().info(
            f'任务二启动订阅：{self.task2_start_topic}，'
            '要求内容为 start'
        )
        self.get_logger().info(
            f'任务三启动发布：{self.task3_start_topic}，'
            '内容为 start'
        )
        self.get_logger().info(
            '未同时收到二维码方向和task2_start之前，'
            '绝不发布非零/cmd_vel'
        )
        self.get_logger().info(
            '控制频率=40Hz；避障状态只由新的真实检测帧更新，'
            '两帧之间保持上一帧状态'
        )
        self.get_logger().info(
            '任务二避障W=d1×d2动态计算；避障激活时完全接管，'
            '不与Stanley W融合'
        )
        self.get_logger().info(
            'RGB边缘忽略：按bottom_y实测曲线插值左右有效边界，'
            f'平滑={self.edge_smoothing_enabled}，'
            f'内缩={self.edge_ignore_margin_px:.1f}px'
        )
        self.get_logger().info('=' * 72)

    # ==============================================================
    # 基础工具
    # ==============================================================
    @staticmethod
    def normalize_angle(angle: float) -> float:
        return (
            angle + math.pi
        ) % (2.0 * math.pi) - math.pi

    @staticmethod
    def clamp(
        value: float,
        lower: float,
        upper: float,
    ) -> float:
        return max(lower, min(value, upper))

    # ==============================================================
    # 摄像头画面边缘忽略
    # ==============================================================
    def _build_edge_boundaries(self) -> None:
        """排序、平滑边缘实测点，供后续按bottom_y插值。"""
        points = sorted(
            (
                (float(bottom_y), float(left_x))
                for bottom_y, left_x in self.edge_measure_points
                if math.isfinite(float(bottom_y))
                and math.isfinite(float(left_x))
            ),
            key=lambda item: item[0],
        )

        if len(points) < 2:
            raise ValueError('摄像头边缘忽略点至少需要2组有效数据')

        y_values: List[float] = []
        raw_x_values: List[float] = []

        for bottom_y, left_x in points:
            # 避免重复bottom_y造成插值分母为0；相同y保留最后一项。
            if y_values and abs(bottom_y - y_values[-1]) < 1e-9:
                raw_x_values[-1] = left_x
                continue

            y_values.append(bottom_y)
            raw_x_values.append(left_x)

        if len(y_values) < 2:
            raise ValueError('摄像头边缘忽略点去重后不足2组')

        if self.edge_smoothing_enabled and len(raw_x_values) >= 3:
            smoothed_x_values: List[float] = []
            last_index = len(raw_x_values) - 1

            for index, current_x in enumerate(raw_x_values):
                if index == 0:
                    # 首点采用3:1平滑，避免端点移动过大。
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
                    # 中间点采用1:2:1平滑。
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
            f'✅ 任务二画面边缘忽略已构建：实测点={len(points)}，'
            f'bottom_y范围=[{y_values[0]:.0f},{y_values[-1]:.0f}]，'
            f'平滑={self.edge_smoothing_enabled}，'
            f'内缩余量={self.edge_ignore_margin_px:.1f}px'
        )

    def _interpolate_left_edge_x(self, bottom_y: float) -> float:
        """按照bottom_y对左侧有效边界做分段线性插值。"""
        y = float(bottom_y)
        y_values = self.edge_y_values
        x_values = self.edge_left_x_values

        # 实测范围外保持端点值，不做不可靠的多项式外推。
        if y <= y_values[0]:
            return x_values[0]

        if y >= y_values[-1]:
            return x_values[-1]

        for index in range(1, len(y_values)):
            y1 = y_values[index]

            if y <= y1:
                y0 = y_values[index - 1]
                x0 = x_values[index - 1]
                x1 = x_values[index]
                ratio = (y - y0) / (y1 - y0)
                return x0 + ratio * (x1 - x0)

        return x_values[-1]

    def get_edge_boundaries(
        self,
        bottom_y: float,
    ) -> Tuple[float, float]:
        """返回指定bottom_y下，加入安全余量后的左右有效边界。"""
        center_x = self.image_width / 2.0

        left_x = (
            self._interpolate_left_edge_x(bottom_y)
            + self.edge_ignore_margin_px
        )

        # 防止参数错误导致左右边界交叉。
        left_x = self.clamp(
            left_x,
            0.0,
            center_x - 1.0,
        )
        right_x = 2.0 * center_x - left_x
        return left_x, right_x

    def is_in_edge_ignore_zone(
        self,
        center_x: float,
        bottom_y: float,
    ) -> bool:
        """检测框中心位于左右有效边界之外时返回True。"""
        left_x, right_x = self.get_edge_boundaries(bottom_y)
        return center_x < left_x or center_x > right_x

    # ==============================================================
    # 二维码方向
    # ==============================================================
    def qr_direction_cb(self, msg: String) -> None:
        text = str(msg.data).strip()

        if text in ('顺时针', 'cw', 'CW', 'clockwise'):
            clockwise = True
            normalized_text = '顺时针'

        elif text in ('逆时针', 'ccw', 'CCW', 'counterclockwise'):
            clockwise = False
            normalized_text = '逆时针'

        else:
            self.get_logger().warn(
                f'⚠️ 收到无效二维码方向：{text!r}；'
                '仅接受顺时针或逆时针',
                throttle_duration_sec=1.0,
            )
            return

        with self.state_lock:
            if self.task2_active:
                # 任务开始后不允许中途切换路线。
                if clockwise != self.is_clockwise:
                    self.get_logger().error(
                        f'❌ 任务二运行中收到不同方向：'
                        f'{normalized_text}，忽略'
                    )
                return

            if (
                self.is_clockwise is not None
                and self.is_clockwise == clockwise
            ):
                return

            self.is_clockwise = clockwise
            self.qr_direction_text = normalized_text

            loaded = self.load_track_for_current_direction()

        if not loaded:
            return

        self.get_logger().warn(
            f'🧭 已缓存二维码方向：{normalized_text}；'
            f'轨迹点={len(self.dense_path)}'
        )

        self.try_activate_task2(
            '收到二维码方向'
        )

    def load_track_for_current_direction(self) -> bool:
        if self.is_clockwise is None:
            return False

        direction = (
            'cw'
            if self.is_clockwise
            else 'ccw'
        )

        path = self.load_and_stitch_track(
            self.track_file,
            direction=direction,
        )

        if not path:
            self.dense_path = []
            self.get_logger().error(
                f'❌ 无法按{self.qr_direction_text}加载轨迹文件：'
                f'{self.track_file}'
            )
            return False

        self.dense_path = path
        self.current_path_index = 0
        self.is_first_run = True
        self.has_reached_middle = False
        return True

    # ==============================================================
    # 任务二启动交接
    # ==============================================================
    def task2_start_cb(self, msg: String) -> None:
        command = str(msg.data).strip().lower()

        if command != 'start':
            self.get_logger().warn(
                f'⚠️ 收到无效 {self.task2_start_topic}：'
                f'{msg.data!r}；要求内容为 start'
            )
            return

        with self.state_lock:
            if self.task2_start_received:
                return

            self.task2_start_received = True

        self.get_logger().warn(
            f'📩 已收到 {self.task2_start_topic}: start'
        )

        self.try_activate_task2(
            '收到去通道完成信号'
        )

    def try_activate_task2(self, reason: str) -> bool:
        with self.state_lock:
            if self.task2_active or self.is_finished:
                return self.task2_active

            if not self.task2_start_received:
                return False

            if self.is_clockwise is None:
                self.get_logger().info(
                    '⏳ 已收到task2_start，但尚未收到二维码方向；'
                    '保持停车等待',
                    throttle_duration_sec=1.0,
                )
                return False

            if not self.dense_path:
                if not self.load_track_for_current_direction():
                    return False

            if not self.pose_received:
                self.get_logger().info(
                    '⏳ task2_start和二维码方向已就绪，'
                    '等待/odom_pose',
                    throttle_duration_sec=1.0,
                )
                return False

            self.current_path_index = 0
            self.is_first_run = True
            self.has_reached_middle = False
            self._clear_rgb_avoid_control_locked()
            self.latest_obs = None
            self.obstacle_frame_sequence = 0
            self.obstacle_frame_consumed_sequence = 0
            self.task2_active = True

        self.get_logger().warn(
            f'🚦 任务二正式激活：{reason}；'
            f'方向={self.qr_direction_text}；'
            f'轨迹点={len(self.dense_path)}'
        )
        return True

    # ==============================================================
    # 轨迹文件
    # ==============================================================
    def resolve_track_file(self, file_path: str) -> Path:
        requested = Path(file_path)

        if requested.is_absolute():
            return requested

        # 优先使用程序所在目录，避免从其他工作目录启动时找不到文件。
        script_directory = Path(__file__).resolve().parent
        script_candidate = script_directory / requested

        if script_candidate.is_file():
            return script_candidate

        return requested

    def load_and_stitch_track(
        self,
        file_path: str,
        direction: str = 'cw',
    ) -> List[PathPoint]:

        lines_data = {
            'D0': [],
            'D1': [],
            'L': [],
            'R': [],
            'T': [],
        }

        resolved_path = self.resolve_track_file(file_path)

        try:
            with resolved_path.open(
                'r',
                encoding='utf-8',
            ) as file:
                for line_raw in file:
                    line = line_raw.strip()

                    if not line or line.startswith('#'):
                        continue

                    parts = [
                        part.strip()
                        for part in line.split(',')
                    ]

                    if len(parts) < 4:
                        continue

                    line_id = parts[0]

                    if line_id not in lines_data:
                        continue

                    try:
                        x = float(parts[2])
                        y = float(parts[3])
                    except ValueError:
                        continue

                    if not (
                        math.isfinite(x)
                        and math.isfinite(y)
                    ):
                        continue

                    lines_data[line_id].append((x, y))

        except Exception as exc:
            self.get_logger().error(
                f'读取轨迹文件失败：{resolved_path}；{exc}'
            )
            return []

        if direction == 'cw':
            stitched = (
                lines_data['D0']
                + lines_data['L'][::-1]
                + lines_data['T'][::-1]
                + lines_data['R']
                + lines_data['D1'][::-1]
            )
        else:
            stitched = (
                lines_data['D1']
                + lines_data['R'][::-1]
                + lines_data['T']
                + lines_data['L']
                + lines_data['D0'][::-1]
            )

        self.get_logger().info(
            f'📂 轨迹文件={resolved_path}，'
            f'方向={direction}，'
            f'D0={len(lines_data["D0"])}，'
            f'D1={len(lines_data["D1"])}，'
            f'L={len(lines_data["L"])}，'
            f'R={len(lines_data["R"])}，'
            f'T={len(lines_data["T"])}，'
            f'拼接后={len(stitched)}'
        )

        return stitched

    # ==============================================================
    # ROS数据回调
    # ==============================================================
    def pose_cb(self, msg: Pose2D) -> None:
        with self.state_lock:
            self.cur_pose = [
                float(msg.x),
                float(msg.y),
                float(msg.theta),
            ]
            self.pose_received = True

        if (
            self.task2_start_received
            and not self.task2_active
        ):
            self.try_activate_task2(
                '收到位姿后条件满足'
            )

    def imu_cb(self, msg: Imu) -> None:
        with self.state_lock:
            self.imu_w_z = float(
                msg.angular_velocity.z
            )
            self.imu_received = True

    def obs_cb(
        self,
        msg: PerceptionTargets,
    ) -> None:
        """只缓存最新真实检测帧；回调内不发布/cmd_vel。"""
        with self.state_lock:
            # 未开始任务二时不缓存旧障碍物，
            # 防止正式启动时处理上一阶段检测帧。
            if not self.task2_active:
                return

            self.latest_obs = msg
            self.obstacle_frame_sequence += 1

    def consume_latest_obstacle_frame_locked(self) -> bool:
        """在40Hz循环中至多消费一次最新障碍物检测帧。"""
        if (
            self.latest_obs is None
            or self.obstacle_frame_sequence
            == self.obstacle_frame_consumed_sequence
        ):
            return False

        msg = self.latest_obs
        self.obstacle_frame_consumed_sequence = (
            self.obstacle_frame_sequence
        )

        # 只有消费到一帧真实检测结果时，才允许改变避障状态。
        # targets为空或无有效锥桶时关闭避障；否则开启避障。
        self.rgb_avoid_active = self.detect_hazard(msg)
        return True

    # ==============================================================
    # 40Hz统一控制状态机
    # ==============================================================
    def control_loop(self) -> None:
        with self.state_lock:
            active = self.task2_active
            finished = self.is_finished
            path_ready = bool(self.dense_path)

            if active and not finished:
                consumed = self.consume_latest_obstacle_frame_locked()
                avoid_active = self.rgb_avoid_active
                avoid_w = self.current_avoid_w
                consumed_sequence = (
                    self.obstacle_frame_consumed_sequence
                )
            else:
                consumed = False
                avoid_active = False
                avoid_w = 0.0
                consumed_sequence = 0

        if finished:
            return

        if not active:
            # 等待期间绝不发布/cmd_vel，避免抢占任务一和去通道。
            return

        if not path_ready:
            self.finish_with_error(
                '任务二已经激活，但轨迹为空'
            )
            return

        if avoid_active:
            source = (
                f'⚡ 任务二RGB避障/检测帧#{consumed_sequence}'
                if consumed
                else '⚡ 任务二RGB避障/保持上一检测帧状态'
            )
            self.execute_drive(
                self.avoid_v,
                avoid_w,
                source,
            )
            return

        self.perform_line_tracking()

    # ==============================================================
    # RGB锥桶避障
    # ==============================================================
    def _clear_rgb_avoid_control_locked(self) -> None:
        """关闭任务二锥桶避障，并清除 d2 的 PD 历史。"""
        self.rgb_avoid_active = False
        self.current_avoid_w = 0.0
        self.avoid_prev_error = None
        self.avoid_prev_time = None
        self.avoid_d_filtered = 0.0
        self.avoid_turn_sign = 0.0
        self.last_avoid_center_x = 0.0
        self.last_avoid_bottom_y = 0.0

    def _update_rgb_avoid_derivative_locked(
        self,
        control_error: float,
        now: float,
    ) -> float:
        """更新 d2 的微分项，并做限幅和一阶低通滤波。"""
        error_d = 0.0

        if (
            self.avoid_prev_error is not None
            and self.avoid_prev_time is not None
        ):
            dt = now - self.avoid_prev_time

            if dt >= 1e-3:
                raw_d = (
                    control_error
                    - self.avoid_prev_error
                ) / dt

                raw_d = self.clamp(
                    raw_d,
                    -self.avoid_d_limit,
                    self.avoid_d_limit,
                )

                alpha = self.avoid_d_filter_alpha
                self.avoid_d_filtered = (
                    alpha * raw_d
                    + (1.0 - alpha)
                    * self.avoid_d_filtered
                )
                error_d = self.avoid_d_filtered

        self.avoid_prev_error = control_error
        self.avoid_prev_time = now
        return error_d

    def detect_hazard(
        self,
        msg: PerceptionTargets,
    ) -> bool:

        candidates = []

        for target in getattr(msg, 'targets', []):
            rois = list(getattr(target, 'rois', []))

            if not rois:
                continue

            for roi in rois:
                confidence = float(
                    getattr(roi, 'confidence', 0.0)
                )

                if confidence <= self.conf_thresh:
                    continue

                rect = getattr(roi, 'rect', None)
                if rect is None:
                    continue

                width = float(
                    getattr(rect, 'width', 0.0)
                )
                height = float(
                    getattr(rect, 'height', 0.0)
                )

                if width <= 0.0 or height <= 0.0:
                    continue

                center_x = (
                    float(getattr(rect, 'x_offset', 0.0))
                    + width / 2.0
                )
                bottom_y = (
                    float(getattr(rect, 'y_offset', 0.0))
                    + height
                )

                if bottom_y <= self.dist_thresh_y:
                    continue

                # 忽略落在摄像头左右无效边缘区域内的检测框。
                if self.is_in_edge_ignore_zone(
                    center_x,
                    bottom_y,
                ):
                    continue

                candidates.append(
                    (
                        bottom_y,
                        center_x,
                        confidence,
                    )
                )

        if not candidates:
            was_active = self.rgb_avoid_active
            old_w = self.current_avoid_w
            self._clear_rgb_avoid_control_locked()

            if was_active:
                self.get_logger().info(
                    f'✅ 当前真实检测帧无触发锥桶，'
                    f'退出任务二RGB避障：旧W={old_w:+.2f}',
                    throttle_duration_sec=0.5,
                )
            return False

        # 选择bottom_y最大的最近锥桶。
        candidates.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        bottom_y, center_x, confidence = candidates[0]

        image_center_x = self.image_width / 2.0
        horizontal_offset = center_x - image_center_x

        # 每个真实检测帧都按当前最近锥桶重新选择方向：
        # 锥桶在右侧 -> 左转（w>0）
        # 锥桶在左侧 -> 右转（w<0）
        # 中心死区 -> 使用 avoid_default_turn_sign。
        if (
            horizontal_offset
            > self.avoid_center_direction_deadband_px
        ):
            new_turn_sign = 1.0
        elif (
            horizontal_offset
            < -self.avoid_center_direction_deadband_px
        ):
            new_turn_sign = -1.0
        else:
            new_turn_sign = self.avoid_default_turn_sign

        # 当前最近锥桶导致绕行方向改变时，清除旧 D 项，
        # 避免上一只锥桶或上一方向的微分污染本帧。
        direction_changed = (
            self.avoid_turn_sign != 0.0
            and new_turn_sign != self.avoid_turn_sign
        )
        if direction_changed:
            self.avoid_prev_error = None
            self.avoid_prev_time = None
            self.avoid_d_filtered = 0.0

        self.avoid_turn_sign = new_turn_sign

        # d2 的误差绝对值由锥桶靠近图像中心的程度决定。
        # 锥桶越靠近中心，center_closeness_px 越大。
        center_distance = abs(horizontal_offset)
        center_closeness_px = max(
            0.0,
            self.avoid_center_effect_width_px
            - center_distance,
        )

        control_error = (
            self.avoid_turn_sign
            * center_closeness_px
        )

        now = time.monotonic()
        control_error_d = (
            self._update_rgb_avoid_derivative_locked(
                control_error,
                now,
            )
        )

        # d1：纵向接近增益。
        # bottom_y 越大，锥桶越接近，d1 越大。
        d1 = (
            bottom_y - self.dist_thresh_y
        ) * self.avoid_near_k
        d1 = self.clamp(
            d1,
            0.0,
            self.avoid_near_max,
        )

        # d2：横向中心接近度的 PD。
        d2_p = self.avoid_kp * control_error
        d2_d = self.avoid_kd * control_error_d
        d2 = d2_p + d2_d

        # D 项只允许削弱本帧方向，不能把本帧控制反向。
        if self.avoid_turn_sign * d2 < 0.0:
            d2 = 0.0

        # 只使用 d1 × d2 得到避障 W。
        # 这里没有与 Stanley W 做相加、加权或任何融合。
        avoid_w_raw = d1 * d2
        avoid_w = self.clamp(
            avoid_w_raw,
            -self.avoid_w_limit,
            self.avoid_w_limit,
        )

        # 保留很小的最低转向保障，避免刚超过触发线时 W 过小。
        if abs(avoid_w) < self.avoid_min_abs_w:
            avoid_w = math.copysign(
                min(
                    self.avoid_min_abs_w,
                    self.avoid_w_limit,
                ),
                self.avoid_turn_sign,
            )

        self.current_avoid_w = avoid_w
        self.last_avoid_center_x = center_x
        self.last_avoid_bottom_y = bottom_y

        direction_text = (
            '左转'
            if self.avoid_turn_sign > 0.0
            else '右转'
        )
        change_text = (
            '，方向已重选'
            if direction_changed
            else ''
        )

        self.get_logger().info(
            f'⚠️ 任务二锥桶d1×d2：'
            f'bottom_y={bottom_y:.1f}，'
            f'd1={d1:.3f}，'
            f'center_x={center_x:.1f}，'
            f'距中心={center_distance:.1f}px，'
            f'中心接近度={center_closeness_px:.1f}px，'
            f'方向={direction_text}{change_text}，'
            f'd2=P({d2_p:+.3f})'
            f'+D({d2_d:+.3f})={d2:+.3f}，'
            f'W原始={avoid_w_raw:+.3f}，'
            f'W避障={avoid_w:+.3f}，'
            f'conf={confidence:.2f}',
            throttle_duration_sec=0.3,
        )

        return True

    # ==============================================================
    # Stanley轨迹跟踪
    # ==============================================================
    def perform_line_tracking(self) -> None:
        with self.state_lock:
            odom_x = self.cur_pose[0]
            odom_y = self.cur_pose[1]
            yaw = self.cur_pose[2]
            w_current = self.imu_w_z

        map_x = odom_x + self.start_offset_x
        map_y = odom_y + self.start_offset_y

        # 当前前轴中心。
        front_x = (
            map_x
            + self.imu_to_front * math.cos(yaw)
        )
        front_y = (
            map_y
            + self.imu_to_front * math.sin(yaw)
        )

        # 舵机时延预测。
        tau = self.servo_delay
        yaw_pred = yaw + w_current * tau
        yaw_mid = yaw + 0.5 * w_current * tau

        predicted_x = (
            front_x
            + self.target_v * tau * math.cos(yaw_mid)
        )
        predicted_y = (
            front_y
            + self.target_v * tau * math.sin(yaw_mid)
        )

        if self.is_first_run:
            search_start = 0
            search_end = len(self.dense_path)
        else:
            search_start = max(
                0,
                self.current_path_index - 2,
            )
            search_end = min(
                len(self.dense_path),
                self.current_path_index + 100,
            )

        minimum_distance = float('inf')
        closest_index = self.current_path_index

        for index in range(search_start, search_end):
            point_x, point_y = self.dense_path[index]

            distance = math.hypot(
                point_x - predicted_x,
                point_y - predicted_y,
            )

            if distance < minimum_distance:
                minimum_distance = distance
                closest_index = index

        if self.is_first_run:
            # 防止起点靠近路线末端时错误锁定到终点。
            if (
                closest_index
                > len(self.dense_path) * 0.95
            ):
                closest_index = 0

            self.is_first_run = False

        self.current_path_index = closest_index

        if (
            closest_index
            > len(self.dense_path)
            * self.middle_index_ratio
        ):
            self.has_reached_middle = True

        goal_x, goal_y = self.dense_path[-1]

        distance_to_goal = math.hypot(
            goal_x - map_x,
            goal_y - map_y,
        )

        reached_goal = (
            self.has_reached_middle
            and closest_index
            > len(self.dense_path)
            * self.goal_index_ratio
            and distance_to_goal
            < self.goal_distance_tolerance
        )

        if reached_goal:
            self.complete_task2(
                distance_to_goal,
                closest_index,
            )
            return

        next_index = min(
            closest_index + self.look_ahead_steps,
            len(self.dense_path) - 1,
        )

        point_x, point_y = (
            self.dense_path[closest_index]
        )
        next_x, next_y = (
            self.dense_path[next_index]
        )

        path_yaw = math.atan2(
            next_y - point_y,
            next_x - point_x,
        )

        dx = predicted_x - point_x
        dy = predicted_y - point_y

        cross_track_error = (
            dx * math.sin(path_yaw)
            - dy * math.cos(path_yaw)
        )

        heading_error = self.normalize_angle(
            path_yaw - yaw_pred
        )

        current_speed = max(
            0.10,
            abs(self.target_v),
        )

        steering_angle = (
            heading_error
            + math.atan2(
                self.k_stanley * cross_track_error,
                current_speed,
            )
        )

        steering_angle = self.clamp(
            steering_angle,
            -self.max_steer_angle,
            self.max_steer_angle,
        )

        angular_speed = (
            current_speed
            / self.wheelbase
        ) * math.tan(steering_angle)

        linear_speed = self.target_v

        if (
            self.has_reached_middle
            and closest_index
            > len(self.dense_path)
            * self.goal_slow_index_ratio
            and distance_to_goal
            < self.goal_slow_distance
        ):
            linear_speed = self.clamp(
                distance_to_goal
                * self.goal_slow_kp,
                0.0,
                self.target_v,
            )

        self.execute_drive(
            linear_speed,
            angular_speed,
            (
                f'📐 Stanley：方向={self.qr_direction_text}，'
                f'IMU-W={w_current:+.2f}，'
                f'横向误差={cross_track_error*100:+.1f}cm，'
                f'舵角={math.degrees(steering_angle):+.1f}°，'
                f'idx={closest_index}/{len(self.dense_path)-1}，'
                f'终点距离={distance_to_goal:.2f}m'
            ),
        )

    # ==============================================================
    # 任务完成与交接任务三
    # ==============================================================
    def complete_task2(
        self,
        distance_to_goal: float,
        closest_index: int,
    ) -> None:

        with self.state_lock:
            if self.is_finished:
                return

            # 先停止控制状态，防止并发定时器继续发运动命令。
            self.is_finished = True
            self.task2_active = False
            self.latest_obs = None
            self.obstacle_frame_sequence = 0
            self.obstacle_frame_consumed_sequence = 0
            self._clear_rgb_avoid_control_locked()

        self.get_logger().warn(
            f'🏁 任务二到达终点：'
            f'方向={self.qr_direction_text}，'
            f'idx={closest_index}/{len(self.dense_path)-1}，'
            f'终点距离={distance_to_goal:.3f}m；'
            '先连续停车，再启动任务三'
        )

        # 必须先停车，后发布任务三启动消息。
        for _ in range(self.finish_stop_burst_count):
            self.publish_zero()
            time.sleep(self.finish_stop_interval)

        self.publish_task3_start()

    def publish_task3_start(self) -> None:
        with self.state_lock:
            if self.task3_start_published:
                return

            self.task3_start_published = True

        message = String()
        message.data = 'start'
        self.task3_start_pub.publish(message)

        self.get_logger().warn(
            f'📤 已发布 {self.task3_start_topic}: start；'
            '任务二此后不再发布/cmd_vel'
        )

    def finish_with_error(self, reason: str) -> None:
        with self.state_lock:
            self.is_finished = True
            self.task2_active = False

        for _ in range(self.finish_stop_burst_count):
            self.publish_zero()
            time.sleep(self.finish_stop_interval)

        self.get_logger().error(
            f'❌ 任务二异常停止：{reason}；'
            '不会发布/task3_start'
        )

    # ==============================================================
    # 速度发布
    # ==============================================================
    def execute_drive(
        self,
        linear_speed: float,
        angular_speed: float,
        log_tag: str,
    ) -> None:

        with self.state_lock:
            if not self.task2_active or self.is_finished:
                return

        linear_speed = self.clamp(
            float(linear_speed),
            0.0,
            self.max_v,
        )

        angular_speed = self.clamp(
            float(angular_speed),
            -self.max_w,
            self.max_w,
        )

        twist = Twist()
        twist.linear.x = linear_speed
        twist.angular.z = angular_speed

        self.cmd_pub.publish(twist)
        self.cmd_publish_count += 1

        self.get_logger().info(
            f'{log_tag} | '
            f'V={linear_speed:.2f}，'
            f'W={angular_speed:.2f}',
            throttle_duration_sec=0.5,
        )

    def publish_zero(self) -> None:
        self.cmd_pub.publish(Twist())
        self.cmd_publish_count += 1

    # ==============================================================
    # 状态输出
    # ==============================================================
    def status_loop(self) -> None:
        if self.is_finished:
            self.get_logger().info(
                f'📊 任务二已完成；'
                f'task3_start已发布='
                f'{self.task3_start_published}'
            )
            return

        if self.task2_active:
            self.get_logger().info(
                f'📊 任务二运行中：'
                f'方向={self.qr_direction_text}，'
                f'idx={self.current_path_index}/'
                f'{max(0, len(self.dense_path)-1)}，'
                f'cmd_count={self.cmd_publish_count}'
            )
            return

        self.get_logger().info(
            f'⏳ 任务二等待：'
            f'二维码方向='
            f'{self.qr_direction_text or "未收到"}，'
            f'task2_start='
            f'{self.task2_start_received}，'
            f'位姿已接收={self.pose_received}，'
            '等待期间不发布/cmd_vel'
        )


def main() -> None:
    rclpy.init()

    node = FileTrackFollowerNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        node.get_logger().warn(
            '🛑 任务二被人工中断，连续发布停车命令'
        )

        for _ in range(8):
            node.publish_zero()
            time.sleep(0.02)

        os.system(
            "ros2 topic pub --once /cmd_vel "
            "geometry_msgs/msg/Twist '{}' "
            "> /dev/null 2>&1"
        )

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
