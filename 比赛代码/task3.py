"""
任务三：等待 /task3_start 后返回 P 点。

目标点：
    P = (0.55, 0.22)

启动规则：
    订阅 /task3_start，std_msgs/msg/String
    只有消息内容严格为 start 时，才正式开始任务三。

休眠规则：
    在收到 /task3_start 前：
    - 可以缓存 /odom_pose；
    - 可以缓存 /cone_coordinates；
    - 不发布任何 /cmd_vel，包括零速度；
    - 不抢占任务一、去通道或任务二的控制权。

正式启动后：
    w_total = w_target + w_boundary + w_cone

    1. 目标点角度 PD；
    2. 地图上下边界 PD；
    3. /cone_coordinates 地图坐标锥桶 PD。

进入以 P 点为中心的 0.40m × 0.35m 矩形范围后：
    - 建立永久停车锁存；
    - 连续发布零速度；
    - 200 Hz 快速停车覆盖一段时间；
    - 之后继续以 40 Hz 发布零速度。

订阅：
    odom_pose
    /cone_coordinates
    /task3_start

发布：
    /cmd_vel
"""

import json
import math
import os
import threading
import time
from typing import List, Tuple

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


MapPoint = Tuple[float, float]


class Task3ReturnPNode(Node):

    def __init__(self):
        super().__init__('task3_return_p_node')

        # ================================================================
        # [配置区 1]：任务三启动交接
        # ================================================================
        self.task3_start_topic = '/task3_start'

        # 收到一次有效 start 后永久锁定，不允许重复重新启动。
        self.task3_start_received = False
        self.task3_active = False
        self.task3_complete = False

        # ================================================================
        # [配置区 2]：P 点、坐标偏移和目标角度 PD
        # ================================================================
        # odom_pose=(0,0) 对应地图坐标 P=(0.55,0.22)。
        self.start_offset_x = 0.55
        self.start_offset_y = 0.20

        # 最终返回起始地图点 P。
        self.target_map_x = self.start_offset_x
        self.target_map_y = self.start_offset_y

        # 线速度：
        #     v = kp_linear * 目标距离
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

        # 到达 P 点的矩形停车范围。
        #
        # 矩形以目标点为中心：
        #     X轴总宽度 = 0.40m，即 |actual_x - target_x| <= 0.20m
        #     Y轴总高度 = 0.35m，即 |actual_y - target_y| <= 0.175m
        #
        # 只要车辆地图坐标进入该矩形，就立即建立永久停车锁存。
        self.arrival_box_width_x = 0.40
        self.arrival_box_height_y = 0.35
        self.arrival_half_x = self.arrival_box_width_x * 0.5
        self.arrival_half_y = self.arrival_box_height_y * 0.5

        # ================================================================
        # [配置区 3]：基于地图坐标的锥桶 PD
        # ================================================================
        self.cone_coordinates_topic = '/cone_coordinates'

        # 只处理车辆前方该距离以内的锥桶。
        self.cone_trigger_longitudinal = 1.60

        # 已经越过或距离车辆位姿点过近的锥桶不参与控制。
        self.cone_min_longitudinal = 0.03

        # 横向距离超过该值时忽略。
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
        #     锥桶位于车辆右侧；
        #     输出正角速度；
        #     车辆向左避障。
        self.cone_kp = 8.00
        self.cone_kd = 0.15
        self.cone_d_filter_alpha = 0.35
        self.cone_d_limit = 3.0
        self.cone_w_limit = 1.80

        # ================================================================
        # [配置区 4]：地图上下边界 PD
        # ================================================================
        self.y_lower_boundary = 0.20
        self.y_upper_boundary = 1.60

        self.boundary_kp = 76.0
        self.boundary_kd = 2.0
        self.boundary_d_filter_alpha = 0.35
        self.boundary_d_limit = 2.0
        self.boundary_w_limit = 18.0

        # ================================================================
        # [配置区 5]：目标点停车
        # ================================================================
        # 首次到达目标时连续发送若干次零速度。
        self.fast_stop_burst_count = 4

        # 到达后200Hz零速度覆盖持续时间。
        self.fast_stop_hold_sec = 0.35

        # 快速停车定时器周期。
        self.fast_stop_timer_period = 0.005

        # ================================================================
        # [状态变量]
        # ================================================================
        self.cur_pose = [0.0, 0.0, 0.0]
        self.pose_received = False

        self.navigation_arrived = False

        self.last_cmd_v = 0.0
        self.last_cmd_w = 0.0
        self.cmd_publish_count = 0

        # ---------------------------------------------------------------
        # 目标点 PD 状态
        # ---------------------------------------------------------------
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
        self.latest_cone_coordinates: List[MapPoint] = []
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

        # 建立后拒绝所有非零速度。
        self.stop_latched = False

        # 快速零速度覆盖结束时间。
        self.fast_stop_until = 0.0

        # ================================================================
        # [回调组]
        # ================================================================
        self.start_callback_group = MutuallyExclusiveCallbackGroup()
        self.stop_callback_group = MutuallyExclusiveCallbackGroup()
        self.control_callback_group = MutuallyExclusiveCallbackGroup()
        self.cone_callback_group = MutuallyExclusiveCallbackGroup()
        self.pose_callback_group = MutuallyExclusiveCallbackGroup()

        # ================================================================
        # [ROS通信]
        # ================================================================

        # ---------------------------------------------------------------
        # 任务三启动订阅
        # ---------------------------------------------------------------
        # 与任务二的 /task3_start 发布器保持一致：
        # RELIABLE + TRANSIENT_LOCAL。
        start_qos = QoSProfile(depth=1)
        start_qos.history = HistoryPolicy.KEEP_LAST
        start_qos.reliability = ReliabilityPolicy.RELIABLE
        start_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

        self.task3_start_sub = self.create_subscription(
            String,
            self.task3_start_topic,
            self.task3_start_cb,
            start_qos,
            callback_group=self.start_callback_group,
        )

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
        # 40Hz控制。
        self.timer = self.create_timer(
            0.025,
            self.control_loop,
            callback_group=self.control_callback_group,
        )

        # 到达目标点后的200Hz快速零速度覆盖。
        self.fast_stop_timer = self.create_timer(
            self.fast_stop_timer_period,
            self.fast_stop_loop,
            callback_group=self.stop_callback_group,
        )
        self.fast_stop_timer.cancel()

        # 休眠/运行状态日志。
        self.status_timer = self.create_timer(
            1.0,
            self.status_loop,
        )

        # ================================================================
        # 启动日志
        # ================================================================
        self.get_logger().info('=' * 72)
        self.get_logger().info(
            '🚀 任务三返回P点节点已经启动'
        )
        self.get_logger().info(
            f'休眠等待：{self.task3_start_topic}，'
            '消息内容必须为 start'
        )
        self.get_logger().info(
            '收到start前只缓存位姿和锥桶，不发布任何/cmd_vel'
        )
        self.get_logger().info(
            f'最终目标P='
            f'({self.target_map_x:.2f},{self.target_map_y:.2f})，'
            f'停车矩形=X±{self.arrival_half_x:.3f}m、'
            f'Y±{self.arrival_half_y:.3f}m，'
            f'总尺寸={self.arrival_box_width_x:.2f}m'
            f'×{self.arrival_box_height_y:.2f}m'
        )
        self.get_logger().info(
            f'目标PD=({self.target_kp:.3f},{self.target_kd:.3f})，'
            f'边界PD=({self.boundary_kp:.3f},{self.boundary_kd:.3f})，'
            f'锥桶PD=({self.cone_kp:.3f},{self.cone_kd:.3f})'
        )
        self.get_logger().info('=' * 72)

    # ==================================================================
    # 通用辅助函数
    # ==================================================================
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

    # ==================================================================
    # 任务三启动交接
    # ==================================================================
    def task3_start_cb(self, msg: String):
        command = str(msg.data).strip().lower()

        if command != 'start':
            self.get_logger().warn(
                f'⚠️ 收到无效 {self.task3_start_topic}：'
                f'{msg.data!r}；要求内容为 start',
                throttle_duration_sec=1.0,
            )
            return

        with self.motion_lock:
            if self.task3_start_received:
                return

            self.task3_start_received = True
            self.task3_active = True
            self.task3_complete = False

            # 确保任务三从干净控制状态开始。
            self.navigation_arrived = False
            self.stop_latched = False
            self.fast_stop_until = 0.0
            self.fast_stop_timer.cancel()

            self.last_cmd_v = 0.0
            self.last_cmd_w = 0.0

            self.target_dist = float('inf')
            self.target_v = 0.0
            self.target_w = 0.0
            self.target_angle_error = 0.0
            self.target_angle_d = 0.0
            self.target_prev_error = None
            self.target_prev_time = None
            self.target_d_filtered = 0.0

            self._reset_boundary_pd_locked()
            self._clear_cone_control_locked()

            now = time.monotonic()

            # 如果任务三启动前已经缓存了位姿，
            # 立即根据最新位姿初始化控制器。
            if self.pose_received:
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

        self.get_logger().warn(
            f'🚦 已收到 {self.task3_start_topic}: start，'
            f'任务三正式接管；'
            f'当前地图位置=({actual_x:.3f},{actual_y:.3f})，'
            f'位姿已接收={self.pose_received}，'
            f'锥桶坐标已接收={self.cone_coordinates_received}'
        )

    # ==================================================================
    # 位姿更新：目标PD + 边界PD + 锥桶PD
    # ==================================================================
    def pose_cb(self, msg: Pose2D):
        now = time.monotonic()

        with self.motion_lock:
            first_pose = not self.pose_received

            self.cur_pose = [
                float(msg.x),
                float(msg.y),
                float(msg.theta),
            ]
            self.pose_received = True

            actual_x = (
                self.cur_pose[0]
                + self.start_offset_x
            )
            actual_y = (
                self.cur_pose[1]
                + self.start_offset_y
            )

            # 休眠阶段只缓存位姿，绝不计算并发布运动命令。
            if not self.task3_active:
                pass

            # 停车锁存后只保存最新位姿，不更新控制器。
            elif not self.stop_latched:
                self._update_target_pd_from_pose_locked(now)
                self._update_boundary_pd_from_pose_locked(now)
                self._update_cone_control_from_coordinates_locked(now)

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

    # ==================================================================
    # 地图边界PD
    # ==================================================================
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

    # ==================================================================
    # 地图坐标锥桶避障
    # ==================================================================
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

    def _parse_cone_coordinates(
        self,
        raw_text: str,
    ) -> List[MapPoint]:
        """
        解析 /cone_coordinates JSON数组：

            [{"x": 1.23, "y": 0.68}, ...]
        """
        data = json.loads(raw_text)

        if not isinstance(data, list):
            raise ValueError(
                'JSON顶层必须是数组'
            )

        coordinates: List[MapPoint] = []

        for index, item in enumerate(data):
            if not isinstance(item, dict):
                self.get_logger().warn(
                    f'忽略第{index}个锥桶：元素不是对象',
                    throttle_duration_sec=1.0,
                )
                continue

            if 'x' not in item or 'y' not in item:
                self.get_logger().warn(
                    f'忽略第{index}个锥桶：缺少x或y',
                    throttle_duration_sec=1.0,
                )
                continue

            try:
                cone_x = float(item['x'])
                cone_y = float(item['y'])

            except (TypeError, ValueError):
                self.get_logger().warn(
                    f'忽略第{index}个锥桶：x/y不是有效数字',
                    throttle_duration_sec=1.0,
                )
                continue

            if (
                not math.isfinite(cone_x)
                or not math.isfinite(cone_y)
            ):
                self.get_logger().warn(
                    f'忽略第{index}个锥桶：x/y非有限值',
                    throttle_duration_sec=1.0,
                )
                continue

            coordinates.append(
                (cone_x, cone_y)
            )

        return coordinates

    def cone_coordinates_cb(self, msg: String):
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
                    f'❌ /cone_coordinates JSON解析失败：'
                    f'{exc}；'
                    f'原始内容={raw_text[:200]!r}',
                    throttle_duration_sec=1.0,
                )
                return

        now = time.monotonic()

        with self.motion_lock:
            # 无论是否启动，都缓存最新锥桶地图坐标。
            self.latest_cone_coordinates = coordinates
            self.cone_coordinates_received = True

            # 任务三休眠时只缓存，绝不发布/cmd_vel。
            if not self.task3_active:
                return

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
        将地图坐标锥桶转换到车辆坐标系。

        返回：
            longitudinal：
                车辆前方为正。

            lateral_right：
                车辆右侧为正。
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

        longitudinal = (
            cos_yaw * dx
            + sin_yaw * dy
        )

        lateral_left = (
            -sin_yaw * dx
            + cos_yaw * dy
        )

        lateral_right = -lateral_left

        return longitudinal, lateral_right

    def collect_valid_cone_candidates_locked(self):
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

        # 选择车辆前方纵向距离最近的锥桶。
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

        if lateral_right > 0.0:
            side_text = '右侧'
        elif lateral_right < 0.0:
            side_text = '左侧'
        else:
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

    # ==================================================================
    # 目标点停车
    # ==================================================================
    def _publish_zero_locked(self):
        zero = Twist()

        self.last_cmd_v = 0.0
        self.last_cmd_w = 0.0

        self.cmd_pub.publish(zero)
        self.cmd_publish_count += 1

    def _activate_target_stop_locked(self):
        """
        建立永久停车锁存。

        1. stop_latched=True；
        2. 清除锥桶和边界控制；
        3. 立即连续发布零速度；
        4. 开启200Hz快速零速度覆盖；
        5. 后续40Hz控制继续发布零速度。
        """
        self.stop_latched = True
        self.task3_complete = True

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
            if not self.task3_active:
                self.fast_stop_timer.cancel()
                return

            if not self.stop_latched:
                self.fast_stop_timer.cancel()
                return

            if (
                time.monotonic()
                <= self.fast_stop_until
            ):
                self._publish_zero_locked()

            else:
                # 后续仍由40Hz控制循环持续停车。
                self.fast_stop_timer.cancel()

    # ==================================================================
    # 统一PD融合控制
    # ==================================================================
    def control_loop(self):
        with self.motion_lock:
            self._publish_fused_control_locked(
                '40Hz保持'
            )

    def _publish_fused_control_locked(self, source):
        # 关键安全规则：
        # 收到/task3_start之前完全静默，不发布零速度或非零速度。
        if not self.task3_active:
            return

        # 到达P点后永久保持停车。
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

        # 进入以P点为中心的矩形停车范围后建立停车锁存。
        actual_x = (
            self.cur_pose[0]
            + self.start_offset_x
        )
        actual_y = (
            self.cur_pose[1]
            + self.start_offset_y
        )

        arrival_error_x = (
            actual_x
            - self.target_map_x
        )
        arrival_error_y = (
            actual_y
            - self.target_map_y
        )

        inside_arrival_box = (
            abs(arrival_error_x)
            <= self.arrival_half_x
            and abs(arrival_error_y)
            <= self.arrival_half_y
        )

        if inside_arrival_box:
            self.navigation_arrived = True

            self._activate_target_stop_locked()

            self.get_logger().warn(
                f'🏁 已进入P点矩形停车范围，'
                f'目标='
                f'({self.target_map_x:.2f},'
                f'{self.target_map_y:.2f})，'
                f'当前位置='
                f'({actual_x:.3f},{actual_y:.3f})，'
                f'X误差={arrival_error_x:+.3f}m'
                f'(允许±{self.arrival_half_x:.3f}m)，'
                f'Y误差={arrival_error_y:+.3f}m'
                f'(允许±{self.arrival_half_y:.3f}m)，'
                f'欧氏距离={self.target_dist:.3f}m；'
                f'已建立永久停车锁存'
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

    # ==================================================================
    # 最终速度发布
    # ==================================================================
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
            # 休眠期间拒绝一切速度发布，包括零速度。
            if not self.task3_active:
                return

            # 停车锁存建立后拒绝所有非零速度。
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
            self.cmd_publish_count += 1

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

    # ==================================================================
    # 状态日志
    # ==================================================================
    def status_loop(self):
        with self.motion_lock:
            active = self.task3_active
            complete = self.task3_complete
            pose_received = self.pose_received
            cone_received = self.cone_coordinates_received
            target_dist = self.target_dist
            cmd_count = self.cmd_publish_count

            actual_x = (
                self.cur_pose[0]
                + self.start_offset_x
            )
            actual_y = (
                self.cur_pose[1]
                + self.start_offset_y
            )

        if not active:
            self.get_logger().info(
                f'💤 任务三休眠：等待 '
                f'{self.task3_start_topic}=start；'
                f'位姿已缓存={pose_received}，'
                f'锥桶坐标已缓存={cone_received}；'
                f'休眠期间不发布/cmd_vel'
            )
            return

        if complete:
            self.get_logger().info(
                f'📊 任务三已完成：'
                f'当前位置=({actual_x:.3f},{actual_y:.3f})，'
                f'停车锁存={self.stop_latched}，'
                f'cmd_count={cmd_count}'
            )
            return

        arrival_error_x = (
            actual_x
            - self.target_map_x
        )
        arrival_error_y = (
            actual_y
            - self.target_map_y
        )

        self.get_logger().info(
            f'📊 任务三运行：'
            f'当前位置=({actual_x:.3f},{actual_y:.3f})，'
            f'距P点={target_dist:.3f}m，'
            f'矩形误差='
            f'X{arrival_error_x:+.3f}/'
            f'±{self.arrival_half_x:.3f}m，'
            f'Y{arrival_error_y:+.3f}/'
            f'±{self.arrival_half_y:.3f}m，'
            f'边界={self.boundary_mode}，'
            f'锥桶PD={self.cone_control_active}，'
            f'cmd_count={cmd_count}'
        )


# ======================================================================
# 主函数
# ======================================================================
def main():
    rclpy.init()

    node = Task3ReturnPNode()

    executor = MultiThreadedExecutor(
        num_threads=4
    )
    executor.add_node(node)

    try:
        executor.spin()

    except KeyboardInterrupt:
        # 任务三还在休眠时不能发布零速度，
        # 否则可能抢占仍在运行的任务一、去通道或任务二。
        if node.task3_active:
            node.get_logger().warn(
                '🛑 任务三退出，连续发布停车命令'
            )

            for _ in range(5):
                node.cmd_pub.publish(Twist())
                time.sleep(0.02)

            os.system(
                "ros2 topic pub --once "
                "/cmd_vel geometry_msgs/msg/Twist '{}' "
                "> /dev/null 2>&1"
            )

        else:
            node.get_logger().warn(
                '任务三在休眠状态退出，'
                '不发布/cmd_vel，避免抢占其他阶段'
            )

    finally:
        executor.shutdown()
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
