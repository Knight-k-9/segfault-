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
    - 可以缓存 /racing_obstacle_detection 视觉检测帧；
    - 不发布任何 /cmd_vel，包括零速度；
    - 不抢占任务一、去通道或任务二的控制权。

正式启动后：
    w_total = w_target + w_upper_boundary + w_visual_cone

    1. 目标点角度 PD；
    2. 地图上边界 PD：接近/越过上边界时强制产生左转分量；
    3. /racing_obstacle_detection 视觉锥桶 PD。

说明：
    - 已取消下边界控制；
    - 三个角速度分量仍直接相加后统一限幅；
    - 视觉避障每个真实检测帧更新一次，两帧之间保持最近结果；
    - 锥桶越靠近画面中心、bottom_y 越大，避障角速度越强。

进入以 P 点为中心的 0.40m × 0.35m 矩形范围后：
    - 建立永久停车锁存；
    - 连续发布零速度；
    - 200 Hz 快速停车覆盖一段时间；
    - 之后继续以 40 Hz 发布零速度。

订阅：
    odom_pose
    /racing_obstacle_detection
    /task3_start

发布：
    /cmd_vel
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
        # [配置区 3]：视觉锥桶 PD（沿用 task_xian 的核心逻辑）
        # ================================================================
        self.obstacle_topic = '/racing_obstacle_detection'

        # 有效检测与触发。
        self.conf_thresh = 0.60
        self.cone_trigger_y = 280.0

        # 纵向接近增益：
        # d1 = clamp(
        #     (bottom_y - cone_trigger_y) * cone_near_k,
        #     0,
        #     cone_near_max,
        # )
        self.cone_near_k = 0.025
        self.cone_near_max = 1.50

        # 锥桶越靠近图像中心，控制误差绝对值越大。
        self.cone_center_effect_width_px = 300.0
        self.cone_center_direction_deadband_px = 8.0

        # W>0 左转，W<0 右转。
        # 锥桶位于中心死区时默认左转。
        self.cone_default_turn_sign = 1.0

        # 视觉横向误差 PD。
        self.cone_kp = 0.004
        self.cone_kd = 0.001
        self.cone_d_filter_alpha = 0.35
        self.cone_d_limit = 800.0
        self.cone_w_limit = 2.00
        self.cone_min_abs_w = 0.60

        self.image_width = 640.0
        self.image_height = 480.0

        # 摄像头画面边缘忽略曲线，与 task_xian 保持一致。
        self.edge_measure_points = list(zip(
            [
                290.0, 295.0, 300.0, 305.0, 310.0, 315.0,
                320.0, 325.0, 330.0, 335.0, 340.0, 345.0,
            ],
            [
                155.0, 144.0, 120.0, 101.0, 92.0, 70.0,
                66.0, 60.0, 56.0, 54.0, 49.0, 30.0,
            ],
        ))
        self.edge_smoothing_enabled = True
        self.edge_ignore_margin_px = 5.0
        self._build_edge_boundaries()

        # 视觉流安全保护。
        self.require_obstacle_frame_before_motion = True
        self.stop_on_obstacle_frame_timeout = True
        self.obstacle_frame_timeout = 0.25

        # ================================================================
        # [配置区 4]：地图上边界 PD（只保留上边界）
        # ================================================================
        self.y_upper_boundary = 1.60

        self.boundary_kp = 76.0
        self.boundary_kd = 2.0
        self.boundary_d_filter_alpha = 0.35
        self.boundary_d_limit = 2.0
        self.boundary_w_limit = 18.0

        # 上边界要求左转：W>0。
        # 下边界控制已完全取消。

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
        # 视觉锥桶 PD 状态
        # ---------------------------------------------------------------
        self.cone_control_active = False
        self.current_cone_w = 0.0

        self.cone_prev_error = None
        self.cone_prev_time = None
        self.cone_d_filtered = 0.0
        self.cone_turn_sign = 0.0

        self.last_cone_center_x = 0.0
        self.last_cone_bottom_y = 0.0

        # 每条 /racing_obstacle_detection 消息对应一帧真实推理结果，
        # 包括 targets 为空的安全帧。回调只缓存，40Hz控制循环统一消费。
        self.latest_obstacle_msg = None
        self.obstacle_frame_received = False
        self.obstacle_frame_sequence = 0
        self.obstacle_frame_consumed_sequence = 0
        self.last_obstacle_frame_time = 0.0
        self.last_consumed_frame_had_hazard = False

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
        self.obstacle_callback_group = MutuallyExclusiveCallbackGroup()
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
        # 视觉锥桶检测订阅
        # ---------------------------------------------------------------
        obstacle_qos = QoSProfile(depth=1)
        obstacle_qos.history = HistoryPolicy.KEEP_LAST
        obstacle_qos.reliability = ReliabilityPolicy.BEST_EFFORT
        obstacle_qos.durability = DurabilityPolicy.VOLATILE

        self.obstacle_sub = self.create_subscription(
            PerceptionTargets,
            self.obstacle_topic,
            self.obs_cb,
            obstacle_qos,
            callback_group=self.obstacle_callback_group,
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
            '收到start前只缓存位姿和视觉检测帧，不发布任何/cmd_vel'
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
            f'上边界左转PD=({self.boundary_kp:.3f},{self.boundary_kd:.3f})，'
            f'视觉锥桶PD=({self.cone_kp:.5f},{self.cone_kd:.5f})'
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
            f'视觉帧已接收={self.obstacle_frame_received}'
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
    # 地图上边界PD：接近/越过上边界时产生左转角速度
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
        actual_y = self.cur_pose[1] + self.start_offset_y

        # 下边界控制已经取消；只有到达或超过上边界才启用保护。
        if actual_y < self.y_upper_boundary:
            self._reset_boundary_pd_locked()
            return

        mode = '上边界左转PD'
        error = actual_y - self.y_upper_boundary
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
                    error - self.boundary_prev_error
                ) / dt
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
            self.boundary_d_filtered = 0.0

        self.boundary_active = True
        self.boundary_mode = mode
        self.boundary_error = error
        self.boundary_error_d = error_d
        self.boundary_prev_error = error
        self.boundary_prev_time = now

        # W>0 为左转。上边界处不再右转，改为明确输出正角速度。
        boundary_w_raw = (
            self.boundary_kp * error
            + self.boundary_kd * error_d
        )

        self.boundary_w = self.clamp(
            boundary_w_raw,
            0.0,
            self.boundary_w_limit,
        )

    # ==================================================================
    # 摄像头画面边缘忽略
    # ==================================================================
    def _build_edge_boundaries(self):
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
            f'✅ 视觉边缘忽略已构建：实测点={len(points)}，'
            f'bottom_y范围=[{y_values[0]:.0f},{y_values[-1]:.0f}]，'
            f'平滑={self.edge_smoothing_enabled}，'
            f'内缩余量={self.edge_ignore_margin_px:.1f}px'
        )

    def _interpolate_left_edge_x(self, bottom_y):
        y = float(bottom_y)
        y_values = self.edge_y_values
        x_values = self.edge_left_x_values

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

    def get_edge_boundaries(self, bottom_y):
        center_x = self.image_width / 2.0
        left_x = (
            self._interpolate_left_edge_x(bottom_y)
            + self.edge_ignore_margin_px
        )
        left_x = self.clamp(left_x, 0.0, center_x - 1.0)
        right_x = 2.0 * center_x - left_x
        return left_x, right_x

    def is_in_edge_ignore_zone(self, center_x, bottom_y):
        left_x, right_x = self.get_edge_boundaries(bottom_y)
        return center_x < left_x or center_x > right_x

    # ==================================================================
    # 视觉锥桶PD
    # ==================================================================
    def _clear_cone_control_locked(self):
        self.cone_control_active = False
        self.current_cone_w = 0.0

        self.cone_prev_error = None
        self.cone_prev_time = None
        self.cone_d_filtered = 0.0
        self.cone_turn_sign = 0.0

        self.last_cone_center_x = 0.0
        self.last_cone_bottom_y = 0.0

    def collect_valid_obstacle_candidates(self, msg):
        """
        保留满足以下条件的视觉锥桶：
        1. 置信度超过阈值；
        2. bottom_y 超过触发阈值；
        3. 检测框中心位于左右有效边界之间。

        返回：(bottom_y, center_x, roi)
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
                        f'🟡 画面边缘锥桶忽略：center_x={center_x:.0f}，'
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
        """
        每个真实视觉检测帧重新选择 bottom_y 最大的最近有效锥桶。

        锥桶在画面右侧 -> 左转（W>0）；
        锥桶在画面左侧 -> 右转（W<0）；
        锥桶位于中心死区 -> 默认左转。
        """
        candidates = self.collect_valid_obstacle_candidates(msg)

        if not candidates:
            was_active = self.cone_control_active
            old_w = self.current_cone_w
            self._clear_cone_control_locked()

            if was_active:
                self.get_logger().info(
                    f'✅ 真实视觉帧无触发锥桶，视觉PD清零：'
                    f'旧W={old_w:+.3f}',
                    throttle_duration_sec=0.3,
                )
            return False

        bottom_y, center_x, roi = max(
            candidates,
            key=lambda item: item[0],
        )

        image_center_x = self.image_width / 2.0
        horizontal_offset = center_x - image_center_x

        if horizontal_offset > self.cone_center_direction_deadband_px:
            new_turn_sign = 1.0
        elif horizontal_offset < -self.cone_center_direction_deadband_px:
            new_turn_sign = -1.0
        else:
            new_turn_sign = self.cone_default_turn_sign

        direction_changed = (
            self.cone_turn_sign != 0.0
            and new_turn_sign != self.cone_turn_sign
        )
        if direction_changed:
            self.cone_prev_error = None
            self.cone_prev_time = None
            self.cone_d_filtered = 0.0

        self.cone_turn_sign = new_turn_sign

        center_distance = abs(horizontal_offset)
        center_closeness_px = max(
            0.0,
            self.cone_center_effect_width_px - center_distance,
        )

        control_error = self.cone_turn_sign * center_closeness_px
        now = time.monotonic()
        control_error_d = self._update_cone_derivative_locked(
            control_error,
            now,
        )

        d1 = (bottom_y - self.cone_trigger_y) * self.cone_near_k
        d1 = self.clamp(d1, 0.0, self.cone_near_max)

        d2_p = self.cone_kp * control_error
        d2_d = self.cone_kd * control_error_d
        d2 = d2_p + d2_d

        # D项可以削弱转向，但不允许把本帧避障方向反转。
        if self.cone_turn_sign * d2 < 0.0:
            d2 = 0.0

        cone_w_raw = d1 * d2
        cone_w = self.clamp(
            cone_w_raw,
            -self.cone_w_limit,
            self.cone_w_limit,
        )

        if abs(cone_w) < self.cone_min_abs_w:
            cone_w = math.copysign(
                min(self.cone_min_abs_w, self.cone_w_limit),
                self.cone_turn_sign,
            )

        self.cone_control_active = True
        self.current_cone_w = cone_w
        self.last_cone_center_x = center_x
        self.last_cone_bottom_y = bottom_y

        direction_text = '左转' if self.cone_turn_sign > 0.0 else '右转'
        change_text = '，本帧方向已重选' if direction_changed else ''

        self.get_logger().info(
            f'🟠 视觉锥桶PD：bottom_y={bottom_y:.0f}，'
            f'd1={d1:.3f}，center_x={center_x:.1f}px，'
            f'距中心={center_distance:.1f}px，'
            f'中心接近度={center_closeness_px:.1f}px，'
            f'方向={direction_text}{change_text}，'
            f'控制误差={control_error:+.1f}，'
            f'D={control_error_d:+.1f}/s，'
            f'd2=P({d2_p:+.3f})+D({d2_d:+.3f})={d2:+.3f}，'
            f'W原始={cone_w_raw:+.3f}，W锥桶={cone_w:+.3f}，'
            f'conf={roi.confidence:.2f}',
            throttle_duration_sec=0.3,
        )
        return True

    def obs_cb(self, msg: PerceptionTargets):
        """只缓存最新真实视觉检测帧；回调内不发布 /cmd_vel。"""
        with self.motion_lock:
            if self.stop_latched or self.navigation_arrived:
                return

            self.latest_obstacle_msg = msg
            self.obstacle_frame_sequence += 1
            self.obstacle_frame_received = True
            self.last_obstacle_frame_time = time.monotonic()

    def _consume_latest_obstacle_frame_locked(self):
        """在40Hz控制循环中至多消费一帧最新视觉检测结果。"""
        if (
            self.latest_obstacle_msg is None
            or self.obstacle_frame_sequence
            == self.obstacle_frame_consumed_sequence
        ):
            return False

        msg = self.latest_obstacle_msg
        sequence = self.obstacle_frame_sequence
        self.obstacle_frame_consumed_sequence = sequence
        self.last_consumed_frame_had_hazard = (
            self.update_cone_control_locked(msg)
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
            # 休眠期间只缓存视觉帧，不消费、不计算控制量、不发布速度。
            if not self.task3_active:
                return

            consumed = self._consume_latest_obstacle_frame_locked()
            if consumed:
                source = (
                    f'40Hz/消费视觉检测帧#'
                    f'{self.obstacle_frame_consumed_sequence}'
                )
            else:
                source = '40Hz/无新视觉帧-保持上一帧锥桶状态'

            self._publish_fused_control_locked(source)

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

        # 视觉避障启用后，未收到第一帧真实检测结果前保持停车。
        if (
            self.require_obstacle_frame_before_motion
            and not self.obstacle_frame_received
        ):
            self.execute_drive(
                0.0,
                0.0,
                '⏳ 等待/racing_obstacle_detection首帧视觉结果',
            )
            return

        # YOLO视觉流失联时停车，避免继续使用过期避障结果。
        if (
            self.stop_on_obstacle_frame_timeout
            and self.obstacle_frame_received
        ):
            frame_age = time.monotonic() - self.last_obstacle_frame_time
            if frame_age > self.obstacle_frame_timeout:
                self.execute_drive(
                    0.0,
                    0.0,
                    f'⛔ 视觉检测流超时{frame_age:.2f}s，停车保护',
                )
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
                '视觉锥桶PD'
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
            f'W上边界={self.boundary_w:+.3f}，'
            f'W视觉锥桶={self.current_cone_w:+.3f}，'
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
            cone_received = self.obstacle_frame_received
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
                f'视觉帧已缓存={cone_received}；'
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
            f'视觉锥桶PD={self.cone_control_active}，'
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
