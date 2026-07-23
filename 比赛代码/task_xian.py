import json
import math
import os
import threading
import time

import rclpy
from ai_msgs.msg import PerceptionTargets
from geometry_msgs.msg import Pose2D, Twist
from sensor_msgs.msg import Imu
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


# 与 drawn_lines_4.txt 完全一致的后备路线。
DEFAULT_TRACK_POINTS = [
    (0.5391, 0.1914),
    (0.5898, 0.1914),
    (0.6406, 0.1914),
    (0.6914, 0.1875),
    (0.7422, 0.1875),
    (0.7891, 0.1875),
    (0.8398, 0.1875),
    (0.8906, 0.1875),
    (0.9414, 0.1875),
    (0.9922, 0.1875),
    (1.0391, 0.1914),
    (1.0898, 0.1914),
    (1.1367, 0.1953),
    (1.1875, 0.1953),
    (1.2383, 0.1992),
    (1.2852, 0.1992),
    (1.3320, 0.2031),
    (1.3828, 0.2070),
    (1.4297, 0.2109),
    (1.4766, 0.2148),
    (1.5273, 0.2188),
    (1.5742, 0.2227),
    (1.6211, 0.2305),
    (1.6680, 0.2344),
    (1.7148, 0.2422),
    (1.7617, 0.2461),
    (1.8086, 0.2539),
    (1.8555, 0.2617),
    (1.9023, 0.2656),
    (1.9492, 0.2734),
    (1.9961, 0.2812),
    (2.0391, 0.2891),
    (2.0859, 0.2969),
    (2.1328, 0.3086),
    (2.1758, 0.3164),
    (2.2227, 0.3242),
    (2.2656, 0.3359),
    (2.3125, 0.3438),
    (2.3555, 0.3555),
    (2.4023, 0.3672),
    (2.4453, 0.3789),
    (2.4883, 0.3906),
    (2.5352, 0.4023),
    (2.5781, 0.4141),
    (2.6211, 0.4258),
    (2.6641, 0.4375),
    (2.7070, 0.4492),
    (2.7500, 0.4648),
    (2.7930, 0.4766),
    (2.8359, 0.4922),
    (2.8789, 0.5039),
    (2.9219, 0.5195),
    (2.9648, 0.5352),
    (3.0078, 0.5508),
    (3.0508, 0.5664),
    (3.0898, 0.5820),
    (3.1328, 0.5977),
    (3.1758, 0.6133),
    (3.2148, 0.6328),
    (3.2578, 0.6484),
    (3.2969, 0.6680),
    (3.3398, 0.6836),
    (3.3789, 0.7031),
    (3.4180, 0.7227),
    (3.4609, 0.7383),
    (3.5000, 0.7578),
    (3.5391, 0.7773),
    (3.5781, 0.7969),
    (3.6172, 0.8203),
    (3.6562, 0.8398),
    (3.6953, 0.8594),
    (3.7344, 0.8828),
    (3.7734, 0.9023),
    (3.8125, 0.9258),
    (3.8516, 0.9453),
    (3.8906, 0.9688),
    (3.9297, 0.9922),
    (3.9688, 1.0156),
    (4.0039, 1.0391),
    (4.0430, 1.0625),
    (4.0781, 1.0859),
    (4.1172, 1.1094),
    (4.1562, 1.1367),
    (4.1914, 1.1602),
    (4.2266, 1.1875),
    (4.2656, 1.2109),
    (4.3008, 1.2383),
    (4.3359, 1.2656),
    (4.3750, 1.2930),
    (4.4102, 1.3203),
    (4.4453, 1.3477),
    (4.4805, 1.3750),
    (4.5156, 1.4023),
    (4.5508, 1.4297),
    (4.5859, 1.4570),
    (4.6211, 1.4883),
    (4.6562, 1.5156),
    (4.6914, 1.5469),
    (4.7266, 1.5781),
    (4.7617, 1.6055),
]


class Task1NavAvoidNode(Node):
    def __init__(self):
        super().__init__('task1_nav_avoid_node')

        # ================================================================
        # ROS2 YAML 参数读取
        # ================================================================
        # 可通过：
        #   python3 task_xian.py --ros-args --params-file task_xian.yaml
        # 加载参数。未在 YAML 中填写的项目使用代码中的默认值。
        def param(name, default):
            self.declare_parameter(name, default)
            return self.get_parameter(name).value

        # ================================================================
        # [配置区 1]：路线、车辆与 Stanley 参数
        # ================================================================
        self.track_file_parameter = str(
            param('track_file', 'drawn_lines_4.txt')
        ).strip()

        # odom_pose=(0,0) 对应的地图坐标偏移。
        self.start_offset_x = float(param('start_offset_x', 0.55))
        self.start_offset_y = float(param('start_offset_y', 0.22))

        # 车辆物理参数。位姿坐标以车体中心为参考。
        self.wheelbase = float(param('wheelbase', 0.144))
        self.vehicle_width = float(param('vehicle_width', 0.20))
        self.vehicle_half_width = self.vehicle_width / 2.0
        self.minimum_turn_radius = float(param('minimum_turn_radius', 0.35))
        self.imu_to_front = float(param('imu_to_front', 0.11))
        self.servo_delay = float(param('servo_delay', 0.15))
        self.imu_filter_alpha = float(param('imu_filter_alpha', 0.20))

        # Stanley 参数。
        self.stanley_target_v = float(param('stanley_target_v', 1.0))
        self.k_stanley = float(param('k_stanley', 2.1))
        self.stanley_softening_speed = float(
            param('stanley_softening_speed', 0.10)
        )
        self.stanley_look_ahead_steps = max(
            1, int(param('stanley_look_ahead_steps', 5))
        )
        self.stanley_search_back = max(
            0, int(param('stanley_search_back', 3))
        )
        self.stanley_search_forward = max(
            1, int(param('stanley_search_forward', 100))
        )
        self.stanley_max_steer_deg = float(
            param('stanley_max_steer_deg', 35.0)
        )

        # R = L / tan(delta)。物理舵角同时受设定角度与最小转弯半径约束。
        self.stanley_max_steer = min(
            math.radians(self.stanley_max_steer_deg),
            math.atan2(self.wheelbase, self.minimum_turn_radius),
        )

        # 路线终点减速与矩形到达区域。
        # 距离路线末点最后 goal_slowdown_distance 米内，线速度按剩余直线距离
        # 线性缩放：距离为1.0m时保持目标速度，越接近末点速度越低。
        self.goal_slowdown_distance = max(
            0.01, float(param('goal_slowdown_distance', 1.5))
        )

        goal_region_x_a = float(param('goal_region_x_min', 4.4))
        goal_region_x_b = float(param('goal_region_x_max', 5.0))
        goal_region_y_a = float(param('goal_region_y_min', 1.5))
        goal_region_y_b = float(param('goal_region_y_max', 2.0))
        self.goal_region_x_min = min(goal_region_x_a, goal_region_x_b)
        self.goal_region_x_max = max(goal_region_x_a, goal_region_x_b)
        self.goal_region_y_min = min(goal_region_y_a, goal_region_y_b)
        self.goal_region_y_max = max(goal_region_y_a, goal_region_y_b)

        # 仍要求路线索引已进入末段，防止车辆意外经过矩形区域时提前判定到达。
        self.path_end_index_ratio = float(param('path_end_index_ratio', 0.80))

        # 总速度/角速度限幅。
        self.max_v = float(param('max_v', 1.0))
        self.max_w = float(param('max_w', 3.0))
        self.stanley_w_limit = float(param('stanley_w_limit', 2.0))

        self.dense_path, self.track_file_used = self.load_track_points(
            self.track_file_parameter
        )
        self.path_ready = len(self.dense_path) >= 2
        if not self.path_ready:
            self.get_logger().error('❌ Stanley 路线为空，任务1保持停车')

        # ================================================================
        # [配置区 2]：锥桶 PD
        # ================================================================
        # 原有避障逻辑保持不变：
        # 1. 每帧仍选择 bottom_y 最大的最近有效锥桶；
        # 2. 不增加目标跟踪、主动“已通过”判断或旧锥桶抑制；
        # 3. 一帧真实无触发锥桶立即退出避障并恢复 Stanley；
        # 4. 不使用跨帧方向锁，40Hz消费机制保持原样。

        # 有效检测与触发。
        self.conf_thresh = float(param('conf_thresh', 0.60))
        self.cone_trigger_y = float(param('cone_trigger_y', 280.0))

        # 纵向接近增益：
        # d1 = clamp(
        #     (bottom_y - cone_trigger_y) * cone_near_k,
        #     0,
        #     cone_near_max
        # )
        self.cone_near_k = float(param('cone_near_k', 0.025))
        self.cone_near_max = float(param('cone_near_max', 1.50))

        # 横向方向和中心增强。
        self.cone_center_effect_width_px = float(
            param('cone_center_effect_width_px', 300.0)
        )
        self.cone_center_direction_deadband_px = float(
            param('cone_center_direction_deadband_px', 8.0)
        )

        # +1=左转，-1=右转。
        # 当前要求：锥桶中心位于死区内时默认左转。
        default_turn_raw = float(param('cone_default_turn_sign', 1.0))
        self.cone_default_turn_sign = (
            1.0 if default_turn_raw >= 0.0 else -1.0
        )

        # PD增益。
        self.cone_kp = float(param('cone_kp', 0.004))
        self.cone_kd = float(param('cone_kd', 0.001))
        self.cone_d_filter_alpha = float(
            param('cone_d_filter_alpha', 0.35)
        )
        self.cone_d_limit = float(param('cone_d_limit', 800.0))

        # 避障角速度范围。
        # 参考纯PD版本：锥桶移向画面边缘时，W应自然减弱；
        # 因此最低角速度只保留很小的起步保障，不再长期强制0.60rad/s。
        self.cone_w_limit = float(param('cone_w_limit', 2.00))
        self.cone_min_abs_w = max(
            0.0, float(param('cone_min_abs_w', 0.60))
        )

        # 图像尺寸。
        self.image_width = float(param('image_width', 640.0))
        self.image_height = float(param('image_height', 480.0))

        # ================================================================
        # [配置区 2.1]：摄像头画面边缘忽略
        # ================================================================
        default_edge_bottom_y = [
            290.0, 295.0, 300.0, 305.0, 310.0, 315.0,
            320.0, 325.0, 330.0, 335.0, 340.0, 345.0,
        ]
        default_edge_left_x = [
            155.0, 144.0, 120.0, 101.0, 92.0, 70.0,
            66.0, 60.0, 56.0, 54.0, 49.0, 30.0,
        ]

        edge_bottom_y = [
            float(value)
            for value in param(
                'edge_measure_bottom_y',
                default_edge_bottom_y,
            )
        ]
        edge_left_x = [
            float(value)
            for value in param(
                'edge_measure_left_x',
                default_edge_left_x,
            )
        ]

        if len(edge_bottom_y) != len(edge_left_x) or len(edge_bottom_y) < 2:
            self.get_logger().warning(
                '⚠️ edge_measure_bottom_y 与 edge_measure_left_x '
                '长度不一致或点数不足，使用代码默认边缘点'
            )
            edge_bottom_y = default_edge_bottom_y
            edge_left_x = default_edge_left_x

        self.edge_measure_points = list(zip(edge_bottom_y, edge_left_x))
        self.edge_smoothing_enabled = bool(
            param('edge_smoothing_enabled', True)
        )
        self.edge_ignore_margin_px = float(
            param('edge_ignore_margin_px', 5.0)
        )
        self._build_edge_boundaries()

        # ================================================================
        # [配置区 3]：分层融合与预测式边界安全过滤
        # ================================================================
        self.map_x_min = float(param('map_x_min', 0.0))
        self.map_x_max = float(param('map_x_max', 5.0))
        self.map_y_min = float(param('map_y_min', 0.0))
        self.map_y_max = float(param('map_y_max', 2.0))

        # 车体中心到物理边界的最小允许距离。
        self.boundary_center_margin = float(
            param('boundary_center_margin', 0.14)
        )
        self.safe_center_x_min = (
            self.map_x_min + self.boundary_center_margin
        )
        self.safe_center_x_max = (
            self.map_x_max - self.boundary_center_margin
        )
        self.safe_center_y_min = (
            self.map_y_min + self.boundary_center_margin
        )
        self.safe_center_y_max = (
            self.map_y_max - self.boundary_center_margin
        )

        self.boundary_prediction_horizon = float(
            param('boundary_prediction_horizon', 0.80)
        )
        self.boundary_prediction_step = float(
            param('boundary_prediction_step', 0.05)
        )
        self.boundary_near_distance = float(
            param('boundary_near_distance', 0.12)
        )
        # 不再使用跨帧方向锁或空帧方向记忆：
        # 每个真实YOLO帧都根据当前bottom_y最大的最近锥桶重新决定绕行方向；
        # 预测式边界过滤器也在每个控制周期独立选择安全候选。
        # 锥桶出现时Stanley仍持续计算，但只按此比例参与融合。
        # 0.0 = 纯锥桶接管；1.0 = 完整Stanley与锥桶角速度直接相加。
        self.cone_stanley_weight = self.clamp(
            float(param('cone_stanley_weight', 0.40)),
            0.0,
            1.0,
        )

        # 保留现有状态机常量。
        self.cone_clear_frames_required = 1
        self.cone_recovery_duration = 0.0

        self.require_obstacle_frame_before_motion = bool(
            param('require_obstacle_frame_before_motion', True)
        )
        self.stop_on_obstacle_frame_timeout = bool(
            param('stop_on_obstacle_frame_timeout', True)
        )
        self.obstacle_frame_timeout = float(
            param('obstacle_frame_timeout', 0.25)
        )

        # 正常巡线和锥桶避障目标速度。
        self.cone_avoid_speed = float(param('cone_avoid_speed', 1.00))
        self.cone_boundary_speed = float(
            param('cone_boundary_speed', 1.00)
        )
        self.recovery_speed_limit = 1.00
        self.boundary_emergency_speed = float(
            param('boundary_emergency_speed', 0.25)
        )

        # ================================================================
        # [配置区 4]：控制周期、停车和二维码交接
        # ================================================================
        self.control_timer_period = float(
            param('control_timer_period', 0.025)
        )
        self.executor_num_threads = max(
            1, int(param('executor_num_threads', 4))
        )

        self.qr_result_topic = str(
            param('qr_result_topic', '/qr_direction_result')
        )
        self.qr_success_topic = str(
            param('qr_success_topic', '/qr_success')
        )
        self.channel_ack_topic = str(
            param('channel_ack_topic', '/channel_navigation_ack')
        )

        self.fast_stop_burst_count = max(
            1, int(param('fast_stop_burst_count', 4))
        )
        self.fast_stop_hold_sec = float(
            param('fast_stop_hold_sec', 0.35)
        )
        self.fast_stop_timer_period = float(
            param('fast_stop_timer_period', 0.005)
        )

        self.handoff_repeat_period = float(
            param('handoff_repeat_period', 0.5)
        )
        self.handoff_repeat_limit = max(
            1, int(param('handoff_repeat_limit', 20))
        )
        self.channel_handoff_min_x = float(
            param('channel_handoff_min_x', 2.5)
        )

        # 历史轨迹记录参数。记录的是加入地图偏置后的实际位姿：
        # (actual_x, actual_y, yaw)。满足距离或航向变化任一条件时采样，
        # 正式交接前强制补入当前停车点，并通过原 /qr_success 一并交接。
        self.path_record_min_distance = max(
            0.0,
            float(param('path_record_min_distance', 0.04)),
        )
        self.path_record_min_yaw_change = math.radians(max(
            0.0,
            float(param('path_record_min_yaw_change_deg', 3.0)),
        ))
        self.path_record_max_points = max(
            2,
            int(param('path_record_max_points', 3000)),
        )

        # ================================================================
        # [状态变量]
        # ================================================================
        self.cur_pose = [0.0, 0.0, 0.0]
        self.pose_received = False
        self.navigation_arrived = False
        self.is_finished = False

        self.last_cmd_v = 0.0
        self.last_cmd_w = 0.0

        # Stanley 路线状态。
        self.current_path_index = 0
        self.path_first_search = True
        self.path_orientation_checked = False
        self.path_goal_distance = float('inf')
        self.path_min_distance = float('inf')
        self.path_error_y = 0.0
        self.path_error_yaw = 0.0
        self.path_steer_angle = 0.0
        self.path_w = 0.0
        self.path_v = 0.0

        # IMU 角速度用于任务二同款舵机时延预测；未收到时按 0 处理。
        self.imu_w_z = 0.0
        self.imu_w_z_filtered = 0.0
        self.imu_received = False

        # 预测式边界过滤状态。边界不再生成独立角速度。
        self.boundary_active = False
        self.boundary_mode = '安全区'
        self.boundary_min_clearance = float('inf')
        self.boundary_override = False
        self.boundary_last_reason = ''
        self.boundary_selected_w = 0.0
        self.boundary_selected_v = 0.0

        # 锥桶 PD 状态，只在收到新检测帧时更新。
        # 两个检测帧之间持续使用最近一次计算结果。
        self.cone_control_active = False
        self.current_cone_w = 0.0
        self.cone_prev_error = None
        self.cone_prev_time = None
        self.cone_d_filtered = 0.0
        # 最近一帧真实YOLO结果选出的方向；每个新检测帧都会重新计算。
        self.cone_turn_sign = 0.0
        self.last_cone_center_x = 0.0
        self.last_cone_bottom_y = 0.0
        self.cone_recovery_active = False
        self.cone_recovery_start_time = 0.0

        # 每条/racing_obstacle_detection消息都对应一帧真实图像推理结果，
        # 包括targets为空的安全帧。只在40Hz循环中消费新帧并更新避障状态。
        self.latest_obstacle_msg = None
        self.obstacle_frame_received = False
        self.obstacle_frame_sequence = 0
        self.obstacle_frame_consumed_sequence = 0
        self.last_obstacle_frame_time = 0.0
        self.last_consumed_frame_had_hazard = False

        # 二维码交接状态。
        self.qr_result_received = False
        self.handoff_complete = False
        self.qr_result = ''
        self.handoff_publish_count = 0
        self.channel_ack_received = False
        self.channel_ack_data = ''

        # 任务一实际运动轨迹：(actual_x, actual_y, yaw)。
        # handoff_payload 在正式交接时一次性生成，之后定时重发同一份数据，
        # 避免重发期间轨迹内容变化。
        self.recorded_path = []
        self.path_recording_frozen = False
        self.handoff_payload = ''

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
        self.imu_callback_group = MutuallyExclusiveCallbackGroup()

        self.pose_sub = self.create_subscription(
            Pose2D,
            'odom_pose',
            self.pose_cb,
            10,
            callback_group=self.pose_callback_group,
        )
        self.imu_sub = self.create_subscription(
            Imu,
            'imu_data',
            self.imu_cb,
            10,
            callback_group=self.imu_callback_group,
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

        # 40 Hz统一消费最新检测帧并发布一次控制命令。
        self.timer = self.create_timer(
            self.control_timer_period,
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

        track_source = self.track_file_used or '代码内嵌后备路线'
        self.get_logger().info(
            '🚀 任务1融合节点启动：Stanley持续计算 + 锥桶PD修正 + 预测边界裁决；'
            'w>0左转，w<0右转'
        )
        self.get_logger().info(
            f'路线点={len(self.dense_path)}，来源={track_source}，'
            f'V={self.stanley_target_v:.2f}m/s，'
            f'Kstanley={self.k_stanley:.2f}，'
            f'轴距={self.wheelbase:.3f}m，预测时延={self.servo_delay:.2f}s；'
            f'终点前{self.goal_slowdown_distance:.2f}m按距离比例降速；'
            f'目标矩形x=[{self.goal_region_x_min:.2f},{self.goal_region_x_max:.2f}]，'
            f'y=[{self.goal_region_y_min:.2f},{self.goal_region_y_max:.2f}]'
        )
        self.get_logger().info(
            f'锥桶PD=({self.cone_kp:.5f},{self.cone_kd:.5f})，'
            f'锥桶阈值={self.cone_trigger_y:.0f}，'
            f'W范围=[{self.cone_min_abs_w:.2f},{self.cone_w_limit:.2f}]，'
            f'避障Stanley权重={self.cone_stanley_weight:.2f}，'
            f'总Wmax={self.max_w:.2f}'
        )
        self.get_logger().info(
            '📷 帧驱动融合：检测回调只缓存；40Hz统一计算Stanley并发布；'
            '一帧真实无锥桶立即退出；无新帧保持上一状态；'
            f'首帧前停车，检测流超时={self.obstacle_frame_timeout:.2f}s'
        )
        self.get_logger().info(
            '↔️ 无方向锁：每个真实YOLO帧按当前最近锥桶重新选边；'
            '预测边界每个控制周期独立裁决；'
            f'中心死区默认={"左转" if self.cone_default_turn_sign > 0 else "右转"}'
        )
        self.get_logger().info(
            f'🧱 预测边界：物理x=[{self.map_x_min:.2f},{self.map_x_max:.2f}]，'
            f'y=[{self.map_y_min:.2f},{self.map_y_max:.2f}]；'
            f'中心安全范围x=[{self.safe_center_x_min:.2f},{self.safe_center_x_max:.2f}]，'
            f'y=[{self.safe_center_y_min:.2f},{self.safe_center_y_max:.2f}]；'
            f'最小转弯半径={self.minimum_turn_radius:.2f}m，'
            f'预测={self.boundary_prediction_horizon:.2f}s'
        )
        self.get_logger().info(
            f'🔄 二维码交接：允许提前缓存；仅当actual_x>'
            f'{self.channel_handoff_min_x:.2f}m且当前不处于锥桶避障状态时，'
            '才停车并发布 /qr_success'
        )
        self.get_logger().info(
            f'🧠 实际运动轨迹记录已开启：距离采样='
            f'{self.path_record_min_distance:.2f}m，航向采样='
            f'{math.degrees(self.path_record_min_yaw_change):.1f}°，'
            f'最多{self.path_record_max_points}点；'
            '正式交接时通过 /qr_success JSON 一并发送'
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

    def _record_path_pose_locked(self, actual_x, actual_y, yaw, force=False):
        """按距离/航向变化记录实际位姿；force用于交接前补终点。"""
        if self.path_recording_frozen:
            return

        point = (
            float(actual_x),
            float(actual_y),
            float(self.normalize_angle(yaw)),
        )

        if not self.recorded_path:
            # 首点只记录实际收到的地图位姿，不人为插入固定起点。
            self.recorded_path.append(point)
            self.get_logger().info(
                f'🧠 运动轨迹首点=({point[0]:.3f},{point[1]:.3f},'
                f'{math.degrees(point[2]):.1f}°)'
            )
            return

        last_x, last_y, last_yaw = self.recorded_path[-1]
        distance = math.hypot(point[0] - last_x, point[1] - last_y)
        yaw_change = abs(self.normalize_angle(point[2] - last_yaw))

        # 距离和航向变化都较小时不重复记录，降低消息体积。
        if not force and (
            distance < self.path_record_min_distance
            and yaw_change < self.path_record_min_yaw_change
        ):
            return

        # 强制补点时，如果与最后一点几乎一致则直接替换，避免重复终点。
        if force and distance < 0.005 and yaw_change < math.radians(0.5):
            self.recorded_path[-1] = point
        else:
            self.recorded_path.append(point)

        if len(self.recorded_path) > self.path_record_max_points:
            # 保留首尾点，对中间轨迹做2倍降采样，防止String无限增长。
            first = self.recorded_path[0]
            middle = self.recorded_path[1:-1:2]
            last = self.recorded_path[-1]
            self.recorded_path = [first, *middle, last]
            self.get_logger().warn(
                f'⚠️ 轨迹点超过{self.path_record_max_points}，已自动降采样；'
                f'当前点数={len(self.recorded_path)}'
            )

    def _build_handoff_payload_locked(self):
        """使用原 /qr_success String 携带二维码结果和历史轨迹。"""
        path = [
            [round(x, 4), round(y, 4), round(yaw, 5)]
            for x, y, yaw in self.recorded_path
        ]
        return json.dumps(
            {
                'type': 'qr_path_v1',
                'qr_result': self.qr_result,
                'path': path,
            },
            ensure_ascii=False,
            separators=(',', ':'),
        )

    # ------------------------------------------------------------------
    # 路线文件与 Stanley 巡线
    # ------------------------------------------------------------------
    def _candidate_track_paths(self, configured_path):
        candidates = []
        configured_path = str(configured_path or '').strip()
        script_dir = os.path.dirname(os.path.abspath(__file__))

        if configured_path:
            candidates.append(configured_path)
            if not os.path.isabs(configured_path):
                candidates.append(os.path.join(script_dir, configured_path))
                candidates.append(os.path.join(os.getcwd(), configured_path))

        # 兼容下载时自动加“(1)”的文件名。
        for name in ('drawn_lines_4.txt', 'drawn_lines_4(1).txt'):
            candidates.append(os.path.join(script_dir, name))
            candidates.append(os.path.join(os.getcwd(), name))

        unique = []
        seen = set()
        for candidate in candidates:
            absolute = os.path.abspath(os.path.expanduser(candidate))
            if absolute not in seen:
                seen.add(absolute)
                unique.append(absolute)
        return unique

    def load_track_points(self, configured_path):
        """读取 Index,x,y 格式路线；失败时使用代码内嵌同一路线。"""
        for candidate in self._candidate_track_paths(configured_path):
            if not os.path.isfile(candidate):
                continue

            indexed_points = []
            try:
                with open(candidate, 'r', encoding='utf-8') as track_file:
                    for raw_line in track_file:
                        line = raw_line.strip()
                        if not line or line.startswith('#'):
                            continue
                        parts = [part.strip() for part in line.split(',')]
                        if len(parts) < 3:
                            continue
                        try:
                            index = int(parts[0])
                            x = float(parts[1])
                            y = float(parts[2])
                        except (TypeError, ValueError):
                            continue
                        if not (math.isfinite(x) and math.isfinite(y)):
                            continue
                        indexed_points.append((index, x, y))
            except OSError as exc:
                self.get_logger().warning(f'读取路线文件失败 {candidate}: {exc}')
                continue

            indexed_points.sort(key=lambda item: item[0])
            points = self._deduplicate_track_points(
                [(x, y) for _, x, y in indexed_points]
            )
            if len(points) >= 2:
                return points, candidate

            self.get_logger().warning(
                f'路线文件 {candidate} 有效点不足，尝试其他路径'
            )

        fallback = self._deduplicate_track_points(DEFAULT_TRACK_POINTS)
        self.get_logger().warning(
            '⚠️ 未找到可用的 drawn_lines_4.txt，使用代码内嵌后备路线'
        )
        return fallback, ''

    @staticmethod
    def _deduplicate_track_points(points):
        result = []
        for x, y in points:
            point = (float(x), float(y))
            if not result or math.hypot(
                point[0] - result[-1][0],
                point[1] - result[-1][1],
            ) > 1e-5:
                result.append(point)
        return result

    def imu_cb(self, msg):
        with self.motion_lock:
            self.imu_w_z = float(msg.angular_velocity.z)
            if not self.imu_received:
                self.imu_w_z_filtered = self.imu_w_z
            else:
                alpha = self.imu_filter_alpha
                self.imu_w_z_filtered = (
                    alpha * self.imu_w_z
                    + (1.0 - alpha) * self.imu_w_z_filtered
                )
            self.imu_received = True

    def _orient_path_for_start_locked(self, actual_x, actual_y):
        if self.path_orientation_checked or len(self.dense_path) < 2:
            return

        self.path_orientation_checked = True
        first_distance = math.hypot(
            actual_x - self.dense_path[0][0],
            actual_y - self.dense_path[0][1],
        )
        last_distance = math.hypot(
            actual_x - self.dense_path[-1][0],
            actual_y - self.dense_path[-1][1],
        )

        if last_distance + 0.05 < first_distance:
            self.dense_path.reverse()
            self.current_path_index = 0
            self.path_first_search = True
            self.get_logger().warning(
                '🔁 当前车辆更靠近路线末端，已自动反转路线方向'
            )

    def pose_cb(self, msg):
        now = time.monotonic()

        with self.motion_lock:
            first_pose = not self.pose_received
            self.cur_pose = [float(msg.x), float(msg.y), float(msg.theta)]
            self.pose_received = True

            actual_x = self.cur_pose[0] + self.start_offset_x
            actual_y = self.cur_pose[1] + self.start_offset_y
            self._orient_path_for_start_locked(actual_x, actual_y)
            self._update_boundary_status_locked(actual_x, actual_y)

            # 任务一仍掌握控制权时持续记录实际运动位姿。二维码可提前缓存，
            # 但从识别二维码到真正交接之间的运动也必须完整记录。
            if not self.handoff_complete and not self.is_finished:
                self._record_path_pose_locked(
                    actual_x,
                    actual_y,
                    self.cur_pose[2],
                )

            # 二维码可能在车辆到达交接区域前就被识别。
            # 此时只缓存结果并继续沿线巡航；位姿首次越过门槛后才正式交接。
            should_complete_handoff = (
                self.qr_result_received
                and not self.handoff_complete
                and bool(self.qr_result)
                and actual_x > self.channel_handoff_min_x
                and not self.cone_control_active
            )

        if first_pose:
            self.get_logger().info(
                f'📍 位姿已连接，actual=({actual_x:.3f},{actual_y:.3f})'
            )

        if should_complete_handoff:
            self._complete_qr_handoff('位姿更新后达到交接门槛')

    def _compute_stanley_control_locked(self):
        if not self.path_ready or not self.pose_received:
            return None

        actual_x = self.cur_pose[0] + self.start_offset_x
        actual_y = self.cur_pose[1] + self.start_offset_y
        actual_yaw = self.cur_pose[2]
        measured_w = self.imu_w_z_filtered if self.imu_received else 0.0

        # 以路线最后一点的直线距离作为最后一米减速依据。
        goal_x, goal_y = self.dense_path[-1]
        goal_distance = math.hypot(goal_x - actual_x, goal_y - actual_y)
        self.path_goal_distance = goal_distance

        base_command_v = self.clamp(
            self.stanley_target_v,
            0.0,
            self.max_v,
        )
        slowdown_ratio = self.clamp(
            goal_distance / self.goal_slowdown_distance,
            0.0,
            1.0,
        )
        planned_v = base_command_v * slowdown_ratio

        # 前轴中心位置。
        front_x = actual_x + self.imu_to_front * math.cos(actual_yaw)
        front_y = actual_y + self.imu_to_front * math.sin(actual_yaw)

        # 用当前计划线速度预测舵机响应延时后的前轴位置和航向；
        # 最后一米减速时，预测速度也同步降低。
        tau = self.servo_delay
        yaw_pred = self.normalize_angle(actual_yaw + measured_w * tau)
        yaw_mid = self.normalize_angle(actual_yaw + 0.5 * measured_w * tau)
        pred_x = front_x + planned_v * tau * math.cos(yaw_mid)
        pred_y = front_y + planned_v * tau * math.sin(yaw_mid)

        if self.path_first_search:
            search_start = 0
            search_end = len(self.dense_path)
        else:
            search_start = max(0, self.current_path_index - self.stanley_search_back)
            search_end = min(
                len(self.dense_path),
                self.current_path_index + self.stanley_search_forward,
            )

        closest_idx = self.current_path_index
        minimum_distance = float('inf')
        for index in range(search_start, search_end):
            path_x, path_y = self.dense_path[index]
            distance = math.hypot(path_x - pred_x, path_y - pred_y)
            if distance < minimum_distance:
                minimum_distance = distance
                closest_idx = index

        if self.path_first_search:
            if closest_idx > int(len(self.dense_path) * 0.95):
                closest_idx = 0
            self.path_first_search = False

        # 不允许因局部回绕让进度大幅倒退。
        if closest_idx + self.stanley_search_back < self.current_path_index:
            closest_idx = max(0, self.current_path_index - self.stanley_search_back)
        self.current_path_index = closest_idx
        self.path_min_distance = minimum_distance

        in_goal_rectangle = (
            self.goal_region_x_min <= actual_x <= self.goal_region_x_max
            and self.goal_region_y_min <= actual_y <= self.goal_region_y_max
        )
        arrived = (
            closest_idx >= int((len(self.dense_path) - 1) * self.path_end_index_ratio)
            and in_goal_rectangle
        )
        if arrived:
            self.path_v = 0.0
            self.path_w = 0.0
            return {
                'arrived': True,
                'v': 0.0,
                'w': 0.0,
                'closest_idx': closest_idx,
                'goal_distance': goal_distance,
                'min_distance': minimum_distance,
                'actual_x': actual_x,
                'actual_y': actual_y,
                'in_goal_rectangle': True,
                'slowdown_ratio': slowdown_ratio,
            }

        next_idx = min(
            closest_idx + self.stanley_look_ahead_steps,
            len(self.dense_path) - 1,
        )
        if next_idx == closest_idx and closest_idx > 0:
            previous_idx = closest_idx - 1
            path_x, path_y = self.dense_path[previous_idx]
            next_x, next_y = self.dense_path[closest_idx]
        else:
            path_x, path_y = self.dense_path[closest_idx]
            next_x, next_y = self.dense_path[next_idx]

        path_yaw = math.atan2(next_y - path_y, next_x - path_x)
        dx = pred_x - path_x
        dy = pred_y - path_y

        # 与任务二保持相同符号约定：车辆在线上方时 e_y<0，需要右转。
        error_y = dx * math.sin(path_yaw) - dy * math.cos(path_yaw)
        error_yaw = self.normalize_angle(path_yaw - yaw_pred)

        # Stanley横向误差项使用当前计划速度；softening避免接近零速时发散。
        velocity_for_control = max(
            self.stanley_softening_speed,
            abs(planned_v),
        )
        steer_angle = error_yaw + math.atan2(
            self.k_stanley * error_y,
            velocity_for_control,
        )
        steer_angle = self.clamp(
            steer_angle,
            -self.stanley_max_steer,
            self.stanley_max_steer,
        )

        command_w = (
            velocity_for_control / self.wheelbase
        ) * math.tan(steer_angle)
        command_w = self.clamp(
            command_w,
            -self.stanley_w_limit,
            self.stanley_w_limit,
        )

        command_v = planned_v

        self.path_error_y = error_y
        self.path_error_yaw = error_yaw
        self.path_steer_angle = steer_angle
        self.path_w = command_w
        self.path_v = command_v

        return {
            'arrived': False,
            'v': command_v,
            'w': command_w,
            'closest_idx': closest_idx,
            'goal_distance': goal_distance,
            'min_distance': minimum_distance,
            'error_y': error_y,
            'error_yaw': error_yaw,
            'steer_angle': steer_angle,
            'imu_w': measured_w,
            'actual_x': actual_x,
            'actual_y': actual_y,
            'in_goal_rectangle': in_goal_rectangle,
            'slowdown_ratio': slowdown_ratio,
        }

    def _reset_boundary_pd_locked(self):
        """保留旧函数名供停车/交接调用；现在只重置过滤器状态。"""
        self.boundary_active = False
        self.boundary_mode = '安全区'
        self.boundary_min_clearance = float('inf')
        self.boundary_override = False
        self.boundary_last_reason = ''
        self.boundary_selected_w = 0.0
        self.boundary_selected_v = 0.0

    def _boundary_gaps(self, x, y):
        return {
            '左边界': x - self.safe_center_x_min,
            '右边界': self.safe_center_x_max - x,
            '下边界': y - self.safe_center_y_min,
            '上边界': self.safe_center_y_max - y,
        }

    def _update_boundary_status_locked(self, actual_x=None, actual_y=None):
        if actual_x is None or actual_y is None:
            actual_x = self.cur_pose[0] + self.start_offset_x
            actual_y = self.cur_pose[1] + self.start_offset_y

        gaps = self._boundary_gaps(actual_x, actual_y)
        nearest_name, nearest_gap = min(gaps.items(), key=lambda item: item[1])
        self.boundary_min_clearance = nearest_gap
        self.boundary_active = nearest_gap <= self.boundary_near_distance
        self.boundary_mode = (
            f'靠近{nearest_name}' if self.boundary_active else '安全区'
        )
        return nearest_name, nearest_gap

    def _kinematic_w_limit(self, v):
        """保证车体中心轨迹转弯半径不小于0.35m。"""
        speed = abs(float(v))
        if speed <= 1e-6:
            return 0.0
        return min(self.max_w, speed / self.minimum_turn_radius)

    def _predict_center_pose(self, x, y, yaw, v, w, duration):
        if abs(w) <= 1e-6:
            return (
                x + v * duration * math.cos(yaw),
                y + v * duration * math.sin(yaw),
                self.normalize_angle(yaw),
            )

        next_yaw = self.normalize_angle(yaw + w * duration)
        radius = v / w
        next_x = x + radius * (math.sin(yaw + w * duration) - math.sin(yaw))
        next_y = y - radius * (math.cos(yaw + w * duration) - math.cos(yaw))
        return next_x, next_y, next_yaw

    def _evaluate_boundary_command_locked(self, v, w):
        """预测命令是否始终保持车体中心在0.14m安全范围内。"""
        x = self.cur_pose[0] + self.start_offset_x
        y = self.cur_pose[1] + self.start_offset_y
        yaw = self.cur_pose[2]

        current_gaps = self._boundary_gaps(x, y)
        minimum_clearance = min(current_gaps.values())
        if minimum_clearance < 0.0:
            return False, minimum_clearance, '当前中心已越过安全范围'

        steps = max(
            1,
            int(math.ceil(
                self.boundary_prediction_horizon
                / self.boundary_prediction_step
            )),
        )
        for index in range(1, steps + 1):
            duration = min(
                self.boundary_prediction_horizon,
                index * self.boundary_prediction_step,
            )
            px, py, _ = self._predict_center_pose(x, y, yaw, v, w, duration)
            gaps = self._boundary_gaps(px, py)
            nearest_name, clearance = min(gaps.items(), key=lambda item: item[1])
            minimum_clearance = min(minimum_clearance, clearance)
            if clearance < 0.0:
                return (
                    False,
                    minimum_clearance,
                    f'{duration:.2f}s后接近{nearest_name}越过中心安全线',
                )

        return True, minimum_clearance, '预测安全'

    def _candidate_w_values(self, v, desired_w):
        """生成双向角速度候选；每个控制周期独立判断，不跨帧锁方向。"""
        limit = self._kinematic_w_limit(v)
        desired = self.clamp(desired_w, -limit, limit)

        values = [desired]
        for ratio in (0.25, 0.50, 0.75, 1.0):
            magnitude = limit * ratio
            values.extend((magnitude, -magnitude))
        values.append(0.0)

        unique = []
        for value in values:
            value = self.clamp(value, -limit, limit)
            if not any(abs(value - existing) < 1e-6 for existing in unique):
                unique.append(value)
        return unique

    def _select_boundary_safe_command_locked(
        self,
        requested_v,
        desired_w,
        preferred_sign=0,
    ):
        """
        对当前期望V/W进行预测式边界过滤。

        每个控制周期独立测试正、负和零角速度候选，不保留任何跨帧方向锁。
        有锥桶时优先当前真实检测帧选出的preferred_sign；若该方向预测不安全，
        可以在本周期选择反方向。无安全候选时逐级降速，最终停车。
        """
        requested_v = self.clamp(requested_v, 0.0, self.max_v)
        preferred_sign = (
            1 if preferred_sign > 0
            else -1 if preferred_sign < 0
            else 0
        )

        speed_candidates = [
            requested_v,
            requested_v * 0.75,
            requested_v * 0.50,
            min(requested_v * 0.35, self.boundary_emergency_speed),
        ]
        speed_candidates = [
            speed
            for index, speed in enumerate(speed_candidates)
            if speed > 0.05
            and not any(
                abs(speed - old) < 1e-6
                for old in speed_candidates[:index]
            )
        ]

        original_reason = ''
        for speed_index, speed in enumerate(speed_candidates):
            safe_options = []
            desired_limit = self._kinematic_w_limit(speed)
            desired_limited = self.clamp(
                desired_w,
                -desired_limit,
                desired_limit,
            )
            desired_magnitude = abs(desired_limited)

            for candidate_w in self._candidate_w_values(speed, desired_w):
                safe, clearance, reason = self._evaluate_boundary_command_locked(
                    speed,
                    candidate_w,
                )
                if not safe:
                    if not original_reason:
                        original_reason = reason
                    continue

                sign = 0
                if candidate_w > 1e-3:
                    sign = 1
                elif candidate_w < -1e-3:
                    sign = -1

                if preferred_sign:
                    effective_threshold = 0.55 * desired_magnitude

                    # 当前帧首选方向且强度足够：最高优先级。
                    if (
                        sign == preferred_sign
                        and abs(candidate_w) >= effective_threshold
                    ):
                        sign_penalty = 0
                        magnitude_error = abs(
                            abs(candidate_w) - desired_magnitude
                        )
                    # 首选方向不安全时，允许当前周期改用反方向。
                    elif sign == -preferred_sign:
                        sign_penalty = 1
                        magnitude_error = abs(
                            abs(candidate_w) - desired_magnitude
                        )
                    # 首选方向过弱，排在足够强的反方向之后。
                    elif sign == preferred_sign:
                        sign_penalty = 2
                        magnitude_error = abs(
                            abs(candidate_w) - desired_magnitude
                        )
                    else:
                        sign_penalty = 3
                        magnitude_error = desired_magnitude
                else:
                    sign_penalty = 0
                    magnitude_error = abs(candidate_w - desired_limited)

                score = (
                    sign_penalty,
                    magnitude_error,
                    -clearance,
                )
                safe_options.append((score, candidate_w, clearance))

            if safe_options:
                safe_options.sort(key=lambda item: item[0])
                _, selected_w, clearance = safe_options[0]

                override = (
                    speed_index > 0
                    or abs(selected_w - desired_limited) > 1e-5
                )
                self.boundary_override = override
                self.boundary_selected_v = speed
                self.boundary_selected_w = selected_w
                self.boundary_min_clearance = clearance
                self.boundary_last_reason = (
                    original_reason if override else '原候选预测安全'
                )
                self._update_boundary_status_locked()
                return speed, selected_w, override, clearance

        self.boundary_override = True
        self.boundary_selected_v = 0.0
        self.boundary_selected_w = 0.0
        self.boundary_last_reason = (
            original_reason or '所有候选轨迹均不安全'
        )
        self._update_boundary_status_locked()
        return 0.0, 0.0, True, self.boundary_min_clearance

    def _current_stanley_weight_locked(self):
        """返回当前Stanley融合比例。无锥桶时始终为100%。"""
        return self.cone_stanley_weight if self.cone_control_active else 1.0

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
        """立即关闭锥桶控制，并清除最近检测帧的方向与PD历史。"""
        self.cone_control_active = False
        self.current_cone_w = 0.0
        self.cone_prev_error = None
        self.cone_prev_time = None
        self.cone_d_filtered = 0.0
        self.cone_turn_sign = 0.0
        self.last_cone_center_x = 0.0
        self.last_cone_bottom_y = 0.0

        # 不使用渐恢复；无锥桶时Stanley立即恢复100%。
        self.cone_recovery_active = False
        self.cone_recovery_start_time = 0.0

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
        """
        只根据一帧真实YOLO结果更新一次锥桶控制。

        不保留跨帧方向锁：
        - 每帧仍选择bottom_y最大的最近有效锥桶；
        - 每个新真实检测帧都按该锥桶当前center_x重新选择左/右绕行；
        - 两个检测帧之间，40Hz控制循环保持最近一次计算结果；
        - 真实无有效锥桶帧到达时立即清零并恢复100% Stanley。
        """
        candidates = self.collect_valid_obstacle_candidates(msg)

        if not candidates:
            was_active = self.cone_control_active
            old_w = self.current_cone_w
            self._clear_cone_control_locked()

            if was_active:
                self.get_logger().info(
                    f'✅ 真实检测帧无触发锥桶：锥桶控制立即清零，'
                    f'100%恢复Stanley；旧W={old_w:+.3f}',
                    throttle_duration_sec=0.3,
                )
            return False

        # 选择bottom_y最大的锥桶，即当前画面中最靠近车辆的有效锥桶。
        bottom_y, center_x, roi = max(
            candidates,
            key=lambda item: item[0],
        )
        self.cone_recovery_active = False
        self.cone_recovery_start_time = 0.0

        image_center_x = self.image_width / 2.0
        horizontal_offset = center_x - image_center_x

        # 每个真实检测帧重新选方向，不继承上一帧或上一只锥桶：
        # 锥桶在右侧 -> 左转(w>0)
        # 锥桶在左侧 -> 右转(w<0)
        # 中心死区 -> 使用cone_default_turn_sign
        if horizontal_offset > self.cone_center_direction_deadband_px:
            new_turn_sign = 1.0
        elif horizontal_offset < -self.cone_center_direction_deadband_px:
            new_turn_sign = -1.0
        else:
            new_turn_sign = self.cone_default_turn_sign

        # 当前最近锥桶导致方向变化时，清除旧D项历史，
        # 避免上一只锥桶的误差微分污染新方向。
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

        # 符号由当前帧方向决定，绝对值由锥桶靠近图像中心的程度决定。
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

        # D项可以削弱当前帧方向，但不能单独将该帧锥桶控制反向。
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

        direction_text = (
            '左转' if self.cone_turn_sign > 0.0 else '右转'
        )
        change_text = '，本帧方向已重选' if direction_changed else ''
        self.get_logger().info(
            f'🟠 真实帧无锁锥桶PD：bottom_y={bottom_y:.0f}，'
            f'd1={d1:.3f}，center_x={center_x:.1f}px，'
            f'距中心={center_distance:.1f}px，'
            f'中心接近度={center_closeness_px:.1f}px，'
            f'本帧方向={direction_text}{change_text}，'
            f'控制误差={control_error:+.1f}，'
            f'D={control_error_d:+.1f}/s，'
            f'd2=P({d2_p:+.3f})+D({d2_d:+.3f})={d2:+.3f}，'
            f'W原始={cone_w_raw:+.3f}，W锥桶={cone_w:+.3f}，'
            f'conf={roi.confidence:.2f}',
            throttle_duration_sec=0.3,
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

        with self.motion_lock:
            if self.handoff_complete:
                return

            # 当前任务只采用本轮收到的第一个有效二维码结果。
            if self.qr_result_received:
                if result != self.qr_result:
                    self.get_logger().warn(
                        f'⚠️ 已缓存二维码结果“{self.qr_result}”，'
                        f'忽略后续不同结果“{result}”'
                    )
                return

            self.qr_result = result
            self.qr_result_received = True

            actual_x = (
                self.cur_pose[0] + self.start_offset_x
                if self.pose_received
                else None
            )
            avoidance_active = self.cone_control_active
            can_handoff_now = (
                actual_x is not None
                and actual_x > self.channel_handoff_min_x
                and not avoidance_active
            )

        if can_handoff_now:
            self.get_logger().warn(
                f'📩 收到二维码结果：“{result}”；'
                f'当前actual_x={actual_x:.3f}m，已超过'
                f'{self.channel_handoff_min_x:.2f}m，开始正式交接'
            )
            self._complete_qr_handoff('收到二维码时已经达到交接门槛')
            return

        if actual_x is None:
            position_text = '尚未收到位姿'
            wait_reason = '收到位姿并满足交接条件后'
        elif actual_x <= self.channel_handoff_min_x:
            position_text = (
                f'当前actual_x={actual_x:.3f}m，尚未超过'
                f'{self.channel_handoff_min_x:.2f}m'
            )
            wait_reason = '越过交接门槛且避障结束后'
        else:
            position_text = (
                f'当前actual_x={actual_x:.3f}m，已超过交接门槛，'
                '但当前仍处于锥桶避障状态'
            )
            wait_reason = '避障结束后'

        self.get_logger().warn(
            f'📥 已缓存二维码结果：“{result}”；{position_text}。'
            f'任务1继续控制车辆，{wait_reason}再停车并发布 /qr_success'
        )

    def _publish_qr_success(self, reason):
        if not self.qr_result:
            self.get_logger().error('无法发布 qr_success：qr_result 为空')
            return
        if not self.handoff_payload:
            self.get_logger().error('无法发布 qr_success：历史轨迹交接载荷为空')
            return

        success_msg = String()
        success_msg.data = self.handoff_payload
        self.qr_success_pub.publish(success_msg)
        self.handoff_publish_count += 1

        self.get_logger().warn(
            f'📤 /qr_success 第{self.handoff_publish_count}次：'
            f'二维码="{self.qr_result}"，轨迹点={len(self.recorded_path)}，'
            f'载荷={len(self.handoff_payload.encode("utf-8"))}B ({reason})'
        )

    def _complete_qr_handoff(self, reason):
        with self.motion_lock:
            if self.handoff_complete:
                return
            if not self.qr_result_received or not self.qr_result:
                self.get_logger().error('无法完成交接：二维码结果尚未缓存')
                return
            if not self.pose_received:
                self.get_logger().info(
                    '⏳ 二维码结果已缓存，但尚未收到位姿，继续等待交接门槛',
                    throttle_duration_sec=1.0,
                )
                return

            actual_x = self.cur_pose[0] + self.start_offset_x
            if actual_x <= self.channel_handoff_min_x:
                self.get_logger().info(
                    f'⏳ 二维码结果已缓存，等待actual_x>'
                    f'{self.channel_handoff_min_x:.2f}m；'
                    f'当前x={actual_x:.3f}m',
                    throttle_duration_sec=1.0,
                )
                return

            # 最终交接前再次检查避障状态。pose_cb/qr_result_cb中的判断
            # 只是提前筛选；这里的兜底检查可避免多线程回调间状态变化，
            # 确保锥桶避障期间绝不会清零避障状态并发布 /qr_success。
            if self.cone_control_active:
                self.get_logger().info(
                    f'⏳ actual_x={actual_x:.3f}m已满足交接门槛，'
                    '但当前锥桶避障仍在执行；继续由任务1控制，'
                    '避障结束后再交接',
                    throttle_duration_sec=0.5,
                )
                return

            result = self.qr_result

            # 交接前强制补入当前停车位姿，然后冻结轨迹并生成固定载荷。
            # 后续 /qr_success 定时重发始终使用同一份JSON，保证内容一致。
            actual_y = self.cur_pose[1] + self.start_offset_y
            self._record_path_pose_locked(
                actual_x,
                actual_y,
                self.cur_pose[2],
                force=True,
            )
            self.path_recording_frozen = True

            if len(self.recorded_path) < 2:
                self.get_logger().error(
                    f'无法完成交接：历史轨迹点不足，当前仅'
                    f'{len(self.recorded_path)}个'
                )
                self.path_recording_frozen = False
                return

            self.handoff_payload = self._build_handoff_payload_locked()

            # 到达交接门槛后先用一组零速度覆盖最后的巡航命令，
            # 再停止本节点控制并发布 /qr_success。
            self.stop_latched = True
            self._clear_cone_control_locked()
            self._reset_boundary_pd_locked()
            for _ in range(self.fast_stop_burst_count):
                self._publish_zero_locked()

            self.handoff_complete = True
            self.is_finished = True
            self.fast_stop_timer.cancel()

        self._publish_qr_success(reason)
        self.handoff_repeat_timer.reset()

        self.get_logger().warn(
            f'🚦 已在actual_x={actual_x:.3f}m完成二维码交接“{result}”：'
            f'已冻结并交接{len(self.recorded_path)}个实际运动轨迹点；'
            '已先发布零速度，巡航节点停止发布 /cmd_vel；'
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
    # 分层控制：边界预测过滤 > 锥桶避障 > Stanley回线
    # ------------------------------------------------------------------
    def obs_cb(self, msg):
        """只缓存最新真实检测帧；这里不发布/cmd_vel。"""
        with self.motion_lock:
            if self.is_finished or self.handoff_complete:
                return

            self.latest_obstacle_msg = msg
            self.obstacle_frame_sequence += 1
            self.obstacle_frame_received = True
            self.last_obstacle_frame_time = time.monotonic()

    def _consume_latest_obstacle_frame_locked(self):
        """在40Hz循环中至多消费一帧最新YOLO结果。"""
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

    def control_loop(self):
        with self.motion_lock:
            consumed = self._consume_latest_obstacle_frame_locked()
            if consumed:
                source = (
                    f'40Hz/消费真实检测帧#'
                    f'{self.obstacle_frame_consumed_sequence}'
                )
            else:
                source = '40Hz/无新检测帧-保持上一帧状态'
            self._publish_fused_control_locked(source)

    def _publish_fused_control_locked(self, source):
        """
        统一控制结构：

            Stanley持续计算
                    +
            当前真实检测帧的锥桶PD修正
                    ↓
            形成期望V/W
                    ↓
            无方向锁的预测式物理边界过滤
                    ↓
            发布/cmd_vel

        边界预测、双向候选、逐级降速和无解停车全部保留；
        但不再跨帧锁住视觉方向或边界替代方向。
        """
        if self.is_finished or self.handoff_complete:
            return

        if self.stop_latched or self.navigation_arrived:
            self.execute_drive(
                0.0,
                0.0,
                '🏁 已到达路线终点，停车等待二维码/交接',
            )
            return

        if not self.pose_received or not self.path_ready:
            return

        if (
            self.require_obstacle_frame_before_motion
            and not self.obstacle_frame_received
        ):
            self.execute_drive(
                0.0,
                0.0,
                '⏳ 等待/racing_obstacle_detection首帧真实图像结果',
            )
            return

        if (
            self.stop_on_obstacle_frame_timeout
            and self.obstacle_frame_received
        ):
            frame_age = time.monotonic() - self.last_obstacle_frame_time
            if frame_age > self.obstacle_frame_timeout:
                self.execute_drive(
                    0.0,
                    0.0,
                    f'⛔ YOLO检测流超时{frame_age:.2f}s，保持内部状态并停车',
                )
                return

        actual_x = self.cur_pose[0] + self.start_offset_x
        actual_y = self.cur_pose[1] + self.start_offset_y
        _, current_clearance = self._update_boundary_status_locked(
            actual_x,
            actual_y,
        )

        # Stanley在有/无锥桶时都持续计算并更新路径索引。
        stanley = self._compute_stanley_control_locked()
        if stanley is None:
            return

        if stanley['arrived']:
            self.navigation_arrived = True
            self._activate_target_stop_locked()
            self.get_logger().warn(
                f'🏁 进入目标矩形并到达路线末段，'
                f'actual=({stanley["actual_x"]:.3f},{stanley["actual_y"]:.3f})，'
                f'末点距离={stanley["goal_distance"]:.3f}m，'
                f'idx={stanley["closest_idx"]}/{len(self.dense_path)-1}；'
                '已停车，继续等待二维码结果或正式交接'
            )
            return

        if self.cone_control_active:
            preferred_sign = (
                1 if self.cone_turn_sign > 0.0 else -1
            )
            stanley_part = self.cone_stanley_weight * stanley['w']
            desired_w_raw = self.current_cone_w + stanley_part

            # 保留当前帧融合防反向：
            # Stanley可削弱当前锥桶修正，但不能把本帧融合结果反向。
            signed_magnitude = preferred_sign * desired_w_raw
            if signed_magnitude < 0.0:
                desired_w = 0.0
                direction_guarded = True
            else:
                desired_w = desired_w_raw
                direction_guarded = False

            cone_requested_v = (
                self.cone_boundary_speed
                if current_clearance <= self.boundary_near_distance
                else self.cone_avoid_speed
            )
            # 锥桶避障也受最后一米减速约束，不能在末段重新升回避障速度。
            command_v = min(cone_requested_v, stanley['v'])

            selected_v, selected_w, boundary_override, predicted_clearance = (
                self._select_boundary_safe_command_locked(
                    command_v,
                    desired_w,
                    preferred_sign=preferred_sign,
                )
            )

            selected_sign = 0
            if selected_w > 1e-3:
                selected_sign = 1
            elif selected_w < -1e-3:
                selected_sign = -1

            boundary_reversed = (
                selected_sign != 0
                and selected_sign == -preferred_sign
            )

            mode_items = [
                f'无锁锥桶PD+Stanley×{self.cone_stanley_weight:.2f}'
            ]
            if direction_guarded:
                mode_items.append('融合防反向')
            if boundary_override:
                mode_items.append('预测边界改写')
            mode = '+'.join(mode_items)

            boundary_text = (
                f'，边界原因={self.boundary_last_reason}'
                if boundary_override else ''
            )
            reverse_text = (
                '，本周期预测边界改为反向绕行'
                if boundary_reversed else ''
            )

            log_tag = (
                f'🟠 融合避障[{source}/{mode}]：'
                f'检测帧={self.obstacle_frame_consumed_sequence}，'
                f'idx={stanley["closest_idx"]}/{len(self.dense_path)-1}，'
                f'center_x={self.last_cone_center_x:.1f}px，'
                f'bottom_y={self.last_cone_bottom_y:.1f}px，'
                f'W线={stanley["w"]:+.3f}，'
                f'W线参与={stanley_part:+.3f}，'
                f'W锥桶={self.current_cone_w:+.3f}，'
                f'W融合={desired_w:+.3f}，'
                f'V线={stanley["v"]:.3f}，V输出={selected_v:.3f}，'
                f'W输出={selected_w:+.3f}，'
                f'预测最小余量={predicted_clearance:.3f}m'
                f'{boundary_text}{reverse_text}'
            )
            self.execute_drive(selected_v, selected_w, log_tag)
            return

        # 无锥桶：100% Stanley，再经过同一个预测式边界过滤器。
        command_v = stanley['v']
        desired_w = stanley['w']

        selected_v, selected_w, boundary_override, predicted_clearance = (
            self._select_boundary_safe_command_locked(
                command_v,
                desired_w,
                preferred_sign=0,
            )
        )

        mode_items = ['Stanley×1.00']
        if boundary_override:
            mode_items.append('预测边界改写')
        mode = '+'.join(mode_items)

        boundary_text = (
            f'，边界原因={self.boundary_last_reason}'
            if boundary_override else ''
        )
        log_tag = (
            f'📐 巡线控制[{source}/{mode}]：'
            f'idx={stanley["closest_idx"]}/{len(self.dense_path)-1}，'
            f'横差={stanley["error_y"]*100:+.1f}cm，'
            f'航差={math.degrees(stanley["error_yaw"]):+.1f}°，'
            f'V线={stanley["v"]:.3f}，'
            f'W线={stanley["w"]:+.3f}，'
            f'W输出={selected_w:+.3f}，'
            f'预测最小余量={predicted_clearance:.3f}m'
            f'{boundary_text}'
        )
        self.execute_drive(selected_v, selected_w, log_tag)

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
        elif any(
            key in log_tag
            for key in ('分层控制', '避障控制', '巡线控制')
        ):
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
    executor = MultiThreadedExecutor(num_threads=node.executor_num_threads)
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
