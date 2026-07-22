#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
二维码交接后的安全通道导航节点。

地图约束：
- 禁止区域 A：0 <= x <= 2 且 2 <= y <= 2.5
- 禁止区域 B：3 <= x <= 5 且 2 <= y <= 2.5
- 中间通道：2 < x < 3

目标：
- 二维码结果允许提前缓存，但只有车辆实际地图x坐标严格大于2.5m才激活去通道；
- 到达 (2.5, 2.0) 附近；
- 最终航向接近 +90°；
- /qr_success 的 String 消息同时携带二维码结果和任务一记录的历史位姿轨迹；
- 激活后先将历史轨迹倒序，使用低速反向Pure Pursuit沿原路返回；
- 当实际地图x坐标进入大厅区域（默认actual_x<=2.0m）后立即结束倒放；
- 大厅返回完成后不再进行Dubins/Bezier路径规划，而是依次使用固定地图航点PD前往通道；
- /cone_coordinates 话题保留订阅，但不再作为控制或规划依据，避免10~20cm深度误差影响；
- RGB相机YOLO结果读取 /racing_obstacle_detection（ai_msgs/msg/PerceptionTargets）；
- 实时锥桶控制参考task.py：接近度×横向像素PD，与路径角速度融合；
- 一次连续避障期间锁定视觉转向方向；锥桶越靠近画面中心，视觉角速度越大，越靠近两侧越小；
- RGB检测框按 bottom_y 判断距离、按 center_x 判断锥桶位于画面左侧或右侧；
- RGB实时避障方向只依据当前RGB检测框左右位置决定，并在单次避障中锁定；地图边界不得反转视觉方向，只允许同向降速/减小角速度或停车；当前帧无危险锥桶时立即退出避障；
- RGB检测回调直接发布避障命令，优先级高于巡点PD；
- 巡点PD依次跟踪通道下方对准点、通道入口点和最终目标点；
- 每个航点根据目标方位角与指定离开航向进行平滑融合，D项抑制转向过冲；
- 一次避障过程锁定转向方向，避免坐标轻微抖动引起左右反复切换；
- RGB避障结束后直接恢复当前航点PD，不保留旧路径，也不触发重规划；
- 所有速度命令在发布前再经过禁止区域预测安全过滤；
- 新增x=5右边界和y=0下边界保护。
"""

import json
import math
import os
import time
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import rclpy
from ai_msgs.msg import PerceptionTargets
from geometry_msgs.msg import Pose2D, Twist
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String


PathPoint = Tuple[float, float, float]
DubinsResult = Tuple[float, float, float, Tuple[str, str, str]]


class ChannelNavigationNode(Node):
    def __init__(self):
        super().__init__('channel_navigation_node')

        # ==============================================================
        # 1. 地图坐标、目标位姿与车辆运动学
        # ==============================================================
        self.start_offset_x = 0.55
        self.start_offset_y = 0.22

        # 用户要求的目标点。
        self.target_map_x = 2.5
        self.target_map_y = 2.0

        # 路径终点略微进入通道内部，使车辆在经过 (2.5, 2.0)
        # 附近时仍有一个正前方参考点，从而自然保持约 90° 航向。
        self.path_goal_x = 2.5
        self.path_goal_y = 2.05
        self.target_yaw = math.radians(90.0)

        # 到达判据同时检查位置和航向。
        # 最终期望航向为90°，允许左右各30°余量，即60°~120°均可停车。
        self.arrival_position_tolerance = 0.20
        self.arrival_yaw_tolerance = math.radians(30.0)

        # 车辆物理最小转弯半径约为 0.35 m，控制中使用 0.40 m。
        self.safe_turn_radius = 0.40
        self.route_candidate_radii = (0.40, 0.45, 0.50, 0.55)

        # 前进导航速度提升到 0.8~1.0 m/s。
        # 最大角速度按 R = |v / w| >= 0.40 m 同步提升：1.0 / 0.40 = 2.5 rad/s。
        self.max_v = 1.00
        self.max_w = 2.50
        self.min_navigation_v = 0.80

        # 任务一历史轨迹倒序返回参数。
        self.history_reverse_speed = -0.20
        self.history_reverse_lookahead = 0.24
        self.history_reverse_final_lookahead = 0.14
        self.history_reverse_yaw_kp = 0.35
        self.history_reverse_cross_track_stop = 0.60
        self.history_reverse_start_gap_limit = 0.50
        self.history_reverse_min_points = 2
        self.reverse_to_forward_stop_hold = 0.50

        # 大厅只使用x坐标区域判据，不要求倒到固定坐标点。
        # actual_x <= 该值时，立即结束历史轨迹倒放并切换前进去通道。
        self.lobby_x_max = 2.00

        # 大厅返回完成后采用固定航点PD巡航，不再调用Dubins/Bezier规划。
        # 航点格式：(地图x, 地图y, 期望离开航向)。
        # 第一航点位于通道下方，第二航点负责进入中线，最后到达任务目标。
        self.channel_pd_waypoint_template: List[PathPoint] = [
            (2.50, 1.20, math.radians(90.0)),
            (2.50, 1.62, math.radians(90.0)),
            (self.target_map_x, self.target_map_y, self.target_yaw),
        ]
        self.enable_forward_path_planning = False
        self.channel_pd_waypoints: List[PathPoint] = []
        self.channel_pd_waypoint_index = 0
        self.channel_pd_approach_y_min = 1.20
        self.channel_pd_approach_y_max = 1.50
        self.channel_pd_waypoint_tolerance = 0.18
        self.channel_pd_heading_blend_distance = 0.30
        self.channel_pd_kp = 2.40
        self.channel_pd_kd = 0.16
        self.channel_pd_derivative_alpha = 0.35
        self.channel_pd_derivative_limit = 4.0
        self.channel_pd_linear_kp = 1.10
        self.channel_pd_min_v = 0.22
        self.channel_pd_max_v = 0.75
        self.channel_pd_sharp_turn_v = 0.24
        self.channel_pd_prev_error = 0.0
        self.channel_pd_prev_time = 0.0
        self.channel_pd_filtered_derivative = 0.0
        self.channel_pd_active = False

        # 初次左转起步约束与主动姿态恢复参数。
        # 只要求第一段为左转；后续段允许直行、左转或右转。
        # 第一段必须是有效左转：角度不少于1°，左转运动距离不少于0.05m。
        self.minimum_first_left_angle = math.radians(1.0)
        self.minimum_first_left_distance = 0.05
        self.reverse_recovery_active = False
        # 保留变量名以兼容原状态机，但恢复方式不再固定为“右后方倒车”。
        # 新逻辑会在前进/后退、左弧/右弧之间主动选择短程姿态调整动作。
        self.reverse_speed = -0.20
        self.recovery_forward_speed = 0.18
        self.reverse_right_w = 0.45
        self.reverse_replan_interval = 0.20
        self.last_reverse_replan_time = 0.0
        self.reverse_start_pose: Optional[PathPoint] = None

        # 每次恢复动作只保持0.45s，然后重新评价周围空间和当前姿态。
        # 这样不会因为1.5s长预测中的某个远期点接近锥桶而永久停车。
        self.recovery_command_hold_time = 0.45
        self.recovery_command_v = 0.0
        self.recovery_command_w = 0.0
        self.recovery_command_until = 0.0
        self.recovery_command_name = ''

        # 恢复动作的地图安全预测只检查真实禁区、保守区和硬边界。
        # 锥桶地图用于“短程恢复轨迹规划与评分”，不会再触发直接停车。
        self.reverse_safety_prediction_times = tuple(
            0.05 * index for index in range(1, 17)
        )
        self.reverse_min_turn_w = 0.05
        self.last_reverse_block_reason = ''

        # Pure Pursuit 参数。
        self.lookahead_distance = 0.38
        self.final_lookahead_distance = 0.26
        self.route_sample_step = 0.04
        # RGB避障结束后，横向偏离原路径超过0.55m才重新规划；
        # 如果已经回到原路径0.25m以内，则认为无需重规划。
        self.route_replan_cross_track = 0.55
        self.route_rejoin_cross_track = 0.25

        # 通道短距离收尾直接使用目标点PD，不再生成额外收尾路径。
        self.goal_pd_start_distance = 0.85
        self.goal_pd_min_y = 1.55
        self.goal_pd_max_yaw_error = math.radians(70.0)
        self.goal_pd_heading_blend_distance = 0.50
        self.goal_pd_kp = 2.20
        self.goal_pd_kd = 0.16
        self.goal_pd_derivative_limit = 3.0
        self.goal_pd_derivative_alpha = 0.35
        self.goal_pd_min_v = 0.35
        self.goal_pd_max_v = 0.70
        self.goal_pd_prev_error = 0.0
        self.goal_pd_prev_time = 0.0
        self.goal_pd_filtered_derivative = 0.0
        self.goal_pd_active = False

        # ==============================================================
        # 2. 禁止区域与安全余量
        # ==============================================================
        self.forbidden_y_min = 2.0
        self.forbidden_y_max = 2.5
        self.forbidden_left_x_max = 2.0
        self.forbidden_right_x_min = 3.0

        # 新增地图硬边界：x=5右边界、y=0下边界。
        self.boundary_x_max = 5.0
        self.boundary_y_min = 0.0

        # 路径规划和速度预测使用更保守的中心点安全范围。
        # y 接近 2.0 时，车体中心尽量保持在 2.15~2.85。
        self.guard_y_min = 1.90
        self.guard_left_x_max = 2.15
        self.guard_right_x_min = 2.85
        self.guard_x_max = 4.90
        self.guard_y_bottom = 0.10

        # 最高速度提升到 1.0m/s 后，加长预测距离；0.75s 最远约前看0.75m。
        self.safety_prediction_times = (0.10, 0.20, 0.35, 0.50, 0.65, 0.75)

        # ==============================================================
        # 3. 锥桶地图障碍物与总开关
        # ==============================================================
        # 只需要修改这一行：
        # True  = /cone_coordinates参与路径规划、碰撞检查和动态重规划；
        # False = 路径规划不考虑地图锥桶。
        # 深度锥桶坐标存在10~20cm误差：保留话题订阅和数据解析，
        # 但不再将其作为硬障碍参与路径规划。前进阶段近距离避障仍由RGB负责。
        self.enable_cone_map_planning = False

        self.cone_coordinates_topic = '/cone_coordinates'

        # /cone_coordinates 中每个 (x, y) 是锥桶正方形中心。
        # 锥桶在地图上的实际投影为 0.20m × 0.20m，即中心两侧各 0.10m。
        self.cone_square_size = 0.20
        self.cone_half_size = self.cone_square_size / 2.0

        # 路径点代表车辆中心，因此在锥桶真实正方形外再扩张安全余量。
        # 默认总半宽 = 0.10 + 0.20 = 0.30m。
        # 若实车仍容易擦碰可增大；场地过窄无法规划时可适当减小。
        self.cone_path_safety_margin = 0.10 #0.20

        # 安全路径之间长度接近时，优先选择离锥桶更远的路径。
        self.cone_preferred_extra_clearance = 0.10
        self.cone_clearance_cost_weight = 1.0

        # 坐标去重和动态重规划参数。
        self.cone_coordinate_merge_distance = 0.08
        self.cone_coordinate_change_threshold = 0.06
        self.cone_replan_min_interval = 0.60
        self.last_cone_replan_time = 0.0

        self.cone_coordinates: List[Tuple[float, float]] = []
        self.cone_coordinates_received = False
        self.cone_parse_error_count = 0

        # ==============================================================
        # 4. RGB相机实时锥桶避障
        # ==============================================================
        # 路径规划与实时避障使用不同数据源：
        # - /cone_coordinates：仅用于地图候选路径规划与重规划；
        # - /racing_obstacle_detection：唯一的逐帧近距离锥桶避障数据源。
        self.enable_rgb_realtime_avoidance = True
        self.rgb_obstacle_topic = '/racing_obstacle_detection'

        # RGB检测框参数直接参考 task.py 的纯PD锥桶避障。
        self.rgb_image_width = 640.0
        self.rgb_image_height = 480.0
        self.rgb_image_center_x = self.rgb_image_width / 2.0
        self.rgb_confidence_threshold = 0.60

        # 只使用一个固定触发阈值，不再使用280/300双阈值。
        self.rgb_trigger_bottom_y = 290.0

        # d1 = (bottom_y - trigger_y) * near_k，用于表达锥桶接近程度。
        self.rgb_cone_near_k = 0.125
        self.rgb_cone_near_max = 1.50

        # 横向控制与 task.py 保持一致：
        # center_closeness = max(0, 320 - abs(center_x - 320))
        # control_error = locked_turn_sign * center_closeness
        # 因此锥桶越靠近画面中心，视觉角速度越大；越靠近两侧越小。
        # 一次连续避障期间锁定转向方向，防止检测框跨过中心时左右抖动。
        self.avoid_v = 1.00
        self.avoid_turn_radius = 0.45
        self.turn_w_deadband = 0.02
        self.rgb_center_effect_width_px = 320.0
        self.rgb_center_direction_deadband_px = 8.0
        self.rgb_default_turn_sign = -1  # 首次正好位于中心死区时默认右转
        self.rgb_avoid_kp = 0.0040
        self.rgb_avoid_kd = 0.0002
        self.rgb_avoid_w_limit = 1.80
        self.rgb_avoid_derivative_limit = 800.0
        self.rgb_avoid_derivative_alpha = 0.35
        self.rgb_avoid_prev_error = 0.0
        self.rgb_avoid_prev_time = 0.0
        self.rgb_avoid_filtered_derivative = 0.0

        # 参考 task.py 的图像边缘忽略曲线：(bottom_y, 左侧有效边界x)。
        self.rgb_edge_measure_points = [
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
        self.rgb_edge_smoothing_enabled = True
        self.rgb_edge_ignore_margin_px = 5.0
        self._build_rgb_edge_boundaries()

        # 视觉PD与进入避障前的路径跟踪角速度融合：
        # W输出 = W路径基础 + W锥桶。
        # 避障期间基础路径分量保持为进入避障前的值，避免逐帧重复累加。
        self.rgb_base_route_v = 0.0
        self.rgb_base_route_w = 0.0

        # 当前RGB帧无危险时立即退出；退出后保留原路径，
        # 只有偏离超过阈值才从当前位置重新规划。
        self.avoid_clear_frame_count = 0
        self.avoid_replan_count = 0
        # 一次连续避障期间锁定视觉转向方向：+1左转，-1右转。
        # 当前帧无危险锥桶、任务结束或重新激活时清零。
        self.avoid_locked_sign = 0
        self.rgb_avoidance_replan_pending = False

        # RGB检测状态。检测话题应逐帧发布，空targets也算一帧安全结果。
        self.rgb_frame_id = 0
        self.rgb_frame_received = False
        self.rgb_last_frame_time = 0.0
        self.rgb_frame_timeout = 0.50
        self.rgb_require_ready_before_motion = True
        self.rgb_avoid_active = False
        self.avoid_hold_v = 0.0
        self.avoid_hold_w = 0.0
        self.last_avoid_center_x = 0.0
        self.last_avoid_bottom_y = 0.0
        self.last_avoid_confidence = 0.0
        self.last_avoid_target_type = ''

        # ==============================================================
        # 5. 交接与运行状态
        # ==============================================================
        self.qr_success_topic = '/qr_success'
        self.channel_ack_topic = '/channel_navigation_ack'

        # 二维码结果可以提前收到，但只有车辆实际地图x坐标严格大于2.5m
        # 时，才正式激活去通道节点、发布交接ACK并开始输出/cmd_vel。
        self.channel_activation_min_x = 2.5
        self.pending_qr_result = ''
        self.pending_forward_history: List[PathPoint] = []

        # WAIT_HANDOFF -> REVERSE_HISTORY -> FORWARD_PAUSE
        # -> FORWARD_CHANNEL(固定航点PD) -> FINISHED
        self.operation_mode = 'WAIT_HANDOFF'
        self.forward_history: List[PathPoint] = []
        self.reverse_history: List[PathPoint] = []
        self.reverse_history_index = 0
        self.reverse_history_complete = False
        self.forward_stage_ready_time = 0.0

        self.cur_pose = [0.0, 0.0, 0.0]
        self.pose_received = False
        self.channel_active = False
        self.is_finished = False
        self.qr_direction = ''

        self.last_cmd_v = 0.0
        self.last_cmd_w = 0.0

        self.route_points: List[PathPoint] = []
        self.route_index = 0
        self.route_type = ''
        self.route_radius = self.safe_turn_radius
        self.route_length = 0.0
        self.last_route_plan_time = 0.0

        # 当前路径是否仍要求第一段左转。初次规划/倒车恢复保持原规则；
        # 锥桶避障结束后的重规划和侧后方恢复规划允许任意起步方向。
        self.route_require_first_left = True

        # 当前前视点位于车身侧后方时禁止继续 Pure Pursuit。
        # 先从当前位姿按“任意起步方向”重新规划；若仍在侧后方则停车，
        # 避免 alpha 穿越 ±pi 时角速度符号来回翻转。
        self.side_rear_alpha_threshold = math.radians(90.0)
        self.side_rear_replan_interval = 0.60
        self.last_side_rear_replan_time = 0.0

        self.qr_success_callback_count = 0
        self.ack_publish_count = 0
        self.cmd_publish_count = 0

        # ==============================================================
        # 6. ROS 通信
        # ==============================================================
        self.pose_sub = self.create_subscription(
            Pose2D,
            '/odom_pose',
            self.pose_cb,
            10,
        )
        self.cone_coordinates_sub = self.create_subscription(
            String,
            self.cone_coordinates_topic,
            self.cone_coordinates_cb,
            10,
        )

        rgb_qos = QoSProfile(depth=1)
        rgb_qos.reliability = ReliabilityPolicy.BEST_EFFORT
        self.rgb_obstacle_sub = self.create_subscription(
            PerceptionTargets,
            self.rgb_obstacle_topic,
            self.rgb_obstacle_cb,
            rgb_qos,
        )
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        handoff_qos = QoSProfile(depth=1)
        handoff_qos.reliability = ReliabilityPolicy.RELIABLE
        handoff_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

        self.qr_success_sub = self.create_subscription(
            String,
            self.qr_success_topic,
            self.qr_success_cb,
            handoff_qos,
        )
        self.channel_ack_pub = self.create_publisher(
            String,
            self.channel_ack_topic,
            handoff_qos,
        )

        self.timer = self.create_timer(0.025, self.control_loop)
        self.status_timer = self.create_timer(1.5, self.status_loop)

        map_switch_text = '开启' if self.enable_cone_map_planning else '关闭'
        rgb_switch_text = '开启' if self.enable_rgb_realtime_avoidance else '关闭'
        self.get_logger().info(
            '🚪 倒序轨迹去通道节点启动：等待原 /qr_success；'
            '消息中读取二维码结果和任务一历史位姿；'
            'actual_x>2.50m后先以0.20m/s闭环倒车沿历史轨迹返回大厅，'
            '返回后依次使用固定地图航点PD进入通道，不再调用Dubins/Bezier或Pure Pursuit；'
            f'路径规划=关闭，锥桶坐标硬地图规划={map_switch_text}，'
            f'RGB实时避障={rgb_switch_text}；'
            f'{self.cone_coordinates_topic}订阅保留但不参与硬碰撞判定；'
            '所有原订阅和发布话题名称保持不变。'
        )

    # ==================================================================
    # 基础工具
    # ==================================================================
    @staticmethod
    def normalize_angle(angle: float) -> float:
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    @staticmethod
    def mod2pi(angle: float) -> float:
        return angle % (2.0 * math.pi)

    def actual_pose(self) -> Tuple[float, float, float]:
        return (
            self.cur_pose[0] + self.start_offset_x,
            self.cur_pose[1] + self.start_offset_y,
            self.normalize_angle(self.cur_pose[2]),
        )

    def angular_direction_sign(self, w: float) -> int:
        if w > self.turn_w_deadband:
            return 1
        if w < -self.turn_w_deadband:
            return -1
        return 0

    @staticmethod
    def direction_name(direction_sign: int) -> str:
        if direction_sign > 0:
            return '左转'
        if direction_sign < 0:
            return '右转'
        return '直行'

    def rgb_stream_motion_ready(self) -> Tuple[bool, str]:
        """前进运动前检查RGB检测流是否已经就绪且没有过期。"""
        if not self.enable_rgb_realtime_avoidance:
            return True, ''
        if self.rgb_require_ready_before_motion and not self.rgb_frame_received:
            return False, f'⏳ 等待 {self.rgb_obstacle_topic} 首帧，保持停车'
        if self.rgb_frame_received:
            frame_age = time.monotonic() - self.rgb_last_frame_time
            if frame_age > self.rgb_frame_timeout:
                return (
                    False,
                    f'⛔ RGB检测帧超时{frame_age:.2f}s，停止等待新帧',
                )
        return True, ''

    # ==================================================================
    # 锥桶地图坐标与正方形占用判定
    # ==================================================================
    def cone_map_ready(self) -> bool:
        """开启锥桶地图规划时，至少收到一帧坐标消息后才允许开始规划。"""
        return (
            not self.enable_cone_map_planning
            or self.cone_coordinates_received
        )

    def deduplicate_cone_coordinates(
        self,
        coordinates: Sequence[Tuple[float, float]],
    ) -> List[Tuple[float, float]]:
        """合并距离很近的重复锥桶坐标，避免同一锥桶重复形成障碍物。"""
        merged: List[List[float]] = []

        for x, y in coordinates:
            matched_index = None
            for index, existing in enumerate(merged):
                if math.hypot(x - existing[0], y - existing[1]) <= (
                    self.cone_coordinate_merge_distance
                ):
                    matched_index = index
                    break

            if matched_index is None:
                merged.append([x, y, 1.0])
            else:
                item = merged[matched_index]
                count = item[2]
                item[0] = (item[0] * count + x) / (count + 1.0)
                item[1] = (item[1] * count + y) / (count + 1.0)
                item[2] = count + 1.0

        return sorted(
            [(float(item[0]), float(item[1])) for item in merged],
            key=lambda point: (point[0], point[1]),
        )

    def cone_coordinate_sets_changed(
        self,
        old_points: Sequence[Tuple[float, float]],
        new_points: Sequence[Tuple[float, float]],
    ) -> bool:
        """判断锥桶集合是否发生足以影响路径规划的变化。"""
        if len(old_points) != len(new_points):
            return True
        if not old_points and not new_points:
            return False

        unmatched = list(new_points)
        for old_x, old_y in old_points:
            best_index = -1
            best_distance = float('inf')
            for index, (new_x, new_y) in enumerate(unmatched):
                distance = math.hypot(new_x - old_x, new_y - old_y)
                if distance < best_distance:
                    best_distance = distance
                    best_index = index

            if (
                best_index < 0
                or best_distance > self.cone_coordinate_change_threshold
            ):
                return True
            unmatched.pop(best_index)

        return bool(unmatched)

    def is_cone_forbidden_with_margin(
        self,
        x: float,
        y: float,
        extra_margin: float = 0.0,
    ) -> bool:
        """判断车辆中心是否进入任意锥桶的扩张正方形。

        extra_margin 仅用于需要更保守的状态，例如倒车恢复。
        """
        if not self.enable_cone_map_planning:
            return False

        half_extent = (
            self.cone_half_size
            + self.cone_path_safety_margin
            + max(0.0, float(extra_margin))
        )
        for cone_x, cone_y in self.cone_coordinates:
            if (
                abs(x - cone_x) <= half_extent
                and abs(y - cone_y) <= half_extent
            ):
                return True
        return False

    def is_cone_forbidden(self, x: float, y: float) -> bool:
        """
        判断车辆中心是否进入任意锥桶的常规膨胀正方形。

        锥桶真实范围：
            [cone_x-0.10, cone_x+0.10] ×
            [cone_y-0.10, cone_y+0.10]

        路径规划时再向四周扩张 cone_path_safety_margin，等价于把车辆
        中心当作质点，同时为车体宽度和定位误差预留空间。
        """
        return self.is_cone_forbidden_with_margin(x, y, 0.0)

    def cone_clearance_with_margin(
        self,
        x: float,
        y: float,
        extra_margin: float = 0.0,
    ) -> float:
        """返回点到最近锥桶扩张正方形边界的欧氏距离。"""
        if not self.enable_cone_map_planning or not self.cone_coordinates:
            return float('inf')

        half_extent = (
            self.cone_half_size
            + self.cone_path_safety_margin
            + max(0.0, float(extra_margin))
        )
        minimum = float('inf')

        for cone_x, cone_y in self.cone_coordinates:
            dx = max(abs(x - cone_x) - half_extent, 0.0)
            dy = max(abs(y - cone_y) - half_extent, 0.0)
            minimum = min(minimum, math.hypot(dx, dy))

        return minimum

    def cone_clearance(self, x: float, y: float) -> float:
        """返回点到最近锥桶常规膨胀正方形边界的欧氏距离。"""
        return self.cone_clearance_with_margin(x, y, 0.0)

    def route_min_cone_clearance(
        self,
        points: Sequence[PathPoint],
    ) -> float:
        if not self.enable_cone_map_planning or not self.cone_coordinates:
            return float('inf')
        if not points:
            return 0.0
        return min(self.cone_clearance(x, y) for x, y, _ in points)

    # ==================================================================
    # 禁止区域判定
    # ==================================================================
    def is_actual_forbidden(self, x: float, y: float) -> bool:
        # 新增x=5右边界和y=0下边界；到达边界即视为禁止。
        if x >= self.boundary_x_max or y <= self.boundary_y_min:
            return True

        if not (self.forbidden_y_min <= y <= self.forbidden_y_max):
            return False
        return (
            x <= self.forbidden_left_x_max
            or x >= self.forbidden_right_x_min
        )

    def is_guard_forbidden(self, x: float, y: float) -> bool:
        """加入车体、定位误差和地图边界余量后的保守禁止区。"""
        if x >= self.guard_x_max or y <= self.guard_y_bottom:
            return True

        if y < self.guard_y_min:
            return False
        return x <= self.guard_left_x_max or x >= self.guard_right_x_min

    def guard_violation_level(self, x: float, y: float) -> float:
        """返回进入保守区的程度，0表示已处于保守安全区。

        该值只用于倒车恢复。车辆可能因为定位误差或上一阶段运动，
        已经轻微进入 guard 区；此时不能把当前点直接判死，而应允许
        一条持续减小该值、最终退出 guard 区的低速倒车轨迹。
        真实禁止区和 x=5/y=0 硬边界仍然绝不放行。
        """
        violation = 0.0

        if x >= self.guard_x_max:
            violation += x - self.guard_x_max
        if y <= self.guard_y_bottom:
            violation += self.guard_y_bottom - y

        if y >= self.guard_y_min:
            if x <= self.guard_left_x_max:
                # 可通过向下退出 y>=guard_y_min，或向右进入中间通道。
                violation += min(
                    y - self.guard_y_min,
                    self.guard_left_x_max - x,
                )
            elif x >= self.guard_right_x_min:
                # 可通过向下退出 y>=guard_y_min，或向左进入中间通道。
                violation += min(
                    y - self.guard_y_min,
                    x - self.guard_right_x_min,
                )

        return max(0.0, violation)

    def runtime_pose_is_safe(self, x: float, y: float) -> bool:
        """实时速度预测安全判定：只检查地图禁区和硬边界。

        这里明确不读取/cone_coordinates。实时锥桶是否危险完全由
        /racing_obstacle_detection的当前RGB检测帧判断。
        """
        if self.is_guard_forbidden(x, y):
            return False
        return -0.2 <= x < self.guard_x_max and self.guard_y_bottom < y <= 2.55

    def planning_pose_is_safe(self, x: float, y: float) -> bool:
        """路径候选点安全判定：地图禁区 + 锥桶地图。"""
        return self.runtime_pose_is_safe(x, y) and not self.is_cone_forbidden(x, y)

    def guard_forbidden_clearance(self, x: float, y: float) -> float:
        """返回到地图保守禁止区域/硬边界的最近距离。

        该函数供RGB避障方向评分使用，故故意不加入锥桶地图距离。
        锥桶实际坐标只在路径规划函数中通过cone_clearance使用。
        """
        if self.is_guard_forbidden(x, y):
            return 0.0

        vertical_gap = max(0.0, self.guard_y_min - y)
        left_horizontal_gap = max(0.0, x - self.guard_left_x_max)
        left_clearance = math.hypot(left_horizontal_gap, vertical_gap)

        right_horizontal_gap = max(0.0, self.guard_right_x_min - x)
        right_clearance = math.hypot(right_horizontal_gap, vertical_gap)

        x5_clearance = max(0.0, self.guard_x_max - x)
        y0_clearance = max(0.0, y - self.guard_y_bottom)

        return min(
            left_clearance,
            right_clearance,
            x5_clearance,
            y0_clearance,
        )

    def route_is_safe(
        self,
        points: Sequence[PathPoint],
        allow_initial_cone_overlap: bool = False,
    ) -> bool:
        """检查路径是否安全。

        地图禁区和硬边界始终严格检查。锥桶膨胀区默认也严格禁止。
        但车辆当前已经位于锥桶膨胀区时，重新规划不能因为候选路径
        的第一个采样点在膨胀区内就把所有路径判死。此时可设置
        allow_initial_cone_overlap=True，允许路径开头连续位于膨胀区，
        只要路径随后驶出；一旦驶出，后续任何点都不允许再次进入。

        这只用于生成“从膨胀区向外脱离”的路径，不会放宽普通路径
        对其他锥桶膨胀区的碰撞约束。
        """
        if not points:
            return False

        start_inside_cone = self.is_cone_forbidden(points[0][0], points[0][1])
        initial_overlap_active = bool(
            allow_initial_cone_overlap and start_inside_cone
        )
        exited_initial_overlap = not initial_overlap_active

        for x, y, _ in points:
            # 地图真实禁区、保守禁区和硬边界始终不能穿越。
            if not self.runtime_pose_is_safe(x, y):
                return False

            inside_cone = self.is_cone_forbidden(x, y)
            if not inside_cone:
                exited_initial_overlap = True
                continue

            # 只允许路径开头从当前所在的膨胀区向外驶出。
            if initial_overlap_active and not exited_initial_overlap:
                continue

            # 已经驶出后再次进入任意锥桶膨胀区，仍判为不安全。
            return False

        # 如果整条路径始终没有离开当前膨胀区，不能作为脱离路径。
        if initial_overlap_active and not exited_initial_overlap:
            return False

        return True

    def route_starts_with_meaningful_left_turn(
        self,
        modes: Sequence[str],
        lengths: Sequence[float],
        radius: float,
    ) -> bool:
        """第一段必须左转，且角度>=1°、左转弧长>=0.05m。"""
        if not modes or not lengths or modes[0] != 'L':
            return False

        first_left_angle = lengths[0]
        first_left_distance = lengths[0] * radius
        return (
            first_left_angle >= self.minimum_first_left_angle
            and first_left_distance >= self.minimum_first_left_distance
        )

    def compact_motion_segments(
        self,
        modes: Sequence[str],
        lengths: Sequence[float],
        radius: float,
    ) -> Tuple[Tuple[str, ...], Tuple[float, ...]]:
        """
        删除几乎为零的段，并合并相邻同类型段。

        因此Dubins解析结果虽然来自三段公式，实际生成的路径可以是
        一段、两段或三段，不再强制保留三个无意义段。
        """
        compact_modes: List[str] = []
        compact_lengths: List[float] = []

        for mode, normalized_length in zip(modes, lengths):
            if normalized_length * radius < 0.015:
                continue
            if compact_modes and compact_modes[-1] == mode:
                compact_lengths[-1] += normalized_length
            else:
                compact_modes.append(mode)
                compact_lengths.append(normalized_length)

        return tuple(compact_modes), tuple(compact_lengths)

    def path_starts_with_meaningful_left_turn(
        self,
        points: Sequence[PathPoint],
    ) -> bool:
        """用于连续Bezier路径的第一段左转判定。"""
        if len(points) < 3:
            return False

        accumulated_angle = 0.0
        accumulated_distance = 0.0
        turn_started = False
        straight_before_turn = 0.0

        for index in range(1, len(points)):
            previous = points[index - 1]
            current = points[index]
            ds = math.hypot(current[0] - previous[0], current[1] - previous[1])
            delta_yaw = self.normalize_angle(current[2] - previous[2])

            if not turn_started:
                if delta_yaw > math.radians(0.15):
                    turn_started = True
                elif delta_yaw < -math.radians(0.15):
                    return False
                else:
                    straight_before_turn += ds
                    if straight_before_turn > 0.03:
                        return False
                    continue

            if delta_yaw < -math.radians(0.15):
                break

            accumulated_distance += ds
            if delta_yaw > 0.0:
                accumulated_angle += delta_yaw

            if (
                accumulated_angle >= self.minimum_first_left_angle
                and accumulated_distance >= self.minimum_first_left_distance
            ):
                return True

        return False

    @staticmethod
    def sampled_path_length(points: Sequence[PathPoint]) -> float:
        return sum(
            math.hypot(
                points[index][0] - points[index - 1][0],
                points[index][1] - points[index - 1][1],
            )
            for index in range(1, len(points))
        )

    def final_path_heading_penalty(self, points: Sequence[PathPoint]) -> float:
        if len(points) < 3:
            return math.pi
        tail_start = points[max(0, len(points) - 6)]
        tail_end = points[-1]
        tail_heading = math.atan2(
            tail_end[1] - tail_start[1],
            tail_end[0] - tail_start[0],
        )
        return abs(self.normalize_angle(self.target_yaw - tail_heading))

    def sample_bezier_path(
        self,
        start_x: float,
        start_y: float,
        start_yaw: float,
        goal_x: float,
        goal_y: float,
        goal_yaw: float,
        control_start: float,
        control_goal: float,
    ) -> Optional[List[PathPoint]]:
        """生成不受固定段数限制的连续三次Bezier候选路径。"""
        p0 = np.array([start_x, start_y], dtype=float)
        p1 = p0 + control_start * np.array(
            [math.cos(start_yaw), math.sin(start_yaw)], dtype=float
        )
        p3 = np.array([goal_x, goal_y], dtype=float)
        p2 = p3 - control_goal * np.array(
            [math.cos(goal_yaw), math.sin(goal_yaw)], dtype=float
        )

        estimate = (
            np.linalg.norm(p1 - p0)
            + np.linalg.norm(p2 - p1)
            + np.linalg.norm(p3 - p2)
        )
        steps = max(12, int(math.ceil(float(estimate) / self.route_sample_step)))
        points: List[PathPoint] = []
        max_allowed_curvature = 1.0 / self.safe_turn_radius + 0.05

        for index in range(steps + 1):
            t = index / steps
            one_minus_t = 1.0 - t
            position = (
                one_minus_t ** 3 * p0
                + 3.0 * one_minus_t ** 2 * t * p1
                + 3.0 * one_minus_t * t ** 2 * p2
                + t ** 3 * p3
            )
            first = (
                3.0 * one_minus_t ** 2 * (p1 - p0)
                + 6.0 * one_minus_t * t * (p2 - p1)
                + 3.0 * t ** 2 * (p3 - p2)
            )
            second = (
                6.0 * one_minus_t * (p2 - 2.0 * p1 + p0)
                + 6.0 * t * (p3 - 2.0 * p2 + p1)
            )

            speed_sq = float(first[0] ** 2 + first[1] ** 2)
            if speed_sq < 1e-10:
                return None

            curvature = abs(
                float(first[0] * second[1] - first[1] * second[0])
            ) / (speed_sq ** 1.5)
            if curvature > max_allowed_curvature:
                return None

            yaw = math.atan2(float(first[1]), float(first[0]))
            points.append((float(position[0]), float(position[1]), self.normalize_angle(yaw)))

        points[0] = (start_x, start_y, self.normalize_angle(start_yaw))
        points[-1] = (goal_x, goal_y, self.normalize_angle(goal_yaw))
        return points

    def build_bezier_route_candidates(
        self,
        start_x: float,
        start_y: float,
        start_yaw: float,
        goal_x: float,
        goal_y: float,
        goal_yaw: float,
        require_first_left: bool,
    ) -> List[Tuple[List[PathPoint], float, float]]:
        """返回安全的连续曲线候选：(路径, 长度, 末端航向误差)。"""
        distance = math.hypot(goal_x - start_x, goal_y - start_y)
        if distance < 0.25:
            return []

        candidates: List[Tuple[List[PathPoint], float, float]] = []
        scales = (0.25, 0.40, 0.60, 0.85)
        minimum_control = 0.18

        for start_scale in scales:
            for goal_scale in scales:
                points = self.sample_bezier_path(
                    start_x,
                    start_y,
                    start_yaw,
                    goal_x,
                    goal_y,
                    goal_yaw,
                    max(minimum_control, distance * start_scale),
                    max(minimum_control, distance * goal_scale),
                )
                if points is None or not self.route_is_safe(
                    points, allow_initial_cone_overlap=True
                ):
                    continue
                if require_first_left and not self.path_starts_with_meaningful_left_turn(points):
                    continue

                length = self.sampled_path_length(points)
                heading_penalty = self.final_path_heading_penalty(points)
                candidates.append((points, length, heading_penalty))

        return candidates

    # ==================================================================
    # Dubins 路径六种组合
    # ==================================================================
    # Dubins 路径六种组合
    # ==================================================================
    def _dubins_lsl(self, a: float, b: float, d: float) -> Optional[DubinsResult]:
        p2 = (
            2.0 + d * d - 2.0 * math.cos(a - b)
            + 2.0 * d * (math.sin(a) - math.sin(b))
        )
        if p2 < 0.0:
            return None
        tmp = math.atan2(
            math.cos(b) - math.cos(a),
            d + math.sin(a) - math.sin(b),
        )
        return (
            self.mod2pi(-a + tmp),
            math.sqrt(p2),
            self.mod2pi(b - tmp),
            ('L', 'S', 'L'),
        )

    def _dubins_rsr(self, a: float, b: float, d: float) -> Optional[DubinsResult]:
        p2 = (
            2.0 + d * d - 2.0 * math.cos(a - b)
            + 2.0 * d * (-math.sin(a) + math.sin(b))
        )
        if p2 < 0.0:
            return None
        tmp = math.atan2(
            math.cos(a) - math.cos(b),
            d - math.sin(a) + math.sin(b),
        )
        return (
            self.mod2pi(a - tmp),
            math.sqrt(p2),
            self.mod2pi(-b + tmp),
            ('R', 'S', 'R'),
        )

    def _dubins_lsr(self, a: float, b: float, d: float) -> Optional[DubinsResult]:
        p2 = (
            -2.0 + d * d + 2.0 * math.cos(a - b)
            + 2.0 * d * (math.sin(a) + math.sin(b))
        )
        if p2 < 0.0:
            return None
        p = math.sqrt(p2)
        tmp = (
            math.atan2(
                -math.cos(a) - math.cos(b),
                d + math.sin(a) + math.sin(b),
            )
            - math.atan2(-2.0, p)
        )
        return (
            self.mod2pi(-a + tmp),
            p,
            self.mod2pi(-self.mod2pi(b) + tmp),
            ('L', 'S', 'R'),
        )

    def _dubins_rsl(self, a: float, b: float, d: float) -> Optional[DubinsResult]:
        p2 = (
            d * d - 2.0 + 2.0 * math.cos(a - b)
            - 2.0 * d * (math.sin(a) + math.sin(b))
        )
        if p2 < 0.0:
            return None
        p = math.sqrt(p2)
        tmp = (
            math.atan2(
                math.cos(a) + math.cos(b),
                d - math.sin(a) - math.sin(b),
            )
            - math.atan2(2.0, p)
        )
        return (
            self.mod2pi(a - tmp),
            p,
            self.mod2pi(b - tmp),
            ('R', 'S', 'L'),
        )

    def _dubins_rlr(self, a: float, b: float, d: float) -> Optional[DubinsResult]:
        tmp = (
            6.0 - d * d + 2.0 * math.cos(a - b)
            + 2.0 * d * (math.sin(a) - math.sin(b))
        ) / 8.0
        if abs(tmp) > 1.0:
            return None
        p = self.mod2pi(2.0 * math.pi - math.acos(tmp))
        t = self.mod2pi(
            a
            - math.atan2(
                math.cos(a) - math.cos(b),
                d - math.sin(a) + math.sin(b),
            )
            + p / 2.0
        )
        q = self.mod2pi(a - b - t + p)
        return t, p, q, ('R', 'L', 'R')

    def _dubins_lrl(self, a: float, b: float, d: float) -> Optional[DubinsResult]:
        tmp = (
            6.0 - d * d + 2.0 * math.cos(a - b)
            + 2.0 * d * (-math.sin(a) + math.sin(b))
        ) / 8.0
        if abs(tmp) > 1.0:
            return None
        p = self.mod2pi(2.0 * math.pi - math.acos(tmp))
        t = self.mod2pi(
            -a
            - math.atan2(
                math.cos(a) - math.cos(b),
                d + math.sin(a) - math.sin(b),
            )
            + p / 2.0
        )
        q = self.mod2pi(self.mod2pi(b) - a - t + self.mod2pi(p))
        return t, p, q, ('L', 'R', 'L')

    def _sample_dubins_path(
        self,
        start_x: float,
        start_y: float,
        start_yaw: float,
        lengths: Sequence[float],
        modes: Sequence[str],
        radius: float,
    ) -> List[PathPoint]:
        x = start_x
        y = start_y
        yaw = self.normalize_angle(start_yaw)
        points: List[PathPoint] = [(x, y, yaw)]

        for normalized_length, mode in zip(lengths, modes):
            remaining = normalized_length * radius

            while remaining > 1e-9:
                ds = min(self.route_sample_step, remaining)

                if mode == 'S':
                    x += ds * math.cos(yaw)
                    y += ds * math.sin(yaw)
                else:
                    curvature = (1.0 / radius) if mode == 'L' else (-1.0 / radius)
                    delta_yaw = curvature * ds
                    x += (
                        math.sin(yaw + delta_yaw) - math.sin(yaw)
                    ) / curvature
                    y += (
                        -math.cos(yaw + delta_yaw) + math.cos(yaw)
                    ) / curvature
                    yaw = self.normalize_angle(yaw + delta_yaw)

                remaining -= ds
                points.append((x, y, yaw))

        return points


    def reset_goal_pd(self) -> None:
        self.goal_pd_prev_error = 0.0
        self.goal_pd_prev_time = 0.0
        self.goal_pd_filtered_derivative = 0.0
        self.goal_pd_active = False

    def should_use_goal_pd(self) -> bool:
        if not self.pose_received:
            return False

        x, y, yaw = self.actual_pose()
        distance = math.hypot(self.target_map_x - x, self.target_map_y - y)
        yaw_error = abs(self.normalize_angle(self.target_yaw - yaw))
        return (
            self.guard_left_x_max < x < self.guard_right_x_min
            and y >= self.goal_pd_min_y
            and distance <= self.goal_pd_start_distance
            and yaw_error <= self.goal_pd_max_yaw_error
        )

    def perform_goal_pd(self) -> None:
        """在通道内短距离直接对目标点进行PD控制，不再生成收尾路径。"""
        x, y, yaw = self.actual_pose()
        dx = self.target_map_x - x
        dy = self.target_map_y - y
        distance = math.hypot(dx, dy)

        if distance > 0.06:
            target_bearing = math.atan2(dy, dx)
        else:
            target_bearing = self.target_yaw

        blend = max(
            0.0,
            min(1.0, (self.goal_pd_heading_blend_distance - distance)
                / self.goal_pd_heading_blend_distance),
        )
        desired_heading = self.normalize_angle(
            target_bearing
            + blend * self.normalize_angle(self.target_yaw - target_bearing)
        )
        error = self.normalize_angle(desired_heading - yaw)

        now = time.monotonic()
        if self.goal_pd_prev_time <= 0.0:
            derivative = 0.0
        else:
            dt = max(0.02, min(0.25, now - self.goal_pd_prev_time))
            derivative = self.normalize_angle(error - self.goal_pd_prev_error) / dt
            derivative = max(
                -self.goal_pd_derivative_limit,
                min(derivative, self.goal_pd_derivative_limit),
            )

        self.goal_pd_filtered_derivative = (
            self.goal_pd_derivative_alpha * derivative
            + (1.0 - self.goal_pd_derivative_alpha)
            * self.goal_pd_filtered_derivative
        )
        self.goal_pd_prev_error = error
        self.goal_pd_prev_time = now
        self.goal_pd_active = True

        command_w = (
            self.goal_pd_kp * error
            + self.goal_pd_kd * self.goal_pd_filtered_derivative
        )
        command_w = max(-self.max_w, min(command_w, self.max_w))

        distance_scale = max(0.0, min(1.0, distance / self.goal_pd_start_distance))
        command_v = self.goal_pd_min_v + (
            self.goal_pd_max_v - self.goal_pd_min_v
        ) * distance_scale
        heading_scale = max(0.45, 1.0 - abs(error) / math.radians(100.0))
        command_v *= heading_scale
        command_v = max(self.goal_pd_min_v, min(command_v, self.goal_pd_max_v))

        self.execute_drive(
            command_v,
            command_w,
            f'🎯 目标点PD收尾：距离={distance:.2f}m，'
            f'期望航向={math.degrees(desired_heading):.1f}°，'
            f'误差={math.degrees(error):.1f}°，D={self.goal_pd_filtered_derivative:.2f}',
            source='goal_pd',
        )

    def reset_channel_waypoint_pd(self, reset_index: bool = False) -> None:
        """重置固定航点PD的微分状态；切换阶段时可同时回到第一个航点。"""
        if reset_index:
            self.channel_pd_waypoint_index = 0
            self.channel_pd_waypoints = list(self.channel_pd_waypoint_template)
        self.channel_pd_prev_error = 0.0
        self.channel_pd_prev_time = 0.0
        self.channel_pd_filtered_derivative = 0.0
        self.channel_pd_active = False

    def start_channel_waypoint_pd(self) -> None:
        """从大厅区域切换到固定航点PD巡航。"""
        _, current_y, _ = self.actual_pose()
        approach_y = max(
            self.channel_pd_approach_y_min,
            min(current_y, self.channel_pd_approach_y_max),
        )

        # 第一航点的y只在一个固定安全范围内跟随当前大厅y，避免车辆
        # 已经位于较高位置时又先向下追逐(2.50,1.20)。这不是路径规划，
        # 只是从固定巡点模板中调整第一个横向对准点。
        candidates: List[PathPoint] = [
            (2.50, approach_y, self.target_yaw),
            (2.50, 1.62, self.target_yaw),
            (self.target_map_x, self.target_map_y, self.target_yaw),
        ]
        self.channel_pd_waypoints = []
        for waypoint in candidates:
            if self.channel_pd_waypoints:
                previous = self.channel_pd_waypoints[-1]
                if math.hypot(
                    waypoint[0] - previous[0],
                    waypoint[1] - previous[1],
                ) < 0.10:
                    self.channel_pd_waypoints[-1] = waypoint
                    continue
            self.channel_pd_waypoints.append(waypoint)

        self.channel_pd_waypoint_index = 0
        self.reset_channel_waypoint_pd(reset_index=False)
        self.clear_route()
        self.exit_reverse_recovery('切换固定航点PD，关闭旧路径恢复状态')
        self.rgb_avoidance_replan_pending = False
        self.get_logger().warn(
            '🎯 已切换固定航点PD：'
            + ' -> '.join(
                f'({x:.2f},{y:.2f},{math.degrees(yaw):.0f}°)'
                for x, y, yaw in self.channel_pd_waypoints
            )
        )

    def current_channel_waypoint(self) -> Optional[PathPoint]:
        if not self.channel_pd_waypoints:
            return None
        index = min(
            self.channel_pd_waypoint_index,
            len(self.channel_pd_waypoints) - 1,
        )
        return self.channel_pd_waypoints[index]

    def advance_channel_waypoint_if_reached(self) -> bool:
        """中间航点到达后切换下一点；最终点由统一到达判据负责停车。"""
        waypoint = self.current_channel_waypoint()
        if waypoint is None:
            return False

        x, y, _ = self.actual_pose()
        target_x, target_y, _ = waypoint
        distance = math.hypot(target_x - x, target_y - y)
        last_index = len(self.channel_pd_waypoints) - 1

        if (
            self.channel_pd_waypoint_index < last_index
            and distance <= self.channel_pd_waypoint_tolerance
        ):
            old_index = self.channel_pd_waypoint_index
            self.channel_pd_waypoint_index += 1
            self.reset_channel_waypoint_pd(reset_index=False)
            next_x, next_y, _ = self.channel_pd_waypoints[
                self.channel_pd_waypoint_index
            ]
            self.get_logger().warn(
                f'✅ 巡点PD航点{old_index + 1}到达：距离={distance:.3f}m；'
                f'切换航点{self.channel_pd_waypoint_index + 1}='
                f'({next_x:.2f},{next_y:.2f})'
            )
            return True
        return False

    def perform_channel_waypoint_pd(self) -> None:
        """不生成路径，直接按固定地图航点执行位置/航向PD控制。"""
        if self.operation_mode != 'FORWARD_CHANNEL':
            return
        if not self.channel_active or self.is_finished or not self.pose_received:
            return

        if self.check_channel_arrival():
            return

        rgb_ready, rgb_reason = self.rgb_stream_motion_ready()
        if not rgb_ready:
            self.publish_stop(rgb_reason)
            return

        if not self.channel_pd_waypoints:
            self.start_channel_waypoint_pd()

        # 一次控制周期最多跳过一个已经到达的中间点，随后立即控制下一点。
        self.advance_channel_waypoint_if_reached()
        waypoint = self.current_channel_waypoint()
        if waypoint is None:
            self.publish_stop('⛔ 固定航点列表为空，禁止前进')
            return

        x, y, yaw = self.actual_pose()
        target_x, target_y, exit_yaw = waypoint
        dx = target_x - x
        dy = target_y - y
        distance = math.hypot(dx, dy)

        if distance > 0.06:
            target_bearing = math.atan2(dy, dx)
        else:
            target_bearing = exit_yaw

        # 接近航点时逐渐从“指向航点”过渡到该点的离开航向，
        # 让阿克曼小车以连续弧线进入下一段，而不是到点后突然转向。
        blend = max(
            0.0,
            min(
                1.0,
                (self.channel_pd_heading_blend_distance - distance)
                / self.channel_pd_heading_blend_distance,
            ),
        )
        desired_heading = self.normalize_angle(
            target_bearing
            + blend * self.normalize_angle(exit_yaw - target_bearing)
        )
        error = self.normalize_angle(desired_heading - yaw)

        now = time.monotonic()
        if self.channel_pd_prev_time <= 0.0:
            derivative = 0.0
        else:
            dt = max(0.02, min(0.25, now - self.channel_pd_prev_time))
            derivative = self.normalize_angle(
                error - self.channel_pd_prev_error
            ) / dt
            derivative = max(
                -self.channel_pd_derivative_limit,
                min(derivative, self.channel_pd_derivative_limit),
            )

        self.channel_pd_filtered_derivative = (
            self.channel_pd_derivative_alpha * derivative
            + (1.0 - self.channel_pd_derivative_alpha)
            * self.channel_pd_filtered_derivative
        )
        self.channel_pd_prev_error = error
        self.channel_pd_prev_time = now
        self.channel_pd_active = True

        command_w = (
            self.channel_pd_kp * error
            + self.channel_pd_kd * self.channel_pd_filtered_derivative
        )
        command_w = max(-self.max_w, min(command_w, self.max_w))

        command_v = min(
            self.channel_pd_max_v,
            max(self.channel_pd_min_v, self.channel_pd_linear_kp * distance),
        )
        heading_abs = abs(error)
        if heading_abs >= math.radians(70.0):
            command_v = min(command_v, self.channel_pd_sharp_turn_v)
        else:
            heading_scale = max(
                0.45,
                1.0 - heading_abs / math.radians(110.0),
            )
            command_v *= heading_scale
            command_v = max(self.channel_pd_min_v, command_v)

        # 接近最后目标点时继续降低速度，提高最终位置和90度航向精度。
        last_index = len(self.channel_pd_waypoints) - 1
        if self.channel_pd_waypoint_index == last_index:
            if distance < 0.55:
                command_v = min(command_v, 0.50)
            if distance < 0.30:
                command_v = min(command_v, 0.32)

        self.execute_drive(
            command_v,
            command_w,
            f'🎯 固定航点PD[{self.channel_pd_waypoint_index + 1}/'
            f'{len(self.channel_pd_waypoints)}]：目标=({target_x:.2f},{target_y:.2f})，'
            f'距离={distance:.2f}m，期望航向={math.degrees(desired_heading):.1f}°，'
            f'角度误差={math.degrees(error):.1f}°，'
            f'D={self.channel_pd_filtered_derivative:.2f}',
            source='waypoint_pd',
        )

    def clear_route(self) -> None:
        self.route_points = []
        self.route_index = 0
        self.route_type = ''
        self.route_length = 0.0

    def enter_reverse_recovery(self, reason: str) -> None:
        if not self.reverse_recovery_active:
            self.reverse_recovery_active = True
            self.reverse_start_pose = self.actual_pose()
            self.last_reverse_replan_time = 0.0
            self.recovery_command_v = 0.0
            self.recovery_command_w = 0.0
            self.recovery_command_until = 0.0
            self.recovery_command_name = ''
            self.get_logger().warn(
                f'🔄 当前位姿没有安全直达路径，进入主动姿态恢复：{reason}'
            )
        self.clear_route()
        self.rgb_avoid_active = False
        self.rgb_avoidance_replan_pending = False
        self.avoid_clear_frame_count = 0
        self.avoid_hold_v = 0.0
        self.avoid_hold_w = 0.0
        self.avoid_locked_sign = 0
        self.rgb_base_route_v = 0.0
        self.rgb_base_route_w = 0.0
        self.reset_avoid_pd()
        self.reset_goal_pd()

    def exit_reverse_recovery(self, reason: str) -> None:
        if self.reverse_recovery_active:
            self.get_logger().warn(f'✅ 退出主动姿态恢复：{reason}')
        self.reverse_recovery_active = False
        self.reverse_start_pose = None
        self.last_reverse_replan_time = 0.0
        self.recovery_command_v = 0.0
        self.recovery_command_w = 0.0
        self.recovery_command_until = 0.0
        self.recovery_command_name = ''

    def reverse_distance(self) -> float:
        if self.reverse_start_pose is None:
            return 0.0
        x, y, _ = self.actual_pose()
        return math.hypot(x - self.reverse_start_pose[0], y - self.reverse_start_pose[1])

    def recovery_command_score(
        self,
        v: float,
        w: float,
    ) -> Optional[Tuple[float, float, float, float, float]]:
        """评价一条短程姿态恢复动作。

        锥桶坐标在这里作为局部轨迹规划代价使用：优先选择能增大锥桶
        间隙的动作，但不会因为某个预测点进入锥桶膨胀区而直接停车。
        地图真实禁区、保守区和x=5/y=0硬边界仍然是硬约束。
        """
        if not self.reverse_command_is_safe(v, w):
            return None

        x, y, yaw = self.actual_pose()
        current_cone_clearance = self.cone_clearance(x, y)
        if not math.isfinite(current_cone_clearance):
            current_cone_clearance = 2.0

        minimum_cone_clearance = float('inf')
        final_x = x
        final_y = y
        final_yaw = yaw

        for duration in self.reverse_safety_prediction_times:
            px, py, pyaw = self.predict_pose(x, y, yaw, v, w, duration)
            clearance = self.cone_clearance(px, py)
            if not math.isfinite(clearance):
                clearance = 2.0
            minimum_cone_clearance = min(minimum_cone_clearance, clearance)
            final_x, final_y, final_yaw = px, py, pyaw

        final_cone_clearance = self.cone_clearance(final_x, final_y)
        if not math.isfinite(final_cone_clearance):
            final_cone_clearance = 2.0
        if not math.isfinite(minimum_cone_clearance):
            minimum_cone_clearance = final_cone_clearance

        goal_distance = math.hypot(
            self.path_goal_x - final_x,
            self.path_goal_y - final_y,
        )
        goal_bearing = math.atan2(
            self.path_goal_y - final_y,
            self.path_goal_x - final_x,
        )
        heading_error = abs(self.normalize_angle(goal_bearing - final_yaw))

        clearance_gain = final_cone_clearance - current_cone_clearance
        clearance_loss = max(
            0.0,
            current_cone_clearance - minimum_cone_clearance,
        )

        # 优先后退调整，但如果后方空间差，会自然选择前进弧线。
        forward_penalty = 0.12 if v > 0.0 else 0.0
        straight_penalty = 0.08 if abs(w) < 0.05 else 0.0
        command_change_penalty = (
            0.12 * abs(v - self.recovery_command_v)
            + 0.05 * abs(w - self.recovery_command_w)
        )

        cost = (
            0.45 * goal_distance
            + 0.18 * heading_error
            + 2.8 * clearance_loss
            - 3.6 * clearance_gain
            - 0.35 * final_cone_clearance
            + forward_penalty
            + straight_penalty
            + command_change_penalty
        )
        return (
            cost,
            minimum_cone_clearance,
            final_cone_clearance,
            goal_distance,
            heading_error,
        )

    def select_pose_recovery_command(
        self,
    ) -> Optional[Tuple[float, float, str, Tuple[float, ...]]]:
        """同时评价全部短程运动原语，并选择最有利于重新规划的一条。

        primitives中的排列只用于保持动作集合可读，不代表依次执行顺序。
        每次动作保持结束后都会重新评价全部动作并择优。
        """
        reverse_v = self.reverse_speed
        forward_v = self.recovery_forward_speed
        reverse_max_w = min(
            self.max_w,
            abs(reverse_v) / max(self.safe_turn_radius, 1e-6),
        )
        forward_max_w = min(
            self.max_w,
            abs(forward_v) / max(self.safe_turn_radius, 1e-6),
        )

        primitives = [
            (reverse_v, +reverse_max_w, '后退左弧'),
            (reverse_v, -reverse_max_w, '后退右弧'),
            (reverse_v, +0.55 * reverse_max_w, '后退小左弧'),
            (reverse_v, -0.55 * reverse_max_w, '后退小右弧'),
            (reverse_v, 0.0, '直线后退'),
            (forward_v, +forward_max_w, '前进左弧'),
            (forward_v, -forward_max_w, '前进右弧'),
            (forward_v, +0.55 * forward_max_w, '前进小左弧'),
            (forward_v, -0.55 * forward_max_w, '前进小右弧'),
            (forward_v, 0.0, '低速前进'),
        ]

        candidates = []
        for v, w, name in primitives:
            metrics = self.recovery_command_score(v, w)
            if metrics is None:
                continue
            candidates.append((metrics[0], v, w, name, metrics))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0])
        _, v, w, name, metrics = candidates[0]
        return float(v), float(w), name, metrics

    def perform_reverse_recovery(self) -> None:
        """通过多方向短程机动主动改变位姿，直到能够重新规划。"""
        if not self.reverse_recovery_active:
            return

        if not self.cone_map_ready():
            self.publish_stop(
                f'⏳ 等待 {self.cone_coordinates_topic} 首帧坐标，'
                '暂不执行姿态恢复'
            )
            return

        now = time.monotonic()

        # 恢复阶段继续强制首段左转，但有效左转门槛已降低为：
        # 左转角>=1°且左转弧长>=0.05m。只有满足该条件的安全路径
        # 出现后，才结束姿态调整并开始跟踪。
        if now - self.last_reverse_replan_time >= self.reverse_replan_interval:
            self.last_reverse_replan_time = now
            if self.plan_safe_route(
                '主动姿态恢复过程中重新规划',
                require_first_left=True,
            ):
                self.exit_reverse_recovery('当前位姿已经找到首段左转的安全路径')
                self.perform_route_following()
                return

        # 当前短程动作执行完后，重新从全部前进/后退原语中选择。
        if now >= self.recovery_command_until:
            selected = self.select_pose_recovery_command()
            if selected is None:
                block_reason = (
                    self.last_reverse_block_reason
                    or '所有前进/后退姿态调整动作都会进入地图禁区或硬边界'
                )
                self.publish_stop(
                    '⛔ 主动姿态恢复无地图安全运动原语；'
                    f'原因={block_reason}。这里只因地图硬约束停车，不因锥桶膨胀区停车'
                )
                return

            (
                self.recovery_command_v,
                self.recovery_command_w,
                self.recovery_command_name,
                metrics,
            ) = selected
            self.recovery_command_until = now + self.recovery_command_hold_time
            (
                _cost,
                min_clearance,
                final_clearance,
                goal_distance,
                heading_error,
            ) = metrics
            self.get_logger().warn(
                f'🧩 选择姿态恢复动作={self.recovery_command_name}，'
                f'预测锥桶最小余量={min_clearance:.2f}m，'
                f'末端余量={final_clearance:.2f}m，'
                f'末端目标距离={goal_distance:.2f}m，'
                f'目标方向误差={math.degrees(heading_error):.1f}°'
            )

        distance = self.reverse_distance()
        self.execute_drive(
            self.recovery_command_v,
            self.recovery_command_w,
            f'🔄 主动姿态恢复[{self.recovery_command_name}]：'
            f'累计位移={distance:.2f}m，动作剩余='
            f'{max(0.0, self.recovery_command_until-now):.2f}s',
            source='recovery',
        )

    def plan_safe_route(
        self,
        reason: str,
        require_first_left: bool = True,
    ) -> bool:
        if not self.enable_forward_path_planning:
            self.get_logger().warn(
                f'🚫 路径规划已禁用[{reason}]：前进阶段只允许固定航点PD',
                throttle_duration_sec=1.0,
            )
            return False
        if not self.pose_received:
            return False

        if not self.cone_map_ready():
            self.get_logger().warn(
                f'⏳ 锥桶避障已开启，但尚未收到 '
                f'{self.cone_coordinates_topic}；暂不生成路径',
                throttle_duration_sec=1.0,
            )
            return False

        sx, sy, syaw = self.actual_pose()
        gx = self.path_goal_x
        gy = self.path_goal_y
        gyaw = self.target_yaw

        if self.is_actual_forbidden(sx, sy):
            self.get_logger().error(
                f'❌ 当前车体中心已经位于禁止区域：({sx:.3f}, {sy:.3f})，'
                '为避免继续扩大风险，不生成前进路径。'
            )
            return False

        planners: Sequence[Tuple[str, Callable[[float, float, float], Optional[DubinsResult]]]] = (
            ('LSL', self._dubins_lsl),
            ('RSR', self._dubins_rsr),
            ('LSR', self._dubins_lsr),
            ('RSL', self._dubins_rsl),
            ('RLR', self._dubins_rlr),
            ('LRL', self._dubins_lrl),
        )

        safe_candidates = []

        # 候选一：Dubins公式，但删除零长度段并合并同类段，实际段数可为1~3段。
        for radius in self.route_candidate_radii:
            dx = gx - sx
            dy = gy - sy
            distance = math.hypot(dx, dy)
            normalized_distance = distance / radius
            theta = self.mod2pi(math.atan2(dy, dx))
            alpha = self.mod2pi(syaw - theta)
            beta = self.mod2pi(gyaw - theta)

            for original_name, planner in planners:
                result = planner(alpha, beta, normalized_distance)
                if result is None:
                    continue

                t, p, q, original_modes = result
                modes, lengths = self.compact_motion_segments(
                    original_modes,
                    (t, p, q),
                    radius,
                )
                if not modes:
                    continue

                if require_first_left and not self.route_starts_with_meaningful_left_turn(
                    modes,
                    lengths,
                    radius,
                ):
                    continue

                points = self._sample_dubins_path(
                    sx,
                    sy,
                    syaw,
                    lengths,
                    modes,
                    radius,
                )
                if not self.route_is_safe(
                    points, allow_initial_cone_overlap=True
                ):
                    continue

                physical_length = sum(lengths) * radius
                final_heading_penalty = self.final_path_heading_penalty(points)
                minimum_cone_clearance = self.route_min_cone_clearance(points)
                cone_clearance_penalty = 0.0
                if math.isfinite(minimum_cone_clearance):
                    cone_clearance_penalty = (
                        self.cone_clearance_cost_weight
                        * max(
                            0.0,
                            self.cone_preferred_extra_clearance
                            - minimum_cone_clearance,
                        )
                    )
                selection_cost = (
                    physical_length
                    + 0.20 * final_heading_penalty
                    + cone_clearance_penalty
                )
                compact_name = ''.join(modes)
                safe_candidates.append(
                    (
                        selection_cost,
                        physical_length,
                        radius,
                        compact_name,
                        modes,
                        points,
                        final_heading_penalty,
                        f'Dubins原型={original_name}',
                    )
                )

        # 候选二：连续Bezier路径，不受一/二/三段结构限制。
        for points, length, final_heading_penalty in self.build_bezier_route_candidates(
            sx,
            sy,
            syaw,
            gx,
            gy,
            gyaw,
            require_first_left,
        ):
            minimum_cone_clearance = self.route_min_cone_clearance(points)
            cone_clearance_penalty = 0.0
            if math.isfinite(minimum_cone_clearance):
                cone_clearance_penalty = (
                    self.cone_clearance_cost_weight
                    * max(
                        0.0,
                        self.cone_preferred_extra_clearance
                        - minimum_cone_clearance,
                    )
                )
            selection_cost = (
                length
                + 0.20 * final_heading_penalty
                + cone_clearance_penalty
            )
            safe_candidates.append(
                (
                    selection_cost,
                    length,
                    self.safe_turn_radius,
                    'BEZIER',
                    ('C',),
                    points,
                    final_heading_penalty,
                    '连续曲线，非固定三段',
                )
            )

        if not safe_candidates:
            self.clear_route()
            start_rule = (
                f'首段左转角>={math.degrees(self.minimum_first_left_angle):.0f}°'
                f'且距离>={self.minimum_first_left_distance:.2f}m'
                if require_first_left
                else '任意起步方向'
            )
            cone_note = (
                f'，当前锥桶={len(self.cone_coordinates)}个'
                if self.enable_cone_map_planning
                else '，锥桶避障已关闭'
            )
            self.get_logger().warn(
                f'❌ 没有找到满足“{start_rule}”且避开禁止区域/锥桶的路径；'
                f'原因={reason}{cone_note}。'
            )
            return False

        safe_candidates.sort(key=lambda item: item[0])
        (
            _,
            length,
            radius,
            route_name,
            modes,
            points,
            final_heading_penalty,
            route_detail,
        ) = safe_candidates[0]

        self.route_points = points
        self.route_index = 0
        self.route_type = route_name
        self.route_radius = radius
        self.route_length = length
        self.last_route_plan_time = time.monotonic()
        self.route_require_first_left = require_first_left
        self.reset_goal_pd()

        start_rule = (
            (
                f'首段L：角度>={math.degrees(self.minimum_first_left_angle):.0f}°'
                f'且距离>={self.minimum_first_left_distance:.2f}m'
            )
            if require_first_left
            else '第一段方向不限'
        )
        selected_cone_clearance = self.route_min_cone_clearance(points)
        if math.isfinite(selected_cone_clearance):
            cone_route_note = (
                f'，锥桶={len(self.cone_coordinates)}个，'
                f'膨胀正方形外最小余量={selected_cone_clearance:.2f}m'
            )
        else:
            cone_route_note = (
                '，无锥桶'
                if self.enable_cone_map_planning
                else '，锥桶避障关闭'
            )

        self.get_logger().warn(
            f'🗺️ 安全路径已生成[{reason}]：类型={route_name}，'
            f'结构={"".join(modes)}（{start_rule}），{route_detail}，'
            f'参考半径={radius:.2f}m，长度={length:.2f}m，'
            f'采样点={len(points)}，'
            f'末端几何航向误差={math.degrees(final_heading_penalty):.1f}°'
            f'{cone_route_note}；'
            f'终点=({self.path_goal_x:.2f},{self.path_goal_y:.2f},90°)'
        )
        return True

    # ==================================================================
    # 任务一历史轨迹解析与倒序闭环跟踪
    # ==================================================================
    def parse_qr_handoff_payload(
        self,
        raw_text: str,
    ) -> Tuple[str, List[PathPoint], str]:
        """兼容新JSON载荷；旧纯字符串只解析二维码，不允许无轨迹启动倒车。"""
        text = str(raw_text).strip()
        if not text:
            return '', [], '空消息'

        try:
            payload = json.loads(text)
        except (json.JSONDecodeError, TypeError, ValueError):
            return text, [], '旧格式纯二维码字符串，未携带历史轨迹'

        if not isinstance(payload, dict):
            return '', [], 'JSON顶层不是对象'

        result = str(payload.get('qr_result', '')).strip()
        raw_path = payload.get('path', [])
        if not result:
            return '', [], 'JSON中qr_result为空'
        if not isinstance(raw_path, list):
            return result, [], 'JSON中path不是数组'

        cleaned: List[PathPoint] = []
        invalid = 0
        for item in raw_path:
            if not isinstance(item, (list, tuple)) or len(item) < 3:
                invalid += 1
                continue
            try:
                x = float(item[0])
                y = float(item[1])
                yaw = self.normalize_angle(float(item[2]))
            except (TypeError, ValueError):
                invalid += 1
                continue
            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(yaw)):
                invalid += 1
                continue

            point = (x, y, yaw)
            if cleaned:
                lx, ly, lyaw = cleaned[-1]
                if (
                    math.hypot(x - lx, y - ly) < 0.01
                    and abs(self.normalize_angle(yaw - lyaw)) < math.radians(1.0)
                ):
                    cleaned[-1] = point
                    continue
            cleaned.append(point)

        note = f'有效轨迹点={len(cleaned)}，无效点={invalid}'
        return result, cleaned, note

    def nearest_reverse_history_info(self) -> Tuple[int, float]:
        if not self.reverse_history:
            return 0, float('inf')

        x, y, _ = self.actual_pose()
        start = max(0, self.reverse_history_index - 6)
        end = min(len(self.reverse_history), self.reverse_history_index + 100)
        best_index = self.reverse_history_index
        best_distance = float('inf')

        for index in range(start, end):
            px, py, _ = self.reverse_history[index]
            distance = math.hypot(px - x, py - y)
            if distance < best_distance:
                best_distance = distance
                best_index = index

        self.reverse_history_index = max(self.reverse_history_index, best_index)
        return self.reverse_history_index, best_distance

    def reverse_history_lookahead_point(self, lookahead: float) -> PathPoint:
        if not self.reverse_history:
            return self.actual_pose()

        index, _ = self.nearest_reverse_history_info()
        accumulated = 0.0
        previous = self.reverse_history[index]
        for next_index in range(index + 1, len(self.reverse_history)):
            current = self.reverse_history[next_index]
            accumulated += math.hypot(
                current[0] - previous[0],
                current[1] - previous[1],
            )
            if accumulated >= lookahead:
                return current
            previous = current
        return self.reverse_history[-1]

    def remaining_reverse_history_distance(self) -> float:
        if not self.reverse_history:
            return float('inf')
        index = min(self.reverse_history_index, len(self.reverse_history) - 1)
        total = 0.0
        previous = self.reverse_history[index]
        for current in self.reverse_history[index + 1:]:
            total += math.hypot(current[0] - previous[0], current[1] - previous[1])
            previous = current
        return total

    def finish_reverse_history(self) -> None:
        x, y, _ = self.actual_pose()
        self.publish_stop(
            f'🏛️ actual_x={x:.3f}m 已进入大厅区域'
            f'(x<={self.lobby_x_max:.2f}m)，当前位置y={y:.3f}m；'
            '停止倒放并准备使用固定航点PD前往通道'
        )
        self.reverse_history_complete = True
        self.operation_mode = 'FORWARD_PAUSE'
        self.forward_stage_ready_time = (
            time.monotonic() + self.reverse_to_forward_stop_hold
        )
        self.reverse_history_index = len(self.reverse_history) - 1

        self.rgb_avoid_active = False
        self.rgb_avoidance_replan_pending = False
        self.avoid_hold_v = 0.0
        self.avoid_hold_w = 0.0
        self.avoid_locked_sign = 0
        self.reset_avoid_pd()
        self.reset_goal_pd()
        self.reset_channel_waypoint_pd(reset_index=True)
        self.clear_route()
        self.exit_reverse_recovery('进入大厅x区域，等待切换固定航点PD')

    def perform_reverse_history(self) -> None:
        """沿任务一记录轨迹倒序闭环倒车，不进行开环速度时间回放。"""
        if (
            not self.channel_active
            or self.is_finished
            or not self.pose_received
            or self.operation_mode != 'REVERSE_HISTORY'
        ):
            return

        if len(self.reverse_history) < self.history_reverse_min_points:
            self.publish_stop('⛔ 历史轨迹点不足，禁止倒车')
            return

        x, y, yaw = self.actual_pose()

        # 只要x进入大厅区域就结束倒放，不再继续追踪历史首点，
        # 也不增加通往固定大厅坐标的额外闭环倒车段。
        if x <= self.lobby_x_max:
            self.finish_reverse_history()
            return

        index, cross_track = self.nearest_reverse_history_info()
        if cross_track > self.history_reverse_cross_track_stop:
            self.publish_stop(
                f'⛔ 倒放轨迹横向偏差={cross_track:.2f}m，超过'
                f'{self.history_reverse_cross_track_stop:.2f}m，停止等待检查定位'
            )
            return

        remaining = self.remaining_reverse_history_distance()
        lookahead = (
            self.history_reverse_final_lookahead
            if remaining < 0.50
            else self.history_reverse_lookahead
        )
        target_x, target_y, target_yaw = self.reverse_history_lookahead_point(lookahead)

        target_bearing = math.atan2(target_y - y, target_x - x)
        reverse_heading = self.normalize_angle(yaw + math.pi)
        alpha = self.normalize_angle(target_bearing - reverse_heading)
        curvature = 2.0 * math.sin(alpha) / max(lookahead, 1e-3)

        speed_abs = abs(self.history_reverse_speed)
        lobby_x_gap = max(0.0, x - self.lobby_x_max)
        if abs(alpha) > math.radians(55.0):
            speed_abs = min(speed_abs, 0.12)
        elif lobby_x_gap < 0.35:
            speed_abs = min(speed_abs, 0.14)

        v_out = -speed_abs
        yaw_error = self.normalize_angle(target_yaw - yaw)
        w_out = speed_abs * curvature + self.history_reverse_yaw_kp * yaw_error

        self.execute_drive(
            v_out,
            w_out,
            f'⏪ 历史轨迹倒放：idx={index}/{len(self.reverse_history)-1}，'
            f'横向误差={cross_track:.2f}m，剩余={remaining:.2f}m，'
            f'距大厅x门槛={max(0.0, x-self.lobby_x_max):.2f}m，'
            f'反向前视误差={math.degrees(alpha):.1f}°，'
            f'记录航向误差={math.degrees(yaw_error):.1f}°',
            source='history_reverse',
        )

    # ==================================================================
    # 路径跟踪
    # ==================================================================
    def nearest_route_info(self) -> Tuple[int, float]:
        if not self.route_points:
            return 0, float('inf')

        x, y, _ = self.actual_pose()
        start = max(0, self.route_index - 8)
        end = min(len(self.route_points), self.route_index + 120)

        best_index = self.route_index
        best_distance = float('inf')

        for index in range(start, end):
            px, py, _ = self.route_points[index]
            distance = math.hypot(px - x, py - y)
            if distance < best_distance:
                best_distance = distance
                best_index = index

        self.route_index = max(self.route_index, best_index)
        return self.route_index, best_distance

    def route_lookahead_point(self, lookahead: float) -> PathPoint:
        if not self.route_points:
            x, y, yaw = self.actual_pose()
            return x, y, yaw

        index, _ = self.nearest_route_info()
        accumulated = 0.0
        previous = self.route_points[index]

        for next_index in range(index + 1, len(self.route_points)):
            current = self.route_points[next_index]
            accumulated += math.hypot(
                current[0] - previous[0],
                current[1] - previous[1],
            )
            if accumulated >= lookahead:
                return current
            previous = current

        return self.route_points[-1]

    def remaining_route_distance(self) -> float:
        if not self.route_points:
            return float('inf')

        index = min(self.route_index, len(self.route_points) - 1)
        total = 0.0
        previous = self.route_points[index]
        for current in self.route_points[index + 1:]:
            total += math.hypot(
                current[0] - previous[0],
                current[1] - previous[1],
            )
            previous = current
        return total

    def check_channel_arrival(self) -> bool:
        if self.operation_mode != 'FORWARD_CHANNEL':
            return False
        if not self.channel_active or self.is_finished or not self.pose_received:
            return self.is_finished

        x, y, yaw = self.actual_pose()
        position_error = math.hypot(
            self.target_map_x - x,
            self.target_map_y - y,
        )
        yaw_error = abs(self.normalize_angle(self.target_yaw - yaw))

        in_safe_corridor = (
            self.guard_left_x_max < x < self.guard_right_x_min
        )

        if not (
            position_error <= self.arrival_position_tolerance
            and yaw_error <= self.arrival_yaw_tolerance
            and in_safe_corridor
        ):
            return False

        self.rgb_avoid_active = False
        self.rgb_avoidance_replan_pending = False
        self.avoid_clear_frame_count = 0
        self.avoid_hold_v = 0.0
        self.avoid_hold_w = 0.0
        self.avoid_locked_sign = 0
        self.rgb_base_route_v = 0.0
        self.rgb_base_route_w = 0.0
        self.reset_avoid_pd()
        self.reset_goal_pd()
        self.publish_stop(
            f'🏁 到达目标位姿：位置误差={position_error:.3f}m，'
            f'航向误差={math.degrees(yaw_error):.1f}°，'
            f'允许误差≤{math.degrees(self.arrival_yaw_tolerance):.0f}°'
        )
        self.is_finished = True
        self.channel_active = False
        self.operation_mode = 'FINISHED'
        return True

    def perform_route_following(self) -> None:
        """兼容旧调用入口；前进阶段实际只执行固定航点PD，不再路径规划。"""
        self.perform_channel_waypoint_pd()

    # ==================================================================
    # 基于RGB检测框的逐帧实时避障
    # ==================================================================
    @staticmethod
    def rgb_target_is_cone(target) -> bool:
        """过滤明显不是锥桶的目标；空类型按锥桶专用话题处理。"""
        target_type = str(getattr(target, 'type', '')).strip().lower()
        if not target_type:
            return True
        if any(token in target_type for token in ('qr', 'barcode', 'image_board')):
            return False
        return True

    def _build_rgb_edge_boundaries(self) -> None:
        """根据task.py实测点构建RGB锥桶左右有效区域。"""
        points = sorted(
            [
                (float(bottom_y), float(left_x))
                for bottom_y, left_x in self.rgb_edge_measure_points
            ],
            key=lambda item: item[0],
        )
        if len(points) < 2:
            raise ValueError('rgb_edge_measure_points至少需要两个点')

        self.rgb_edge_y_values = [item[0] for item in points]
        raw_x_values = [item[1] for item in points]

        if self.rgb_edge_smoothing_enabled and len(raw_x_values) >= 3:
            smoothed = []
            last_index = len(raw_x_values) - 1
            for index, current_x in enumerate(raw_x_values):
                if index == 0:
                    value = 0.75 * current_x + 0.25 * raw_x_values[index + 1]
                elif index == last_index:
                    value = 0.25 * raw_x_values[index - 1] + 0.75 * current_x
                else:
                    value = (
                        0.25 * raw_x_values[index - 1]
                        + 0.50 * current_x
                        + 0.25 * raw_x_values[index + 1]
                    )
                smoothed.append(value)
            self.rgb_edge_left_x_values = smoothed
        else:
            self.rgb_edge_left_x_values = raw_x_values

    def _interpolate_rgb_left_edge_x(self, bottom_y: float) -> float:
        y = float(bottom_y)
        y_values = self.rgb_edge_y_values
        x_values = self.rgb_edge_left_x_values

        if y <= y_values[0]:
            return x_values[0]
        if y >= y_values[-1]:
            return x_values[-1]

        for index in range(1, len(y_values)):
            if y <= y_values[index]:
                y0 = y_values[index - 1]
                y1 = y_values[index]
                x0 = x_values[index - 1]
                x1 = x_values[index]
                ratio = (y - y0) / max(y1 - y0, 1e-6)
                return x0 + ratio * (x1 - x0)

        return x_values[-1]

    def get_rgb_edge_boundaries(self, bottom_y: float) -> Tuple[float, float]:
        left_x = (
            self._interpolate_rgb_left_edge_x(bottom_y)
            + self.rgb_edge_ignore_margin_px
        )
        left_x = max(0.0, min(left_x, self.rgb_image_center_x - 1.0))
        right_x = 2.0 * self.rgb_image_center_x - left_x
        return left_x, right_x

    def rgb_is_in_edge_ignore_zone(
        self,
        center_x: float,
        bottom_y: float,
    ) -> bool:
        left_x, right_x = self.get_rgb_edge_boundaries(bottom_y)
        return center_x < left_x or center_x > right_x

    def collect_rgb_obstacle_candidates(self, msg: PerceptionTargets) -> list:
        """提取当前RGB帧中可触发task式PD的锥桶框。"""
        candidates = []
        for target in getattr(msg, 'targets', []):
            if not self.rgb_target_is_cone(target):
                continue

            for roi in getattr(target, 'rois', []):
                confidence = float(getattr(roi, 'confidence', 0.0))
                if confidence <= self.rgb_confidence_threshold:
                    continue

                rect = getattr(roi, 'rect', None)
                if rect is None:
                    continue

                x_offset = float(getattr(rect, 'x_offset', 0.0))
                y_offset = float(getattr(rect, 'y_offset', 0.0))
                width = float(getattr(rect, 'width', 0.0))
                height = float(getattr(rect, 'height', 0.0))
                if width <= 0.0 or height <= 0.0:
                    continue

                center_x = x_offset + width / 2.0
                bottom_y = y_offset + height

                if bottom_y <= self.rgb_trigger_bottom_y:
                    continue

                if self.rgb_is_in_edge_ignore_zone(center_x, bottom_y):
                    left_x, right_x = self.get_rgb_edge_boundaries(bottom_y)
                    self.get_logger().info(
                        f'🟡 RGB画面边缘忽略：center_x={center_x:.0f}px，'
                        f'bottom_y={bottom_y:.0f}px，'
                        f'有效范围=[{left_x:.0f},{right_x:.0f}]px，'
                        f'conf={confidence:.2f}',
                        throttle_duration_sec=0.5,
                    )
                    continue

                candidates.append(
                    (
                        bottom_y,
                        center_x,
                        confidence,
                        str(getattr(target, 'type', '')).strip(),
                    )
                )

        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates

    def reset_avoid_pd(self) -> None:
        self.rgb_avoid_prev_error = 0.0
        self.rgb_avoid_prev_time = 0.0
        self.rgb_avoid_filtered_derivative = 0.0

    def compute_rgb_pd_w(
        self,
        center_x: float,
        bottom_y: float,
    ) -> Tuple[float, float, float, float, float, float, float]:
        """
        计算中心增强型RGB锥桶PD。

        横向控制不再使用 center_x-320 的绝对偏差，而是使用中心接近度：
            center_closeness = max(0, 320 - abs(center_x-320))

        一次连续避障期间锁定转向方向：
            首次锥桶在右侧 -> 左转（w>0）
            首次锥桶在左侧 -> 右转（w<0）
        """
        horizontal_offset = center_x - self.rgb_image_center_x

        if self.avoid_locked_sign == 0:
            if horizontal_offset > self.rgb_center_direction_deadband_px:
                self.avoid_locked_sign = 1
            elif horizontal_offset < -self.rgb_center_direction_deadband_px:
                self.avoid_locked_sign = -1
            else:
                self.avoid_locked_sign = self.rgb_default_turn_sign

        center_distance = abs(horizontal_offset)
        center_closeness = max(
            0.0,
            self.rgb_center_effect_width_px - center_distance,
        )
        control_error = float(self.avoid_locked_sign) * center_closeness
        now = time.monotonic()

        if self.rgb_avoid_prev_time <= 0.0:
            derivative = 0.0
        else:
            dt = max(0.01, min(0.20, now - self.rgb_avoid_prev_time))
            derivative = (control_error - self.rgb_avoid_prev_error) / dt
            derivative = max(
                -self.rgb_avoid_derivative_limit,
                min(derivative, self.rgb_avoid_derivative_limit),
            )

        self.rgb_avoid_filtered_derivative = (
            self.rgb_avoid_derivative_alpha * derivative
            + (1.0 - self.rgb_avoid_derivative_alpha)
            * self.rgb_avoid_filtered_derivative
        )
        self.rgb_avoid_prev_error = control_error
        self.rgb_avoid_prev_time = now

        d1 = (
            bottom_y - self.rgb_trigger_bottom_y
        ) * self.rgb_cone_near_k
        d1 = max(0.0, min(d1, self.rgb_cone_near_max))

        d2_p = self.rgb_avoid_kp * control_error
        d2_d = self.rgb_avoid_kd * self.rgb_avoid_filtered_derivative
        d2 = d2_p + d2_d

        cone_w_raw = d1 * d2
        cone_w_limit = min(self.max_w, self.rgb_avoid_w_limit)
        cone_w = max(-cone_w_limit, min(cone_w_raw, cone_w_limit))

        return (
            cone_w,
            control_error,
            self.rgb_avoid_filtered_derivative,
            d1,
            d2,
            center_distance,
            center_closeness,
        )

    def detect_rgb_hazard(self, msg: PerceptionTargets) -> bool:
        """处理一帧RGB检测结果；使用task式PD与进入避障前的巡点PD角速度融合。"""
        if not self.enable_rgb_realtime_avoidance:
            return False

        candidates = self.collect_rgb_obstacle_candidates(msg)
        if not candidates:
            return False

        # 与task.py一致：选择bottom_y最大的锥桶，即图像中最近的锥桶。
        bottom_y, center_x, confidence, target_type = candidates[0]

        if not self.rgb_avoid_active:
            # 只在本次避障开始时保存一次巡点PD基础分量，避免后续帧重复累加。
            self.rgb_base_route_v = (
                self.last_cmd_v if self.last_cmd_v > 1e-3 else self.avoid_v
            )
            self.rgb_base_route_w = self.last_cmd_w
            self.avoid_locked_sign = 0
            self.reset_avoid_pd()

        (
            cone_w,
            control_error,
            derivative,
            d1,
            d2,
            center_distance,
            center_closeness,
        ) = self.compute_rgb_pd_w(
            center_x,
            bottom_y,
        )

        command_v = max(
            self.channel_pd_min_v,
            min(self.avoid_v, self.rgb_base_route_v),
        )
        command_w_raw = self.rgb_base_route_w + cone_w

        # 路径基础角速度可以参与融合，但绝不允许把当前视觉PD方向反转。
        # 例如锥桶在画面左侧时cone_w<0，最终命令不得变成左转。
        cone_sign = self.angular_direction_sign(cone_w)
        fused_sign = self.angular_direction_sign(command_w_raw)
        if cone_sign != 0 and fused_sign not in (0, cone_sign):
            command_w_raw = cone_w

        command_w = max(-self.max_w, min(command_w_raw, self.max_w))

        self.rgb_avoid_active = True
        self.avoid_clear_frame_count = 0
        self.avoid_hold_v = command_v
        self.avoid_hold_w = command_w
        self.last_avoid_center_x = center_x
        self.last_avoid_bottom_y = bottom_y
        self.last_avoid_confidence = confidence
        self.last_avoid_target_type = target_type

        side_text = (
            '画面右侧'
            if center_x > self.rgb_image_center_x
            else '画面左侧'
        )
        if abs(center_x - self.rgb_image_center_x) < 1.0:
            side_text = '画面正中'

        locked_direction_text = self.direction_name(self.avoid_locked_sign)
        self.execute_drive(
            command_v,
            command_w,
            f'⚡ 中心增强RGB锥桶PD：center_x={center_x:.1f}px({side_text})，'
            f'bottom_y={bottom_y:.1f}px，阈值={self.rgb_trigger_bottom_y:.0f}px，'
            f'conf={confidence:.2f}，d1={d1:.3f}，'
            f'距中心={center_distance:.1f}px，中心接近度={center_closeness:.1f}px，'
            f'锁定方向={locked_direction_text}，控制误差={control_error:+.1f}，'
            f'D={derivative:+.1f}/s，d2={d2:+.3f}，'
            f'W巡点基础={self.rgb_base_route_w:+.3f}，'
            f'W锥桶={cone_w:+.3f}，W融合={command_w:+.3f}',
            source='cone',
        )
        return True

    def finish_rgb_avoidance_immediately(self) -> None:
        """当前RGB帧无危险时立即退出避障，并恢复当前固定航点PD。"""
        self.rgb_avoid_active = False
        self.rgb_avoidance_replan_pending = False
        self.avoid_clear_frame_count = 0
        self.avoid_hold_v = 0.0
        self.avoid_hold_w = 0.0
        self.avoid_locked_sign = 0
        self.rgb_base_route_v = 0.0
        self.rgb_base_route_w = 0.0
        self.reset_avoid_pd()
        self.reset_channel_waypoint_pd(reset_index=False)

        waypoint = self.current_channel_waypoint()
        waypoint_text = (
            f'当前航点=({waypoint[0]:.2f},{waypoint[1]:.2f})'
            if waypoint is not None
            else '当前航点尚未初始化'
        )
        self.get_logger().warn(
            '✅ RGB当前帧未检测到危险锥桶，立即退出避障；'
            f'{waypoint_text}，直接恢复固定航点PD，不进行路径重规划'
        )
        self.perform_channel_waypoint_pd()

    def rgb_obstacle_cb(self, msg: PerceptionTargets) -> None:
        """RGB检测结果逐帧处理；危险帧立即抢占，安全帧负责退出。"""
        self.rgb_frame_id += 1
        self.rgb_frame_received = True
        self.rgb_last_frame_time = time.monotonic()

        if not self.enable_rgb_realtime_avoidance:
            return
        if not self.channel_active or self.is_finished or not self.pose_received:
            return
        if self.operation_mode != 'FORWARD_CHANNEL':
            # 摄像头朝前，倒车阶段不允许前视检测框抢占历史轨迹控制。
            return
        if not self.cone_map_ready():
            return
        if self.reverse_recovery_active:
            return

        if self.detect_rgb_hazard(msg):
            return

        if not self.rgb_avoid_active:
            return

        # 当前这一帧没有危险锥桶：立即结束避障，不再等待连续安全帧。
        self.finish_rgb_avoidance_immediately()

    # ==================================================================
    # 预测安全过滤
    # ==================================================================
    def predict_pose(
        self,
        x: float,
        y: float,
        yaw: float,
        v: float,
        w: float,
        duration: float,
    ) -> Tuple[float, float, float]:
        if abs(w) < 1e-6:
            return (
                x + v * duration * math.cos(yaw),
                y + v * duration * math.sin(yaw),
                yaw,
            )

        next_yaw = yaw + w * duration
        return (
            x + (v / w) * (math.sin(next_yaw) - math.sin(yaw)),
            y - (v / w) * (math.cos(next_yaw) - math.cos(yaw)),
            self.normalize_angle(next_yaw),
        )

    def command_is_safe(self, v: float, w: float) -> bool:
        x, y, yaw = self.actual_pose()

        for duration in self.safety_prediction_times:
            px, py, _ = self.predict_pose(x, y, yaw, v, w, duration)
            if not self.runtime_pose_is_safe(px, py):
                return False
        return True

    def reverse_command_is_safe(self, v: float, w: float) -> bool:
        """检查倒车恢复命令是否安全，并允许从轻微 guard 侵入中退出。

        原逻辑只要当前点位于保守 guard 区，就会立即返回 False。
        这会造成：前进路径因起点不安全而失败，倒车又因同一起点失败，
        最终永久停车。现在区分“真实禁止区”和“可恢复的保守区”：

        - 当前点/预测点进入真实禁止区、x=5或y=0：仍然拒绝；
        - 当前点已轻微进入 guard 区：允许倒车轨迹暂时仍在 guard 区，
          但 guard 侵入程度不得增加，且整段预测结束时必须明显改善；
        - 当前点本来安全：仍要求所有预测点都保持 guard 安全。
        """
        x, y, yaw = self.actual_pose()
        self.last_reverse_block_reason = ''

        # 真实禁止区和地图硬边界不允许通过恢复逻辑穿越。
        if self.is_actual_forbidden(x, y):
            self.last_reverse_block_reason = '当前车体中心位于真实禁止区域'
            return False
        if not (-0.2 <= x < self.boundary_x_max and self.boundary_y_min < y <= 2.55):
            self.last_reverse_block_reason = '当前车体中心越过地图硬边界'
            return False
        current_guard_violation = self.guard_violation_level(x, y)
        maximum_allowed_violation = current_guard_violation + 0.003
        final_guard_violation = current_guard_violation

        for duration in self.reverse_safety_prediction_times:
            px, py, _ = self.predict_pose(x, y, yaw, v, w, duration)

            if self.is_actual_forbidden(px, py):
                self.last_reverse_block_reason = (
                    f'{duration:.2f}s预测点进入真实禁止区域'
                )
                return False
            if not (
                -0.2 <= px < self.boundary_x_max
                and self.boundary_y_min < py <= 2.55
            ):
                self.last_reverse_block_reason = (
                    f'{duration:.2f}s预测点越过x=5/y=0硬边界'
                )
                return False
            predicted_violation = self.guard_violation_level(px, py)
            final_guard_violation = predicted_violation

            if current_guard_violation <= 1e-6:
                # 从安全区开始时，不允许倒车轨迹进入任何保守禁止区。
                if predicted_violation > 1e-6:
                    self.last_reverse_block_reason = (
                        f'{duration:.2f}s预测点将进入地图保守禁止区'
                    )
                    return False
            elif predicted_violation > maximum_allowed_violation:
                # 已在保守区时只允许向外逃逸，不能进一步深入。
                self.last_reverse_block_reason = (
                    f'{duration:.2f}s预测使保守区侵入加深：'
                    f'{current_guard_violation:.3f}->{predicted_violation:.3f}m'
                )
                return False

        if (
            current_guard_violation > 1e-6
            and final_guard_violation
            > max(0.0, current_guard_violation - 0.005)
        ):
            self.last_reverse_block_reason = (
                '倒车预测没有明显减小当前保守区侵入程度：'
                f'{current_guard_violation:.3f}->{final_guard_violation:.3f}m'
            )
            return False

        return True

    def apply_forbidden_safety_filter(
        self,
        v: float,
        requested_w: float,
        source: str,
    ) -> Tuple[float, float, str]:
        if abs(v) <= 1e-6:
            return 0.0, 0.0, ''

        max_w_by_radius = min(self.max_w, abs(v) / self.safe_turn_radius)
        requested_w = max(-max_w_by_radius, min(requested_w, max_w_by_radius))

        safety_check = (
            self.reverse_command_is_safe
            if source in ('reverse', 'recovery', 'history_reverse')
            else self.command_is_safe
        )

        if safety_check(v, requested_w):
            return v, requested_w, ''

        candidate_values = [requested_w]
        for index in range(25):
            ratio = -1.0 + 2.0 * index / 24.0
            candidate_values.append(ratio * max_w_by_radius)

        # RGB锥桶避障时，安全过滤器绝不允许反转视觉方向。
        # 只可在视觉指定方向内减小角速度、退化为直行或停车；
        # 实时安全过滤只检查地图禁区和硬边界，不检查锥桶坐标膨胀区。
        requested_sign = self.angular_direction_sign(requested_w)
        if source == 'cone':
            if requested_sign != 0:
                candidate_values = [
                    candidate
                    for candidate in candidate_values
                    if self.angular_direction_sign(candidate) in (0, requested_sign)
                ]
            else:
                # task式PD在锥桶位于中心附近时允许W锥桶自然降为0；
                # 地图安全过滤不得自行选择左转或右转，只能直行或停车。
                candidate_values = [0.0]
        elif source == 'reverse':
            # 仅兼容旧调用；主动姿态恢复使用source='recovery'，允许左右两侧调整。
            candidate_values = [candidate for candidate in candidate_values if candidate > 0.0]

        x, y, yaw = self.actual_pose()
        safe_candidates = []

        for candidate_w in candidate_values:
            if not safety_check(v, candidate_w):
                continue

            px, py, _ = self.predict_pose(x, y, yaw, v, candidate_w, 0.60)
            corridor_penalty = max(0.0, py - 1.70) * abs(px - 2.5)
            change_penalty = abs(candidate_w - requested_w)
            steering_penalty = 0.05 * abs(candidate_w)
            cost = change_penalty + 2.5 * corridor_penalty + steering_penalty
            safe_candidates.append((cost, candidate_w))

        if not safe_candidates:
            if source in ('reverse', 'recovery', 'history_reverse'):
                return (
                    0.0,
                    0.0,
                    ' | ⛔ 姿态恢复预测将进入地图禁止区域，停车',
                )
            return 0.0, 0.0, ' | ⛔ 预测将进入禁止区域，无安全转向，停车'

        safe_candidates.sort(key=lambda item: item[0])
        safe_w = safe_candidates[0][1]
        return (
            v,
            safe_w,
            f' | 🛡️ 禁区安全修正 W:{requested_w:.2f}->{safe_w:.2f}',
        )

    # ==================================================================
    # ROS 回调、状态机和速度发布
    # ==================================================================
    def cone_coordinates_cb(self, msg: String) -> None:
        """接收 JSON 锥桶中心坐标，并在当前路径被挡住时立即重规划。"""
        raw_text = str(msg.data).strip()
        if not raw_text:
            self.get_logger().warn(
                f'收到空 {self.cone_coordinates_topic} 消息，保留上一帧坐标',
                throttle_duration_sec=1.0,
            )
            return

        try:
            payload = json.loads(raw_text)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            self.cone_parse_error_count += 1
            self.get_logger().error(
                f'❌ {self.cone_coordinates_topic} JSON解析失败'
                f'（第{self.cone_parse_error_count}次）：{exc}；'
                '保留上一帧有效锥桶坐标',
                throttle_duration_sec=1.0,
            )
            return

        if not isinstance(payload, list):
            self.cone_parse_error_count += 1
            self.get_logger().error(
                f'❌ {self.cone_coordinates_topic} 顶层必须是JSON数组，'
                f'实际类型={type(payload).__name__}；保留上一帧坐标',
                throttle_duration_sec=1.0,
            )
            return

        parsed_points: List[Tuple[float, float]] = []
        invalid_items = 0

        for item in payload:
            if not isinstance(item, dict) or 'x' not in item or 'y' not in item:
                invalid_items += 1
                continue

            try:
                cone_x = float(item['x'])
                cone_y = float(item['y'])
            except (TypeError, ValueError):
                invalid_items += 1
                continue

            if not math.isfinite(cone_x) or not math.isfinite(cone_y):
                invalid_items += 1
                continue

            parsed_points.append((cone_x, cone_y))

        new_points = self.deduplicate_cone_coordinates(parsed_points)
        first_message = not self.cone_coordinates_received
        changed = self.cone_coordinate_sets_changed(
            self.cone_coordinates,
            new_points,
        )

        self.cone_coordinates = new_points
        self.cone_coordinates_received = True

        if first_message or changed:
            self.get_logger().info(
                f'📦 锥桶地图更新：有效锥桶={len(new_points)}个，'
                f'无效条目={invalid_items}，'
                f'单个锥桶={self.cone_square_size:.2f}m×'
                f'{self.cone_square_size:.2f}m，'
                f'车辆中心总避让半宽='
                f'{self.cone_half_size + self.cone_path_safety_margin:.2f}m'
            )

        if not self.enable_cone_map_planning:
            return
        if not self.channel_active or self.is_finished or not self.pose_received:
            return
        if self.rgb_avoid_active:
            # 避障过程中只更新地图，不抢占RGB实时控制。
            return
        if self.rgb_avoidance_replan_pending:
            # 避障刚结束时保留原路径；是否重规划由横向偏差阈值决定。
            return
        if self.reverse_recovery_active:
            # 倒车恢复会按固定周期使用最新坐标重新规划候选前进路径；
            # 实际倒车速度安全判定本身不读取锥桶坐标。
            return

        remaining_route = self.route_points[
            max(0, self.route_index - 2):
        ]
        route_blocked = bool(
            remaining_route
            and not self.route_is_safe(remaining_route)
        )

        if remaining_route and not route_blocked:
            return

        now = time.monotonic()
        if (
            not first_message
            and not changed
            and now - self.last_cone_replan_time
            < self.cone_replan_min_interval
        ):
            return

        if (
            route_blocked
            and now - self.last_cone_replan_time
            < self.cone_replan_min_interval
        ):
            # 锥桶坐标只用于规划。即使最新膨胀区覆盖当前路径，
            # 限频等待期间也不因坐标地图直接停车；继续由RGB实时避障
            # 和原路径控制，达到重规划间隔后再尝试生成新路径。
            self.get_logger().warn(
                '⚠️ 最新锥桶膨胀区与当前路径相交，但已取消坐标停车；'
                '暂时继续原路径/RGB控制，等待重规划限频时间',
                throttle_duration_sec=0.5,
            )
            return

        self.last_cone_replan_time = now

        # 首次等待锥桶地图后生成路径时，继续保留原来的“第一段左转”规则；
        # 已有路径被新锥桶阻断时，属于动态重规划，不再限制第一段方向。
        require_first_left = (
            self.route_require_first_left
            if not remaining_route
            else False
        )

        self.clear_route()
        reason = (
            f'{self.cone_coordinates_topic} 更新后'
            f'{"当前路径与锥桶膨胀正方形相交" if route_blocked else "开始首次锥桶地图规划"}'
        )

        if self.plan_safe_route(
            reason,
            require_first_left=require_first_left,
        ):
            direction_rule = (
                '保留首段左转规则'
                if require_first_left
                else '动态重规划不限制第一段左右方向'
            )
            self.get_logger().warn(
                '🔄 已根据最新锥桶地图清除旧路径并生成新路径；'
                f'{direction_rule}'
            )
            self.perform_route_following()
            return

        # 不因为进入锥桶膨胀区先发布停车命令。若确实无法生成安全
        # 前进路径，直接切换到恢复流程；恢复阶段是否能够运动仍只由
        # 地图真实禁区/硬边界安全检查决定。
        self.get_logger().warn(
            '⚠️ 最新锥桶地图下暂时没有安全前进路径；'
            '已取消锥桶膨胀区停车，直接进入主动姿态恢复'
        )
        self.enter_reverse_recovery('锥桶地图更新后安全规划失败')
        self.perform_reverse_recovery()

    def pose_cb(self, msg: Pose2D) -> None:
        first_pose = not self.pose_received
        self.cur_pose = [msg.x, msg.y, msg.theta]
        self.pose_received = True

        if first_pose:
            x, y, yaw = self.actual_pose()
            self.get_logger().info(
                f'📍 首次位姿：({x:.3f}, {y:.3f})，'
                f'yaw={math.degrees(yaw):.1f}°'
            )

        # 二维码可以提前到达。每次位姿更新都检查actual_x门槛；
        # 只有actual_x严格大于2.5m才真正激活、发布ACK并接管/cmd_vel。
        if (
            not self.channel_active
            and not self.is_finished
            and self.pending_qr_result
        ):
            if self.try_activate_pending_qr('位姿更新后达到x门槛'):
                return

        # 前进阶段由40Hz控制循环执行固定航点PD；位姿回调不再生成路径。

    def publish_channel_ack(self, result: str, reason: str) -> None:
        ack = String()
        ack.data = (
            f'channel_active={self.channel_active};result={result};'
            f'pose_received={self.pose_received};reason={reason}'
        )
        self.channel_ack_pub.publish(ack)
        self.ack_publish_count += 1

    def activate_channel_navigation(
        self,
        result: str,
        reason: str,
        forward_history: Sequence[PathPoint],
    ) -> None:
        """二维码、x门槛和完整历史轨迹满足后，先进入倒序返回阶段。"""
        history = list(forward_history)
        if len(history) < self.history_reverse_min_points:
            self.channel_active = False
            self.operation_mode = 'WAIT_HANDOFF'
            self.publish_channel_ack(
                result,
                f'拒绝激活：历史轨迹点不足({len(history)})',
            )
            self.publish_stop('⛔ /qr_success 未携带足够历史轨迹，保持停车')
            return

        # 历史轨迹至少要覆盖到大厅x区域，否则倒序走到轨迹末端后
        # 仍无法满足切换条件。这里只检查覆盖范围，不人为补轨迹或补固定点。
        minimum_history_x = min(point[0] for point in history)
        if minimum_history_x > self.lobby_x_max:
            self.channel_active = False
            self.operation_mode = 'WAIT_HANDOFF'
            self.publish_channel_ack(
                result,
                f'拒绝激活：历史轨迹最小x={minimum_history_x:.2f}m未进入大厅区域',
            )
            self.publish_stop(
                f'⛔ 历史轨迹未覆盖大厅x区域：最小x={minimum_history_x:.2f}m，'
                f'要求x<={self.lobby_x_max:.2f}m；禁止在轨迹末端继续盲目倒车'
            )
            return

        current_x, current_y, current_yaw = self.actual_pose()
        last_x, last_y, _ = history[-1]
        start_gap = math.hypot(current_x - last_x, current_y - last_y)
        if start_gap > self.history_reverse_start_gap_limit:
            self.channel_active = False
            self.operation_mode = 'WAIT_HANDOFF'
            self.publish_channel_ack(
                result,
                f'拒绝激活：当前位姿与轨迹末点相差{start_gap:.2f}m',
            )
            self.publish_stop(
                f'⛔ 当前位姿与任务一历史轨迹末点相差{start_gap:.2f}m，'
                '超过安全门槛，禁止盲目倒车'
            )
            return

        self.qr_direction = result
        self.pending_qr_result = ''
        self.pending_forward_history = []
        self.channel_active = True
        self.operation_mode = 'REVERSE_HISTORY'

        self.forward_history = history
        # 只倒序使用任务一真实记录的轨迹，不修改末点、不追加固定大厅点。
        self.reverse_history = list(reversed(history))

        self.reverse_history_index = 0
        self.reverse_history_complete = False

        self.rgb_avoid_active = False
        self.rgb_avoidance_replan_pending = False
        self.avoid_clear_frame_count = 0
        self.avoid_hold_v = 0.0
        self.avoid_hold_w = 0.0
        self.avoid_locked_sign = 0
        self.rgb_base_route_v = 0.0
        self.rgb_base_route_w = 0.0
        self.reset_avoid_pd()
        self.reset_goal_pd()
        self.reset_channel_waypoint_pd(reset_index=True)

        self.route_require_first_left = True
        self.last_side_rear_replan_time = 0.0
        self.clear_route()
        self.exit_reverse_recovery('激活倒序轨迹返回，重置旧恢复状态')

        self.publish_channel_ack(
            result,
            f'{reason};mode=REVERSE_HISTORY;path_points={len(history)}',
        )

        self.get_logger().warn(
            f'🚦 倒序轨迹导航正式激活：二维码={result}，'
            f'当前位置=({current_x:.3f},{current_y:.3f},'
            f'{math.degrees(current_yaw):.1f}°)，'
            f'历史轨迹点={len(history)}，轨迹末点间隙={start_gap:.3f}m，'
            f'历史最小x={minimum_history_x:.3f}m；'
            f'沿真实历史轨迹倒车，actual_x<={self.lobby_x_max:.2f}m后'
            '立即切换固定航点PD前往通道'
        )
        self.perform_reverse_history()

    def try_activate_pending_qr(self, reason: str) -> bool:
        """二维码提前到达时缓存；actual_x严格大于门槛后才激活。"""
        if (
            self.channel_active
            or self.is_finished
            or not self.pending_qr_result
            or not self.pose_received
        ):
            return False

        x, _, _ = self.actual_pose()
        if x <= self.channel_activation_min_x:
            self.get_logger().info(
                f'⏳ 二维码结果已缓存，等待actual_x>'
                f'{self.channel_activation_min_x:.2f}m；当前x={x:.3f}m',
                throttle_duration_sec=1.0,
            )
            return False

        result = self.pending_qr_result
        history = list(self.pending_forward_history)
        self.activate_channel_navigation(result, reason, history)
        return True

    def qr_success_cb(self, msg: String) -> None:
        self.qr_success_callback_count += 1
        raw_text = str(msg.data).strip()

        if not raw_text:
            self.get_logger().warn('收到空 /qr_success，保持休眠')
            return

        result, history, parse_note = self.parse_qr_handoff_payload(raw_text)
        if not result:
            self.get_logger().error(f'❌ /qr_success 解析失败：{parse_note}')
            return

        if self.channel_active:
            self.publish_channel_ack(
                result,
                f'节点已经激活;mode={self.operation_mode};{parse_note}',
            )
            return

        if self.is_finished:
            self.publish_channel_ack(result, '通道任务已经完成')
            return

        previous_result = self.pending_qr_result
        self.pending_qr_result = result
        self.pending_forward_history = history
        self.qr_direction = result

        if len(history) < self.history_reverse_min_points:
            self.get_logger().error(
                f'❌ 已收到二维码={result}，但历史轨迹不可用：{parse_note}；'
                '不会接管 /cmd_vel'
            )
            return

        if previous_result and previous_result != result:
            self.get_logger().warn(
                f'📩 更新提前缓存的二维码结果：{previous_result} -> {result}；'
                f'{parse_note}'
            )
        else:
            self.get_logger().warn(
                f'📩 已缓存二维码结果和历史轨迹：二维码={result}；{parse_note}'
            )

        if not self.pose_received:
            self.get_logger().info(
                '⏳ 二维码和历史轨迹已缓存，尚未收到位姿；'
                '收到位姿且actual_x>2.50m后再激活倒放'
            )
            return

        self.try_activate_pending_qr('收到二维码和历史轨迹时车辆x已超过门槛')

    def control_loop(self) -> None:
        if not self.channel_active or self.is_finished or not self.pose_received:
            return

        x, y, _ = self.actual_pose()
        if self.is_actual_forbidden(x, y):
            self.publish_stop(
                f'❌ 检测到车体中心进入禁止区域 ({x:.3f},{y:.3f})，紧急停车'
            )
            return

        if self.operation_mode == 'REVERSE_HISTORY':
            self.perform_reverse_history()
            return

        if self.operation_mode == 'FORWARD_PAUSE':
            if time.monotonic() < self.forward_stage_ready_time:
                self.publish_stop('⏸️ 倒车结束，短暂停车后切换固定航点PD')
                return

            self.operation_mode = 'FORWARD_CHANNEL'
            self.start_channel_waypoint_pd()
            self.perform_channel_waypoint_pd()
            return

        if self.operation_mode != 'FORWARD_CHANNEL':
            return

        if self.check_channel_arrival():
            return

        rgb_ready, rgb_reason = self.rgb_stream_motion_ready()
        if not rgb_ready:
            self.publish_stop(rgb_reason)
            return

        if self.enable_rgb_realtime_avoidance and self.rgb_avoid_active:
            # 两帧RGB之间保持上一帧避障命令；安全帧到来后恢复当前航点PD。
            self.execute_drive(
                self.avoid_hold_v,
                self.avoid_hold_w,
                '🧲 保持上一帧RGB融合结果；安全帧后恢复固定航点PD',
                source='cone',
            )
            return

        self.perform_channel_waypoint_pd()

    def publish_stop(self, reason: str) -> None:
        self.last_cmd_v = 0.0
        self.last_cmd_w = 0.0
        twist = Twist()
        self.cmd_pub.publish(twist)
        self.cmd_publish_count += 1
        self.get_logger().warn(reason, throttle_duration_sec=0.5)

    def execute_drive(self, v: float, w: float, log_tag: str, source: str) -> None:
        v = max(self.reverse_speed, min(float(v), self.max_v))
        w = max(-self.max_w, min(float(w), self.max_w))

        # 第一层：满足最小转弯半径，前进和倒车都使用速度绝对值。
        radius_note = ''
        if abs(v) > 1e-6 and abs(w) > 1e-6:
            max_w_by_radius = abs(v) / self.safe_turn_radius
            if abs(w) > max_w_by_radius:
                old_w = w
                w = math.copysign(max_w_by_radius, w)
                radius_note = f' | 半径限制 W:{old_w:.2f}->{w:.2f}'

        # 第二层：预测未来轨迹，只禁止进入地图禁区和硬边界；
        # 锥桶实时危险由RGB检测回调独立处理。
        v, w, safety_note = self.apply_forbidden_safety_filter(v, w, source)

        self.last_cmd_v = float(v)
        self.last_cmd_w = float(w)

        twist = Twist()
        twist.linear.x = float(v)
        twist.angular.z = float(w)
        self.cmd_pub.publish(twist)
        self.cmd_publish_count += 1

        self.get_logger().info(
            f'{log_tag}{radius_note}{safety_note} | V={v:.2f}, W={w:.2f}',
            throttle_duration_sec=0.5,
        )

    def status_loop(self) -> None:
        if self.is_finished:
            return

        if (
            not self.channel_active
            and self.pose_received
            and self.pending_qr_result
        ):
            x, y, yaw = self.actual_pose()
            self.get_logger().info(
                f'📨 等待去通道激活：二维码={self.pending_qr_result}，'
                f'位置=({x:.2f},{y:.2f})，yaw={math.degrees(yaw):.1f}°，'
                f'要求actual_x>{self.channel_activation_min_x:.2f}m',
                throttle_duration_sec=1.0,
            )
            return

        if not self.channel_active or not self.pose_received:
            return

        if self.operation_mode == 'FORWARD_PAUSE':
            self.get_logger().info(
                '⏸️ 历史轨迹倒放已完成，正在停车切换前进阶段',
                throttle_duration_sec=1.0,
            )
            return

        if self.operation_mode == 'REVERSE_HISTORY':
            x, y, yaw = self.actual_pose()
            index, cross_track = self.nearest_reverse_history_info()
            remaining = self.remaining_reverse_history_distance()
            self.get_logger().info(
                f'📊 倒放状态：位置=({x:.2f},{y:.2f})，'
                f'yaw={math.degrees(yaw):.1f}°，'
                f'轨迹进度={index}/{max(0,len(self.reverse_history)-1)}，'
                f'横向误差={cross_track:.2f}m，剩余={remaining:.2f}m，'
                f'大厅条件=x<={self.lobby_x_max:.2f}m，'
                f'倒车速度={self.history_reverse_speed:.2f}m/s'
            )
            return

        x, y, yaw = self.actual_pose()
        position_error = math.hypot(
            self.target_map_x - x,
            self.target_map_y - y,
        )
        yaw_error = abs(self.normalize_angle(self.target_yaw - yaw))
        waypoint = self.current_channel_waypoint()
        if waypoint is None:
            waypoint_text = '未初始化'
            waypoint_distance = float('inf')
        else:
            waypoint_text = f'({waypoint[0]:.2f},{waypoint[1]:.2f})'
            waypoint_distance = math.hypot(waypoint[0] - x, waypoint[1] - y)

        self.get_logger().info(
            f'📊 状态：位置=({x:.2f},{y:.2f})，'
            f'yaw={math.degrees(yaw):.1f}°，'
            f'最终目标距离={position_error:.2f}m，'
            f'最终航向误差={math.degrees(yaw_error):.1f}°，'
            f'固定航点进度={self.channel_pd_waypoint_index + 1}/'
            f'{max(1,len(self.channel_pd_waypoints))}，'
            f'当前航点={waypoint_text}，航点距离={waypoint_distance:.2f}m，'
            f'巡点PD激活={self.channel_pd_active}，'
            f'锥桶地图规划=关闭，'
            f'RGB实时避障开关={self.enable_rgb_realtime_avoidance}，'
            f'RGB帧已接收={self.rgb_frame_received}，'
            f'RGB避障中={self.rgb_avoid_active}，'
            f'最近center_x={self.last_avoid_center_x:.1f}px，'
            f'最近bottom_y={self.last_avoid_bottom_y:.1f}px，'
            f'当前锥桶PD方向={self.direction_name(self.avoid_locked_sign)}，'
            f'速度上限=({self.channel_pd_max_v:.2f}m/s,'
            f'{self.max_w:.2f}rad/s)'
        )



def main() -> None:
    rclpy.init()
    node = ChannelNavigationNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node.channel_active and not node.is_finished:
            node.get_logger().warn('🛑 通道导航被中断，发送停车命令')
            for _ in range(5):
                node.cmd_pub.publish(Twist())
                time.sleep(0.02)
            os.system(
                "ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist '{}' "
                '> /dev/null 2>&1'
            )
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()