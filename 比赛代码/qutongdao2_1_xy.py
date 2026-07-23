#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
二维码交接后的去通道节点（无路径规划、无未来轨迹预测版）。

运行流程：
1. 等待任务一通过原 /qr_success 发送二维码结果与历史位姿轨迹；
2. 车辆地图 x>2.50m 后接管 /cmd_vel；
3. 将任务一历史轨迹倒序，低速闭环倒车；
4. 加偏置后的地图坐标严格满足 map_x<2.50m 时，立即切换前进；
5. 不等待、不生成路径、不预测未来位置，直接使用PD巡点前往
   唯一通道目标 (2.50, 2.00, 90°)；
6. 前进过程中保留RGB近距离锥桶避障；只有收到危险检测框时才抢占；
   未收到检测消息、收到空帧或检测流暂时中断，都不阻塞PD巡点；
7. 可通过 enable_forbidden_auto_recovery 开关决定禁区处理方式：
   True 时，进入左右硬禁止区后自动生成安全回退点并闭环倒车退出；
   False 时，保持原来的禁区立即停车，不发布自动倒车命令；
8. 自动恢复开启时，连续确认离开禁区后恢复原来的倒车历史或前进PD巡点；
9. 仍不预测未来0.1~0.75秒轨迹，也不枚举候选转向。

坐标约定：
    map_x = odom_x + 0.55
    map_y = odom_y + 0.22

角速度约定：
    w > 0：左转
    w < 0：右转

保持原话题名称：
订阅：/odom_pose、/cone_coordinates、/racing_obstacle_detection、/qr_success
发布：/cmd_vel、/channel_navigation_ack、/task2_start
"""

import json
import math
import os
import time
from typing import List, Sequence, Tuple

import rclpy
from ai_msgs.msg import PerceptionTargets
from geometry_msgs.msg import Pose2D, Twist
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String


PathPoint = Tuple[float, float, float]


class ChannelNavigationNode(Node):
    def __init__(self) -> None:
        super().__init__('channel_navigation_node')

        # ==============================================================
        # 1. 地图坐标和目标点
        # ==============================================================
        # /odom_pose=(0,0) 对应地图坐标 (0.55,0.22)。
        self.start_offset_x = 0.55
        self.start_offset_y = 0.20

        # 使用“PD巡点”结构，但当前只有一个目标点。
        # 后续需要增加中间点时，只需在列表中追加 (x,y,yaw)。
        self.channel_waypoints: List[PathPoint] = [
            (2.50, 2.00, math.radians(90.0)),
        ]
        self.waypoint_index = 0

        self.target_map_x = self.channel_waypoints[-1][0]
        self.target_map_y = self.channel_waypoints[-1][1]
        self.target_yaw = self.channel_waypoints[-1][2]

        # 与任务二的最终交接区域（严格不包含边界）。
        self.arrival_corridor_x_min = 2.00
        self.arrival_corridor_x_max = 3.00
        self.arrival_corridor_y_min = 2.00

        # 当前地图硬禁止区域，只做“当前位姿”检查，不做未来预测。
        self.forbidden_y_min = 2.00
        self.forbidden_y_max = 2.50
        self.forbidden_left_x_max = 2.00
        self.forbidden_right_x_min = 3.00
        self.map_x_min = -0.20
        self.map_x_max = 5.00
        self.map_y_min = 0.00
        self.map_y_max = 2.55

        # 禁区自动倒车恢复总开关：
        # True  = 进入左右禁区后自动倒车并调整角度退出禁区；
        # False = 关闭自动恢复，进入任何禁区后立即停车。
        # 注意：Python布尔值首字母必须大写，只能填写 True 或 False。
        self.enable_forbidden_auto_recovery = True

        # 禁区恢复：只对左右两块通道外禁区执行自动倒车恢复。
        # 地图总边界越界仍然停车，避免盲目倒车进一步驶出场地。
        self.forbidden_recovery_speed = -0.24
        self.forbidden_recovery_min_speed = 0.12
        self.forbidden_recovery_reverse_distance = 0.42
        self.forbidden_recovery_inward_shift = 0.38
        self.forbidden_recovery_corridor_margin = 0.18
        self.forbidden_recovery_y_margin = 0.14
        self.forbidden_recovery_lookahead_min = 0.25
        self.forbidden_recovery_lookahead_max = 0.55
        self.forbidden_recovery_target_tolerance = 0.18
        self.forbidden_recovery_safe_frames_required = 4
        self.forbidden_recovery_max_duration = 8.0
        self.forbidden_recovery_max_distance = 1.40
        self.forbidden_recovery_min_turn_w = 0.22

        # ==============================================================
        # 2. 车辆输出限制
        # ==============================================================
        self.max_v = 1.00
        self.max_w = 2.50

        # 仅保留阿克曼小车物理转弯半径约束，不属于路径预测。
        self.safe_turn_radius = 0.40

        # ==============================================================
        # 3. 历史轨迹倒车
        # ==============================================================
        self.history_reverse_speed = -0.80  #-0.20
        self.history_reverse_lookahead = 0.45 #0.28
        self.history_reverse_final_lookahead = 0.28 #0.14
        self.history_reverse_yaw_kp = 0.22 #0.365
        self.history_reverse_cross_track_stop = 0.60
        self.history_reverse_start_gap_limit = 0.50
        self.history_reverse_min_points = 2

        # 必须使用加过偏置的地图坐标判断。
        self.reverse_exit_map_x = 2.90

        # ==============================================================
        # 4. 前进PD巡点参数
        # ==============================================================
        # 远处朝向目标点，进入该距离后逐渐融合到目标点要求的90°航向。
        self.waypoint_heading_blend_distance = 0.70 #0.50

        # 航向误差PD。
        self.waypoint_kp = 2.45 #2.30 #2.20
        self.waypoint_kd = 0.20 #0.18 #0.16
        self.waypoint_derivative_limit = 3.50 #3.00
        self.waypoint_derivative_alpha = 0.25 #0.28 #0.35

        # 线速度。
        self.waypoint_linear_kp = 1.55 #1.30 #1.00
        self.waypoint_min_v = 0.28 #0.25 #0.22
        self.waypoint_max_v = 0.95 #0.85 #0.70
        self.waypoint_sharp_turn_v = 0.38 #0.32 #0.24
        self.waypoint_sharp_turn_angle = math.radians(82.0) #78.0#70.0

        # 中间点切换容差。当前只有最终点，保留该参数便于以后扩展。
        self.intermediate_waypoint_tolerance = 0.18

        self.waypoint_prev_error = 0.0
        self.waypoint_prev_time = 0.0
        self.waypoint_filtered_derivative = 0.0
        self.waypoint_pd_active = False
        self.last_waypoint_v = 0.0
        self.last_waypoint_w = 0.0

        # ==============================================================
        # 5. RGB实时锥桶避障
        # ==============================================================
        self.enable_rgb_realtime_avoidance = True
        self.rgb_obstacle_topic = '/racing_obstacle_detection'

        self.rgb_image_width = 640.0
        self.rgb_image_height = 480.0
        self.rgb_image_center_x = self.rgb_image_width / 2.0
        self.rgb_confidence_threshold = 0.60
        self.rgb_trigger_bottom_y = 290.0

        # d1=(bottom_y-trigger)*near_k，表示锥桶接近程度。
        self.rgb_cone_near_k = 0.125
        self.rgb_cone_near_max = 1.50

        # 中心增强型横向PD。
        self.rgb_center_effect_width_px = 320.0
        self.rgb_center_direction_deadband_px = 8.0
        self.rgb_default_turn_sign = -1  # 中心死区默认右转
        self.rgb_avoid_kp = 0.0040
        self.rgb_avoid_kd = 0.0002
        self.rgb_avoid_w_limit = 1.80
        self.rgb_avoid_derivative_limit = 800.0
        self.rgb_avoid_derivative_alpha = 0.35
        self.turn_w_deadband = 0.02

        # 图像边缘忽略曲线：(bottom_y, 左侧有效边界x)。
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

        # RGB数据流仅用于避障，不再作为PD巡点的启动条件。
        # 未收到任何帧时直接执行PD；若避障激活后超过0.50s没有新检测帧，
        # 按“当前没有检测到锥桶”处理，清除旧避障指令并恢复PD巡点。
        self.rgb_frame_received = False
        self.rgb_last_frame_time = 0.0
        self.rgb_frame_timeout = 0.50
        self.rgb_frame_id = 0

        # RGB避障状态。
        self.rgb_avoid_active = False
        self.avoid_locked_sign = 0
        self.avoid_hold_v = 0.0
        self.avoid_hold_w = 0.0
        self.rgb_base_waypoint_v = 0.0
        self.rgb_base_waypoint_w = 0.0
        self.rgb_avoid_prev_error = 0.0
        self.rgb_avoid_prev_time = 0.0
        self.rgb_avoid_filtered_derivative = 0.0
        self.last_avoid_center_x = 0.0
        self.last_avoid_bottom_y = 0.0
        self.last_avoid_confidence = 0.0
        self.last_avoid_target_type = ''

        # ==============================================================
        # 6. 交接和运行状态
        # ==============================================================
        self.qr_success_topic = '/qr_success'
        self.channel_ack_topic = '/channel_navigation_ack'
        self.cone_coordinates_topic = '/cone_coordinates'

        # 去通道完成后触发任务二。task2节点要求消息内容严格为 start。
        self.task2_start_topic = '/task2_start'
        self.task2_start_published = False
        self.task2_start_publish_count = 0

        # 到达通道目标后，先连续发送停车命令，再交接任务二。
        # 这样可以避免任务二刚接管时仍残留本节点上一周期的运动命令。
        self.finish_stop_burst_count = 8
        self.finish_stop_interval = 0.02

        # 二维码可提前缓存，但地图x严格大于2.5m才激活倒车。
        self.channel_activation_min_x = 2.50
        self.pending_qr_result = ''
        self.pending_forward_history: List[PathPoint] = []

        # WAIT_HANDOFF -> REVERSE_HISTORY -> FORWARD_CHANNEL -> FINISHED
        # 任一运动阶段进入左右禁区时，会临时切换到 FORBIDDEN_RECOVERY。
        self.operation_mode = 'WAIT_HANDOFF'
        self.forbidden_recovery_resume_mode = 'FORWARD_CHANNEL'
        self.forbidden_recovery_side = ''
        self.forbidden_recovery_target: PathPoint = (0.0, 0.0, 0.0)
        self.forbidden_recovery_safe_frames = 0
        self.forbidden_recovery_start_time = 0.0
        self.forbidden_recovery_start_pose: PathPoint = (0.0, 0.0, 0.0)
        self.forbidden_recovery_failed = False
        self.forward_history: List[PathPoint] = []
        self.reverse_history: List[PathPoint] = []
        self.reverse_history_index = 0
        self.reverse_history_complete = False

        self.cur_pose = [0.0, 0.0, 0.0]
        self.pose_received = False
        self.channel_active = False
        self.is_finished = False
        self.qr_direction = ''

        self.last_cmd_v = 0.0
        self.last_cmd_w = 0.0
        self.qr_success_callback_count = 0
        self.ack_publish_count = 0
        self.cmd_publish_count = 0
        self.cone_coordinates_received = False

        # ==============================================================
        # 7. ROS通信
        # ==============================================================
        self.pose_sub = self.create_subscription(
            Pose2D,
            '/odom_pose',
            self.pose_cb,
            10,
        )

        # 按要求保留原话题，但回调只做接收状态记录，不参与任何控制。
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

        # 与task2订阅端保持一致：可靠、TRANSIENT_LOCAL。
        # 即使task2稍晚启动，也能收到最后一次 start。
        self.task2_start_pub = self.create_publisher(
            String,
            self.task2_start_topic,
            handoff_qos,
        )

        self.timer = self.create_timer(0.025, self.control_loop)  # 40Hz
        self.status_timer = self.create_timer(1.5, self.status_loop)

        self.get_logger().info(
            '🚪 无预测去通道节点启动：历史轨迹倒车；地图x<2.50m后'
            '立即切换PD巡点；无等待、无Dubins/Bezier/Pure Pursuit、'
            '无未来位姿预测、无候选角速度搜索；'
            '唯一前进目标=(2.50,2.00,90°)；RGB实时避障保留，'
            '未收到RGB消息或RGB超时均不阻塞PD巡点；'
            f'禁区自动倒车恢复={self.enable_forbidden_auto_recovery}；'
            f'到达后发布{self.task2_start_topic}: start触发任务二。'
        )

    # ==================================================================
    # 基础工具
    # ==================================================================
    @staticmethod
    def normalize_angle(angle: float) -> float:
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    @staticmethod
    def clamp(value: float, lower: float, upper: float) -> float:
        return max(lower, min(value, upper))

    def actual_pose(self) -> PathPoint:
        """将原始odom转换为统一地图坐标。"""
        return (
            float(self.cur_pose[0]) + self.start_offset_x,
            float(self.cur_pose[1]) + self.start_offset_y,
            self.normalize_angle(float(self.cur_pose[2])),
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

    def current_forbidden_region(self) -> str:
        """返回当前所在禁区；空字符串表示安全。"""
        x, y, _ = self.actual_pose()

        if x < self.map_x_min:
            return 'MAP_LEFT'
        if x >= self.map_x_max:
            return 'MAP_RIGHT'
        if y <= self.map_y_min:
            return 'MAP_BOTTOM'
        if y > self.map_y_max:
            return 'MAP_TOP'

        if self.forbidden_y_min <= y <= self.forbidden_y_max:
            if x <= self.forbidden_left_x_max:
                return 'LEFT_FORBIDDEN'
            if x >= self.forbidden_right_x_min:
                return 'RIGHT_FORBIDDEN'

        return ''

    def current_pose_is_forbidden(self) -> bool:
        """只检查当前地图位置；不预测未来轨迹。"""
        return bool(self.current_forbidden_region())

    @staticmethod
    def recoverable_forbidden_side(region: str) -> str:
        if region == 'LEFT_FORBIDDEN':
            return 'left'
        if region == 'RIGHT_FORBIDDEN':
            return 'right'
        return ''

    def forbidden_recovery_pose_is_safe(self) -> bool:
        """恢复结束条件：已出禁区，且车体回到通道附近的安全侧。"""
        x, y, _ = self.actual_pose()
        if self.current_forbidden_region():
            return False

        corridor_min = self.forbidden_left_x_max + 0.05
        corridor_max = self.forbidden_right_x_min - 0.05
        inside_corridor = corridor_min < x < corridor_max
        below_forbidden_band = y < (
            self.forbidden_y_min - 0.02
        )
        return inside_corridor and below_forbidden_band

    def build_forbidden_recovery_target(self, side: str) -> PathPoint:
        """在车辆后方并朝通道内侧生成一个安全回退点。"""
        x, y, yaw = self.actual_pose()

        rear_x = x - self.forbidden_recovery_reverse_distance * math.cos(yaw)
        rear_y = y - self.forbidden_recovery_reverse_distance * math.sin(yaw)

        inward_sign = 1.0 if side == 'left' else -1.0
        raw_x = rear_x + inward_sign * self.forbidden_recovery_inward_shift
        raw_y = rear_y

        corridor_min = (
            self.forbidden_left_x_max
            + self.forbidden_recovery_corridor_margin
        )
        corridor_max = (
            self.forbidden_right_x_min
            - self.forbidden_recovery_corridor_margin
        )
        target_x = self.clamp(raw_x, corridor_min, corridor_max)

        # 恢复点放在禁止带下方，防止刚横移进通道就再次向前触发禁区。
        target_y = min(
            raw_y,
            self.forbidden_y_min - self.forbidden_recovery_y_margin,
        )
        target_y = self.clamp(
            target_y,
            self.map_y_min + 0.15,
            self.forbidden_y_min - self.forbidden_recovery_y_margin,
        )

        return target_x, target_y, yaw

    def start_forbidden_recovery(self, side: str, reason: str) -> None:
        if not self.enable_forbidden_auto_recovery:
            return
        if side not in ('left', 'right'):
            return

        previous_mode = self.operation_mode
        if previous_mode == 'FORBIDDEN_RECOVERY':
            previous_mode = self.forbidden_recovery_resume_mode

        if previous_mode not in ('REVERSE_HISTORY', 'FORWARD_CHANNEL'):
            previous_mode = 'FORWARD_CHANNEL'

        self.forbidden_recovery_resume_mode = previous_mode
        self.forbidden_recovery_side = side
        self.forbidden_recovery_target = self.build_forbidden_recovery_target(side)
        self.forbidden_recovery_safe_frames = 0
        self.forbidden_recovery_start_time = time.monotonic()
        self.forbidden_recovery_start_pose = self.actual_pose()
        self.forbidden_recovery_failed = False
        self.operation_mode = 'FORBIDDEN_RECOVERY'

        self.reset_rgb_avoidance_state()
        self.reset_waypoint_pd()

        x, y, yaw = self.actual_pose()
        target_x, target_y, _ = self.forbidden_recovery_target
        side_name = '左侧禁区' if side == 'left' else '右侧禁区'
        rear_motion = '车尾向右移入通道' if side == 'left' else '车尾向左移入通道'
        self.get_logger().error(
            f'🚨 进入{side_name}：map=({x:.3f},{y:.3f})，'
            f'yaw={math.degrees(yaw):.1f}°；{reason}。'
            f'切换禁区恢复，目标回退点=({target_x:.2f},{target_y:.2f})，'
            f'{rear_motion}，退出后恢复{previous_mode}'
        )

    def finish_forbidden_recovery(self) -> None:
        resume_mode = self.forbidden_recovery_resume_mode
        side = self.forbidden_recovery_side
        x, y, yaw = self.actual_pose()

        self.forbidden_recovery_side = ''
        self.forbidden_recovery_safe_frames = 0
        self.forbidden_recovery_failed = False
        self.reset_rgb_avoidance_state()
        self.reset_waypoint_pd()

        if resume_mode not in ('REVERSE_HISTORY', 'FORWARD_CHANNEL'):
            resume_mode = 'FORWARD_CHANNEL'
        self.operation_mode = resume_mode

        self.get_logger().warn(
            f'✅ 禁区恢复完成：原禁区={side}，'
            f'map=({x:.3f},{y:.3f})，yaw={math.degrees(yaw):.1f}°，'
            f'连续安全帧={self.forbidden_recovery_safe_frames_required}；'
            f'恢复模式={resume_mode}'
        )

        if resume_mode == 'REVERSE_HISTORY':
            if x < self.reverse_exit_map_x:
                self.finish_reverse_history()
            else:
                self.perform_reverse_history()
            return

        self.perform_waypoint_pd()

    def perform_forbidden_recovery(self) -> None:
        if (
            not self.channel_active
            or self.is_finished
            or not self.pose_received
            or self.operation_mode != 'FORBIDDEN_RECOVERY'
        ):
            return

        x, y, yaw = self.actual_pose()
        region = self.current_forbidden_region()
        current_side = self.recoverable_forbidden_side(region)

        # 若恢复过程中跨到另一侧禁区，重新生成对应侧的回退点。
        if current_side and current_side != self.forbidden_recovery_side:
            self.start_forbidden_recovery(
                current_side,
                '恢复过程中进入另一侧禁区，重新选择回退方向',
            )
            x, y, yaw = self.actual_pose()
            region = self.current_forbidden_region()

        # 地图总边界不属于可恢复禁区，不能继续盲目倒车。
        if region.startswith('MAP_'):
            self.forbidden_recovery_failed = True
            self.publish_stop(
                f'⛔ 禁区恢复时触发地图总边界{region}，停止并检查定位'
            )
            return

        start_x, start_y, _ = self.forbidden_recovery_start_pose
        recovered_distance = math.hypot(x - start_x, y - start_y)
        elapsed = time.monotonic() - self.forbidden_recovery_start_time
        if (
            elapsed > self.forbidden_recovery_max_duration
            or recovered_distance > self.forbidden_recovery_max_distance
        ):
            self.forbidden_recovery_failed = True
            self.publish_stop(
                f'⛔ 禁区恢复超过安全限制：时间={elapsed:.1f}s，'
                f'位移={recovered_distance:.2f}m；停止并检查定位/车体方向'
            )
            return

        target_x, target_y, target_yaw = self.forbidden_recovery_target
        dx = target_x - x
        dy = target_y - y
        target_distance = math.hypot(dx, dy)

        if self.forbidden_recovery_pose_is_safe():
            self.forbidden_recovery_safe_frames += 1
        else:
            self.forbidden_recovery_safe_frames = 0

        if self.forbidden_recovery_safe_frames >= (
            self.forbidden_recovery_safe_frames_required
        ):
            self.finish_forbidden_recovery()
            return

        target_bearing = math.atan2(dy, dx)
        reverse_heading = self.normalize_angle(yaw + math.pi)
        alpha = self.normalize_angle(target_bearing - reverse_heading)
        lookahead = self.clamp(
            target_distance,
            self.forbidden_recovery_lookahead_min,
            self.forbidden_recovery_lookahead_max,
        )
        curvature = 2.0 * math.sin(alpha) / max(lookahead, 1e-3)

        speed_abs = abs(self.forbidden_recovery_speed)
        if target_distance < 0.35:
            speed_abs = max(
                self.forbidden_recovery_min_speed,
                min(speed_abs, 0.70 * target_distance),
            )

        # 与历史轨迹倒车保持相同符号约定：使用正的速度绝对值乘曲率。
        command_w = speed_abs * curvature

        # 左禁区倒车时应让车尾向右（w>0）；右禁区相反。
        expected_sign = 1 if self.forbidden_recovery_side == 'left' else -1
        if abs(target_x - x) > 0.06:
            command_sign = self.angular_direction_sign(command_w)
            if command_sign != expected_sign:
                command_w = expected_sign * max(
                    abs(command_w),
                    self.forbidden_recovery_min_turn_w,
                )

        command_w = self.clamp(command_w, -self.max_w, self.max_w)
        yaw_error = self.normalize_angle(target_yaw - yaw)

        self.execute_drive(
            -speed_abs,
            command_w,
            f'↩️ 禁区倒车恢复[{self.forbidden_recovery_side}]：'
            f'map=({x:.2f},{y:.2f})，目标=({target_x:.2f},{target_y:.2f})，'
            f'距离={target_distance:.2f}m，反向前视误差='
            f'{math.degrees(alpha):.1f}°，进入时航向差='
            f'{math.degrees(yaw_error):.1f}°，安全帧='
            f'{self.forbidden_recovery_safe_frames}/'
            f'{self.forbidden_recovery_safe_frames_required}',
            source='forbidden_recovery',
        )

    def release_stale_rgb_avoidance(self) -> bool:
        """RGB消息缺失不阻塞主运动；只清理已经过期的旧避障命令。"""
        if not self.rgb_avoid_active:
            return False
        if not self.rgb_frame_received:
            self.reset_rgb_avoidance_state()
            self.reset_waypoint_pd()
            return True

        frame_age = time.monotonic() - self.rgb_last_frame_time
        if frame_age <= self.rgb_frame_timeout:
            return False

        self.reset_rgb_avoidance_state()
        self.reset_waypoint_pd()
        self.get_logger().warn(
            f'⚠️ RGB避障帧已{frame_age:.2f}s未更新，按当前无锥桶处理；'
            '清除上一帧避障指令并恢复PD巡点',
            throttle_duration_sec=0.5,
        )
        return True

    # ==================================================================
    # /cone_coordinates：保留话题，不参与控制
    # ==================================================================
    def cone_coordinates_cb(self, msg: String) -> None:
        """只确认数据流存在；不解析为障碍、不规划、不预测。"""
        first_message = not self.cone_coordinates_received
        self.cone_coordinates_received = True
        if first_message:
            self.get_logger().info(
                f'📦 已收到 {self.cone_coordinates_topic}；'
                '当前版本仅保留订阅，不参与PD、避障、规划或安全判断'
            )

    # ==================================================================
    # 二维码和历史轨迹交接
    # ==================================================================
    def parse_qr_handoff_payload(
        self,
        raw_text: str,
    ) -> Tuple[str, List[PathPoint], str]:
        """解析任务一通过 /qr_success 发送的JSON轨迹。"""
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
                last_x, last_y, last_yaw = cleaned[-1]
                if (
                    math.hypot(x - last_x, y - last_y) < 0.01
                    and abs(self.normalize_angle(yaw - last_yaw))
                    < math.radians(1.0)
                ):
                    cleaned[-1] = point
                    continue
            cleaned.append(point)

        return result, cleaned, f'有效轨迹点={len(cleaned)}，无效点={invalid}'

    def publish_channel_ack(self, result: str, reason: str) -> None:
        ack = String()
        ack.data = (
            f'channel_active={self.channel_active};result={result};'
            f'pose_received={self.pose_received};reason={reason}'
        )
        self.channel_ack_pub.publish(ack)
        self.ack_publish_count += 1

    def try_activate_pending_qr(self, reason: str) -> bool:
        if (
            self.channel_active
            or self.is_finished
            or not self.pending_qr_result
            or not self.pose_received
        ):
            return False

        map_x, _, _ = self.actual_pose()
        if map_x <= self.channel_activation_min_x:
            self.get_logger().info(
                f'⏳ 二维码与历史轨迹已缓存，等待地图x>'
                f'{self.channel_activation_min_x:.2f}m；当前map_x={map_x:.3f}m',
                throttle_duration_sec=1.0,
            )
            return False

        result = self.pending_qr_result
        history = list(self.pending_forward_history)
        self.activate_channel_navigation(result, reason, history)
        return self.channel_active

    def activate_channel_navigation(
        self,
        result: str,
        reason: str,
        forward_history: Sequence[PathPoint],
    ) -> None:
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

        minimum_history_x = min(point[0] for point in history)
        if minimum_history_x >= self.reverse_exit_map_x:
            self.channel_active = False
            self.operation_mode = 'WAIT_HANDOFF'
            self.publish_channel_ack(
                result,
                f'拒绝激活：历史轨迹最小地图x={minimum_history_x:.2f}m，'
                f'未覆盖x<{self.reverse_exit_map_x:.2f}m区域',
            )
            self.publish_stop('⛔ 历史轨迹没有覆盖倒车切换区域，禁止盲目倒车')
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
                f'⛔ 当前位姿与历史轨迹末点相差{start_gap:.2f}m，禁止盲目倒车'
            )
            return

        self.qr_direction = result
        self.pending_qr_result = ''
        self.pending_forward_history = []
        self.channel_active = True
        self.operation_mode = 'REVERSE_HISTORY'

        self.forward_history = history
        self.reverse_history = list(reversed(history))
        self.reverse_history_index = 0
        self.reverse_history_complete = False

        self.waypoint_index = 0
        self.reset_waypoint_pd()
        self.reset_rgb_avoidance_state()

        self.publish_channel_ack(
            result,
            f'{reason};mode=REVERSE_HISTORY;path_points={len(history)}',
        )

        self.get_logger().warn(
            f'🚦 去通道激活：二维码={result}，'
            f'当前map=({current_x:.3f},{current_y:.3f},'
            f'{math.degrees(current_yaw):.1f}°)，'
            f'历史点={len(history)}，末点间隙={start_gap:.3f}m；'
            f'先沿历史轨迹倒车，map_x<{self.reverse_exit_map_x:.2f}m后'
            '立即切换PD巡点'
        )

        self.perform_reverse_history()

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

        self.pending_qr_result = result
        self.pending_forward_history = history
        self.qr_direction = result

        if len(history) < self.history_reverse_min_points:
            self.get_logger().error(
                f'❌ 已收到二维码={result}，但历史轨迹不可用：{parse_note}；'
                '不会接管 /cmd_vel'
            )
            return

        self.get_logger().warn(
            f'📩 已缓存二维码和历史轨迹：二维码={result}；{parse_note}'
        )

        if not self.pose_received:
            self.get_logger().info(
                '⏳ 尚未收到位姿；收到位姿且地图x>2.50m后再激活倒车'
            )
            return

        self.try_activate_pending_qr('收到二维码时地图x已超过激活门槛')

    # ==================================================================
    # 历史轨迹倒车
    # ==================================================================
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
        """地图x严格小于2.5m后，同一周期切换PD巡点。"""
        map_x, map_y, map_yaw = self.actual_pose()
        odom_x = self.cur_pose[0]
        odom_y = self.cur_pose[1]

        self.reverse_history_complete = True
        self.operation_mode = 'FORWARD_CHANNEL'
        self.reverse_history_index = len(self.reverse_history) - 1
        self.waypoint_index = 0
        self.reset_waypoint_pd()
        self.reset_rgb_avoidance_state()

        self.get_logger().warn(
            f'🔄 倒车立即切换PD巡点：odom=({odom_x:.3f},{odom_y:.3f})，'
            f'map=({map_x:.3f},{map_y:.3f})，'
            f'map_x<{self.reverse_exit_map_x:.2f}m；'
            f'无等待、无路径规划、无未来轨迹预测，直接前往'
            f'({self.target_map_x:.2f},{self.target_map_y:.2f},'
            f'{math.degrees(self.target_yaw):.0f}°)，'
            f'当前yaw={math.degrees(map_yaw):.1f}°'
        )

        self.perform_waypoint_pd()

    def perform_reverse_history(self) -> None:
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

        # 严格使用地图x判断。
        if x < self.reverse_exit_map_x:
            self.finish_reverse_history()
            return

        index, cross_track = self.nearest_reverse_history_info()
        if cross_track > self.history_reverse_cross_track_stop:
            self.publish_stop(
                f'⛔ 倒放轨迹横向误差={cross_track:.2f}m，超过'
                f'{self.history_reverse_cross_track_stop:.2f}m，停止检查定位'
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
        reverse_exit_gap = max(0.0, x - self.reverse_exit_map_x)
        if abs(alpha) > math.radians(55.0):
            speed_abs = min(speed_abs, 0.12)
        elif reverse_exit_gap < 0.35:
            speed_abs = min(speed_abs, 0.14)

        v_out = -speed_abs
        yaw_error = self.normalize_angle(target_yaw - yaw)
        w_out = speed_abs * curvature + self.history_reverse_yaw_kp * yaw_error

        self.execute_drive(
            v_out,
            w_out,
            f'⏪ 历史轨迹倒放：idx={index}/{len(self.reverse_history)-1}，'
            f'横向误差={cross_track:.2f}m，剩余={remaining:.2f}m，'
            f'距map_x切换门槛={max(0.0, x-self.reverse_exit_map_x):.2f}m，'
            f'反向前视误差={math.degrees(alpha):.1f}°，'
            f'记录航向误差={math.degrees(yaw_error):.1f}°',
            source='history_reverse',
        )

    # ==================================================================
    # 前进PD巡点
    # ==================================================================
    def reset_waypoint_pd(self) -> None:
        self.waypoint_prev_error = 0.0
        self.waypoint_prev_time = 0.0
        self.waypoint_filtered_derivative = 0.0
        self.waypoint_pd_active = False
        self.last_waypoint_v = 0.0
        self.last_waypoint_w = 0.0

    def current_waypoint(self) -> PathPoint:
        index = min(self.waypoint_index, len(self.channel_waypoints) - 1)
        return self.channel_waypoints[index]

    def check_channel_arrival(self) -> bool:
        if self.operation_mode != 'FORWARD_CHANNEL':
            return False
        if not self.channel_active or self.is_finished or not self.pose_received:
            return self.is_finished

        x, y, yaw = self.actual_pose()
        position_error = math.hypot(self.target_map_x - x, self.target_map_y - y)
        yaw_error = abs(self.normalize_angle(self.target_yaw - yaw))
        in_handoff_region = (
            self.arrival_corridor_x_min < x < self.arrival_corridor_x_max
            and y > self.arrival_corridor_y_min
        )

        if not in_handoff_region:
            return False

        self.complete_channel_navigation(
            position_error=position_error,
            yaw_error=yaw_error,
        )
        return True

    def publish_task2_start(self) -> None:
        """只发布一次任务二启动指令。"""
        if self.task2_start_published:
            return

        message = String()
        message.data = 'start'
        self.task2_start_pub.publish(message)
        self.task2_start_published = True
        self.task2_start_publish_count += 1

        self.get_logger().warn(
            f'📤 已发布 {self.task2_start_topic}: start；'
            '去通道节点已停止控制，任务二可以接管 /cmd_vel'
        )

    def complete_channel_navigation(
        self,
        position_error: float,
        yaw_error: float,
    ) -> None:
        """先停车并退出控制，再触发task2，避免两个节点抢占速度。"""
        if self.is_finished:
            return

        # 先关闭本节点运动状态。即使在停车脉冲期间有其他回调触发，
        # control_loop也不会再发布非零速度。
        self.is_finished = True
        self.channel_active = False
        self.operation_mode = 'FINISHED'
        self.reset_rgb_avoidance_state()
        self.reset_waypoint_pd()

        self.get_logger().warn(
            f'🏁 到达通道目标：位置误差={position_error:.3f}m，'
            f'航向误差={math.degrees(yaw_error):.1f}°；'
            '先连续停车，再触发任务二'
        )

        for _ in range(self.finish_stop_burst_count):
            self.publish_zero()
            time.sleep(self.finish_stop_interval)

        # 停车命令全部发完后再交接，避免start发出后继续用零速度压住task2。
        self.publish_task2_start()

    def perform_waypoint_pd(self) -> None:
        """不规划路径、不预测未来位置，直接对当前目标点做PD。"""
        if self.operation_mode != 'FORWARD_CHANNEL':
            return
        if not self.channel_active or self.is_finished or not self.pose_received:
            return
        if self.check_channel_arrival():
            return

        x, y, yaw = self.actual_pose()
        odom_x = self.cur_pose[0]
        odom_y = self.cur_pose[1]
        target_x, target_y, target_yaw = self.current_waypoint()

        dx = target_x - x
        dy = target_y - y
        distance = math.hypot(dx, dy)

        # 若以后增加中间点，到达中间点后立即切换下一点。
        is_final_waypoint = self.waypoint_index >= len(self.channel_waypoints) - 1
        if not is_final_waypoint and distance <= self.intermediate_waypoint_tolerance:
            self.waypoint_index += 1
            self.reset_waypoint_pd()
            self.get_logger().warn(
                f'📍 到达巡航点，切换到第{self.waypoint_index + 1}/'
                f'{len(self.channel_waypoints)}个目标点'
            )
            self.perform_waypoint_pd()
            return

        if distance > 0.06:
            target_bearing = math.atan2(dy, dx)
        else:
            target_bearing = target_yaw

        blend = self.clamp(
            (self.waypoint_heading_blend_distance - distance)
            / max(self.waypoint_heading_blend_distance, 1e-6),
            0.0,
            1.0,
        )
        desired_heading = self.normalize_angle(
            target_bearing
            + blend * self.normalize_angle(target_yaw - target_bearing)
        )
        error = self.normalize_angle(desired_heading - yaw)

        now = time.monotonic()
        if self.waypoint_prev_time <= 0.0:
            derivative = 0.0
        else:
            dt = self.clamp(now - self.waypoint_prev_time, 0.02, 0.25)
            derivative = self.normalize_angle(error - self.waypoint_prev_error) / dt
            derivative = self.clamp(
                derivative,
                -self.waypoint_derivative_limit,
                self.waypoint_derivative_limit,
            )

        self.waypoint_filtered_derivative = (
            self.waypoint_derivative_alpha * derivative
            + (1.0 - self.waypoint_derivative_alpha)
            * self.waypoint_filtered_derivative
        )
        self.waypoint_prev_error = error
        self.waypoint_prev_time = now
        self.waypoint_pd_active = True

        command_w = (
            self.waypoint_kp * error
            + self.waypoint_kd * self.waypoint_filtered_derivative
        )
        command_w = self.clamp(command_w, -self.max_w, self.max_w)

        command_v = self.clamp(
            self.waypoint_linear_kp * distance,
            self.waypoint_min_v,
            self.waypoint_max_v,
        )

        heading_abs = abs(error)
        if heading_abs >= self.waypoint_sharp_turn_angle:
            command_v = min(command_v, self.waypoint_sharp_turn_v)
        else:
            heading_scale = max(
                0.45,
                1.0 - heading_abs / math.radians(110.0),
            )
            command_v = max(self.waypoint_min_v, command_v * heading_scale)

        # 只做距离减速，不做路径预测。
        if distance < 0.55:
            command_v = min(command_v, 0.8)#0.45
        if distance < 0.30:
            command_v = min(command_v, 0.6)#0.28

        self.last_waypoint_v = command_v
        self.last_waypoint_w = command_w

        self.execute_drive(
            command_v,
            command_w,
            f'🎯 PD巡点[{self.waypoint_index + 1}/{len(self.channel_waypoints)}]：'
            f'odom=({odom_x:.3f},{odom_y:.3f})，'
            f'map=({x:.3f},{y:.3f})，'
            f'目标=({target_x:.2f},{target_y:.2f},'
            f'{math.degrees(target_yaw):.0f}°)，'
            f'距离={distance:.2f}m，期望航向={math.degrees(desired_heading):.1f}°，'
            f'角度误差={math.degrees(error):.1f}°，'
            f'D={self.waypoint_filtered_derivative:.2f}',
            source='waypoint_pd',
        )

    # ==================================================================
    # RGB实时避障
    # ==================================================================
    @staticmethod
    def rgb_target_is_cone(target) -> bool:
        target_type = str(getattr(target, 'type', '')).strip().lower()
        if not target_type:
            return True
        if any(token in target_type for token in ('qr', 'barcode', 'image_board')):
            return False
        return True

    def _build_rgb_edge_boundaries(self) -> None:
        points = sorted(
            [(float(y), float(x)) for y, x in self.rgb_edge_measure_points],
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
                y0, y1 = y_values[index - 1], y_values[index]
                x0, x1 = x_values[index - 1], x_values[index]
                ratio = (y - y0) / max(y1 - y0, 1e-6)
                return x0 + ratio * (x1 - x0)

        return x_values[-1]

    def get_rgb_edge_boundaries(self, bottom_y: float) -> Tuple[float, float]:
        left_x = (
            self._interpolate_rgb_left_edge_x(bottom_y)
            + self.rgb_edge_ignore_margin_px
        )
        left_x = self.clamp(left_x, 0.0, self.rgb_image_center_x - 1.0)
        right_x = 2.0 * self.rgb_image_center_x - left_x
        return left_x, right_x

    def rgb_is_in_edge_ignore_zone(self, center_x: float, bottom_y: float) -> bool:
        left_x, right_x = self.get_rgb_edge_boundaries(bottom_y)
        return center_x < left_x or center_x > right_x

    def collect_rgb_obstacle_candidates(self, msg: PerceptionTargets) -> list:
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

    def reset_rgb_avoidance_state(self) -> None:
        self.rgb_avoid_active = False
        self.avoid_locked_sign = 0
        self.avoid_hold_v = 0.0
        self.avoid_hold_w = 0.0
        self.rgb_base_waypoint_v = 0.0
        self.rgb_base_waypoint_w = 0.0
        self.reset_avoid_pd()

    def compute_rgb_pd_w(
        self,
        center_x: float,
        bottom_y: float,
    ) -> Tuple[float, float, float, float, float, float, float]:
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
            dt = self.clamp(now - self.rgb_avoid_prev_time, 0.01, 0.20)
            derivative = (control_error - self.rgb_avoid_prev_error) / dt
            derivative = self.clamp(
                derivative,
                -self.rgb_avoid_derivative_limit,
                self.rgb_avoid_derivative_limit,
            )

        self.rgb_avoid_filtered_derivative = (
            self.rgb_avoid_derivative_alpha * derivative
            + (1.0 - self.rgb_avoid_derivative_alpha)
            * self.rgb_avoid_filtered_derivative
        )
        self.rgb_avoid_prev_error = control_error
        self.rgb_avoid_prev_time = now

        d1 = (bottom_y - self.rgb_trigger_bottom_y) * self.rgb_cone_near_k
        d1 = self.clamp(d1, 0.0, self.rgb_cone_near_max)

        d2 = (
            self.rgb_avoid_kp * control_error
            + self.rgb_avoid_kd * self.rgb_avoid_filtered_derivative
        )
        cone_w = self.clamp(
            d1 * d2,
            -min(self.max_w, self.rgb_avoid_w_limit),
            min(self.max_w, self.rgb_avoid_w_limit),
        )

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
        candidates = self.collect_rgb_obstacle_candidates(msg)
        if not candidates:
            return False

        bottom_y, center_x, confidence, target_type = candidates[0]

        if not self.rgb_avoid_active:
            # 记录进入避障前的PD基础值；不保存负的倒车速度。
            self.rgb_base_waypoint_v = (
                self.last_waypoint_v
                if self.last_waypoint_v > 1e-3
                else self.waypoint_min_v
            )
            self.rgb_base_waypoint_w = self.last_waypoint_w
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
        ) = self.compute_rgb_pd_w(center_x, bottom_y)

        command_v = self.clamp(
            self.rgb_base_waypoint_v,
            self.waypoint_min_v,
            self.waypoint_max_v,
        )
        command_w_raw = self.rgb_base_waypoint_w + cone_w

        # 不允许目标点PD把视觉避障方向反转。
        cone_sign = self.angular_direction_sign(cone_w)
        fused_sign = self.angular_direction_sign(command_w_raw)
        if cone_sign != 0 and fused_sign not in (0, cone_sign):
            command_w_raw = cone_w

        command_w = self.clamp(command_w_raw, -self.max_w, self.max_w)

        self.rgb_avoid_active = True
        self.avoid_hold_v = command_v
        self.avoid_hold_w = command_w
        self.last_avoid_center_x = center_x
        self.last_avoid_bottom_y = bottom_y
        self.last_avoid_confidence = confidence
        self.last_avoid_target_type = target_type

        side_text = '画面右侧' if center_x > self.rgb_image_center_x else '画面左侧'
        if abs(center_x - self.rgb_image_center_x) < 1.0:
            side_text = '画面正中'

        self.execute_drive(
            command_v,
            command_w,
            f'⚡ RGB锥桶PD：center_x={center_x:.1f}px({side_text})，'
            f'bottom_y={bottom_y:.1f}px，conf={confidence:.2f}，'
            f'd1={d1:.3f}，中心接近度={center_closeness:.1f}px，'
            f'锁定方向={self.direction_name(self.avoid_locked_sign)}，'
            f'误差={control_error:+.1f}，D={derivative:+.1f}/s，'
            f'd2={d2:+.3f}，W基础={self.rgb_base_waypoint_w:+.3f}，'
            f'W锥桶={cone_w:+.3f}，W融合={command_w:+.3f}',
            source='cone',
        )
        return True

    def finish_rgb_avoidance_immediately(self) -> None:
        self.reset_rgb_avoidance_state()
        self.reset_waypoint_pd()
        self.get_logger().warn(
            '✅ 当前RGB帧无危险锥桶，立即退出避障并恢复PD巡点'
        )
        self.perform_waypoint_pd()

    def rgb_obstacle_cb(self, msg: PerceptionTargets) -> None:
        self.rgb_frame_id += 1
        self.rgb_frame_received = True
        self.rgb_last_frame_time = time.monotonic()

        if not self.enable_rgb_realtime_avoidance:
            return
        if not self.channel_active or self.is_finished or not self.pose_received:
            return
        if self.operation_mode != 'FORWARD_CHANNEL':
            # 前置摄像头不参与倒车控制。
            return

        if self.detect_rgb_hazard(msg):
            return

        if self.rgb_avoid_active:
            self.finish_rgb_avoidance_immediately()

    # ==================================================================
    # ROS状态机和速度发布
    # ==================================================================
    def pose_cb(self, msg: Pose2D) -> None:
        first_pose = not self.pose_received
        self.cur_pose = [float(msg.x), float(msg.y), float(msg.theta)]
        self.pose_received = True

        if first_pose:
            x, y, yaw = self.actual_pose()
            self.get_logger().info(
                f'📍 首次地图位姿=({x:.3f},{y:.3f})，'
                f'yaw={math.degrees(yaw):.1f}°'
            )

        if (
            not self.channel_active
            and not self.is_finished
            and self.pending_qr_result
        ):
            self.try_activate_pending_qr('位姿更新后达到激活门槛')

    def control_loop(self) -> None:
        if not self.channel_active or self.is_finished or not self.pose_received:
            return

        if self.operation_mode == 'FORBIDDEN_RECOVERY':
            if not self.enable_forbidden_auto_recovery:
                # 兼容运行中通过调试手段关闭开关的情况：立即停止恢复动作，
                # 返回原运动状态，但只要车辆仍在禁区内就持续停车。
                self.operation_mode = self.forbidden_recovery_resume_mode
                self.forbidden_recovery_side = ''
                self.forbidden_recovery_safe_frames = 0
                self.reset_rgb_avoidance_state()
                self.reset_waypoint_pd()
                self.publish_stop('⛔ 禁区自动倒车恢复已关闭，立即停止恢复动作')
                return
            self.perform_forbidden_recovery()
            return

        forbidden_region = self.current_forbidden_region()
        recover_side = self.recoverable_forbidden_side(forbidden_region)
        if self.enable_forbidden_auto_recovery and recover_side:
            x, y, _ = self.actual_pose()
            self.start_forbidden_recovery(
                recover_side,
                f'当前地图位置=({x:.3f},{y:.3f})',
            )
            self.perform_forbidden_recovery()
            return

        if forbidden_region:
            x, y, _ = self.actual_pose()
            if recover_side and not self.enable_forbidden_auto_recovery:
                reason = '禁区自动倒车恢复开关=False'
            else:
                reason = '该区域不可自动恢复'
            self.publish_stop(
                f'❌ 当前地图位置进入硬禁止区{forbidden_region}：'
                f'({x:.3f},{y:.3f})；{reason}，立即停车'
            )
            return

        if self.operation_mode == 'REVERSE_HISTORY':
            self.perform_reverse_history()
            return

        if self.operation_mode != 'FORWARD_CHANNEL':
            return

        if self.check_channel_arrival():
            return

        # RGB从未发布消息时不停车；如果上一帧曾触发避障但数据流随后中断，
        # 超过rgb_frame_timeout后按“当前没有锥桶”释放旧避障命令。
        if self.enable_rgb_realtime_avoidance:
            self.release_stale_rgb_avoidance()

        if self.enable_rgb_realtime_avoidance and self.rgb_avoid_active:
            # 危险帧仍在有效时间内时，在两帧之间保持上一帧避障指令。
            self.execute_drive(
                self.avoid_hold_v,
                self.avoid_hold_w,
                '🧲 保持当前有效RGB避障指令；空帧或超时后恢复PD巡点',
                source='cone',
            )
            return

        self.perform_waypoint_pd()

    def publish_zero(self) -> None:
        """发布单次零速度，不改变任务状态。"""
        self.last_cmd_v = 0.0
        self.last_cmd_w = 0.0
        self.cmd_pub.publish(Twist())
        self.cmd_publish_count += 1

    def publish_stop(self, reason: str) -> None:
        self.publish_zero()
        self.get_logger().warn(reason, throttle_duration_sec=0.5)

    def execute_drive(self, v: float, w: float, log_tag: str, source: str) -> None:
        """
        最终速度发布：
        - 只做速度/角速度限幅；
        - 只做阿克曼最小转弯半径限制；
        - 只检查当前地图位置；
        - 不预测未来轨迹，不搜索候选角速度，不生成路径。
        """
        forbidden_region = self.current_forbidden_region()
        recover_side = self.recoverable_forbidden_side(forbidden_region)
        if forbidden_region:
            recovery_command_allowed = (
                self.enable_forbidden_auto_recovery
                and source == 'forbidden_recovery'
                and bool(recover_side)
                and float(v) < 0.0
            )
            if not recovery_command_allowed:
                self.publish_stop(
                    f'⛔ 当前位姿位于{forbidden_region}，'
                    f'禁区自动恢复={self.enable_forbidden_auto_recovery}，'
                    '拒绝当前运动命令'
                )
                return

        v = self.clamp(float(v), self.history_reverse_speed, self.max_v)
        w = self.clamp(float(w), -self.max_w, self.max_w)

        radius_note = ''
        if abs(v) > 1e-6 and abs(w) > 1e-6:
            max_w_by_radius = min(self.max_w, abs(v) / self.safe_turn_radius)
            if abs(w) > max_w_by_radius:
                old_w = w
                w = math.copysign(max_w_by_radius, w)
                radius_note = f' | 半径限制W:{old_w:.2f}->{w:.2f}'

        self.last_cmd_v = v
        self.last_cmd_w = w

        twist = Twist()
        twist.linear.x = v
        twist.angular.z = w
        self.cmd_pub.publish(twist)
        self.cmd_publish_count += 1

        self.get_logger().info(
            f'{log_tag}{radius_note} | V={v:.2f}, W={w:.2f}，'
            '无未来轨迹预测',
            throttle_duration_sec=0.5,
        )

    def status_loop(self) -> None:
        if self.is_finished:
            self.get_logger().info(
                f'📊 去通道已完成；task2_start已发布='
                f'{self.task2_start_published}，'
                f'发布次数={self.task2_start_publish_count}'
            )
            return

        if (
            not self.channel_active
            and self.pose_received
            and self.pending_qr_result
        ):
            x, y, yaw = self.actual_pose()
            self.get_logger().info(
                f'📨 等待激活：二维码={self.pending_qr_result}，'
                f'map=({x:.2f},{y:.2f})，yaw={math.degrees(yaw):.1f}°，'
                f'要求map_x>{self.channel_activation_min_x:.2f}m'
            )
            return

        if not self.channel_active or not self.pose_received:
            return

        x, y, yaw = self.actual_pose()

        if self.operation_mode == 'FORBIDDEN_RECOVERY':
            target_x, target_y, _ = self.forbidden_recovery_target
            elapsed = max(0.0, time.monotonic() - self.forbidden_recovery_start_time)
            self.get_logger().info(
                f'📊 禁区恢复状态：side={self.forbidden_recovery_side}，'
                f'map=({x:.2f},{y:.2f})，yaw={math.degrees(yaw):.1f}°，'
                f'回退目标=({target_x:.2f},{target_y:.2f})，'
                f'安全帧={self.forbidden_recovery_safe_frames}/'
                f'{self.forbidden_recovery_safe_frames_required}，'
                f'耗时={elapsed:.1f}s，恢复后={self.forbidden_recovery_resume_mode}，'
                f'失败锁定={self.forbidden_recovery_failed}，'
                f'自动恢复开关={self.enable_forbidden_auto_recovery}'
            )
            return

        if self.operation_mode == 'REVERSE_HISTORY':
            index, cross_track = self.nearest_reverse_history_info()
            remaining = self.remaining_reverse_history_distance()
            self.get_logger().info(
                f'📊 倒车状态：map=({x:.2f},{y:.2f})，'
                f'yaw={math.degrees(yaw):.1f}°，'
                f'轨迹={index}/{max(0,len(self.reverse_history)-1)}，'
                f'横向误差={cross_track:.2f}m，剩余={remaining:.2f}m，'
                f'切换条件=map_x<{self.reverse_exit_map_x:.2f}m'
            )
            return

        target_x, target_y, target_yaw = self.current_waypoint()
        distance = math.hypot(target_x - x, target_y - y)
        yaw_error = abs(self.normalize_angle(target_yaw - yaw))

        self.get_logger().info(
            f'📊 PD巡点状态：map=({x:.2f},{y:.2f})，'
            f'yaw={math.degrees(yaw):.1f}°，'
            f'巡点={self.waypoint_index + 1}/{len(self.channel_waypoints)}，'
            f'目标=({target_x:.2f},{target_y:.2f},'
            f'{math.degrees(target_yaw):.0f}°)，'
            f'距离={distance:.2f}m，最终航向误差={math.degrees(yaw_error):.1f}°，'
            f'RGB帧已接收={self.rgb_frame_received}，'
            f'RGB避障中={self.rgb_avoid_active}，'
            f'禁区自动恢复={self.enable_forbidden_auto_recovery}，'
            'RGB无消息不阻塞，路径规划=关闭，未来轨迹预测=关闭'
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
