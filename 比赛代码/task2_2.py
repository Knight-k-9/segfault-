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
    边界判断点不再使用小车中心，而使用沿当前航向前方 0.20m 的车头点：
        head_x = x_actual + 0.20 * cos(yaw)
        head_y = y_actual + 0.20 * sin(yaw)
    上边界：仅 head_x < 3.9 时生效，e = head_y - y_upper_boundary
    下边界：e = head_y - y_lower_boundary
    右边界：物理边界 x=5.0，车头距边界 0.20m 开始保护，
              e = x_right_protect_start - head_x（负值），输出左转角速度
    w_boundary = -(boundary_kp * e + boundary_kd * de/dt)

3. 锥桶 PD：
    d1 = (bottom_y - cone_trigger_y) * cone_near_k
    center_distance = abs(cone_center_x - image_center_x)
    center_closeness = max(0, 320 - center_distance)
    control_error = locked_turn_sign * center_closeness
    d2 = cone_kp * control_error + cone_kd * d(control_error)/dt
    w_cone = d1 * d2

    现在横向规律为：锥桶越靠近画面中心，角速度越大；
    锥桶越靠近画面边缘，角速度越小。一次连续避障期间锁定转向方向，
    防止检测框跨过中心时角速度符号来回翻转。

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

二维码交接规则：
- 二维码可以提前识别并缓存；
- 提前识别时任务1继续巡航，不会立即停发 /cmd_vel；
- 只有 actual_x > 2.5m 且当前不处于锥桶避障状态时，才先发布零速度，
  再发送 /qr_success 完成交接；
- 若越过交接门槛时正在避障，只缓存二维码结果并继续避障，避障结束后再交接；
- 历史轨迹只记录实际位姿，不再人为添加固定大厅坐标。
"""

import json
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
        self.target_map_x = 5.0-0.25
        self.target_map_y = 2.0-0.20

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

        # 横向控制改成“越靠近画面中心，角速度越大”。
        # center_closeness_px = max(0, 320 - abs(center_x - 320))
        # 由于新的控制误差最大为 320px，P/D 参数按该量级重新缩放，
        # 避免继续使用原 cone_kp=0.3 时绝大多数情况直接撞到角速度限幅。
        self.cone_center_effect_width_px = 320.0
        self.cone_center_direction_deadband_px = 8.0
        self.cone_default_turn_sign = -1.0  # 中心死区首次触发时默认右转

        self.cone_kp = 0.40 #0.0040
        self.cone_kd = 0.0002
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
        # 边界检测使用车头点，而不是小车中心。
        # 车头点 = 小车中心 + 当前航向前方 front_offset_m。
        self.front_offset_m = 0.20

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

        self.boundary_kp = 150.0 #76
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

        # 二维码允许提前识别，但不能立即停止任务1巡航。
        # 只有车辆实际地图 x 严格大于该门槛时，才停车并向通道节点交接。
        # qutongdao.py 使用相同门槛，二者必须保持一致。
        self.channel_handoff_min_x = 2.5

        # 历史轨迹记录参数。轨迹使用实际地图坐标，随 /qr_success 一起交接。
        # 记录会持续到任务一真正停止输出 /cmd_vel，避免二维码提前识别后
        # 到 actual_x>2.5 之间出现未记录的路径空档。
        self.path_record_min_distance = 0.04
        self.path_record_min_yaw_change = math.radians(3.0)
        self.path_record_max_points = 3000

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
        # 最近一次用于边界判断的车头地图坐标，供调试输出使用。
        self.boundary_head_x = 0.0
        self.boundary_head_y = 0.0

        # 锥桶 PD 状态，只在收到新检测帧时更新。
        # 两个检测帧之间持续使用最近一次计算结果。
        self.cone_control_active = False
        self.current_cone_w = 0.0
        self.cone_prev_error = None
        self.cone_prev_time = None
        self.cone_d_filtered = 0.0
        self.cone_turn_sign = 0.0
        self.last_cone_center_x = 0.0
        self.last_cone_bottom_y = 0.0

        # 二维码交接状态。
        self.qr_result_received = False
        self.handoff_complete = False
        self.qr_result = ''
        self.handoff_publish_count = 0
        self.channel_ack_received = False
        self.channel_ack_data = ''

        # 任务一实际走过的轨迹栈：(actual_x, actual_y, yaw)。
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
            f'🚗 边界检测点：车体中心沿当前航向前方 '
            f'{self.front_offset_m:.2f}m 的车头点'
        )
        self.get_logger().info(
            f'🧱 边界规则：上边界在车头x=['
            f'{self.upper_boundary_disable_x_start:.2f},'
            f'{self.upper_boundary_disable_x_end:.2f}] 禁用；'
            f'右边界 x={self.x_right_boundary:.2f}，'
            f'x>={self.x_right_protect_start:.2f} 开始左转保护'
        )
        self.get_logger().info(
            f'🔄 二维码交接：允许提前缓存；仅当actual_x>'
            f'{self.channel_handoff_min_x:.2f}m且锥桶避障未激活时，'
            '才停车并发布 /qr_success'
        )
        self.get_logger().info(
            f'🧠 历史轨迹记录已开启：距离采样={self.path_record_min_distance:.2f}m，'
            f'航向采样={math.degrees(self.path_record_min_yaw_change):.1f}°；'
            '只记录实际位姿，不再强制添加固定大厅点；'
            '轨迹将通过原 /qr_success 话题一并交接'
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
        """按距离/航向变化压入实际轨迹栈；force用于交接前补最后一点。"""
        if self.path_recording_frozen:
            return

        point = (
            float(actual_x),
            float(actual_y),
            float(self.normalize_angle(yaw)),
        )

        if not self.recorded_path:
            # 首点只记录任务一实际收到的位姿，不人为插入固定大厅坐标。
            self.recorded_path.append(point)
            self.get_logger().info(
                f'🧠 历史轨迹首点=({point[0]:.3f},{point[1]:.3f},'
                f'{math.degrees(point[2]):.1f}°)'
            )
            return

        last_x, last_y, last_yaw = self.recorded_path[-1]
        distance = math.hypot(point[0] - last_x, point[1] - last_y)
        yaw_change = abs(self.normalize_angle(point[2] - last_yaw))
        if not force and (
            distance < self.path_record_min_distance
            and yaw_change < self.path_record_min_yaw_change
        ):
            return

        # force时若最后一点几乎相同则直接替换，避免重复终点。
        if force and distance < 0.005 and yaw_change < math.radians(0.5):
            self.recorded_path[-1] = point
        else:
            self.recorded_path.append(point)

        if len(self.recorded_path) > self.path_record_max_points:
            # 保留首点和最新点，对中间轨迹做2倍降采样，避免String消息无限增长。
            first = self.recorded_path[0]
            middle = self.recorded_path[1:-1:2]
            last = self.recorded_path[-1]
            self.recorded_path = [first, *middle, last]
            self.get_logger().warn(
                f'⚠️ 轨迹点超过{self.path_record_max_points}，已自动降采样；'
                f'当前点数={len(self.recorded_path)}'
            )

    def _build_handoff_payload_locked(self):
        """使用原 /qr_success String 携带二维码结果和压缩轨迹。"""
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

            # 任务一仍掌握控制权时持续记录实际位姿。二维码可能提前识别，
            # 但直到真正交接前仍需记录，确保倒放路径从当前停车点连续开始。
            if not self.handoff_complete and not self.is_finished:
                self._record_path_pose_locked(
                    actual_x, actual_y, self.cur_pose[2]
                )

            # 二维码可能在车辆到达交接区域前就被识别。
            # 只有越过门槛且当前不处于锥桶避障状态时才正式交接。
            # 若正在避障，则继续保留任务一控制权；避障结束后的下一帧位姿
            # 会再次检查并完成交接。
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
        actual_yaw = self.cur_pose[2]

        # 使用车头点进行边界判断。车头补偿会随航向角旋转：
        # yaw=0       -> 车头点位于中心点 x 正方向 0.20m；
        # yaw=+pi/2   -> 车头点位于中心点 y 正方向 0.20m；
        # yaw=-pi/2   -> 车头点位于中心点 y 负方向 0.20m。
        head_x = actual_x + self.front_offset_m * math.cos(actual_yaw)
        head_y = actual_y + self.front_offset_m * math.sin(actual_yaw)
        self.boundary_head_x = head_x
        self.boundary_head_y = head_y

        # 右边界优先级最高：车头进入右侧保护区时必须向左转。
        # error 在保护区内为负值，因此沿用统一公式
        # w_boundary = -(Kp*error + Kd*error_d) 后得到正角速度（左转）。
        if head_x >= self.x_right_protect_start:
            mode = '右边界PD'
            error = self.x_right_protect_start - head_x

        # 车头 x 位于 3.9~5.0 时，明确取消上边界保护。
        elif (
            head_y >= self.y_upper_boundary
            and not (
                self.upper_boundary_disable_x_start
                <= head_x
                <= self.upper_boundary_disable_x_end
            )
        ):
            mode = '上边界PD'
            error = head_y - self.y_upper_boundary

        elif head_y <= self.y_lower_boundary:
            mode = '下边界PD'
            error = head_y - self.y_lower_boundary
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
        self.cone_turn_sign = 0.0
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
        horizontal_offset = center_x - image_center_x

        # 一次连续避障期间锁定方向：
        # 锥桶首次在右侧 -> 左转(w>0)；首次在左侧 -> 右转(w<0)。
        # 锁定只在当前连续检测期间有效；当前帧无触发锥桶时会清零。
        if self.cone_turn_sign == 0.0:
            if horizontal_offset > self.cone_center_direction_deadband_px:
                self.cone_turn_sign = 1.0
            elif horizontal_offset < -self.cone_center_direction_deadband_px:
                self.cone_turn_sign = -1.0
            else:
                self.cone_turn_sign = self.cone_default_turn_sign

        center_distance = abs(horizontal_offset)
        center_closeness_px = max(
            0.0,
            self.cone_center_effect_width_px - center_distance,
        )

        # 符号决定避障方向，绝对值决定角速度大小。
        # center_closeness_px 越大，说明锥桶越靠近画面中心。
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

        direction_text = '左转' if self.cone_turn_sign > 0.0 else '右转'
        self.get_logger().info(
            f'🟠 中心增强锥桶PD：bottom_y={bottom_y:.0f}，'
            f'd1={d1:.3f}，center_x={center_x:.1f}px，'
            f'距中心={center_distance:.1f}px，'
            f'中心接近度={center_closeness_px:.1f}px，'
            f'锁定方向={direction_text}，'
            f'控制误差={control_error:+.1f}，'
            f'D={control_error_d:+.1f}/s，'
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
            cone_avoid_active = self.cone_control_active
            can_handoff_now = (
                actual_x is not None
                and actual_x > self.channel_handoff_min_x
                and not cone_avoid_active
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
            wait_text = '收到位姿并满足交接条件后再交接'
        elif actual_x <= self.channel_handoff_min_x:
            position_text = (
                f'当前actual_x={actual_x:.3f}m，尚未超过'
                f'{self.channel_handoff_min_x:.2f}m'
            )
            wait_text = '越过交接门槛后再交接'
        else:
            position_text = (
                f'当前actual_x={actual_x:.3f}m，已超过'
                f'{self.channel_handoff_min_x:.2f}m，但锥桶避障正在执行'
            )
            wait_text = '等待锥桶避障结束后再交接'

        self.get_logger().warn(
            f'📥 已缓存二维码结果：“{result}”；{position_text}。'
            f'任务1继续保持控制，{wait_text}并发布 /qr_success'
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

            # 最终安全门：交接只能发生在锥桶避障未激活时。
            # 即使位姿回调与检测回调并发，避障在条件判断后突然激活，
            # 这里也会阻止停车、冻结轨迹和发布 /qr_success。
            if self.cone_control_active:
                self.get_logger().info(
                    f'⏳ 已满足交接位置条件，但当前处于锥桶避障状态：'
                    f'W锥桶={self.current_cone_w:+.3f}；等待避障结束后再交接',
                    throttle_duration_sec=0.5,
                )
                return

            if actual_x <= self.channel_handoff_min_x:
                self.get_logger().info(
                    f'⏳ 二维码结果已缓存，等待actual_x>'
                    f'{self.channel_handoff_min_x:.2f}m；'
                    f'当前x={actual_x:.3f}m',
                    throttle_duration_sec=1.0,
                )
                return

            result = self.qr_result

            # 交接前强制补入当前停车位姿，然后冻结轨迹并生成固定载荷。
            actual_y = self.cur_pose[1] + self.start_offset_y
            self._record_path_pose_locked(
                actual_x, actual_y, self.cur_pose[2], force=True
            )
            self.path_recording_frozen = True
            if len(self.recorded_path) < 2:
                self.get_logger().error(
                    f'无法完成交接：历史轨迹点不足，当前仅{len(self.recorded_path)}个'
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
            f'已冻结并交接{len(self.recorded_path)}个历史轨迹点；'
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
            f'🧮 角速度调试[{source}/{mode}]：'
            f'W目标={self.target_w:+.3f}，'
            f'W边界={self.boundary_w:+.3f}，'
            f'W锥桶={self.current_cone_w:+.3f}，'
            f'W三项和={w_raw:+.3f}，'
            f'W总输出={w_total:+.3f}，'
            f'车头=({self.boundary_head_x:.3f},'
            f'{self.boundary_head_y:.3f})'
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
        elif '角速度调试' in log_tag:
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