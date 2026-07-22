#!/usr/bin/env python3
# -*- coding: utf-8 -*-


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

        # ==========================================
        # [配置区 1]：PID 寻点参数
        # ==========================================
        self.target_map_x = 5.0
        self.target_map_y = 2.0
        self.start_offset_x = 0.55  # 地图坐标偏移
        self.start_offset_y = 0.22

        # PID 增益（关键调参项）
        self.kp_linear = 1.0        # 线速度比例系数：越大起步越快
        self.kp_angular = 3.0       # 角速度比例系数：越大转向越灵敏

        self.max_v = 1.0            # 最大巡航线速度
        self.max_w = 2.0            # 最大转向角速度
        self.arrival_tolerance = 0.5

        # ==========================================
        # [配置区 1.1]：二维码结果直接交接与目标点停车
        # ==========================================
        # 不再订阅二维码 YOLO 检测，也不再根据二维码画面位置停车。
        # 只要 /qr_direction_result 收到非空解析结果，就立即向通道导航节点
        # 发布 /qr_success，并永久停止本巡航节点发布 /cmd_vel。
        self.qr_result_topic = '/qr_direction_result'
        self.qr_success_topic = '/qr_success'
        self.channel_ack_topic = '/channel_navigation_ack'

        # 只有到达第一目标点 (5.0, 2.0) 时才建立停车锁存。
        # 先连续发布多条零速度，再在短时间内以 200 Hz 重复零速度，
        # 最后由 40 Hz 控制循环持续保持停车。
        self.fast_stop_burst_count = 4
        self.fast_stop_hold_sec = 0.35
        self.fast_stop_timer_period = 0.005

        # 交接后，在收到通道节点 ACK 前每 0.5 秒重发一次 qr_success，
        # 最多重发 20 次，降低节点启动时序导致的交接消息丢失风险。
        self.handoff_repeat_period = 0.5
        self.handoff_repeat_limit = 20

        # ==========================================
        # [配置区 2]：视觉避障参数
        # ==========================================
        self.conf_thresh = 0.6

        # 正常触发阈值：当前角速度方向与视觉要求方向相同或近似直行时，
        # 按 bottom_y > 300 触发。
        self.dist_thresh_y = 300

        # 提前触发阈值：当前角速度方向与视觉要求方向相反时，
        # 按 bottom_y > 280 提前触发。
        self.early_dist_thresh_y = 280

        self.avoid_v = 1
        self.avoid_w_fixed = 1.0

        # 上侧区域视觉避障保护：
        # 实际地图 y > 1.5 时，视觉避障不允许继续向左转。
        # 如果视觉原本要求右转，保持正常右转角速度；
        # 如果视觉原本要求左转，则改为以 1.5 rad/s 向右转。
        self.upper_avoid_y_threshold = 1.5
        self.upper_forced_right_w = 1.8

        # 下侧区域视觉避障保护：
        # 实际地图 y < 0.5 且 x > 1.0 时，视觉避障不允许继续向右转。
        # x > 1.0 用于排除起点附近本来就处于 y < 0.5 的区域。
        # 如果视觉原本要求左转，保持正常左转角速度；
        # 如果视觉原本要求右转，则改为以 1.5 rad/s 向左转。
        self.lower_avoid_y_threshold = 0.5
        self.lower_avoid_x_min = 1.0
        self.lower_forced_left_w = 1.5

        # 判断当前角速度方向时使用的死区。
        # abs(w) <= 该值时认为车辆基本直行，不判定为左转或右转。
        self.turn_w_deadband = 0.00

        # 视觉避障采用“保持到下一帧”机制：
        # 某一帧触发避障后，在下一条检测消息到来之前，持续发布该避障指令。
        # 不再使用固定秒数，防止图像/检测帧率较低或偶发丢帧时被 PID 提前覆盖。

        # 图像中央近距离障碍物增强：
        # 当选中的最近障碍物 bottom_y > 320，且检测框中心位于
        # 图像中心 x=320 左右 35 像素范围内时，本次避障角速度
        # 的绝对值临时增加 0.2 rad/s，转向方向保持不变。
        self.center_close_bottom_y_threshold = 320
        self.center_close_x_half_range = 40
        self.center_close_w_bonus = 0.2

        # 图像尺寸参数
        self.image_width = 640
        self.image_height = 480

        # ==========================================
        # [配置区 2.1]：曲线边缘忽略参数
        # ==========================================
        # 左侧边缘曲线控制点：(x, y)
        # y < 300 时向上延伸并保持 x 不变。
        self.left_edge_points = [
            (145, 300),
            (136, 305),
            (127, 310),
            (114, 315),
            (103, 320),
        ]

        # 右侧边缘会自动关于图像中心对称生成
        self._build_edge_boundaries()

        # ==========================================
        # [配置区 3]：虚拟墙边界参数
        # ==========================================
        self.y_lower_danger_min = 0.0
        self.y_lower_danger_max = 0.20
        self.y_upper_danger_min = 1.8
        self.y_upper_danger_max = 2.0
        self.boundary_w = 1.8
        self.boundary_v = 1.0

        # 上侧“航向角 + y”强制右转保护。
        # Pose2D.theta 按弧度处理，正角度表示逆时针方向。
        # 当实际 y > 1.2 且航向角 >= 45° 时，立即强制右转。
        #
        # 为了让车辆真正转过去，这里降低线速度，并使用最大右转角速度。
        self.upper_heading_y_threshold = 1.60 #1.20
        self.upper_heading_yaw_threshold = math.radians(30.0) #45 
        self.upper_heading_force_v = 0.60
        self.upper_heading_force_w = 2.00

        # ==========================================
        # [状态变量]
        # ==========================================
        self.cur_pose = [0.0, 0.0, 0.0]  # [x, y, yaw]
        self.pose_received = False
        # navigation_arrived 表示已经到达第一目标点 (5.0, 2.0) 并停车；
        # is_finished 表示已经把二维码结果交接给通道导航节点。
        self.navigation_arrived = False
        self.is_finished = False

        # 最近一次真正发布到 /cmd_vel 的速度。
        # 角速度正数表示左转，负数表示右转。
        self.last_cmd_v = 0.0
        self.last_cmd_w = 0.0

        # 上侧航向角保护状态，仅用于记录进入/退出并清除旧视觉指令。
        self.upper_heading_force_active = False

        # 视觉避障“保持到下一帧”状态。
        # visual_avoid_active=True 时，如果没有收到新的检测消息，
        # 控制循环会持续发布上一帧确定的避障指令。
        self.visual_avoid_active = False
        self.avoid_hold_v = self.avoid_v
        self.avoid_hold_w = 0.0

        # 二维码结果与任务交接状态。
        self.qr_result_received = False
        self.handoff_complete = False
        self.qr_result = ''

        # 速度发布原子锁与目标点停车锁存。
        # 锁覆盖“检查状态 -> 更新状态 -> 发布 /cmd_vel”全过程，
        # 防止到达目标点后避障线程或 PID 又补发一条非零速度。
        self.motion_lock = threading.RLock()
        self.stop_latched = False
        self.fast_stop_until = 0.0

        # 可靠交接状态。
        self.handoff_publish_count = 0
        self.channel_ack_received = False
        self.channel_ack_data = ''

        # 二维码结果回调使用独立回调组，避免被控制循环或视觉避障阻塞。
        self.qr_callback_group = MutuallyExclusiveCallbackGroup()
        self.stop_callback_group = MutuallyExclusiveCallbackGroup()
        # 普通控制、锥桶检测和位姿分别使用独立回调组。
        # 原代码的100 Hz控制定时器和锥桶订阅都在默认互斥组中，
        # 高频定时器可能连续占用回调组，导致锥桶帧延迟。
        self.control_callback_group = MutuallyExclusiveCallbackGroup()
        self.obstacle_callback_group = MutuallyExclusiveCallbackGroup()
        self.pose_callback_group = MutuallyExclusiveCallbackGroup()

        # 通信接口
        self.pose_sub = self.create_subscription(
            Pose2D,
            'odom_pose',
            self.pose_cb,
            10,
            callback_group=self.pose_callback_group,
        )
        # 锥桶检测只保留最新一帧，并使用BEST_EFFORT传感器型QoS。
        # 这样不会因为可靠重传或旧检测排队而延迟避障。
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

        # 如果本节点运行在 namespace 中，同时监听该 namespace 下的相对
        # qr_direction_result。根 namespace 下二者相同，此时不会重复订阅。
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

        # 使用 TRANSIENT_LOCAL，通道节点即使稍晚启动，也能收到最近一次
        # 成功消息。通道节点必须使用相同 durability。
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

        # /cmd_vel 只保留最新一条可靠消息。depth=1 能避免底盘在收到
        # 停车命令前继续消费队列里残留的旧前进/转向命令。
        cmd_qos = QoSProfile(depth=1)
        cmd_qos.history = HistoryPolicy.KEEP_LAST
        cmd_qos.reliability = ReliabilityPolicy.RELIABLE
        cmd_qos.durability = DurabilityPolicy.VOLATILE
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', cmd_qos)

        # 普通控制保持 40 Hz。到达第一目标点后的低延迟停车由独立
        # 200 Hz 快速停车定时器负责。
        self.timer = self.create_timer(
            0.025,
            self.control_loop,
            callback_group=self.control_callback_group,
        )

        # 独立 200 Hz 快速停车定时器，只在到达目标点后的短窗口内发布。
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
            '🚀 巡航节点启动：目标点=(5.0, 2.0)；'
            '已删除二维码检测停车逻辑；收到二维码解析结果立即交接；'
            '到达目标点后快速停车并保持'
        )

    # -------------------------------------------------------------------------
    # [边缘曲线构建块]
    # -------------------------------------------------------------------------
    def _build_edge_boundaries(self):
        """
        根据实测控制点构建左右边缘曲线插值函数。
        - 左侧曲线：基于 self.left_edge_points
        - 右侧曲线：关于图像中心对称
        """
        center_x = self.image_width / 2.0
        extended_left_points = []

        min_y_point = min(self.left_edge_points, key=lambda p: p[1])
        min_y = min_y_point[1]
        min_x = min_y_point[0]

        # 向上延伸到 y=100
        if min_y > 100:
            extended_left_points.append((min_x, 100))
            extended_left_points.append((min_x, min_y))

        sorted_points = sorted(self.left_edge_points, key=lambda p: p[1])
        extended_left_points.extend(sorted_points)

        # 向下延伸到图像底部
        max_y_point = max(sorted_points, key=lambda p: p[1])
        max_y = max_y_point[1]
        max_x = max_y_point[0]

        if max_y < self.image_height - 1:
            extended_left_points.append((max_x, self.image_height - 1))

        # 去除可能重复的 y，避免插值坐标重复
        unique_by_y = {}
        for x, y in extended_left_points:
            unique_by_y[y] = x
        extended_left_points = sorted(
            [(x, y) for y, x in unique_by_y.items()],
            key=lambda p: p[1],
        )

        left_y_coords = np.array([p[1] for p in extended_left_points], dtype=float)
        left_x_coords = np.array([p[0] for p in extended_left_points], dtype=float)

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
            f'y 范围 [{left_y_coords[0]:.0f}, {left_y_coords[-1]:.0f}]'
        )

    def is_in_edge_ignore_zone(self, center_x, bottom_y):
        """检测框中心是否落在左右曲线边缘之外。"""
        left_x = self.left_boundary_func(bottom_y)
        right_x = self.right_boundary_func(bottom_y)
        return center_x < left_x or center_x > right_x

    # -------------------------------------------------------------------------
    # [方向辅助函数]
    # -------------------------------------------------------------------------
    def angular_direction_sign(self, w):
        """
        将角速度转换为方向符号：
        +1：左转
        -1：右转
         0：近似直行
        """
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

    # -------------------------------------------------------------------------
    # [回调函数块]
    # -------------------------------------------------------------------------
    def pose_cb(self, msg):
        first_pose = not self.pose_received
        self.cur_pose = [msg.x, msg.y, msg.theta]
        self.pose_received = True

        if first_pose:
            actual_x = msg.x + self.start_offset_x
            actual_y = msg.y + self.start_offset_y
            self.get_logger().info(
                f'📍 位姿已连接，actual=({actual_x:.3f}, {actual_y:.3f})'
            )

    def _publish_zero_locked(self):
        """调用者必须持有 motion_lock；无日志地直接发布零速度。"""
        zero = Twist()
        self.last_cmd_v = 0.0
        self.last_cmd_w = 0.0
        self.cmd_pub.publish(zero)

    def _activate_target_stop_locked(self):
        """调用者必须持有 motion_lock；到达第一目标点后建立停车锁存。"""
        self.stop_latched = True
        self.fast_stop_until = time.monotonic() + self.fast_stop_hold_sec
        self.fast_stop_timer.reset()

        # 清除所有可能继续保持的非零控制状态。
        self.visual_avoid_active = False
        self.upper_heading_force_active = False
        self.avoid_hold_v = 0.0
        self.avoid_hold_w = 0.0

        # 先不经过日志和其他判断，连续发布零速度覆盖底盘旧指令。
        for _ in range(self.fast_stop_burst_count):
            self._publish_zero_locked()

    def fast_stop_loop(self):
        """到达第一目标点后，短时间以 200 Hz 重复零速度。"""
        with self.motion_lock:
            # 一旦二维码结果已经完成交接，巡航节点必须立刻停止所有速度发布。
            if self.handoff_complete or self.is_finished:
                self.fast_stop_timer.cancel()
                return

            if not self.stop_latched:
                self.fast_stop_timer.cancel()
                return

            if time.monotonic() <= self.fast_stop_until:
                self._publish_zero_locked()
            else:
                # 快速覆盖窗口结束后，由 40 Hz 控制循环继续保持停车。
                self.fast_stop_timer.cancel()

    def qr_result_cb(self, msg):
        """收到非空二维码解析结果后，立即交接给通道导航节点。"""
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
        """发布一次成功消息，并记录发布次数和连接状态。"""
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
        """二维码结果一旦有效，立即结束巡航控制并发布交接消息。"""
        if self.handoff_complete:
            return
        if self.qr_result_received:
            return
        if not self.qr_result:
            self.get_logger().error('无法完成交接：二维码结果为空')
            return

        result = self.qr_result

        # 先原子关闭巡航节点的一切速度输出，再发布 qr_success。
        # 这里不主动发布停车命令，避免二维码结果到达时形成额外停车阶段；
        # 后续速度由通道导航节点接管。
        with self.motion_lock:
            self.qr_result_received = True
            self.handoff_complete = True
            self.is_finished = True
            self.visual_avoid_active = False
            self.upper_heading_force_active = False
            self.avoid_hold_v = 0.0
            self.avoid_hold_w = 0.0
            self.fast_stop_timer.cancel()

        self._publish_qr_success('收到二维码解析结果，立即正式交接')
        self.handoff_repeat_timer.reset()

        self.get_logger().warn(
            f'🚦 已交接二维码结果“{result}”：巡航节点停止发布 /cmd_vel；'
            f'等待通道节点通过 {self.channel_ack_topic} 返回 ACK'
        )

    def handoff_repeat_loop(self):
        """在未收到通道 ACK 时有限次数重发成功消息。"""
        if not self.handoff_complete or self.channel_ack_received:
            self.handoff_repeat_timer.cancel()
            return

        if self.handoff_publish_count >= self.handoff_repeat_limit:
            self.get_logger().error(
                f'❌ 已重发 {self.qr_success_topic} '
                f'{self.handoff_publish_count} 次仍未收到 ACK；'
                f'请检查 channel_navigation 是否运行、话题和 ROS_DOMAIN_ID'
            )
            self.handoff_repeat_timer.cancel()
            return

        self._publish_qr_success('等待通道 ACK，定时重发')

    def channel_ack_cb(self, msg):
        """接收通道节点唤醒确认。"""
        ack = str(msg.data).strip()
        self.channel_ack_received = True
        self.channel_ack_data = ack
        self.handoff_repeat_timer.cancel()
        self.get_logger().warn(
            f'✅ 收到通道节点 ACK："{ack}"；'
            f'交接消息共发布 {self.handoff_publish_count} 次，'
            f'巡航节点确认不再发布 /cmd_vel'
        )

    def obs_cb(self, msg):
        """
        检测结果到达后立即判断并发布 /cmd_vel。

        当 y > 1.2 且航向角 >= 45° 时：
        1. 当前方向固定按“右转”参与原双阈值判断；
        2. 有障碍物时仍执行原左右方向比较和提前阈值逻辑；
        3. 最终控制命令始终保持强制右转。
        """
        # 整个锥桶决策与速度发布使用同一把可重入锁。
        # 控制定时器无法在锥桶帧处理到一半时插入PID命令。
        with self.motion_lock:
            if self.is_finished or self.handoff_complete or self.stop_latched:
                return

            # 到达第一目标点后保持停车，障碍检测帧不再恢复 PID 或避障速度。
            if self.navigation_arrived:
                return

            # 已经进入真实上下危险区时，仍由原虚拟墙最高优先级接管。
            if self.check_boundary_protection():
                return

            upper_force = self.update_upper_heading_force_state()

            # 有效障碍仍接入原来的双阈值判断。
            # detect_hazard() 内部会读取 upper_force 状态：
            # - 当前方向按右转参与同向/反向判断；
            # - 最终命令仍强制右转。
            if self.detect_hazard(msg):
                return

            # 当前检测帧没有触发障碍，但角度+y条件成立：
            # 不执行 PID，直接强制右转。
            if upper_force:
                self.visual_avoid_active = False
                self.avoid_hold_w = 0.0
                self.execute_drive(
                    self.upper_heading_force_v,
                    -abs(self.upper_heading_force_w),
                    self.upper_heading_force_log('无有效障碍，直接接管'),
                )
                return

            # 当前帧不触发危险，解除上一视觉避障并立即恢复 PID。
            if self.visual_avoid_active:
                old_w = self.avoid_hold_w
                self.visual_avoid_active = False
                self.avoid_hold_w = 0.0

                self.get_logger().info(
                    f'✅ 当前检测帧无危险，立即解除上一避障指令 '
                    f'W={old_w:.2f}',
                    throttle_duration_sec=0.5,
                )

            self.perform_navigation_pid()

    # -------------------------------------------------------------------------
    # [核心控制块]
    # -------------------------------------------------------------------------
    def control_loop(self):
        """40 Hz 保底控制循环。"""
        with self.motion_lock:
            # 二维码结果已经交接后，本节点永久停止发布 /cmd_vel。
            if self.is_finished or self.handoff_complete:
                return

            # 只有到达第一目标点 (5.0, 2.0) 后才保持停车。
            if self.stop_latched or self.navigation_arrived:
                self.execute_drive(
                    0.0,
                    0.0,
                    '🏁 已到达第一目标点 (5.0, 2.0)，停车保持',
                )
                return

            # 实际危险边界具有最高运动控制优先级。
            if self.check_boundary_protection():
                return

            # y > 1.2 且航向角达到阈值时持续强制右转。
            if self.update_upper_heading_force_state():
                self.execute_drive(
                    self.upper_heading_force_v,
                    -abs(self.upper_heading_force_w),
                    self.upper_heading_force_log('40Hz持续接管'),
                )
                return

            # 无新检测帧时，保持上一帧视觉避障命令。
            if self.visual_avoid_active:
                self.execute_drive(
                    self.avoid_hold_v,
                    self.avoid_hold_w,
                    '🧲 无新检测帧，持续保持视觉避障',
                )
                return

            self.perform_navigation_pid()

    @staticmethod
    def normalize_angle(angle):
        """将角度归一化到 [-pi, pi)。"""
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    def upper_heading_force_condition(self):
        """
        实际 y > 1.2 且航向角 >= 45° 时返回 True。

        注意：这里假设 Pose2D.theta 使用弧度，并且朝 +y 方向转动时
        theta 为正。如果你的里程计方向相反，可根据运行日志调整符号。
        """
        actual_y = self.cur_pose[1] + self.start_offset_y
        yaw = self.normalize_angle(self.cur_pose[2])

        return (
            actual_y > self.upper_heading_y_threshold
            and yaw >= self.upper_heading_yaw_threshold
        )

    def update_upper_heading_force_state(self):
        """更新强制右转状态，并在首次进入时清除旧视觉避障指令。"""
        active_now = self.upper_heading_force_condition()

        if active_now and not self.upper_heading_force_active:
            self.upper_heading_force_active = True

            # 进入保护时清除可能残留的旧左转视觉指令。
            self.visual_avoid_active = False
            self.avoid_hold_w = 0.0

            actual_y = self.cur_pose[1] + self.start_offset_y
            yaw_deg = math.degrees(self.normalize_angle(self.cur_pose[2]))
            self.get_logger().warn(
                f'🧭 进入上侧航向保护：actual_y={actual_y:.3f}>'
                f'{self.upper_heading_y_threshold:.2f}，'
                f'yaw={yaw_deg:.1f}°>='
                f'{math.degrees(self.upper_heading_yaw_threshold):.1f}°，'
                f'开始强制右转',
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

    def check_boundary_protection(self):
        """
        原有真实边界保护：
        - 下危险区强制左转；
        - 上危险区强制右转。

        这里只处理已经进入实际危险区的情况。
        y>1.2 且角度>=45°的提前保护由独立航向逻辑处理。
        """
        actual_y = self.cur_pose[1] + self.start_offset_y

        if self.y_lower_danger_min <= actual_y <= self.y_lower_danger_max:
            self.upper_heading_force_active = False
            self.visual_avoid_active = False
            self.avoid_hold_w = 0.0
            self.execute_drive(
                self.boundary_v,
                +self.boundary_w,
                f'🚧 下边界保护！actual_y={actual_y:.3f} 强制左转',
            )
            return True

        if self.y_upper_danger_min <= actual_y <= self.y_upper_danger_max:
            self.upper_heading_force_active = True
            self.visual_avoid_active = False
            self.avoid_hold_w = 0.0
            self.execute_drive(
                self.boundary_v,
                -self.boundary_w,
                f'🚧 上边界保护！actual_y={actual_y:.3f} 强制右转',
            )
            return True

        return False

    # -------------------------------------------------------------------------
    # [视觉候选目标筛选]
    # -------------------------------------------------------------------------
    def collect_valid_obstacle_candidates(self, msg):
        """
        收集满足以下条件的检测框：
        1. 有效 ROI；
        2. 置信度大于阈值；
        3. bottom_y 大于提前阈值，已经进入提前预判区域；
        4. 不在曲线边缘忽略区。

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

                # 提前阈值以下既不避障，也不参与方向冲突判断
                if bottom_y <= self.early_dist_thresh_y:
                    continue

                if self.is_in_edge_ignore_zone(center_x, bottom_y):
                    left_x = self.left_boundary_func(bottom_y)
                    right_x = self.right_boundary_func(bottom_y)
                    self.get_logger().info(
                        f'🟡 曲线边缘忽略：center_x={center_x:.0f}, '
                        f'bottom_y={bottom_y:.0f} | '
                        f'边界范围 [{left_x:.0f}, {right_x:.0f}]',
                        throttle_duration_sec=0.5,
                    )
                    continue

                candidates.append((bottom_y, center_x, roi))

        return candidates

    # -------------------------------------------------------------------------
    # [避障逻辑块]
    # -------------------------------------------------------------------------
    def detect_hazard(self, msg):
        """
        双阈值方向判断：

        1. 根据锥桶中心位置得到需要执行的避障方向：
           - 锥桶在画面左侧：向右避让，w < 0；
           - 锥桶在画面右侧：向左避让，w > 0。

        2. 将避障方向与当前实际发布的角速度方向比较：
           - 同方向或当前近似直行：完全忽略提前阈值，只按正常阈值触发；
           - 相反方向：仅此时使用提前阈值触发。

        每收到一条新检测消息，都重新比较当前角速度和视觉要求方向。
        当前帧确定的最终避障指令会保持到下一条检测消息到来，
        但不会锁存“方向冲突”判断，因此同向时仍只按正常阈值触发。
        """
        candidates = self.collect_valid_obstacle_candidates(msg)

        if not candidates:
            return False

        # 选择 bottom_y 最大的有效锥桶，即图像中最近的锥桶。
        bottom_y, center_x, roi = max(candidates, key=lambda item: item[0])

        image_center_x = self.image_width / 2.0

        # ROS 角速度约定：w > 0 左转，w < 0 右转。
        # 锥桶在左侧时向右避让；锥桶在右侧时向左避让。
        desired_w = (
            -self.avoid_w_fixed
            if center_x < image_center_x
            else +self.avoid_w_fixed
        )
        desired_sign = self.angular_direction_sign(desired_w)

        # 当前消息就是允许改变避障状态的“下一帧”。
        # 因此新目标与旧保持方向相反时，可在本帧直接重新计算并接管。

        upper_force = self.upper_heading_force_active

        # 处于 y>1.2 且航向角>=45°保护时，
        # 不管上一条 /cmd_vel 是什么，都固定按“当前正在右转”参与
        # 原有同向/反向双阈值判断。
        if upper_force:
            current_sign = -1
            current_w_for_log = -abs(self.upper_heading_force_w)
        else:
            current_sign = self.angular_direction_sign(self.last_cmd_w)
            current_w_for_log = self.last_cmd_w

        # 只有当前转向方向和视觉要求方向相反时，才允许使用提前阈值。
        directions_opposite = (
            current_sign != 0
            and desired_sign != 0
            and current_sign != desired_sign
        )

        if directions_opposite:
            trigger_threshold = self.early_dist_thresh_y
            trigger_mode = (
                f'方向相反，使用 {self.early_dist_thresh_y} 提前触发'
            )
        else:
            # 当前直行或方向一致：完全不采用提前阈值，
            # 直接执行正常触发阈值。
            trigger_threshold = self.dist_thresh_y
            if current_sign == desired_sign and current_sign != 0:
                trigger_mode = (
                    f'方向一致，忽略 {self.early_dist_thresh_y}，'
                    f'按 {self.dist_thresh_y}'
                )
            else:
                trigger_mode = (
                    f'当前直行，忽略 {self.early_dist_thresh_y}，'
                    f'按 {self.dist_thresh_y}'
                )

        self.get_logger().info(
            f'👀 锥桶预判：当前={self.direction_name(current_sign)} '
            f'(W={current_w_for_log:.2f})，'
            f'需要={self.direction_name(desired_sign)}，'
            f'bottom_y={bottom_y:.0f}，阈值={trigger_threshold}（{trigger_mode}）',
            throttle_duration_sec=0.5,
        )

        if bottom_y <= trigger_threshold:
            return False

        # --------------------------------------------------------------
        # 上下侧区域视觉避障保护
        # --------------------------------------------------------------
        # 使用带起点偏移的实际地图坐标，与边界保护逻辑保持一致。
        actual_x = self.cur_pose[0] + self.start_offset_x
        actual_y = self.cur_pose[1] + self.start_offset_y
        command_v = self.avoid_v
        command_w = desired_w
        boundary_avoid_mode = ''

        if upper_force:
            # 即使障碍物原逻辑要求左转，也不允许车辆继续靠近上边界。
            # 障碍方向仍参与上面的双阈值判断，但最终命令固定强制右转。
            command_v = self.upper_heading_force_v
            command_w = -abs(self.upper_heading_force_w)
            boundary_avoid_mode = (
                f' | y>1.2且航向角>=45°，'
                f'以强制右转接入原双阈值逻辑，'
                f'最终保持 W={command_w:.2f}'
            )

        elif actual_y > self.upper_avoid_y_threshold:
            # 上侧区域只允许视觉避障向右转。
            if desired_sign < 0:
                # 视觉原本就要求右转：保持正常右转角速度。
                command_w = desired_w
                boundary_avoid_mode = (
                    f' | 上侧区域 y={actual_y:.2f}>'
                    f'{self.upper_avoid_y_threshold:.2f}，原本右转，保持正常右转'
                )
            else:
                # 视觉原本要求左转：强制改成 1.5 rad/s 右转。
                command_w = -abs(self.upper_forced_right_w)
                boundary_avoid_mode = (
                    f' | 上侧区域 y={actual_y:.2f}>'
                    f'{self.upper_avoid_y_threshold:.2f}，'
                    f'原本左转，强制右转 W={command_w:.2f}'
                )

        elif (
            actual_y < self.lower_avoid_y_threshold
            and actual_x > self.lower_avoid_x_min
        ):
            # 下侧区域只允许视觉避障向左转。
            if desired_sign > 0:
                # 视觉原本就要求左转：保持正常左转角速度。
                command_w = desired_w
                boundary_avoid_mode = (
                    f' | 下侧区域 y={actual_y:.2f}<'
                    f'{self.lower_avoid_y_threshold:.2f} 且 '
                    f'x={actual_x:.2f}>{self.lower_avoid_x_min:.2f}，'
                    f'原本左转，保持正常左转'
                )
            else:
                # 视觉原本要求右转：强制改成 1.5 rad/s 左转。
                command_w = abs(self.lower_forced_left_w)
                boundary_avoid_mode = (
                    f' | 下侧区域 y={actual_y:.2f}<'
                    f'{self.lower_avoid_y_threshold:.2f} 且 '
                    f'x={actual_x:.2f}>{self.lower_avoid_x_min:.2f}，'
                    f'原本右转，强制左转 W={command_w:.2f}'
                )

        # --------------------------------------------------------------
        # 图像中央近距离障碍物：本次角速度临时增强 0.2
        # --------------------------------------------------------------
        # 只检查当前选中的最近障碍物：
        # 1. bottom_y 必须严格大于 320；
        # 2. center_x 必须在图像中心左右 35 像素范围内，即 285~355。
        # 满足时只增加角速度绝对值，不改变最终转向方向。
        center_close_mode = ''
        center_offset_x = abs(center_x - image_center_x)

        if (
            bottom_y > self.center_close_bottom_y_threshold
            and center_offset_x <= self.center_close_x_half_range
        ):
            original_command_w = command_w
            enhanced_abs_w = min(
                abs(command_w) + self.center_close_w_bonus,
                self.max_w,
            )
            command_w = math.copysign(enhanced_abs_w, command_w)

            if abs(command_w) > abs(original_command_w):
                center_close_mode = (
                    f' | 🎯 中央近距离增强：bottom_y={bottom_y:.0f}>'
                    f'{self.center_close_bottom_y_threshold}，'
                    f'center_x={center_x:.0f} 位于 '
                    f'[{image_center_x - self.center_close_x_half_range:.0f}, '
                    f'{image_center_x + self.center_close_x_half_range:.0f}]，'
                    f'角速度临时 +{self.center_close_w_bonus:.2f}，'
                    f'W:{original_command_w:.2f}->{command_w:.2f}'
                )
            else:
                center_close_mode = (
                    f' | 🎯 中央近距离增强条件满足，但受 '
                    f'max_w={self.max_w:.2f} 限制，W仍为 {command_w:.2f}'
                )

        # 将本帧最终视觉指令锁存。
        # 在下一条检测消息到来之前，控制循环持续发布该指令；
        # 只有下一帧才能更新方向或解除避障。
        self.visual_avoid_active = True
        self.avoid_hold_v = command_v
        self.avoid_hold_w = command_w

        self.execute_drive(
            command_v,
            command_w,
            f'⚡ 视觉劫持[{trigger_mode}]：'
            f'center_x={center_x:.0f}, bottom_y={bottom_y:.0f}, '
            f'conf={roi.confidence:.2f}, threshold={trigger_threshold}'
            f'{boundary_avoid_mode}{center_close_mode}',
        )
        return True

    # -------------------------------------------------------------------------
    # [导航逻辑块]
    # -------------------------------------------------------------------------
    def perform_navigation_pid(self):
        """常规 PID 寻点逻辑；第一目标点为实际地图坐标 (5.0, 2.0)。"""
        if self.handoff_complete or self.is_finished or self.navigation_arrived:
            return

        mx = self.cur_pose[0] + self.start_offset_x
        my = self.cur_pose[1] + self.start_offset_y
        m_yaw = self.cur_pose[2]

        dx = self.target_map_x - mx
        dy = self.target_map_y - my
        dist = math.hypot(dx, dy)

        if dist < self.arrival_tolerance:
            with self.motion_lock:
                if self.handoff_complete or self.is_finished:
                    return
                self.navigation_arrived = True
                self._activate_target_stop_locked()

            self.get_logger().warn(
                f'🏁 到达第一目标点 ({self.target_map_x:.1f}, '
                f'{self.target_map_y:.1f})，距离={dist:.3f}m；'
                f'已锁存停车并继续等待二维码解析结果'
            )
            return

        target_yaw = math.atan2(dy, dx)
        angle_error = self.normalize_angle(target_yaw - m_yaw)

        v_out = dist * self.kp_linear
        w_out = angle_error * self.kp_angular

        self.execute_drive(v_out, w_out, '🍀 PID 寻点中')

    def execute_drive(self, v, w, log_tag):
        """原子检查状态并发布速度，防止到达目标点后出现晚到非零指令。"""
        requested_nonzero = abs(v) > 1e-6 or abs(w) > 1e-6

        with self.motion_lock:
            # 交接完成后无条件禁止巡航节点继续发布。
            if self.handoff_complete or self.is_finished:
                return

            # 到达第一目标点建立停车锁存后，拦截任何非零速度。
            if self.stop_latched and requested_nonzero:
                return

            v = max(0.0, min(v, self.max_v))
            w = max(-self.max_w, min(w, self.max_w))

            self.last_cmd_v = float(v)
            self.last_cmd_w = float(w)

            twist = Twist()
            twist.linear.x = float(v)
            twist.angular.z = float(w)
            self.cmd_pub.publish(twist)

        # 日志在解锁和发布之后执行，避免阻塞关键速度消息。
        if v == 0.0 and w == 0.0:
            self.get_logger().info(
                f'{log_tag} | V:0.00 W:0.00',
                throttle_duration_sec=1.0,
            )
        elif '视觉劫持' in log_tag or '边界保护' in log_tag:
            self.get_logger().info(
                f'{log_tag} | V:{v:.2f} W:{w:.2f}',
                throttle_duration_sec=1.0,
            )

# -----------------------------------------------------------------------------
# [主函数块]
# -----------------------------------------------------------------------------
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
