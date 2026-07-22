#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
import math
import threading
import time
import os
from geometry_msgs.msg import Pose2D, Twist
from sensor_msgs.msg import Imu  # 引入 IMU 消息类型
from std_msgs.msg import String
from ai_msgs.msg import PerceptionTargets

class FileTrackFollowerNode(Node):
    def __init__(self):
        super().__init__('file_track_follower_node')

        # ==========================================
        # [配置区 1]：轨迹文件与方向配置
        # ==========================================
        self.track_file = 'merged_lines.txt'

        # 不再使用固定方向。节点启动后等待 /qr_direction_result：
        #   顺时针 -> cw
        #   逆时针 -> ccw
        self.is_clockwise = None
        self.direction_received = False
        self.direction_text = ''
        self.dense_path = []
        self.current_path_index = 0

        # 防护状态标志位
        self.is_first_run = True
        self.has_reached_middle = False

        # ==========================================
        # [配置区 2]：★ 阿克曼物理参数与 IMU 预测参数 ★
        # ==========================================
        self.start_offset_x = 0.55
        self.start_offset_y = 0.20

        # 根据您的车模物理数据精确输入（单位：米）
        self.wheelbase = 0.144            # 1. 轴距
        self.imu_to_front = 0.11         # 2. IMU 距离前轮中心的物理距离

        self.target_v = 1.0              # 恒定跑图线速度
        self.k_stanley = 2.1             # Stanley横向增益 (纠偏灵敏度)

        # ⚡ 舵机响应延时的预测时间补偿 (单位：秒)
        self.servo_delay = 0.15

        self.max_v = 1.0
        self.max_w = 5.0

        # ==========================================
        # [配置区 3]：视觉避障参数与状态机 ★
        # ==========================================
        self.conf_thresh = 0.6
        self.dist_thresh_y = 300
        self.avoid_v = 1.0
        self.avoid_w_fixed = 1.0 #0.8
        self.image_width = 640
        self.avoid_edge_margin = 0

        # ⚡ 新增：避障状态机参数
        self.avoid_timeout = 0.1       # 避障动作保持时间(秒) - 建议0.5~0.8
        self.avoid_end_time = 0.0      # 避障结束的倒计时时间戳
        self.current_avoid_w = 0.0     # 当前锁定的避障角速度

        # ==========================================
        # 接收数据变量
        # ==========================================
        self.cur_pose = [0.0, 0.0, 0.0]  # [x, y, theta] 来自 odom
        self.imu_w_z = 0.0               # 瞬时角速度 (来自 IMU)
        self.latest_obs = None
        self.is_finished = False
        self.lock = threading.Lock()
        self.dt = 0.05

        # ROS 2 订阅者与发布者
        self.pose_sub = self.create_subscription(Pose2D, 'odom_pose', self.pose_cb, 10)
        self.imu_sub = self.create_subscription(Imu, 'imu_data', self.imu_cb, 10)
        self.obs_sub = self.create_subscription(PerceptionTargets, 'racing_obstacle_detection', self.obs_cb, 10)
        self.direction_sub = self.create_subscription(
            String, '/qr_direction_result', self.direction_cb, 10
        )
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        self.timer = self.create_timer(self.dt, self.control_loop)

        self.get_logger().info(
            f"🚀 [IMU预测补偿阿克曼控制] 启动 | 轴距: {self.wheelbase}m | "
            f"前移量: {self.imu_to_front}m"
        )
        self.get_logger().warn(
            "⏳ 正在等待 /qr_direction_result 的顺时针/逆时针结果，收到前任务二不会运动"
        )

    def direction_cb(self, msg):
        """根据二维码方向结果选择任务二轨迹方向，只锁存第一次有效结果。"""
        raw_text = str(msg.data).strip()
        normalized = raw_text.lower().replace(' ', '').replace('_', '').replace('-', '')

        # 已经开始运行后不允许切换方向，避免重新加载轨迹导致小车突然改变路线。
        if self.direction_received:
            self.get_logger().info(
                f"🔒 任务二方向已锁定为{self.direction_text}，忽略重复结果: {raw_text}",
                throttle_duration_sec=2.0
            )
            return

        # 必须先判断逆时针，因为英文 counterclockwise 中包含 clockwise。
        ccw_tokens = ('逆时针', '逆時針', 'counterclockwise', 'anticlockwise', 'ccw')
        cw_tokens = ('顺时针', '順時針', 'clockwise', 'cw')

        if any(token in normalized for token in ccw_tokens):
            is_clockwise = False
            direction_str = 'ccw'
            direction_text = '逆时针'
        elif any(token in normalized for token in cw_tokens):
            is_clockwise = True
            direction_str = 'cw'
            direction_text = '顺时针'
        else:
            self.get_logger().warn(
                f"⚠️ 收到无法识别的方向结果: {raw_text!r}，应包含“顺时针”或“逆时针”"
            )
            return

        new_path = self.load_and_stitch_track(self.track_file, direction=direction_str)
        if not new_path:
            self.get_logger().error(
                f"❌ 按{direction_text}读取轨迹失败或轨迹为空，任务二保持停止"
            )
            self.execute_drive(0.0, 0.0, "轨迹加载失败")
            return

        # 初始化本次任务二的全部轨迹状态。
        self.is_clockwise = is_clockwise
        self.direction_text = direction_text
        self.dense_path = new_path
        self.current_path_index = 0
        self.is_first_run = True
        self.has_reached_middle = False
        self.is_finished = False
        self.avoid_end_time = 0.0
        self.current_avoid_w = 0.0
        with self.lock:
            self.latest_obs = None

        # 最后设置该标志，确保控制循环不会读到初始化一半的轨迹。
        self.direction_received = True
        self.get_logger().warn(
            f"✅ 收到二维码方向结果: {raw_text!r} -> 任务二按{direction_text}运行 | "
            f"轨迹点数: {len(self.dense_path)}"
        )

    def load_and_stitch_track(self, file_path, direction='cw'):
        lines_data = {'D0': [], 'D1': [], 'L': [], 'R': [], 'T': []}
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_raw in f:
                    line = line_raw.strip()
                    if not line or line.startswith('#'): continue
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 4:
                        line_id = parts[0]
                        try:
                            x = float(parts[2])
                            y = float(parts[3])
                            if line_id in lines_data:
                                lines_data[line_id].append((x, y))
                        except ValueError:
                            pass
        except Exception as e:
            self.get_logger().error(f"读取文件出错: {e}")
            return []

        if direction == 'cw':
            return lines_data['D0'] + lines_data['L'][::-1] + lines_data['T'][::-1] + lines_data['R'] + lines_data['D1'][::-1]
        else:
            return lines_data['D1'] + lines_data['R'][::-1] + lines_data['T'] + lines_data['L'] + lines_data['D0'][::-1]

    def pose_cb(self, msg):
        self.cur_pose = [msg.x, msg.y, msg.theta]

    def imu_cb(self, msg):
        self.imu_w_z = msg.angular_velocity.z

    def obs_cb(self, msg):
        with self.lock:
            self.latest_obs = msg

    def control_loop(self):
        if self.is_finished:
            return

        # 没收到二维码方向结果前，不发布运动指令，避免默认方向误启动。
        if not self.direction_received:
            self.get_logger().info(
                "⏳ 等待 /qr_direction_result，任务二尚未启动",
                throttle_duration_sec=2.0
            )
            return

        if not self.dense_path:
            self.execute_drive(0.0, 0.0, "轨迹为空，保持停车")
            return

        current_time = time.time()

        with self.lock:
            obs_msg = self.latest_obs
            self.latest_obs = None  # 提取后清空，防止处理过期旧数据

        # 1. 尝试检测当前画面是否有障碍物
        is_hazard_now = False
        if obs_msg:
            is_hazard_now = self.detect_hazard(obs_msg, current_time)

        # 2. 判断是否处于“避障保持期”
        if is_hazard_now or (current_time < self.avoid_end_time):
            # 持续输出避障指令，屏蔽巡线逻辑
            self.execute_drive(self.avoid_v, self.current_avoid_w, "⚡ 避障强制切出中!")
            return

        # 3. 既没有障碍物，也不在避障冷却期，执行正常巡线
        self.perform_line_tracking()

    def detect_hazard(self, msg, current_time):
        if len(msg.targets) == 0:
            return False

        target = max(msg.targets, key=lambda t: t.rois[0].rect.y_offset + t.rois[0].rect.height)
        roi = target.rois[0]
        rect = roi.rect
        bottom_y = rect.y_offset + rect.height

        if roi.confidence > self.conf_thresh and bottom_y > self.dist_thresh_y:
            center_x = rect.x_offset + rect.width / 2

            # 如果目标已经在极度边缘，不刷新保持时间
            if center_x < self.avoid_edge_margin or center_x > (self.image_width - self.avoid_edge_margin):
                return False

            # 计算需要往哪边打方向盘
            avoid_dir_w = -self.avoid_w_fixed if center_x < 320 else self.avoid_w_fixed

            # ★ 锁定避障指令，并刷新保持时间 ★
            self.current_avoid_w = avoid_dir_w
            self.avoid_end_time = current_time + self.avoid_timeout

            return True

        return False

    def perform_line_tracking(self):
        # 1. 锁存当前里程计姿态
        mx_imu = self.cur_pose[0] + self.start_offset_x
        my_imu = self.cur_pose[1] + self.start_offset_y
        m_yaw = self.cur_pose[2]
        w_curr = self.imu_w_z

        # 第一步：计算当前前轴中心在世界坐标下的投影位置
        mx_fa = mx_imu + self.imu_to_front * math.cos(m_yaw)
        my_fa = my_imu + self.imu_to_front * math.sin(m_yaw)

        # 第二步：融入物理角速度，预测时滞后前轴的未来位置
        tau = self.servo_delay
        yaw_pred = m_yaw + w_curr * tau
        yaw_mid = m_yaw + 0.5 * w_curr * tau

        mx_pred = mx_fa + self.target_v * tau * math.cos(yaw_mid)
        my_pred = my_fa + self.target_v * tau * math.sin(yaw_mid)

        # 第三步：寻找最近点
        if self.is_first_run:
            search_start = 0
            search_end = len(self.dense_path)
        else:
            search_start = max(0, self.current_path_index - 2)
            search_end = min(len(self.dense_path), self.current_path_index + 100)

        min_dist = float('inf')
        closest_idx = self.current_path_index
        for i in range(search_start, search_end):
            px, py = self.dense_path[i]
            d = math.sqrt((px - mx_pred)**2 + (py - my_pred)**2)
            if d < min_dist:
                min_dist = d
                closest_idx = i

        if self.is_first_run:
            if closest_idx > len(self.dense_path) * 0.95:
                closest_idx = 0
            self.is_first_run = False

        self.current_path_index = closest_idx

        # 终点判定
        if closest_idx > len(self.dense_path) * 0.5:
            self.has_reached_middle = True

        dist_to_goal = math.sqrt((self.dense_path[-1][0] - mx_imu)**2 + (self.dense_path[-1][1] - my_imu)**2)
        if self.has_reached_middle and closest_idx > len(self.dense_path) * 0.95 and dist_to_goal < 0.2:
            self.execute_drive(0.0, 0.0, "🏁 顺利回港，停车！")
            self.is_finished = True
            return

        # 第四步：计算误差
        look_ahead_steps = 5
        next_idx = min(closest_idx + look_ahead_steps, len(self.dense_path) - 1)
        px, py = self.dense_path[closest_idx]
        nx, ny = self.dense_path[next_idx]

        path_yaw = math.atan2(ny - py, nx - px)
        dx = mx_pred - px
        dy = my_pred - py
        e_y = dx * math.sin(path_yaw) - dy * math.cos(path_yaw)
        e_yaw = (path_yaw - yaw_pred + math.pi) % (2.0 * math.pi) - math.pi

        # 第五步：带有航向预测的 Stanley 核心控制
        v_current = max(0.1, abs(self.target_v))
        delta = e_yaw + math.atan2(self.k_stanley * e_y, v_current)

        max_delta = math.radians(35.0)
        delta = max(-max_delta, min(delta, max_delta))

        # 第六步：阿克曼逆运动学
        w_out = (v_current / self.wheelbase) * math.tan(delta)

        # 临近终点降速
        v_out = self.target_v
        if self.has_reached_middle and closest_idx > len(self.dense_path) * 0.9 and dist_to_goal < 0.6:
            v_out = max(0.0, dist_to_goal * 1.5)

        self.execute_drive(v_out, w_out,
                           f"📐 实测w:{w_curr:+.2f} | 偏差:{e_y*100:+.1f}cm | 舵角:{math.degrees(delta):+.1f}° | idx:{closest_idx}")

    def execute_drive(self, v, w, log_tag):
        v = max(0.0, min(v, self.max_v))
        w = max(-self.max_w, min(w, self.max_w))
        twist = Twist()
        twist.linear.x = float(v)
        twist.angular.z = float(w)
        self.cmd_pub.publish(twist)
        self.get_logger().info(f"{log_tag} | V:{v:.2f} W:{w:.2f}", throttle_duration_sec=0.5)

def main():
    rclpy.init()
    node = FileTrackFollowerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().warn("🛑 紧急强制停车")
        for _ in range(5):
            node.cmd_pub.publish(Twist())
            time.sleep(0.02)
        os.system("ros2 topic pub --once cmd_vel geometry_msgs/msg/Twist '{}' > /dev/null 2>&1")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
