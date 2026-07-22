#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
import math
import time
import os
import threading
from geometry_msgs.msg import Pose2D, Twist
from ai_msgs.msg import PerceptionTargets
from std_msgs.msg import Int8

class ReturnToHomeNode(Node):
    def __init__(self):
        super().__init__('return_to_home_node')

        # ==========================================
        # [配置区 1]：全局寻点与地图参数
        # ==========================================
        self.start_offset_x = 0.55
        self.start_offset_y = 0.20

        # ★ 终极目标：P点 (发车点)
        self.target_p_x = 0.55
        self.target_p_y = 0.20

        self.kp_linear = 1.0
        self.kp_angular = 3.0
        self.max_v = 1.0
        self.max_w = 2.0
        self.arrival_tolerance = 0.35  # 到达 P 点的判定阈值

        # ==========================================
        # [配置区 2]：视觉避障参数 (保持大厅防撞)
        # ==========================================
        self.conf_thresh = 0.6
        self.dist_thresh_y = 300
        self.avoid_v = 1.0
        self.avoid_w_fixed = 0.8
        self.image_width = 640
        self.avoid_edge_margin = 100

        # ==========================================
        # [状态变量]：跨包静默锁
        # ==========================================
        self.is_active = False  # 初始休眠
        self.cur_pose = [0.0, 0.0, 0.0]
        self.latest_obs = None
        self.lock = threading.Lock()

        # 通信接口
        self.pose_sub = self.create_subscription(Pose2D, 'odom_pose', self.pose_cb, 10)
        self.obs_sub = self.create_subscription(PerceptionTargets, 'racing_obstacle_detection', self.obs_cb, 10)
        self.state_sub = self.create_subscription(Int8, '/competition_state', self.state_cb, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # 定时器 (20Hz)
        self.timer = self.create_timer(0.05, self.control_loop)
        self.get_logger().info("🚀 [任务三] 归巢节点已加载！静默等待任务二移交兵权...")

    def state_cb(self, msg):
        if msg.data == 3 and not self.is_active:
            self.is_active = True
            self.get_logger().info("⚡ [任务三] 成功唤醒！全速冲刺返回 P 点！")

    def pose_cb(self, msg):
        self.cur_pose = [msg.x, msg.y, msg.theta]

    def obs_cb(self, msg):
        with self.lock:
            self.latest_obs = msg

    def control_loop(self):
        # ★ 拦截墙：没收到信号绝不抢夺底盘
        if not self.is_active:
            return

        with self.lock:
            obs_msg = self.latest_obs
            self.latest_obs = None
        
        # 处理避障
        if obs_msg and self.detect_hazard(obs_msg):
            return

        # 执行回源 PID 寻点
        self.perform_pid_nav()

    def perform_pid_nav(self):
        mx = self.cur_pose[0] + self.start_offset_x
        my = self.cur_pose[1] + self.start_offset_y
        m_yaw = self.cur_pose[2]

        dx = self.target_p_x - mx
        dy = self.target_p_y - my
        dist = math.sqrt(dx**2 + dy**2)

        # 最终到达判定
        if dist < self.arrival_tolerance:
            self.get_logger().info("🎉 抵达起点！比赛圆满结束，强制停车！")
            # 发送多次零速度，确保物理刹车完全响应
            for _ in range(3):
                self.execute_drive(0.0, 0.0, "🛑 物理急刹死锁！")
                time.sleep(0.05)
            self.is_active = False # 自毁挂起
            return

        # PID 计算
        target_yaw = math.atan2(dy, dx)
        num_yaw = target_yaw - m_yaw
        angle_error = (num_yaw + math.pi) % (2.0 * math.pi) - math.pi

        v_out = dist * self.kp_linear
        w_out = angle_error * self.kp_angular

        self.execute_drive(v_out, w_out, f"🏁 返回P点寻点中 | 距目标 {dist:.2f}m")

    def detect_hazard(self, msg):
        if len(msg.targets) == 0: return False
        target = max(msg.targets, key=lambda t: t.rois[0].rect.y_offset + t.rois[0].rect.height)
        roi = target.rois[0]
        rect = roi.rect
        bottom_y = rect.y_offset + rect.height

        if roi.confidence > self.conf_thresh and bottom_y > self.dist_thresh_y:
            center_x = rect.x_offset + rect.width / 2
            if center_x < self.avoid_edge_margin or center_x > (self.image_width - self.avoid_edge_margin):
                return False
            avoid_dir_w = -self.avoid_w_fixed if center_x < 320 else self.avoid_w_fixed
            self.execute_drive(self.avoid_v, avoid_dir_w, "⚡ 视觉避障劫持!")
            return True
        return False

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
    node = ReturnToHomeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
