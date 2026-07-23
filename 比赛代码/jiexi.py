#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
import serial
import struct
import math
import signal

# 导入 ROS 2 标准消息
from geometry_msgs.msg import Pose2D, Twist, TransformStamped, Quaternion
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from tf2_ros import TransformBroadcaster

# 协议常量 (严格对照 C++ 和 STM32 代码)
FRAME_HEADER = 0x7B
FRAME_TAIL = 0x7D
RECEIVE_FRAME_LEN = 36  # STM32 发给上位机的长度
SEND_FRAME_LEN = 11     # 上位机发给 STM32 的长度

class OriginCarBaseNode(Node):
    def __init__(self):
        super().__init__('origincar_base_node')
        
        # 1. 声明并获取参数 (默认使用 /dev/ttyACM0，根据你的日志，这是你的设备名)
        self.declare_parameter('port', '/dev/ttyACM0')
        self.declare_parameter('baudrate', 115200)
        port = self.get_parameter('port').value
        baud = self.get_parameter('baudrate').value
        
        # 2. 初始化串口
        try:
            self.ser = serial.Serial(port, baud, timeout=0.01)
            self.get_logger().info(f'串口已成功连接: {port} 波特率: {baud}')
        except Exception as e:
            self.get_logger().error(f'无法打开串口 {port}: {e}')
            return

        # 3. 创建发布者 (用于和其他模块联合)
        self.odom_pub = self.create_publisher(Odometry, 'odom', 10)
        self.imu_pub = self.create_publisher(Imu, 'imu_data', 10)
        self.pose_pub = self.create_publisher(Pose2D, 'odom_pose', 10)
        
        # 4. 创建订阅者 (接收控制指令)
        self.cmd_sub = self.create_subscription(Twist, 'cmd_vel', self.cmd_vel_callback, 10)
        
        # 5. TF 广播器 (建立 odom 到 base_link 的坐标变换)
        self.tf_broadcaster = TransformBroadcaster(self)

        # 6. 定时器：50Hz 频率读取串口
        self.timer = self.create_timer(0.02, self.receive_loop)
        self.buffer = b''

    def check_sum(self, data):
        """ XOR 校验计算 """
        checksum = 0
        for byte in data:
            checksum ^= byte
        return checksum

    def cmd_vel_callback(self, msg):
        """ 
        订阅 cmd_vel 并下发至 STM32
        对应 C++ 中的 Cmd_Vel_Callback 逻辑
        """
        send_buffer = bytearray(SEND_FRAME_LEN)
        send_buffer[0] = FRAME_HEADER
        send_buffer[1] = 0 
        send_buffer[2] = 0 

        # 转换单位: m/s -> mm/s (乘以 1000)
        v_x = int(msg.linear.x * 1000)
        v_y = int(msg.linear.y * 1000)
        v_z = int(msg.angular.z * 1000)

        # 打包为大端序 short (2字节)
        struct.pack_into('>h', send_buffer, 3, v_x)
        struct.pack_into('>h', send_buffer, 5, v_y)
        struct.pack_into('>h', send_buffer, 7, v_z)

        # 计算校验位并填入第 9 位 (索引)
        send_buffer[9] = self.check_sum(send_buffer[:9])
        send_buffer[10] = FRAME_TAIL
        
        try:
            self.ser.write(send_buffer)
        except Exception as e:
            self.get_logger().error(f'发送指令失败: {e}')

    def euler_to_quaternion(self, roll, pitch, yaw):
        """ 欧拉角转四元数辅助函数 """
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        q = [0] * 4
        q[0] = sr * cp * cy - cr * sp * sy # x
        q[1] = cr * sp * cy + sr * cp * sy # y
        q[2] = cr * cp * sy - sr * sp * cy # z
        q[3] = cr * cp * cy + sr * sp * sy # w
        return q

    def receive_loop(self):
        """ 串口接收及解析逻辑 """
        if self.ser.in_waiting > 0:
            try:
                self.buffer += self.ser.read(self.ser.in_waiting)
            except Exception as e:
                self.get_logger().error(f'读取串口出错: {e}')
                return
            
            # 帧对齐处理
            while len(self.buffer) >= RECEIVE_FRAME_LEN:
                header_idx = self.buffer.find(bytes([FRAME_HEADER]))
                if header_idx == -1:
                    self.buffer = b''
                    break
                if header_idx > 0:
                    self.buffer = self.buffer[header_idx:] # 移除头部噪声
                
                if len(self.buffer) < RECEIVE_FRAME_LEN:
                    break
                
                frame = self.buffer[:RECEIVE_FRAME_LEN]
                
                # 校验：检查包尾和校验和
                if frame[RECEIVE_FRAME_LEN-1] == FRAME_TAIL:
                    received_sum = frame[34]
                    calculated_sum = self.check_sum(frame[:34])
                    
                    if received_sum == calculated_sum:
                        self.parse_and_publish(frame)
                    else:
                        self.get_logger().warn('校验和失败，丢弃一帧数据')
                
                # 移除此处理过的帧
                self.buffer = self.buffer[RECEIVE_FRAME_LEN:]

    def parse_and_publish(self, frame):
        """ 解析 36 字节协议并分发到各个 ROS 话题 """
        try:
            # 1. 解析实时速度 (大端序 short, 偏移 2, 4, 6)
            vx = struct.unpack('>h', frame[2:4])[0] / 1000.0
            vy = struct.unpack('>h', frame[4:6])[0] / 1000.0
            vz = struct.unpack('>h', frame[6:8])[0] / 1000.0

            # 2. 解析 IMU 数据 (大端序 short, 偏移 8-19)
            # 加速度转换为 m/s^2 (假设原始数据单位是 1mg)
            ax = struct.unpack('>h', frame[8:10])[0] / 1000.0 * 9.8
            ay = struct.unpack('>h', frame[10:12])[0] / 1000.0 * 9.8
            az = struct.unpack('>h', frame[12:14])[0] / 1000.0 * 9.8
            # 角速度转换为 rad/s (原单位是 100*deg/s)
            gx = math.radians(struct.unpack('>h', frame[14:16])[0] / 100.0)
            gy = math.radians(struct.unpack('>h', frame[16:18])[0] / 100.0)
            gz = math.radians(struct.unpack('>h', frame[18:20])[0] / 100.0)

            # 3. 解析全局座标 (小端序 float, 偏移 22, 26, 30)
            gx_pos, gy_pos, g_yaw = struct.unpack('<fff', frame[22:34])
            
            # --- 构造通用数据 ---
            curr_time = self.get_clock().now().to_msg()
            q = self.euler_to_quaternion(0, 0, g_yaw)

            # --- A. 发布 Pose2D (你的原始需求) ---
            p_msg = Pose2D()
            p_msg.x = float(gx_pos)
            p_msg.y = float(gy_pos)
            p_msg.theta = float(g_yaw)
            self.pose_pub.publish(p_msg)

            # --- B. 发布标准 Odometry ---
            odom = Odometry()
            odom.header.stamp = curr_time
            odom.header.frame_id = "odom"
            odom.child_frame_id = "base_link"
            odom.pose.pose.position.x = float(gx_pos)
            odom.pose.pose.position.y = float(gy_pos)
            odom.pose.pose.position.z = 0.0
            # 修复：参数名是 z 而不是 qz
            odom.pose.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
            odom.twist.twist.linear.x = float(vx)
            odom.twist.twist.linear.y = float(vy)
            odom.twist.twist.angular.z = float(vz)
            self.odom_pub.publish(odom)

            # --- C. 发布 IMU ---
            imu = Imu()
            imu.header.stamp = curr_time
            imu.header.frame_id = "imu_link"
            imu.linear_acceleration.x = ax
            imu.linear_acceleration.y = ay
            imu.linear_acceleration.z = az
            imu.angular_velocity.x = gx
            imu.angular_velocity.y = gy
            imu.angular_velocity.z = gz
            self.imu_pub.publish(imu)

            # --- D. 发布 TF 坐标转换 ---
            t = TransformStamped()
            t.header.stamp = curr_time
            t.header.frame_id = "odom"
            t.child_frame_id = "base_link"
            t.transform.translation.x = float(gx_pos)
            t.transform.translation.y = float(gy_pos)
            t.transform.translation.z = 0.0
            t.transform.rotation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
            self.tf_broadcaster.sendTransform(t)

        except Exception as e:
            self.get_logger().error(f'数据解析关键错误: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = OriginCarBaseNode()
    
    # 优雅退出处理
    def handle_sigint(sig, frame):
        node.get_logger().info('正在关闭节点...')
        rclpy.shutdown()
    
    signal.signal(signal.SIGINT, handle_sigint)

    try:
        rclpy.spin(node)
    except rclpy.executors.ExternalShutdownException:
        pass
    finally:
        if rclpy.ok():
            node.ser.close()
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()
