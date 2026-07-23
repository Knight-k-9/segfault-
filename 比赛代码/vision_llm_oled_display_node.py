#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import sys
import threading
from typing import List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSDurabilityPolicy,
    QoSReliabilityPolicy,
)
from std_msgs.msg import String

from PIL import ImageFont
from luma.core.interface.serial import i2c
from luma.oled.device import ssd1309
from luma.core.render import canvas


class VisionLLMOLEDDisplayNode(Node):
    """
    OLED 四行显示节点：

    第 1 行：固定显示 /qr_full_result 中解析出的二维码数字。
    第 2～4 行：显示 /vision_llm_result 的最新累计文本，并自动换行、
                只保留最后三行，实现随模型输出实时向上滚动。

    图生文发送节点每收到一个字符都会发布一次“累计文本”；本节点不再
    人工回放完整结果，因此无需等待模型全部生成完成。
    """

    def __init__(self):
        super().__init__('vision_llm_oled_display_node')

        # ===================== 参数 =====================
        self.declare_parameter('result_topic', '/vision_llm_result')
        self.declare_parameter(
            'stream_status_topic',
            '/vision_llm_stream_status'
        )
        self.declare_parameter('qr_full_result_topic', '/qr_full_result')
        self.declare_parameter(
            'font_path',
            '/root/zuizhong/task_2/tushengwen/zpix.ttf'
        )
        self.declare_parameter('font_size', 11)
        self.declare_parameter('i2c_port', 0)
        self.declare_parameter('i2c_address', 0x3C)
        self.declare_parameter('line_height', 16)
        self.declare_parameter('text_display_lines', 3)
        self.declare_parameter('qr_placeholder', '--')
        self.declare_parameter('qr_prefix', '')
        self.declare_parameter('show_stream_cursor', True)

        self.result_topic = str(self.get_parameter('result_topic').value)
        self.stream_status_topic = str(
            self.get_parameter('stream_status_topic').value
        )
        self.qr_full_result_topic = str(
            self.get_parameter('qr_full_result_topic').value
        )
        self.font_path = str(self.get_parameter('font_path').value)
        self.font_size = int(self.get_parameter('font_size').value)
        self.i2c_port = int(self.get_parameter('i2c_port').value)
        self.i2c_address = int(self.get_parameter('i2c_address').value)
        self.line_height = max(
            1,
            int(self.get_parameter('line_height').value)
        )
        self.text_display_lines = max(
            1,
            min(3, int(self.get_parameter('text_display_lines').value))
        )
        self.qr_placeholder = str(
            self.get_parameter('qr_placeholder').value
        )
        self.qr_prefix = str(self.get_parameter('qr_prefix').value)
        self.show_stream_cursor = bool(
            self.get_parameter('show_stream_cursor').value
        )

        # ===================== OLED =====================
        self.oled_device = None
        self.font_zh = None
        self.display_lock = threading.Lock()
        self.init_oled()

        # ===================== 显示状态 =====================
        self.state_lock = threading.RLock()
        self.qr_number = self.qr_placeholder
        self.llm_text = ''
        self.is_streaming = False
        self.state_version = 0

        self.stop_event = threading.Event()
        self.redraw_event = threading.Event()
        self.display_thread: Optional[threading.Thread] = None

        # 三个话题均使用“可靠 + TRANSIENT_LOCAL”。OLED 晚启动时可以收到
        # 二维码数字、图生文最新累计文本以及流式完成状态。
        latched_qos = QoSProfile(depth=1)
        latched_qos.reliability = QoSReliabilityPolicy.RELIABLE
        latched_qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL

        self.result_sub = self.create_subscription(
            String,
            self.result_topic,
            self.result_callback,
            latched_qos
        )
        self.status_sub = self.create_subscription(
            String,
            self.stream_status_topic,
            self.stream_status_callback,
            latched_qos
        )
        self.qr_sub = self.create_subscription(
            String,
            self.qr_full_result_topic,
            self.qr_full_result_callback,
            latched_qos
        )

        if self.oled_device is not None and self.font_zh is not None:
            self.display_thread = threading.Thread(
                target=self.display_loop,
                name='vision_llm_oled_live_display',
                daemon=True
            )
            self.display_thread.start()
            self.request_redraw()

        self.get_logger().info('二维码数字 + 图生文流式 OLED 节点已启动')
        self.get_logger().info(
            f'第1行二维码数字：{self.qr_full_result_topic}'
        )
        self.get_logger().info(
            f'第2～4行累计文本：{self.result_topic}'
        )
        self.get_logger().info(
            f'流式状态：{self.stream_status_topic}'
        )
        self.get_logger().info(
            f'OLED：I2C port={self.i2c_port}, '
            f'address=0x{self.i2c_address:02X}'
        )

    # =========================================================
    # OLED 初始化和文字测量
    # =========================================================
    def init_oled(self):
        try:
            serial_i2c = i2c(
                port=self.i2c_port,
                address=self.i2c_address
            )
            self.oled_device = ssd1309(serial_i2c)
            self.font_zh = ImageFont.truetype(
                self.font_path,
                self.font_size
            )
            self.oled_device.clear()
            self.get_logger().info('OLED 屏幕及中文字体初始化成功')
        except Exception as exc:
            self.get_logger().error(
                f'OLED 初始化失败；仍会接收并打印结果：{exc}'
            )
            self.oled_device = None
            self.font_zh = None

    def text_width(self, text: str) -> int:
        if not self.font_zh:
            return len(text)

        try:
            return int(self.font_zh.getlength(text))
        except AttributeError:
            return int(self.font_zh.getsize(text)[0])

    def fit_single_line(self, text: str, max_width: int) -> str:
        """把二维码数字限制在第一行宽度内。"""
        if self.text_width(text) <= max_width:
            return text

        fitted = ''
        for char in text:
            candidate = fitted + char
            if self.text_width(candidate) > max_width:
                break
            fitted = candidate
        return fitted

    def get_wrapped_lines(self, text: str, max_width: int) -> List[str]:
        if not text:
            return []

        lines: List[str] = []
        current_line = ''

        for char in text:
            if char == '\r':
                continue

            if char == '\n':
                lines.append(current_line)
                current_line = ''
                continue

            candidate = current_line + char
            if not current_line or self.text_width(candidate) <= max_width:
                current_line = candidate
            else:
                lines.append(current_line)
                current_line = char

        if current_line or text.endswith('\n'):
            lines.append(current_line)

        return lines

    # =========================================================
    # ROS 回调
    # =========================================================
    def qr_full_result_callback(self, msg: String):
        raw = str(msg.data).strip()
        if not raw:
            return

        number_text = ''
        try:
            payload = json.loads(raw)
            if isinstance(payload, dict) and 'number' in payload:
                number_text = str(payload['number']).strip()
        except json.JSONDecodeError:
            # 兼容直接只发布数字字符串的情况。
            if raw.isdigit():
                number_text = raw

        if not number_text:
            self.get_logger().warning(
                f'无法从二维码完整结果中提取 number：{raw!r}'
            )
            return

        with self.state_lock:
            self.qr_number = number_text
            self.state_version += 1

        self.get_logger().warning(f'OLED 第一行二维码数字：{number_text}')
        self.request_redraw()

    def result_callback(self, msg: String):
        text = str(msg.data)
        if not text:
            return

        with self.state_lock:
            self.llm_text = text
            self.state_version += 1
            char_count = len(text)

        # 不在每个字符都打印 warning，避免流式输出时日志刷屏。
        if char_count == 1 or char_count % 20 == 0:
            self.get_logger().info(
                f'OLED 已收到图生文累计文本：{char_count} 字符'
            )

        self.request_redraw()

    def stream_status_callback(self, msg: String):
        status = str(msg.data).strip().lower()
        if not status:
            return

        with self.state_lock:
            if status == 'start':
                self.llm_text = ''
                self.is_streaming = True
            elif status in ('done', 'completed'):
                self.is_streaming = False
            elif status.startswith('error'):
                self.is_streaming = False
            else:
                return

            self.state_version += 1

        self.get_logger().info(f'图生文流式状态：{status}')
        self.request_redraw()

    # =========================================================
    # 绘制线程
    # =========================================================
    def request_redraw(self):
        self.redraw_event.set()

    def get_state_snapshot(self) -> Tuple[str, str, bool, int]:
        with self.state_lock:
            return (
                self.qr_number,
                self.llm_text,
                self.is_streaming,
                self.state_version,
            )

    def draw_current_screen(self):
        if not self.oled_device or not self.font_zh:
            return

        qr_number, llm_text, is_streaming, _ = self.get_state_snapshot()
        width = int(self.oled_device.width)

        first_line = self.fit_single_line(
            f'{self.qr_prefix}{qr_number}',
            width
        )

        display_text = llm_text
        if is_streaming and self.show_stream_cursor:
            display_text += '_'

        wrapped_lines = self.get_wrapped_lines(display_text, width)
        text_lines = wrapped_lines[-self.text_display_lines:]

        with self.display_lock:
            with canvas(self.oled_device) as draw:
                # 第 1 行固定为二维码数字。
                draw.text(
                    (0, 0),
                    first_line,
                    fill='white',
                    font=self.font_zh
                )

                # 第 2～4 行显示图生文最后三行，文本增长后自然向上滚动。
                for index, line in enumerate(text_lines):
                    draw.text(
                        (0, (index + 1) * self.line_height),
                        line,
                        fill='white',
                        font=self.font_zh
                    )

    def display_loop(self):
        last_drawn_version = -1

        try:
            while rclpy.ok() and not self.stop_event.is_set():
                self.redraw_event.wait(timeout=0.2)
                self.redraw_event.clear()

                if self.stop_event.is_set():
                    break

                _, _, _, version = self.get_state_snapshot()
                if version == last_drawn_version:
                    continue

                self.draw_current_screen()
                last_drawn_version = version

        except Exception as exc:
            self.get_logger().warning(f'OLED 实时显示线程异常：{exc}')

    def clear_oled(self):
        if not self.oled_device:
            return

        try:
            with self.display_lock:
                self.oled_device.clear()
        except Exception as exc:
            self.get_logger().warning(f'清空 OLED 失败：{exc}')

    def cleanup(self):
        self.stop_event.set()
        self.redraw_event.set()

        if (
            self.display_thread is not None
            and self.display_thread.is_alive()
            and self.display_thread is not threading.current_thread()
        ):
            self.display_thread.join(timeout=1.0)

        self.clear_oled()


def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = VisionLLMOLEDDisplayNode()
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    except Exception as exc:
        print(f'节点启动失败：{exc}', file=sys.stderr)

    finally:
        if node is not None:
            node.cleanup()
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
