#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
二维码解析节点（多节点协作版）

输入：
  /aurora/rgb/image_raw        sensor_msgs/msg/Image
  /yolo_sd_erweima             std_msgs/msg/String
  /qr_direction_result         std_msgs/msg/String (订阅其他节点的结果)

输出：
  /qr_direction_result         std_msgs/msg/String

功能：
1. 严格按照 JSON 中的原图 header.stamp 匹配同一帧图像
2. 同时订阅 qr_direction_result，若其他节点已发布有效结果则停止
3. 测试多个 ROI 比例和预处理方法进行解码
4. 解码成功或收到其他节点结果后停止发布
"""

from collections import deque
import json
import os
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String

from pyzbar.pyzbar import ZBarSymbol, decode

class QRDecodeNode(Node):
    def __init__(self):
        super().__init__('qr_decode_node')

        # --------------------------------------------------------------
        # 参数
        # --------------------------------------------------------------
        self.declare_parameter('image_topic', '/aurora/rgb/image_raw')
        self.declare_parameter('yolo_topic', '/yolo_sd_erweima')
        self.declare_parameter('result_topic', '/qr_direction_result')
        self.declare_parameter('min_confidence', 0.35)
        self.declare_parameter('min_roi_size', 15)
        self.declare_parameter('image_cache_size', 120)

        self.image_topic = str(self.get_parameter('image_topic').value)
        self.yolo_topic = str(self.get_parameter('yolo_topic').value)
        self.result_topic = str(self.get_parameter('result_topic').value)
        self.min_confidence = float(
            self.get_parameter('min_confidence').value
        )
        self.min_roi_size = int(
            self.get_parameter('min_roi_size').value
        )
        cache_size = max(
            10,
            int(self.get_parameter('image_cache_size').value),
        )

        self.qr_class_name = 'QR CODE BOARD'
        self.qr_class_id = 0
        self.expand_ratios = [1.0, 1.2, 1.5, 2.0, 2.5]

        # --------------------------------------------------------------
        # 状态
        # --------------------------------------------------------------
        self.state_lock = threading.RLock()
        self.image_cache = deque(maxlen=cache_size)
        self.is_done = False
        self.callback_group = ReentrantCallbackGroup()

        self.frame_count = 0
        self.yolo_count = 0
        self.exact_match_count = 0
        self.sync_fail_count = 0
        self.decode_attempt_count = 0
        self.decode_success_count = 0

        self._last_sync_warn_time = 0.0
        self._last_empty_warn_time = 0.0
        self._last_summary_time = time.time()

        # --------------------------------------------------------------
        # 图像转换
        # --------------------------------------------------------------
        self.bridge = None
        self.use_cv_bridge = False
        try:
            from cv_bridge import CvBridge
            self.bridge = CvBridge()
            self.use_cv_bridge = True
            self.get_logger().info('✓ CvBridge 加载成功')
        except Exception as exc:
            self.get_logger().warning(
                f'⚠ CvBridge 不可用：{exc}，改用手动转换'
            )

        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        # 订阅图像
        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_cb,
            image_qos,
            callback_group=self.callback_group,
        )

        # 订阅 YOLO 检测结果
        self.yolo_sub = self.create_subscription(
            String,
            self.yolo_topic,
            self.yolo_cb,
            10,
            callback_group=self.callback_group,
        )

        # 订阅其他节点发布的二维码方向结果
        self.result_sub = self.create_subscription(
            String,
            self.result_topic,
            self.result_cb,
            10,
            callback_group=self.callback_group,
        )

        # 发布二维码方向结果
        self.result_pub = self.create_publisher(
            String,
            self.result_topic,
            10,
        )

        self.get_logger().info('=' * 76)
        self.get_logger().info('二维码解析节点已启动（多节点协作模式）')
        self.get_logger().info(f'图像输入：{self.image_topic}')
        self.get_logger().info(f'YOLO输入：{self.yolo_topic}')
        self.get_logger().info(
            f'结果话题：{self.result_topic} (订阅+发布)'
        )
        self.get_logger().info(
            '同步方式：sec 和 nanosec 必须完全相等'
        )
        self.get_logger().info(
            f'ROI比例：{self.expand_ratios}'
        )
        self.get_logger().info('=' * 76)

    # --------------------------------------------------------------
    # 结果回调 - 监听其他节点的成功结果
    # --------------------------------------------------------------
    def result_cb(self, msg):
        result = msg.data.strip()
        if result in ('顺时针', '逆时针'):
            with self.state_lock:
                if not self.is_done:
                    self.is_done = True
                    self.get_logger().info(
                        f'✓ 收到有效结果：{result}，停止本节点解析与发布'
                    )

    # --------------------------------------------------------------
    # 基础工具
    # --------------------------------------------------------------
    @staticmethod
    def stamp_to_ns(stamp):
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    @staticmethod
    def ns_to_text(stamp_ns):
        if stamp_ns <= 0:
            return '0.000000000'
        sec = stamp_ns // 1_000_000_000
        nanosec = stamp_ns % 1_000_000_000
        return f'{sec}.{nanosec:09d}'

    def _ros_image_to_cv2(self, msg):
        encoding = str(getattr(msg, 'encoding', '')).strip().lower()

        try:
            if encoding in ('nv12', 'yuv420sp'):
                height = int(msg.height)
                width = int(msg.width)
                expected = height * width * 3 // 2
                raw = np.frombuffer(msg.data, dtype=np.uint8)
                if raw.size < expected:
                    return None
                nv12 = raw[:expected].reshape(height * 3 // 2, width)
                return cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)
        except Exception as exc:
            self.get_logger().warning(f'NV12转换失败：{exc}')
            return None

        if self.use_cv_bridge and self.bridge is not None:
            try:
                return self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            except Exception as exc:
                self.get_logger().debug(f'CvBridge转换失败：{exc}')

        try:
            height = int(msg.height)
            width = int(msg.width)
            step = int(getattr(msg, 'step', 0))
            raw = np.frombuffer(msg.data, dtype=np.uint8)

            if encoding in ('bgr8', '8uc3'):
                if step >= width * 3:
                    rows = raw.reshape(height, step)
                    return rows[:, :width * 3].reshape(
                        height, width, 3
                    ).copy()
                return raw.reshape(height, width, 3).copy()

            if encoding == 'rgb8':
                if step >= width * 3:
                    rows = raw.reshape(height, step)
                    rgb = rows[:, :width * 3].reshape(
                        height, width, 3
                    )
                else:
                    rgb = raw.reshape(height, width, 3)
                return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            if encoding in ('mono8', '8uc1'):
                if step >= width:
                    rows = raw.reshape(height, step)
                    gray = rows[:, :width]
                else:
                    gray = raw.reshape(height, width)
                return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            if encoding == 'bgra8':
                bgra = raw.reshape(height, width, 4)
                return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)

            if encoding == 'rgba8':
                rgba = raw.reshape(height, width, 4)
                return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)

            self.get_logger().warning(
                f'暂不支持图像编码：{encoding}'
            )
            return None
        except Exception as exc:
            self.get_logger().warning(f'手动图像转换失败：{exc}')
            return None

    # --------------------------------------------------------------
    # 图像缓存与严格同步
    # --------------------------------------------------------------
    def image_cb(self, msg):
        with self.state_lock:
            if self.is_done:
                return

        image = self._ros_image_to_cv2(msg)
        if image is None:
            return

        stamp_ns = self.stamp_to_ns(msg.header.stamp)

        with self.state_lock:
            if self.is_done:
                return
            self.image_cache.append(
                (
                    stamp_ns,
                    image.copy(),
                    str(msg.header.frame_id),
                )
            )
            self.frame_count += 1

        self._maybe_log_summary()

    def _get_exact_image(self, target_stamp_ns):
        with self.state_lock:
            if not self.image_cache:
                return None, None, None

            cached = list(self.image_cache)

        for stamp_ns, image, frame_id in reversed(cached):
            if stamp_ns == target_stamp_ns:
                self.exact_match_count += 1
                return image.copy(), frame_id, 0

        nearest_stamp_ns, _, _ = min(
            cached,
            key=lambda item: abs(item[0] - target_stamp_ns),
        )
        nearest_gap_ns = nearest_stamp_ns - target_stamp_ns
        self.sync_fail_count += 1
        return None, None, nearest_gap_ns

    # --------------------------------------------------------------
    # JSON解析
    # --------------------------------------------------------------
    @staticmethod
    def _extract_json_header(payload):
        if not isinstance(payload, dict):
            return 0, ''

        header = payload.get('header', {})
        if not isinstance(header, dict):
            header = {}

        stamp = header.get('stamp', payload.get('stamp', {}))
        if not isinstance(stamp, dict):
            stamp = {}

        try:
            sec = int(stamp.get('sec', 0))
            nanosec = int(
                stamp.get('nanosec', stamp.get('nsec', 0))
            )
        except (TypeError, ValueError):
            sec = 0
            nanosec = 0

        frame_id = str(
            header.get('frame_id', payload.get('frame_id', ''))
        )
        return sec * 1_000_000_000 + nanosec, frame_id

    @staticmethod
    def _extract_bbox(det):
        if not isinstance(det, dict):
            return None

        bbox = det.get(
            'bbox',
            det.get('box', det.get('xyxy', None)),
        )

        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            try:
                return [float(v) for v in bbox[:4]]
            except (TypeError, ValueError):
                return None

        source = bbox if isinstance(bbox, dict) else det

        for keys in (
            ('x1', 'y1', 'x2', 'y2'),
            ('xmin', 'ymin', 'xmax', 'ymax'),
            ('left', 'top', 'right', 'bottom'),
        ):
            if all(key in source for key in keys):
                try:
                    return [float(source[key]) for key in keys]
                except (TypeError, ValueError):
                    return None

        if all(key in source for key in ('x', 'y')) and (
            all(key in source for key in ('w', 'h'))
            or all(key in source for key in ('width', 'height'))
        ):
            try:
                x = float(source['x'])
                y = float(source['y'])
                width = float(
                    source.get('w', source.get('width'))
                )
                height = float(
                    source.get('h', source.get('height'))
                )
                return [x, y, x + width, y + height]
            except (TypeError, ValueError):
                return None

        return None

    def _is_qr_detection(self, det):
        class_name = str(
            det.get('class_name', det.get('name', ''))
        ).strip().upper()
        class_id = det.get(
            'class_id',
            det.get('id', None),
        )

        name_match = class_name == self.qr_class_name.upper()

        try:
            id_match = (
                class_id is not None
                and int(class_id) == self.qr_class_id
            )
        except (TypeError, ValueError):
            id_match = False

        return name_match or id_match

    def _get_confidence(self, det):
        try:
            return float(
                det.get(
                    'confidence',
                    det.get('score', 1.0),
                )
            )
        except (TypeError, ValueError):
            return 0.0

    # --------------------------------------------------------------
    # ROI
    # --------------------------------------------------------------
    @staticmethod
    def _expand_bbox(bbox, ratio, image_w, image_h):
        x1, y1, x2, y2 = map(float, bbox)

        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        box_w = x2 - x1
        box_h = y2 - y1
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0

        new_w = box_w * float(ratio)
        new_h = box_h * float(ratio)

        nx1 = max(0, int(round(center_x - new_w / 2.0)))
        ny1 = max(0, int(round(center_y - new_h / 2.0)))
        nx2 = min(image_w, int(round(center_x + new_w / 2.0)))
        ny2 = min(image_h, int(round(center_y + new_h / 2.0)))

        return nx1, ny1, nx2, ny2

    # --------------------------------------------------------------
    # 解码
    # --------------------------------------------------------------
    @staticmethod
    def _resize(image, scale):
        h, w = image.shape[:2]
        return cv2.resize(
            image,
            (
                max(1, int(round(w * scale))),
                max(1, int(round(h * scale))),
            ),
            interpolation=cv2.INTER_LANCZOS4,
        )

    def _build_decode_variants(self, roi):
        gray = (
            cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            if roi.ndim == 3
            else roi.copy()
        )

        up2 = self._resize(roi, 2.0)
        up3 = self._resize(roi, 3.0)

        gray3 = (
            cv2.cvtColor(up3, cv2.COLOR_BGR2GRAY)
            if up3.ndim == 3
            else up3.copy()
        )

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8),
        ).apply(gray3)

        adaptive = cv2.adaptiveThreshold(
            gray3,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            5,
        )

        return [
            ('original', roi),
            ('gray', gray),
            ('up2', up2),
            ('up3', up3),
            ('up3_clahe', clahe),
            ('up3_adaptive', adaptive),
        ]

    def _decode_one_roi(self, roi, ratio):
        decoded_any = False
        numeric_direction = None
        seen_contents = set()

        for method_name, candidate in self._build_decode_variants(roi):
            self.decode_attempt_count += 1
            try:
                objects = decode(
                    candidate,
                    symbols=[ZBarSymbol.QRCODE],
                )
            except Exception as exc:
                self.get_logger().warning(
                    f'pyzbar异常 | ROI={ratio:.1f}x '
                    f'方法={method_name} | {exc}'
                )
                continue

            if not objects:
                continue

            decoded_any = True

            for obj in objects:
                try:
                    content = obj.data.decode(
                        'utf-8',
                        errors='replace',
                    ).strip()
                except Exception:
                    content = repr(obj.data)

                key = (method_name, content)
                if key in seen_contents:
                    continue
                seen_contents.add(key)

                self.get_logger().info(
                    f'🔎 pyzbar读到内容 | ROI={ratio:.1f}x '
                    f'方法={method_name} | 原始内容={content!r}'
                )

                if content.isdigit():
                    number = int(content)
                    numeric_direction = (
                        '顺时针'
                        if number % 2 != 0
                        else '逆时针'
                    )
                    self.get_logger().info(
                        f'✅ 数字二维码有效 | 数字={number} | '
                        f'方向={numeric_direction}'
                    )
                    return decoded_any, numeric_direction

                self.get_logger().info(
                    '⚠ 已识别二维码，但内容不是纯数字，不发布方向结果'
                )

        return decoded_any, numeric_direction

    # --------------------------------------------------------------
    # YOLO回调
    # --------------------------------------------------------------
    def yolo_cb(self, msg):
        with self.state_lock:
            if self.is_done:
                return

        self.yolo_count += 1

        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().error(
                f'YOLO JSON格式错误：{exc}'
            )
            return

        if isinstance(payload, dict):
            detections = payload.get(
                'detections',
                payload.get(
                    'results',
                    payload.get('objects', []),
                ),
            )
        elif isinstance(payload, list):
            detections = payload
            payload = {}
        else:
            self.get_logger().warning(
                f'不支持的JSON顶层类型：{type(payload).__name__}'
            )
            return

        if not isinstance(detections, list):
            self.get_logger().warning(
                'detections/results/objects 不是列表'
            )
            return

        if not detections:
            now = time.time()
            if now - self._last_empty_warn_time > 2.0:
                self._last_empty_warn_time = now
                self.get_logger().info(
                    'YOLO消息正常，但当前 detections 为空'
                )
            return

        target_stamp_ns, json_frame_id = (
            self._extract_json_header(payload)
        )

        if target_stamp_ns <= 0:
            self.get_logger().error(
                'YOLO JSON中没有有效header.stamp，无法进行严格同帧匹配'
            )
            return

        image, image_frame_id, nearest_gap_ns = (
            self._get_exact_image(target_stamp_ns)
        )

        if image is None:
            now = time.time()
            if now - self._last_sync_warn_time > 1.0:
                self._last_sync_warn_time = now
                self.get_logger().warning(
                    '❌ 严格同步失败 | '
                    f'YOLO时间戳={self.ns_to_text(target_stamp_ns)} | '
                    f'最近图像差={nearest_gap_ns / 1_000_000.0:.3f} ms | '
                    '本次不使用最近帧、不进行二维码解析'
                )
            return

        self.get_logger().info(
            '✓ 严格同步成功 | '
            f'时间戳={self.ns_to_text(target_stamp_ns)} | 时间差=0 ns'
        )

        image_h, image_w = image.shape[:2]
        found_qr = False

        for index, det in enumerate(detections):
            if not isinstance(det, dict):
                continue

            if not self._is_qr_detection(det):
                continue

            found_qr = True
            confidence = self._get_confidence(det)
            bbox = self._extract_bbox(det)

            self.get_logger().info(
                f'检测[{index}] | confidence={confidence:.3f} | '
                f'bbox={bbox}'
            )

            if confidence < self.min_confidence:
                self.get_logger().warning(
                    f'检测[{index}]置信度不足：'
                    f'{confidence:.3f} < {self.min_confidence:.3f}'
                )
                continue

            if bbox is None:
                self.get_logger().warning(
                    f'检测[{index}]无法提取bbox'
                )
                continue

            x1, y1, x2, y2 = bbox
            if (
                max(abs(x1), abs(x2)) <= 1.5
                and max(abs(y1), abs(y2)) <= 1.5
            ):
                bbox = [
                    x1 * image_w,
                    y1 * image_h,
                    x2 * image_w,
                    y2 * image_h,
                ]

            strict_box = self._expand_bbox(
                bbox,
                1.0,
                image_w,
                image_h,
            )
            sx1, sy1, sx2, sy2 = strict_box
            strict_w = sx2 - sx1
            strict_h = sy2 - sy1

            if (
                strict_w < self.min_roi_size
                or strict_h < self.min_roi_size
            ):
                self.get_logger().warning(
                    f'严格YOLO框过小：{strict_w}x{strict_h}'
                )
                continue

            for ratio in self.expand_ratios:
                rx1, ry1, rx2, ry2 = self._expand_bbox(
                    bbox,
                    ratio,
                    image_w,
                    image_h,
                )
                roi = image[ry1:ry2, rx1:rx2].copy()

                self.get_logger().info(
                    f'开始解析 ROI={ratio:.1f}x | '
                    f'范围=({rx1},{ry1})-({rx2},{ry2}) | '
                    f'尺寸={roi.shape[1]}x{roi.shape[0]}'
                )

                decoded_any, direction = self._decode_one_roi(
                    roi,
                    ratio,
                )

                if direction is not None:
                    # 发布前再次检查是否已被其他节点抢先完成
                    with self.state_lock:
                        if self.is_done:
                            return
                        self.is_done = True

                    result = String()
                    result.data = direction
                    self.result_pub.publish(result)
                    self.decode_success_count += 1
                    self.get_logger().info(
                        f'📤 已发布 {self.result_topic}: {direction}，'
                        '本节点停止'
                    )
                    return

                if decoded_any:
                    self.get_logger().info(
                        f'ROI={ratio:.1f}x 已读到二维码，'
                        '但未得到纯数字方向结果'
                    )
                else:
                    self.get_logger().info(
                        f'ROI={ratio:.1f}x 所有方法均未解码'
                    )

        if not found_qr:
            self.get_logger().warning(
                'JSON中有检测数据，但没有通过类别筛选的二维码'
            )

        self._maybe_log_summary()

    def _maybe_log_summary(self):
        now = time.time()
        if now - self._last_summary_time < 5.0:
            return

        self._last_summary_time = now
        with self.state_lock:
            cache_len = len(self.image_cache)

        self.get_logger().info(
            '统计 | '
            f'图像={self.frame_count} | '
            f'YOLO消息={self.yolo_count} | '
            f'严格匹配成功={self.exact_match_count} | '
            f'严格匹配失败={self.sync_fail_count} | '
            f'解码调用={self.decode_attempt_count} | '
            f'数字解码成功={self.decode_success_count} | '
            f'缓存={cache_len}/{self.image_cache.maxlen}'
        )

def main(args=None):
    try:
        os.sched_setaffinity(0, {2})
    except Exception:
        pass

    rclpy.init(args=args)
    node = QRDecodeNode()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()