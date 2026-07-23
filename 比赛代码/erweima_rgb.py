#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
下相机二维码识别节点。

订阅：
    /image
        sensor_msgs/msg/CompressedImage
        下相机压缩图像。

    /qr_code_detection
        ai_msgs/msg/PerceptionTargets
        X3 YOLO 发布的二维码检测结果。
        YOLO 只负责触发全画面扫码，不再根据检测框裁剪 ROI。
        首次触发时优先解析与检测消息 header.stamp 对应的完整图像。

    /qr_direction_result
        std_msgs/msg/String
        最终扫码结果。收到非空结果后停止扫码。

发布：
    /qr_direction_result
        std_msgs/msg/String
        仅发布方向，保持现有导航链路兼容：
        奇数二维码 -> 顺时针
        偶数二维码 -> 逆时针

    /qr_full_result
        std_msgs/msg/String
        发布完整 JSON，例如：{"number":123,"direction":"顺时针"}
        供车载屏幕或语音节点展示原始数字和运行方向。

工作方式：
    1. 平时只缓存 /image 的压缩数据，不做 JPEG 解码和二维码解析；
    2. /qr_code_detection 中出现二维码后，YOLO 只作为扫码触发器；
    3. 优先解析与 YOLO 检测时间戳完全匹配的完整图像；
    4. 若首帧失败，进入短时间全画面连续抢扫；
    5. 抢扫期间只保留最新图像，不积压旧帧；
    6. 二维码解析在独立工作线程中执行，不阻塞 ROS 图像回调；
    7. 成功后立即发布原有两个结果话题并停止扫码。

停止条件：
    1. 本节点成功识别二维码并发布结果；
    2. 收到其他节点发布的非空扫码结果。

停止后会注销 /image 和 /qr_code_detection 订阅，
不再接收图像，也不再执行二维码解码。
"""

import json
import threading
import time
from collections import deque
from typing import Deque, Optional, Set, Tuple

import cv2
import numpy as np
import rclpy

from ai_msgs.msg import PerceptionTargets
from pyzbar.pyzbar import decode
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String


# (stamp_ns, frame_sequence, compressed_bytes)
CompressedFrame = Tuple[int, int, bytes]

# (stamp_ns, frame_sequence, compressed_bytes, source_description)
ScanFrame = Tuple[int, int, bytes, str]


class BottomCameraQRNode(Node):

    def __init__(self):
        super().__init__('bottom_camera_qr_node')

        # ============================================================
        # 参数
        # ============================================================

        self.declare_parameter('image_topic', '/image')
        self.declare_parameter(
            'detection_topic',
            '/qr_code_detection'
        )
        self.declare_parameter(
            'result_topic',
            '/qr_direction_result'
        )
        self.declare_parameter(
            'full_result_topic',
            '/qr_full_result'
        )

        self.declare_parameter('qr_target_type', 'qr_code')

        # 保留旧参数名，防止现有 YAML 或启动命令传入后失去兼容性。
        # 当前版本不再裁剪 ROI，因此这两个参数不会参与图像处理。
        self.declare_parameter('roi_expand_ratio', 2.5)
        self.declare_parameter('min_roi_size', 15)

        # 缓存约 3 秒压缩图像（30 FPS 时 90 帧），
        # 仅用于寻找与 YOLO 检测时间戳完全匹配的完整图像。
        self.declare_parameter('image_cache_size', 90)

        # YOLO 触发后连续全画面扫码的持续时间。
        # 新的二维码检测会刷新此时间窗口。
        self.declare_parameter('burst_scan_duration', 0.45)

        # 每隔多少个实际处理帧执行一次较重的增强方法。
        # 原图、灰度与 OpenCV 解码每帧都会尝试；
        # CLAHE、锐化、自适应二值化等按帧轮换，避免处理过重。
        self.declare_parameter('heavy_process_every_n', 3)

        self.image_topic = str(
            self.get_parameter('image_topic').value
        )
        self.detection_topic = str(
            self.get_parameter('detection_topic').value
        )
        self.result_topic = str(
            self.get_parameter('result_topic').value
        )
        self.full_result_topic = str(
            self.get_parameter('full_result_topic').value
        )
        self.qr_target_type = str(
            self.get_parameter('qr_target_type').value
        )

        self.image_cache_size = max(
            10,
            int(self.get_parameter('image_cache_size').value)
        )
        self.burst_scan_duration = max(
            0.05,
            float(self.get_parameter('burst_scan_duration').value)
        )
        self.heavy_process_every_n = max(
            1,
            int(self.get_parameter('heavy_process_every_n').value)
        )

        # 类型名统一忽略大小写、空格、下划线和连字符。
        # 因此 QR CODE / qr_code / QR-CODE 都会归一化成 qrcode。
        self.qr_target_type_normalized = self._normalize_target_type(
            self.qr_target_type
        )

        # ============================================================
        # 状态
        # ============================================================

        # RLock 同时供 Condition 使用，并允许同一线程内安全嵌套。
        self._lock = threading.RLock()
        self._worker_condition = threading.Condition(self._lock)

        # 缓存的是压缩字节，不在 30 FPS 图像回调里执行 cv2.imdecode。
        self._image_cache: Deque[CompressedFrame] = deque(
            maxlen=self.image_cache_size
        )
        self._latest_frame: Optional[CompressedFrame] = None
        self._frame_sequence = 0

        # 精确时间戳匹配帧优先于普通 burst 最新帧。
        self._priority_scan_frame: Optional[ScanFrame] = None

        # burst 模式只保留最新帧；新帧直接覆盖旧帧，不形成积压队列。
        self._latest_burst_frame: Optional[ScanFrame] = None
        self._burst_active = False
        self._burst_deadline = 0.0
        self._burst_session_count = 0

        self._is_done = False
        self._worker_shutdown = False

        self._last_sync_warning_time = 0.0
        self._exact_match_count = 0
        self._sync_miss_count = 0

        # 防止同一个相机帧既作为精确匹配帧、又作为 burst 最新帧重复处理。
        self._processed_sequences: Set[int] = set()
        self._processed_sequence_order: Deque[int] = deque()

        self.attempt_count = 0
        self.first_attempt_time = None
        self.success_method = None

        # OpenCV 解码器作为 pyzbar 的另一条解码路径。
        self._opencv_qr_detector = cv2.QRCodeDetector()

        # ============================================================
        # ROS 通信
        # 以下订阅、发布话题、消息类型与 QoS 保持原代码不变。
        # ============================================================

        # /image 在 yolo_513.py 中是 CompressedImage。
        self.image_sub = self.create_subscription(
            CompressedImage,
            self.image_topic,
            self.image_callback,
            qos_profile_sensor_data
        )

        # X3 YOLO 发布的二维码检测结果。
        self.detection_sub = self.create_subscription(
            PerceptionTargets,
            self.detection_topic,
            self.detection_callback,
            10
        )

        # 扫码结果发布器。
        self.result_pub = self.create_publisher(
            String,
            self.result_topic,
            10
        )

        # 完整二维码结果发布器：保留原始数字，供屏幕/语音展示。
        # 与 OLED 节点保持一致：RELIABLE + TRANSIENT_LOCAL + depth=1。
        # 这样即使 OLED 稍晚启动，也能收到最近一次二维码数字。
        full_result_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.full_result_pub = self.create_publisher(
            String,
            self.full_result_topic,
            full_result_qos
        )

        # 同时监听结果。如果其他节点已经得到结果，
        # 本节点收到后立即停止继续扫码。
        self.result_sub = self.create_subscription(
            String,
            self.result_topic,
            self.result_callback,
            10
        )

        # 二维码解析放到独立线程，不占用 ROS 回调线程。
        self._worker_thread = threading.Thread(
            target=self._scan_worker,
            name='bottom_camera_qr_scan_worker',
            daemon=True,
        )
        self._worker_thread.start()

        self.get_logger().info(
            '下相机扫码节点已启动：'
            f'图像={self.image_topic}，'
            f'二维码检测={self.detection_topic}，'
            f'目标类型={self.qr_target_type}，'
            f'图像缓存={self.image_cache_size}帧，'
            f'方向结果={self.result_topic}，'
            f'完整结果={self.full_result_topic}'
        )
        self.get_logger().info(
            '扫码模式：YOLO只负责触发，不裁剪ROI；'
            f'触发后连续全画面抢扫={self.burst_scan_duration:.2f}s；'
            '工作线程只处理最新帧'
        )

    # ================================================================
    # 时间戳与类型名工具
    # ================================================================

    @staticmethod
    def _stamp_to_ns(stamp):
        """把 ROS builtin_interfaces/Time 转成纳秒；无效时间戳返回 0。"""
        if stamp is None:
            return 0

        try:
            sec = int(getattr(stamp, 'sec', 0))
            nanosec = int(getattr(stamp, 'nanosec', 0))
        except (TypeError, ValueError):
            return 0

        if sec < 0 or nanosec < 0:
            return 0

        return sec * 1_000_000_000 + nanosec

    @classmethod
    def _message_stamp_ns(cls, msg):
        header = getattr(msg, 'header', None)
        return cls._stamp_to_ns(
            getattr(header, 'stamp', None)
        )

    @staticmethod
    def _normalize_target_type(value):
        """忽略大小写及分隔符，统一 QR CODE、qr_code 等写法。"""
        return ''.join(
            character
            for character in str(value or '').strip().lower()
            if character.isalnum()
        )

    def _target_is_qr(self, target_type):
        """只接收二维码类别；兼容历史发布端的 QR CODE/QR CODE BOARD。"""
        normalized = self._normalize_target_type(target_type)

        if not normalized:
            # 旧发布端可能没有填写 type；
            # 该话题本身是二维码专用话题。
            return True

        return normalized in {
            self.qr_target_type_normalized,
            'qrcode',
            'qrcodeboard',
        }

    def _warn_sync(self, message):
        now = time.monotonic()

        if now - self._last_sync_warning_time < 1.0:
            return

        self._last_sync_warning_time = now
        self.get_logger().warning(message)

    # ================================================================
    # 图像缓存与检测帧匹配
    # ================================================================

    def _frame_for_detection(
        self,
        msg: PerceptionTargets,
    ) -> Tuple[Optional[CompressedFrame], int, str]:
        """
        优先返回与检测消息 header.stamp 完全相同的完整压缩图像。

        完全匹配失败时仍然使用最新完整图像启动救急 burst，
        因为 YOLO 在本节点中只负责确认二维码已经进入视野。
        """
        detection_stamp_ns = self._message_stamp_ns(msg)

        with self._lock:
            latest_frame = self._latest_frame

            if detection_stamp_ns > 0:
                for frame in reversed(self._image_cache):
                    image_stamp_ns, _, _ = frame

                    if image_stamp_ns == detection_stamp_ns:
                        self._exact_match_count += 1
                        return frame, detection_stamp_ns, 'exact'

                self._sync_miss_count += 1

                oldest_stamp = (
                    self._image_cache[0][0]
                    if self._image_cache
                    else 0
                )
                newest_stamp = (
                    self._image_cache[-1][0]
                    if self._image_cache
                    else 0
                )
            else:
                oldest_stamp = 0
                newest_stamp = 0

        if detection_stamp_ns <= 0:
            if latest_frame is None:
                self._warn_sync(
                    '二维码检测消息没有有效 header.stamp，'
                    '且尚无图像可用于全画面救急扫码'
                )
                return None, 0, 'invalid'

            self._warn_sync(
                '二维码检测消息没有有效 header.stamp，'
                '使用最新完整图像触发救急扫码；'
                '请确认相机与YOLO消息都保留原始header.stamp'
            )
            return latest_frame, 0, 'latest_fallback'

        if latest_frame is None:
            self._warn_sync(
                '没有找到二维码检测对应的同时间戳图像，'
                '且当前没有最新图像可降级使用：'
                f'detection={detection_stamp_ns}，'
                f'cache=[{oldest_stamp}, {newest_stamp}]，'
                f'miss={self._sync_miss_count}'
            )
            return None, detection_stamp_ns, 'miss_no_image'

        self._warn_sync(
            '没有找到二维码检测对应的同时间戳图像，'
            '本次先使用最新完整图像并继续burst抢扫：'
            f'detection={detection_stamp_ns}，'
            f'cache=[{oldest_stamp}, {newest_stamp}]，'
            f'miss={self._sync_miss_count}'
        )
        return latest_frame, detection_stamp_ns, 'latest_after_miss'

    # ================================================================
    # 状态控制
    # ================================================================

    def _done(self):
        with self._lock:
            return self._is_done

    def _claim_result(self):
        """
        原子地取得最终结果发布权。

        防止工作线程与外部结果回调同时触发，
        或多个解码方法在很短时间内重复发布。
        """
        with self._worker_condition:
            if self._is_done:
                return False

            self._is_done = True
            self._burst_active = False
            self._priority_scan_frame = None
            self._latest_burst_frame = None
            self._worker_condition.notify_all()
            return True

    def _start_or_refresh_burst(
        self,
        matched_frame: Optional[CompressedFrame],
        sync_mode: str,
    ):
        """由一次真实二维码 YOLO 检测启动或刷新连续抢扫窗口。"""
        now = time.monotonic()

        with self._worker_condition:
            if self._is_done or self._worker_shutdown:
                return False

            was_active = (
                self._burst_active
                and now < self._burst_deadline
            )

            self._burst_active = True
            self._burst_deadline = now + self.burst_scan_duration

            if not was_active:
                self._burst_session_count += 1

            if matched_frame is not None:
                stamp_ns, sequence, compressed_bytes = matched_frame

                # YOLO 对应帧优先处理。
                self._priority_scan_frame = (
                    stamp_ns,
                    sequence,
                    compressed_bytes,
                    f'YOLO触发完整图/{sync_mode}',
                )

            self._worker_condition.notify_all()
            return not was_active

    def _stop_scanning(self, reason):
        """
        停止本节点继续接收图像和 YOLO 检测结果。

        result_sub 不注销，使节点仍能保持运行状态；
        image_sub 和 detection_sub 注销后不再产生扫码计算。
        """
        with self._worker_condition:
            already_done = self._is_done

            self._is_done = True
            self._burst_active = False
            self._burst_deadline = 0.0
            self._latest_frame = None
            self._priority_scan_frame = None
            self._latest_burst_frame = None
            self._image_cache.clear()

            # 扫码成功或收到外部结果后，工作线程不再需要继续运行。
            self._worker_shutdown = True
            self._worker_condition.notify_all()

        image_sub = self.image_sub
        detection_sub = self.detection_sub

        self.image_sub = None
        self.detection_sub = None

        if image_sub is not None:
            try:
                self.destroy_subscription(image_sub)
            except Exception as exc:
                self.get_logger().debug(
                    f'停止图像订阅失败: {exc}'
                )

        if detection_sub is not None:
            try:
                self.destroy_subscription(detection_sub)
            except Exception as exc:
                self.get_logger().debug(
                    f'停止检测结果订阅失败: {exc}'
                )

        if not already_done:
            self.get_logger().info(
                f'扫码监听已停止：{reason}'
            )

    def shutdown_worker(self):
        """节点销毁前停止并回收扫码工作线程。"""
        with self._worker_condition:
            self._worker_shutdown = True
            self._worker_condition.notify_all()

        worker = getattr(self, '_worker_thread', None)

        if (
            worker is not None
            and worker.is_alive()
            and threading.current_thread() is not worker
        ):
            worker.join(timeout=2.0)

    # ================================================================
    # /image 图像回调
    # ================================================================

    def image_callback(self, msg: CompressedImage):
        """
        只缓存压缩图像。

        不在 ROS 回调中执行 JPEG 解码或二维码解析，
        确保能够尽量跟上高帧率下相机。
        """
        if self._done():
            return

        try:
            if not msg.data:
                return

            # 转为不可变 bytes，避免消息对象生命周期结束后数据失效。
            compressed_bytes = bytes(msg.data)
            stamp_ns = self._message_stamp_ns(msg)
            now = time.monotonic()

            with self._worker_condition:
                if self._is_done or self._worker_shutdown:
                    return

                self._frame_sequence += 1
                frame = (
                    stamp_ns,
                    self._frame_sequence,
                    compressed_bytes,
                )
                self._latest_frame = frame

                # 有效时间戳才进入精确匹配缓存；
                # 时间戳为0时只保留最新帧作为降级来源。
                if stamp_ns > 0:
                    self._image_cache.append(frame)

                if self._burst_active:
                    if now < self._burst_deadline:
                        # 只保留最新一帧，永远不积压旧扫码任务。
                        self._latest_burst_frame = (
                            stamp_ns,
                            self._frame_sequence,
                            compressed_bytes,
                            'burst最新完整图',
                        )
                        self._worker_condition.notify_all()
                    else:
                        self._burst_active = False
                        self._latest_burst_frame = None
                        self._worker_condition.notify_all()

        except Exception as exc:
            self.get_logger().debug(
                f'/image 图像缓存异常: {exc}'
            )

    # ================================================================
    # /qr_code_detection 回调
    # ================================================================

    def detection_callback(
        self,
        msg: PerceptionTargets
    ):
        """
        YOLO 只负责判断是否存在二维码并触发全画面扫码。

        检测框坐标不会用于裁剪。
        """
        if self._done():
            return

        qr_found = False
        best_confidence = 0.0

        try:
            for target in msg.targets:
                if not self._target_is_qr(target.type):
                    continue

                # 保持原检测语义：目标至少要带一个尺寸有效的ROI，
                # 但ROI坐标只用于确认YOLO确实输出了有效二维码检测，
                # 后续绝不裁剪图像。
                for roi in target.rois:
                    rect = getattr(roi, 'rect', None)

                    if rect is None:
                        continue

                    width = int(getattr(rect, 'width', 0))
                    height = int(getattr(rect, 'height', 0))

                    if width <= 0 or height <= 0:
                        continue

                    qr_found = True
                    best_confidence = max(
                        best_confidence,
                        float(getattr(roi, 'confidence', 0.0))
                    )

        except Exception as exc:
            self.get_logger().debug(
                f'解析二维码检测结果异常: {exc}'
            )
            return

        if not qr_found:
            return

        matched_frame, matched_stamp_ns, sync_mode = (
            self._frame_for_detection(msg)
        )

        new_session = self._start_or_refresh_burst(
            matched_frame,
            sync_mode,
        )

        self.get_logger().info(
            'YOLO检测到二维码，启动全画面扫码：'
            f'置信度={best_confidence:.3f}，'
            f'图像同步={sync_mode}，'
            f'stamp_ns={matched_stamp_ns}，'
            f'新burst={new_session}',
            throttle_duration_sec=0.5
        )

    # ================================================================
    # /qr_direction_result 回调
    # ================================================================

    def result_callback(self, msg: String):
        """
        监听最终扫码结果。

        如果其他节点已经发布了非空结果，本节点立即停止扫码。
        本节点自己发布结果后也会收到该消息，但此时已经停止，
        不会重复执行识别。
        """
        result = msg.data.strip()

        if not result:
            return

        if self._done():
            return

        self.get_logger().info(
            f'收到已有扫码结果: {result}'
        )
        self._stop_scanning(
            f'已有结果 {result}'
        )

    # ================================================================
    # 扫码工作线程
    # ================================================================

    def _sequence_already_processed_locked(
        self,
        sequence: int,
    ) -> bool:
        return sequence in self._processed_sequences

    def _mark_sequence_processed_locked(
        self,
        sequence: int,
    ):
        if sequence in self._processed_sequences:
            return

        self._processed_sequences.add(sequence)
        self._processed_sequence_order.append(sequence)

        # 只保存最近128个序号，防止集合无限增长。
        while len(self._processed_sequence_order) > 128:
            old_sequence = self._processed_sequence_order.popleft()
            self._processed_sequences.discard(old_sequence)

    def _next_scan_frame_locked(
        self,
    ) -> Optional[ScanFrame]:
        """
        取下一张要处理的图像。

        优先级：
            1. YOLO时间戳匹配帧；
            2. burst期间最新相机帧。
        """
        if self._priority_scan_frame is not None:
            frame = self._priority_scan_frame
            self._priority_scan_frame = None
            return frame

        if (
            self._burst_active
            and self._latest_burst_frame is not None
        ):
            frame = self._latest_burst_frame
            self._latest_burst_frame = None
            return frame

        return None

    def _scan_worker(self):
        """独立线程：只处理最新完整图像，不阻塞ROS回调。"""
        burst_timeout_to_log = None

        while True:
            with self._worker_condition:
                frame_to_process = None

                while frame_to_process is None:
                    if self._worker_shutdown or self._is_done:
                        return

                    now = time.monotonic()

                    if (
                        self._burst_active
                        and now >= self._burst_deadline
                    ):
                        self._burst_active = False
                        self._latest_burst_frame = None
                        burst_timeout_to_log = self._burst_session_count

                    frame_to_process = self._next_scan_frame_locked()

                    if frame_to_process is not None:
                        _, sequence, _, _ = frame_to_process

                        if self._sequence_already_processed_locked(sequence):
                            frame_to_process = None
                            continue

                        # 在真正解码前先标记，避免同帧通过两条入口重复提交。
                        self._mark_sequence_processed_locked(sequence)
                        break

                    if burst_timeout_to_log is not None:
                        break

                    wait_timeout = 0.5

                    if self._burst_active:
                        wait_timeout = max(
                            0.001,
                            min(
                                0.5,
                                self._burst_deadline - now,
                            )
                        )

                    self._worker_condition.wait(
                        timeout=wait_timeout
                    )

            if burst_timeout_to_log is not None:
                self.get_logger().info(
                    f'本轮全画面burst扫码超时：'
                    f'session={burst_timeout_to_log}，'
                    f'累计处理帧={self.attempt_count}；'
                    '继续等待下一次YOLO二维码触发'
                )
                burst_timeout_to_log = None

                if frame_to_process is None:
                    continue

            if frame_to_process is None:
                continue

            (
                stamp_ns,
                _,
                compressed_bytes,
                source_description,
            ) = frame_to_process

            try:
                compressed_array = np.frombuffer(
                    compressed_bytes,
                    dtype=np.uint8,
                )
                image = cv2.imdecode(
                    compressed_array,
                    cv2.IMREAD_COLOR,
                )

                if image is None or image.size == 0:
                    self.get_logger().warning(
                        '扫码工作线程解压完整图像失败',
                        throttle_duration_sec=2.0
                    )
                    continue

                if self._process_full_image(
                    image,
                    source_description,
                    stamp_ns,
                ):
                    return

            except Exception as exc:
                self.get_logger().debug(
                    f'完整图像扫码工作线程异常: {exc}'
                )

    # ================================================================
    # 完整图像处理
    # ================================================================

    def _process_full_image(
        self,
        image,
        source_description,
        stamp_ns,
    ):
        """
        直接解析完整640x480图像，不裁剪ROI。

        每帧先走快速通道，再轮换一种增强方法；
        既利用多帧机会，又避免每帧把所有重处理全部执行一遍。
        """
        if self._done():
            return False

        if self.attempt_count == 0:
            self.first_attempt_time = time.time()

        self.attempt_count += 1
        attempt = self.attempt_count

        try:
            gray = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY,
            )

            # 拉普拉斯方差仅用于日志观察清晰度，不直接丢弃模糊帧。
            sharpness = float(
                cv2.Laplacian(
                    gray,
                    cv2.CV_64F,
                ).var()
            )

            self.get_logger().info(
                f'全画面扫码：attempt={attempt}，'
                f'来源={source_description}，'
                f'stamp_ns={stamp_ns}，'
                f'清晰度={sharpness:.1f}',
                throttle_duration_sec=0.5
            )

            # --------------------------------------------------------
            # 快速通道：每个实际处理帧都会执行
            # --------------------------------------------------------

            if self._try_pyzbar(
                image,
                '完整彩色/pyzbar'
            ):
                return True

            if self._try_pyzbar(
                gray,
                '完整灰度/pyzbar'
            ):
                return True

            if self._try_opencv(
                image,
                '完整彩色/OpenCV'
            ):
                return True

            # --------------------------------------------------------
            # 轻量增强轮换
            # 不把所有重方法都压在同一帧上。
            # --------------------------------------------------------

            enhancement_slot = (
                (attempt - 1)
                % self.heavy_process_every_n
            )

            if enhancement_slot == 0:
                # Otsu对光照整体变化、黑白对比不足比较便宜。
                _, otsu = cv2.threshold(
                    gray,
                    0,
                    255,
                    cv2.THRESH_BINARY + cv2.THRESH_OTSU,
                )

                if self._try_pyzbar(
                    otsu,
                    '完整图Otsu/pyzbar'
                ):
                    return True

                if self._try_opencv(
                    otsu,
                    '完整图Otsu/OpenCV'
                ):
                    return True

            elif enhancement_slot == 1:
                # CLAHE改善局部明暗不均。
                clahe = cv2.createCLAHE(
                    clipLimit=3.0,
                    tileGridSize=(8, 8),
                )
                enhanced = clahe.apply(gray)

                if self._try_pyzbar(
                    enhanced,
                    '完整图CLAHE/pyzbar'
                ):
                    return True

                if self._try_opencv(
                    enhanced,
                    '完整图CLAHE/OpenCV'
                ):
                    return True

            else:
                # 轻度反遮罩锐化，避免旧9中心核过度放大噪声。
                blurred = cv2.GaussianBlur(
                    gray,
                    (0, 0),
                    1.0,
                )
                sharpened = cv2.addWeighted(
                    gray,
                    1.6,
                    blurred,
                    -0.6,
                    0.0,
                )

                if self._try_pyzbar(
                    sharpened,
                    '完整图轻度锐化/pyzbar'
                ):
                    return True

                # 自适应二值化作为同一轮的模糊/光照兜底。
                adaptive = cv2.adaptiveThreshold(
                    gray,
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    31,
                    5,
                )

                if self._try_pyzbar(
                    adaptive,
                    '完整图自适应二值化/pyzbar'
                ):
                    return True

                if self._try_opencv(
                    sharpened,
                    '完整图轻度锐化/OpenCV'
                ):
                    return True

        except Exception as exc:
            self.get_logger().debug(
                f'完整图像处理异常: {exc}'
            )

        return False

    # ================================================================
    # 二维码解码
    # ================================================================

    def _try_pyzbar(
        self,
        image,
        method_name,
    ):
        try:
            decoded_objects = decode(image)

            for decoded_object in decoded_objects:
                try:
                    qr_text = decoded_object.data.decode(
                        'utf-8'
                    ).strip()
                except UnicodeDecodeError:
                    continue

                if self._handle_qr_text(
                    qr_text,
                    method_name,
                ):
                    return True

            return False

        except Exception as exc:
            self.get_logger().debug(
                f'{method_name}异常: {exc}'
            )
            return False

    def _try_opencv(
        self,
        image,
        method_name,
    ):
        """使用OpenCV QRCodeDetector作为pyzbar之外的兜底。"""
        try:
            qr_text, _, _ = (
                self._opencv_qr_detector.detectAndDecode(image)
            )

            if qr_text:
                return self._handle_qr_text(
                    qr_text.strip(),
                    method_name,
                )

            # 某些OpenCV版本支持多二维码接口；
            # 单二维码失败后再轻量尝试一次，多版本返回值均做兼容处理。
            detect_multi = getattr(
                self._opencv_qr_detector,
                'detectAndDecodeMulti',
                None,
            )

            if detect_multi is None:
                return False

            result = detect_multi(image)

            if not isinstance(result, tuple) or len(result) < 2:
                return False

            retval = bool(result[0])
            decoded_info = result[1]

            if not retval or decoded_info is None:
                return False

            for qr_text_multi in decoded_info:
                qr_text_multi = str(
                    qr_text_multi or ''
                ).strip()

                if (
                    qr_text_multi
                    and self._handle_qr_text(
                        qr_text_multi,
                        f'{method_name}/Multi',
                    )
                ):
                    return True

            return False

        except Exception as exc:
            self.get_logger().debug(
                f'{method_name}异常: {exc}'
            )
            return False

    def _handle_qr_text(
        self,
        qr_text,
        method_name,
    ):
        """校验二维码内容并保持原有发布规则。"""
        try:
            qr_text = str(qr_text).strip()

            # 保持原来的规则，只接受纯数字二维码。
            if not qr_text.isdigit():
                self.get_logger().debug(
                    f'忽略非数字二维码: {qr_text}'
                )
                return False

            qr_number = int(qr_text)

            if qr_number % 2 != 0:
                direction = '顺时针'
            else:
                direction = '逆时针'

            # 原子地保证只发布一次。
            if not self._claim_result():
                return True

            self.success_method = method_name

            elapsed = 0.0

            if self.first_attempt_time is not None:
                elapsed = (
                    time.time()
                    - self.first_attempt_time
                )

            # 原方向话题保持不变，避免影响task_xian.py等导航节点。
            result_msg = String()
            result_msg.data = direction
            self.result_pub.publish(result_msg)

            # 完整结果话题保持不变，保留二维码原始数字供展示节点使用。
            full_result_msg = String()
            full_result_msg.data = json.dumps(
                {
                    'number': qr_number,
                    'direction': direction,
                },
                ensure_ascii=False,
                separators=(',', ':'),
            )
            self.full_result_pub.publish(full_result_msg)

            self.get_logger().info(
                '二维码识别成功：'
                f'数字={qr_number}，'
                f'结果={direction}，'
                f'完整消息={full_result_msg.data}，'
                f'方法={method_name}，'
                f'处理帧数={self.attempt_count}，'
                f'耗时={elapsed:.3f}s'
            )

            # _claim_result已经将_is_done设为True。
            # 这里继续注销图像和检测结果订阅。
            self._stop_scanning(
                f'本节点识别成功，结果为 {direction}'
            )
            return True

        except Exception as exc:
            self.get_logger().debug(
                f'处理二维码内容异常: {exc}'
            )
            return False


def main(args=None):
    # 不设置 CPU 亲和性，由操作系统自行调度。
    rclpy.init(args=args)
    node = BottomCameraQRNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown_worker()
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
