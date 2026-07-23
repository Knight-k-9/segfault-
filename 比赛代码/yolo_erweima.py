#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
与 qr.bin 实际结构匹配的二维码 YOLO ROS 2 节点。

根据 qr.bin 静态元数据：
- 模型：QR_code_mew_tag_v7.0_detect_640x640_Bayese_nv12
- 训练输入：1x3x640x640，RGB，NCHW
- 运行时输入：NV12 / pyramid
- 输出：单个 output0，带 HzDequantize
- Detect 层已经包含 Sigmoid、Add、Mul、Pow、Reshape、Concat
- 因此 output0 是已解码的 [cx, cy, w, h, objectness, class_probability]
- 不允许再次使用 anchor/grid/sigmoid 做 raw YOLO 解码

输出话题：
- /yolo_sd_erweima              std_msgs/msg/String
"""

import json
import math
import os
import threading
import time
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from hobot_dnn import pyeasy_dnn as dnn
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String


class QrYoloDecodedNode(Node):
    def __init__(self):
        super().__init__('qr_yolo_decoded_node')

        self.declare_parameter(
            'bpu_model_path',
            '/root/zuizhong/task_1/yolo/models/qr.bin',
        )
        self.declare_parameter(
            'image_topic',
            '/aurora/rgb/image_raw',
        )
        self.declare_parameter(
            'detection_topic',
            '/yolo_sd_erweima',
        )
        self.declare_parameter('score_threshold', 0.35)
        self.declare_parameter('nms_threshold', 0.45)
        self.declare_parameter('max_pre_nms', 300)
        self.declare_parameter('max_detections', 20)

        self.model_path = str(
            self.get_parameter('bpu_model_path').value
        )
        self.image_topic = str(
            self.get_parameter('image_topic').value
        )
        self.detection_topic = str(
            self.get_parameter('detection_topic').value
        )
        self.score_threshold = float(
            self.get_parameter('score_threshold').value
        )
        self.nms_threshold = float(
            self.get_parameter('nms_threshold').value
        )
        self.max_pre_nms = int(
            self.get_parameter('max_pre_nms').value
        )
        self.max_detections = int(
            self.get_parameter('max_detections').value
        )

        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(
                f'二维码模型不存在：{self.model_path}'
            )

        models = dnn.load(self.model_path)
        if not models:
            raise RuntimeError(
                f'二维码模型加载失败：{self.model_path}'
            )

        self.model = models[0]

        input_shape = tuple(
            int(value)
            for value in self.model.inputs[0].properties.shape
        )
        self.input_h, self.input_w = self._infer_input_hw(
            input_shape
        )

        # 对 640×640、三尺度、每格3个anchor：
        # 80×80×3 + 40×40×3 + 20×20×3 = 25200。
        self.expected_rows = sum(
            (self.input_h // stride)
            * (self.input_w // stride)
            * 3
            for stride in (8, 16, 32)
        )

        self.bridge = CvBridge()
        self.infer_lock = threading.Lock()
        self.output_shape_logged = False

        self.frame_count = 0
        self.error_count = 0
        self.busy_drop_count = 0
        self.last_log_time = time.monotonic()

        # 相机通常使用 BEST_EFFORT，必须使用 sensor-data QoS。
        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            qos_profile_sensor_data,
        )

        self.detection_pub = self.create_publisher(
            String,
            self.detection_topic,
            10,
        )


        self.get_logger().info(
            '二维码YOLO节点已启动 | '
            f'模型={self.model_path} | '
            f'输入={self.input_w}x{self.input_h} | '
            f'输出解析=decoded xywh | '
            f'理论候选行数={self.expected_rows} | '
            f'检测话题={self.detection_topic}'
        )

    @staticmethod
    def _infer_input_hw(
        shape: Sequence[int],
    ) -> Tuple[int, int]:
        values = [int(value) for value in shape]

        if len(values) != 4:
            raise ValueError(
                f'无法识别模型输入shape：{values}'
            )

        # 训练形状通常为 NCHW：1×3×640×640。
        if values[1] in (1, 3, 4):
            return values[2], values[3]

        # 兼容运行库返回 NHWC。
        if values[3] in (1, 3, 4):
            return values[1], values[2]

        raise ValueError(
            f'无法从模型输入shape推断宽高：{values}'
        )

    @staticmethod
    def _bgr_to_nv12(
        bgr_image: np.ndarray,
    ) -> np.ndarray:
        height, width = bgr_image.shape[:2]

        if height % 2 != 0 or width % 2 != 0:
            raise ValueError(
                f'NV12图像宽高必须为偶数：{width}x{height}'
            )

        i420 = cv2.cvtColor(
            bgr_image,
            cv2.COLOR_BGR2YUV_I420,
        ).reshape(-1)

        y_size = height * width
        y_plane = i420[:y_size]
        u_plane = i420[
            y_size:y_size + y_size // 4
        ]
        v_plane = i420[
            y_size + y_size // 4:
            y_size + y_size // 2
        ]

        uv_plane = np.empty(
            y_size // 2,
            dtype=np.uint8,
        )
        uv_plane[0::2] = u_plane
        uv_plane[1::2] = v_plane

        return np.concatenate(
            (y_plane, uv_plane)
        )

    def _letterbox_to_nv12(
        self,
        image: np.ndarray,
    ) -> Tuple[np.ndarray, float, int, int]:
        original_h, original_w = image.shape[:2]

        scale = min(
            self.input_w / original_w,
            self.input_h / original_h,
        )

        resized_w = max(
            1,
            int(round(original_w * scale)),
        )
        resized_h = max(
            1,
            int(round(original_h * scale)),
        )

        resized = cv2.resize(
            image,
            (resized_w, resized_h),
            interpolation=cv2.INTER_LINEAR,
        )

        left = (self.input_w - resized_w) // 2
        right = self.input_w - resized_w - left
        top = (self.input_h - resized_h) // 2
        bottom = self.input_h - resized_h - top

        padded = cv2.copyMakeBorder(
            resized,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(127, 127, 127),
        )

        return (
            self._bgr_to_nv12(padded),
            scale,
            left,
            top,
        )

    @staticmethod
    def _runtime_shape(tensor) -> Tuple[int, ...]:
        try:
            return tuple(
                int(value)
                for value in tensor.properties.shape
            )
        except Exception:
            return tuple(
                int(value)
                for value in np.asarray(tensor.buffer).shape
            )

    def _reshape_decoded_output(
        self,
        tensor,
    ) -> np.ndarray:
        """
        将 output0 整理成 N×6。

        兼容：
        - (1, 25200, 6, 1)
        - (1, 25200, 6)
        - (25200, 6)
        - (1, 6, 25200, 1)
        - 一维 151200 元素
        """
        runtime_shape = self._runtime_shape(tensor)
        output = np.asarray(
            tensor.buffer,
            dtype=np.float32,
        )

        if not self.output_shape_logged:
            self.output_shape_logged = True
            self.get_logger().info(
                '模型实际输出 | '
                f'properties.shape={runtime_shape} | '
                f'buffer.shape={output.shape} | '
                f'元素数={output.size}'
            )

        squeezed = np.squeeze(output)

        if squeezed.ndim == 2:
            if squeezed.shape[1] == 6:
                predictions = squeezed
            elif squeezed.shape[0] == 6:
                predictions = squeezed.T
            else:
                raise ValueError(
                    '二维输出不包含长度为6的属性维：'
                    f'shape={squeezed.shape}'
                )
        else:
            if output.size % 6 != 0:
                raise ValueError(
                    'output0元素数量不能按6列解析：'
                    f'shape={runtime_shape}，'
                    f'元素数={output.size}'
                )
            predictions = output.reshape(-1, 6)

        if predictions.shape[1] != 6:
            raise ValueError(
                f'模型输出列数不是6：{predictions.shape}'
            )

        if predictions.shape[0] != self.expected_rows:
            raise ValueError(
                '模型输出行数不符合640×640 YOLOv5三尺度结构：'
                f'实际={predictions.shape[0]}，'
                f'理论={self.expected_rows}，'
                f'运行时shape={runtime_shape}'
            )

        if not np.all(np.isfinite(predictions)):
            raise ValueError(
                '模型输出包含NaN或Inf'
            )

        return predictions

    def _parse_decoded_output(
        self,
        tensor,
        scale: float,
        left: int,
        top: int,
        original_w: int,
        original_h: int,
    ) -> List[dict]:
        """
        qr.bin 已在模型内部完成：
        sigmoid + grid offset + anchor宽高变换 + 三尺度拼接。

        此处只执行：
        score阈值 -> 映射回原图 -> NMS。
        """
        predictions = self._reshape_decoded_output(tensor)

        coordinates = predictions[:, :4]
        objectness = predictions[:, 4].copy()
        class_probability = predictions[:, 5].copy()

        # 模型末端存在 Sigmoid；正常情况下这两列已经在0~1。
        # 量化反量化可能产生极小越界，因此只clip，绝不再次sigmoid。
        probability_valid_ratio = float(np.mean(
            (objectness >= -0.02)
            & (objectness <= 1.02)
            & (class_probability >= -0.02)
            & (class_probability <= 1.02)
        ))

        if probability_valid_ratio < 0.995:
            raise ValueError(
                '模型输出与decoded结构不符：'
                'objectness/class_probability不在0~1附近，'
                f'有效比例={probability_valid_ratio:.4f}'
            )

        objectness = np.clip(
            objectness,
            0.0,
            1.0,
        )
        class_probability = np.clip(
            class_probability,
            0.0,
            1.0,
        )

        scores = objectness * class_probability
        selected_indices = np.flatnonzero(
            scores >= self.score_threshold
        )

        if selected_indices.size == 0:
            return []

        # 防止异常输出把NMS和网页端拖慢。
        if selected_indices.size > self.max_pre_nms:
            top_order = np.argsort(
                scores[selected_indices]
            )[-self.max_pre_nms:]
            selected_indices = selected_indices[top_order]

        selected_boxes = coordinates[
            selected_indices
        ].copy()
        selected_scores = scores[
            selected_indices
        ]

        # 兼容归一化xywh；当前模型通常是640×640像素坐标。
        coordinate_p99 = float(
            np.percentile(
                np.abs(selected_boxes),
                99,
            )
        )
        if coordinate_p99 <= 2.0:
            selected_boxes[:, 0] *= self.input_w
            selected_boxes[:, 2] *= self.input_w
            selected_boxes[:, 1] *= self.input_h
            selected_boxes[:, 3] *= self.input_h

        detections: List[dict] = []

        for row, score in zip(
            selected_boxes,
            selected_scores,
        ):
            center_x, center_y, box_w, box_h = [
                float(value)
                for value in row
            ]

            if not all(
                math.isfinite(value)
                for value in (
                    center_x,
                    center_y,
                    box_w,
                    box_h,
                    float(score),
                )
            ):
                continue

            if box_w <= 1.0 or box_h <= 1.0:
                continue

            # decoded模型坐标位于模型输入平面。
            # 允许少量边界外延伸，但拒绝明显异常值。
            if (
                abs(center_x) > self.input_w * 3.0
                or abs(center_y) > self.input_h * 3.0
                or box_w > self.input_w * 4.0
                or box_h > self.input_h * 4.0
            ):
                continue

            x1 = (
                center_x - box_w * 0.5 - left
            ) / scale
            y1 = (
                center_y - box_h * 0.5 - top
            ) / scale
            x2 = (
                center_x + box_w * 0.5 - left
            ) / scale
            y2 = (
                center_y + box_h * 0.5 - top
            ) / scale

            x1 = float(np.clip(
                x1,
                0.0,
                max(0, original_w - 1),
            ))
            y1 = float(np.clip(
                y1,
                0.0,
                max(0, original_h - 1),
            ))
            x2 = float(np.clip(
                x2,
                0.0,
                original_w,
            ))
            y2 = float(np.clip(
                y2,
                0.0,
                original_h,
            ))

            mapped_w = x2 - x1
            mapped_h = y2 - y1

            if mapped_w < 2.0 or mapped_h < 2.0:
                continue

            # 几乎覆盖全图的框通常是结构或数据异常。
            if (
                mapped_w * mapped_h
                > original_w * original_h * 0.95
            ):
                continue

            detections.append({
                'id': 0,
                'class_id': 0,
                'class_name': 'QR CODE BOARD',
                'score': float(score),
                'confidence': float(score),
                'bbox': [x1, y1, x2, y2],
            })

        return detections

    def _apply_nms(
        self,
        detections: List[dict],
    ) -> List[dict]:
        if not detections:
            return []

        boxes = []
        scores = []

        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            boxes.append([
                float(x1),
                float(y1),
                float(x2 - x1),
                float(y2 - y1),
            ])
            scores.append(
                float(detection['score'])
            )

        indices = cv2.dnn.NMSBoxes(
            boxes,
            scores,
            self.score_threshold,
            self.nms_threshold,
        )

        if indices is None or len(indices) == 0:
            return []

        selected = [
            detections[int(index)]
            for index in np.asarray(
                indices
            ).reshape(-1)
        ]

        selected.sort(
            key=lambda item: item['score'],
            reverse=True,
        )

        return selected[:self.max_detections]

    def _publish_detection_message(
        self,
        image_message: Image,
        detections: List[dict],
        processing_ms: float,
    ) -> None:
        payload = {
            'header': {
                'stamp': {
                    'sec': int(
                        image_message.header.stamp.sec
                    ),
                    'nanosec': int(
                        image_message.header.stamp.nanosec
                    ),
                },
                'frame_id': str(
                    image_message.header.frame_id
                ),
            },
            'model': {
                'input_width': self.input_w,
                'input_height': self.input_h,
                'output_format': 'decoded_xywh_obj_cls',
            },
            'processing_ms': float(processing_ms),
            'detections': detections,
        }

        output_message = String()
        output_message.data = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(',', ':'),
        )
        self.detection_pub.publish(
            output_message
        )

    def _periodic_log(
        self,
        detection_count: int,
        processing_ms: float,
    ) -> None:
        now = time.monotonic()
        if now - self.last_log_time < 2.0:
            return

        self.last_log_time = now
        self.get_logger().info(
            '运行状态 | '
            f'帧={self.frame_count} | '
            f'检测={detection_count} | '
            f'耗时={processing_ms:.1f}ms | '
            f'忙碌丢帧={self.busy_drop_count} | '
            f'错误={self.error_count}'
        )

    def image_callback(
        self,
        image_message: Image,
    ) -> None:
        if not self.infer_lock.acquire(
            blocking=False
        ):
            self.busy_drop_count += 1
            return

        start_time = time.perf_counter()
        image: Optional[np.ndarray] = None

        try:
            image = self.bridge.imgmsg_to_cv2(
                image_message,
                desired_encoding='bgr8',
            )

            original_h, original_w = (
                image.shape[:2]
            )

            input_data, scale, left, top = (
                self._letterbox_to_nv12(image)
            )

            outputs = self.model.forward(
                input_data
            )

            if len(outputs) != 1:
                raise ValueError(
                    'qr.bin应只有一个output0，'
                    f'实际输出数量={len(outputs)}'
                )

            candidates = self._parse_decoded_output(
                outputs[0],
                scale,
                left,
                top,
                original_w,
                original_h,
            )

            final_detections = self._apply_nms(
                candidates
            )

            processing_ms = (
                time.perf_counter() - start_time
            ) * 1000.0

            self._publish_detection_message(
                image_message,
                final_detections,
                processing_ms,
            )


            self.frame_count += 1
            self._periodic_log(
                len(final_detections),
                processing_ms,
            )

        except Exception as exception:
            self.error_count += 1
            processing_ms = (
                time.perf_counter() - start_time
            ) * 1000.0

            self.get_logger().error(
                f'二维码YOLO处理出错：{exception}'
            )

            # 出错时发布空结果，避免下游继续使用上一帧错误框。
            if image is not None:
                self._publish_detection_message(
                    image_message,
                    [],
                    processing_ms,
                )

        finally:
            self.infer_lock.release()


def main(args=None):
    try:
        os.sched_setaffinity(0, {4})
    except (AttributeError, OSError):
        pass

    rclpy.init(args=args)
    node = QrYoloDecodedNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
