#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import threading
import traceback

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from hobot_dnn import pyeasy_dnn as dnn
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
    QoSDurabilityPolicy,
)
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import String


class BPUYOLONode(Node):
    """BPU YOLO二维码板检测节点。

    优化重点：
    1. ROS回调只保存最新帧，推理放到独立线程，主动丢弃过期帧；
    2. 使用SensorData风格QoS，避免可靠传输导致排队；
    3. 模型输出尺寸动态计算，不再写死25200；
    4. NMS前限制候选数；
    5. 调试图像可降频或关闭，避免JPEG压缩拖慢检测；
    6. JSON携带原始图像时间戳，供二维码节点严格匹配同一帧。
    """

    def __init__(self):
        super().__init__('bpu_yolo_node')

        # ---------------- 参数 ----------------
        self.declare_parameter('bpu_model_path', '/root/task/yolov5/models/QR.bin')
        self.declare_parameter('image_topic', '/aurora/rgb/image_raw')
        self.declare_parameter('score_threshold', 0.35)
        self.declare_parameter('nms_threshold', 0.45)
        self.declare_parameter('max_detections', 3)
        self.declare_parameter('pre_nms_topk', 300)
        self.declare_parameter('publish_debug_image', True)
        self.declare_parameter('debug_image_every_n', 2)
        self.declare_parameter('jpeg_quality', 40)
        self.declare_parameter('publish_legacy_json', True)

        model_path = self.get_parameter('bpu_model_path').value
        image_topic = self.get_parameter('image_topic').value

        self.score_threshold = float(self.get_parameter('score_threshold').value)
        self.nms_threshold = float(self.get_parameter('nms_threshold').value)
        self.max_detections = max(1, int(self.get_parameter('max_detections').value))
        self.pre_nms_topk = max(20, int(self.get_parameter('pre_nms_topk').value))
        self.publish_debug_image = bool(self.get_parameter('publish_debug_image').value)
        self.debug_image_every_n = max(1, int(self.get_parameter('debug_image_every_n').value))
        self.jpeg_quality = int(np.clip(self.get_parameter('jpeg_quality').value, 20, 95))
        self.publish_legacy_json = bool(self.get_parameter('publish_legacy_json').value)

        self.class_names = ['QR CODE BOARD']
        self.colors = [(0, 255, 0)]
        self.class_num = len(self.class_names)

        # 当前进程已经固定到单核时，关闭OpenCV内部多线程可减少线程调度开销。
        try:
            cv2.setNumThreads(1)
        except Exception:
            pass

        # ---------------- 模型 ----------------
        self.get_logger().info(f'正在加载 BPU 模型: {model_path}')
        self.models = dnn.load(model_path)
        if not self.models:
            raise RuntimeError(f'模型加载结果为空: {model_path}')
        self.model = self.models[0]

        input_shape = self.model.inputs[0].properties.shape
        if len(input_shape) < 4:
            raise RuntimeError(f'不支持的模型输入尺寸: {input_shape}')
        self.input_h = int(input_shape[2])
        self.input_w = int(input_shape[3])
        self.get_logger().info(
            f'BPU 模型加载成功，输入尺寸: {self.input_w}x{self.input_h}'
        )

        # ---------------- ROS通信 ----------------
        self.bridge = CvBridge()

        sensor_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        result_qos = QoSProfile(depth=1)

        self.image_sub = self.create_subscription(
            Image, image_topic, self.image_callback, sensor_qos
        )
        self.detection_pub = self.create_publisher(
            String, '/racing_obstacle_detection', result_qos
        )
        # 旧话题继续发布原来的list格式，避免破坏已有订阅节点。
        self.json_detection_pub = self.create_publisher(
            String, '/yolo_detections_json', result_qos
        )
        # 新话题包含图像时间戳，二维码节点用它匹配同一帧。
        self.stamped_json_detection_pub = self.create_publisher(
            String, '/yolo_detections_stamped_json', result_qos
        )
        self.compressed_pub = self.create_publisher(
            CompressedImage, '/racing_result_image/compressed', sensor_qos
        )

        # ---------------- 最新帧工作线程 ----------------
        self._frame_lock = threading.Lock()
        self._frame_event = threading.Event()
        self._stop_event = threading.Event()
        self._latest_msg = None
        self._received_frames = 0
        self._processed_frames = 0
        self._last_finish_time = None
        self._output_hz_ema = 0.0

        self._worker = threading.Thread(
            target=self._inference_worker,
            name='bpu_yolo_worker',
            daemon=True,
        )
        self._worker.start()

        self.get_logger().info(
            '节点启动完成。提示：相机输入为5 FPS时，实际检测发布频率上限也是5 FPS。'
        )

    @staticmethod
    def _stamp_to_ns(msg: Image) -> int:
        return int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)

    def image_callback(self, msg: Image):
        """只覆盖保存最新帧，避免旧帧排队。"""
        if self._stop_event.is_set():
            return

        with self._frame_lock:
            self._latest_msg = msg
            self._received_frames += 1
            self._frame_event.set()

    def _take_latest_frame(self):
        self._frame_event.wait(timeout=0.1)
        if self._stop_event.is_set():
            return None

        with self._frame_lock:
            msg = self._latest_msg
            self._latest_msg = None
            self._frame_event.clear()
        return msg

    def _inference_worker(self):
        while not self._stop_event.is_set():
            msg = self._take_latest_frame()
            if msg is None:
                continue

            try:
                self._process_frame(msg)
            except Exception as exc:
                self.get_logger().error(f'处理过程中发生错误: {exc}')
                self.get_logger().error(traceback.format_exc())

    def bgr_to_nv12_fast(self, bgr_img: np.ndarray) -> np.ndarray:
        """BGR -> I420 -> NV12，使用单个目标缓冲区减少一次拼接分配。"""
        h, w = bgr_img.shape[:2]
        if (h & 1) or (w & 1):
            raise ValueError(f'NV12要求宽高为偶数，当前为 {w}x{h}')

        i420 = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YUV_I420).reshape(-1)
        y_size = h * w
        uv_plane_size = y_size // 4

        nv12 = np.empty(y_size * 3 // 2, dtype=np.uint8)
        nv12[:y_size] = i420[:y_size]

        u_plane = i420[y_size:y_size + uv_plane_size]
        v_plane = i420[y_size + uv_plane_size:y_size + 2 * uv_plane_size]
        uv = nv12[y_size:]
        uv[0::2] = u_plane
        uv[1::2] = v_plane
        return nv12

    def get_nv12_letterbox(self, img: np.ndarray):
        """等比例缩放并补边。"""
        orig_h, orig_w = img.shape[:2]
        scale = min(self.input_w / orig_w, self.input_h / orig_h)

        new_w = max(2, int(round(orig_w * scale)))
        new_h = max(2, int(round(orig_h * scale)))
        new_w = min(new_w, self.input_w)
        new_h = min(new_h, self.input_h)

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        left = (self.input_w - new_w) // 2
        top = (self.input_h - new_h) // 2

        canvas = np.full((self.input_h, self.input_w, 3), 114, dtype=np.uint8)
        canvas[top:top + new_h, left:left + new_w] = resized
        scale_x = new_w / float(orig_w)
        scale_y = new_h / float(orig_h)
        return self.bgr_to_nv12_fast(canvas), scale_x, scale_y, left, top

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        # 防止极端logit导致exp溢出。
        return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))

    def parse_tensors(self, dnn_outputs, scale_x, scale_y, left, top, orig_h, orig_w):
        if not dnn_outputs:
            return []

        raw = np.asarray(dnn_outputs[0].buffer)
        columns = 5 + self.class_num
        if raw.size % columns != 0:
            raise ValueError(
                f'模型输出元素数 {raw.size} 无法按每行 {columns} 个字段解析；'
                f'请检查模型类别数或输出格式。原始shape={raw.shape}'
            )

        data = raw.reshape(-1, columns)
        cx = data[:, 0]
        cy = data[:, 1]
        box_w = data[:, 2]
        box_h = data[:, 3]
        obj_conf = data[:, 4]
        cls_probs = data[:, 5:]

        # 只做一次min/max判断。
        obj_min = float(np.min(obj_conf))
        obj_max = float(np.max(obj_conf))
        if obj_min < 0.0 or obj_max > 1.0:
            obj_conf = self._sigmoid(obj_conf)
            cls_probs = self._sigmoid(cls_probs)

        if self.class_num == 1:
            cls_ids = np.zeros(data.shape[0], dtype=np.int32)
            cls_max_probs = cls_probs[:, 0]
        else:
            cls_ids = np.argmax(cls_probs, axis=1).astype(np.int32)
            cls_max_probs = cls_probs[np.arange(cls_probs.shape[0]), cls_ids]

        final_scores = obj_conf * cls_max_probs
        valid_indices = np.flatnonzero(final_scores >= self.score_threshold)
        if valid_indices.size == 0:
            return []

        # NMS前只保留高分候选，避免低阈值时产生大量Python对象。
        if valid_indices.size > self.pre_nms_topk:
            local_scores = final_scores[valid_indices]
            top_local = np.argpartition(
                local_scores, -self.pre_nms_topk
            )[-self.pre_nms_topk:]
            valid_indices = valid_indices[top_local]

        cx_f = cx[valid_indices].astype(np.float32, copy=True)
        cy_f = cy[valid_indices].astype(np.float32, copy=True)
        w_f = box_w[valid_indices].astype(np.float32, copy=True)
        h_f = box_h[valid_indices].astype(np.float32, copy=True)
        scores_f = final_scores[valid_indices]
        cls_ids_f = cls_ids[valid_indices]

        # 兼容0~1归一化坐标。
        if max(float(np.max(cx_f)), float(np.max(cy_f))) <= 1.5:
            cx_f *= self.input_w
            cy_f *= self.input_h
            w_f *= self.input_w
            h_f *= self.input_h

        x1 = np.clip((cx_f - w_f * 0.5 - left) / scale_x, 0, orig_w - 1)
        y1 = np.clip((cy_f - h_f * 0.5 - top) / scale_y, 0, orig_h - 1)
        x2 = np.clip((cx_f + w_f * 0.5 - left) / scale_x, 0, orig_w - 1)
        y2 = np.clip((cy_f + h_f * 0.5 - top) / scale_y, 0, orig_h - 1)

        valid_box = (x2 > x1 + 1) & (y2 > y1 + 1)
        results = []
        for i in np.flatnonzero(valid_box):
            results.append({
                'id': int(cls_ids_f[i]),
                'score': float(scores_f[i]),
                'bbox': [
                    float(x1[i]), float(y1[i]),
                    float(x2[i]), float(y2[i]),
                ],
            })
        return results

    def _run_nms(self, detections):
        if not detections:
            return []

        boxes = []
        scores = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            boxes.append([x1, y1, x2 - x1, y2 - y1])
            scores.append(det['score'])

        indices = cv2.dnn.NMSBoxes(
            boxes,
            scores,
            self.score_threshold,
            self.nms_threshold,
        )
        if indices is None or len(indices) == 0:
            return []

        kept = [detections[int(i)] for i in np.asarray(indices).reshape(-1)]
        kept.sort(key=lambda item: item['score'], reverse=True)
        return kept[:self.max_detections]

    def _publish_json(self, msg: Image, orig_w: int, orig_h: int, detections):
        detection_list = [
            {
                'id': int(det['id']),
                'class_name': self.class_names[int(det['id'])],
                'score': float(det['score']),
                'bbox': [float(v) for v in det['bbox']],
            }
            for det in detections
        ]

        # 保留旧格式，其他已有节点无需修改。
        if self.publish_legacy_json:
            legacy = String()
            legacy.data = json.dumps(
                detection_list, ensure_ascii=False, separators=(',', ':')
            )
            self.json_detection_pub.publish(legacy)

        payload = {
            'stamp_ns': self._stamp_to_ns(msg),
            'frame_id': msg.header.frame_id,
            'image_width': orig_w,
            'image_height': orig_h,
            'detections': detection_list,
        }
        stamped = String()
        stamped.data = json.dumps(payload, ensure_ascii=False, separators=(',', ':'))
        self.stamped_json_detection_pub.publish(stamped)

    def _publish_debug(self, msg: Image, cv_img: np.ndarray, detections, process_fps: float):
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            cls_id = int(det['id'])
            color = self.colors[cls_id % len(self.colors)]
            cv2.rectangle(cv_img, (x1, y1), (x2, y2), color, 2)
            label = f'{self.class_names[cls_id]}: {det["score"]:.2f}'
            text_y = max(18, y1 - 8)
            cv2.putText(
                cv_img,
                label,
                (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )

        cv2.putText(
            cv_img,
            f'PROC: {process_fps:.1f} FPS  OUT: {self._output_hz_ema:.1f} Hz',
            (15, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        ok, encoded = cv2.imencode(
            '.jpg',
            cv_img,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            return

        compressed_msg = CompressedImage()
        compressed_msg.header = msg.header
        compressed_msg.format = 'jpeg'
        compressed_msg.data = encoded.tobytes()
        self.compressed_pub.publish(compressed_msg)

    def _process_frame(self, msg: Image):
        start = time.perf_counter()

        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        if cv_img is None or cv_img.size == 0:
            return
        orig_h, orig_w = cv_img.shape[:2]

        input_data, scale_x, scale_y, left, top = self.get_nv12_letterbox(cv_img)
        outputs = self.model.forward(input_data)
        raw = self.parse_tensors(
            outputs, scale_x, scale_y, left, top, orig_h, orig_w
        )
        final_res = self._run_nms(raw)

        elapsed = max(time.perf_counter() - start, 1e-6)
        process_fps = 1.0 / elapsed
        now = time.perf_counter()
        if self._last_finish_time is not None:
            instantaneous_hz = 1.0 / max(now - self._last_finish_time, 1e-6)
            if self._output_hz_ema <= 0.0:
                self._output_hz_ema = instantaneous_hz
            else:
                self._output_hz_ema = 0.85 * self._output_hz_ema + 0.15 * instantaneous_hz
        self._last_finish_time = now
        self._processed_frames += 1

        # JSON始终发布，空检测也携带对应帧时间戳，方便下游判断。
        self._publish_json(msg, orig_w, orig_h, final_res)

        status = String()
        status.data = (
            f'proc_fps:{process_fps:.1f}, output_hz:{self._output_hz_ema:.1f}, '
            f'count:{len(final_res)}, process_ms:{elapsed * 1000.0:.1f}'
        )
        self.detection_pub.publish(status)

        # JPEG绘制与编码是明显CPU耗时项，可以关闭或降频。
        if (
            self.publish_debug_image
            and self._processed_frames % self.debug_image_every_n == 0
        ):
            self._publish_debug(msg, cv_img, final_res, process_fps)

    def destroy_node(self):
        self._stop_event.set()
        self._frame_event.set()
        if hasattr(self, '_worker') and self._worker.is_alive():
            self._worker.join(timeout=1.0)
        return super().destroy_node()


def main(args=None):
    try:
        os.sched_setaffinity(0, {3})
        print('Check: bpu_yolo_node tied to Core 3')
    except Exception as exc:
        print(f'CPU affinity skipped: {exc}')

    rclpy.init(args=args)
    node = BPUYOLONode()
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