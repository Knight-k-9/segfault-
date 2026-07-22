#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import threading
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data

# 只保留核心消息格式
from ai_msgs.msg import PerceptionTargets, Target, Point, Roi
from geometry_msgs.msg import Point32
from hobot_dnn import pyeasy_dnn as dnn

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

class TurboDetectorNode(Node):
    def __init__(self):
        super().__init__('turbo_detector_node')

        self.declare_parameter('bpu_model_path', '/root/workspace/ros2_work/src/yolov5/models/yolov5_best_x5.bin')
        model_path = self.get_parameter('bpu_model_path').value

        self.yolo5_config = {
            'strides': [8, 16, 32],
            'anchors_table': np.array([
                [[10, 13], [16, 30], [33, 23]],
                [[30, 61], [62, 45], [59, 119]],
                [[116, 90], [156, 198], [373, 326]]
            ], dtype=np.float32),
            'class_num': 4,
            'score_threshold': 0.35,
            'nms_threshold': 0.45
        }

        self.models = dnn.load(model_path)
        self.model = self.models[0]
        self.input_h = self.model.inputs[0].properties.shape[2]
        self.input_w = self.model.inputs[0].properties.shape[3]

        self.bridge = CvBridge()
        self.infer_sem = threading.Semaphore(1)

        self.image_sub = self.create_subscription(
            CompressedImage, 
            '/image', 
            self.image_callback, 
            qos_profile_sensor_data
        )

        # 【重新梳理的 4 个独立发布者】
        # ID 1 -> 车道线
        self.line_pub = self.create_publisher(PerceptionTargets, "racing_track_center_detection", 10)
        # ID 3 -> 锥桶 (严格确保！)
        self.obs_pub = self.create_publisher(PerceptionTargets, "racing_obstacle_detection", 10)
        # ID 0 -> 二维码
        self.qr_pub = self.create_publisher(PerceptionTargets, "qr_code_detection", 10)
        # ID 2 -> 停车P点
        self.parking_pub = self.create_publisher(PerceptionTargets, "parking_p_detection", 10)

        self.get_logger().info("🚀 极速压缩图模式已启动 (实测标签修正版)")

    def bgr_to_nv12_fast(self, bgr_img):
        h, w = bgr_img.shape[:2]
        yuv_i420 = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YUV_I420)
        y_plane = yuv_i420[:h, :].flatten()
        uv_interleaved = np.empty(h * w // 2, dtype=np.uint8)
        u_plane = yuv_i420[h : h + h//4, :].flatten()
        v_plane = yuv_i420[h + h//4 :, :].flatten()
        uv_interleaved[0::2] = u_plane
        uv_interleaved[1::2] = v_plane
        return np.concatenate([y_plane, uv_interleaved])

    def get_nv12_letterbox(self, img):
        h, w = img.shape[:2]
        scale = min(self.input_h / h, self.input_w / w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized_img = cv2.resize(img, (new_w, new_h))
        top, left = (self.input_h - new_h) // 2, (self.input_w - new_w) // 2
        letterbox_img = cv2.copyMakeBorder(resized_img, top, self.input_h-new_h-top, left, self.input_w-new_w-left,
                                          cv2.BORDER_CONSTANT, value=(127, 127, 127))
        return self.bgr_to_nv12_fast(letterbox_img), scale, left, top

    def parse_tensors(self, dnn_outputs, scale, left, top, orig_h, orig_w):
        all_results = []
        conf = self.yolo5_config
        logits_threshold = -np.log(1.0 / conf['score_threshold'] - 1.0)
        for tensor in dnn_outputs:
            h_feat, w_feat = tensor.properties.shape[1:3]
            stride = self.input_h // h_feat
            anchors = conf['anchors_table'][[8, 16, 32].index(stride)]
            data = np.array(tensor.buffer).reshape(h_feat, w_feat, 3, conf['class_num'] + 5)
            mask = data[..., 4] > logits_threshold
            if not np.any(mask): continue
            f_h, f_w, f_a = np.where(mask)
            valid_data, curr_anchors = data[mask], anchors[f_a]
            obj_scores, cls_probs = sigmoid(valid_data[:, 4]), sigmoid(valid_data[:, 5:])
            f_s = obj_scores * np.max(cls_probs, axis=1)
            score_mask = f_s > conf['score_threshold']
            if not np.any(score_mask): continue

            f_data, f_v_s, f_h_idx, f_w_idx, f_anc, f_cls = \
                valid_data[score_mask], f_s[score_mask], f_h[score_mask], f_w[score_mask], curr_anchors[score_mask], np.argmax(cls_probs[score_mask], axis=1)

            cx = (sigmoid(f_data[:, 0])*2 - 0.5 + f_w_idx) * stride
            cy = (sigmoid(f_data[:, 1])*2 - 0.5 + f_h_idx) * stride
            cw, ch = (sigmoid(f_data[:, 2])*2)**2 * f_anc[:, 0], (sigmoid(f_data[:, 3])*2)**2 * f_anc[:, 1]

            for i in range(len(f_v_s)):
                x1, y1 = (cx[i]-cw[i]/2-left)/scale, (cy[i]-ch[i]/2-top)/scale
                x2, y2 = (cx[i]+cw[i]/2-left)/scale, (cy[i]+ch[i]/2-top)/scale
                all_results.append({
                    'id': int(f_cls[i]),
                    'score': float(f_v_s[i]),
                    'bbox': [max(0, x1), max(0, y1), min(orig_w, x2), min(orig_h, y2)]
                })
        return all_results

    def image_callback(self, msg):
        self.get_logger().info("📷 成功接收到相机图像帧！", throttle_duration_sec=2.0)
        
        if not self.infer_sem.acquire(blocking=False): return
        try:
            cv_img = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
            
            orig_h, orig_w = cv_img.shape[:2]
            input_data, scale, left, top = self.get_nv12_letterbox(cv_img)
            outputs = self.model.forward(input_data)
            raw_detections = self.parse_tensors(outputs, scale, left, top, orig_h, orig_w)

            # NMS
            final_res = []
            if raw_detections:
                boxes = [[d['bbox'][0], d['bbox'][1], d['bbox'][2]-d['bbox'][0], d['bbox'][3]-d['bbox'][1]] for d in raw_detections]
                scores = [d['score'] for d in raw_detections]
                indices = cv2.dnn.NMSBoxes(boxes, scores, self.yolo5_config['score_threshold'], self.yolo5_config['nms_threshold'])
                if len(indices) > 0:
                    for i in np.array(indices).flatten():
                        final_res.append(raw_detections[i])

            if len(final_res) > 0:
                detected_ids = [d['id'] for d in final_res]
                self.get_logger().info(f"🎯 识别到目标 ID 列表: {detected_ids}", throttle_duration_sec=0.5)

            # --- 初始化 4 个独立的消息体 ---
            line_msg = PerceptionTargets()
            obs_msg = PerceptionTargets()
            qr_msg = PerceptionTargets()
            parking_msg = PerceptionTargets()
            
            line_msg.header = obs_msg.header = qr_msg.header = parking_msg.header = msg.header
            
            best_line = None

            for det in final_res:
                det_id = det['id']
                
                if det_id == 1: # 【ID 1: 车道线】
                    if best_line is None or det['score'] > best_line['score']:
                        best_line = det
                        
                elif det_id == 3: # 【ID 3: 锥桶】 -> 绑定到 obs_pub
                    t = Target()
                    t.type = "construction_cone"
                    roi = Roi()
                    roi.rect.x_offset, roi.rect.y_offset = int(det['bbox'][0]), int(det['bbox'][1])
                    roi.rect.width, roi.rect.height = int(det['bbox'][2]-det['bbox'][0]), int(det['bbox'][3]-det['bbox'][1])
                    roi.confidence = det['score']
                    t.rois.append(roi)
                    obs_msg.targets.append(t)
                    
                elif det_id == 0: # 【ID 0: 二维码】 -> 绑定到 qr_pub
                    t = Target()
                    t.type = "qr_code" 
                    roi = Roi()
                    roi.rect.x_offset, roi.rect.y_offset = int(det['bbox'][0]), int(det['bbox'][1])
                    roi.rect.width, roi.rect.height = int(det['bbox'][2]-det['bbox'][0]), int(det['bbox'][3]-det['bbox'][1])
                    roi.confidence = det['score']
                    t.rois.append(roi)
                    qr_msg.targets.append(t)

                elif det_id == 2: # 【ID 2: 停车P点】 -> 绑定到 parking_pub
                    t = Target()
                    t.type = "parking_p" 
                    roi = Roi()
                    roi.rect.x_offset, roi.rect.y_offset = int(det['bbox'][0]), int(det['bbox'][1])
                    roi.rect.width, roi.rect.height = int(det['bbox'][2]-det['bbox'][0]), int(det['bbox'][3]-det['bbox'][1])
                    roi.confidence = det['score']
                    t.rois.append(roi)
                    parking_msg.targets.append(t)

            # --- 封装车道线 ---
            if best_line:
                x1, y1, x2, y2 = best_line['bbox']
                t = Target()
                t.type = "line"
                p = Point()
                pt = Point32()
                pt.x, pt.y = float((x1 + x2) / 2.0), float(y2)
                p.point.append(pt)
                t.points.append(p)
                line_msg.targets.append(t)

            # --- 统一发布 4 个话题 ---
            self.line_pub.publish(line_msg)
            self.obs_pub.publish(obs_msg)
            self.qr_pub.publish(qr_msg)
            self.parking_pub.publish(parking_msg)

        except Exception as e:
            self.get_logger().error(f"❌ 推理过程中发生崩溃: {str(e)}")
        finally:
            self.infer_sem.release()

def main(args=None):
    rclpy.init(args=args)
    node = TurboDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()