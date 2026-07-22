import os
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image, CompressedImage # 🔍 新增 CompressedImage 支持
import cv2
import numpy as np
import threading
import json
from pyzbar.pyzbar import decode
import time

# 🔍 导入下相机的专用消息类型
from ai_msgs.msg import PerceptionTargets
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
import cv2
import numpy as np
import threading
import json
from pyzbar.pyzbar import decode

class YOLOQRFusionNode(Node):
    def __init__(self):
        super().__init__('depth_yolo_qr_fusion_node')

        # ==========================================
        # [配置区]
        # ==========================================
        # self.image_topic = "/aurora/rgb/image_raw"
        self.image_topic = "/aurora/ir/image_raw"
        self.yolo_json_topic = "/yolo_detections_json"
        self.pub_topic = "/qr_direction_result"

        self.qr_class_name = "QR CODE BOARD"
        self.qr_class_id = 0

        self.roi_expand_ratio = 2.0
        self.min_roi_size = 15

        # ==========================================
        # [状态变量]
        # ==========================================
        self.latest_image = None
        self.lock = threading.Lock()
        self.is_done = False

        # ==========================================
        # [安全初始化CvBridge]
        # ==========================================
        self.bridge = None
        self.use_cv_bridge = False
        try:
            from cv_bridge import CvBridge
            self.bridge = CvBridge()
            self.use_cv_bridge = True
            self.get_logger().info("✓ CvBridge加载成功")
        except Exception as e:
            self.get_logger().warn(f"⚠ CvBridge不可用: {e}，使用手动转换")

        # ==========================================
        # [通信接口]
        # ==========================================
        self.image_sub = self.create_subscription(Image, self.image_topic, self.image_cb, 5)
        self.yolo_sub = self.create_subscription(String, self.yolo_json_topic, self.yolo_cb, 5)
        self.result_pub = self.create_publisher(String, self.pub_topic, 10)

        # 与板子B共用同一个结果话题：任意一块板识别成功后，另一块立即停止继续扫码
        # 本节点自己发布结果时，self.is_done 已提前置为 True，因此不会重复处理自己的消息。
        self.shared_result_sub = self.create_subscription(
            String, self.pub_topic, self.shared_result_cb, 10
        )

        self.get_logger().info("=" * 70)
        self.get_logger().info("🎯 扫码节点已启动 [全场景适配]")
        self.get_logger().info("   - 远距离: 3.0x/4.0x 放大")
        self.get_logger().info("   - 斜角度: 透视矫正")
        self.get_logger().info("   - 模糊/低对比: CLAHE增强")
        self.get_logger().info("=" * 70)

    # ==========================================
    # [图像转换]
    # ==========================================
    def _ros_image_to_cv2(self, msg):
        """安全地转换ROS Image到OpenCV格式"""
        if self.use_cv_bridge and self.bridge:
            try:
                return self.bridge.imgmsg_to_cv2(msg, "bgr8")
            except:
                pass

        try:
            if msg.encoding == 'bgr8':
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            elif msg.encoding == 'rgb8':
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif msg.encoding == 'mono8':
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                return None
            return img
        except Exception as e:
            self.get_logger().debug(f"图像转换失败: {e}")
            return None

    # ==========================================
    # [回调函数]
    # ==========================================
    def shared_result_cb(self, msg):
        """接收板子B的最终扫码结果；不改变本节点原有识别和发布逻辑。"""
        direction = msg.data.strip()

        if direction not in ("顺时针", "逆时针"):
            return

        # 本节点已经完成时，收到的通常是自己刚发布的消息，直接忽略。
        if self.is_done:
            return

        # 板子B先识别成功：停止本节点后续图像处理和二维码解码。
        self.is_done = True
        self.get_logger().warn(
            f"🏁 已收到板子B扫码结果 [{direction}]，深度摄像头扫码节点停止继续识别"
        )

    def image_cb(self, msg):
        if self.is_done:
            return

        try:
            cv_image = self._ros_image_to_cv2(msg)
            if cv_image is None:
                return

            with self.lock:
                self.latest_image = cv_image.copy()
        except Exception as e:
            self.get_logger().debug(f"图像回调异常: {e}")

    def yolo_cb(self, msg):
        if self.is_done:
            return

        try:
            detections = json.loads(msg.data)
            qr_detections = [
                det for det in detections
                if det.get('class_name') == self.qr_class_name or det.get('id') == self.qr_class_id
            ]

            if not qr_detections:
                return

            with self.lock:
                if self.latest_image is None:
                    return
                cv_image = self.latest_image.copy()

            for det in qr_detections:
                if self._process_qr_roi(cv_image, det['bbox']):
                    return

        except Exception as e:
            self.get_logger().debug(f"YOLO回调异常: {e}")

    # ==========================================
    # [核心处理逻辑 - 全场景策略]
    # ==========================================
    def _process_qr_roi(self, image, bbox):
        """处理二维码ROI - 覆盖远/近/斜/模糊所有场景"""

        try:
            x1, y1, x2, y2 = map(int, bbox)
            orig_h, orig_w = image.shape[:2]

            roi_w, roi_h = x2 - x1, y2 - y1
            if roi_w < self.min_roi_size or roi_h < self.min_roi_size:
                return False

            # 扩展ROI
            expand_w = int(roi_w * (self.roi_expand_ratio - 1.0) / 2)
            expand_h = int(roi_h * (self.roi_expand_ratio - 1.0) / 2)

            x1_exp = max(0, x1 - expand_w)
            y1_exp = max(0, y1 - expand_h)
            x2_exp = min(orig_w, x2 + expand_w)
            y2_exp = min(orig_h, y2 + expand_h)

            roi = image[y1_exp:y2_exp, x1_exp:x2_exp]
            if roi.size == 0:
                return False

            # ==========================================
            # 【路径A】原图路径 - 适合近距离正面
            # ==========================================
            # A1: 原图直接解码
            if self._try_decode(roi, "原图"):
                return True

            # A2: 原图透视矫正
            corrected_orig = self._correct_perspective(roi)
            if corrected_orig is not None and self._try_decode(corrected_orig, "原图+矫正"):
                return True

            # ==========================================
            # 【路径B】3.0倍放大路径 - 适合中距离
            # ==========================================
            upscaled_3x = self._upscale_image(roi, 3.0)

            # B1: 3.0倍直接解码
            if self._try_decode(upscaled_3x, "3.0倍"):
                return True

            # B2: 3.0倍透视矫正
            corrected_3x = self._correct_perspective(upscaled_3x)
            if corrected_3x is not None and self._try_decode(corrected_3x, "3.0倍+矫正"):
                return True

            # B3: 3.0倍CLAHE增强
            enhanced_3x = self._apply_clahe(upscaled_3x)
            if self._try_decode(enhanced_3x, "3.0倍+CLAHE"):
                return True

            # B4: 3.0倍矫正+CLAHE
            if corrected_3x is not None:
                enhanced_corrected_3x = self._apply_clahe(corrected_3x)
                if self._try_decode(enhanced_corrected_3x, "3.0倍+矫正+CLAHE"):
                    return True

            # ==========================================
            # 【路径C】4.0倍放大路径 - 适合远距离
            # ==========================================
            upscaled_4x = self._upscale_image(roi, 4.0)

            # C1: 4.0倍直接解码
            if self._try_decode(upscaled_4x, "4.0倍"):
                return True

            # C2: 4.0倍透视矫正
            corrected_4x = self._correct_perspective(upscaled_4x)
            if corrected_4x is not None and self._try_decode(corrected_4x, "4.0倍+矫正"):
                return True

            # C3: 4.0倍CLAHE增强
            enhanced_4x = self._apply_clahe(upscaled_4x)
            if self._try_decode(enhanced_4x, "4.0倍+CLAHE"):
                return True

            # C4: 4.0倍矫正+CLAHE（终极杀手锏）
            if corrected_4x is not None:
                enhanced_corrected_4x = self._apply_clahe(corrected_4x)
                if self._try_decode(enhanced_corrected_4x, "4.0倍+矫正+CLAHE"):
                    return True

            return False

        except Exception as e:
            self.get_logger().debug(f"处理ROI异常: {e}")
            return False

    # ==========================================
    # [图像处理工具函数]
    # ==========================================
    def _upscale_image(self, img, scale):
        """高质量图像放大"""
        try:
            h, w = img.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)

            upscaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

            # 轻度锐化
            kernel = np.array([[0, -0.5, 0],
                              [-0.5, 3, -0.5],
                              [0, -0.5, 0]], dtype=np.float32)
            upscaled = cv2.filter2D(upscaled, -1, kernel)
            upscaled = np.clip(upscaled, 0, 255).astype(np.uint8)

            return upscaled
        except:
            h, w = img.shape[:2]
            return cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)

    def _correct_perspective(self, img):
        """透视矫正 - 把斜的二维码掰正"""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()

            # 二值化
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # 找轮廓
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return None

            # 找最大轮廓
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)

            # 如果轮廓太小，跳过
            if area < 100:
                return None

            # 获取最小外接矩形的四个角点
            rect = cv2.minAreaRect(largest_contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            # 计算矩形的宽高
            width = int(rect[1][0])
            height = int(rect[1][1])

            if width == 0 or height == 0:
                return None

            # 确保宽 > 高
            if width < height:
                width, height = height, width

            # 目标矩形
            dst_points = np.array([
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1]
            ], dtype=np.float32)

            # 源矩形排序
            box = box.astype(np.float32)
            box = self._order_points(box)

            # 透视变换
            matrix = cv2.getPerspectiveTransform(box, dst_points)
            corrected = cv2.warpPerspective(img, matrix, (width, height))

            return corrected

        except Exception as e:
            self.get_logger().debug(f"透视矫正失败: {e}")
            return None

    def _order_points(self, pts):
        """按左上、右上、右下、左下排序四个点"""
        rect = np.zeros((4, 2), dtype=np.float32)

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect

    def _apply_clahe(self, img):
        """CLAHE对比度增强"""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            return enhanced
        except:
            return img

    def _try_decode(self, img, method_name):
        """尝试解码"""
        try:
            decoded_objects = decode(img)
            return self._process_decoded_result(decoded_objects, method_name)
        except:
            return False

    def _process_decoded_result(self, decoded_objects, method_name):
        """处理解码结果"""
        for obj in decoded_objects:
            try:
                qr_data = obj.data.decode('utf-8').strip()
                if not qr_data.isdigit():
                    continue

                qr_number = int(qr_data)
                direction = "顺时针" if (qr_number % 2 != 0) else "逆时针"

                self.is_done = True

                msg = String()
                msg.data = direction
                self.result_pub.publish(msg)

                self.get_logger().info("=" * 70)
                self.get_logger().info("✅ 识别成功!")
                self.get_logger().info(f"   二维码数字: [{qr_number}]")
                self.get_logger().info(f"   输出指令:   [{direction}]")
                self.get_logger().info(f"   成功方法:   {method_name}")
                self.get_logger().info("=" * 70)

                return True

            except Exception as e:
                self.get_logger().debug(f"解码处理异常: {e}")
                continue

        return False

# ==========================================
# [主函数]
# ==========================================
def main(args=None):
    try:
        os.sched_setaffinity(0, {2})
    except Exception:
        pass

    rclpy.init(args=args)
    node = YOLOQRFusionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

class YOLOQRFusionNode(Node):
    def __init__(self):
        super().__init__('yolo_qr_fusion_node')

        # ==========================================
        # [配置区]
        # ==========================================
        # 上相机配置
        self.image_topic = "/aurora/rgb/image_raw"
        self.yolo_json_topic = "/yolo_detections_json"
        self.qr_class_name = "QR CODE BOARD"
        self.qr_class_id = 0

        # 🔍 下相机配置
        self.bottom_image_topic = "/image"
        self.bottom_yolo_topic = "/qr_code_detection"

        self.pub_topic = "qr_direction_result"

        # 图像处理参数（针对斜角度优化）
        self.roi_expand_ratio = 2.5      # 🔥 增大到2.5倍！
        self.min_roi_size = 15

        # ==========================================
        # [状态变量]
        # ==========================================
        self.latest_image = None
        self.latest_bottom_image = None  # 🔍 新增下相机图像缓存

        self.lock = threading.Lock()
        self.is_done = False

        # 统计信息
        self.attempt_count = 0
        self.success_method = None
        self.decode_time = 0
        self.first_attempt_time = None
        self.total_decode_time = 0

        # ==========================================
        # [安全初始化CvBridge]
        # ==========================================
        self.bridge = None
        self.use_cv_bridge = False
        try:
            from cv_bridge import CvBridge
            self.bridge = CvBridge()
            self.use_cv_bridge = True
            self.get_logger().info("✓ CvBridge加载成功")
        except Exception as e:
            self.get_logger().warn(f"⚠ CvBridge不可用: {e}，使用手动转换")

        # ==========================================
        # [通信接口]
        # ==========================================
        # 上相机订阅
        self.image_sub = self.create_subscription(Image, self.image_topic, self.image_cb, 5)
        self.yolo_sub = self.create_subscription(String, self.yolo_json_topic, self.yolo_cb, 5)

        # 🔍 下相机订阅 (假设 /image 是压缩图像，如果你用的是原生 Image，把类型改回 Image 即可)
        self.bottom_img_sub = self.create_subscription(CompressedImage, self.bottom_image_topic, self.bottom_image_cb, 5)
        self.bottom_yolo_sub = self.create_subscription(PerceptionTargets, self.bottom_yolo_topic, self.bottom_yolo_cb, 5)

        # 发布结果
        self.result_pub = self.create_publisher(String, self.pub_topic, 10)

        self.get_logger().info("=" * 70)
        self.get_logger().info("🎯 双摄融合扫码节点已启动 [高速斜角度优化]")
        self.get_logger().info("   上相机监听: /aurora/rgb/image_raw")
        self.get_logger().info("   下相机监听: /image")
        self.get_logger().info("   策略1: 3.0倍+直接解码（快速）")
        self.get_logger().info("   策略2: 3.0倍+CLAHE（抗模糊核心）⭐⭐⭐⭐⭐")
        self.get_logger().info("=" * 70)

    # ==========================================
    # [上相机回调处理]
    # ==========================================
    def _ros_image_to_cv2(self, msg):
        """安全地转换ROS Image到OpenCV格式"""
        if self.use_cv_bridge and self.bridge:
            try:
                return self.bridge.imgmsg_to_cv2(msg, "bgr8")
            except:
                pass

        try:
            if msg.encoding == 'bgr8':
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            elif msg.encoding == 'rgb8':
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif msg.encoding == 'mono8':
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                return None
            return img
        except Exception as e:
            self.get_logger().debug(f"图像转换失败: {e}")
            return None

    def image_cb(self, msg):
        if self.is_done: return
        try:
            cv_image = self._ros_image_to_cv2(msg)
            if cv_image is None: return
            with self.lock:
                self.latest_image = cv_image.copy()
        except Exception as e:
            self.get_logger().debug(f"上相机图像回调异常: {e}")

    def yolo_cb(self, msg):
        if self.is_done: return
        try:
            detections = json.loads(msg.data)
            qr_detections = [
                det for det in detections
                if det.get('class_name') == self.qr_class_name or det.get('id') == self.qr_class_id
            ]

            if not qr_detections: return
            with self.lock:
                if self.latest_image is None: return
                cv_image = self.latest_image.copy()

            for det in qr_detections:
                # 传入来源标识
                if self._process_qr_roi(cv_image, det['bbox'], source="上相机"):
                    return
        except Exception as e:
            self.get_logger().debug(f"上相机YOLO回调异常: {e}")

    # 🔍 ==========================================
    # [新增：下相机回调处理]
    # ==========================================
    def bottom_image_cb(self, msg):
        if self.is_done: return
        try:
            # 假设你的 /image 是 CompressedImage，这里使用专门的解码方法
            if self.bridge:
                cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
            else:
                # 手动解码备用方案
                np_arr = np.frombuffer(msg.data, np.uint8)
                cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if cv_image is None: return
            with self.lock:
                self.latest_bottom_image = cv_image.copy()
        except Exception as e:
            self.get_logger().debug(f"下相机图像回调异常: {e}")

    def bottom_yolo_cb(self, msg):
        if self.is_done: return
        try:
            with self.lock:
                if self.latest_bottom_image is None: return
                cv_image = self.latest_bottom_image.copy()

            # 解析 PerceptionTargets 消息
            for target in msg.targets:
                if target.type == "qr_code":
                    rect = target.rois[0].rect
                    # 转换坐标为 [x1, y1, x2, y2]
                    bbox = [
                        rect.x_offset,
                        rect.y_offset,
                        rect.x_offset + rect.width,
                        rect.y_offset + rect.height
                    ]
                    # 传入来源标识
                    if self._process_qr_roi(cv_image, bbox, source="下相机"):
                        return
        except Exception as e:
            self.get_logger().debug(f"下相机YOLO回调异常: {e}")

    # ==========================================
    # [核心处理逻辑]
    # ==========================================
    def _process_qr_roi(self, image, bbox, source="未知相机"):
        """处理二维码ROI - 增加相机来源标识"""
        if self.attempt_count == 0:
            self.first_attempt_time = time.time()

        start_time = time.time()
        self.attempt_count += 1

        try:
            x1, y1, x2, y2 = map(int, bbox)
            orig_h, orig_w = image.shape[:2]

            roi_w, roi_h = x2 - x1, y2 - y1
            if roi_w < self.min_roi_size or roi_h < self.min_roi_size:
                return False

            expand_w = int(roi_w * (self.roi_expand_ratio - 1.0) / 2)
            expand_h = int(roi_h * (self.roi_expand_ratio - 1.0) / 2)

            x1_exp = max(0, x1 - expand_w)
            y1_exp = max(0, y1 - expand_h)
            x2_exp = min(orig_w, x2 + expand_w)
            y2_exp = min(orig_h, y2 + expand_h)

            roi = image[y1_exp:y2_exp, x1_exp:x2_exp]
            if roi.size == 0:
                return False

            upscaled_3x = self._upscale_image(roi, 3.0)

            if self._try_decode_direct(upscaled_3x, "⭐3.0倍", source):
                self.decode_time = time.time() - start_time
                self.total_decode_time = time.time() - self.first_attempt_time
                return True

            if self._try_clahe(upscaled_3x, "🔥3.0倍+CLAHE", source):
                self.decode_time = time.time() - start_time
                self.total_decode_time = time.time() - self.first_attempt_time
                return True

            if self._try_sharpen(upscaled_3x, "3.0倍+锐化", source):
                self.decode_time = time.time() - start_time
                self.total_decode_time = time.time() - self.first_attempt_time
                return True

            upscaled_4x = self._upscale_image(roi, 4.0)

            if self._try_decode_direct(upscaled_4x, "4.0倍", source):
                self.decode_time = time.time() - start_time
                self.total_decode_time = time.time() - self.first_attempt_time
                return True

            if self._try_clahe(upscaled_4x, "4.0倍+CLAHE", source):
                self.decode_time = time.time() - start_time
                self.total_decode_time = time.time() - self.first_attempt_time
                return True

            return False

        except Exception as e:
            self.get_logger().debug(f"处理ROI异常: {e}")
            return False

    # ==========================================
    # [图像放大]
    # ==========================================
    def _upscale_image(self, img, scale):
        try:
            h, w = img.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            upscaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

            kernel = np.array([[0, -0.5, 0],
                              [-0.5, 3, -0.5],
                              [0, -0.5, 0]], dtype=np.float32)
            upscaled = cv2.filter2D(upscaled, -1, kernel)
            upscaled = np.clip(upscaled, 0, 255).astype(np.uint8)

            return upscaled
        except:
            h, w = img.shape[:2]
            return cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)

    # ==========================================
    # [解码函数]
    # ==========================================
    def _try_decode_direct(self, img, method_name, source):
        try:
            decoded_objects = decode(img)
            if self._process_decoded_result(decoded_objects, method_name, source):
                return True
        except: pass
        return False

    def _try_clahe(self, img, method_name, source):
        try:
            if len(img.shape) == 3: gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else: gray = img.copy()
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            decoded_objects = decode(enhanced)
            if self._process_decoded_result(decoded_objects, method_name, source): return True
        except: pass
        return False

    def _try_sharpen(self, img, method_name, source):
        try:
            if len(img.shape) == 3: gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else: gray = img.copy()
            kernel = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]], dtype=np.float32)
            sharpened = cv2.filter2D(gray, -1, kernel)
            decoded_objects = decode(sharpened)
            if self._process_decoded_result(decoded_objects, method_name, source): return True
        except: pass
        return False

    def _process_decoded_result(self, decoded_objects, method_name, source):
        for obj in decoded_objects:
            try:
                qr_data = obj.data.decode('utf-8').strip()
                if not qr_data.isdigit(): continue

                qr_number = int(qr_data)
                direction = "顺时针" if (qr_number % 2 != 0) else "逆时针"

                self.is_done = True
                self.success_method = method_name

                msg = String()
                msg.data = direction
                self.result_pub.publish(msg)

                self.get_logger().info("=" * 70)
                self.get_logger().info(f"✅ 识别成功! (立功设备: {source})")
                self.get_logger().info(f"   二维码数字: [{qr_number}]")
                self.get_logger().info(f"   输出指令:   [{direction}]")
                self.get_logger().info(f"   成功方法:   {self.success_method}")
                self.get_logger().info(f"   尝试次数:   {self.attempt_count}")
                self.get_logger().info("=" * 70)
                return True
            except Exception as e:
                continue
        return False

def main(args=None):
    try:
        os.sched_setaffinity(0, {2})
    except Exception:
        pass

    rclpy.init(args=args)
    node = YOLOQRFusionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()