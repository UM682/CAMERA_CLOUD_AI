import cv2
import numpy as np
import time
import logging
from typing import Dict, List, Tuple
import os
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

logger = logging.getLogger(__name__)

class FaceRecognizer:
    def __init__(self):
        self.face_cascade = None
        self.mp_face_detection = None
        self.total_requests = 0
        self.total_faces_detected = 0
        self.load_models()
    
    def load_models(self):
        """
        顔検出モデルを読み込み（MediaPipe優先、フォールバックでOpenCV）
        """
        try:
            # MediaPipeが利用可能な場合は優先使用
            if MEDIAPIPE_AVAILABLE:
                mp_face_detection = mp.solutions.face_detection
                self.mp_face_detection = mp_face_detection.FaceDetection(
                    model_selection=0, min_detection_confidence=0.5
                )
                logger.info("MediaPipe顔検出モデルの読み込みが完了しました")
            
            # OpenCVのHaar Cascade分類器もロード（フォールバック用）
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            if self.face_cascade.empty():
                logger.warning("Haar Cascade分類器の読み込みに失敗しました")
            else:
                logger.info("OpenCV Haar Cascade分類器の読み込みが完了しました")
            
        except Exception as e:
            logger.error(f"モデル読み込みエラー: {e}")
            raise
    
    def detect_faces(self, image: np.ndarray) -> Dict:
        """
        画像から顔を検出（MediaPipe優先、フォールバックでHaar Cascade）
        
        Args:
            image: OpenCVで読み込んだ画像 (BGR format)
            
        Returns:
            検出結果の辞書
        """
        # MediaPipeが利用可能な場合は優先使用
        if self.mp_face_detection:
            return self._detect_faces_mediapipe(image)
        else:
            return self._detect_faces_opencv(image)
    
    def _detect_faces_mediapipe(self, image: np.ndarray) -> Dict:
        """MediaPipeを使用した顔検出"""
        start_time = time.time()
        
        try:
            # BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.mp_face_detection.process(rgb_image)
            
            processing_time = time.time() - start_time
            
            # 統計を更新
            self.total_requests += 1
            
            face_locations = []
            confidence_scores = []
            
            if results.detections:
                h, w = image.shape[:2]
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)
                    
                    face_locations.append({
                        "x": x,
                        "y": y,
                        "width": width,
                        "height": height
                    })
                    confidence_scores.append(round(detection.score[0], 2))
            
            self.total_faces_detected += len(face_locations)
            
            result = {
                "face_count": len(face_locations),
                "face_locations": face_locations,
                "confidence_scores": confidence_scores,
                "processing_time": round(processing_time, 3),
                "method": "MediaPipe",
                "image_size": {
                    "width": image.shape[1],
                    "height": image.shape[0]
                }
            }
            
            logger.info(f"MediaPipe顔検出完了: {len(face_locations)}個の顔を検出")
            return result
            
        except Exception as e:
            logger.error(f"MediaPipe顔検出エラー: {e}")
            return self._detect_faces_opencv(image)
    
    def _detect_faces_opencv(self, image: np.ndarray) -> Dict:
        """OpenCV Haar Cascadeを使用した顔検出"""
        start_time = time.time()
        
        try:
            # グレースケールに変換
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # ヒストグラム平坦化で画像を改善
            gray = cv2.equalizeHist(gray)
            
            # 顔検出を実行
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,        # 画像ピラミッドのスケーリング係数
                minNeighbors=5,         # 各候補矩形が持つべき近傍矩形の最小数
                minSize=(30, 30),       # 検出対象の最小サイズ
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            processing_time = time.time() - start_time
            
            # 統計を更新
            self.total_requests += 1
            self.total_faces_detected += len(faces)
            
            # 結果を整理
            face_locations = []
            confidence_scores = []
            
            for (x, y, w, h) in faces:
                face_locations.append({
                    "x": int(x),
                    "y": int(y),
                    "width": int(w),
                    "height": int(h)
                })
                # Haar Cascadeは信頼度スコアを直接提供しないため、
                # サイズベースの簡易スコアを計算
                area = w * h
                confidence = min(1.0, area / (100 * 100))  # 正規化
                confidence_scores.append(round(confidence, 2))
            
            result = {
                "face_count": len(faces),
                "face_locations": face_locations,
                "confidence_scores": confidence_scores,
                "processing_time": round(processing_time, 3),
                "image_size": {
                    "width": image.shape[1],
                    "height": image.shape[0]
                }
            }
            
            logger.info(f"顔検出完了: {len(faces)}個の顔を検出 (処理時間: {processing_time:.3f}秒)")
            
            return result
            
        except Exception as e:
            logger.error(f"顔検出エラー: {e}")
            return {
                "face_count": 0,
                "face_locations": [],
                "confidence_scores": [],
                "processing_time": 0,
                "error": str(e)
            }
    
    def detect_faces_with_dnn(self, image: np.ndarray, confidence_threshold: float = 0.5) -> Dict:
        """
        DNNを使用した高精度な顔検出（オプション）
        
        Args:
            image: OpenCVで読み込んだ画像
            confidence_threshold: 信頼度の閾値
            
        Returns:
            検出結果の辞書
        """
        start_time = time.time()
        
        try:
            # DNNモデルのパス（事前にダウンロードが必要）
            model_path = "models/opencv_face_detector_uint8.pb"
            config_path = "models/opencv_face_detector.pbtxt"
            
            if not (os.path.exists(model_path) and os.path.exists(config_path)):
                logger.warning("DNNモデルが見つかりません。Haar Cascadeにフォールバック")
                return self.detect_faces(image)
            
            # DNNネットワークを読み込み
            net = cv2.dnn.readNetFromTensorflow(model_path, config_path)
            
            # 画像を前処理
            h, w = image.shape[:2]
            blob = cv2.dnn.blobFromImage(
                image, 1.0, (300, 300), 
                [104, 117, 123], False, False
            )
            
            # 推論を実行
            net.setInput(blob)
            detections = net.forward()
            
            processing_time = time.time() - start_time
            
            # 結果を処理
            face_locations = []
            confidence_scores = []
            
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                
                if confidence > confidence_threshold:
                    x1 = int(detections[0, 0, i, 3] * w)
                    y1 = int(detections[0, 0, i, 4] * h)
                    x2 = int(detections[0, 0, i, 5] * w)
                    y2 = int(detections[0, 0, i, 6] * h)
                    
                    face_locations.append({
                        "x": x1,
                        "y": y1,
                        "width": x2 - x1,
                        "height": y2 - y1
                    })
                    confidence_scores.append(round(float(confidence), 2))
            
            # 統計を更新
            self.total_requests += 1
            self.total_faces_detected += len(face_locations)
            
            result = {
                "face_count": len(face_locations),
                "face_locations": face_locations,
                "confidence_scores": confidence_scores,
                "processing_time": round(processing_time, 3),
                "method": "DNN",
                "image_size": {
                    "width": w,
                    "height": h
                }
            }
            
            logger.info(f"DNN顔検出完了: {len(face_locations)}個の顔を検出")
            
            return result
            
        except Exception as e:
            logger.error(f"DNN顔検出エラー: {e}")
            # フォールバックとしてHaar Cascadeを使用
            return self.detect_faces(image)
    
    def annotate_image(self, image: np.ndarray, detection_result: Dict) -> np.ndarray:
        """
        検出結果を画像に描画
        
        Args:
            image: 元の画像
            detection_result: detect_facesの結果
            
        Returns:
            注釈付きの画像
        """
        annotated = image.copy()
        
        for i, face_loc in enumerate(detection_result["face_locations"]):
            x, y, w, h = face_loc["x"], face_loc["y"], face_loc["width"], face_loc["height"]
            
            # 矩形を描画
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # 信頼度スコアを表示
            if i < len(detection_result["confidence_scores"]):
                confidence = detection_result["confidence_scores"][i]
                label = f"Face: {confidence}"
                cv2.putText(annotated, label, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return annotated
    
    def get_total_requests(self) -> int:
        """総リクエスト数を取得"""
        return self.total_requests
    
    def get_total_faces(self) -> int:
        """検出した総顔数を取得"""
        return self.total_faces_detected