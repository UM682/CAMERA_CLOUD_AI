#!/usr/bin/env python3

import os
import time
import requests
import json
from datetime import datetime
from picamera2 import Picamera2
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CameraCloudAI:
    def __init__(self, config_path='../config/config.json'):
        self.config = self.load_config(config_path)
        self.picam2 = Picamera2()
        self.setup_camera()
        
    def load_config(self, config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def setup_camera(self):
        camera_config = self.picam2.create_still_configuration(
            main={"size": (640, 480)},
            lores={"size": (320, 240)},
            display="lores"
        )
        self.picam2.configure(camera_config)
        self.picam2.start()
        time.sleep(2)  # カメラの初期化を待つ
        
    def capture_image(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"capture_{timestamp}.jpg"
        image_path = os.path.join(self.config['local_image_path'], filename)
        
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        
        self.picam2.capture_file(image_path)
        logger.info(f"画像を撮影しました: {image_path}")
        
        return image_path
    
    def upload_to_cloud(self, image_path):
        try:
            with open(image_path, 'rb') as f:
                files = {'image': f}
                data = {
                    'timestamp': datetime.now().isoformat(),
                    'device_id': self.config['device_id']
                }
                
                response = requests.post(
                    self.config['cloud_api_url'] + '/upload',
                    files=files,
                    data=data,
                    timeout=30
                )
                
            if response.status_code == 200:
                result = response.json()
                logger.info(f"クラウド処理完了: {result}")
                return result
            else:
                logger.error(f"アップロード失敗: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"アップロード中にエラー: {e}")
            return None
    
    def process_detection_result(self, result):
        if result and 'faces_detected' in result:
            face_count = result['faces_detected']
            logger.info(f"検出された顔の数: {face_count}")
            
            if face_count > 0:
                logger.info("人が検出されました！")
                # ここで必要な処理を実行（アラート、通知など）
                self.send_notification(result)
            else:
                logger.info("人は検出されませんでした")
        
        return result
    
    def send_notification(self, detection_result):
        # 通知機能の実装（メール、Slack、LINEなど）
        logger.info("通知を送信中...")
        pass
    
    def cleanup_old_images(self):
        image_dir = self.config['local_image_path']
        max_age = self.config.get('max_image_age_hours', 24)
        
        if os.path.exists(image_dir):
            for filename in os.listdir(image_dir):
                file_path = os.path.join(image_dir, filename)
                file_age = time.time() - os.path.getctime(file_path)
                
                if file_age > max_age * 3600:
                    os.remove(file_path)
                    logger.info(f"古い画像を削除: {filename}")
    
    def run_continuous(self, interval_seconds=10):
        logger.info(f"連続撮影モードを開始（間隔: {interval_seconds}秒）")
        
        try:
            while True:
                # 画像撮影
                image_path = self.capture_image()
                
                # クラウドにアップロードして顔認識
                result = self.upload_to_cloud(image_path)
                
                # 結果処理
                self.process_detection_result(result)
                
                # 古い画像の削除
                self.cleanup_old_images()
                
                # 次の撮影まで待機
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            logger.info("撮影を停止します...")
        finally:
            self.cleanup()
    
    def run_single_shot(self):
        logger.info("1回撮影モード")
        
        # 画像撮影
        image_path = self.capture_image()
        
        # クラウドにアップロードして顔認識
        result = self.upload_to_cloud(image_path)
        
        # 結果処理
        self.process_detection_result(result)
        
        self.cleanup()
        
        return result
    
    def cleanup(self):
        self.picam2.stop()
        logger.info("カメラを停止しました")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Raspberry Pi カメラクラウドAIシステム')
    parser.add_argument('--mode', choices=['single', 'continuous'], default='single',
                      help='実行モード: single(1回撮影) または continuous(連続撮影)')
    parser.add_argument('--interval', type=int, default=10,
                      help='連続撮影モードでの撮影間隔（秒）')
    parser.add_argument('--config', default='config/config.json',
                      help='設定ファイルのパス')
    
    args = parser.parse_args()
    
    try:
        camera_ai = CameraCloudAI(args.config)
        
        if args.mode == 'continuous':
            camera_ai.run_continuous(args.interval)
        else:
            camera_ai.run_single_shot()
            
    except FileNotFoundError:
        logger.error(f"設定ファイルが見つかりません: {args.config}")
    except Exception as e:
        logger.error(f"エラーが発生しました: {e}")

if __name__ == '__main__':
    main()