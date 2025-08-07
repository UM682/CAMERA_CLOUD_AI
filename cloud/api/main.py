from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from datetime import datetime
import json
import os
import logging
from typing import Optional
import tempfile
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from face_recognition import FaceRecognizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Camera Cloud AI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 顔認識エンジンの初期化
face_recognizer = FaceRecognizer()

@app.get("/")
async def root():
    """
    API情報を表示
    """
    return {
        "service": "Camera Cloud AI API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "stats": "/stats", 
            "upload": "/upload (POST)",
            "docs": "/docs"
        },
        "description": "Raspberry Pi Zero 2W camera capture with cloud-based face recognition",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/upload")
async def upload_and_analyze(
    image: UploadFile = File(...),
    timestamp: str = Form(...),
    device_id: str = Form(...)
):
    """
    画像をアップロードして顔認識を実行
    """
    try:
        # アップロードファイルの検証
        if image.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
            raise HTTPException(status_code=400, detail="サポートされていない画像形式です")
        
        # 画像を一時ファイルに保存
        contents = await image.read()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(contents)
            tmp_file_path = tmp_file.name
        
        # OpenCVで画像を読み込み
        cv_image = cv2.imread(tmp_file_path)
        if cv_image is None:
            raise HTTPException(status_code=400, detail="画像の読み込みに失敗しました")
        
        # 顔認識を実行
        detection_result = face_recognizer.detect_faces(cv_image)
        
        # 結果をログに出力
        logger.info(f"デバイス {device_id} からの画像を処理: {detection_result}")
        
        # 一時ファイルを削除
        os.unlink(tmp_file_path)
        
        # レスポンスを作成
        response = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "device_id": device_id,
            "faces_detected": detection_result["face_count"],
            "face_locations": detection_result["face_locations"],
            "confidence_scores": detection_result["confidence_scores"],
            "processing_time": detection_result["processing_time"]
        }
        
        return response
        
    except Exception as e:
        logger.error(f"処理中にエラー: {e}")
        raise HTTPException(status_code=500, detail=f"内部サーバーエラー: {str(e)}")

@app.get("/health")
async def health_check():
    """
    ヘルスチェック
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "Camera Cloud AI API"
    }

@app.get("/stats")
async def get_stats():
    """
    処理統計を取得
    """
    return {
        "total_requests": face_recognizer.get_total_requests(),
        "total_faces_detected": face_recognizer.get_total_faces(),
        "service_uptime": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)