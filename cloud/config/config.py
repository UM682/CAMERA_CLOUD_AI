import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # API設定
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    
    # セキュリティ設定
    api_key: str = ""
    allowed_origins: list = ["*"]
    
    # 画像処理設定
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    supported_formats: list = ["image/jpeg", "image/png", "image/jpg"]
    
    # 顔認識設定
    confidence_threshold: float = 0.5
    use_dnn_model: bool = False
    
    # ログ設定
    log_level: str = "INFO"
    log_file: str = "camera_ai.log"
    
    # データベース設定（将来的な拡張用）
    database_url: str = ""
    
    # ストレージ設定
    temp_dir: str = "/tmp"
    
    class Config:
        env_file = ".env"
        env_prefix = "CAMERA_AI_"

# グローバル設定インスタンス
settings = Settings()