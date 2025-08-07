# Camera Cloud AI システム

Raspberry Pi Zero 2 Wのカメラで画像を取得し、クラウド上でAI顔認識を行うシステムです。

## 概要

このシステムは以下の構成で動作します：

1. **Raspberry Pi側**: カメラで画像を撮影し、クラウドAPIにアップロード
2. **クラウド側**: 画像を受信し、AI（OpenCV）で顔認識を実行
3. **結果返却**: 検出結果をRaspberry Piに返却

## プロジェクト構造

```
camera_cloud_AI/
├── raspberry_pi/           # Raspberry Pi用コード
│   ├── scripts/
│   │   ├── camera_capture.py    # メインの撮影・アップロードスクリプト
│   │   └── install.sh          # インストールスクリプト
│   └── config/
│       └── config.json         # 設定ファイル
├── cloud/                  # クラウドサーバー用コード
│   ├── api/
│   │   └── main.py            # FastAPI メインアプリケーション
│   ├── face_recognition/
│   │   ├── __init__.py
│   │   └── face_recognizer.py  # 顔認識エンジン
│   ├── config/
│   │   └── config.py          # サーバー設定
│   ├── Dockerfile             # Docker設定
│   └── requirements-cloud.txt # クラウド用依存関係
├── shared/                 # 共通ユーティリティ（将来拡張用）
├── docker-compose.yml      # Docker Compose設定
├── requirements.txt        # 依存関係リスト
└── README.md              # このファイル
```

## 必要な環境

### Raspberry Pi側

- Raspberry Pi Zero 2 W
- Raspberry Pi Camera Module（V1、V2、HQ Camera対応）
- Python 3.9+
- インターネット接続

### クラウド側

- Python 3.11+
- Docker & Docker Compose（推奨）
- または直接Python環境

## インストール・セットアップ

### 1. Raspberry Pi側のセットアップ

```bash
# プロジェクトをクローン
git clone <repository-url> /home/umetsu/camera_cloud_AI
cd /home/umetsu/camera_cloud_AI

# インストールスクリプトを実行
chmod +x raspberry_pi/scripts/install.sh
./raspberry_pi/scripts/install.sh
```

### 2. 設定ファイルの編集

`raspberry_pi/config/config.json` を編集：

```json
{
    "device_id": "raspberrypi_001",
    "local_image_path": "/home/pi/camera_images",
    "cloud_api_url": "http://YOUR_CLOUD_SERVER_IP:8000",
    ...
}
```

### 3. クラウドサーバーのセットアップ

#### Render（推奨）

1. GitHubにプロジェクトをプッシュ
2. [Render](https://render.com)でアカウント作成
3. 新しいWebサービスを作成し、GitHubリポジトリを接続
4. render.yamlが自動的に検出され、設定が適用されます

必要に応じて環境変数を追加設定：
- `CAMERA_AI_CONFIDENCE_THRESHOLD`: 顔認識の信頼度閾値（デフォルト: 0.5）
- その他の設定は.env.exampleを参照

#### Docker Composeを使用（ローカル開発）

```bash
# docker-compose.ymlがあるディレクトリで実行
docker-compose up -d
```

#### 直接Python環境で実行

```bash
cd cloud
pip install -r requirements-cloud.txt
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## 使用方法

### 基本的な使用方法

#### 1回撮影モード

```bash
cd /home/umetsu/camera_cloud_AI/raspberry_pi/scripts
python3 camera_capture.py --mode single
```

#### 連続撮影モード

```bash
python3 camera_capture.py --mode continuous --interval 30
```

### システムサービスとして実行

```bash
# サービスを開始
sudo systemctl start camera-ai.service

# サービス状態を確認
sudo systemctl status camera-ai.service

# サービスを停止
sudo systemctl stop camera-ai.service

# ログを確認
sudo journalctl -u camera-ai.service -f
```

## API仕様

### クラウドAPI

#### POST /upload

画像をアップロードして顔認識を実行

**リクエスト:**
- `image`: 画像ファイル（JPEG, PNG）
- `timestamp`: タイムスタンプ
- `device_id`: デバイスID

**レスポンス:**
```json
{
    \"status\": \"success\",
    \"timestamp\": \"2024-01-01T12:00:00\",
    \"device_id\": \"raspberrypi_001\",
    \"faces_detected\": 2,
    \"face_locations\": [
        {\"x\": 100, \"y\": 50, \"width\": 80, \"height\": 80}
    ],
    \"confidence_scores\": [0.95],
    \"processing_time\": 0.123
}
```

#### GET /health

ヘルスチェック

#### GET /stats

処理統計情報

## 設定項目

### Raspberry Pi設定（config.json）

| 設定項目 | 説明 | デフォルト値 |
|----------|------|--------------|
| device_id | デバイス識別子 | \"raspberrypi_001\" |
| cloud_api_url | クラウドAPIのURL | - |
| max_image_age_hours | 画像の保存期間（時間） | 24 |
| camera_settings.resolution | カメラ解像度 | 640x480 |

### クラウド設定（環境変数）

| 環境変数 | 説明 | デフォルト値 |
|----------|------|--------------|
| CAMERA_AI_HOST | APIホスト | \"0.0.0.0\" |
| CAMERA_AI_PORT | APIポート | 8000 |
| CAMERA_AI_DEBUG | デバッグモード | false |
| CAMERA_AI_CONFIDENCE_THRESHOLD | 顔認識の信頼度閾値 | 0.5 |

## トラブルシューティング

### よくある問題

1. **カメラが認識されない**
   ```bash
   # カメラの状態確認
   vcgencmd get_camera
   
   # カメラを有効化
   sudo raspi-config
   ```

2. **APIサーバーに接続できない**
   ```bash
   # サーバーの状態確認
   curl http://YOUR_SERVER_IP:8000/health
   
   # ファイアウォール設定確認
   sudo ufw status
   ```

3. **依存関係のエラー**
   ```bash
   # 仮想環境を再作成
   rm -rf ~/camera_ai_env
   python3 -m venv ~/camera_ai_env
   source ~/camera_ai_env/bin/activate
   pip install -r requirements.txt
   ```

### ログの確認

```bash
# Raspberry Pi側のログ
sudo journalctl -u camera-ai.service -f

# クラウド側のログ（Docker）
docker-compose logs -f camera-ai-api
```

## セキュリティ注意事項

- APIキーやトークンは環境変数で管理
- HTTPS通信の使用を推奨
- 不要な画像ファイルの定期的な削除
- ファイアウォール設定の適切な構成

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 貢献

バグレポートや機能要求はIssueとして報告してください。
プルリクエストも歓迎します。

## 開発・テスト

### ローカル開発環境

```bash
# 開発用依存関係をインストール
pip install -r requirements.txt
pip install pytest black flake8

# コードフォーマット
black .

# リント
flake8 .

# テスト実行
pytest
```

### パフォーマンス最適化

- Raspberry Pi Zero 2 Wは性能が限られているため、画像サイズを調整
- ネットワーク帯域を考慮した画像圧縮
- バッチ処理による効率化

## 将来の拡張予定

- [ ] 顔認識の精度向上（深層学習モデル）
- [ ] 人物識別機能
- [ ] リアルタイム動画ストリーミング
- [ ] モバイルアプリ連携
- [ ] データベース連携
- [ ] 複数デバイス対応