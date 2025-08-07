#!/bin/bash

# Raspberry Pi Zero 2 W用のインストールスクリプト

echo "===== Camera Cloud AI システムのインストール ====="

# パッケージリストを更新
echo "パッケージリストを更新中..."
sudo apt update

# 必要なシステムパッケージをインストール
echo "システムパッケージをインストール中..."
sudo apt install -y python3-pip python3-venv git

# カメラを有効化（既に有効になっている場合はスキップ）
echo "カメラモジュールを有効化中..."
sudo raspi-config nonint do_camera 0

# Python仮想環境を作成
echo "Python仮想環境を作成中..."
python3 -m venv ~/camera_ai_env

# 仮想環境をアクティブ化
source ~/camera_ai_env/bin/activate

# Python依存関係をインストール
echo "Python依存関係をインストール中..."
pip install --upgrade pip

# Raspberry Pi特有の依存関係
pip install picamera2
pip install requests
pip install Pillow
pip install python-dotenv

# 画像保存ディレクトリを作成
echo "画像保存ディレクトリを作成中..."
mkdir -p /home/pi/camera_images

# 設定ファイルをコピー（必要に応じて編集）
echo "設定ファイルを確認してください:"
echo "- /home/umetsu/camera_cloud_AI/raspberry_pi/config/config.json"
echo "  クラウドAPIのURLを実際のサーバーアドレスに変更してください"

# 実行権限を付与
chmod +x /home/umetsu/camera_cloud_AI/raspberry_pi/scripts/camera_capture.py

# システムサービスファイルを作成（オプション）
echo "システムサービスファイルを作成中..."
sudo tee /etc/systemd/system/camera-ai.service > /dev/null <<EOF
[Unit]
Description=Camera Cloud AI Service
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/umetsu/camera_cloud_AI/raspberry_pi/scripts
Environment=PATH=/home/pi/camera_ai_env/bin
ExecStart=/home/pi/camera_ai_env/bin/python camera_capture.py --mode continuous --interval 30
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# サービスを有効化（開始はしない）
sudo systemctl daemon-reload
sudo systemctl enable camera-ai.service

echo "===== インストール完了 ====="
echo ""
echo "次のステップ:"
echo "1. config/config.json のcloud_api_urlを実際のサーバーアドレスに変更"
echo "2. テスト実行: python3 camera_capture.py --mode single"
echo "3. サービスの開始: sudo systemctl start camera-ai.service"
echo "4. サービス状態確認: sudo systemctl status camera-ai.service"
echo ""
echo "注意: カメラモジュールが正しく接続され、有効になっていることを確認してください"