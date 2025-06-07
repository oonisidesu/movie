#!/bin/bash

echo "動画顔変換ツール - セットアップスクリプト"
echo "========================================"

# Python version check
echo -n "Pythonバージョンチェック... "
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    echo "OK (Python $python_version)"
else
    echo "エラー: Python 3.8以上が必要です"
    exit 1
fi

# Create virtual environment
echo -n "仮想環境を作成中... "
python3 -m venv venv
echo "完了"

# Activate virtual environment
echo "仮想環境を有効化中..."
source venv/bin/activate

# Upgrade pip
echo "pipをアップグレード中..."
pip install --upgrade pip

# Install dependencies
echo "依存関係をインストール中..."
pip install -r requirements.txt

# Create necessary directories
echo "必要なディレクトリを作成中..."
mkdir -p output temp faces target_faces test_videos logs cache

# Create .gitkeep files
touch faces/.gitkeep
touch target_faces/.gitkeep
touch test_videos/.gitkeep

echo ""
echo "セットアップ完了！"
echo ""
echo "使用方法:"
echo "1. 仮想環境を有効化: source venv/bin/activate"
echo "2. テストを実行: pytest tests/"
echo "3. 開発を開始してください！"
echo ""
echo "注意: OpenCVとdlibの依存関係により、初回インストールには時間がかかる場合があります。"