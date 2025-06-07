# 動画顔変換ツール

動画内の人物の顔を別の人物の顔に変換するツールです。

## プロジェクト概要

- **目的**: 動画内の顔を別の顔に自然に変換する
- **用途**: 個人使用・教育目的限定
- **動作環境**: ローカル環境（クラウド不要）

## システム要件

- **OS**: Windows 10/11、macOS、Linux
- **Python**: 3.8以上
- **メモリ**: 8GB以上（16GB推奨）
- **GPU**: NVIDIA GPU（CUDA対応）推奨（CPUでも動作可能）

## セットアップ

### 自動セットアップ（推奨）

```bash
# リポジトリをクローン
git clone https://github.com/oonisidesu/movie.git
cd movie

# セットアップスクリプトを実行
./scripts/setup.sh
```

### 手動セットアップ

```bash
# 仮想環境を作成
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 依存関係をインストール
pip install -r requirements.txt

# 必要なディレクトリを作成
mkdir -p output temp faces target_faces test_videos
```

## プロジェクト構造

```
movie/
├── src/                    # ソースコード
│   ├── core/              # コア機能（顔検出、顔交換）
│   ├── ui/                # ユーザーインターフェース
│   └── utils/             # ユーティリティ
├── tests/                 # テストコード
├── docs/                  # ドキュメント
├── scripts/               # セットアップスクリプト
├── requirements.txt       # 本番用依存関係
├── requirements-dev.txt   # 開発用依存関係
└── .gitignore            # Git除外設定
```

## 開発

### 開発環境のセットアップ

```bash
# 開発用依存関係をインストール
pip install -r requirements-dev.txt

# Pre-commitフックを設定
pre-commit install
```

### 実行コマンド

```bash
# 仮想環境を有効化
source venv/bin/activate

# テストを実行
pytest tests/

# コードフォーマット
black src/ tests/

# リンター実行
flake8 src/ tests/

# 型チェック
mypy src/
```

## 使用方法

### 基本的な使用方法

```bash
# CLIでの実行（実装予定）
python -m src.ui.cli --input video.mp4 --face target_face.jpg --output result.mp4
```

### 対応ファイル形式

- **入力動画**: MP4、AVI、MOV
- **出力動画**: MP4（音声付き）
- **顔画像**: JPG、PNG

## 開発状況

現在MVPフェーズの開発中です。

### Phase 1: MVP
- [x] プロジェクト構造とセットアップ (#1)
- [ ] 顔検出モジュールの実装 (#2)
- [ ] 動画ファイル処理 (#3)
- [ ] CLIインターフェース (#4)
- [ ] 顔交換アルゴリズム調査 (#5)

### Phase 2: 機能拡張（予定）
- 複数人物対応
- 品質調整機能
- 他の動画形式サポート

### Phase 3: 改善（予定）
- パフォーマンス最適化
- UI/UX改善
- バッチ処理機能

## ライセンス・倫理的配慮

- **使用制限**: 個人使用または教育目的に限定
- **プライバシー**: すべての処理はローカル環境で実行
- **責任**: 悪意のある使用を禁止

## 貢献

Issues や Pull Requests は歓迎します。詳細は `PARALLEL_DEVELOPMENT.md` を参照してください。

## サポート

問題が発生した場合は、GitHubの Issues でお知らせください。