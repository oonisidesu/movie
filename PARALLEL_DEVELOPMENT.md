# 並行開発ガイド

## ブランチ戦略

各ISSUEごとに独立したfeatureブランチで開発を進めます。

### ブランチ一覧
- `feature/issue-1-project-setup` - Issue #1: プロジェクト構造とセットアップ
- `feature/issue-2-face-detection` - Issue #2: 顔検出モジュール
- `feature/issue-3-video-processing` - Issue #3: 動画処理
- `feature/issue-4-cli-interface` - Issue #4: CLIインターフェース
- `feature/issue-5-algorithm-research` - Issue #5: アルゴリズム調査

## 作業の進め方

### 1. ブランチの作成（初回のみ）
```bash
# セットアップスクリプトを実行
./setup_branches.sh
```

### 2. 作業開始時
```bash
# 作業したいISSUEのブランチに切り替え
git checkout feature/issue-1-project-setup

# 最新のmainブランチの変更を取り込む（必要に応じて）
git pull origin main
git merge main
```

### 3. 作業中
```bash
# 変更をコミット
git add .
git commit -m "feat: [#1] 実装内容の説明"

# リモートにプッシュ
git push origin feature/issue-1-project-setup
```

### 4. 別のISSUEに切り替える
```bash
# 現在の作業を保存
git add .
git commit -m "WIP: 作業中の内容"

# 別のブランチに切り替え
git checkout feature/issue-2-face-detection
```

### 5. 作業完了時
```bash
# GitHub上でPull Requestを作成
gh pr create --base main --head feature/issue-1-project-setup \
  --title "feat: プロジェクト構造とセットアップ #1" \
  --body "Closes #1"
```

## 並行作業時の注意点

1. **依存関係の管理**
   - Issue #1（セットアップ）は他の作業の基盤となるため、優先的に完了させる
   - Issue #2, #3, #4は比較的独立して進められる
   - Issue #5（調査）は他の実装と並行して進められる

2. **コンフリクトの回避**
   - 各ブランチで異なるファイル/ディレクトリを扱うように心がける
   - 共通ファイル（requirements.txt等）の変更は早めにmainにマージ

3. **定期的な同期**
   - mainブランチの更新を定期的に各featureブランチに取り込む
   - 小さな単位で頻繁にPRを作成してマージする

## 推奨される作業分担例

### 開発者A
- Issue #1: プロジェクトセットアップ（最優先）
- Issue #2: 顔検出モジュール

### 開発者B
- Issue #5: アルゴリズム調査（最優先）
- Issue #3: 動画処理

### 開発者C
- Issue #4: CLIインターフェース
- テストコードの作成

このように分担することで、効率的に並行開発を進められます。