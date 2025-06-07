#\!/bin/bash

# 各ISSUEに対応するブランチを作成
echo "Creating branches for parallel development..."

# Issue #1: プロジェクト構造とセットアップ
git checkout -b feature/issue-1-project-setup main
git push -u origin feature/issue-1-project-setup

# Issue #2: 顔検出モジュール
git checkout -b feature/issue-2-face-detection main
git push -u origin feature/issue-2-face-detection

# Issue #3: 動画処理
git checkout -b feature/issue-3-video-processing main
git push -u origin feature/issue-3-video-processing

# Issue #4: CLIインターフェース
git checkout -b feature/issue-4-cli-interface main
git push -u origin feature/issue-4-cli-interface

# Issue #5: 顔交換アルゴリズム調査
git checkout -b feature/issue-5-algorithm-research main
git push -u origin feature/issue-5-algorithm-research

# mainブランチに戻る
git checkout main

echo "All branches created successfully\!"
echo "Use 'git checkout feature/issue-X-name' to switch between branches"
