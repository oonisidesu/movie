"""
セットアップのテスト
"""
import os
import sys
import pytest


def test_python_version():
    """Python バージョンが3.8以上であることを確認"""
    assert sys.version_info >= (3, 8), "Python 3.8以上が必要です"


def test_project_structure():
    """プロジェクト構造が正しく作成されていることを確認"""
    expected_dirs = [
        "src",
        "src/core",
        "src/ui", 
        "src/utils",
        "tests",
        "docs",
        "scripts"
    ]
    
    for dir_path in expected_dirs:
        assert os.path.isdir(dir_path), f"ディレクトリ {dir_path} が存在しません"


def test_required_files():
    """必要なファイルが存在することを確認"""
    expected_files = [
        "README.md",
        "requirements.txt",
        "requirements-dev.txt",
        ".gitignore",
        "CLAUDE.md",
        "scripts/setup.sh"
    ]
    
    for file_path in expected_files:
        assert os.path.isfile(file_path), f"ファイル {file_path} が存在しません"


def test_imports():
    """基本的なパッケージがインポートできることを確認"""
    try:
        import src
        assert hasattr(src, '__version__')
    except ImportError:
        pytest.fail("srcパッケージがインポートできません")