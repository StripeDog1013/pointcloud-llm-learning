"""
path_utils.py

パス・ディレクトリ関連の共通ユーティリティ
"""

from pathlib import Path


def create_directory(path: str | Path) -> Path:
    """
    ディレクトリを作成してPathとして返す。
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    return path


def ensure_parent_directory(file_path: str | Path) -> Path:
    """
    ファイル保存前に親ディレクトリを作成する。
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    return file_path


def get_project_root() -> Path:
    """
    common/ の1つ上をプロジェクトルートとして返す。
    """
    return Path(__file__).resolve().parents[1]


def resolve_from_project_root(*paths: str) -> Path:
    """
    プロジェクトルート基準でパスを解決する。
    """
    return get_project_root().joinpath(*paths)