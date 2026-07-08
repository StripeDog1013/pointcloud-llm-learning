"""
log_utils.py

ログ・JSON・タイムスタンプ関連の共通ユーティリティ
"""

import json
from datetime import datetime
from pathlib import Path

from common.path_utils import create_directory


def get_timestamp() -> str:
    """
    現在時刻を YYYYMMDD_HHMMSS 形式で返す。
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_json(
    data: dict,
    prefix: str,
    log_dir: str | Path = "../logs",
) -> Path:
    """
    dictをJSON形式で保存する。

    ファイル名:
        {prefix}_{timestamp}.json
    """
    log_dir = create_directory(log_dir)
    output_path = log_dir / f"{prefix}_{get_timestamp()}.json"

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=4,
            ensure_ascii=False,
        )

    return output_path


def load_json(file_path: str | Path) -> dict:
    """
    JSONファイルを読み込む。
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)