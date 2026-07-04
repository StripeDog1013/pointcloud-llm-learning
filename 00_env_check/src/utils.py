"""
utils.py

共通ユーティリティ
"""

import platform
import sys
import time
from functools import wraps


def print_header(title: str) -> None:
    """
    セクションタイトルを表示する。

    Parameters
    ----------
    title : str
        タイトル
    """
    print("=" * 60)
    print(title)
    print("=" * 60)


def print_subheader(title: str) -> None:
    """
    サブタイトルを表示する。

    Parameters
    ----------
    title : str
        サブタイトル
    """
    print("-" * 60)
    print(title)
    print("-" * 60)


def print_python_info() -> None:
    """
    Python実行環境を表示する。
    """
    print(f"Python Version : {sys.version}")
    print(f"Python Executable : {sys.executable}")
    print(f"Platform : {platform.platform()}")
    print(f"Machine : {platform.machine()}")
    print(f"Processor : {platform.processor()}")


def timer(func):
    """
    関数の実行時間を表示するデコレータ。
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()

        result = func(*args, **kwargs)

        elapsed = time.perf_counter() - start
        print(f"\nExecution Time : {elapsed:.3f} sec")

        return result

    return wrapper