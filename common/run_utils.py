"""
run_utils.py

一括実行・実行時間計測用ユーティリティ
"""

import time
from collections.abc import Callable
from functools import wraps


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


def run_steps(
    steps: list[tuple[str, Callable]],
) -> None:
    """
    サンプルプログラムを順番に実行する。
    """
    total_steps = len(steps)

    for index, (title, func) in enumerate(
        steps,
        start=1,
    ):
        print(f"\n[{index}/{total_steps}] {title}")
        func()