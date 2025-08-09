import logging
import sys


def setup_logger() -> logging.Logger:
    logger = logging.getLogger()

    # 既にハンドラーが設定されている場合は重複を避ける
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    )

    file_handler = logging.FileHandler("app.log")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    return logger


def get_logger(name: str = None) -> logging.Logger:
    """アプリケーション用のロガーを取得する関数"""
    # ルートロガーがまだ設定されていない場合は設定
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        setup_logger()

    # 指定された名前のロガーを返す
    return logging.getLogger(name)
