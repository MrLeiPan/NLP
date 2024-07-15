import logging
import os


class Logger:
    def __init__(self, log_file='app.log', log_level=logging.INFO):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

        # Create file handler which logs even debug messages
        fh = logging.FileHandler(log_file)
        fh.setLevel(log_level)

        # Create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.ERROR)

        # Create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - \n%(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # Add the handlers to the logger
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)

    def close(self):
        """Close all handlers."""
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)


# 使用示例
if __name__ == "__main__":
    # 创建日志实例
    log = Logger(log_file='logs/my_app.log', log_level=logging.INFO)

    # 记录日志信息
    log.info("这是一个信息日志")

    # 关闭日志文件
    log.close()
    #
    # # 删除日志文件
    # if os.path.exists('logs/my_app.log'):
    #     os.remove('logs/my_app.log')
    #     print("日志文件已删除")
    # else:
    #     print("日志文件不存在")
