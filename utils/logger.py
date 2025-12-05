import os
import logging
from torch.utils.tensorboard import SummaryWriter

class Logger:
    """
    日志记录器，支持控制台输出和TensorBoard
    """
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=log_dir)
        
        # 配置logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, "train.log")),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger()

    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)
        
    def info(self, msg):
        self.logger.info(msg)
        
    def close(self):
        self.writer.close()
