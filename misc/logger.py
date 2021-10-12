import logging
import os
import time
import numpy as np
from tensorboardX import SummaryWriter


class TensorBoardLogger():
    def __init__(self, logdir, run_name):
        self.log_name = logdir + '/' + run_name
        self.tf_writer = None
        self.start_time = time.time()
        self.n_eps = 0
        self.total_options = 0

        if not os.path.exists(self.log_name):
            os.makedirs(self.log_name)

        self.writer = SummaryWriter(self.log_name)

        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.log_name + '/logger.log'),
            ],
            datefmt='%Y/%m/%d %I:%M:%S %p'
        )

    def log_data(self, tag_value, total_steps, value):
        self.writer.add_scalar(tag=tag_value, scalar_value=value, global_step=total_steps)
