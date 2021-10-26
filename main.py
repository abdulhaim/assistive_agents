
import time
import os
import random
import torch
import numpy as np
import gym
import pickle
import berrygrid

from misc.utils import make_env, set_log, load_config
from misc.logger import TensorBoardLogger
from misc.arguments import args
from algorithms.replay_buffer import ReplayBufferPhaseI
from algorithms.agents import AssistiveModel
from trainer import train_phaseI, train_phaseII

torch.set_num_threads(3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    # Create directories
    if not os.path.exists("./logs/"):
        os.makedirs("./logs/")
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    # Create loggings
    log = set_log(args)
    tb_writer = TensorBoardLogger(logdir="./logs_tensorboard/", run_name=args.log_name + time.ctime())

    # Set seeds
    env = make_env(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if device == torch.device("cuda"):
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(args.seed)

    if args.phaseII:
        file = open("buffer.obj", 'rb')
        buffer = pickle.load(file)
    else:
        observation_shape = np.array(env.reset()['image']).shape
        buffer = ReplayBufferPhaseI(obs_dim=observation_shape[1:],
                              act_dim=env.action_space.shape,
                              size=args.buffer_size)

    # Set Assistant Agent
    r_agent = AssistiveModel(env = env,
                            args = args,
                            log = log,
                            tb_writer = tb_writer)

    # Set Phase 1 Training
    if args.phaseII:
        r_agent.network.load_phaseII(args)
        train_phaseII(args, r_agent, env, buffer)
    else:
        train_phaseI(args, r_agent, env, buffer)

if __name__ == '__main__':
    # Load experiment specific config if provided
    if args.config is not None:
        load_config(args)

    # Set log name
    args.log_name = \
        "%s_model_name::%s_seed" % (args.model_name, args.seed)

    main(args)
