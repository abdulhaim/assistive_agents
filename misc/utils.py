import logging
import yaml
import numpy as np
import random
import torch

# Setting CUDA USE
use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
device = torch.device("cuda" if use_cuda else "cpu")

COLOR_TO_IDX = {
    'red'   : 0,
    'orange': 1,
    'green' : 2,
    'blue'  : 3,
    'purple': 4,
    'yellow': 5,
    'grey'  : 6
}

def load_config(args, path="."):
    """Loads and replaces default parameters with experiment
    specific parameters
    Args:
        args (argparse): Python argparse that contains arguments
        path (str): Root directory to load config from. Default: "."
    """
    with open(path + "/config/" + args.config, 'r') as f:
        config = yaml.safe_load(f)

    for key, value in config.items():
        args.__dict__[key] = value


def set_logger(logger_name, log_file, level=logging.INFO):
    log = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    log.setLevel(level)
    log.addHandler(fileHandler)
    log.addHandler(streamHandler)
    log.propagate = False  # otherwise root logger prints things again


def set_log(args):
    log = {}
    set_logger(
        logger_name=args.log_name,
        log_file=r'{0}{1}'.format("./logs/", args.log_name))
    log[args.log_name] = logging.getLogger(args.log_name)

    for arg, value in sorted(vars(args).items()):
        log[args.log_name].info("%s: %r", arg, value)

    return log

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def make_env(seed, task=None):
    import gym
    import berrygrid
    if task:
        env = gym.make("MultiGrid-Color-Gather-Env-8x8-v0", kwargs={'color_pick': task})
        env.max_episode_steps = env.max_steps

    else:
        tasks = ["red", "green", "blue", "purple", "grey"]
        random_color = random.choice(tasks)
        env = gym.make("MultiGrid-Color-Gather-Env-8x8-v0",  kwargs={'color_pick': random_color})
        env.max_episode_steps = env.max_steps

    env.seed(seed)
    env.action_space.seed(seed)
    return env

def format_obs(obs):
    obs = np.array(obs['image'])
    obs_shape = torch.tensor(obs).shape
    obs = torch.reshape(tensor(obs), (obs_shape[0], obs_shape[3], obs_shape[1], obs_shape[2]))
    return obs

def get_color(obs):
    return  COLOR_TO_IDX[obs['color']]


def to_onehot(value, dim):
    """Convert batch of numbers to onehot
    Args:
        value (numpy.ndarray): Batch of numbers to convert to onehot. Shape: (batch,)
        dim (int): Dimension of onehot
    Returns:
        onehot (numpy.ndarray): Converted onehot. Shape: (batch, dim)
    """
    one_hot = torch.zeros(value.shape[0], dim)
    one_hot[torch.arange(value.shape[0]), value.long()] = 1
    return one_hot

def tensor(x):
    if isinstance(x, torch.Tensor):
        return x.to(device).float()
    x = torch.tensor(x, device=device)
    return x.to(device).float()