import argparse

parser = argparse.ArgumentParser(description='provide arguments for AdInfoHRLTD3 algorithms')

# Actor Critic Parameters
parser.add_argument('--lr', type=float, default=0.001, help='learning rate, can increase to 0.005')
parser.add_argument('--gamma', help='discount factor for critic updates', default=0.99)
parser.add_argument('--alpha', help='Entropy regularization coefficient', default=0.2)
parser.add_argument('--polyak', help='averaging for target networks', default=0.995)
parser.add_argument('--buffer-size', help='max size of the replay buffer', default=1000000)
parser.add_argument('--hidden-size', help='number of units in the hidden layers', default=64)
parser.add_argument('--batch-size', help='size of minibatch for minibatch-SGD', default=100)
parser.add_argument("--max-grad-clip", type=float, default=5.0, help="Max norm gradient clipping value")
parser.add_argument("--tasks", type=list, default=['blue'], help="tasks")

# Episodes and Exploration Parameters
parser.add_argument('--total-step-num', help='total number of time steps', default=10000000)
parser.add_argument('--test-num', help='number of episode for recording the return', default=10)
parser.add_argument('--update-every', help='update model after certain number steps', type=int, default=50)

# Environment Parameters
parser.add_argument('--env_name', help='name of env', type=str, default="CartPole-v1")
parser.add_argument('--seed', type=int, help='random seed for repeatability', default=7)

# Plotting Parameters
parser.add_argument('--save-model-every', help='Save model every certain number of steps', type=int, default=10000)
parser.add_argument('--exp-name', help='Experiment Name', type=str, default="sac_1")
parser.add_argument('--model_dir', help='Model directory', type=str, default="model/")
parser.add_argument('--model_type', help='Model Type', type=str, default="SAC")
parser.add_argument('--config', help='config name', type=str, default="berrygrid.yaml")
parser.add_argument('--model_name', help='Model Name', type=str, default="")
parser.add_argument('--load_model', type=bool, default=False, help='load model to visualize')

args = parser.parse_args()