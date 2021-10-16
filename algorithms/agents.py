import torch
import torch.nn as nn
from torch.optim import Adam
from algorithms.models import Network
from misc.utils import to_onehot

class AssistiveModel(nn.Module):
    def __init__(self, env, args, log, tb_writer):
        super(AssistiveModel, self).__init__()
        self.args = args
        self.tb_writer = tb_writer
        self.log = log
        self.human_phase = args.human_phase
        self.env = env

        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.num_tasks = len(args.tasks)
        self.num_actions = 7
        self.num_demonstrators = 1
        self.network = Network(self.num_actions, self.args.num_cumulants, args.human_phase, self.num_tasks)
        self.mse_loss = nn.MSELoss()
        self.softmax = nn.Softmax()
        self.cross_entropy_loss = nn.CrossEntropyLoss().double()
        self.optimizer = Adam(self.network.parameters(), lr=args.lr)
        self.iteration = 0

    def get_action(self, obs, task_id, evaluation=False):
        if self.iteration < self.args.start_steps:
            action = self.env.action_space.sample() # TODO: epsilon
            return action.item()
        else:
            embedding = self.network.embedding(obs)
            after_norm = self.network.normalization(embedding)
            assistive_psi = self.network.assistive_psi(after_norm)
            assistive_psi = torch.reshape(assistive_psi, (1, self.args.num_cumulants, self.num_actions))
            assistive_actions = torch.einsum("bca, c  -> ba", assistive_psi, self.network.w[task_id])
            if evaluation:
                assistive_action = torch.argmax(assistive_actions.squeeze(0)).item()
            else:
                policy = torch.distributions.Categorical(logits=assistive_actions)
                assistive_action = policy.sample()
            return assistive_action

    def predict_task(self, obs):
        # TODO: Should we have net for predicting task?
        return 0 # for now

    def itd_loss_fn(self, phi, psi, psi_prev, done):
        target = phi + self.args.gamma * torch.einsum("bca, b  -> bca", psi.detach(), done)
        return self.mse_loss(target, psi_prev)

    def q_learning_loss_fn(self, q, r, done, q_prev, a_prev):
        target = r + self.args.gamma * done * torch.max(q).detach()
        return self.mse_loss(target,  torch.gather(q_prev, 1, a_prev.type(torch.int64)))

    def nll_loss_fn(self, action, q_prev, prev_action):
        # only for human - cross entropy between (one-hot) real action of human and the softmax of q of human.
        return self.cross_entropy_loss(to_onehot(action), self.softmax(torch.gather(q_prev, 1, prev_action)))

    def reward_loss(self, reward, computed_reward,action, task_id):
        computed =  computed_reward[:, task_id]
        computed = torch.gather(computed, 1, action.type(torch.int64))
        return self.mse_loss(reward.unsqueeze(-1), computed)

    def update_loss(self, data, task_id):
        # Get Assistant
        if self.human_phase:
            obs, assistant_action, human_action, assistant_reward, prev_obs, assistant_prev_action, human_prev_action, done  = \
                data['obs'],  data['assistant_action'], data['human_action'], data['assistant_reward'], data['next_obs'], data['assistant_prev_action'], data['human_prev_action'], data['done']
        else:
            obs, assistant_action, assistant_reward, prev_obs, assistant_prev_action, done  = \
                data['obs'],  data['assistant_action'], data['assistant_reward'], data['prev_obs'], data['assistant_prev_action'], data['done']

        self.optimizer.zero_grad()

        # For Assistant Agent Phase I
        # Calculate the successor features for time-step `t` and `t+1` (next)
        phi, assistant_psi, human_psi, w_params, \
            assistant_rewards, human_rewards, \
            assistant_q, human_q = self.network(obs)

        phi_prev, assistant_psi_prev, human_psi_prev, w_params_prev, \
            assistant_rewards_prev, human_rewards_prev, \
            assistant_q_prev, human_q_prev = self.network(prev_obs)

        if self.args.human_phase:
            task_id = self.predict_task()    # TODO: Determine TASK ID

        # For Assistant
        itd_loss_assistant = self.itd_loss_fn(phi, assistant_psi, assistant_psi_prev, done)
        self.tb_writer.log_data("assistant_loss/itd_loss", self.iteration, itd_loss_assistant.item())

        assistant_q_k = assistant_q[:, task_id]
        assistant_q_prev_k = assistant_q_prev[:, task_id]
        dqn_loss_assistant = self.q_learning_loss_fn(assistant_q_k, assistant_reward, done, assistant_q_prev_k, assistant_prev_action)
        self.tb_writer.log_data("assistant_loss/dqn_loss", self.iteration, dqn_loss_assistant.item())

        reward_loss_assistant = self.reward_loss(assistant_reward, assistant_rewards, assistant_action, task_id)
        self.tb_writer.log_data("assistant_loss/reward_loss", self.iteration, reward_loss_assistant.item())

        if self.human_phase:
            itd_loss_human = torch.mean(self.itd_loss_fn(phi, human_psi, human_psi_prev, done))
            self.tb_writer.log_data("human_loss/itd_loss", self.iteration, itd_loss_human.item())

            human_q_prev_k = human_q_prev[:, task_id]
            bc_loss_human = torch.mean(self.nll_loss_fn(human_action, human_q_prev_k, human_prev_action))
            self.tb_writer.log_data("human_loss/bc_loss", self.iteration, bc_loss_human.item())

            total_loss = itd_loss_human + itd_loss_assistant
            total_loss += dqn_loss_assistant + reward_loss_assistant
            total_loss += + bc_loss_human
        else:
            total_loss = itd_loss_assistant + dqn_loss_assistant + reward_loss_assistant

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.args.max_grad_clip)
        self.optimizer.step()
