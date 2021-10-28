import pickle
import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from algorithms.models import Network
from misc.utils import to_onehot
from algorithms.replay_buffer import ReplayBufferPhaseII
from copy import deepcopy
from misc.utils import tensor


class AssistiveModel(nn.Module):
    def __init__(self, env, args, log, tb_writer):
        super(AssistiveModel, self).__init__()
        self.args = args
        self.tb_writer = tb_writer
        self.log = log
        self.phaseII = args.phaseII
        self.env = env

        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.num_tasks = len(args.tasks)
        self.num_actions = 7
        self.num_demonstrators = 1
        self.network = Network(self.num_actions, self.args.num_cumulants, args.phaseII, self.num_tasks)
        self.target_network = deepcopy(self.network)

        self.mse_loss = nn.MSELoss()
        self.softmax = nn.Softmax()
        self.cross_entropy_loss = nn.CrossEntropyLoss().double()
        self.optimizer = Adam(self.network.parameters(), lr=args.lr)
        self.gradient_steps = 0
        self.iteration = 0
        self.index_alternate = False

    def get_action(self, obs, evaluation=False):
        assert obs.shape[0] == 1
        p = np.random.random()

        if p < self.args.epsilon:
            action = self.env.action_space.sample() # TODO: epsilon
            return action.item()
        else:
            embedding = self.network.embedding(obs).squeeze(-1).squeeze(-1)
            if not self.phaseII:
                one_hot_task = to_onehot(tensor(self.network.task_id).unsqueeze(0), dim=self.num_tasks)
                one_hot_task = one_hot_task.expand(embedding.shape[0], -1)
                embedding = torch.cat([embedding, one_hot_task], dim=-1)

            after_norm = self.network.normalization(embedding)
            assistive_psi = self.network.assistive_psi(after_norm)
            assistive_psi = torch.reshape(assistive_psi, (self.args.num_cumulants, self.num_actions))
            assistive_actions = torch.matmul(assistive_psi.T, self.network.w[self.network.task_id])
            if evaluation:
                assistive_action = torch.argmax(assistive_actions.squeeze(0)).item()
            else:
                policy = torch.distributions.Categorical(logits=assistive_actions)
                assistive_action = policy.sample()
            return assistive_action

    def itd_loss_fn(self, phi, psi, next_psi, done):
        target = phi + self.args.gamma * torch.einsum("bca, b  -> bca", next_psi.detach(), done)
        return self.mse_loss(target, psi)

    def q_learning_loss_fn(self, q, a, r, done, q_next):
        target = r + self.args.gamma * done * torch.max(q_next).detach()
        return self.mse_loss(target,  torch.gather(q, 1, a.type(torch.int64)))

    def nll_loss_fn(self, action, q_next):
        # only for human - cross entropy between (one-hot) real action of human and the softmax of q of human.
        return self.cross_entropy_loss(to_onehot(action), self.softmax(torch.gather(q_next)))

    def reward_loss(self, real_reward, computed_reward, action):
        computed =  computed_reward[:, self.network.task_id]
        computed = torch.gather(computed, 1, action.type(torch.int64))
        return self.mse_loss(real_reward.unsqueeze(-1), computed)

    def update_loss(self, data):
        if self.phaseII:
            obs, human_action, next_obs = data['obs'], data['human_action'], data['next_obs']
        else:
            obs, assistant_action, real_assistant_reward, next_obs, next_assistant_action, done  = \
                    data['obs'],  data['assistant_action'], data['assistant_reward'], data['next_obs'], data['next_assistant_action'], data['done']

        self.optimizer.zero_grad()

        # Calculate the successor features for time-step `t` and `t+1`
        phi, assistant_psi, human_psi, \
            computed_assistant_rewards, human_rewards, \
            assistant_q, human_q = self.network(obs)

        _, target_next_assistant_psi, target_next_human_psi, _, _, target_next_assistant_q, _ = self.target_network(next_obs)

        itd_loss_assistant = self.itd_loss_fn(phi, assistant_psi, target_next_assistant_psi, done)
        self.tb_writer.log_data("assistant_loss/itd_loss", self.iteration, itd_loss_assistant.item())

        if self.phaseII:
            itd_loss_human = torch.mean(self.itd_loss_fn(phi, human_psi, target_next_human_psi, done))
            self.tb_writer.log_data("human_loss/itd_loss", self.iteration, itd_loss_human.item())

            bc_loss_human = torch.mean(self.nll_loss_fn(human_action, human_q))
            self.tb_writer.log_data("human_loss/bc_loss", self.iteration, bc_loss_human.item())

            total_loss = itd_loss_assistant

            if self.args.alternate_loss:
                total_loss += itd_loss_human if self.index_alternate else bc_loss_human
                self.index_alternate = not self.index_alternate
            else:
                total_loss += itd_loss_human + bc_loss_human

        else:
            assistant_q_k = assistant_q[:, self.network.task_id]
            target_next_assistant_q_k = target_next_assistant_q[:, self.network.task_id]
            dqn_loss_assistant = self.q_learning_loss_fn(assistant_q_k, assistant_action, real_assistant_reward, done, target_next_assistant_q_k)
            self.tb_writer.log_data("assistant_loss/dqn_loss", self.iteration, dqn_loss_assistant.item())

            reward_loss_assistant = self.reward_loss(real_assistant_reward, computed_assistant_rewards, assistant_action)
            self.tb_writer.log_data("assistant_loss/reward_loss", self.iteration, reward_loss_assistant.item())

            total_loss = itd_loss_assistant + dqn_loss_assistant + reward_loss_assistant

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.args.max_grad_clip)
        self.optimizer.step()
        self.gradient_steps += 1

        if self.gradient_steps % self.args.target_update_freq == 0:
            self.target_network.assistive_psi.load_state_dict(self.network.assistive_psi.state_dict())

    def save_human_expert(self):
        observation_shape = np.array(self.env.reset()['image']).shape

        buffer = ReplayBufferPhaseII(obs_dim=observation_shape[1:],
                              act_dim=self.env.action_space.shape,
                              size=self.args.buffer_size)
        filehandler = open("buffer.obj", "wb")
        pickle.dump(buffer, filehandler)
        filehandler.close()