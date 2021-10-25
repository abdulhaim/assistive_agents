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

    def get_action(self, obs, evaluation=False):
        if self.iteration < self.args.start_steps:
            action = self.env.action_space.sample() # TODO: epsilon
            return action.item()
        else:
            embedding = self.network.embedding(obs)
            after_norm = self.network.normalization(embedding)
            assistive_psi = self.network.assistive_psi(after_norm)
            assistive_psi = torch.reshape(assistive_psi, (1, self.args.num_cumulants, self.num_actions))
            assistive_actions = torch.einsum("bca, c  -> ba", assistive_psi, self.network.w[self.task])
            if evaluation:
                assistive_action = torch.argmax(assistive_actions.squeeze(0)).item()
            else:
                policy = torch.distributions.Categorical(logits=assistive_actions)
                assistive_action = policy.sample()
            return assistive_action

    def itd_loss_fn(self, phi, psi, next_psi, done):
        target = phi + self.args.gamma * torch.einsum("bca, b  -> bca", psi.detach(), done)
        return self.mse_loss(target, next_psi)

    def q_learning_loss_fn(self, q, a, r, done, q_next):
        target = r + self.args.gamma * done * torch.max(q).detach()
        return self.mse_loss(target,  torch.gather(q_next, 1, a.type(torch.int64)))

    def nll_loss_fn(self, action, q_next, next_action):
        # only for human - cross entropy between (one-hot) real action of human and the softmax of q of human.
        return self.cross_entropy_loss(to_onehot(action), self.softmax(torch.gather(q_next)))

    def reward_loss(self, real_reward, computed_reward, action):
        computed =  computed_reward[:, self.task]
        computed = torch.gather(computed, 1, action.type(torch.int64))
        return self.mse_loss(real_reward.unsqueeze(-1), computed)

    def update_loss(self, data):
        obs, assistant_action, real_assistant_reward, next_obs, next_assistant_action, done  = \
                data['obs'],  data['assistant_action'], data['assistant_reward'], data['next_obs'], data['next_assistant_action'], data['done']

        self.optimizer.zero_grad()

        # For Assistant Agent Phase I
        # Calculate the successor features for time-step `t` and `t+1`
        phi, assistant_psi, human_psi, w_params, \
            computed_assistant_rewards, human_rewards, \
            assistant_q, human_q = self.network(obs)

        next_phi, next_assistant_psi, next_human_phi, next_w_params, \
            next_computed_assistant_rewards, next_human_rewards, \
            next_assistant_q, next_human_q = self.network(next_obs)

        # For Assistant
        itd_loss_assistant = self.itd_loss_fn(phi, assistant_psi, next_assistant_psi, done)
        self.tb_writer.log_data("assistant_loss/itd_loss", self.iteration, itd_loss_assistant.item())

        assistant_q_k = assistant_q[:, self.task]
        next_assistant_q_k = next_assistant_q[:, self.task]
        dqn_loss_assistant = self.q_learning_loss_fn(assistant_q_k, assistant_action, real_assistant_reward, done, next_assistant_q_k)
        self.tb_writer.log_data("assistant_loss/dqn_loss", self.iteration, dqn_loss_assistant.item())

        reward_loss_assistant = self.reward_loss(real_assistant_reward, computed_assistant_rewards, assistant_action)
        self.tb_writer.log_data("assistant_loss/reward_loss", self.iteration, reward_loss_assistant.item())

        total_loss = itd_loss_assistant + dqn_loss_assistant + reward_loss_assistant
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.args.max_grad_clip)
        self.optimizer.step()


    def update_loss_phaseII(self, data):
        # Get Assistant
        if self.human_phase:
            obs, assistant_action, human_action, real_assistant_reward, next_obs, next_assistant_action, next_human_action, done  = \
                data['obs'],  data['assistant_action'], data['human_action'], data['assistant_reward'], data['next_obs'], data['next_assistant_action'], data['next_human_action'], data['done']
        else:
            obs, assistant_action, real_assistant_reward, next_obs, next_assistant_action, done  = \
                data['obs'],  data['assistant_action'], data['assistant_reward'], data['next_obs'], data['next_assistant_action'], data['done']

        self.optimizer.zero_grad()

        # For Assistant Agent Phase I
        # Calculate the successor features for time-step `t` and `t+1`
        phi, assistant_psi, human_psi, w_params, \
            computed_assistant_rewards, human_rewards, \
            assistant_q, human_q = self.network(obs)

        next_phi, next_assistant_psi, next_human_phi, next_w_params, \
            next_computed_assistant_rewards, next_human_rewards, \
            next_assistant_q, next_human_q = self.network(next_obs)

        # For Assistant
        itd_loss_assistant = self.itd_loss_fn(phi, assistant_psi, next_assistant_psi, done)
        self.tb_writer.log_data("assistant_loss/itd_loss", self.iteration, itd_loss_assistant.item())

        assistant_q_k = assistant_q[:, self.task]
        next_assistant_q_k = next_assistant_q[:, self.task]
        dqn_loss_assistant = self.q_learning_loss_fn(assistant_q_k, assistant_action, real_assistant_reward, done, next_assistant_q_k)
        self.tb_writer.log_data("assistant_loss/dqn_loss", self.iteration, dqn_loss_assistant.item())

        reward_loss_assistant = self.reward_loss(real_assistant_reward, computed_assistant_rewards, assistant_action, task_id)
        self.tb_writer.log_data("assistant_loss/reward_loss", self.iteration, reward_loss_assistant.item())

        if self.human_phase:
            itd_loss_human = torch.mean(self.itd_loss_fn(phi, human_psi, next_human_phi, done))
            self.tb_writer.log_data("human_loss/itd_loss", self.iteration, itd_loss_human.item())

            human_q_k = human_q[:, task_id]
            bc_loss_human = torch.mean(self.nll_loss_fn(human_action, human_q_k, next_human_action))
            self.tb_writer.log_data("human_loss/bc_loss", self.iteration, bc_loss_human.item())

            total_loss = itd_loss_assistant + dqn_loss_assistant + reward_loss_assistant
            if self.args.alternate_loss:
                total_loss += itd_loss_human if self.index_alternate else bc_loss_human
            else:
                total_loss += itd_loss_human + bc_loss_human

        else:
            total_loss = itd_loss_assistant + dqn_loss_assistant + reward_loss_assistant

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.args.max_grad_clip)
        self.optimizer.step()

class HumanModel(nn.Module):
    def __init__(self, args):
        super(HumanModel, self).__init__()
