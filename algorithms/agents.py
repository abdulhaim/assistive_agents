import torch
import torch.nn as nn
from torch.optim import Adam
from algorithms.models import ANetwork

class AAgent(nn.Module):
    def __init__(self, env, args, log, tb_writer):
        super(AAgent, self).__init__()
        self.args = args
        self.tb_writer = tb_writer
        self.log = log
        self.human_phase = args.human_phase

        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.num_tasks = len(args.tasks)
        self.num_actions = 7
        self.num_demonstrators = 1
        self.network = ANetwork(self.num_actions, self.args.num_cumulants, args.human_phase, self.num_tasks)
        self.optimizer = Adam(self.network.parameters(), lr=args.lr)

    def get_action(self, obs, task_id):
        embedding = self.network.embedding(obs)
        after_norm = self.network.normalization(embedding)
        assistive_psi = self.network.assistive_psi(after_norm)
        assistive_psi = torch.reshape(assistive_psi, (1, self.args.num_cumulants, self.num_actions))
        assistive_actions = torch.einsum("bca, c  -> ba", assistive_psi, self.network.w[task_id])
        assistive_actions = assistive_actions.squeeze(0)
        action = torch.argmax(assistive_actions[task_id])
        if self.args.human_phase:
            # TODO: Not sure what to do here since there shouldn't be access to human action?
            return
        else:
            return action

    def predict_task(self, obs):
        # TODO: Should we have net for predicting task?
        return 0 # for now

    def itd_loss_fn(self, phi, psi, psi_next, done):
        target = phi + self.args.gamma * torch.einsum("bca, b  -> bca", psi_next, done)
        return target - psi

    def q_learning_loss_fn(self, q, a, r, done, q_next):
        target = r + self.args.gamma * done * torch.max(q_next)
        a = torch.tensor(a, dtype=torch.int64)
        q_gather = torch.gather(q, 1, a)
        return target - q_gather

    def nll_loss_fn(self, logits, target):
        distribution = torch.distributions.Categorical(logits=logits)
        return -distribution.log_prob(target).sum(axis=-1)

    def update_loss(self, data, task_id):
        # Get Assistant
        if self.human_phase:
            obs, assistant_action, human_action, assistant_reward, next_obs, assistant_next_action, human_next_action, done  = \
                data['obs'],  data['assistant_action'], data['human_action'], data['assistant_reward'], data['next_obs'], data['assistant_next_action'], data['human_next_action'], data['done']
        else:
            obs, assistant_action, assistant_reward, next_obs, assistant_next_action, done  = \
                data['obs'],  data['assistant_action'], data['assistant_reward'], data['next_obs'], data['assistant_next_action'], data['done']

        # For Assistant Agent Phase I
        # Calculate the successor features for time-step `t` and `t+1` (next)
        phi, assistant_psi, human_psi, w_params, \
        assistant_rewards, human_rewards, \
        assistant_policy_params, human_policy_params = self.network(obs)

        phi_next, assistant_psi_next, human_psi_next, w_params_next, \
        assistant_rewards_next, human_rewards_next, \
        assistant_policy_params_next, human_policy_params_next = self.network(next_obs)

        if self.args.human_phase:
            task_id = self.predict_task()    # TODO: Determine TASK ID

        # For Assistant
        q_input_assistive = torch.einsum("bca, c  -> ba", assistant_psi, w_params[task_id]) #psi*w
        itd_loss_assistant = torch.mean(self.itd_loss_fn(phi, assistant_psi, assistant_psi_next, done))
        dqn_loss_assistant = torch.mean(self.q_learning_loss_fn(q_input_assistive, assistant_action, assistant_reward, done, assistant_next_action))
        bc_loss_assistant = torch.mean(self.nll_loss_fn(q_input_assistive, assistant_action)) # TODO: not sure if this is correct

        if self.human_phase:
            itd_loss_human = torch.mean(self.itd_loss_fn(phi, human_psi, human_psi_next, done))
            q_input_human = torch.einsum("bca, c  -> ba", human_psi, w_params[task_id])  # psi*w
            bc_loss_human = torch.mean(self.nll_loss_fn(q_input_human, human_action))  # TODO: not sure if this is correct
            total_loss = itd_loss_human + itd_loss_assistant
            total_loss += dqn_loss_assistant # TODO: No DQN loss for human cause no reward?
            total_loss += + bc_loss_assistant + bc_loss_human
        else:
            total_loss = itd_loss_assistant + dqn_loss_assistant + bc_loss_assistant


        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()














