import torch
import torch.nn as nn

class ANetwork(nn.Module):
    def __init__(self, num_actions, num_cumulants, human_phase, num_tasks):
        super().__init__()
        self.num_actions = num_actions
        self.num_cumulants = num_cumulants
        self.human_phase = human_phase
        self.num_tasks = num_tasks

        self.embedding = ConvEncoder().double()
        self.normalization = LayerNormNLP(output_sizes=(64, 64)).double()
        self.phi = mlp(sizes=(64, self.num_cumulants*self.num_actions), # cumulants - [b, num_cumulants, num_actions]
                             activation=nn.ReLU, output_activation=nn.Identity).double()
        self.assistive_psi = mlp(sizes = (64, self.num_cumulants*self.num_actions),
                                      activation=nn.ReLU, output_activation=nn.Identity).double() # successor - [b, num_cumulants, num_actions]

        if self.human_phase:
            self.human_psi = self.assistive_psi.copy()

        self.w = nn.Parameter(torch.randn(self.num_tasks, self.num_cumulants)).double() # preference vector: [task_size, num_cumulants]
        # TODO: Selecting right preference vector based on task ID

    def forward(self, obs):
        embedding = self.embedding(obs.double())
        after_norm = self.normalization(embedding)
        phi = self.phi(after_norm)
        phi = torch.reshape(phi, (obs.shape[0], self.num_cumulants, self.num_actions))
        assistive_psi = self.assistive_psi(after_norm)
        assistive_psi = torch.reshape(assistive_psi, (obs.shape[0], self.num_cumulants, self.num_actions))

        assistive_rewards = torch.einsum("tc, bca  -> bta", self.w, phi)
        assistive_policy_params =  torch.einsum("tc, bca  -> ba", self.w, assistive_psi) # [b, num_cumulants, num_actions]

        if self.human_phase:
            human_psi = self.human_psi(after_norm)  # [b, num_demonstrators, num_cumulants, num_actions]
            human_psi = torch.reshape(human_psi, (obs.shape[0], self.num_tasks, self.num_cumulants, self.num_actions))  # [num_demonstrators, num_cumulants]
            human_rewards = torch.einsum("tc, bca  -> bta", self.w, phi)  # [b, task_size, num_actions]
            human_policy_params = torch.einsum("tc, btca  -> bca", self.w, human_psi)  # [b, num_cumulants, num_actions]
            return phi, assistive_psi, human_psi, self.w, assistive_rewards, human_rewards, assistive_policy_params, human_policy_params
        else:
            return phi, assistive_psi, None, self.w, assistive_rewards, None, assistive_policy_params, None


def mlp(sizes, activation, output_activation):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

class ConvEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding =  nn.Sequential(
            nn.Conv2d(3, 16, (8, 8),stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, (1, 1), stride=2),
            nn.ReLU(),
            nn.Flatten()).double()

    def forward(self, obs):
        embedding = self.embedding(obs)
        return embedding


class LayerNormNLP(nn.Module):
    def __init__(self, output_sizes):
        super().__init__()
        self.output_sizes = output_sizes

        # first linear layer.
        self.linear_layer = nn.Linear(32, self.output_sizes[0])

        # normalisation layer.
        self.norm_layer = nn.Sequential(nn.LayerNorm(64), # over the last dimension, not sure how to create_scale & create_offset
                                        nn.Tanh())
        # mlp module
        self.mlp_layer = mlp(output_sizes[1:], nn.ReLU(), output_activation=nn.ReLU)


    def forward(self, embedding):
        embedding = self.linear_layer(embedding)
        embedding = self.norm_layer(embedding)
        return self.mlp_layer(embedding)



