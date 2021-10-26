import torch
import torch.nn as nn
from misc.utils import tensor
from misc.utils import to_onehot

class Network(nn.Module):
    def __init__(self, num_actions, num_cumulants, phaseII, num_tasks):
        super().__init__()
        self.num_actions = num_actions
        self.num_cumulants = num_cumulants
        self.phaseII = phaseII
        self.num_tasks = num_tasks

        self.embedding = ConvEncoder()
        self.normalization = LayerNormNLP(output_sizes=(64, 128), num_tasks=self.num_tasks)
        self.phi = mlp(sizes=(64, self.num_cumulants*self.num_actions), # cumulants - [b, num_cumulants, num_actions]
                             activation=nn.ReLU, output_activation=nn.Identity) # TODO: add action to this in phaseIII

        self.assistive_psi = mlp(sizes = (64, self.num_cumulants*self.num_actions),
                                      activation=nn.ReLU, output_activation=nn.Identity) # successor - [b, num_cumulants, num_actions] #TODO: add actionn to this

        self.task_id = 0


        if self.phaseII:
            self.w = nn.Parameter(torch.randn(self.num_cumulants))
            self.human_psi = mlp(sizes = (64, self.num_cumulants*self.num_actions),
                                      activation=nn.ReLU, output_activation=nn.Identity) # successor - [b, num_cumulants, num_actions] #TODO: add actionn to this
        else:
            self.w = nn.Parameter(torch.randn(self.num_tasks, self.num_cumulants)) # preference vector: [task_size, num_cumulants]


    def forward(self, obs):
        embedding = self.embedding(tensor(obs))
        if not self.phaseII:
            one_hot_task = to_onehot(tensor(self.task_id).unsqueeze(0), dim=self.num_tasks)
            one_hot_task = one_hot_task.expand(embedding.shape[0],-1)
            embedding = torch.cat([embedding, one_hot_task], dim=-1)
        after_norm = self.normalization(embedding)
        phi = self.phi(after_norm)
        phi = torch.reshape(phi, (obs.shape[0], self.num_cumulants, self.num_actions))
        assistive_psi = self.assistive_psi(after_norm)
        assistive_psi = torch.reshape(assistive_psi, (obs.shape[0], self.num_cumulants, self.num_actions))
        assistive_rewards = torch.einsum("tc, bca  -> bta", self.w, phi)
        assistive_q =  torch.einsum("tc, bca  -> bta", self.w, assistive_psi) # [b, num_cumulants, num_actions]

        if self.phaseII:
            human_psi = self.human_psi(after_norm)
            human_psi = torch.reshape(human_psi, (obs.shape[0], self.num_cumulants, self.num_actions))
            human_q = torch.einsum("tc, bca  -> bta", self.w, human_psi)
            return phi, assistive_psi, human_psi, assistive_rewards, assistive_q, human_q
        else:
            return phi, assistive_psi, None, assistive_rewards, None, assistive_q, None


    def load_phaseII(self, args):
        self.load_state_dict(torch.load(args.human_model), strict=False)
        self.human_psi = self.assistive_psi.copy()
        self.w = nn.Parameter(torch.randn(self.num_cumulants))

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
            nn.Conv2d(3, 16, (3, 3),stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, (1, 1), stride=2),
            nn.ReLU(),
            nn.Flatten())

    def forward(self, obs):
        embedding = self.embedding(obs)
        return embedding

class LayerNormNLP(nn.Module):
    def __init__(self, output_sizes, num_tasks):
        super().__init__()
        self.output_sizes = output_sizes
        self.num_tasks = num_tasks

        # first linear layer.
        self.linear_layer = nn.Linear(32 + self.num_tasks, self.output_sizes[0])

        # normalisation layer.
        self.norm_layer = nn.Sequential(nn.LayerNorm(self.output_sizes[0]), # over the last dimension, not sure how to create_scale & create_offset
                                        nn.Tanh())
        # mlp module
        self.mlp_layer = mlp(output_sizes[1:], nn.ReLU(), output_activation=nn.ReLU)

    def forward(self, embedding):
        embedding = self.linear_layer(embedding)
        embedding = self.norm_layer(embedding)
        return self.mlp_layer(embedding)
