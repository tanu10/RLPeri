#from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import warnings
#from torch.distributions import Categorical
#from sklearn.metrics import mean_squared_error
warnings.simplefilter("ignore")


lkern_channel = 8
vkern_channel = 16

class ConvAdapt(nn.Module):
    def __init__(self, in_channels, out_channels, p):
        super(ConvAdapt, self).__init__()
        self.lkern = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=int(p/lkern_channel), bias=True)
        self.vkern = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=int(p/vkern_channel), bias=True)

    def forward(self, x):
        return self.lkern(x) + self.vkern(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.lhc1 = ConvAdapt(planes, planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.lhc2 = ConvAdapt(planes, planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                ConvAdapt(self.expansion*planes, self.expansion *
                          planes, int(self.expansion*planes)),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.lhc1(self.conv1(x))))
        out = self.bn2(self.lhc2(self.conv2(out)))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class QNetwork(nn.Module):
    '''action_dim = [dim_action1, dim_action2]'''
    def __init__(self, action_dim, hidden_dim=256):
        super(QNetwork, self).__init__()
        block = BasicBlock
        num_blocks = [2, 2, 2, 2]
        self.in_planes = 64

        self.conv1 = nn.Conv2d(41, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.linear = nn.Linear(128*2, hidden_dim)

        self.value_head = nn.Linear(hidden_dim, 1)
        self.adv_heads = nn.ModuleList([nn.Linear(hidden_dim, action_dim[i]) for i in range(len(action_dim))])
        self.max_len = max(action_dim)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x1, x2):
        out1 = self.conv1(x1)
        out1 = F.relu(self.bn1(out1))
        out1 = self.layer1(out1)
        out1 = self.layer2(out1)
        out1 = F.avg_pool2d(out1, 4)

        out2 = self.conv1(x2)
        out2 = F.relu(self.bn1(out2))
        out2 = self.layer1(out2)
        out2 = self.layer2(out2)
        out2 = F.avg_pool2d(out2, 4)

        out = torch.cat((out1, out2), dim=1)

        feat = out.view(out.size(0), -1)
        out = self.linear(feat)
        value = self.value_head(out)
        advs = []
        means = []
        for l in self.adv_heads:
            ad = l(out)
            '''As dimnsions of each action branch is different, use padding to make them of same size'''
            advs.append(F.pad(ad, (0, self.max_len-ad.shape[-1])))
            means.append(l(out).mean(1, keepdim=True))
        advs = torch.stack(advs, dim=1)
        means = torch.stack(means, dim=1)
        q_val = value.unsqueeze(2) + advs - means
        return q_val


class RLAgent(nn.Module):
    def __init__(self, state_dim, action_dim, target_update_freq, learning_rate, gamma, hidden_dim, td_target, device):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.beta = 1

        self.network = QNetwork(action_dim, hidden_dim)
        self.target_network = QNetwork(action_dim, hidden_dim)
        self.target_network.load_state_dict(self.network.state_dict())
        self.optim = optim.Adam(self.network.parameters(), lr=learning_rate, weight_decay=1e-3)

        self.network.to(device)
        self.target_network.to(device)
        self.device = device

        self.target_update_freq = target_update_freq
        self.update_counter = 0

        self.td_target = td_target
        self.stimuli_mask = [0 if i<action_dim[1] else 1 for i in range(action_dim[0])]


    '''Action: [num_locations, stimuli_value]
        action_mask is to choose from only valid available actions
        stimuli_mask is to mask padding added to perform matrix operations'''
    def get_action(self, state_seen, state_not_seen, action_mask):
        with torch.no_grad():
            state_seen = torch.tensor(state_seen).to(self.device, dtype=torch.float)
            state_not_seen = torch.tensor(state_not_seen).to(self.device, dtype=torch.float)
            out = self.network(state_seen, state_not_seen).squeeze(0)
        out = out.detach().cpu().numpy()
        out = np.ma.masked_array(out, mask=[action_mask, self.stimuli_mask])
        action = np.argmax(out, axis=1)
        return action

    def update_policy(self, st, st_n, act, r, nst, nst_n, term):
        states_seen = torch.tensor(st).float().to(self.device, dtype=torch.float)
        states_not_seen = torch.tensor(st_n).float().to(self.device, dtype=torch.float)
        actions = torch.tensor(act).long().to(self.device)
        rewards = torch.tensor(r).float().to(self.device)
        next_states_seen = torch.tensor(nst).float().to(self.device, dtype=torch.float)
        next_states_not_seen = torch.tensor(nst_n).float().to(self.device, dtype=torch.float)
        term = torch.tensor(term).float().to(self.device, dtype=torch.float)
        
        current_Q = self.network(states_seen, states_not_seen).gather(2, actions).squeeze(-1)

        with torch.no_grad():
            argmax = torch.argmax(self.network(next_states_seen, next_states_not_seen), dim=2)
            max_next_Q = self.target_network(next_states_seen, next_states_not_seen).gather(2, argmax.unsqueeze(2)).squeeze(-1)
            if self.td_target == "mean":
                max_next_Q = max_next_Q.mean(1, keepdim=True)
            elif self.td_target == "max":
                max_next_Q, _ = max_next_Q.max(1, keepdim=True)

        expected_Q = rewards + max_next_Q * self.gamma * term
        loss = F.mse_loss(expected_Q, current_Q)

        self.optim.zero_grad()
        loss.backward()

        for p in self.network.parameters():
            p.grad.data.clamp_(-1., 1.)
        self.optim.step()

        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.update_counter = 0
            self.target_network.load_state_dict(self.network.state_dict())

        return loss.detach().cpu()



