import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from matplotlib import pyplot as plt
import os

class DQNbn(nn.Module):
    def __init__(self, in_channels=1, n_actions=14):
        """
        Initialize Deep Q Network

        Args:
            in_channels (int): number of input channels
            n_actions (int): number of outputs
        """
        super(DQNbn, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.head = nn.Linear(512, n_actions)
        
    def forward(self, x):
        x = x.float() / 255
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.head(x)

class NoisyLinear(nn.Module):
  def __init__(self, in_features, out_features, std_init=0.5):
    super(NoisyLinear, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.std_init = std_init
    self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
    self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
    self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
    self.bias_mu = nn.Parameter(torch.empty(out_features))
    self.bias_sigma = nn.Parameter(torch.empty(out_features))
    self.register_buffer('bias_epsilon', torch.empty(out_features))
    self.reset_parameters()
    self.reset_noise()

  def reset_parameters(self):
    mu_range = 1 / math.sqrt(self.in_features)
    self.weight_mu.data.uniform_(-mu_range, mu_range)
    self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
    self.bias_mu.data.uniform_(-mu_range, mu_range)
    self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

  def _scale_noise(self, size):
    x = torch.randn(size, device=self.weight_mu.device)
    return x.sign().mul_(x.abs().sqrt_())

  def reset_noise(self):
    epsilon_in = self._scale_noise(self.in_features)
    epsilon_out = self._scale_noise(self.out_features)
    self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
    self.bias_epsilon.copy_(epsilon_out)

  def forward(self, input):
    if self.training:
      return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)
    else:
      return F.linear(input, self.weight_mu, self.bias_mu)


class DQN(nn.Module):
    def __init__(self, in_channels=1, n_actions=4):
        """
        Initialize Deep Q Network

        Args:
            in_channels (int): number of input channels
            n_actions (int): number of outputs
        """
        super(DQN, self).__init__()
        self.atoms = args.atoms
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=4, stride=1) # kernel_size=4
        # self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1) # kernel_size=3
        # self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=2, stride=1) # kernel_size=2
        # self.bn3 = nn.BatchNorm2d(64)
        # self.flatten = nn.Flatten()
        self.fc4 = nn.Linear(32*25*9, 128)
        # 내 위치랑 목표 위치를 받아서 append
        # Layer 5 추가
        self.fc5 = nn.Linear(128, 128)
        self.conv_output_size = 128
        self.head = nn.Linear(128, n_actions)
        self.fc_h_v = NoisyLinear(self.conv_output_size, 512, std_init=0.1)
        self.fc_h_a = NoisyLinear(self.conv_output_size, 512, std_init=0.1)
        self.fc_z_v = NoisyLinear(512, self.atoms, std_init=0.1)
        self.fc_z_a = NoisyLinear(512, n_actions * self.atoms, std_init=0.1)
        
    def forward(self, x, plot=False, log=False, steps=0):
        # x를 map, my_pos, terminal로 나누기
        # map, pos, ter = x
        # x를 map으로 치환
        # pos, ter을 append(torch.cat으로)

        # print(x.shape) torch.Size[32, 1, 21, 21]
        # Mazeview2D map test 코드
        # x_numpy = x.cpu().numpy()
        # plt.imshow(x_numpy[0,0,:,:])
        # plt.savefig('figures3/maze-image{}.png'.format(steps))
        # print(os.path.isfile('figures3/maze-image{}.png'.format(steps)))
        if plot:
            my_maze = x.cpu().numpy()
            plt.subplots(1, 1)
            plt.imshow(my_maze[0,0,:,:])
            plt.axis('off')
            plt.savefig('test/maze-image{}.png'.format(steps))
            
        batch = x.size(0)
        x = x.float() / 255
        x = F.relu(self.conv1(x))
        # print(x.shape)
        if plot:
            x_numpy = x.cpu().detach().numpy()
            fig, axes = plt.subplots(2,8)
            for i in range(16):
                axes[i//8, i%8].imshow(x_numpy[0,i,:,:])
            plt.savefig('test/conv1_{}.png'.format(steps))

        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # print(x.shape)
        if plot:
            x_numpy = x.cpu().detach().numpy()
            fig, axes = plt.subplots(4,8)
            for i in range(32):
                axes[i//8, i%8].imshow(x_numpy[0,i,:,:])
            plt.savefig('test/conv3_{}.png'.format(steps))

        # x = self.flatten(x)
        # print("x shape: ",x.shape)
        x = F.relu(self.fc4(x.view(batch, -1)))
        x = F.relu(self.fc5(x))

        x = self.convs(x)
        x = x.view(-1, self.conv_output_size)
        v = self.fc_z_v(F.relu(self.fc_h_v(x)))  # Value stream
        a = self.fc_z_a(F.relu(self.fc_h_a(x)))  # Advantage stream
        v, a = v.view(-1, 1, self.atoms), a.view(-1, self.action_space, self.atoms)
        q = v + a - a.mean(1, keepdim=True)  # Combine streams
        if log:  # Use log softmax for numerical stability
            q = F.log_softmax(q, dim=2)  # Log probabilities with action over second dimension
        else:
            q = F.softmax(q, dim=2)  # Probabilities with action over second dimension
        return q


        return self.head(x)