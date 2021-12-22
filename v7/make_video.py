import imageio
import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import make_env, Storage, orthogonal_init

def compute_returns(rewards, discount_factor):
    """Compute discounted returns."""
    returns = np.zeros(len(rewards))
    returns[-1] = rewards[-1]
    for t in reversed(range(len(rewards)-1)):
        returns[t] = rewards[t] + discount_factor * returns[t+1]
    return returns

def clip(rtheta, at, eps = 0.2):
    result = []
    rtat = rtheta*at
    clip = torch.clamp(rtheta,min = 1 - eps,max = 1 + eps)*at
    return torch.min(rtat, clip)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

# Hyperparameters
total_steps = 4e6 #maybe 25e6?
num_envs = 1
num_levels = 200
num_steps = 256
num_epochs = 3
batch_size = 512 # maybe 8
eps = .2
grad_eps = .5
value_coef = .5
entropy_coef = .01
#TODO: choose appropriate values for c1 and c2
c1 = 0.5
c2 = 0.01

num_actions = 15
in_channels = 3
feature_dim = 64

env_name = 'starpilot'
use_background = False

### num_levels easy = 200, hard = 500
make_env_kwargs = {'num_levels':num_levels, 'env_name':env_name}

class IMPALAEncoder(nn.Module):
  def __init__(self, in_channels, feature_dim):
    super().__init__()
    self.layers = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1),
        nn.MaxPool2d(3,2)
    )

    self.residual = nn.Sequential(
        nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding = 1),
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding = 1),
    )

    self.residual2 = nn.Sequential(
        nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1),
    )

    self.rest = nn.Sequential(
        Flatten(),
        nn.ReLU(),
        nn.Linear(in_features=28800, out_features=feature_dim),
        nn.ReLU(),
    )
    self.apply(orthogonal_init)

  def forward(self, x):
    x = self.layers(x)
    x += self.residual(x)
    x += self.residual(x)
    x = self.rest(x)

    return x

class PPOEncoder(nn.Module):
  def __init__(self, in_channels, feature_dim):
    super().__init__()
    self.layers = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4), nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2), nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1), nn.ReLU(),
        Flatten(),
        nn.Linear(in_features=1024, out_features=feature_dim), nn.ReLU()
    )
    self.apply(orthogonal_init)

  def forward(self, x):
    return self.layers(x)


class Policy(nn.Module):
  def __init__(self, encoder, feature_dim, num_actions):
    super().__init__()
    self.encoder = encoder
    self.policy = orthogonal_init(nn.Linear(feature_dim, num_actions), gain=.01)
    self.value = orthogonal_init(nn.Linear(feature_dim, 1), gain=1.)

  def act(self, x):
    with torch.no_grad():
      x = x.contiguous()
      dist, value = self.forward(x)
      action = dist.sample()
      log_prob = dist.log_prob(action)
    
    return action.cpu(), log_prob.cpu(), value.cpu()

  def forward(self, x):
    x = self.encoder(x)
    logits = self.policy(x)
    value = self.value(x).squeeze(1)
    dist = torch.distributions.Categorical(logits=logits)

    return dist, value

# Make evaluation environment
eval_env = make_env(**make_env_kwargs, start_level=300, use_backgrounds=False, n_envs=1)
obs = eval_env.reset()

frames = []
total_reward = []

encoder = IMPALAEncoder(in_channels, feature_dim)
policy =  Policy(encoder, feature_dim, eval_env.action_space.n)

state_dict = torch.load("checkpoints\\starpilot_nosta_impala_no_re_500_checkpoint2021-12-07 10_30_25.pt", map_location=torch.device('cpu'))
# v6\output\checkpoints\\ppo_1checkpoint2021-12-05 22_02_58.pt
# v6\output\checkpoints\\ppo_1checkpoint2021-12-06 00_48_59.pt
policy.load_state_dict(state_dict())

# policy = Policy(state_dict['encoder'], state_dict['feature_dim'], state_dict['num_actions'])
# Evaluate policy
policy.eval()
count = 0
while count < 101:

  # Use policy
  action, log_prob, value = policy.act(obs)

  # Take step in environment
  obs, reward, done, info = eval_env.step(action)
  # print(done)
  if done.any():
    print(obs)
    print(done)
    count += 1
    print(f'done {count}')
  total_reward.append(torch.Tensor(reward))

  # Render environment and store
  frame = (torch.Tensor(eval_env.render(mode='rgb_array'))*255.).byte()
  frames.append(frame)

# print(total_reward)
# Calculate average return
total_reward2 = torch.stack(total_reward).sum(0).mean(0)
print('Average return:', total_reward2)
print('Some other thing: ', torch.stack(total_reward).mean(0))

# Save frames as video
frames = torch.stack(frames)
imageio.mimsave('vid2nd.mp4', frames, fps=25)