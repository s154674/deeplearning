import imageio
import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import make_env, Storage, orthogonal_init
import os
import json

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

final_result_dict = {}

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

seeds = [234,654,682,501,984]
for file_name in os.listdir('./checkpoints'):
  if 'starpilot_nosta' in file_name:
    for seed in seeds:
      if 'impala' in file_name:
        if 'no_re' in file_name:
          if '_75_' in file_name:
              eval_env = make_env(**make_env_kwargs, start_level=500, use_backgrounds=False, seed=seed)
              obs = eval_env.reset()
              total_reward = []
              encoder = IMPALAEncoder(in_channels, feature_dim)
              policy =  Policy(encoder, feature_dim, eval_env.action_space.n)
              eval_env2 = make_env(**make_env_kwargs, start_level=500, use_backgrounds=True, seed=seed)
              obs2 = eval_env2.reset()
              total_reward2 = []
              encoder2 = IMPALAEncoder(in_channels, feature_dim)
              policy2 =  Policy(encoder2, feature_dim, eval_env2.action_space.n)
          if '_200_' in file_name:
              eval_env = make_env(**make_env_kwargs, start_level=500, use_backgrounds=False, seed=seed)
              obs = eval_env.reset()
              total_reward = []
              encoder = IMPALAEncoder(in_channels, feature_dim)
              policy =  Policy(encoder, feature_dim, eval_env.action_space.n)
              eval_env2 = make_env(**make_env_kwargs, start_level=500, use_backgrounds=True, seed=seed)
              obs2 = eval_env2.reset()
              total_reward2 = []
              encoder2 = IMPALAEncoder(in_channels, feature_dim)
              policy2 =  Policy(encoder2, feature_dim, eval_env2.action_space.n)
          if '_500_' in file_name:
              eval_env = make_env(**make_env_kwargs, start_level=500, use_backgrounds=False, seed=seed)
              obs = eval_env.reset()
              total_reward = []
              encoder = IMPALAEncoder(in_channels, feature_dim)
              policy =  Policy(encoder, feature_dim, eval_env.action_space.n)
              eval_env2 = make_env(**make_env_kwargs, start_level=500, use_backgrounds=True, seed=seed)
              obs2 = eval_env2.reset()
              total_reward2 = []
              encoder2 = IMPALAEncoder(in_channels, feature_dim)
              policy2 =  Policy(encoder2, feature_dim, eval_env2.action_space.n)
        else:
          if '_75_' in file_name:
              eval_env = make_env(**make_env_kwargs, start_level=500, use_backgrounds=False, seed=seed)
              obs = eval_env.reset()
              total_reward = []
              encoder = IMPALAEncoder(in_channels, feature_dim)
              policy =  Policy(encoder, feature_dim, eval_env.action_space.n)
              eval_env2 = make_env(**make_env_kwargs, start_level=500, use_backgrounds=True, seed=seed)
              obs2 = eval_env2.reset()
              total_reward2 = []
              encoder2 = IMPALAEncoder(in_channels, feature_dim)
              policy2 =  Policy(encoder2, feature_dim, eval_env2.action_space.n)
          if '_200_' in file_name:
              eval_env = make_env(**make_env_kwargs, start_level=500, use_backgrounds=False, seed=seed)
              obs = eval_env.reset()
              total_reward = []
              encoder = IMPALAEncoder(in_channels, feature_dim)
              policy =  Policy(encoder, feature_dim, eval_env.action_space.n)
              eval_env2 = make_env(**make_env_kwargs, start_level=500, use_backgrounds=True, seed=seed)
              obs2 = eval_env2.reset()
              total_reward2 = []
              encoder2 = IMPALAEncoder(in_channels, feature_dim)
              policy2 =  Policy(encoder2, feature_dim, eval_env2.action_space.n)
          if '_500_' in file_name:
              eval_env = make_env(**make_env_kwargs, start_level=500, use_backgrounds=False, seed=seed)
              obs = eval_env.reset()
              total_reward = []
              encoder = IMPALAEncoder(in_channels, feature_dim)
              policy =  Policy(encoder, feature_dim, eval_env.action_space.n)
              eval_env2 = make_env(**make_env_kwargs, start_level=500, use_backgrounds=True, seed=seed)
              obs2 = eval_env2.reset()
              total_reward2 = []
              encoder2 = IMPALAEncoder(in_channels, feature_dim)
              policy2 =  Policy(encoder2, feature_dim, eval_env2.action_space.n)
      else:
        if 'no_re' in file_name:
          if '_75_' in file_name:
              eval_env = make_env(**make_env_kwargs, start_level=500, use_backgrounds=False, seed=seed)
              obs = eval_env.reset()
              total_reward = []
              encoder = PPOEncoder(in_channels, feature_dim)
              policy =  Policy(encoder, feature_dim, eval_env.action_space.n)
              eval_env2 = make_env(**make_env_kwargs, start_level=500, use_backgrounds=True, seed=seed)
              obs2 = eval_env2.reset()
              total_reward2 = []
              encoder2 = PPOEncoder(in_channels, feature_dim)
              policy2 =  Policy(encoder2, feature_dim, eval_env2.action_space.n)
          if '_200_' in file_name:
              eval_env = make_env(**make_env_kwargs, start_level=500, use_backgrounds=False, seed=seed)
              obs = eval_env.reset()
              total_reward = []
              encoder = PPOEncoder(in_channels, feature_dim)
              policy =  Policy(encoder, feature_dim, eval_env.action_space.n)
              eval_env2 = make_env(**make_env_kwargs, start_level=500, use_backgrounds=True, seed=seed)
              obs2 = eval_env2.reset()
              total_reward2 = []
              encoder2 = PPOEncoder(in_channels, feature_dim)
              policy2 =  Policy(encoder2, feature_dim, eval_env2.action_space.n)
          if '_500_' in file_name:
              eval_env = make_env(**make_env_kwargs, start_level=500, use_backgrounds=False, seed=seed)
              obs = eval_env.reset()
              total_reward = []
              encoder = PPOEncoder(in_channels, feature_dim)
              policy =  Policy(encoder, feature_dim, eval_env.action_space.n)
              eval_env2 = make_env(**make_env_kwargs, start_level=500, use_backgrounds=True, seed=seed)
              obs2 = eval_env2.reset()
              total_reward2 = []
              encoder2 = PPOEncoder(in_channels, feature_dim)
              policy2 =  Policy(encoder2, feature_dim, eval_env2.action_space.n)
        else:
          if '_75_' in file_name:
              eval_env = make_env(**make_env_kwargs, start_level=500, use_backgrounds=False, seed=seed)
              obs = eval_env.reset()
              total_reward = []
              encoder = PPOEncoder(in_channels, feature_dim)
              policy =  Policy(encoder, feature_dim, eval_env.action_space.n)
              eval_env2 = make_env(**make_env_kwargs, start_level=500, use_backgrounds=True, seed=seed)
              obs2 = eval_env2.reset()
              total_reward2 = []
              encoder2 = PPOEncoder(in_channels, feature_dim)
              policy2 =  Policy(encoder2, feature_dim, eval_env2.action_space.n)
          if '_200_' in file_name:
              eval_env = make_env(**make_env_kwargs, start_level=500, use_backgrounds=False, seed=seed)
              obs = eval_env.reset()
              total_reward = []
              encoder = PPOEncoder(in_channels, feature_dim)
              policy =  Policy(encoder, feature_dim, eval_env.action_space.n)
              eval_env2 = make_env(**make_env_kwargs, start_level=500, use_backgrounds=True, seed=seed)
              obs2 = eval_env2.reset()
              total_reward2 = []
              encoder2 = PPOEncoder(in_channels, feature_dim)
              policy2 =  Policy(encoder2, feature_dim, eval_env2.action_space.n)
          if '_500_' in file_name:
              eval_env = make_env(**make_env_kwargs, start_level=500, use_backgrounds=False, seed=seed)
              obs = eval_env.reset()
              total_reward = []
              encoder = PPOEncoder(in_channels, feature_dim)
              policy =  Policy(encoder, feature_dim, eval_env.action_space.n)
              eval_env2 = make_env(**make_env_kwargs, start_level=500, use_backgrounds=True, seed=seed)
              obs2 = eval_env2.reset()
              total_reward2 = []
              encoder2 = PPOEncoder(in_channels, feature_dim)
              policy2 =  Policy(encoder2, feature_dim, eval_env2.action_space.n)
      
      state_dict = torch.load("checkpoints\\" + file_name, map_location=torch.device('cpu'))
      state_dict2 = torch.load("checkpoints\\" + file_name, map_location=torch.device('cpu'))
      
      policy.load_state_dict(state_dict())
      policy2.load_state_dict(state_dict2())

      policy.eval()
      policy2.eval()

      for i in range(512):
        # Use policy
        action, log_prob, value = policy.act(obs)
        # Take step in environment
        obs, reward, done, info = eval_env.step(action)
        total_reward.append(torch.Tensor(reward))

        # Use policy
        action2, log_prob2, value2 = policy2.act(obs2)
        # Take step in environment
        obs2, reward2, done2, info2 = eval_env2.step(action2)
        total_reward2.append(torch.Tensor(reward2))
        if i % 100 == 0:
          print(i)

      total_reward = torch.stack(total_reward).sum(0).mean(0)
      print('Average return:', total_reward)
      final_result_dict[file_name + f'_bg_false_seed_{seed}'] = total_reward

      total_reward2 = torch.stack(total_reward2).sum(0).mean(0)
      print('Average return:', total_reward2)
      final_result_dict[file_name + f'_bg_true_seed_{seed}'] = total_reward2

print('FINAL RESULTS:')
print(final_result_dict)
with open('final_result_dict2.txt', 'w') as f:
  f.write(json.dumps(final_result_dict))