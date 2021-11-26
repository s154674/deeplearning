# Hyperparameters
total_steps = 8e6 #maybe 25e6?
num_envs = 32
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
c2 = 0.1

num_actions = 4
in_channels = 3
feature_dim = 64

env_name = 'maze'
use_background=False

### num_levels easy = 200, hard = 500
make_env_kwargs = {'num_levels':num_levels, 'env_name':env_name, 'use_backgrounds':use_background}

dct = {}
for var in ['total_steps', 'num_envs', 'num_steps', 'num_levels','num_epochs', 'batch_size','eps','grad_eps','value_coef','entropy_coef', 'c1', 'c2', 'num_actions','in_channels','feature_dim', 'env_name','use_background']:
    dct[var] = eval(var)
print(dct)
####################################################

import numpy as np
import torch
def compute_returns(rewards, discount_factor):
    """Compute discounted returns."""
    returns = np.zeros(len(rewards))
    returns[-1] = rewards[-1]
    for t in reversed(range(len(rewards)-1)):
        returns[t] = rewards[t] + discount_factor * returns[t+1]
    return returns

def clip(rt, at, eps = 0.2):
    result = []
    rtat = rt*at
    clip = torch.clip(rt,1 - eps, 1 + eps)*at
    for i in range(len(rt)):
      result.append(min(rtat[i], clip[i]))
    return torch.tensor(result).cuda()



#####################################################

#https://spinningup.openai.com/en/latest/algorithms/ppo.html#documentation
#https://blog.varunajayasiri.com/ml/ppo_pytorch.html
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import make_env, Storage, orthogonal_init
# import numpy as np


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Encoder(nn.Module):
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


class Policy(nn.Module):
  def __init__(self, encoder, feature_dim, num_actions):
    super().__init__()
    self.encoder = encoder
    self.policy = orthogonal_init(nn.Linear(feature_dim, num_actions), gain=.01)
    self.value = orthogonal_init(nn.Linear(feature_dim, 1), gain=1.)

  def act(self, x):
    with torch.no_grad():
      x = x.cuda().contiguous()
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


# Define environment
# check the utils.py file for info on arguments
env = make_env(**make_env_kwargs)
print('Observation space:', env.observation_space)
print('Action space:', env.action_space.n)

# Define network
encoder = Encoder(in_channels, feature_dim)
policy =  Policy(encoder, feature_dim, env.action_space.n)
policy.cuda()

# Define optimizer
# these are reasonable values but probably not optimal
optimizer = torch.optim.Adam(policy.parameters(), lr=5e-4, eps=1e-5)

# Define temporary storage
# we use this to collect transitions during each iteration
storage = Storage(
    env.observation_space.shape,
    num_steps,
    num_envs
)

# Run training
obs = env.reset()
step = 0
while step < total_steps:

  # Use policy to collect data for num_steps steps
  policy.eval()
  for _ in range(num_steps):
    # Use policy
    action, log_prob, value = policy.act(obs)
    
    # Take step in environment
    next_obs, reward, done, info = env.step(action)

    # Store data
    storage.store(obs, action, reward, done, info, log_prob, value)
    
    # Update current observation
    obs = next_obs

  # Add the last observation to collected data
  _, _, value = policy.act(obs)
  storage.store_last(obs, value)

  # Compute return and advantage
  storage.compute_return_advantage()

  # Optimize policy
  policy.train()
  for epoch in range(num_epochs):

    # Iterate over batches of transitions
    generator = storage.get_generator(batch_size)
    for batch in generator:
      b_obs, b_action, b_log_prob, b_value, b_returns, b_advantage = batch

      # Get current policy outputs
      new_dist, new_value = policy(b_obs)
      new_log_prob = new_dist.log_prob(b_action)

      #rewards to go
      r_t = torch.tensor(compute_returns(b_returns, 0.99)).cuda() # changed to 0.99 from 0.09 after trying to train 3 times
      #advantage estimate = discounted_rewards - value_function or baseline estimate
      a_t = r_t - b_value

      # Clipped policy objective
      #r_theta=(b_log_prob-new_log_prob).exp()
      # most probably we should do torch.mean(clip(r_thet, a_t, eps))
      #old pi_loss = torch.mean(clip(r_t, a_t, eps))
      r_theta = (b_log_prob-new_log_prob).exp()
      pi_loss =  torch.mean(clip(r_theta, a_t, eps))

      # # Clipped value function objective
      value_loss = torch.mean((r_t - b_value)**2)

      # Entropy loss
      # entropy_loss = torch.mean(torch.tensor(nn.CrossEntropyLoss()).cuda())
      # entropy_loss = nn.CrossEntropyLoss()
      # ce_loss = entropy_loss(r_t, new_value)
      ce_loss = new_dist.entropy().mean()


      # Backpropagate losses
      # loss = -torch.mean(torch.mul(torch.log(b_log_prob), b_returns))
      loss = -(pi_loss - c1 * value_loss + c2 * ce_loss)
      loss.backward()

      # Clip gradients
      torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_eps)

      # Update policy
      optimizer.step()
      optimizer.zero_grad()

  # Update stats
  step += num_envs * num_steps
  print(f'Step: {step}\tMean reward: {storage.get_reward()}')

print('Completed training!')
from datetime import datetime
print(datetime.utcnow().replace(microsecond=0))
torch.save(policy.state_dict, f'checkpoint{datetime.utcnow().replace(microsecond=0)}.pt')
