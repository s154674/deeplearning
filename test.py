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