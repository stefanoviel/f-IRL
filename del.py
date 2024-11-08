# load this /home/viel/f-IRL/expert_data/actions/HalfCheetahFH-v0_airl.pt

import torch

# Specify the path to your .pt file
file_path = '/home/viel/f-IRL/expert_data/actions/HalfCheetahFH-v0_airl.pt'

# Load the checkpoint
checkpoint = torch.load(file_path)

print(file_path)
print(checkpoint.shape)

# Specify the path to your .pt file
file_path = '/home/viel/f-IRL/expert_data/states/Hopper-v5.pt'

# Load the checkpoint
checkpoint = torch.load(file_path)

print(file_path)
print(checkpoint.shape)

# Specify the path to your .pt file
file_path = '/home/viel/f-IRL/expert_data/states/Hopper-v5_1_sto.pt'

# Load the checkpoint
checkpoint = torch.load(file_path)

print(file_path)
print(checkpoint.shape)