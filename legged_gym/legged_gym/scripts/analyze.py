import torch

n_obstacle_passed = torch.load('a1_leap_metra_2_2000_n_obstacle_3.pt')
success_bins = torch.zeros((10,1))
for i in range(10):
    success_bins[i] = (n_obstacle_passed[:1000][i*100:(i+1)*100]>=1).sum()
print("mean: ", success_bins.mean())
print("std: ", success_bins.std())
import ipdb;ipdb.set_trace()