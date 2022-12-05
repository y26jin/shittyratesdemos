#%%
from timeit import timeit
import numpy as np
import scipy 

# Plain vanilla call 
def bs_py(s0, strike, T, impliedVol, r):
    S = s0
    K = strike
    dt = T
    sigma = impliedVol
    phi = scipy.stats.norm.cdf
    d1 = (np.log(S/K)+(r+sigma**2/2)*dt)/(sigma*np.sqrt(dt))
    d2 = d1-sigma*np.sqrt(dt)
    return S*phi(d1) - K*np.exp(-r*dt)*phi(d2)

# %%
import torch

# Py Torch version
def bs_torch(s0, strike, T, impliedVol, r):
    S = s0
    K = strike
    dt = T
    sigma = impliedVol
    phi = torch.distributions.Normal(0,1).cdf
    d1 = (torch.log(S/K)+(r+sigma**2/2)*dt)/(sigma*torch.sqrt(dt))
    d2 = d1-sigma*torch.sqrt(dt)
    return S*phi(d1) - K*torch.exp(-r*dt)*phi(d2)
#%%
# call backwards to calculate first order derivatives(aka Greeks)

S0=torch.tensor([100.], requires_grad=True)
K=torch.tensor([101.], requires_grad=True)
T=torch.tensor([1.], requires_grad=True)
sigma=torch.tensor([0.3], requires_grad=True)
r=torch.tensor([0.01], requires_grad=True)
npv_torch = bs_torch(S0,K,T,sigma,r)
#npv_torch.backward()
#%%
# delta and gamma
gradient = torch.autograd.grad(npv_torch, S0, create_graph=True)
delta, = gradient
delta.backward(retain_graph=True)
print('Delta: ', delta)
print('Gamma: ', S0.grad)
# %%

'''
Try some Barrier options!
'''

# 1. Barrier with numpy

def monte_carlo_down_out_py(S_0, strike, time_to_expiry, implied_vol, riskfree_rate, barrier, steps, samples):
    stdnorm_random_variates = np.random.randn(samples, steps)
    S = S_0
    K = strike
    dt = time_to_expiry / stdnorm_random_variates.shape[1]
    sigma = implied_vol
    r = riskfree_rate
    B = barrier
    # See Advanced Monte Carlo methods for barrier and related exotic options by Emmanuel Gobet
    B_shift = B*np.exp(0.5826*sigma*np.sqrt(dt))
    S_T = S * np.cumprod(np.exp((r-sigma**2/2)*dt+sigma*np.sqrt(dt)*stdnorm_random_variates), axis=1)
    non_touch = (np.min(S_T, axis=1) > B_shift)*1
    call_payout = np.maximum(S_T[:,-1] - K, 0)
    npv = np.mean(non_touch * call_payout)
    return np.exp(-time_to_expiry*r)*npv

# test py version

S0=100.
K=110.
dt=2.
impliedVol = .2
riskfree_rate = .03
barrier = 90.
steps = 1000
samples = 100000

print(monte_carlo_down_out_py(S0,K,dt,impliedVol,riskfree_rate,barrier,steps,samples))

# 2. Now, try torch

def monte_carlo_down_out_torch(S_0, strike, time_to_expiry, implied_vol, riskfree_rate, barrier, steps, samples):
    stdnorm_random_variates = torch.distributions.Normal(0,1).sample((samples, steps))
    S = S_0
    K = strike
    dt = time_to_expiry / stdnorm_random_variates.shape[1]
    sigma = implied_vol
    r = riskfree_rate
    B = barrier
    # See Advanced Monte Carlo methods for barrier and related exotic options by Emmanuel Gobet
    B_shift = B*torch.exp(0.5826*sigma*torch.sqrt(dt))
    S_T = S * torch.cumprod(torch.exp((r-sigma**2/2)*dt+sigma*torch.sqrt(dt)*stdnorm_random_variates), dim=1)
    non_touch = torch.min(S_T, dim=1)[0] > B_shift
    call_payout = S_T[:,-1] - K
    call_payout[call_payout<0]=0
    npv = torch.mean(non_touch.type(torch.FloatTensor) * call_payout)
    return torch.exp(-time_to_expiry*r)*npv

S0_torch = torch.tensor([100.], requires_grad=True)
K_torch = torch.tensor([110.], requires_grad=True)
dt_torch = torch.tensor([2.], requires_grad=True)
sigma_torch = torch.tensor([.2], requires_grad=True)
r_torch = torch.tensor([.03], requires_grad=True)
barrier_torch = torch.tensor([90.], requires_grad=True)
npv_torch_mc = monte_carlo_down_out_torch(S0_torch, K_torch, dt_torch, sigma_torch, r_torch, barrier_torch, 1000, 100000)
print(npv_torch_mc)

# %%
'''
Now, CUDA version
This is gonna be deployed on cloud
'''

def monte_carlo_down_out_torch_cuda(S_0, strike, time_to_expiry, implied_vol, riskfree_rate, barrier, steps, samples):
    stdnorm_random_variates = torch.cuda.FloatTensor(steps, samples).normal_()
    S = S_0
    K = strike
    dt = time_to_expiry / stdnorm_random_variates.shape[1]
    sigma = implied_vol
    r = riskfree_rate
    B = barrier
    # See Advanced Monte Carlo methods for barrier and related exotic options by Emmanuel Gobet
    B_shift = B*torch.exp(0.5826*sigma*torch.sqrt(dt))
    S_T = S * torch.cumprod(torch.exp((r-sigma**2/2)*dt+sigma*torch.sqrt(dt)*stdnorm_random_variates), dim=1)
    non_touch = torch.min(S_T, dim=1)[0] > B_shift
    non_touch = non_touch.type(torch.cuda.FloatTensor)
    call_payout = S_T[:,-1] - K
    call_payout[call_payout<0]=0
    npv = torch.mean(non_touch * call_payout)
    return torch.exp(-time_to_expiry*r)*npv
# %%
