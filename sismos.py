from differential_evolution import *
from timer import Timer

import numpy as np

import torch.nn.functional as F
import torch.nn as nn

import argparse
from functools import partial

def pairwise_dist_sq(P,Q):
    # P.shape = (batch, m, 2)
    # Q.shape = (batch, n, 2)
    # result.shape = (batch, m, n)
    P = P[...,None,:] # (batch, m, 1, 2)
    Q = Q[...,None,:,:] # (batch, 1, n, 2)
    R = P-Q
    R *= R
    return R.sum(dim=-1)

def pairwise_dist2_batch(x):
    return pairwise_dist_sq(x,x)

def l1_batch(x,A,smooth=False):
    D = torch.abs(pairwise_dist2_batch(x)-A)
    if smooth:
        smalls = D[D<1.0]
        D[D<1.0] =  smalls**2 # smooth L1
    return torch.mean(D,dim=(1,2))

def l2_batch(x,A):
    D = pairwise_dist2_batch(x)-A
    return torch.mean(D*D,dim=(1,2))

def clamp(x):
    #x -= x.mean(dim=0)*0.9
    mask = (x > 4.0) + (x < -4.0)
    x[mask] = torch.randn_like(x[mask])*2
    return x

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('matrix',type=str, help="file with (symmetric) distance matrix")

    
    
    parser.add_argument("-t","--time", type=float, default=16, help="Number of seconds to spend finding a good solution (per restart)")
    
    parser.add_argument("-e","--error", type=str, default="L2", help="either use L1, L2, or smoothL1")
    
    parser.add_argument("-n","--num_restarts", type=int, default=8, help="Number of times that we try to restart")
    parser.add_argument("-p","--pop_size", type=int, default=64, help="Population size for diff evo")
    parser.add_argument("-s","--shuffles", type=int, default=1, help="Number of times populations mix.")
    
    parser.add_argument("-d","--disable_cuda", dest='disable_cuda', action='store_true', help="If true, disable CUDA (even when available). If false, use CUDA only if available. Ignored if you don't have CUDA.")
    
    parser.set_defaults(disable_cuda=False)
    
    args = parser.parse_args()
    
    use_cuda = False if args.disable_cuda else torch.cuda.is_available()
    
    errstr = args.error.lower()
    if errstr == 'l2':
        error_func = l2_batch
    elif errstr == 'l1':
        error_func = partial(l1_batch,smooth=False)
    else:
        error_func = partial(l1_batch,smooth=True)

    D = torch.tensor(np.loadtxt(args.matrix, delimiter=" ")).double()
    
    scale = D.max()/2
    
    M = D/scale
    
    A = (M*M)[None]
    
    if use_cuda:
        print("Using CUDA!")
        A = A.cuda()
    else:
        print("NOT using CUDA!")
    
    best_x = None
    best_cost = 1e15
    
    initial_pop = 2*torch.randn(args.pop_size*args.num_restarts, A.shape[1], 2).double()
    
    loss, x = optimize(lambda x: error_func(x,A), 
                        initial_pop=initial_pop, 
                        epochs=Timer(args.time),
                        num_populations=args.num_restarts,
                        shuffles=args.shuffles,
                        use_cuda=use_cuda, 
                        mut=(0.1,0.9),crossp=(0.3,0.7),
                        proj_to_domain=clamp,
                        prob_choosing_method='auto')
    
    if loss < best_cost:
        best_cost = loss
        best_x = x
        
    x = best_x.cpu()
    dx = torch.sqrt(pairwise_dist2_batch(x[None])).squeeze()
        
    x *= scale
    dx *= scale
        
    diff = D - dx
    error = torch.mean(torch.abs(diff)).item()
    
    np.set_printoptions(precision=6, suppress=True, linewidth=10000, floatmode='maxprec_equal')
    
    print("Solution found: \n",x.numpy())
    print("\n Distance Matrix: \n", dx.numpy())
    print("\n Original distance matrix: \n", D.numpy())
    
    diff = D - dx
    print("\n Difference: \n", diff.numpy())
    
    print(f"\n Avg difference: {error}")
    print(f"\n Worst difference: {torch.max(torch.abs(diff)).item()}")
    
