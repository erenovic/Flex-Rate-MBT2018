#!/usr/bin/env python
# coding: utf-8


# # 1. Test code

# In[25]:


import torch
import numpy as np
import os
import sys
import warnings

warnings.filterwarnings('ignore')
import imageio

from compressai.zoo import mbt2018_mean

    
import argparse
import logging
import time
import random
import math
import matplotlib.pyplot as plt
from torch import optim

from torch.utils.data import DataLoader, RandomSampler

model_path = os.path.abspath('..')
sys.path.insert(1, model_path)

from i_model.model import gained_mbt2018

from utils import float_to_uint8, MSE, PSNR, calculate_distortion_loss
from utils import KodakTestDataset, load_model, image_compress
from utils import TestInfographic

torch.backends.cudnn.benchmark = True


# Argument parser
parser = argparse.ArgumentParser()

# Hyperparameters, paths and settings are given
# prior the test
parser.add_argument("--project_name", type=str, default="ICIP2022_test")                        # Project name
parser.add_argument("--model_name", type=str, default="flex_rate_test_new")              # Model name
          
parser.add_argument("--test_path", type=str, default="/datasets/kodak/")                        # Dataset paths

parser.add_argument("--test_num_frames", type=int, default=1)                                   # Test gop sizes

parser.add_argument("--device", type=str, default="cuda")                                       # device "cuda" or "cpu"
parser.add_argument("--workers", type=int, default=4)                                           # number of workers

parser.add_argument("--levels", type=int, default=4)                                            # Number of points on rate-distortion curve

parser.add_argument("--i_pretrained", type=str, default="../mbt2018_gained.pth")         # Load model from this file

parser.add_argument("--log_results", type=bool, default=True)                                   # Store results in log

args = parser.parse_args()

args.save_name = args.model_name

logging.basicConfig(filename= args.save_name + "_test.log", level=logging.INFO)

# args.levels_intervals = [(0, 1.), (0, 0.66), (0, 0.33), (1, 1.), (1, 0.66), (1, 0.33), (2, 1.), (2, 0.66), (2, 0.33), (3, 1.)]
#args.levels_intervals = [(0, 1.), (0, 0.75), (0, 0.5), (0, 0.25), 
#                         (1, 1.), (1, 0.75), (1, 0.5), (1, 0.25), 
#                         (2, 1.), (2, 0.75), (2, 0.5), (2, 0.25), (3, 1.)]

args.levels_intervals = []
for level in range(0, 3):
    for interval in np.arange(1., 0., -0.1):
        args.levels_intervals.append((level, interval))
        
args.levels_intervals.append((3, 1.))
        

# ### Test Function

# In[ ]:

def mbt_image_compress(im_batch, model, n, l):
    _, _, h, w = im_batch.shape
    num_pixels = h * w
    
    output = model(
        im_batch, 
        n=n,
        l=l,
    )
    
    size = sum(
        (torch.log(likelihoods).sum(dim=(1, 2, 3)) / (-math.log(2)))
        for likelihoods in output["likelihoods"].values()
    )
    rate = size / num_pixels
    
    dec = output["x_hat"]
    return dec, rate, size
    

def test(i_model, device, args):
    """
    test_loader: Test loader for UVG
    model: Composite B-frame compressor model
    betas: Rate-distortion tradeoff (rate coeff.)
    device: cuda or cpu
    """
    with torch.no_grad():
        
        psnr_dict = {k: 0 for k in args.levels_intervals}
        size_dict = {k: 0 for k in args.levels_intervals}
        frame_num_dict = {k: 0 for k in args.levels_intervals}
        pixel_num_dict = {k: 0 for k in args.levels_intervals}
   
        test_dataset = KodakTestDataset(args.test_path)
        test_loader = DataLoader(test_dataset, batch_size=args.test_num_frames, shuffle=False, num_workers=args.workers)

        for x in test_loader:
            x = x.to(device)
            b, _, h, w = x.shape
            
            for level, interval in args.levels_intervals:

                dec, rate, size = mbt_image_compress(x, i_model, n=[level], l=interval)

                uint8_real = float_to_uint8(x[0, :, :h, :w].cpu().numpy())
                uint8_dec_out = float_to_uint8(dec[0, :, :h, :w].cpu().numpy())

                cur_psnr = PSNR(
                    MSE(uint8_dec_out.astype(np.float32), uint8_real.astype(np.float32)), 
                    data_range=255
                )

                psnr_dict[(level, interval)] += cur_psnr
                size_dict[(level, interval)] += size.item()
                frame_num_dict[(level, interval)] += 1
                pixel_num_dict[(level, interval)] += uint8_real.shape[0] * uint8_real.shape[1]
                
    total_frames = sum(frame_num_dict.values(), 0.0)
    total_pixels = sum(pixel_num_dict.values(), 0.0)
    total_size = sum(size_dict.values(), 0.0)
    total_psnr = sum(psnr_dict.values(), 0.0)
    
    average_psnr_dict = {k: (v / frame_num_dict[k]) for k, v in psnr_dict.items()}
    average_bpp_dict = {k: (v / pixel_num_dict[k]) for k, v in size_dict.items()}
    
    average_psnr = total_psnr / total_frames
    average_bpp = total_size / total_pixels

    return average_psnr, average_bpp, average_psnr_dict, average_bpp_dict

# ### Main Function

# In[ ]:
# We just train the b-coding model


def plot_with_dots(bpp, psnr, label, c):
    plt.plot(bpp, psnr, ".", color=c, markersize=15, label=label)
    plt.plot(bpp, psnr, "-", color=c)


def main(args):

    device = torch.device(args.device)
    
    # Build I model
    i_model = gained_mbt2018(N=192, M=320, n=4).to(device).float()
    
    if args.i_pretrained:
        checkpoint = torch.load(args.i_pretrained, map_location=device)
        i_model = load_model(i_model, checkpoint, exceptions=[])
           
    # print(model)
    
    i_model = i_model.eval()

    time_start = time.perf_counter()

    average_psnr, average_bpp, average_psnr_dict, average_bpp_dict = test(
        i_model=i_model, 
        device=device, 
        args=args
    )
    
    time_end = time.perf_counter()
    duration = time_end - time_start
                    
    # Log to logfile if wanted
    if args.log_results:
        logging.info("-------------------------------")
    
        logging.info("Average PSNR: " + str(average_psnr))
        logging.info("Average bpp: " + str(average_bpp))
        logging.info("Duration (sec): " + str(duration))
        
        logging.info("---------------------------------")
        
        psnrs = list(average_psnr_dict.values())
        bpps = list(average_bpp_dict.values())
        
        logging.info("PSNRs: %s" % str(psnrs))
        logging.info("bpps: %s" % str(bpps))

        fig = plt.figure(figsize= (13,13))
  
        HCVR_psnr = [31.622085547563817, 31.88118613390959, 32.121417874343365, 
                     32.359775139187136, 32.61165956177868, 32.87280250262936, 
                     33.14926935203545, 33.44890018396181, 33.87163380593382, 
                     34.32590662037793, 34.81581849379523, 34.943057727123666, 
                     35.07680780796605, 35.229856795264205, 35.37573338647558, 
                     35.51260525449112, 35.647828051491686, 35.77458076374411, 
                     35.89466694534939, 36.01216812651749, 36.1296433536629, 
                     36.231436671487764, 36.36660020721107, 36.48525808812263, 
                     36.590966297824984, 36.69154033316701, 36.786670592804846, 
                     36.88150712005423, 36.97343839032113, 37.0622648235922, 37.14558007113424]
        HCVR_bpp = [0.3720136880874634, 0.39901456236839294, 0.4520415663719177, 
                    0.5499646067619324, 0.5852422118186951, 0.6185129284858704, 
                    0.648993194103241, 0.6784154176712036, 0.6435035467147827, 
                    0.6758296489715576, 0.7246824502944946, 0.7460921406745911, 
                    0.7673801779747009, 0.7889041304588318, 0.8114141225814819, 
                    0.8341883420944214, 0.8567377328872681, 0.8792592287063599, 
                    0.9020118117332458, 0.925216019153595, 0.9489445686340332, 
                    0.9730197191238403, 0.9975293874740601, 1.028427243232727, 
                    1.0564765930175781, 1.0779005289077759, 1.0988943576812744, 
                    1.1217992305755615, 1.1462048292160034, 1.1711864471435547, 1.1972018480300903]
        
        plot_with_dots(
            bpps,
            psnrs,
            args.model_name,
            c="blue"
        )
        plot_with_dots(
            HCVR_bpp,
            HCVR_psnr,
            "HCVR",
            c="red"
        )
        
        plt.grid()
        plt.ylabel("PSNR (dB)")
        plt.xlabel("bpp (bit/sec)")
        plt.legend()
        
        plt.savefig(args.model_name + ".png")
        


# In[ ]:


if __name__ == '__main__':
    main(args)


# In[ ]:




