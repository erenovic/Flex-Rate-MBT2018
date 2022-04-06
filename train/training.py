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
from utils import VimeoTrainDataset, KodakTestDataset
from utils import image_compress, save_model, load_optimizer
from utils import configure_seeds, configure_optimizer_parameters
from utils import load_model, Infographic

torch.backends.cudnn.benchmark = True


# Argument parser
parser = argparse.ArgumentParser()

# Hyperparameters, paths and settings are given
# prior the test
parser.add_argument("--project_name", type=str, default="ICIP2022_test")                        # Project name
parser.add_argument("--model_name", type=str, default="flex_rate_mbt")                          # Model name

parser.add_argument("--random_seed", type=int, default=None)                                    # Get the seeds if available
parser.add_argument("--torch_seed", type=int, default=None)

parser.add_argument("--train_path", type=str, default="/datasets/vimeo_septuplet/sequences/")   # Dataset paths
parser.add_argument("--val_path", type=str, default="/datasets/kodak/")
parser.add_argument("--total_train_step", type=int, default=2000000)                            # # of total iterations
parser.add_argument("--train_step", type=int, default=5000)                                     # # of iterations for recording
parser.add_argument("--learning_rate", type=float, default=1.e-4)                               # learning rate
parser.add_argument("--aux_learning_rate", type=float, default=1.e-3)
# parser.add_argument("--min_lr", type=float, default=1.e-7)                                    # min. learning rate
# parser.add_argument("--patience", type=int, default=20)                                       # scheduler patience
parser.add_argument("--batch_size", type=int, default=4)                                        # Batch size
parser.add_argument("--patch_size", type=int, default=256)                                      # Train patch sizes

parser.add_argument("--train_gop_size", type=int, default=1)                                    # Train gop sizes
parser.add_argument("--train_num_frames", type=int, default=1)                                  # Train number of frames
parser.add_argument("--val_num_frames", type=int, default=1)                                    # Val number of frames
parser.add_argument("--train_skip_frames", type=int, default=1)                                 # Train number of frames skipped

parser.add_argument("--i_save_dir", type=str, default="../mbt2018_gained_lr1e_4.pth")           # Save file for the bidir. compressor

parser.add_argument("--device", type=str, default="cuda")                                       # device "cuda" or "cpu"
parser.add_argument("--workers", type=int, default=4)                                           # number of workers

parser.add_argument("--i_pretrained", type=str, default="../mbt_2018_q8.pth")                   # Load model from this file

parser.add_argument("--cont_train", type=bool, default=False)                                   # load optimizer
parser.add_argument("--log_results", type=bool, default=True)                                   # Store results in log

args = parser.parse_args()

args.save_name = args.model_name

logging.basicConfig(filename= args.save_name + ".log", level=logging.INFO)

rng = configure_seeds(args.random_seed, args.torch_seed)

device = torch.device(args.device)

# CompressAI trade-off values (For each trade-off, we pick one above I-compressor quality mbt2018_mean)
# args.betas_mse = torch.tensor([0.0067*(255**2), 0.0250*(255**2), 0.0483*(255**2), 
#                                0.0932*(255**2)]).to(device)                                     # beta for rate-distortion trade-off
                               
args.betas_mse = torch.tensor([0.0250*(255**2), 0.0483*(255**2), 
                               0.0932*(255**2), 0.1800*(255**2)]).to(device)                    # beta for rate-distortion trade-off      

args.levels = args.betas_mse.shape[0]                                                           # Number of points on rate-distortion curve


# In[ ]:

def mbt_image_compress(im_batch, model, n, l=1.):
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


def train_one_step(im_batch, i_model, optimizer, aux_optimizer, betas, device):
    """
    im_batch: video frames of shape (b, c * gop_size, h, w)
    i_model: Image compressor model
    optimizer: Optimizer of the model
    aux_optimizer: Auxiliary optimizer for the entropy model
    betas: Rate-distortion tradeoff (distortion coeff.)
    device: cuda or cpu
    """

    b, _, h, w = im_batch.shape

    level = torch.arange(0, args.levels).to(device)
    
    dec, rate, _ = mbt_image_compress(im_batch, i_model, n=level, l=1)    
    
    dist_loss = calculate_distortion_loss(dec, im_batch, dim=(1, 2, 3))
    dist_loss = torch.mean(betas * dist_loss)
    rate_loss = torch.mean(rate)
    
    loss = dist_loss + rate_loss
    
    # AUXILIARY LOSS
    aux_loss = i_model.aux_loss()
    
    optimizer.zero_grad()

    loss.backward()
    aux_loss.backward()
    
    torch.nn.utils.clip_grad_norm_(i_model.parameters(), 1.0)

    optimizer.step()
    
    return dist_loss.item(), rate_loss.item(), loss.item()
    

def validate(i_model, betas, device, args):
    """
    test_loader: Test loader for UVG
    model: Composite B-frame compressor model
    betas: Rate-distortion tradeoff (rate coeff.)
    device: cuda or cpu
    """
    with torch.no_grad():
        
        rate_loss = 0
        dist_loss = 0
        
        psnr_dict = {k: 0 for k in range(args.levels)}
        size_dict = {k: 0 for k in range(args.levels)}
        frame_num_dict = {k: 0 for k in range(args.levels)}
        pixel_num_dict = {k: 0 for k in range(args.levels)}
   
        test_dataset = KodakTestDataset(args.val_path)   
        test_loader = DataLoader(test_dataset, batch_size=args.val_num_frames, shuffle=False, num_workers=args.workers)

        for x in test_loader:
            x = x.to(device)
            b, _, h, w = x.shape
            
            for level in range(args.levels):

                dec, rate, size = mbt_image_compress(x, i_model, n=[level], l=1.)

                dist_loss += betas[level] * calculate_distortion_loss(dec, x, dim=(0, 1, 2, 3))
                rate_loss += rate

                uint8_real = float_to_uint8(x[0, :, :h, :w])
                uint8_dec_out = float_to_uint8(dec[0, :, :h, :w])

                cur_psnr = PSNR(
                    MSE(uint8_dec_out.type(torch.float), uint8_real.type(torch.float)), 
                    data_range=255
                )

                psnr_dict[level] += cur_psnr
                size_dict[level] += size
                frame_num_dict[level] += 1
                pixel_num_dict[level] += uint8_real.shape[1] * uint8_real.shape[2]
                
    total_frames = sum(frame_num_dict.values(), 0.0)
    total_pixels = sum(pixel_num_dict.values(), 0.0)
    total_size = sum(size_dict.values(), 0.0)
    total_psnr = sum(psnr_dict.values(), 0.0)
    
    average_psnr_dict = {k: (v / frame_num_dict[k]).item() for k, v in psnr_dict.items()}
    average_bpp_dict = {k: (v / pixel_num_dict[k]).item() for k, v in size_dict.items()}
    
    average_psnr = total_psnr / total_frames
    average_loss = ((dist_loss + rate_loss) / total_frames) * args.val_num_frames
    average_bpp = total_size / total_pixels

    return average_loss.item(), average_psnr.item(), average_bpp.item(), average_psnr_dict, average_bpp_dict


def val_before_start(i_model, device, args):
    i_model = i_model.eval()
    
    avg_val_loss, avg_psnr, avg_bpp, avg_psnr_dict, avg_bpp_dict = validate(
        i_model=i_model,
        betas=args.betas_mse,
        device=device,
        args=args
    )

    # Log to logfile if wanted
    if args.log_results:
        logging.info("---- Validation Before Start ----")
        logging.info("Validation PSNR: " + str(avg_psnr))
        logging.info("Validation bpp: " + str(avg_bpp))
        logging.info("Validation loss: " + str(avg_val_loss))
        logging.info("---------------------------------")
        logging.info("-- PSNR per level --")
        for k, v in avg_psnr_dict.items():
            logging.info("Level " + str(k) + " PSNR: " + str(v))
        logging.info("-- bpp per level --")
        for k, v in avg_bpp_dict.items():
            logging.info("Level " + str(k) + " bpp: " + str(v))
        logging.info("*********************************")

# ### Main Function

# In[ ]:

def main(args):
        
    device = torch.device(args.device)

    # Build I model
    i_model = gained_mbt2018(N=192, M=320, n=4).to(device).float()
    
    if args.i_pretrained:
        checkpoint = torch.load(args.i_pretrained, map_location=device)
        i_model = load_model(i_model, checkpoint, exceptions=[])
  
    parameters, aux_parameters = configure_optimizer_parameters(i_model)
  
    optimizer = optim.Adam((p for (n, p) in parameters if p.requires_grad),
                           lr=args.learning_rate)
    aux_optimizer = optim.Adam((p for (n, p) in aux_parameters if p.requires_grad),
                               lr=args.aux_learning_rate)
    learning_rate = args.learning_rate
    
    i_model_parameters = filter(lambda p: p.requires_grad, i_model.parameters())
    i_params = sum([np.prod(p.size()) for p in i_model_parameters])
    
    if args.log_results:
        logging.info("I-compression Model Num. params: " + str(i_params))
        
    train_dataset = VimeoTrainDataset(
                        args.train_path, 
                        patch_size=args.patch_size,
                        gop_size=args.train_gop_size, 
                        skip_frames=args.train_skip_frames,
                        num_frames=args.train_num_frames,
                        rng=rng,
                        dtype="png"
                    )                    
    train_sampler = RandomSampler(train_dataset, replacement=True)

    infographic = Infographic()
    
    # VALIDATE BEFORE START
    val_before_start(i_model, device, args)
    
    time_start = time.perf_counter()
    
    # If we want to continue training using a checkpoint we load the number of iterations
    if args.cont_train:
        iteration = checkpoint["iter"]
    else:
        iteration = 0
        
    i_model = i_model.train()

    while iteration <= args.total_train_step:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.workers, drop_last=True)

        for gop_im_batch in train_loader:
         
            dist_loss, rate_loss, loss = train_one_step(
                im_batch=gop_im_batch.to(args.device).float(), 
                i_model=i_model,
                optimizer=optimizer, 
                aux_optimizer=aux_optimizer, 
                betas=args.betas_mse,
                device=device
            )
             
            infographic.update_train_info(dist_loss, rate_loss, loss)
            iteration += 1

            if iteration % args.train_step == 0:
                
                i_model = i_model.eval()
                
                avg_val_loss, avg_psnr, avg_bpp, avg_psnr_dict, avg_bpp_dict = validate(
                    i_model=i_model,
                    betas=args.betas_mse,
                    device=device,
                    args=args
                )

                infographic.update_val_info(avg_val_loss, avg_psnr, avg_bpp)

                time_end = time.perf_counter()
                duration = time_end - time_start

                if avg_val_loss < infographic.best_val_loss:
                    # Save every submodule of the model separately
                    save_model(
                        model=i_model, 
                        optimizer=optimizer, 
                        aux_optimizer=aux_optimizer, 
                        scheduler=None, 
                        num_iter=iteration,
                        exceptions=[], 
                        save_name= args.i_save_dir
                    )
                    
                    infographic.update_best_val_info()
                    
                    if args.log_results:
                        logging.info("********** NEW BEST! **********")
                
                    
                # Log to logfile if wanted
                if args.log_results:
                    logging.info("Iteration: " + str(iteration))
                    logging.info("Time: " + str(duration))
                    logging.info("Learning rate: " + str(learning_rate))
                    logging.info("Distortion loss: " + str(infographic.step_train_dist_loss / args.train_step))
                    logging.info("Rate loss: " + str(infographic.step_train_rate_loss / args.train_step))
                    logging.info("Train loss: " + str(infographic.step_train_loss / args.train_step))
                    logging.info("Validation PSNR: " + str(infographic.avg_psnr_dec))
                    logging.info("Validation bpp: " + str(infographic.avg_bpp))
                    logging.info("Validation loss: " + str(infographic.avg_val_loss))
                    logging.info("Best Validation loss: " + str(infographic.best_val_loss))
                    logging.info("PSNR at best Validation loss: " + str(infographic.psnr_dec_at_best_loss))
                    logging.info("bpp at best Validation loss: " + str(infographic.bpp_at_best_loss))
                    logging.info("---------------------------------")
                    logging.info("-- PSNR per level --")
                    for k, v in avg_psnr_dict.items():
                        logging.info("Level " + str(k) + " PSNR: " + str(v))
                    logging.info("-- bpp per level --")
                    for k, v in avg_bpp_dict.items():
                        logging.info("Level " + str(k) + " bpp: " + str(v))
                    logging.info("*********************************")

                infographic.zero_train_info()
                
                # Take the model back into training mode                                                                  
                i_model = i_model.train()
                   
                time_start = time.perf_counter()
              
            if iteration >= args.total_train_step:
                break
                

# In[ ]:


if __name__ == '__main__':
    main(args)

