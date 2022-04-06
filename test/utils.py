#!/usr/bin/env python
# coding: utf-8

# # 2. Utils

# In[ ]:


import torch
from torch import optim
import numpy as np
from natsort import natsorted
import glob
import random
import imageio
import math
import torch.nn as nn
import logging


# In[1]:


def normalize(tensor):
    norm = (tensor) / 255.
    return norm


# In[2]:


def float_to_uint8(image):
    clip = np.clip(image, 0, 1) * 255.
    im_uint8 = np.round(clip).astype(np.uint8).transpose(1, 2, 0)
    return im_uint8


# In[3]:


def MSE(gt, pred):
    mse = np.mean((gt - pred) ** 2)
    return mse


# In[4]:


def PSNR(mse, data_range):
    psnr = 10 * np.log10((data_range ** 2) / mse)
    return psnr


# In[5]:


def calculate_distortion_loss(out, real):
    """Mean Squared Error"""
    distortion_loss = torch.mean((out - real) ** 2)
    return distortion_loss


# In[6]:


def pad(im):
    """Padding to fix size at validation"""
    (b, c, w, h) = im.size()

    p1 = (64 - (w % 64)) % 64
    p2 = (64 - (h % 64)) % 64

    pad = nn.ReflectionPad2d(padding=(0, p2, 0, p1))
    return pad(im).squeeze(0)
        

# ### Training & Test Video & Image Datasets


from torch.utils.data import Dataset


def tensor_crop(frames, patch_size=256):
    """
    Crop frames according to the patch size
    Output is a numpy array
    """
    X_train = []
    sample_im = imageio.imread(frames[0])
    
    x = random.randint(0, sample_im.shape[1] - patch_size)
    y = random.randint(0, sample_im.shape[0] - patch_size)

    for k in range(len(frames)):

        img = imageio.imread(frames[k])
        img_cropped = img[y:y + patch_size, x:x + patch_size]
        img_cropped = img_cropped.transpose(2, 0, 1)
        
        if k == 0:
            img_concat = np.array(img_cropped)
        else:
            img_concat = np.concatenate((img_concat, img_cropped), axis=0)

    return img_concat


class VimeoTrainDataset(Dataset):
    """Dataset for custom vimeo"""

    def __init__(self, data_path, patch_size, gop_size, skip_frames, num_frames=3, dtype=".png"):
        """
        data_path: path to folders of videos,
        patch_size: size to crop for training,
        gop_size: GoP size,
        skip_frames: do we skip frames (int),
        num_frames: whether we limit the number of frames in the GoP,
        dype: png or jpeg
        """

        self.data_path = data_path

        # Pick the videos with sufficient resolution
        videos = []
        folders = natsorted(glob.glob(data_path + "*"))
        for folder in folders:
            videos += natsorted(glob.glob(folder + "/*"))
        
        self.videos = videos
        
        self.patch_size = patch_size
        self.gop_size = gop_size
        self.skip_frames = skip_frames
        self.dtype = dtype
        
        # How many frames to take
        if num_frames:
            self.num_frames = num_frames
        else:
            self.num_frames = (self.gop_size // self.skip_frames) + 1

        self.dataset_size = len(self.videos)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, item):
        video = self.videos[item]
        video_im_list = natsorted(glob.glob(video + "/*." + self.dtype))
        
        length = len(video_im_list)

        s = random.randint(0, length - 1 - (self.num_frames - 1) * self.skip_frames)
        video_split = video_im_list[s:s + self.skip_frames * self.num_frames:self.skip_frames]
        
        video_split = tensor_crop(video_split, self.patch_size)
        video_split = normalize(video_split)
        
        return video_split


class UVGTestDataset(Dataset):
    """Dataset for UVG"""

    def __init__(self, data_path, video_names, gop_size, skip_frames, test_size=2):
        """
        data_path: path to folders of videos,
        video_name: video name (e.g. beauty),
        skip_frames: do we skip frames (int, e.g. 1),
        """
        # Get the frame paths for each frame
        self.data_path = data_path
        self.skip_frames = skip_frames
        self.gop_size = gop_size
        self.test_size = test_size
        self.frames = []
        
        for video_name in video_names:
            video = data_path + video_name
            if test_size:
                frames = natsorted(glob.glob(video + "/*.png"))[:test_size*gop_size+1]
            else:
                frames = natsorted(glob.glob(video + "/*.png"))
            
            for idx, frame in enumerate(frames):
                self.frames.append(frame)
                if (idx % gop_size == 0) and (idx != 0) and (idx // gop_size != test_size):
                    self.frames.append(frame)
        
        self.dataset_size = len(self.frames)
        self.orig_img_size = imageio.imread(self.frames[0]).shape
        
    def __len__(self):
        return self.dataset_size

    def __getitem__(self, item):        
        frame = self.frames[item]

        im = imageio.imread(frame).transpose(2, 0, 1)
        im = normalize(torch.from_numpy(im)).unsqueeze(0)
        im = pad(im)

        return im
    

class KodakTestDataset(Dataset):
    """Dataset for Kodak"""

    def __init__(self, data_path):
        """
        data_path: path to folders of videos,
        video_name: video name (e.g. beauty),
        skip_frames: do we skip frames (int, e.g. 1),
        """
        # Get the frame paths for each frame
        self.data_path = data_path
        self.images = natsorted(glob.glob(self.data_path + "*.png"))
               
    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        im = self.images[item]
        
        im = imageio.imread(im).transpose(2, 0, 1)
        im = normalize(torch.from_numpy(im))
        
        return im


# ### I-Frame image compressor

# In[11]:


def image_compress(im, compressor, level, interval):
    # out = compressor(im, level)
    out = compressor(im, level, interval)
    dec = out["x_hat"]

    size_image = sum(
        (torch.log(likelihoods).sum() / (-math.log(2)))
        for likelihoods in out["likelihoods"].values()
    )

    return dec, size_image
    
    
#class Limiter():
#    """
#    Build a limiter class to limit PSNR or bpp
#    """
#    def __init__(self, limit_type, limit):
#        self.limit_type = limit_type
#        self.limit = limit
#        
#        self.prev_value = None
#        self.prev_level = None
#        self.prev_interval = None
#                
#    def decide(self, level, interval, psnr, bpp):
#        if limit_type == "psnr":




# ### Save and load pmodel

# In[12]:


def save_model(model, optimizer, aux_optimizer, scheduler, num_iter, exceptions, save_name="checkpoint.pth"):
    """
    Save a model with its optimizer, aux_optimizer, scheduler and # of iteration info.
    If some of them are not desired, give None as input instead of it
    """
    
    save_dict = {}
    if optimizer:
        save_dict["optimizer"] = optimizer.state_dict()
    if aux_optimizer:
        save_dict["aux_optimizer"] = aux_optimizer.state_dict()
    
    if scheduler:
        save_dict["scheduler"] = scheduler.state_dict()
        
    if num_iter:
        save_dict["iter"] = num_iter
    
    for child, module in model.named_children():
        # If we don't want to save a child, we skip it
        if child in exceptions:
            continue
        save_dict[child] = module.state_dict()
        logging.info("Saved " + child + " at " + save_name)
        
    torch.save(save_dict, save_name)


# In[14]:


def load_model(model, pretrained_dict, exceptions):
    """
    Load the model parameters from a dictionary. The dictionary must have key names same
    as the model attributes (which are submodules). The save_model() function is designed
    to be matching with this function.
    """
    
    model_child_names = [name for name, _ in model.named_children()]
    
    for name, submodule in pretrained_dict.items():
        # If we don't want to load a module, we skip it
        if name in exceptions:
            continue
            
        if name in model_child_names:
            message = getattr(model, name).load_state_dict(submodule)
            logging.info(name + ": " + str(message))
    return model


# In[14]:


def configure_optimizers(model, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""
    # Use list of tuples instead of dict to be able to later check the elements are unique and there is no intersection
    parameters = []
    aux_parameters = []
    parameter_dict = {}
    for name, param in model.named_parameters():
        parameter_dict[name] = param
        if not name.endswith(".quantiles"):
            parameters.append((name, param))
        else:
            aux_parameters.append((name, param))
    
    aux_param_set = set(p for n, p in aux_parameters)
    num_aux_params = sum([np.prod(p.size()) for p in aux_param_set])
    
    logging.info("There are " + str(num_aux_params) + " aux_parameters")
    
    # Make sure we don't have an intersection of parameters
    parameters_name_set = set(n for n,p in parameters)
    aux_parameters_name_set = set(n for n, p in aux_parameters)
    assert len(parameters) == len(parameters_name_set)
    assert len(aux_parameters) == len(aux_parameters_name_set)
    
    inter_params = parameters_name_set & aux_parameters_name_set
    union_params = parameters_name_set | aux_parameters_name_set
    assert len(inter_params) == 0
    assert len(union_params) - len(parameter_dict.keys()) == 0

    optimizer = optim.Adam((p for (n, p) in parameters if p.requires_grad),
                           lr=args.learning_rate)
    aux_optimizer = optim.Adam((p for (n, p) in aux_parameters if p.requires_grad),
                               lr=args.aux_learning_rate)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                                     patience=5, min_lr=args.min_lr)
    
    return optimizer, aux_optimizer, scheduler


# In[15]:

    
def load_optimizer(pretrained_dict, device, optimizer, aux_optimizer=None):
    """
    Load the optimizer parameters from a dictionary that was saved using the save_model() 
    function.
    """
    
    message = optimizer.load_state_dict(pretrained_dict["optimizer"])
    
    logging.info("Optimizer: " + str(message))
    
    if aux_optimizer:
        aux_optimizer.load_state_dict(pretrained_dict["aux_optimizer"])
      
        logging.info("Aux Optimizer: " + str(message))

    return optimizer, aux_optimizer


# ### Info passing during training and validation

# In[15]:

#class TestInfographic():
#    """
#    Build a logging class to save & load the test results
#    """
#    def __init__(self, levels_intervals, folder_names):
#        self.frame_types = ["I", "P", "B"]
#        
#        self.psnr_dict = {}
#        self.size_dict = {}
#        self.frame_num_dict = {}
#        self.pixel_num_dict = {}
#        
#        self.psnr_specs = {}
#        self.size_specs = {}
#        self.frame_specs = {}
#        self.pixel_specs = {}
#        
#        self.average_bpp_specs = {}
#        self.average_psnr_specs = {}
#        
#        for level, interval in levels_intervals:
#            self.psnr_dict[(level, interval)] = 0
#            self.size_dict[(level, interval)] = 0
#            self.frame_num_dict[(level, interval)] = 0
#            self.pixel_num_dict[(level, interval)] = 0
#            
#        for folder in folder_names:
#            self.psnr_specs[folder] = {}
#            self.size_specs[folder] = {}
#            self.frame_specs[folder] = {}
#            self.pixel_specs[folder] = {}
#            
#            for level, interval in levels_intervals:
#                self.psnr_specs[folder][(level, interval)] = 0
#                self.size_specs[folder][(level, interval)] = 0
#                self.frame_specs[folder][(level, interval)] = 0
#                self.pixel_specs[folder][(level, interval)] = 0
#                
#    def update(self, level, interval, folder, psnr, size, pixels):
#        self.psnr_dict[(level, interval)] += psnr
#        self.size_dict[(level, interval)] += size
#        self.frame_num_dict[(level, interval)] += 1
#        self.pixel_num_dict[(level, interval)] += pixels
#        
#        self.psnr_specs[folder][(level, interval)] += psnr
#        self.size_specs[folder][(level, interval)] += size
#        self.frame_specs[folder][(level, interval)] += 1
#        self.pixel_specs[folder][(level, interval)] += pixels
#        
#    def total(self):
#        self.total_frames = sum(self.frame_num_dict.values(), 0.0)
#        self.total_pixels = sum(self.pixel_num_dict.values(), 0.0)
#        self.total_size = sum(self.size_dict.values(), 0.0)
#        self.total_psnr = sum(self.psnr_dict.values(), 0.0)
#        
#    def video_level_average(self, video):
#        self.average_bpp_specs[video] = {k: (v / self.pixel_specs[video][k]).item() for k, v in self.size_specs[video].items()}
#        self.average_psnr_specs[video] = {k: (v / self.frame_specs[video][k]).item() for k, v in self.psnr_specs[video].items()}
#        
#    def level_average(self):
#        self.average_bpp_dict = {k: (v / self.pixel_num_dict[k]).item() for k, v in self.size_dict.items()}
#        self.average_psnr_dict = {k: (v / self.frame_num_dict[k]).item() for k, v in self.psnr_dict.items()}
#        
#        self.average_psnr = self.total_psnr / self.total_frames
#        self.average_bpp = self.total_size / self.total_pixels
#        
#    def frame_level_average(self, video):
#        
#    def print_per_level(self):
#        logging.info("-- PSNR per level --")
#        for k, v in self.average_psnr_dict.items():
#            logging.info("Level " + str(k) + " PSNR: " + str(v))
#        logging.info("-- bpp per level --")
#        for k, v in self.average_bpp_dict.items():
#            logging.info("Level " + str(k) + " bpp: " + str(v))
#            
#    def print_per_video_level(self, video):
#        logging.info("-- PSNR per video, level pair --")
#        for k, v in self.average_psnr_specs[video].items():
#            logging.info("Video: " + str(video) + ", Level: " + str(k) + " PSNR: " + str(v))
#        logging.info("-- bpp per video, level pair --")
#        for k, v in self.average_bpp_specs[video].items():
#            logging.info("Video: " + str(video) + ", Level: " + str(k) + " bpp: " + str(v))
        
import pandas as pd

class TestInfographic():
    """
    Build a logging class to save & load the test results
    """        
    def __init__(self, levels_intervals, folder_names):
        self.frame_types = ["I", "P", "B"]
        self.levels_intervals = levels_intervals
        self.folder_names = folder_names
        
        self.column_names = ["video", "level", "interval", "frame_type", "psnr", "size", "pixels"]
        self.frame_df = pd.DataFrame(columns=self.column_names)
                
    def update(self, frame_type, level, interval, video, psnr, size, pixels):
        data_point = {"video": video, "level": level, "interval": interval, 
                      "frame_type": frame_type, "psnr": psnr, "size": size, "pixels": pixels}
        
        self.frame_df = self.frame_df.append(data_point, ignore_index=True)
    
    def print_per_level(self):
        grouped = self.frame_df.groupby(["level", "interval"])
        
        self.totals_per_level = grouped.sum()
        self.num_frames_per_level = grouped.size()
        
        self.average_psnr_dict = {}
        self.average_bpp_dict = {}
        
        logging.info("----- Per Level -----")
        for (level_interval, sums), (_, num_frames) in zip(self.totals_per_level.iterrows(), 
                                                           self.num_frames_per_level.iteritems()):
            psnr = sums["psnr"]/num_frames
            bpp = sums["size"]/sums["pixels"]
            logging.info("Level " + str(level_interval) + " PSNR: " + str(psnr))
            logging.info("Level " + str(level_interval) + " bpp: " + str(bpp))
            
            self.average_psnr_dict[level_interval] = psnr
            self.average_bpp_dict[level_interval] = bpp
            
    def print_per_video_level(self, video):
        grouped = self.frame_df.groupby(["video", "level", "interval"])
        
        self.totals_per_video_level = grouped.sum().xs(video, level=0, drop_level=False)
        self.num_frames_per_video_level = grouped.size().xs(video, level=0, drop_level=False)
        
        logging.info("----- Per video, level pair -----")
        for (vid_level_interval, sums), (_, num_frames) in zip(self.totals_per_video_level.iterrows(), 
                                                               self.num_frames_per_video_level.iteritems()):
            video, level, interval = vid_level_interval
            logging.info("Video: " + str(video) + ", Level: " + str((level, interval)) + " PSNR: " + str(sums["psnr"]/num_frames))
            logging.info("Video: " + str(video) + ", Level: " + str((level, interval)) + " bpp: " + str(sums["size"]/sums["pixels"]))
        
    def print_per_frame_type_level(self):
        grouped = self.frame_df.groupby(["level", "interval", "frame_type"])
        
        self.totals_per_level_frame = grouped.sum()
        self.num_frames_per_level_frame = grouped.size()
        
        logging.info("----- Per frame type, level pair -----")
        for (level_interval_frame, sums), (_, num_frames) in zip(self.totals_per_level_frame.iterrows(), 
                                                                 self.num_frames_per_level_frame.iteritems()):
            level, interval, frame = level_interval_frame
            logging.info("Frame: " + str(frame) + ", Level: " + str((level, interval)) + " PSNR: " + str(sums["psnr"]/num_frames))
            logging.info("Frame: " + str(frame) + ", Level: " + str((level, interval)) + " bpp: " + str(sums["size"]/sums["pixels"]))
            
    def print_per_video_level_frame_type(self, video):
        grouped = self.frame_df.groupby(["video", "level", "interval", "frame_type"])
        
        self.totals_per_video_level_frame = grouped.sum().xs(video, level=0, drop_level=False)
        self.num_frames_per_video_level_frame = grouped.size().xs(video, level=0, drop_level=False)
        
        logging.info("----- Per video, level, frame type pair -----")
        for (video_level_interval_frame, sums), (_, num_frames) in zip(self.totals_per_video_level_frame.iterrows(), 
                                                                       self.num_frames_per_video_level_frame.iteritems()):
            video, level, interval, frame = video_level_interval_frame
            logging.info("Video: " + str(video) + ", Frame: " + str(frame) + ", Level: " + str((level, interval)) + " PSNR: " + str(sums["psnr"]/num_frames))
            logging.info("Video: " + str(video) + ", Frame: " + str(frame) + ", Level: " + str((level, interval)) + " bpp: " + str(sums["size"]/sums["pixels"]))
        
