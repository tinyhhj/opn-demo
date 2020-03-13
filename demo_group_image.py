# -*- coding: utf-8 -*-

from __future__ import division
import torch

import torch.nn as nn


# general libs
import cv2
from PIL import Image
import numpy as np

import sys
import argparse



### My libs
# sys.path.append('utils/')
# sys.path.append('models/')
from .utils.helpers import *
from .models.OPN import OPN

def get_arguments():
    parser = argparse.ArgumentParser(description="args")
    parser.add_argument("--input", type=str, default='3e91f10205_2', required=True)
    return parser.parse_args()
args = get_arguments()
aug = 4
seq_name = args.input
dir = os.path.join('Image_inputs',args.input)
num = len([f for f in os.listdir(dir) if f.startswith('gt')]) * aug

#################### Load image
T, H, W = num, 240, 424
frames = np.empty((T*aug, H, W, 3), dtype=np.float32)
holes = np.empty((T*aug, H, W, 1), dtype=np.float32)
dists = np.empty((T*aug, H, W, 1), dtype=np.float32)


def shift_down(frames,orig,copy,stride):
    frames[copy, :stride] = frames[orig, 0]
    frames[copy, stride:] = frames[orig, :-stride]
def shift_up(frames,orig,copy,stride):
    frames[copy, :-stride] = frames[orig, stride:]
    frames[copy, -stride:] = frames[orig, -1]
def shift_right(frames,orig,copy,stride):
    # print(frames[copy,:,:stride].shape, np.stack([frames[orig,:,0]]*stride,1).shape)
    frames[copy,:,:stride] = np.stack([frames[orig,:,0]],1)
    frames[copy,:,stride:] = frames[orig,:,:-stride]
def shift_left(frames,orig,copy,stride):
    frames[copy, :, :-stride] = frames[orig, :, stride:]
    frames[copy, :, -stride:] = np.stack([frames[orig, :, -1]]*stride,1)
def horizontal_flip(frames,orig,copy):
    frames[copy] = frames[orig,:,::-1,...]


stride = 30
for i in range(T // aug):
    #### rgb
    img_file = os.path.join('Image_inputs', seq_name, 'gt_{}.jpg'.format(i))
    raw_frame = np.array(Image.open(img_file).convert('RGB'))/255.
    raw_frame = cv2.resize(raw_frame, dsize=(W, H), interpolation=cv2.INTER_CUBIC)
    frames[i] = raw_frame
    frames[i+1] = raw_frame.copy()
    horizontal_flip(frames,i,i+2)
    shift_down(frames, i+2, i + 3, stride)
    # shift_up(frames, i, i + 2, stride)
    # shift_right(frames, i, i + 3, stride)
    # shift_left(frames, i, i + 4, stride)

    #### mask
    mask_file = os.path.join('Image_inputs', seq_name, 'mask_{}.png'.format(i))
    raw_mask = np.array(Image.open(mask_file).convert('P'), dtype=np.uint8)
    raw_mask = (raw_mask > 0.5).astype(np.uint8)
    raw_mask = cv2.resize(raw_mask, dsize=(W, H), interpolation=cv2.INTER_NEAREST)
    raw_mask = cv2.dilate(raw_mask, cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3)))
    holes[i,:,:,0] = raw_mask.astype(np.float32)
    holes[i+1,:,:,0] = raw_mask.astype(np.float32).copy()
    horizontal_flip(holes,i,i+2)
    shift_down(holes, i+2, i + 3, stride)
    # shift_up(holes, i, i + 2, stride)
    # shift_right(holes, i, i + 3, stride)
    # shift_left(holes, i, i + 4, stride)

    #### dist
    dists[i,:,:,0] = cv2.distanceTransform(raw_mask, cv2.DIST_L2, maskSize=5)
    dists[i+1,:,:,0] = cv2.distanceTransform(raw_mask, cv2.DIST_L2, maskSize=5).copy()
    horizontal_flip(dists, i, i + 2)
    shift_down(dists, i+2, i + 3, stride)
    # shift_up(dists, i, i + 2, stride)
    # shift_right(dists, i, i + 3, stride)
    # shift_left(dists, i, i + 4, stride)

frames = torch.from_numpy(np.transpose(frames, (3, 0, 1, 2)).copy()).float()
holes = torch.from_numpy(np.transpose(holes, (3, 0, 1, 2)).copy()).float()
dists = torch.from_numpy(np.transpose(dists, (3, 0, 1, 2)).copy()).float()
# remove hole
frames = frames * (1-holes) + holes*torch.tensor([0.485, 0.456, 0.406]).view(3,1,1,1)
# valids area
valids = 1-holes
# unsqueeze to batch 1
frames = frames.unsqueeze(0)
holes = holes.unsqueeze(0)
dists = dists.unsqueeze(0)
valids = valids.unsqueeze(0)


#################### Load Model
model = nn.DataParallel(OPN())
if torch.cuda.is_available():
    model.cuda()
model.load_state_dict(torch.load(os.path.join('OPN.pth')), strict=False)
model.eval() 


################### Inference
# memory encoding 
midx = list( range(0, T) )
with torch.no_grad():
    mkey, mval, mhol = model(frames[:,:,midx], valids[:,:,midx], dists[:,:,midx])

for f in range(T):
    # memory selection
    ridx = [i for i in range(len(midx)) if i != f] # memory minus self
    fkey, fval, fhol = mkey[:,:,ridx], mval[:,:,ridx], mhol[:,:,ridx]
    # inpainting..
    for r in range(999): 
        if r == 0:
            comp = frames[:,:,f]
            dist = dists[:,:,f]
        with torch.no_grad(): 
            comp, dist = model(fkey, fval, fhol, comp, valids[:,:,f], dist)
        
        # update
        comp, dist = comp.detach(), dist.detach()
        if torch.sum(dist).item() == 0:
            break
        
    # visualize..
    est = (comp[0].permute(1,2,0).detach().cpu().numpy() * 255.).astype(np.uint8)
    true = (frames[0,:,f].permute(1,2,0).detach().cpu().numpy() * 255.).astype(np.uint8) # h,w,3
    mask = (dists[0,0,f].detach().cpu().numpy() > 0).astype(np.uint8) # h,w,1
    ov_true = overlay_davis(true, mask, colors=[[0,0,0],[100,100,0]], cscale=2, alpha=0.4)

    canvas = np.concatenate([ov_true, est], axis=0)
    save_path = os.path.join('Image_results', seq_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    canvas = Image.fromarray(canvas)
    canvas.save(os.path.join(save_path, 'res_{}.jpg'.format(f)))

print('Results are saved: ./{}'.format(save_path))
