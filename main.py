from .models.OPN import OPN
import torch.nn as nn
import torch
import os
import numpy as np
from PIL import Image
import cv2
from datetime import datetime
import torchvision.transforms as transforms
thickness = int(os.getenv('OPN_THICKNESS')) or 8
# result make dir
root = os.getenv('OPN_ROOT') or 'checkpoints/opn'
results = os.path.join(root,'results')

model = nn.DataParallel(OPN(thickness=thickness))
if torch.cuda.is_available():
    model.cuda()
model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__),'OPN.pth')), strict=False)
model.eval()

if not os.path.exists(results):
    os.makedirs(results)
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
def inference(image, mask):
    stride = 10
    T = aug = 4
    input_size = int(os.getenv('INPUT_SIZE'))
    H, W = input_size, input_size
    frames = np.empty((T * aug, H, W, 3), dtype=np.float32)
    holes = np.empty((T * aug, H, W, 1), dtype=np.float32)
    dists = np.empty((T * aug, H, W, 1), dtype=np.float32)
    mt = transforms.Compose([
        transforms.Resize((input_size, input_size)),
    ])
    orig_image = mt(image)
    orig_mask = mt(mask)
    for i in range(T // aug):
        #### rgb
        raw_frame = np.array(image) / 255.
        raw_frame = cv2.resize(raw_frame, dsize=(W, H), interpolation=cv2.INTER_CUBIC)
        frames[i] = raw_frame
        frames[i + 1] = raw_frame.copy()
        horizontal_flip(frames, i, i + 2)
        shift_down(frames, i + 2, i + 3, stride)
        # shift_up(frames, i, i + 2, stride)
        # shift_right(frames, i, i + 3, stride)
        # shift_left(frames, i, i + 4, stride)

        #### mask
        raw_mask = np.array(mask, dtype=np.uint8)
        raw_mask = (raw_mask > 0.5).astype(np.uint8)
        raw_mask = cv2.resize(raw_mask, dsize=(W, H), interpolation=cv2.INTER_NEAREST)
        raw_mask = cv2.dilate(raw_mask, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)))
        holes[i, :, :, 0] = raw_mask.astype(np.float32)
        holes[i + 1, :, :, 0] = raw_mask.astype(np.float32).copy()
        horizontal_flip(holes, i, i + 2)
        shift_down(holes, i + 2, i + 3, stride)
        # shift_up(holes, i, i + 2, stride)
        # shift_right(holes, i, i + 3, stride)
        # shift_left(holes, i, i + 4, stride)

        #### dist
        dists[i, :, :, 0] = cv2.distanceTransform(raw_mask, cv2.DIST_L2, maskSize=5)
        dists[i + 1, :, :, 0] = cv2.distanceTransform(raw_mask, cv2.DIST_L2, maskSize=5).copy()
        horizontal_flip(dists, i, i + 2)
        shift_down(dists, i + 2, i + 3, stride)
        # shift_up(dists, i, i + 2, stride)
        # shift_right(dists, i, i + 3, stride)
        # shift_left(dists, i, i + 4, stride)

    frames = torch.from_numpy(np.transpose(frames, (3, 0, 1, 2)).copy()).float()
    holes = torch.from_numpy(np.transpose(holes, (3, 0, 1, 2)).copy()).float()
    dists = torch.from_numpy(np.transpose(dists, (3, 0, 1, 2)).copy()).float()
    # remove hole
    frames = frames * (1 - holes) + holes * torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
    # valids area
    valids = 1 - holes
    # unsqueeze to batch 1
    frames = frames.unsqueeze(0)
    holes = holes.unsqueeze(0)
    dists = dists.unsqueeze(0)
    valids = valids.unsqueeze(0)

    # memory encoding
    midx = list(range(0, T))
    with torch.no_grad():
        mkey, mval, mhol = model(frames[:, :, midx], valids[:, :, midx], dists[:, :, midx])
    original_est = None
    for f in range(T):
        # memory selection
        ridx = [i for i in range(len(midx)) if i != f]  # memory minus self
        fkey, fval, fhol = mkey[:, :, ridx], mval[:, :, ridx], mhol[:, :, ridx]
        # inpainting..
        for r in range(999):
            if r == 0:
                comp = frames[:, :, f]
                dist = dists[:, :, f]
            with torch.no_grad():
                comp, dist = model(fkey, fval, fhol, comp, valids[:, :, f], dist)

            # update
            comp, dist = comp.detach(), dist.detach()
            if torch.sum(dist).item() == 0:
                break

        # visualize..
        est = (comp[0].permute(1, 2, 0).detach().cpu().numpy() * 255.).astype(np.uint8)
        mask_3d = (np.stack([orig_mask]*3,2) / 255).astype(np.uint8)
        est = np.array(orig_image) * (1 - mask_3d) + est * mask_3d
        if f == 0:
            original_est = est
        # true = (frames[0, :, f].permute(1, 2, 0).detach().cpu().numpy() * 255.).astype(np.uint8)  # h,w,3
        # mask = (dists[0, 0, f].detach().cpu().numpy() > 0).astype(np.uint8)  # h,w,1
        # ov_true = overlay_davis(true, mask, colors=[[0, 0, 0], [100, 100, 0]], cscale=2, alpha=0.4)

        # canvas = np.concatenate([ov_true, est], axis=0)
        # save_path = os.path.join(os.path.join(root,'results'))
        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)
        # canvas = Image.fromarray(canvas)
        # canvas.save(os.path.join(save_path, 'res_{}.jpg'.format(f)))
        filename = os.path.splitext(image.filename)
        Image.fromarray(est).save(os.path.join(results,f'{datetime.today().strftime("%Y_%m_%d_%H_%M")}_{filename[0]}_{f}{filename[1]}'))
    return Image.fromarray(original_est)

        # print('Results are saved: ./{}'.format(save_path))