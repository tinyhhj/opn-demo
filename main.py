from .models.OPN import OPN
from .models.TCN import TCN
from .models.helpers import overlay_davis
import torch.nn as nn
import torch
import os
import numpy as np
from PIL import Image
import cv2
from datetime import datetime
import torchvision.transforms as transforms
thickness = int(os.getenv('OPN_THICKNESS') or 8)
# result make dir
date = datetime.today().strftime("%Y/%m/%d")

root = os.getenv('OPN_ROOT') or os.getenv('UPLOAD_PATH') or 'checkpoints/opn'
results = os.path.join(root,date,'results')
env = os.getenv('FLASK_ENV') or 'PROD'

model = nn.DataParallel(OPN(thickness=thickness))
if torch.cuda.is_available():
    model.cuda()
model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__),'OPN.pth')), strict=False)
model.eval()

pp_model = nn.DataParallel(TCN())
if torch.cuda.is_available():
    pp_model.cuda()
pp_model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__),'TCN.pth')), strict=False)
pp_model.eval()

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
def batch(iterable, n=1):
    l = len(iterable)
    for i in range(0,l,n):
        yield iterable[i:min(i+n,l)]

def inference(images, masks, filename,memory_period = 1, batch_size=10, frame_path=None):
    '''
    :param images: frames T C H W
    :param masks: masks T C H W
    :param memory_period: image 1 video over 1
    :return: image, video inpainting result
    '''
    # input_size = int(os.getenv('INPUT_SIZE'))
    global pp_model, model
    B,_,H, W = images.size()
    # at least we need 2 images
    T = 2 if B == 1 else B
    if B == 1:
        images = torch.cat([images, images])
        masks = torch.cat([masks,masks])
    mt = transforms.Compose([
        transforms.ToPILImage(),
    ])
    orig_image = mt(images[0])
    orig_mask = mt(masks[0])

    for ii in range(0,len(images), batch_size):
        # total is batch_size but last can be small than batch size
        batch_images = images[ii: min(ii+batch_size, len(images))]
        batch_masks = masks[ii: min(ii+batch_size, len(masks))]
        T = min(len(batch_images),len(batch_masks))
        orig_frames = np.empty((T , H, W, 3), dtype=np.float32)
        holes = np.empty((T , H, W, 1), dtype=np.float32)
        dists = np.empty((T , H, W, 1), dtype=np.float32)

        for i in range(T):
            #### rgb
            raw_frame = batch_images[i].permute((1,2,0)).float().numpy()
            raw_frame = cv2.resize(raw_frame, dsize=(W, H), interpolation=cv2.INTER_CUBIC)
            orig_frames[i] = raw_frame
            # frames[i + 1] = raw_frame.copy()
            # horizontal_flip(frames, i, i + 2)
            # shift_down(frames, i + 2, i + 3, stride)
            # shift_up(frames, i, i + 2, stride)
            # shift_right(frames, i, i + 3, stride)
            # shift_left(frames, i, i + 4, stride)

            #### mask
            raw_mask = batch_masks[i].permute((1,2,0)).byte().numpy()
            raw_mask = (raw_mask > 0.5).astype(np.uint8)
            raw_mask = cv2.resize(raw_mask, dsize=(W, H), interpolation=cv2.INTER_NEAREST)
            raw_mask = cv2.dilate(raw_mask, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)))
            holes[i, :, :, 0] = raw_mask.astype(np.float32)
            # holes[i + 1, :, :, 0] = raw_mask.astype(np.float32).copy()
            # horizontal_flip(holes, i+1, i + 2)
            # shift_down(holes, i + 2, i + 3, stride)
            # shift_up(holes, i, i + 2, stride)
            # shift_right(holes, i, i + 3, stride)
            # shift_left(holes, i, i + 4, stride)

            #### dist
            dists[i, :, :, 0] = cv2.distanceTransform(raw_mask, cv2.DIST_L2, maskSize=5)
        # dists[i + 1, :, :, 0] = cv2.distanceTransform(raw_mask, cv2.DIST_L2, maskSize=5).copy()
        # horizontal_flip(dists, i+1, i + 2)
        # shift_down(dists, i + 2, i + 3, stride)
        # shift_up(dists, i, i + 2, stride)
        # shift_right(dists, i, i + 3, stride)
        # shift_left(dists, i, i + 4, stride)
        frames = torch.from_numpy(np.transpose(orig_frames, (3, 0, 1, 2)).copy()).float()
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
        midx = list(range(0, T, memory_period))
        comp_size = list(frames.size())
        # 왜했는지 기억 x 나중에 기록
        comp_size[2] += 1
        comps = torch.zeros(comp_size)
        # 전 batch에서 기록해둔 마지막 comps를 로드
        if ii != 0:
            comps[:,:,-1] = last
        with torch.no_grad():
            mkey, mval, mhol = model(frames[:, :, midx], valids[:, :, midx], dists[:, :, midx])
        original_est = None
        for f in range(T):
            # memory selection
            ridx = [i for i in range(len(midx)) if i*memory_period != f]  # memory minus self
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
            comps[:,:,f] = comp
            # visualize..
            est = (comp[0].permute(1, 2, 0).detach().cpu().numpy() * 255.).astype(np.uint8)
            mask_3d = (np.stack([orig_mask]*3,2) / 255).astype(np.uint8)
            # cv2.GaussianBlur
            # est = cv2.bilateralFilter(est*mask_3d,5,75,75)
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
            if memory_period == 1:
                # only image processing save result images
                time = datetime.today().strftime('%H_%M_%S')
                Image.fromarray(est).save(os.path.join(results,f'{time}_{filename}_{f}.jpg'))
        if ii == 0:
            comps[:,:,-1] = comps[:,:,ii].clone()
        last = comps[:,:,T-1].clone()

        if memory_period > 1 :
            hidden = None
            # only video processing do video post process
            for f in range(T):
                with torch.no_grad():
                    comps[:, :, f], hidden = pp_model(comps[:, :, f - 1], holes[:, :, f - 1], comps[:, :, f], holes[:, :, f], hidden)

            for f in range(T):
                # visualize..
                est = (comps[0, :, f].permute(1, 2, 0).detach().cpu().numpy() * 255.).astype(np.uint8)
                true = (orig_frames[f]*255).astype(np.uint8)  # h,w,3
                mask = (dists[0, 0, f].detach().cpu().numpy() > 0).astype(np.uint8)  # h,w,1
                if env == 'PROD':
                    canvas = est
                    target_image = Image.open(os.path.join(frame_path,f'{ii+f:05d}.jpg'))
                    canvas = cv2.resize(canvas,target_image.size, interpolation=cv2.INTER_CUBIC)
                    mask = cv2.resize(mask,target_image.size, interpolation=cv2.INTER_NEAREST)
                    target_image = np.array(target_image)
                    canvas = target_image * (1- mask[:,:,None]) + mask[:,:,None] * canvas
                else:
                    ov_true = overlay_davis(true, mask, colors=[[0, 0, 0], [128, 0, 0]], cscale=2, alpha=0.4)
                    canvas = np.concatenate([ov_true, est], axis=0)

                save_path = os.path.join(results, filename)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                canvas = Image.fromarray(canvas)
                canvas.save(os.path.join(save_path, '{:05d}.jpg'.format(ii+f)))
    return Image.fromarray(original_est)
