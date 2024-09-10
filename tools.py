import numpy as np
import os
import shutil
from PIL import Image,  ImageFont, ImageDraw
import torch
mask = torch.tensor([[1, 1, 1, 0, 0, 0, 0, 1, 1]
                     , [1, 1, 0, 0, 0, 0, 0, 0, 1]
                     , [1, 0, 0, 0, 0, 0, 0, 0, 0]
                     , [0, 0, 0, 0, 0, 0, 0, 0, 0]
                     , [0, 0, 0, 0, 0, 0, 0, 0, 0]
                     , [1, 0, 0, 0, 0, 0, 0, 0, 0]
                     , [1, 1, 0, 0, 0, 0, 0, 0, 1]
                     , [1, 1, 1, 0, 0, 0, 0, 1, 1]])

def visualize(actions, rewards, final_rewards, final_guesses, ground_truths, inits, output_path, suffix, mses, vis=False):
    gt_path = '{}/{}/gt/'.format(output_path, suffix)
    if os.path.exists(gt_path):
        shutil.rmtree(gt_path)
    os.makedirs(gt_path)
    print('gt path', gt_path)
    reward_path = '{}/{}/reward/'.format(output_path, suffix)
    if os.path.exists(reward_path):
        shutil.rmtree(reward_path)
    os.makedirs(reward_path)
    print('reward path', reward_path)
    init_path = '{}/{}/init/'.format(output_path, suffix)
    if os.path.exists(init_path):
        shutil.rmtree(init_path)
    os.makedirs(init_path)
    print('init path', init_path)
    final_guess_path = '{}/{}/final_guess/'.format(output_path, suffix)
    if os.path.exists(final_guess_path):
        shutil.rmtree(final_guess_path)
    os.makedirs(final_guess_path)
    print('init path', init_path)
    index_path = '{}/{}/index/'.format(output_path, suffix) 
    if os.path.exists(index_path):
        shutil.rmtree(index_path)
    os.makedirs(index_path)
    print('index path', index_path)

    rewardst_path = '{}/{}/rewardst/'.format(output_path, suffix)
    if os.path.exists(rewardst_path):
        shutil.rmtree(rewardst_path)
    os.makedirs(rewardst_path)
    print('reward path', rewardst_path)
    initst_path = '{}/{}/initst/'.format(output_path, suffix)
    if os.path.exists(initst_path):
        shutil.rmtree(initst_path)
    os.makedirs(initst_path)
    vf_loc = torch.where(torch.rot90(mask,1,[1,0]) == 0)
    vf_loc = torch.stack((7-vf_loc[1],vf_loc[0]))
    for i, action in enumerate(actions):
        reward = rewards[i]
        final_guess = final_guesses[i]
        ground_truth = ground_truths[i]
        init = inits[i]
        reward_2d = np.ones((8, 9)) * 255
        init_2d = np.ones((8, 9)) * 255
        final_guesses_2d = np.ones((8, 9)) * 255
        index_2d = np.ones((8, 9)) * 255
        gt_2d = np.ones((8, 9)) * 255
        rw_steps, init_steps = [], []
        for j, (loc, init_val) in enumerate(action):
            x, y = vf_loc[0][loc], vf_loc[1][loc]
            reward_2d[x, y] = reward[x, y]
            init_2d[x, y] = init[x, y]
            final_guesses_2d[x, y] = final_guess[x,y]
            index_2d[x, y] = j
            gt_2d[x,y]=ground_truth[x, y]
            
            if vis:
                rw_steps.append(reward_2d.copy())
                init_steps.append(init_2d.copy())
        final_reward = final_rewards[i]
        output = vis_vf(gt_2d)
        output.save(os.path.join(gt_path, '{}_rw{}_{:.2f}.png'.format(i, final_reward, mses[i])))
    
        output = vis_vf(reward_2d)
        output.save(os.path.join(reward_path, '{}_rw{}.png'.format(i, final_reward)))

        output = vis_vf(init_2d)
        output.save(os.path.join(init_path, '{}_rw{}.png'.format(i, final_reward)))

        output = vis_vf(final_guesses_2d)
        output.save(os.path.join(final_guess_path, '{}_rw{}.png'.format(i, final_reward)))

        output = vis_vf(index_2d)
        output.save(os.path.join(index_path, '{}_rw{}.png'.format(i, final_reward)))
        if vis:
            for k, img in enumerate(rw_steps):
                output = vis_vf_step(img, init_color=125)
                output.save(
                    os.path.join(rewardst_path, '{}_{}_rw{}.png'.format(i, k, final_reward)))
            for k, img in enumerate(init_steps):
                output = vis_vf_step(img, init_color=125)
                output.save(
                    os.path.join(initst_path, '{}_{}_rw{}.png'.format(i, k, final_reward)))

def vis_vf(vf, is_int=True, font=None):
    font = ImageFont.truetype(
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf", 20)
    if is_int:
        vf = vf.astype(int)
    img = Image.new(mode="L",size=(288,256), color =255)
    draw = ImageDraw.Draw(img)
    scale = 32
    map_index = [3, 6, 7, 6, 5, 5, 6, 9]
    h,w = vf.shape
    for i in range(h):
        for j in range(w):
            if vf[i,j]==255:
                continue
            im = Image.new(mode="L", size=(32, 32),color=int(vf[i,j])*4)
            img.paste(im,(j*scale, i*scale))
            if font is None:
                draw.text((j*scale+10, i*scale+10),str(vf[i,j]) ,fill=255, align='center')
            else:
                if vf[i,j] < 10:
                    draw.text((j * scale + 10, i * scale + 9),str(vf[i, j]), fill=255, font=font)
                else:
                    draw.text((j * scale + 5, i * scale + 9),str(vf[i, j]), fill=255, font=font)
            draw.text((j*scale, i*scale), str(h*i+j-map_index[i]), fill=255)
    return img

def vis_vf_step(vf, is_int=True, font=None, init_color=255):
    font = ImageFont.truetype(
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf", 18)
    if is_int:
        vf = vf.astype(int)
    img = Image.new(mode="L",size=(288,256), color = init_color)
    draw = ImageDraw.Draw(img)
    scale = 32
    map_index = [3, 6, 7, 6, 5, 5, 6, 9]
    h,w = vf.shape
    for i in range(h):
        for j in range(w):
            if vf[i,j]==255 and mask[i,j]==0:
                continue
            if mask[i,j]==1:
                im = Image.new(mode="L", size=(32, 32), color=init_color)
                img.paste(im, (j * scale, i * scale))
            else:
                im = Image.new(mode="L", size=(32, 32),color=int(vf[i,j])*4)
                img.paste(im,(j*scale, i*scale))
                if font is None:
                    draw.text((j*scale+10, i*scale+10),str(vf[i,j]) ,fill=255)
                else:
                    if vf[i,j] < 10:
                        draw.text((j * scale + 10, i * scale + 9),str(vf[i, j]), fill=255, font=font)
                    else:
                        draw.text((j * scale + 5, i * scale + 9),str(vf[i, j]), fill=255, font=font)
                draw.text((j*scale, i*scale), str(h*i+j-map_index[i]), fill=255)
    return img
