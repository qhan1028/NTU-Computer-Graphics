#
#   Background Replacer Utilities
#   Modified by Qhan
#

import numpy as np
import cv2


# Blur
def blur(im, mask=45):
    kernel = np.ones((mask, mask), np.float32) / (mask ** 2)
    return cv2.filter2D(np.array(im), -1, kernel)


# Gray only (BGR)
def gray(bgr_im, contrast=80, brightness=0.75):
    gray_im = cv2.cvtColor(bgr_im, cv2.COLOR_BGR2GRAY).astype(np.int)
    #F = (259 / 255) * (contrast + 255) / (259 - contrast)
    #result = (F * (gray_im - 128) + 128) * brightness
    return gray_im * brightness


def green(im):
    im[:, :, 0] = 0
    im[:, :, 1] = 255
    im[:, :, 2] = 0
    return im


def to3dim(im): return cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)

def rgb2bgr(im): return cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

def bgr2rgb(im): return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

def amap2im(amap): return (amap * 255).astype(np.uint8)

# Change saturation & brightness (BGR, HSV)
def change_sv(bgr_im, dsv):
    ds = dsv['ds']
    dv = dsv['dv']

    if len(bgr_im.shape) == 2:
        gray_im = np.clip(bgr_im.astype(int) + dv, 0, 255).astype(np.uint8)
        return gray_im
    
    hsv_im = cv2.cvtColor(bgr_im , cv2.COLOR_BGR2HSV)

    saturation = np.array(hsv_im[:, :, 1]).astype(int)
    brightness = np.array(hsv_im[:, :, 2]).astype(int)
    
    hsv_im[:, :, 1] = np.clip(saturation + ds, 0, 255).astype(np.uint8)
    hsv_im[:, :, 2] = np.clip(brightness + dv, 0, 255).astype(np.uint8)

    return cv2.cvtColor(hsv_im, cv2.COLOR_HSV2BGR)
    

# Blend bg, fg
def blend(fg, bg, amap, power=1, continuous=True, threshold=None):
    if power == 1:
        pass
    else:
        amap = amap ** power
    
    if continuous:
        pass
    else:
        if type(threshold) is float: # discrete blend
            amap = np.where(amap >= threshold, 1, 0)
        elif type(threshold) is list:
            cond = [amap < threshold[0], amap < threshold[1], amap <= 1.0]
            choice = [0, 0.6, 1]
            amap = np.select(cond, choice)
        else:
            pass
    
    fg_weight = amap
    bg_weight = 1 - fg_weight

    bg_channel = 1 if len(bg.shape) == 2 else 3
    amap_channel = 1 if len(amap.shape) == 2 else 3

    if amap_channel == 3 and bg_channel == 3:
        blend_im = fg * fg_weight + bg * bg_weight

    elif amap_channel == 1 and bg_channel == 3:
        blend_im = np.zeros_like(fg)
        blend_im[:, :, 0] = fg[:, :, 0] * fg_weight + bg[:, :, 0] * bg_weight
        blend_im[:, :, 1] = fg[:, :, 1] * fg_weight + bg[:, :, 1] * bg_weight
        blend_im[:, :, 2] = fg[:, :, 2] * fg_weight + bg[:, :, 2] * bg_weight

    elif amap_channel == 1 and bg_channel == 1:
        blend_im = np.zeros_like(fg)
        blend_im[:, :, 0] = fg[:, :, 0] * fg_weight + bg * bg_weight
        blend_im[:, :, 1] = fg[:, :, 1] * fg_weight + bg * bg_weight
        blend_im[:, :, 2] = fg[:, :, 2] * fg_weight + bg * bg_weight

    else:
        print('> [Info] Unable to blend, bg channel: %d, amap channel: %d' % (bg_channel, amap_channel))
    
    return blend_im.astype(np.uint8)

