import cv2
import math
import random
import colorsys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from .utils import bounding_box

####
def gen_colors(N, random=True, bright=True):
    """
    Generate colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    if (random): random.shuffle(colors)
    return colors

####
def visualize_instances(mask, canvas=None, color_info=None):
    """
    Args:
        mask: array of NW
    Return:
        Image with the instance overlaid
    """
    if color_info is not None:
        num_colors = color_info[0]

    canvas = np.full(mask.shape + (3,), 200, dtype=np.uint8) \
                if canvas is None else np.copy(canvas)

    insts_list = list(np.unique(mask)) # [0,1,2,3,4,..,820]
    insts_list.remove(0) # remove background ?? is 0 (first elem) always background? # [1,2,3,4,..820]

    if num_colors is None:
        inst_colors = gen_colors(len(insts_list))
        inst_colors = np.array(inst_colors) * 255

   # assuming that colors and inst_colors are sorted equally
    if num_colors is not None:
        unique_colors = np.array(gen_colors(num_colors, random=False)) * 255

    for idx, inst_id in enumerate(insts_list):
        inst_color = inst_colors[idx] if num_colors is None else unique_colors[int(color_info[1][idx][0])]

        inst_map = np.array(mask == inst_id, np.uint8)
        y1, y2, x1, x2  = bounding_box(inst_map)
        y1 = y1 - 2 if y1 - 2 >= 0 else y1
        x1 = x1 - 2 if x1 - 2 >= 0 else x1
        x2 = x2 + 2 if x2 + 2 <= mask.shape[1] - 1 else x2
        y2 = y2 + 2 if y2 + 2 <= mask.shape[0] - 1 else y2
        inst_map_crop = inst_map[y1:y2, x1:x2]
        inst_canvas_crop = canvas[y1:y2, x1:x2]
        contours = cv2.findContours(inst_map_crop, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # For opencv-python==4.1.0.25
        # cv2.drawContours(inst_canvas_crop, contours[0], -1, inst_color, 2)

        cv2.drawContours(inst_canvas_crop, contours[1], -1, inst_color, 2)
        canvas[y1:y2, x1:x2] = inst_canvas_crop
    return canvas

####
def gen_figure(imgs_list, titles, fig_inch, shape=None,
                share_ax='all', show=False, colormap=plt.get_cmap('jet')):

    num_img = len(imgs_list)
    if shape is None:
        ncols = math.ceil(math.sqrt(num_img))
        nrows = math.ceil(num_img / ncols)
    else:
        nrows, ncols = shape

    # generate figure
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                        sharex=share_ax, sharey=share_ax)
    axes = [axes] if nrows == 1 else axes

    # not very elegant
    idx = 0
    for ax in axes:
        for cell in ax:
            cell.set_title(titles[idx])
            cell.imshow(imgs_list[idx], cmap=colormap)
            cell.tick_params(axis='both',
                            which='both',
                            bottom='off',
                            top='off',
                            labelbottom='off',
                            right='off',
                            left='off',
                            labelleft='off')
            idx += 1
            if idx == len(titles):
                break
        if idx == len(titles):
            break

    fig.tight_layout()
    return fig
####
