import os
import math

import argparse
import importlib
import json
import requests

from tempfile import NamedTemporaryFile
from datetime import datetime
from collections import deque

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

from scipy.ndimage import filters, measurements
from scipy.ndimage.morphology import (binary_dilation, binary_fill_holes,
                                      distance_transform_cdt,
                                      distance_transform_edt)
from skimage.morphology import remove_small_objects, watershed

from misc.viz_utils import visualize_instances
from misc.utils import get_inst_centroid
from metrics.stats_utils import remap_label

# from scripts.viz_utils import visualize_instances
# import scripts.process_utils as proc_utils

class InfererURL():
    """
    Make sure that ENDPOINT and CUDA_VISIBLE_DEVICES are set up.
    
    input_img argument can be PIL image, numpy array or just path to .png file.
    """

    def __init__(self, input_img, save_dir):
        # self.nr_types = 6
        # self.input_shape = [256, 256]
        # self.mask_shape = [164, 164] 
        # self.input_norm  = True

        # input_img as PIL
        # values for np_hv model graph
        self.server_url = os.environ['ENDPOINT'] if 'ENDPOINT' in os.environ else 'http://localhost:8501/v1/models/hover:predict'
        self.infer_mask_shape = [80,  80]
        self.infer_input_shape = [270, 270]
        self.inf_batch_size = 16
        self.nr_types = 5 # 6  # denotes number of classes (including BG) for nuclear type classification
        self.eval_inf_input_tensor_names = ['images:0']
        self.eval_inf_output_tensor_names = ['predmap-coded:0']
        self.save_dir = save_dir

        # if it is PIL image
        if isinstance(input_img, Image.Image): 
            self.input_img = cv2.cvtColor(np.array(input_img, dtype=np.float32), cv2.COLOR_BGR2RGB)
        # if it is np array (f.eks. cv2 image)
        elif isinstance(input_img, np.ndarray):  
            if isinstance(input_img.flat[0], np.uint8): 
                self.input_img = cv2.cvtColor(np.array(Image.fromarray(input_img, 'RGB'), dtype=np.float32), cv2.COLOR_BGR2RGB)
            elif isinstance(input_img.flat[0], np.floating): 
                self.input_img = cv2.cvtColor(np.float32(input_img), cv2.COLOR_BGR2RGB)
        # if it is filename
        elif os.path.isfile(input_img): 
            self.input_img = cv2.cvtColor(cv2.imread(input_img), cv2.COLOR_BGR2RGB)
        else: 
            raise Exception('Unsupported type of input image.')

        self.remap_labels = False

    def proc_np_hv(self, pred, marker_mode=2, energy_mode=2, rgb=None):
        """
        Process Nuclei Prediction with XY Coordinate Map

        Args:
            pred: prediction output, assuming 
                    channel 0 contain probability map of nuclei
                    channel 1 containing the regressed X-map
                    channel 2 containing the regressed Y-map
        """

        blb_raw = pred[...,0]
        h_dir_raw = pred[...,1]
        v_dir_raw = pred[...,2]

        ##### Processing 
        blb = np.copy(blb_raw)
        blb[blb >= 0.5] = 1
        blb[blb <  0.5] = 0

        blb = measurements.label(blb)[0]
        blb = remove_small_objects(blb, min_size=10)
        blb[blb > 0] = 1 # back ground is 0 already
        #####

        if energy_mode == 2 or marker_mode == 2:
            h_dir = cv2.normalize(h_dir_raw, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            v_dir = cv2.normalize(v_dir_raw, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

            sobelh = cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=21)
            sobelv = cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=21)

            sobelh = 1 - (cv2.normalize(sobelh, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F))
            sobelv = 1 - (cv2.normalize(sobelv, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F))

            overall = np.maximum(sobelh, sobelv)
            overall = overall - (1 - blb)
            overall[overall < 0] = 0

            if energy_mode == 2:
                dist = (1.0 - overall) * blb
                ## nuclei values form mountains so inverse to get basins
                dist = -cv2.GaussianBlur(dist,(3, 3),0)

            if marker_mode == 2:
                overall[overall >= 0.4] = 1
                overall[overall <  0.4] = 0
                
                marker = blb - overall
                marker[marker < 0] = 0
                marker = binary_fill_holes(marker).astype('uint8')
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5, 5))
                marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)
                marker = measurements.label(marker)[0]
                marker = remove_small_objects(marker, min_size=10)
        
        proced_pred = watershed(dist, marker, mask=blb)

        return proced_pred

    def process_image(self, image, pred):
        pred = np.squeeze(pred['result'])

        pred_inst = pred[...,self.nr_types:]
        pred_type = pred[...,:self.nr_types]

        pred_inst = np.squeeze(pred_inst)
        pred_type = np.argmax(pred_type, axis=-1)
        pred_inst = pred

        pred_inst = self.proc_np_hv(pred_inst,
                    marker_mode=2,
                    energy_mode=2, rgb=image)

        # ! will be extremely slow on WSI/TMA so it's advisable to comment this out
        # * remap once so that further processing faster (metrics calculation, etc.)
        if (self.remap_labels):
            pred_inst = remap_label(pred_inst, by_size=True)

        #### * Get class of each instance id, stored at index id-1
        pred_id_list = list(np.unique(pred_inst))[1:] # exclude background ID
        pred_inst_type = np.full(len(pred_id_list), 0, dtype=np.int32)
        for idx, inst_id in enumerate(pred_id_list):
            inst_type = pred_type[pred_inst == inst_id]
            type_list, type_pixels = np.unique(inst_type, return_counts=True)
            type_list = list(zip(type_list, type_pixels))
            type_list = sorted(type_list, key=lambda x: x[1], reverse=True)
            inst_type = type_list[0][0]
            if inst_type == 0: # ! pick the 2nd most dominant if exist
                if len(type_list) > 1:
                    inst_type = type_list[1][0]
                else:
                    pass
                    # print('[Warn] Instance has `background` type')
            pred_inst_type[idx] = inst_type
        pred_inst_centroid = get_inst_centroid(pred_inst)

        pred = {'inst_map': pred_inst,
                'type_map': pred_type,
                'inst_type': pred_inst_type[:, None],
                'inst_centroid': pred_inst_centroid}
        overlaid_output = visualize_instances(pred_inst, image, (self.nr_types, pred_inst_type[:, None])) #cfg.nr_types + 1
        overlaid_output = cv2.cvtColor(overlaid_output, cv2.COLOR_BGR2RGB)
        return (overlaid_output, pred)
        # with open(os.path.join(proc_dir, f'{basename}.log'), 'w') as log_file:
        #     unique, counts = np.unique(pred_inst_type[:, None], return_counts=True)
        #     print(f'{basename} : {dict(zip(unique, counts))}', file = log_file)
    

    def __predict_subpatch(self, subpatch):
        """
        subpatch : numpy.ndarray
        """

        predict_request = json.dumps({"inputs": np.array(subpatch).tolist()})
        response = requests.post(self.server_url, data=predict_request)
        response.raise_for_status()
        prediction = np.array(response.json()['outputs'])
        return prediction # [0]


    def __gen_prediction(self, x):

        step_size = self.infer_mask_shape
        msk_size = self.infer_mask_shape
        win_size = self.infer_input_shape

        def get_last_steps(length, msk_size, step_size):
            nr_step = math.ceil((length - msk_size) / step_size)
            last_step = (nr_step + 1) * step_size
            return int(last_step), int(nr_step + 1)

        im_h = x.shape[0]
        im_w = x.shape[1]

        last_h, nr_step_h = get_last_steps(im_h, msk_size[0], step_size[0])
        last_w, nr_step_w = get_last_steps(im_w, msk_size[1], step_size[1])

        diff_h = win_size[0] - step_size[0]
        padt = diff_h // 2
        padb = last_h + win_size[0] - im_h

        diff_w = win_size[1] - step_size[1]
        padl = diff_w // 2
        padr = last_w + win_size[1] - im_w

        x = np.lib.pad(x, ((padt, padb), (padl, padr), (0, 0)), 'reflect')

        sub_patches = []

        for row in range(0, last_h, step_size[0]):
            for col in range (0, last_w, step_size[1]):
                win = x[row:row+win_size[0],
                        col:col+win_size[1]]
                sub_patches.append(win)
        pred_map = deque()

        while len(sub_patches) > self.inf_batch_size:
            mini_batch  = sub_patches[:self.inf_batch_size]
            sub_patches = sub_patches[self.inf_batch_size:]
            mini_output = self.__predict_subpatch(mini_batch)
            mini_output = np.split(mini_output, self.inf_batch_size, axis=0)
            pred_map.extend(mini_output)
        if len(sub_patches) != 0:
            mini_output = self.__predict_subpatch(sub_patches)
            mini_output = np.split(mini_output, len(sub_patches), axis=0)
            pred_map.extend(mini_output)

        #### Assemble back into full image
        output_patch_shape = np.squeeze(pred_map[0]).shape
        ch = 1 if len(output_patch_shape) == 2 else output_patch_shape[-1]

        #### Assemble back into full image
        pred_map = np.squeeze(np.array(pred_map))
        pred_map = np.reshape(pred_map, (nr_step_h, nr_step_w) + pred_map.shape[1:])
        pred_map = np.transpose(pred_map, [0, 2, 1, 3, 4]) if ch != 1 else \
                        np.transpose(pred_map, [0, 2, 1, 3])
        pred_map = np.reshape(pred_map, (pred_map.shape[0] * pred_map.shape[1],
                                         pred_map.shape[2] * pred_map.shape[3], ch))
        pred_map = np.squeeze(pred_map[:im_h,:im_w]) # just crop back to original size

        return pred_map

    def __process(self):
        pred_map = {'result': [self.__gen_prediction(self.input_img)]} # {'result':[pred_map]}
        # np.save(f"{name_out}_map.npy", pred_map) 

        # pred_inst, pred_type = proc_utils.process_instance(pred_map, nr_types=self.nr_types)
        # overlaid_output = visualize_instances(img, pred_inst, pred_type)
        # overlaid_output = cv2.cvtColor(overlaid_output, cv2.COLOR_BGR2RGB)
        # # combine instance and type arrays for saving
        # pred_inst = np.expand_dims(pred_inst, -1)
        # pred_type = np.expand_dims(pred_type, -1)
        # pred = np.dstack([pred_inst, pred_type])

        overlaid_output, pred = self.process_image(self.input_img, pred_map)
        return (overlaid_output, pred)

    def run(self):
        temp_file = NamedTemporaryFile()
        name_out = os.path.join(self.save_dir, os.path.split(temp_file.name)[1])

        proc_result = self.__process()
        
        cv2.imwrite(f'{name_out}.png', proc_result[0])
        print(f"Saved processed image to <{name_out}.png>. {datetime.now().strftime('%H:%M:%S.%f')}")

        np.save(f'{name_out}.npy', proc_result[1])
        print(f"Saved pred to <{name_out}.npy>. {datetime.now().strftime('%H:%M:%S.%f')}")
        


if __name__ == '__main__':
    """
    Example: 
        python external_infer_url.py --input_img '/data/input/data_consep/data/test/Images/test_1.png' --save_dir '/data/output/'
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='Comma separated list of GPU(s) to use.', default="0")
    parser.add_argument('--input_img', help='Full path to input image', required=True)
    parser.add_argument('--save_dir', help='Path to the directory to save result', required=True)
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    n_gpus = len(args.gpu.split(','))

    inferer = InfererURL(args.input_img, args.save_dir)
    inferer.run()