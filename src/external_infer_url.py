import os
import math
import timeit
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

from scripts.viz_utils import visualize_instances
import scripts.process_utils as proc_utils

class InfererURL():
    """
    Make sure that ENDPOINT is set up.
    
    input_img argument can be PIL image, numpy array or just path to .png file.
    """

    def __init__(self, input_img, save_dir):
        # input_img as PIL
        # values for np_hv model graph
        self.server_url = os.environ['ENDPOINT']
        assert requests.get(':'.join(self.server_url.split(':')[:-1])).ok == True
        
        ### graph specific
        self.mask_shape = [80,  80] # [164, 164] # 
        self.input_shape = [270, 270] # [256, 256] # 
        self.nr_types = 5 # 6  # denotes number of classes (including BG) for nuclear type classification
        ###

        self.inf_batch_size = 25
        self.eval_inf_input_tensor_names = ['images:0']
        self.eval_inf_output_tensor_names = ['predmap-coded:0']

        self.input_norm  = True
        self.remap_labels = False
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

    def _timer(func):
        def wrapped(self, *args, **kwargs):
            start_time = timeit.default_timer()
            func(self, *args, **kwargs)
            elapsed = timeit.default_timer() - start_time 
            if (kwargs['logging']==True): print(f"Finished {func.__name__}. Time elapsed: {'{:.3f}'.format(elapsed)} sec")
        return wrapped

    def __gen_prediction(self, x):

        step_size = self.mask_shape
        msk_size = self.mask_shape
        win_size = self.input_shape

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

    def __process_instance(self, pred_map, output_dtype='uint16'):
        """
        Post processing script for image tiles

        Args:
            pred_map: commbined output of nc, np and hv branches
            output_dtype: data type of output
        
        Returns:
            pred_inst:     pixel-wise nuclear instance segmentation prediction
            pred_type_out: pixel-wise nuclear type prediction 
        """
        
        # pred = np.squeeze(pred_map['result'])

        pred_inst = pred_map[..., self.nr_types:]
        pred_type = pred_map[..., :self.nr_types]

        pred_inst = np.squeeze(pred_inst)
        pred_type = np.argmax(pred_type, axis=-1)
        pred_type = np.squeeze(pred_type)

        pred_inst = proc_utils.proc_np_hv(pred_inst)

        if (self.remap_labels):
            pred_inst = proc_utils.remap_label(pred_inst, by_size=True)

        pred_type_out = np.zeros([pred_type.shape[0], pred_type.shape[1]])               
        #### * Get class of each instance id, stored at index id-1
        pred_id_list = list(np.unique(pred_inst))[1:] # exclude background ID
        pred_inst_type = np.full(len(pred_id_list), 0, dtype=np.int32)
        for idx, inst_id in enumerate(pred_id_list):
            inst_tmp = pred_inst == inst_id
            inst_type = pred_type[pred_inst == inst_id]
            type_list, type_pixels = np.unique(inst_type, return_counts=True)
            type_list = list(zip(type_list, type_pixels))
            type_list = sorted(type_list, key=lambda x: x[1], reverse=True)
            inst_type = type_list[0][0]
            if inst_type == 0: # ! pick the 2nd most dominant if exist
                if len(type_list) > 1:
                    inst_type = type_list[1][0]
            pred_type_out += (inst_tmp * inst_type)
        pred_type_out = pred_type_out.astype(output_dtype)
        pred_inst = pred_inst.astype(output_dtype)

        return pred_inst, pred_type_out
        # pred = {'inst_map': pred_inst,
        #         'type_map': pred_type,
        #         'inst_type': pred_inst_type[:, None],
        #         'inst_centroid': pred_inst_centroid}
        # overlaid_output = visualize_instances(pred_inst, image, (self.nr_types, pred_inst_type[:, None])) #cfg.nr_types + 1
        # overlaid_output = cv2.cvtColor(overlaid_output, cv2.COLOR_BGR2RGB)
        # return (overlaid_output, pred)

        # with open(os.path.join(proc_dir, f'{basename}.log'), 'w') as log_file:
        #     unique, counts = np.unique(pred_inst_type[:, None], return_counts=True)
        #     print(f'{basename} : {dict(zip(unique, counts))}', file = log_file)
    

    def __predict_subpatch(self, subpatch):
        """
        subpatch : numpy.ndarray
        inputs - outputs
        instances - predictions
        """
        predict_request = json.dumps({"inputs": np.array(subpatch).tolist()})
        response = requests.post(self.server_url, data=predict_request)
        response.raise_for_status()
        prediction = np.array(response.json()['outputs'])
        return prediction # [0]

    @_timer    
    def run(self, logging=False):

        temp_file = NamedTemporaryFile()
        name_out = os.path.join(self.save_dir, os.path.split(temp_file.name)[1])

        # pred_map = {'result': [self.__gen_prediction(self.input_img)]} # {'result':[pred_map]}
        # np.save(f"{name_out}_map.npy", pred_map) 

        pred_map = self.__gen_prediction(self.input_img)
        pred_inst, pred_type = self.__process_instance(pred_map)
        
        overlaid_output = visualize_instances(self.input_img, pred_inst, pred_type)
        overlaid_output = cv2.cvtColor(overlaid_output, cv2.COLOR_BGR2RGB)

        # combine instance and type arrays for saving
        pred_inst = np.expand_dims(pred_inst, -1)
        pred_type = np.expand_dims(pred_type, -1)
        pred = np.dstack([pred_inst, pred_type])
        
        cv2.imwrite(f'{name_out}.png', overlaid_output)
        if logging: print(f"Saved processed image to <{name_out}.png>. {datetime.now().strftime('%H:%M:%S')}") # '%H:%M:%S.%f'

        np.save(f'{name_out}.npy', pred)
        if logging: print(f"Saved pred to <{name_out}.npy>. {datetime.now().strftime('%H:%M:%S')}")


if __name__ == '__main__':
    """
    Example: ENDPOINT=http://localhost:8501/v1/models/hover:predict \
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
    inferer.run(logging=True)