import os
import importlib
import random

import cv2
import numpy as np
import tensorflow as tf
from tensorpack import imgaug

from loader.augs import (BinarizeLabel, GaussianBlur, GenInstanceDistance,
                         GenInstanceHV, MedianBlur, GenInstanceUnetMap,
                         GenInstanceContourMap)
####
class Config(object):
    def __init__(self, ):

        self.seed = 10
        mode = 'hover'
        self.model_type = 'np_hv'

        self.type_classification = False
        self.nr_types = 5  # denotes number of classes for nuclear type classification
        # ! some semantic segmentation network like micronet,
        # ! nr_types will replace nr_classes if type_classification=True
        self.nr_classes = 2 # Nuclei Pixels vs Background

        #### Dynamically setting the config file into variable
        if mode == 'hover':
            config_file = importlib.import_module('opt.hover') # np_hv, np_dist
        else:
            config_file = importlib.import_module('opt.other') # fcn8, dcan, etc.
        config_dict = config_file.__getattribute__(self.model_type)

        for variable, value in config_dict.items():
            self.__setattr__(variable, value)
        #### Training data

        # patches are stored as numpy arrays with N channels
        # ordering as [Image][Nuclei Pixels][Nuclei Type][Additional Map]
        # Ex: with type_classification=True
        #     HoVer-Net: RGB - Nuclei Pixels - Type Map - Horizontal and Vertical Map
        # Ex: with type_classification=False
        #     Dist     : RGB - Nuclei Pixels - Distance Map
        data_code_dict = {
            'unet'     : '536x536_84x84',
            'dist'     : '536x536_84x84',
            'fcn8'     : '512x512_256x256',
            'dcan'     : '512x512_256x256',
            'segnet'   : '512x512_256x256',
            'micronet' : '504x504_252x252',
            'np_hv'    : '540x540_80x80',
            'np_dist'  : '540x540_80x80',
        }

        self.data_ext = '.npy'
        # list of directories containing validation patches
        self.train_dir = ['/data/input/data_kumar/train/'] # ['/data/input/data_cpm17/train/']
        self.valid_dir = ['/data/input/data_kumar/test_diff/','/data/input/data_kumar/test_same/'] # ['/data/input/data_cpm17/test']

        # self.valid_dir = ['../../../train/Kumar_old_GT/%s/valid_same/XXXX/' % data_code_dict[self.model_type],
        #                   '../../../train/Kumar_old_GT/%s/valid_diff/XXXX/' % data_code_dict[self.model_type]]

        # nr of processes for parallel processing input
        self.nr_procs_train = 8
        self.nr_procs_valid = 4

        self.input_norm = True # normalize RGB to 0-1 range

        ####
        #
        exp_id = '1.0'
        model_id = '%s' % self.model_type
        self.model_name = os.path.join(exp_id, model_id)
        #self.model_name = '%s/%s' % (exp_id, model_id)
        # loading chkpts in tensorflow, the path must not contain extra '/'
        self.log_path = '/data/output/log' # log root path
        self.save_dir = os.path.join(self.log_path, self.model_name)
        #self.save_dir = '%s/%s' % (self.log_path, self.model_name) # log file destination

        #### Info for running inference
        self.inf_auto_find_chkpt = False
        # path to checkpoints will be used for inference, replace accordingly
        self.inf_model_path  = '/data/input/weights_hv_kumar.npz' # '/data/input/weights_hv_CPM17.npz'
        #self.save_dir + '/model-19640.index'

        # output will have channel ordering as [Nuclei Type][Nuclei Pixels][Additional]
        # where [Nuclei Type] will be used for getting the type of each instance
        # while [Nuclei Pixels][Additional] will be used for extracting instances

        # TODO: encode the file extension for each folder?
        # list of [[root_dir1, codeX, subdirA, subdirB], [root_dir2, codeY, subdirC, subdirD] etc.]
        # code is used together with 'inf_output_dir' to make output dir for each set
        self.inf_imgs_ext = '.tif'

        # rootdir, outputdirname, subdir1, subdir2(opt) ...
        self.inf_data_list = [
            # ['/data/input/data_cpm17', 'output_cpm_mat', 'test/Images'],
            ['/data/input/data_kumar', 'output_kumar_mat', 'test_diff/Images'],
            # ['/data/input/data_uit_he', 'output_uit_he_mat', 'test/Images'],
            # ['../../../data/NUC_Kumar/train-set/orig_split/', 'XXXX', 'valid_diff'],
        ]
        self.inf_output_dir = '/data/output/%s/%s' % (exp_id, model_id)

        # for inference during evalutaion mode i.e run by inferer.py
        self.eval_inf_input_tensor_names = ['images']
        self.eval_inf_output_tensor_names = ['predmap-coded']
        # for inference during training mode i.e run by trainer.py
        self.train_inf_output_tensor_names = ['predmap-coded', 'truemap-coded']


    def get_model(self):
        if self.model_type == 'np_hv':
            model_constructor = importlib.import_module('model.graph')
            model_constructor = model_constructor.Model_NP_HV
        elif self.model_type == 'np_dist':
            model_constructor = importlib.import_module('model.graph')
            model_constructor = model_constructor.Model_NP_DIST
        else:
            model_constructor = importlib.import_module('model.%s' % self.model_type)
            model_constructor = model_constructor.Graph
        return model_constructor # NOTE return alias, not object

    # refer to https://tensorpack.readthedocs.io/modules/dataflow.imgaug.html for
    # information on how to modify the augmentation parameters
    def get_train_augmentors(self, input_shape, output_shape, view=False):
        print(input_shape, output_shape)
        shape_augs = [
            imgaug.Affine(
                        shear=5, # in degree
                        scale=(0.8, 1.2),
                        rotate_max_deg=179,
                        translate_frac=(0.01, 0.01),
                        interp=cv2.INTER_NEAREST,
                        border=cv2.BORDER_CONSTANT),
            imgaug.Flip(vert=True),
            imgaug.Flip(horiz=True),
            imgaug.CenterCrop(input_shape),
        ]

        input_augs = [
            imgaug.RandomApplyAug(
                imgaug.RandomChooseAug(
                    [
                    GaussianBlur(),
                    MedianBlur(),
                    imgaug.GaussianNoise(),
                    ]
                ), 0.5),
            # standard color augmentation
            imgaug.RandomOrderAug(
                [imgaug.Hue((-8, 8), rgb=True),
                imgaug.Saturation(0.2, rgb=True),
                imgaug.Brightness(26, clip=True),
                imgaug.Contrast((0.75, 1.25), clip=True),
                ]),
            imgaug.ToUint8(),
        ]

        label_augs = []
        if self.model_type == 'unet' or self.model_type == 'micronet':
            label_augs =[GenInstanceUnetMap(crop_shape=output_shape)]
        if self.model_type == 'dcan':
            label_augs =[GenInstanceContourMap(crop_shape=output_shape)]
        if self.model_type == 'dist':
            label_augs = [GenInstanceDistance(crop_shape=output_shape, inst_norm=False)]
        if self.model_type == 'np_hv':
            label_augs = [GenInstanceHV(crop_shape=output_shape)]
        if self.model_type == 'np_dist':
            label_augs = [GenInstanceDistance(crop_shape=output_shape, inst_norm=True)]

        if not self.type_classification:
            label_augs.append(BinarizeLabel())

        if not view:
            label_augs.append(imgaug.CenterCrop(output_shape))

        return shape_augs, input_augs, label_augs

    def get_valid_augmentors(self, input_shape, output_shape, view=False):
        print(input_shape, output_shape)
        shape_augs = [
            imgaug.CenterCrop(input_shape),
        ]

        input_augs = None

        label_augs = []
        if self.model_type == 'unet' or self.model_type == 'micronet':
            label_augs =[GenInstanceUnetMap(crop_shape=output_shape)]
        if self.model_type == 'dcan':
            label_augs =[GenInstanceContourMap(crop_shape=output_shape)]
        if self.model_type == 'dist':
            label_augs = [GenInstanceDistance(crop_shape=output_shape, inst_norm=False)]
        if self.model_type == 'np_hv':
            label_augs = [GenInstanceHV(crop_shape=output_shape)]
        if self.model_type == 'np_dist':
            label_augs = [GenInstanceDistance(crop_shape=output_shape, inst_norm=True)]
        label_augs.append(BinarizeLabel())

        if not view:
            label_augs.append(imgaug.CenterCrop(output_shape))

        return shape_augs, input_augs, label_augs
