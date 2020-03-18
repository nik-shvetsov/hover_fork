import glob
import os

import cv2
import numpy as np

from misc.patch_extractor import PatchExtractor
from misc.utils import rm_n_mkdir

from config import Config

###########################################################################
if __name__ == '__main__':
    cfg = Config()

    for data_mode in os.listdir(cfg.extract_data_dir):
        if data_mode in cfg.data_modes:
            xtractor = PatchExtractor(cfg.win_size, cfg.step_size)
            work_dir = os.path.join(cfg.extract_data_dir, data_mode)
            img_dir = os.path.join(work_dir, 'Images')
            ann_dir = os.path.join(work_dir, 'Labels')
            out_dir = os.path.join(cfg.out_extract_root, ('{}_{}x{}_{}x{}'\
                .format(cfg.model_config, cfg.win_size[0], cfg.win_size[1], cfg.step_size[0], cfg.step_size[1])), data_mode, 'Labels')
            file_list = glob.glob(os.path.join(img_dir, '*{}'.format(cfg.img_ext)))
            file_list.sort()

            rm_n_mkdir(out_dir)
            for filename in file_list:
                filename = os.path.basename(filename)
                basename = filename.split('.')[0]
                print('Mode: {}, filename - {}'.format(data_mode, filename))

                img = cv2.imread(os.path.join(img_dir, '{}{}'.format(basename, cfg.img_ext)))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                if cfg.type_classification:
                    # # assumes that ann is HxWx2 (nuclei class labels are available at index 1 of C)
                    # ann = np.load(os.path.join(ann_dir, '{}.npy'.format(basename)))
                    # ann_inst = ann[...,0]
                    # ann_type = ann[...,1]
                    ann = sio.loadmat(os.path.join(ann_dir, '{}.mat'.format(basename)))
                    ann_inst = ann['inst_map']
                    ann_type = ann['class_map']

                    # merge classes for CoNSeP (in paper we only utilise 3 nuclei classes and background)
                    # If own dataset is used, then the below may need to be modified
                    ann_type[(ann_type == 3) | (ann_type == 4)] = 3
                    ann_type[(ann_type == 5) | (ann_type == 6) | (ann_type == 7)] = 4

                    assert np.max(ann_type) <= cfg.nr_types - 1, \
                                    ("Only {} types of nuclei are defined for training"\
                                    "but there are {} types found in the input image.").format(cfg.nr_types, np.max(ann_type))

                    ann = np.dstack([ann_inst, ann_type])
                    ann = ann.astype('int32')
                else:
                    # assumes that ann is HxW
                    ann_inst = sio.loadmat(os.path.join(ann_dir, '{}.mat'.format(basename))) # .mat?
                    ann_inst = (ann_inst['inst_map']).astype('int32')
                    ann = np.expand_dims(ann_inst, -1)

                img = np.concatenate([img, ann], axis=-1)
                sub_patches = xtractor.extract(img, cfg.extract_type)
                for idx, patch in enumerate(sub_patches):
                    np.save("{0}/{1}_{2:03d}.npy".format(out_dir, basename, idx), patch)
