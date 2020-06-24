import glob
import os
import argparse
from datetime import datetime

import cv2
import numpy as np
import scipy.io as sio
from scipy.ndimage import filters, measurements
from scipy.ndimage.morphology import (binary_dilation, binary_fill_holes,
                                      distance_transform_cdt,
                                      distance_transform_edt)
from skimage.morphology import remove_small_objects, watershed

import postproc.hover
import postproc.dist
import postproc.other

from config import Config

from misc.viz_utils import visualize_instances
from misc.utils import get_inst_centroid
from metrics.stats_utils import remap_label

from joblib import Parallel, delayed


AV_CPU = os.cpu_count()

###################

def process(parallel=False):

    ## ! WARNING:
    ## check the prediction channels, wrong ordering will break the code !
    ## the prediction channels ordering should match the ones produced in augs.py

    cfg = Config()

    # * flag for HoVer-Net only
    # 1 - threshold, 2 - sobel based
    energy_mode = 2
    marker_mode = 2

    for num, data_dir in enumerate(cfg.inf_data_list):
        pred_dir = os.path.join(cfg.inf_output_dir, str(num))
        proc_dir = '{}_processed'.format(pred_dir)

        file_list = glob.glob(os.path.join(pred_dir, '*.mat'))
        file_list.sort() # ensure same order

        if not os.path.isdir(proc_dir):
            os.makedirs(proc_dir)

        def process_image(filename):
            filename = os.path.basename(filename)
            basename = filename.split('.')[0]
            if not parallel: print(pred_dir, basename, flush=True)

            ##
            img = cv2.imread(os.path.join(data_dir, '{}{}'.format(basename, cfg.inf_imgs_ext)))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            pred = sio.loadmat(os.path.join(pred_dir, '{}.mat'.format(basename)))
            pred = np.squeeze(pred['result'])

            if hasattr(cfg, 'type_classification') and cfg.type_classification:
                pred_inst = pred[...,cfg.nr_types:]
                pred_type = pred[...,:cfg.nr_types]

                pred_inst = np.squeeze(pred_inst)
                pred_type = np.argmax(pred_type, axis=-1)

                if cfg.model_type == 'micronet':
                    # dilate prediction of all type to match it with
                    # the instance segmentation post-proc code
                    kernel = np.array([[0, 1, 0],
                                    [1, 1, 1],
                                    [0, 1, 0]], np.uint8)
                    canvas = np.zeros_like(pred_type, dtype=np.int32)
                    for type_id in range(1, cfg.nr_classes):
                        type_map = (pred_type == type_id).astype('uint8')
                        type_map = cv2.dilate(type_map, kernel, iterations=1)
                        canvas[type_map > 0] = type_id
            else:
                pred_inst = pred

            if cfg.model_type == 'np_hv' or cfg.model_type == 'np_hv_opt':
                pred_inst = postproc.hover.proc_np_hv(pred_inst,
                                marker_mode=marker_mode,
                                energy_mode=energy_mode, rgb=img)
            elif cfg.model_type == 'np_dist':
                pred_inst = postproc.hover.proc_np_dist(pred_inst)
            elif cfg.model_type == 'dist':
                pred_inst = postproc.dist.process(pred_inst)
            else:
                pred_inst = postproc.other.process(pred_inst, cfg.model_type)

            # ! will be extremely slow on WSI/TMA so it's advisable to comment this out
            # * remap once so that further processing faster (metrics calculation, etc.)
            if (cfg.remap_labels):
                pred_inst = remap_label(pred_inst, by_size=True)

            # for instance segmentation only
            if cfg.type_classification:
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
                            if not parallel: 
                                pass # print('[Warn] Instance has `background` type')
                    pred_inst_type[idx] = inst_type
                pred_inst_centroid = get_inst_centroid(pred_inst)

                sio.savemat(os.path.join(proc_dir, '{}.mat'.format(basename)),
                            {'inst_map'  :     pred_inst,
                            'type_map'  :     pred_type,
                            'inst_type' :     pred_inst_type[:, None],
                            'inst_centroid' : pred_inst_centroid,
                            })
                overlaid_output = visualize_instances(pred_inst, img, ((cfg.nuclei_type_dict, cfg.color_palete), pred_inst_type[:, None]), cfg.outline)
                overlaid_output = cv2.cvtColor(overlaid_output, cv2.COLOR_BGR2RGB)
                cv2.imwrite(os.path.join(proc_dir, '{}.png'.format(basename)), overlaid_output)
                with open(os.path.join(proc_dir, f'{basename}.log'), 'w') as log_file:
                    unique, counts = np.unique(pred_inst_type[:, None], return_counts=True)
                    unique = list(unique)
                    if 0 in unique: # remove backround entries
                        counts = np.delete(counts, unique.index(0))
                        unique.remove(0)
                    print(f'{basename} : {dict(zip([{str(v): str(k) for k, v in cfg.nuclei_type_dict.items()}[str(item)] for item in unique], counts))}', file = log_file)

            else:
                sio.savemat(os.path.join(proc_dir, '{}.mat'.format(basename)),
                            {'inst_map'  : pred_inst})
                overlaid_output = visualize_instances(pred_inst, img)
                overlaid_output = cv2.cvtColor(overlaid_output, cv2.COLOR_BGR2RGB)
                cv2.imwrite(os.path.join(proc_dir, '{}_uc.png'.format(basename)), overlaid_output)

            ##
            if not parallel: print(f"Finished for {basename} {datetime.now().strftime('%H:%M:%S.%f')}")

        start = datetime.now()
        if parallel: 
            print(f"Stared parallel process for ```{data_dir}``` {start.strftime('%d/%m %H:%M:%S.%f')}")
            print(f"Using {AV_CPU - 2} CPU cores")
            Parallel(n_jobs=AV_CPU - 2, prefer='threads')(delayed(process_image)(filename) for filename in file_list)
            end = datetime.now()
            print(f"Done parallel process for ```{data_dir}``` {end.strftime('%d/%m %H:%M:%S.%f')}")
        else:
            for filename in file_list:
                process_image(filename)
            end = datetime.now()
        print(f"Overall time elapsed: {end - start}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parallel', help='Whether to run process in parallel, consider using with large images', default=False, action='store_true')
    args = parser.parse_args()
    process(args.parallel)
