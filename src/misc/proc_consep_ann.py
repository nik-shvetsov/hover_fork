import os
import scipy.io as sio
import glob
import numpy as np

path_to_dataset = '/input/data_consep/CoNSeP/'

folders = ['Test', 'Train']

for mode in folders:
    file_list = glob.glob(os.path.join(path_to_dataset, mode, 'Labels', '*{}'.format('.mat')))
    file_list.sort()
    for file_mat in file_list:
        filename = os.path.basename(file_mat)
        basename = filename.split('.')[0]
        to_transform = sio.loadmat(file_mat)
        iter_to_transform = to_transform.copy()
        for key in iter_to_transform.keys():
            if str(key).startswith('__'):
                to_transform.pop(key, None)
        # save only maps
        result = np.dstack((to_transform['inst_map'], to_transform['type_map']))
        np.save(f"{file_mat.split('.')[0]}.npy", result)