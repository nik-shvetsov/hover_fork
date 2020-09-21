import os
import scipy.io as sio
import glob
import numpy as np

def transform(input_dir, output_dir, modes):
    for mode in modes:
        file_list = glob.glob(os.path.join(input_dir, mode, 'Labels', '*{}'.format('.mat')))
        file_list.sort()
        assert (len(file_list) != 0)
        for file_mat in file_list:
            filename = os.path.basename(file_mat)
            basename = filename.split('.')[0]
            to_transform = sio.loadmat(file_mat)
            iter_to_transform = to_transform.copy()
            for key in iter_to_transform.keys():
                if str(key).startswith('__'):
                    to_transform.pop(key, None)

            ## * for converting the GT type in CoNSeP
            to_transform['type_map'][(to_transform['type_map'] == 3) | (to_transform['type_map'] == 4)] = 3
            to_transform['type_map'][(to_transform['type_map'] == 5) | (to_transform['type_map'] == 6) | (to_transform['type_map'] == 7)] = 4
            assert (np.max(to_transform['type_map']) <= 4)

            # save only maps
            result = np.dstack((to_transform['inst_map'], to_transform['type_map']))
            np.save(f"{os.path.join(output_dir, mode.lower(), 'Labels', basename)}.npy", result)
            print(f"{os.path.join(output_dir, mode.lower(), 'Labels', basename)}.npy saved.")


if __name__ == '__main__':
    transform('/data/input/data_consep/CoNSeP/', '/data/input/data_consep/data/', ['Train', 'Test'])