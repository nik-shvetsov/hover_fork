import os
import cv2
import glob
import staintools

from misc.utils import rm_n_mkdir

def stain_normilize(img_dir, save_dir, targets_dir, norm_brightness=False):
    file_list = glob.glob(os.path.join(img_dir, '*.png'))
    file_list.sort()

    if norm_brightness:
        standardizer = staintools.LuminosityStandardizer()
    stain_normalizer = staintools.StainNormalizer(method='vahadane')

    # dict of paths to target image and dir code to make output folder
    # {'/data/TCGA-21-5784-01Z-00-DX1.tif' : '5784'}
    stain_norm_target = {k : v for k, v in zip(glob.glob(os.path.join(targets_dir, '*.*')), range(len(glob.glob(os.path.join(targets_dir, '*.*')))))}

    for target_path, target_code in stain_norm_target.items():
        target_img = cv2.imread(target_path)
        target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
        if norm_brightness:
            target_img = standardizer.transform(target_img)
        stain_normalizer.fit(target_img)

        norm_dir = os.path.join(save_dir, target_code)
        rm_n_mkdir(norm_dir)

        for img_path in file_list:
            filename = os.path.basename(img_path)
            basename = filename.split('.')[0]
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if norm_brightness:
                img = standardizer.transform(img)

            img = stain_normalizer.transform(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(norm_dir, '{}.png'.format(basename)), img)

if __name__ == '__main__':
    '''
    Example:
        python stain_norm.py --img_dir /data/input/ --save_dir /data/output/ --targets_dir /data/input_targets/ --nb
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', help='Path to images folder', required=True)
    parser.add_argument('--save_dir', help='Where to save normalized image', required=True)
    parser.add_argument('--targets_dir', help='Path to folder with target for normalization', required=True)
    parser.add_argument('--nb', help='Whether to normalize brightness', default=False, action='store_true')
    args = parser.parse_args()

    stain_normilize(args.img_dir, args.save_dir, args.targets_dir, norm_brightness=args.nb)
