import cv2
import numpy as np
import time
import skimage
from skimage.morphology import remove_small_objects, remove_small_holes, disk, watershed
from skimage.filters import rank, threshold_otsu
from scipy import ndimage
from scipy.ndimage import filters, measurements

from scipy.ndimage.morphology import (binary_dilation, binary_erosion, binary_closing, binary_fill_holes)
from .viz_utils import bounding_box

def proc_np_hv(pred, return_coords=False):
    """
    Process Nuclei Prediction with XY Coordinate Map

    Args:
        pred:           prediction output, assuming 
                        channel 0 contain probability map of nuclei
                        channel 1 containing the regressed X-map
                        channel 2 containing the regressed Y-map
        return_coords: return coordinates of extracted instances
    """

    blb_raw = pred[...,0]
    h_dir_raw = pred[...,1]
    v_dir_raw = pred[...,2]

    # Processing 
    blb = np.copy(blb_raw)
    blb[blb >= 0.5] = 1
    blb[blb <  0.5] = 0

    blb = measurements.label(blb)[0]
    blb = remove_small_objects(blb, min_size=10)
    blb[blb > 0] = 1 # background is 0 already
    #####

    h_dir = cv2.normalize(h_dir_raw, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    v_dir = cv2.normalize(v_dir_raw, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    h_dir_raw = None  # clear variable
    v_dir_raw = None  # clear variable

    sobelh = cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=21)
    sobelv = cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=21)
    h_dir = None  # clear variable
    v_dir = None  # clear variable

    sobelh = 1 - (cv2.normalize(sobelh, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F))
    sobelv = 1 - (cv2.normalize(sobelv, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F))

    overall = np.maximum(sobelh, sobelv)
    sobelh = None  # clear variable
    sobelv = None  # clear variable
    overall = overall - (1 - blb)
    overall[overall < 0] = 0

    dist = (1.0 - overall) * blb
    # nuclei values form peaks so inverse to get basins
    dist = -cv2.GaussianBlur(dist,(3, 3),0)

    overall[overall >= 0.5] = 1
    overall[overall <  0.5] = 0
    marker = blb - overall
    overall = None # clear variable
    marker[marker < 0] = 0
    marker = binary_fill_holes(marker).astype('uint8')
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5, 5))
    marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)
    marker = measurements.label(marker)[0]
    marker = remove_small_objects(marker, min_size=10)
 
    pred_inst = watershed(dist, marker, mask=blb, watershed_line=False)
    if return_coords:
        label_idx = np.unique(pred_inst)
        coords = measurements.center_of_mass(blb, pred_inst, label_idx[1:])
        return pred_inst, coords
    else:
        return pred_inst


def process_instance(pred_map, nr_types, remap_label=False, output_dtype='uint16'):
    """
    Post processing script for image tiles

    Args:
        pred_map: commbined output of nc, np and hv branches
        nr_types: number of types considered at output of nc branch
        remap_label: whether to map instance labels from 1 to N (N = number of nuclei)
        output_dtype: data type of output
    
    Returns:
        pred_inst:     pixel-wise nuclear instance segmentation prediction
        pred_type_out: pixel-wise nuclear type prediction 
    """

    pred_inst = pred_map[..., nr_types:]
    pred_type = pred_map[..., :nr_types]

    pred_inst = np.squeeze(pred_inst)
    pred_type = np.argmax(pred_type, axis=-1)
    pred_type = np.squeeze(pred_type)
    
    pred_inst = proc_np_hv(pred_inst)

    # remap label is very slow - only uncomment if necessary to map labels in order
    if remap_label:
        pred_inst = remap_label(pred_inst, by_size=True)
    
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
####

def crop_array(pred_inst, pred_type, pred_cent, shape_tile, crop_shape=(70,70)):
    """
    Crop the instance and class array with a given nucleus at the centre.
    Done to decrease the search space and consequently processing time.

    Args:
        pred_inst:  predicted nuclear instances for a given tile
        pred_type:  predicted nuclear types (pixel based) for a given tile
        pred_cent:  predicted centroid for a given nucleus
        shape_tile: shape of tile 
        crop_shape: output crop shape (saved as (y,x))

    Returns:
        crop_pred_inst: cropped pred_inst of shape crop_shape
        crop_pred_type: cropped pred_type of shape crop_shape
    """
    pred_x = pred_cent[1] # x coordinate
    pred_y = pred_cent[0] # y coordinate

    if pred_x < (crop_shape[0]/2):
        x_crop = 0
    elif pred_x > (shape_tile[1] - (crop_shape[1]/2)):
        x_crop = shape_tile[1] - crop_shape[1]
    else:
        x_crop = (pred_cent[1] - (crop_shape[1]/2))
    
    if pred_y < (crop_shape[0]/2):
        y_crop = 0
    elif pred_y > (shape_tile[0] - (crop_shape[0]/2)):
        y_crop = shape_tile[0] - crop_shape[0]
    else:
        y_crop = (pred_cent[0] - (crop_shape[0]/2))
    
    x_crop = int(x_crop)
    y_crop = int(y_crop)
    
    # perform the crop
    crop_pred_inst = pred_inst[y_crop:y_crop+crop_shape[1], x_crop:x_crop+crop_shape[0]]
    crop_pred_type = pred_type[y_crop:y_crop+crop_shape[1], x_crop:x_crop+crop_shape[0]]

    return crop_pred_inst, crop_pred_type
####

def img_min_axis(img):
    """
    Get the minimum of the x and y axes for an input array

    Args:
        img: input array
    """
    try:
        return min(img.shape[:2])
    except AttributeError:
        return min(img.size)
####

def stain_entropy_otsu(img):
    """
    Binarise an input image by calculating the entropy on the 
    hameatoxylin and eosin channels and then using otsu threshold 

    Args:
        img: input array
    """

    img_copy = img.copy()
    hed = skimage.color.rgb2hed(img_copy)  # convert colour space
    hed = (hed * 255).astype(np.uint8)
    h = hed[:, :, 0]
    e = hed[:, :, 1]
    d = hed[:, :, 2]
    selem = disk(4)  # structuring element
    # calculate entropy for each colour channel
    h_entropy = rank.entropy(h, selem)
    e_entropy = rank.entropy(e, selem)
    d_entropy = rank.entropy(d, selem)
    entropy = np.sum([h_entropy, e_entropy], axis=0) - d_entropy
    # otsu threshold
    threshold_global_otsu = threshold_otsu(entropy)
    mask = entropy > threshold_global_otsu

    return mask
####

def morphology(mask, proc_scale):
    """
    Applies a series of morphological operations
    to refine the binarised tissue mask

    Args:
        mask: input binary mask to refine
        proc_scale: scale at which to process
    
    Return:
        processed binary image
    """

    mask_scale = img_min_axis(mask)
    # Join together large groups of small components ('salt')
    radius = int(8 * proc_scale)
    selem = disk(radius)
    mask = binary_dilation(mask, selem)

    # Remove thin structures
    radius = int(16 * proc_scale)
    selem = disk(radius)
    mask = binary_erosion(mask, selem)

    # Remove small disconnected objects
    mask = remove_small_holes(
        mask,
        area_threshold=int(40 * proc_scale)**2,
        connectivity=1,
    )

    # Close up small holes ('pepper')
    mask = binary_closing(mask, selem)

    mask = remove_small_objects(
        mask,
        min_size=int(120 * proc_scale)**2,
        connectivity=1,
    )

    radius = int(16 * proc_scale)
    selem = disk(radius)
    mask = binary_dilation(mask, selem)

    mask = remove_small_holes(
        mask,
        area_threshold=int(40 * proc_scale)**2,
        connectivity=1,
    )

    # Fill holes in mask
    mask = ndimage.binary_fill_holes(mask)

    return mask
####

def get_tissue_mask(img, proc_scale=0.5):
    """
    Obtains tissue mask for a given image

    Args:
        img: input WSI as a np array
        proc_scale: scale at which to process
    
    Returns:
        binarised tissue mask
    """
    img_copy = img.copy()
    if proc_scale != 1.0:
        img_resize = cv2.resize(img_copy, None, fx=proc_scale, fy=proc_scale)
    else:
        img_resize = img_copy

    mask = stain_entropy_otsu(img_resize)
    mask = morphology(mask, proc_scale)
    mask = mask.astype('uint8')

    if proc_scale != 1.0:
        mask = cv2.resize(
            mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    return mask
####

def remap_label(pred, by_size=False):
    """
    Rename all instance id so that the id is contiguous i.e [0, 1, 2, 3] 
    not [0, 2, 4, 6]. The ordering of instances (which one comes first) 
    is preserved unless by_size=True, then the instances will be reordered
    so that bigger nucler has smaller ID
    Args:
        pred    : the 2d array contain instances where each instances is marked
                  by non-zero integer
        by_size : renaming with larger instances has smaller id (on-top)
    """
    pred_id = list(np.unique(pred))
    pred_id.remove(0)
    if len(pred_id) == 0:
        return pred # no label
    if by_size:
        pred_size = []
        for inst_id in pred_id:
            size = (pred == inst_id).sum()
            pred_size.append(size)
        # sort the id by size in descending order
        pair_list = zip(pred_id, pred_size)
        pair_list = sorted(pair_list, key=lambda x: x[1], reverse=True)
        pred_id, pred_size = zip(*pair_list)

    new_pred = np.zeros(pred.shape, np.int32)
    for idx, inst_id in enumerate(pred_id):
        new_pred[pred == inst_id] = idx + 1    
    return new_pred
#####