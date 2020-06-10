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


class InfererURL():
	"""
	Make sure that ENDPOINT and CUDA_VISIBLE_DEVICES are set up.
	
	input_img argument can be PIL image or just path to .png file.
	"""

	def __init__(self, input_img, save_dir):
		# input_img as PIL
		# values for np_hv model graph

		self.server_url = os.environ['ENDPOINT'] if 'ENDPOINT' in os.environ else 'http://localhost:8501/v1/models/hover:predict'
		self.infer_mask_shape = [80,  80]
		self.infer_input_shape = [270, 270]
		self.inf_batch_size = 16
		self.eval_inf_input_tensor_names = ['images:0']
		self.eval_inf_output_tensor_names = ['predmap-coded:0']
		self.save_dir = save_dir
		self.input_img = np.array(input_img) if isinstance(input_img, Image.Image) else np.array(Image.open(input_img))


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

	def __process():

	def run(self):
		pred_map = self.__gen_prediction(self.input_img)
		temp_file = NamedTemporaryFile()
		out = os.path.join(self.save_dir, os.path.split(temp_file.name)[1])
		np.save(out, pred_map) # {'result':[pred_map]}
		print(f"Saved pred_map to <{out}.npy>. {datetime.now().strftime('%H:%M:%S.%f')}")

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