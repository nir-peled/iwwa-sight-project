import torch as th
from torch import nn
from torchvision import datasets
import torchvision.transforms as transforms
import torch.utils.data
from skimage.io import imread
import numpy as np
import math
import os
from datetime import datetime
from multiprocessing import freeze_support
from sklearn.metrics import classification_report


"""
 * add:
 * git?
 * runnable in console or similar, to keep the xec env alive
 * save model between epochs?
 * 
 * add for test:
 * ? make and save graph of predictions by class ?
 *
 * consider and find out:
 * use more/other parameters for training - accuracy, recall, f1?
 * different loss function?
 * different optimizer? *Adam?*
"""

class CopepodImageSet(torch.utils.data.Dataset):
	"""
	 * <transforms> is iterator of items like (transformation, count), 
	 * each one creates <count> images with applied <transformation>
	 * when count=-1, the list <transformation> generates is flattened once
	"""
	def __init__(self, root, loader, grayscale=False, transform_list=None):
		super(CopepodImageSet, self).__init__()

		class_dict = {"negative":0, "negative1":0, "copepod":1, "copepod1":1}
		self.root = root
		self.image_paths = datasets.DatasetFolder.make_dataset(
			directory=root, 
			class_to_idx=class_dict, 
			extensions=("jpg",)
		)
		self.grayscale=grayscale
		self.transform = transform_list
		self.io_loader = loader
		print("first image path:")
		print(self.image_paths[0][0])

		self.image_ex = self.load(self.image_paths[0][0])
		self.image_shape = self.image_ex.shape

		self.length = self._calc_length()
		self.patches = {}

	def load(self, path):
		im = th.from_numpy(self.io_loader(path, as_gray=self.grayscale))
		# maxv , minv = im.max(), im.min() # debug
		# print(f"image max={maxv:>3f}, min={minv:>3f}") # debug
		return im.unsqueeze(0)


	# looks slow. find ways to make faster?
	def _generate_image_patches(self, path):
		results = [self.load(path)]
		for trans, count in self.transform:
				image_list = []
				if count > 0:
					# trans() returns single image
					# call <count> times
					for _ in range(count):
						image_list += [trans(im) for im in results]
				elif count == -1:
					# trans() returns list of images
					# attach the same label to each one
					for image in results:
						transformed = trans(image)
						image_list += list(transformed)
				# else ignore

				if len(image_list) > 0:
					results = image_list

		return results

	def _calc_length(self):
		print(f"image_paths = {len(self.image_paths)}")
		self._calc_image_multiplier()
		print(f"multiplier = {self.trans_multiplier}")
		return len(self.image_paths) * self.trans_multiplier

	def _calc_image_multiplier(self):
		print("calc multiplier")
		mul = 1
		for trans, count in self.transform:
			if count > 0:
				print(f"pos -> * {count}")
				mul *= count
			elif count == -1:
				trans_count = trans.get_count(self.image_shape)
				print(f"-1 -> * {trans_count}")
				mul *= trans_count

		self.trans_multiplier = mul

	def _get_patches(self, i):
		path, label = self.image_paths[i]
		patches = self._generate_image_patches(path)
		patches = [(pch, label) for pch in patches]
		# print(f"{os.getpid()}* patches length = {len(patches)}") # debug

		offset = i * self.trans_multiplier
		# print(f"{os.getpid()}* offset = {offset}") # debug
		patches_dict = dict( (offset + idx, pch) for idx, pch in enumerate(patches) )
		self.patches.update(patches_dict)

	def __len__(self):
		return self.length

	def __getitem__(self, i):
		# print(f"{os.getpid()}* looking for {i}") # debug
		if i not in self.patches.keys():
			image_i = i // self.trans_multiplier
			# print(f"{os.getpid()}* loading image {image_i}") # debug
			self._get_patches(image_i)
		return self.patches[i]


class PatchImage(object):
	"""
	 * one-to-many transformation that takes an image
	 * and returns patches of the image
	 * inspired by tensorflow.image.make_patches
	 * <patch_size> is the shape of the created patches
	 * <strides> is the strides between the starting point
	 * of a patch to the starting point of the next one
	 * <padding> is 'valid' if patches that are cut at the end
	 * are to be discarded, or 'same' if they are to be padded
	 * with zeroes
	 * 
	 * __call__() returns list of patches
	"""
	def __init__(self, patch_size, strides, padding=False):
		super(PatchImage, self).__init__()
		self.patch_size = patch_size
		self.strides = strides
		self.pad_mode = padding

	def __call__(self, image):
		channels, height, width = image.shape
		patch_h, patch_w = self.patch_size
		stride_h, stride_w = self.strides

		# pregenerate patch array
		patch_count = self.get_count(image.shape)
		patch_list = th.empty(patch_count, channels, 
			patch_h, patch_w, dtype=image.dtype)

		patch_shape = (channels, patch_h, patch_w)
		curr_h = 0
		i = 0

		# each iteration, creating patch starting at (curr_h, curr_w)
		# to (curr_h + patch_h, curr_w + patch_w)
		while curr_h < height:
			end_h = curr_h + patch_h
			curr_w = 0
			while curr_w < width:
				end_w = curr_w + patch_w
				new_patch = image[:, curr_h:end_h, curr_w:end_w]
				if tuple(new_patch.shape) == patch_shape:
					patch_list[i] = new_patch
					i += 1
				elif self.pad_mode:
					patch_list[i] = self._pad(new_patch, patch_shape)
					i += 1
				# else ignore non-matching patch

				curr_w += stride_w
			# end while curr_w < width
			curr_h += stride_h

		return patch_list

	# calculate number of patches generated based
	# on image shape and pad mode
	def get_count(self, im_shape):
		height, width = im_shape[1:]
		patch_h, patch_w = self.patch_size
		stride_h, stride_w = self.strides

		if self.pad_mode:
			# calc how many strides in the image, round up
			return math.ceil((height / stride_h) * (width / stride_w))
		else:
			# calc how many strides with full patch left, 
			# round down
			max_in_h = ((height - patch_h) // stride_h) + 1
			max_in_w = ((width - patch_w) // stride_w) + 1
			return max_in_h * max_in_w
		

	# pad <patch> with zeroes from the end
	# to <shape>
	def _pad(self, patch, shape):
		new_h, new_w = patch.shape[1:]
		fixed_patch = th.zeros(shape, dtype=patch.dtype)
		fixed_patch[:, :new_h, :new_w] = patch
		return fixed_patch


class CopepodNetwork(nn.Module):
	"""docstring for CopepodNetwork"""
	def __init__(self):
		super().__init__()
		self.flatten = nn.Flatten()
		self.conv1 = nn.Conv2d(1, 3, 5)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(3, 9, 5)
		self.fc1 = nn.LazyLinear(120)
		self.fc2 = nn.LazyLinear(84)
		self.fc3 = nn.LazyLinear(1)

	def set_loss_fn(self, loss_fn):
		self.loss_fn = loss_fn

	def set_optimizer(self, optimizer):
		self.optimizer = optimizer

	def forward(self, x):
		x = x.float()
		res = self.pool(nn.functional.relu(self.conv1(x)))
		res = self.pool(nn.functional.relu(self.conv2(res)))
		res = torch.flatten(res, 1)
		res = nn.functional.relu(self.fc1(res))
		res = nn.functional.relu(self.fc2(res))
		logits = self.fc3(res)
		return logits

	def train_model(self, dataloader, device):
		size = len(dataloader.dataset)
		print("train dataset size: ", size)
		self.train()

		for batch, (items, labels) in enumerate(dataloader):
			# initialize environment
			items, labels = items.to(device), labels.to(device)
			self.optimizer.zero_grad()

			# compute prediction error
			pred = self(items)
			loss = self.loss_fn(pred, labels.unsqueeze(1).float())

			# Backprogression
			loss.backward()
			self.optimizer.step()

			# print current status
			if batch % 40 == 39:
				loss_p, current = loss.item(), batch * len(items)
				print(f"loss: {loss_p:>7f}  [{current:>5d}/{size:>5d}]")

	def test_model(self, dataloader, device):
		size = len(dataloader.dataset)
		num_batches = len(dataloader)
		self.eval()
		test_loss = 0
		pred_list = np.zeros(size, dtype=np.float64)
		tag_list = np.zeros(size, dtype=np.float64)
		i = 0
		with torch.no_grad():
			for item, label in dataloader:
				item_num = label.size(dim=0)
				item, label = item.to(device), label.to(device)
				pred = self(item)
				pred_perc = th.sigmoid(pred)
				test_loss += self.loss_fn(pred, label.unsqueeze(1).float()).item()
				pred_list[i:i+item_num] = pred_perc.cpu().numpy().reshape(item_num)
				tag_list[i:i+item_num] = label.cpu().numpy().reshape(item_num)
				i += item_num
    

		test_loss /= num_batches
		print(f"Test  Avg loss: {test_loss:>8f} \n")
		self.print_test_data(tag_list, pred_list)
		

	def print_test_data(self, tag_list, pred_list):
		tag_arr = np.array(tag_list)
		pred_arr = np.array(pred_list)
		positive_preds = pred_arr[tag_arr == 1]
		negative_preds = pred_arr[tag_arr == 0]

		p_avg = np.average(positive_preds)
		p_mean = np.mean(positive_preds)
		p_max, p_min = np.max(positive_preds), np.min(positive_preds)
		n_avg = np.average(negative_preds)
		n_mean = np.mean(negative_preds)
		n_max, n_min = np.max(negative_preds), np.min(negative_preds)

		print(f"positive range:({p_min:>5f},{p_max:>5f}) avg:{p_avg:>5f} mean:{p_mean:>5f}")
		print(f"negative range:({n_min:>5f},{n_max:>5f}) avg:{n_avg:>5f} mean:{n_mean:>5f}")

		print(classification_report(tag_list, pred_list.round()))

def time_now():
	return datetime.now().strftime("%H:%M:%S")

def project_main():
	freeze_support()

	patch_height = 486
	patch_width = 648
	item_height = 200
	item_width = 260

	# values after GrayScale() are in [0,1], 
	# so no need to normalize
	transform_list = [
		(transforms.RandomRotation(180), 5),
		(PatchImage((patch_height, patch_width), (patch_height, patch_width), True), -1), 
		(transforms.Resize((item_height, item_width)), 1)
	]

	device = "cuda" if torch.cuda.is_available() else "cpu"
	print(f"Using {device} device")

	root = "C:\\Users\\User\\Desktop\\univercity\\telhai\\comp_vision\\project\\pictures"
	image_ds = CopepodImageSet(root, imread, True, transform_list)
	print(f"image set length = {len(image_ds)}")

	train_size = int(len(image_ds) * 0.8)
	test_size = len(image_ds) - train_size
	train_ds, test_ds = torch.utils.data.random_split(image_ds, (train_size, test_size))

	batch_size = 64
	train_loader = torch.utils.data.DataLoader(train_ds,
	 batch_size=batch_size, shuffle=True, num_workers=2)
	test_loader =torch.utils.data. DataLoader(test_ds,
	 batch_size=batch_size, shuffle=False, num_workers=2)

	model = CopepodNetwork().to(device)
	print(model)

	loss_fn = nn.BCEWithLogitsLoss(pos_weight=th.tensor([0.65/0.35]).to(device))
	optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.9)
	# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
	model.set_loss_fn(loss_fn)
	model.set_optimizer(optimizer)

	epochs = 50
	for t in range(epochs):
		print(f"Epoch {t+1} at {time_now()}\n-------------------------------")
		model.train_model(train_loader, device)
		print(f"testing at {time_now()}:")
		model.test_model(test_loader, device)
	print("done!")
	# try:
	# 	epochs = 50
	# 	for t in range(epochs):
	# 		print(f"Epoch {t+1} at {time_now()}\n-------------------------------")
	# 		model.train_model(train_loader, device)
	# 		print(f"testing at {time_now()}:")
	# 		model.test_model(test_loader, device)
	# 	print("done!")
	# except KeyboardInterrupt as e:
	# 	print("interrupred")
	# except Exception as e:
	# 	raise e
	# finally:
	# 	return model
	


if __name__ == '__main__':
	project_main()
