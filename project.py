import torch as th
from torch import nn
from torchvision import datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as transfuncs
import torch.utils.data
from skimage.io import imread
import numpy as np
import math
import os
from datetime import datetime
from multiprocessing import freeze_support
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt



class CopepodImageSet(torch.utils.data.Dataset):
	"""
	 * <transforms> is iterator of items like (transformation, count), 
	 * each one creates <count> images with applied <transformation>
	 * when count=-1, the list <transformation> generates is flattened once
	"""
	def __init__(self, root, loader, grayscale=False, transform_list=None):
		super(CopepodImageSet, self).__init__()

		class_dict = {"negative":0, "copepod":1, "copepod1":1, "copepod_maybe":1}
		self.root = root
		self.image_paths = datasets.DatasetFolder.make_dataset(
			directory=root, 
			class_to_idx=class_dict, 
			is_valid_file= lambda path: any(folder in path for folder in class_dict.keys()) and path.endswith("jpg")
		)
		self.grayscale=grayscale
		self.transform = transform_list
		self.io_loader = loader
		print("first image path:")
		print(self.image_paths[0][0])

		self.image_ex = self.load(self.image_paths[0][0])
		self.image_shape = self.image_ex.shape
		self.image_dtype = self.image_ex.dtype

		self.length = self._calc_length()
		positives, negatives = self.count_labels()
		print(f"positives = {positives}, negatives = {negatives}")
		self.patches = {}

	def count_labels(self):
		positives = sum(l for _, l in self.image_paths)
		negatives = len(self.image_paths) - positives
		return positives, negatives

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
		patches = [(pch, label, path) for pch in patches]
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

class AllDirectionsImage(object):
	def __init__(self):
		super(AllDirectionsImage, self).__init__()

	def __call__(self, image):
		results = [image]
		vert_flipped_image = transfuncs.vflip(image)
		results.append(vert_flipped_image)
		results.append(transfuncs.hflip(image))
		results.append(transfuncs.hflip(vert_flipped_image))

		return results

	def get_count(self, image_size):
		return 4


class CopepodNetwork(nn.Module):
	"""docstring for CopepodNetwork"""
	def __init__(self):
		super().__init__()
		# self.flatten = nn.Flatten()
		self.conv1 = nn.Conv2d(1, 8, 5)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(8, 16, 5)
		self.conv3 = nn.Conv2d(16, 32, 5)
		self.conv4 = nn.Conv2d(32, 64, 5)
		self.conv5 = nn.Conv2d(64, 64, 5)
		self.conv6 = nn.Conv2d(64, 64, 5)
		self.fc = nn.LazyLinear(1024)
		self.fc1 = nn.LazyLinear(512)
		self.fc2 = nn.LazyLinear(256)
		self.fc3 = nn.LazyLinear(128)
		self.fc4 = nn.LazyLinear(64)
		self.fc5 = nn.LazyLinear(1)

		self.history = History()

	def set_loss_fn(self, loss_fn):
		self.loss_fn = loss_fn

	def set_optimizer(self, optimizer):
		self.optimizer = optimizer

	def forward(self, x):
		x = x.float()
		res = self.pool( nn.functional.relu( self.conv1(x) ) )
		res = self.pool( nn.functional.relu(self.conv2(res)) )
		res = self.pool( nn.functional.relu(self.conv3(res)) )
		res = self.pool( nn.functional.relu(self.conv4(res)) )
		res = self.pool( nn.functional.relu(self.conv5(res)) )
		res = torch.flatten(res, 1)
		res = nn.functional.relu(self.fc(res))
		res = nn.functional.relu(self.fc1(res))
		res = nn.functional.relu(self.fc2(res))
		res = nn.functional.relu(self.fc3(res))
		res = nn.functional.relu(self.fc4(res))
		logits = self.fc5(res)
		return logits

	def train_model(self, dataloader, device):
		size = len(dataloader.dataset)
		num_batches = len(dataloader)
		print("train dataset size: ", size)
		total_loss = 0
		self.train()

		for batch, (items, labels, _) in enumerate(dataloader):
			# initialize environment
			items, labels = items.to(device), labels.to(device)
			self.optimizer.zero_grad()

			# compute prediction error
			pred = self(items)
			loss = self.loss_fn(pred, labels.unsqueeze(1).float())

			# Backprogression
			loss.backward()
			self.optimizer.step()

			total_loss += loss.item()
			# print current status
			if batch % 20 == 19:
				loss_p, current = loss.item(), batch * len(items)
				print(f"loss: {loss_p:>7f}  [{current:>5d}/{size:>5d}]")

		total_loss /= num_batches
		self.history.add_train_loss(total_loss)

	def test_model(self, dataloader, device):
		size = len(dataloader.dataset)
		num_batches = len(dataloader)
		self.eval()
		test_loss = 0
		pred_list = np.zeros(size, dtype=np.float64)
		tag_list = np.zeros(size, dtype=np.float64)
		i = 0

		with torch.no_grad():
			for items, label, _ in dataloader:
				item_num = label.size(dim=0)
				items, label = items.to(device), label.to(device)

				pred = self(items)
				pred_perc = th.sigmoid(pred)
				test_loss += self.loss_fn(pred, label.unsqueeze(1).float()).item()
				pred_list[i:i+item_num] = pred_perc.squeeze().cpu().numpy()
				tag_list[i:i+item_num] = label.squeeze().cpu().numpy()
				i += item_num

		test_loss /= num_batches
		self.history.add_test_loss(test_loss)
		print(f"Test  Avg loss: {test_loss:>8f} \n")
		self.print_test_data(tag_list, pred_list)
		

	def print_test_data(self, tag_arr, pred_arr):
		# tag_arr = np.array(tag_list)
		# pred_arr = np.array(pred_list)
		positive_preds = pred_arr[tag_arr == 1]
		negative_preds = pred_arr[tag_arr == 0]

		p_mean = np.mean(positive_preds)
		p_std = np.std(positive_preds)
		p_max, p_min = np.max(positive_preds), np.min(positive_preds)
		n_mean = np.mean(negative_preds)
		n_std = np.std(negative_preds)
		n_max, n_min = np.max(negative_preds), np.min(negative_preds)

		print(f"positive range:({p_min:>5f},{p_max:>5f}) mean:{p_mean:>5f} std:{p_std:>5f}")
		print(f"negative range:({n_min:>5f},{n_max:>5f}) mean:{n_mean:>5f} std:{n_std:>5f}")

		print( classification_report(tag_arr, pred_arr.round(), digits=4) )
		hist_range = (min(p_min, n_min), max(p_max, n_max))
		self.history.make_graphs(positive_preds, negative_preds, hist_range)

class History():
	"""docstring for History"""
	def __init__(self):
		self.test_loss = []
		self.train_loss = []
		self.test_accuracy = []

	def add_train_loss(self, loss):
		self.train_loss.append(loss)

	def add_test_loss(self, loss):
		self.test_loss.append(loss)

	def add_test_accuracy(self, accuracy):
		self.test_accuracy.append(accuracy)

	def make_graphs(self, positive_preds, negative_preds, pred_range):
		epoch_range = range(1, len(self.test_loss) + 1)

		# loss graph
		fig, axes = plt.subplots(2, 2, figsize=(18, 12))
		self.make_loss_graph(axes[0,0], epoch_range)

		# hist graph
		self.make_hist_graph(axes[0,1], positive_preds, negative_preds, pred_range)

		# accuracy graph
		axes[1,0].sharex(axes[0,0])
		self.make_accuracy_graph(axes[1,0], epoch_range, positive_preds, negative_preds)

		fig.tight_layout()
		plt.show()

	def make_loss_graph(self, subplot, epoch_range):
		subplot.plot(epoch_range, self.test_loss, label="Test Loss")
		subplot.plot(epoch_range, self.train_loss, label="Train Loss")
		subplot.legend(loc='upper right')
		subplot.set_title("Train And Test Loss")

	def make_hist_graph(self, subplot, positive_preds, negative_preds, pred_range):
		subplot.set_xticks(np.arange(0, 1, step=0.05))
		subplot.hist([positive_preds, negative_preds], bins=180, alpha=0.5
		 , label=['positives', 'negatives'], range=pred_range )
		subplot.legend(loc='upper right')
		subplot.set_title("Test Prediction Distribution")

	def make_accuracy_graph(self, subplot, epoch_range, positive_preds, negative_preds):
		positive_correct = positive_preds.round().sum()
		negative_correct = negative_preds.size - negative_preds.round().sum()
		accuracy = (positive_correct + negative_correct) / (positive_preds.size + negative_preds.size)
		self.test_accuracy.append(accuracy)
		
		subplot.plot(epoch_range, self.test_accuracy)
		subplot.set_title("Test Accuracy")

def time_now():
	return datetime.now().strftime("%H:%M:%S")


original_height = 1944
original_width = 2592
patch_height = 972
patch_width = 1296
item_height = 200
item_width = 260

# values in grayscale are in [0,1], 
# need to normalize images?
transform_list = [
	# create more, different images
	(transforms.RandomVerticalFlip(0.5), 2),
	(transforms.RandomHorizontalFlip(0.5), 2),
	(transforms.RandomCrop((patch_height, patch_width)), 4),
	# (PatchImage((patch_height, patch_width), (patch_height, patch_width), True), -1), 
	# normalize images (converting to uint8 and back because torch is annoying)
	# seems to make it worse, so don't do it
	# (transforms.ConvertImageDtype(th.uint8), 1 ),
	# (transformFuncs.equalize, 1),
	# (transforms.ConvertImageDtype(th.float64), 1),
	
	# not sure why this helps, looks like it just does -0.5
	# (transforms.Normalize((0.5,), (1.0,)), 1), 
	(transforms.Resize((item_height, item_width)), 1)
]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

root = "/datasets"
image_ds = CopepodImageSet(root, imread, True, transform_list)
print(f"image set length = {len(image_ds)}")

train_size = int(len(image_ds) * 0.8)
test_size = len(image_ds) - train_size
train_ds, test_ds = torch.utils.data.random_split(image_ds, (train_size, test_size))

batch_size = 64
train_loader = torch.utils.data.DataLoader(train_ds,
 batch_size=batch_size, shuffle=True, num_workers=0)
test_loader =torch.utils.data. DataLoader(test_ds,
 batch_size=batch_size, shuffle=False, num_workers=0)

model = CopepodNetwork().to(device)
dummy_item = th.zeros([batch_size, 1, item_height, item_width],
                      dtype=image_ds.image_dtype).to(device)
model.forward(dummy_item)
print(model)

# loss_fn = nn.BCEWithLogitsLoss(pos_weight=th.tensor([0.65/0.35]).to(device))
# loss_fn = nn.BCEWithLogitsLoss()
positives, negatives = image_ds.count_labels()
loss_fn = nn.BCEWithLogitsLoss(pos_weight=th.tensor([negatives/positives]).to(device))
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.9)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5) # if not working, try lr=1e-5
# optimizer = th.optim.Adagrad(model.parameters(), lr=1e-5)
model.set_loss_fn(loss_fn)
model.set_optimizer(optimizer)

epochs = 200
for t in range(epochs):
	print(f"Epoch {t+1} at {time_now()}\n-------------------------------")
	model.train_model(train_loader, device)
	print(f"testing at {time_now()}:")
	model.test_model(test_loader, device)
print("done!")
