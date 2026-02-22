from dataclasses import dataclass
from pathlib import Path
from functools import reduce

import jax.numpy as jnp
import numpy as np
import torch
from jax.tree_util import tree_map
from PIL import Image
from torch.utils import data
from torch.utils.data import Dataset, random_split
from torchvision.transforms import v2
from jax.nn import one_hot

HEIGHT_DIM = -3
WIDTH_DIM = -2


class DummyAugment():
    def __init__(self):
        self.extension_factor = 1

    def __call__(self, img, idx):
        return img


class Rotate90Augment():
    def __init__(self, angles=[0, 1, 2, 3]):
        self.angles = angles
        self.extension_factor = len(angles)

    def __call__(self, img, idx):
        return np.rot90(img, self.angles[idx], axes=[HEIGHT_DIM, WIDTH_DIM])


class TranslateAugment():
    def __init__(self, height: int, width: int):
        self.height = height
        self.width = width
        self.extension_factor = height * width

    def __call__(self, img, idx):
        return np.roll(img, (idx // self.width, idx % self.width), axis=(HEIGHT_DIM, WIDTH_DIM))


class StackAugment():
    def __init__(self, augmentations):
        self.augmentations = augmentations
        self.extension_factors = [aug.extension_factor for aug in augmentations]
        self.extension_factor = reduce(lambda x, y: x * y, self.extension_factors)

    def __call__(self, img, idx):
        for aug in self.augmentations:
            img = aug(img, idx % aug.extension_factor)
            idx //= aug.extension_factor
        return img


class NCTCRCHE(Dataset):
    def __init__(self, dir, transform=None, target_transform=None,
                 keep_in_mem=False, augmentation=DummyAugment(), augmen_dim=False):
        # if augmented, the dataset will have shape (batch_dims, augmen_size, H, W, C)
        self.dir = Path(dir)
        folders = sorted([x for x in self.dir.iterdir() if x.is_dir()])
        self.classes = [x.name for x in folders]
        self.augmentation = augmentation
        self.augmen_dim = augmen_dim

        self.img_files = []
        self.img_labels = []
        self.class_samples_count = []
        for (i, folder) in enumerate(folders):
            img_files = list(folder.iterdir())
            self.img_files += img_files
            self.class_samples_count.append(len(img_files))
            self.img_labels += [i] * len(img_files)

        # multiply by augmentation size
        self.augmen_size = augmentation.extension_factor
        if self.augmen_dim:
            self.size = len(self.img_files)
        else:
            self.size = len(self.img_files) * self.augmen_size
        self.transform = staticmethod(transform)
        self.target_transform = target_transform
        self.keep_in_mem = keep_in_mem
        if keep_in_mem:
            print("Keeping data set in memory")
            # should have shape (H, W, C)
            tmp_img = self.load_image(0)[0]
            if self.augmen_dim:
                img_shape = [self.augmen_size, *tmp_img.shape]
            else:
                img_shape = tmp_img.shape
            self.images = np.empty((self.size, *img_shape))
            self.images_loaded = np.full(self.size, False)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if self.keep_in_mem and self.images_loaded[idx]:
            return self.images[idx], self.img_labels[idx]
        else:
            if self.augmen_dim:
                orig_image, label = self.load_image(idx)
                images = []
                for i in range(self.augmen_size):
                    image = self.augmentation(orig_image, i)
                    images.append(image)
                image = np.stack(images, axis=0)
            else:
                img_idx = idx // self.augmen_size
                augm_idx = idx % self.augmen_size
                tmp_image, label = self.load_image(img_idx)
                image = self.augmentation(tmp_image, augm_idx)

            if self.keep_in_mem:
                self.images[idx] = image
                self.images_loaded[idx] = True

            return image, label

    def __repr__(self):
        message = "NCT-CRC-HE dataset with the following classes:\n"
        for i, class_name in enumerate(self.classes):
            message += f"{i:2d}: {class_name:4s} consisting of {self.class_samples_count[i]:5d}\n"
        return message

    def load_image(self, idx):
        """Will load image and label with image shape [..., C, H, W]."""
        img_path = self.img_files[idx]
        image = Image.open(img_path)
        image = v2.ToImage()(image)
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def numpy_collate(batch):
    return tree_map(np.asarray, data.default_collate(batch))


class NumpyLoader(data.DataLoader):
    def __init__(self, dataset, batch_size=1,
                 shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0,
                 pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None):
        super(self.__class__, self).__init__(dataset,
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             sampler=sampler,
                                             batch_sampler=batch_sampler,
                                             num_workers=num_workers,
                                             collate_fn=numpy_collate,
                                             pin_memory=pin_memory,
                                             drop_last=drop_last,
                                             timeout=timeout,
                                             worker_init_fn=worker_init_fn)


class CastToNumpy(object):
    def __call__(self, pic):
        return np.transpose(np.array(pic, dtype=jnp.float32), axes=(1, 2, 0))


@dataclass
class DataloaderSplit:
    train: NumpyLoader
    test: NumpyLoader
    val: NumpyLoader


def create_dataset_for_jax(path, seed, pixels, n_train_samples, n_test_samples,
                           augmentation=DummyAugment(), augmen_dim=False):
    img_dim = (pixels, pixels)
    n_val_samples = n_test_samples

    transforms = v2.Compose([v2.Resize(img_dim), CastToNumpy()])
    dataset = NCTCRCHE(path, transform=transforms, keep_in_mem=False, augmentation=augmentation,
                       augmen_dim=augmen_dim)
    print("Loaded dataset\n", dataset)
    generator1 = torch.Generator().manual_seed(seed)
    train_dataset, test_dataset, val_dataset, _ = random_split(
        dataset,
        [
            n_train_samples,
            n_test_samples,
            n_val_samples,
            len(dataset) - n_train_samples - n_test_samples - n_val_samples,
        ],
        generator=generator1,
    )
    # with open("dataset_w_val.log", "w") as f:
    #     f.write(str(train_dataset.indices) + "\n" + str(test_dataset.indices))
    return train_dataset, test_dataset, val_dataset


def create_dataset_loader_for_jax(path, seed, pixels, n_train_samples,
                                  n_test_samples, batch_size):
    train_dataset, test_dataset, val_dataset = create_dataset_for_jax(
        path, seed, pixels, n_train_samples, n_test_samples
    )
    train_dataloader = NumpyLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )
    test_dataloader = NumpyLoader(
        test_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )

    val_dataloader = NumpyLoader(
        val_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )

    data = DataloaderSplit(
        train=train_dataloader,
        test=test_dataloader,
        val=val_dataloader
    )
    return data


@dataclass
class DatasetTensorCollection:
    train_data: jnp.ndarray
    train_labels: jnp.ndarray
    test_data: jnp.ndarray
    test_labels: jnp.ndarray


def create_dataset_tensors(train_dataset, test_dataset):
    train_data = jnp.stack([x[0] for x in train_dataset], axis=0)
    train_labels = jnp.stack([one_hot(x[1], 9) for x in train_dataset])
    if len(test_dataset) == 0:
        return DatasetTensorCollection(train_data, train_labels, None, None)

    test_data = jnp.stack([x[0] for x in test_dataset], axis=0)
    test_labels = jnp.stack([one_hot(x[1], 9) for x in test_dataset])

    return DatasetTensorCollection(train_data, train_labels, test_data,
                                   test_labels)


if __name__ == "__main__":
    path = "./data/NCT-CRC-HE-100K"
    dataset = NCTCRCHE(path, transform=CastToNumpy())

    training_generator = NumpyLoader(dataset, batch_size=4, num_workers=0)
    print(dataset)
    print(dataset[0])
    print(next(iter(training_generator)))
