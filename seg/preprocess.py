# dataset.py
import numpy as np
import nibabel as nib
import torch
import random


# 2D预处理类
class RandomCrop2D(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if len(image.shape) == 4:  # 如果形状是 (4, 240, 240, 1)
            image = image.squeeze(axis=3)  # 去掉最后的1维度，变成 (4, 240, 240)
        if len(label.shape) == 4:  # 如果形状是 (4, 240, 240, 1)
            image = image.squeeze(axis=3)  # 去掉最后的1维度，变成 (4, 240, 240)

        c, h, w = image.shape

        top = np.random.randint(0, h - self.output_size[0])
        left = np.random.randint(0, w - self.output_size[1])

        image = image[:, top:top + self.output_size[0], left:left + self.output_size[1]]
        label = label[top:top + self.output_size[0], left:left + self.output_size[1]]

        return {'image': image, 'label': label}


class CenterCrop2D(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if len(image.shape) == 4:  # 如果形状是 (4, 240, 240, 1)
            image = image.squeeze(axis=3)  # 去掉最后的1维度，变成 (4, 240, 240)
        if len(label.shape) == 4:  # 如果形状是 (4, 240, 240, 1)
            image = image.squeeze(axis=3)  # 去掉最后的1维度，变成 (4, 240, 240)
        c, h, w = image.shape

        top = (h - self.output_size[0]) // 2
        left = (w - self.output_size[1]) // 2

        image = image[:, top:top + self.output_size[0], left:left + self.output_size[1]]
        label = label[top:top + self.output_size[0], left:left + self.output_size[1]]

        return {'image': image, 'label': label}


# 修改 RandomRotFlip2D 类
class RandomRotFlip2D(object):
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if len(image.shape) == 4:  # 如果形状是 (4, 240, 240, 1)
            image = image.squeeze(axis=3)  # 去掉最后的1维度，变成 (4, 240, 240)
        if len(label.shape) == 4:  # 如果形状是 (4, 240, 240, 1)
            image = image.squeeze(axis=3)  # 去掉最后的1维度，变成 (4, 240, 240)
        # 随机旋转 (0, 90, 180, 270度)
        k = np.random.randint(0, 4)
        image = np.stack([np.rot90(channel, k) for channel in image], axis=0)
        label = np.rot90(label, k)

        # 随机翻转 (水平或垂直)
        flip_type = np.random.choice(['none', 'h', 'v'], p=[0.5, 0.25, 0.25])
        if flip_type == 'h':
            image = np.flip(image, axis=2).copy()  # 使用.copy()确保连续性
            label = np.flip(label, axis=1).copy()  # 使用.copy()确保连续性
        elif flip_type == 'v':
            image = np.flip(image, axis=1).copy()  # 使用.copy()确保连续性
            label = np.flip(label, axis=0).copy()  # 使用.copy()确保连续性

        return {'image': image, 'label': label}


def augment_gaussian_noise(data_sample, noise_variance=(0, 0.1)):
    if noise_variance[0] == noise_variance[1]:
        variance = noise_variance[0]
    else:
        variance = random.uniform(noise_variance[0], noise_variance[1])
    data_sample = data_sample + np.random.normal(0.0, variance, size=data_sample.shape)
    return data_sample


class GaussianNoise(object):
    def __init__(self, noise_variance=(0, 0.1), p=0.5):
        self.prob = p
        self.noise_variance = noise_variance

    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        if np.random.uniform() < self.prob:
            image = augment_gaussian_noise(image, self.noise_variance)
        return {'image': image, 'label': label}


def augment_contrast(data_sample, contrast_range=(0.75, 1.25), preserve_range=True, per_channel=True):
    if not per_channel:
        mn = data_sample.mean()
        if preserve_range:
            minm = data_sample.min()
            maxm = data_sample.max()
        if np.random.random() < 0.5 and contrast_range[0] < 1:
            factor = np.random.uniform(contrast_range[0], 1)
        else:
            factor = np.random.uniform(max(contrast_range[0], 1), contrast_range[1])
        data_sample = (data_sample - mn) * factor + mn
        if preserve_range:
            data_sample[data_sample < minm] = minm
            data_sample[data_sample > maxm] = maxm
    else:
        for c in range(data_sample.shape[0]):
            mn = data_sample[c].mean()
            if preserve_range:
                minm = data_sample[c].min()
                maxm = data_sample[c].max()
            if np.random.random() < 0.5 and contrast_range[0] < 1:
                factor = np.random.uniform(contrast_range[0], 1)
            else:
                factor = np.random.uniform(max(contrast_range[0], 1), contrast_range[1])
            data_sample[c] = (data_sample[c] - mn) * factor + mn
            if preserve_range:
                data_sample[c][data_sample[c] < minm] = minm
                data_sample[c][data_sample[c] > maxm] = maxm
    return data_sample


class ContrastAugmentationTransform(object):
    def __init__(self, contrast_range=(0.75, 1.25), preserve_range=True, per_channel=True, p_per_sample=1.):
        self.p_per_sample = p_per_sample
        self.contrast_range = contrast_range
        self.preserve_range = preserve_range
        self.per_channel = per_channel

    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        if np.random.uniform() < self.p_per_sample:
            image = augment_contrast(image, contrast_range=self.contrast_range,
                                     preserve_range=self.preserve_range, per_channel=self.per_channel)
        return {'image': image, 'label': label}


def augment_brightness_additive(data_sample, mu: float, sigma: float, per_channel: bool = True,
                                p_per_channel: float = 1.):
    if not per_channel:
        rnd_nb = np.random.normal(mu, sigma)
        for c in range(data_sample.shape[0]):
            if np.random.uniform() <= p_per_channel:
                data_sample[c] += rnd_nb
    else:
        for c in range(data_sample.shape[0]):
            if np.random.uniform() <= p_per_channel:
                rnd_nb = np.random.normal(mu, sigma)
                data_sample[c] += rnd_nb
    return data_sample


class BrightnessTransform(object):
    def __init__(self, mu, sigma, per_channel=True, p_per_sample=1., p_per_channel=1.):
        self.p_per_sample = p_per_sample
        self.mu = mu
        self.sigma = sigma
        self.per_channel = per_channel
        self.p_per_channel = p_per_channel

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if np.random.uniform() < self.p_per_sample:
            image = augment_brightness_additive(image, self.mu, self.sigma, self.per_channel,
                                                p_per_channel=self.p_per_channel)
        return {'image': image, 'label': label}


# 修改 ToTensorDict 类
class ToTensorDict(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        # 确保数组是连续的
        if not image.flags.contiguous:
            image = np.ascontiguousarray(image)
        if not label.flags.contiguous:
            label = np.ascontiguousarray(label)

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long()
        return {'image': image, 'label': label}


# 在预处理流程中添加强制连续性
class EnsureContiguous(object):
    """确保数组在内存中是连续的"""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if not image.flags.contiguous:
            image = np.ascontiguousarray(image)
        if not label.flags.contiguous:
            label = np.ascontiguousarray(label)

        return {'image': image, 'label': label}