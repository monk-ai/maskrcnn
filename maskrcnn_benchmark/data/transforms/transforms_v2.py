# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import random

import torch
import torchvision
from torchvision.transforms import functional as F
import numpy as np
import imgaug as ia
ia.seed(1)
import imgaug.augmenters as iaa
from imgaug.augmentables.polys import Polygon
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Resize(object):
    """
    TODO: To ndarray
    """
    def __init__(self, min_size, max_size):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image, target=None):
        size = self.get_size(image.size)
        image = F.resize(image, size)
        if target is None:
            return image
        target = target.resize(image.size)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        return F.to_tensor(image.copy()), target


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, target=None):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image
        return image, target


class Augmenter:
    def __init__(self, modes, fliplr_prob, crop_max, gaussian_blur_sigma_max, contrast_normalization_range,
                 linear_contrast_range, brightness_range, hue_sat_add_range, gaussian_noise_scale_max, input_zoom_range,
                 translation_range, rotation_range, shear_range, perspective_sigma_max):
        self.modes = modes
        self.aug_color = iaa.Sequential([
            iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, gaussian_blur_sigma_max))),
            iaa.Sometimes(0.5, iaa.ContrastNormalization(contrast_normalization_range)),
            iaa.Sometimes(0.5, iaa.LinearContrast(linear_contrast_range, per_channel=0.5)),
            iaa.Sometimes(0.5, iaa.AddToHueAndSaturation(hue_sat_add_range)),
            iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, gaussian_noise_scale_max), per_channel=0.5)),
            iaa.Multiply(brightness_range, per_channel=0.4),
        ])
        self.aug_others = iaa.Sequential([
            # iaa.Resize({"height": 32, "width": 64}),
            iaa.Fliplr(fliplr_prob),
            iaa.Affine(
                scale={"x": input_zoom_range, "y": input_zoom_range},
                translate_percent={"x": translation_range, "y": translation_range},
                rotate=rotation_range,
                shear=shear_range
            ),
            iaa.Crop(percent=(0, crop_max)),
            iaa.Sometimes(0.5, iaa.PerspectiveTransform((0.0, perspective_sigma_max), keep_size=True)),
        ], random_order=False)  # apply augmenters in random order

    def __call__(self, img, target, counter=0):
        img_aug, target_aug = self.aug_color(image=img, polygons=target)
        img_aug, target_aug = self.aug_others(image=img_aug, polygons=target_aug)
        # TODO: Fix dat shit
        bugged = False
        try:
            target_aug = self.select_polygons(target_aug, img_aug)
        except:
            bugged = True
            print("BUGGED IMAGE")
        if len(target_aug) == 0 or bugged:
            # if counter < 3:
            #     return self(img=img, polygons=target, counter=counter+1)
            return self.aug_color(image=img, polygons=target)
        return img_aug, target_aug

    def pol_to_bbox(self, pol):
        bbox = pol.to_bounding_box()
        return [bbox.x1, bbox.y1, bbox.x2, bbox.y2]

    def select_polygons(self, polygons, img, policy="clip_instances_partly_out"):
        """
        TODO: When a polygone has several groups, take bbox and subpolygons accordingly as maskrcnn does
        """
        if policy == "remove_instances_partly_out":
            to_take = [not p.is_out_of_image(img, fully=False, partly=True) for p in polygons]
            return [pol for i, pol in enumerate(polygons) if to_take[i] is True]

        if policy == "clip_instances_partly_out":
            to_take = [not p.is_out_of_image(img, fully=True, partly=False) for p in polygons]
            pol_to_take = [pol for i, pol in enumerate(polygons) if to_take[i] is True]
            pol_clipped = [pol.clip_out_of_image(img) for pol in pol_to_take]
            # return pol_clipped
            biggest_subpols = []
            for pol in pol_clipped:
                areas = []
                for subpol in pol:
                    areas.append(subpol.area)
                biggest_subpols.append(pol[int(np.argmax(areas))])
            return biggest_subpols


class PILToArray:
    """
    Convert PIL image to ndarray
    """
    def __init__(self):
        pass

    def __call__(self, img, target):
        return np.asarray(img), target


class ToImgaugPolygons:
    """
    Convert this repo target object to imgaug polygons
    """
    def __init__(self):
        pass

    def __call__(self, img, target):
        polygons = []
        for i, p in enumerate(target.get_field('masks').instances.polygons):
            points = p.polygons[0].numpy()
            points = points.reshape((int(len(points) / 2), 2))
            label = target.get_field('labels').numpy()[i]
            polygons.append(Polygon(points, label=label))
        return img, polygons


class ToMaskrcnnPolygons:
    """
    Convert imgaug polygons to this repo target
    """
    def __init__(self, modes):
        self.modes = modes
        pass

    def pol_to_bbox(self, pol):
        bbox = pol.to_bounding_box()
        return [bbox.x1, bbox.y1, bbox.x2, bbox.y2]

    def __call__(self, img, target):
        boxes = [self.pol_to_bbox(pol) for pol in target]
        labels = [pol.label for pol in target]

        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        new_target = BoxList(boxes, (img.shape[1], img.shape[0]), mode="xyxy")

        classes = torch.tensor(labels)
        new_target.add_field("labels", classes)

        if 'masks' in self.modes:
            masks = []
            for pol in target:
                x = pol.xx[:, np.newaxis]
                y = pol.yy[:, np.newaxis]
                # TODO: change if several groups inside one instance
                masks.append([list(np.concatenate((x, y), axis=1).flatten())])
            masks = SegmentationMask(masks, (img.shape[1], img.shape[0]), mode='poly')
            new_target.add_field("masks", masks)

        if 'keypoints' in self.modes:
            pass
        return img, new_target.clip_to_image(remove_empty=True)
