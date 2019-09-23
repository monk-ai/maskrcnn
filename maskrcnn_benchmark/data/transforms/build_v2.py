# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from . import transforms_v2 as T


def build_transforms(cfg, is_train=True):
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        fliplr_prob = cfg.INPUT.FLIPLR_PROB
        crop_max = cfg.INPUT.CROP_MAX
        gaussian_blur_sigma_max = cfg.INPUT.GAUSSIAN_BLUR_SIGMA_MAX
        contrast_normalization_range = cfg.INPUT.CONTRAST_NORMALIZATION_RANGE
        linear_contrast_range = cfg.INPUT.LINEAR_CONTRAST_RANGE
        brightness_range = cfg.INPUT.BRIGHTNESS_RANGE
        hue_sat_add_range = cfg.INPUT.HUE_SAT_ADD_RANGE
        gaussian_noise_scale_max = cfg.INPUT.GAUSSIAN_NOISE_SCALE_MAX
        input_zoom_range = cfg.INPUT.ZOOM_RANGE
        translation_range = cfg.INPUT.TRANSLATION_RANGE
        rotation_range = cfg.INPUT.ROTATE_DEGREES_RANGE
        shear_range = cfg.INPUT.SHEAR_RANGE
        perspective_sigma_max = cfg.INPUT.PERSPECTIVE_SIGMA_MAX
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        fliplr_prob = 0.0
        crop_max = 0.0
        gaussian_blur_sigma_max = 0.0
        contrast_normalization_range = 1.0
        linear_contrast_range = 1.0
        brightness_range = 1.0
        hue_sat_add_range = 0.0
        gaussian_noise_scale_max = 0.0
        input_zoom_range = 1.0
        translation_range = 0.0
        rotation_range = 0.0
        shear_range = 0.0
        perspective_sigma_max = 0.0

    to_bgr255 = cfg.INPUT.TO_BGR255
    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=to_bgr255
    )

    augmenter = T.Augmenter(
        modes=cfg.INPUT.MODES,
        fliplr_prob=fliplr_prob,
        crop_max=crop_max,
        gaussian_blur_sigma_max=gaussian_blur_sigma_max,
        contrast_normalization_range=contrast_normalization_range,
        linear_contrast_range=linear_contrast_range,
        brightness_range=brightness_range,
        hue_sat_add_range=hue_sat_add_range,
        gaussian_noise_scale_max=gaussian_noise_scale_max,
        input_zoom_range=input_zoom_range,
        translation_range=translation_range,
        rotation_range=rotation_range,
        shear_range=shear_range,
        perspective_sigma_max=perspective_sigma_max
    )

    transform = T.Compose(
        [
            T.Resize(min_size, max_size),
            normalize_transform,
            T.PILToArray(),
            T.ToImgaugPolygons(),
            augmenter,
            T.ToMaskrcnnPolygons(modes=['masks']),
            T.ToTensor(),
            normalize_transform,
        ]
    )
    return transform
