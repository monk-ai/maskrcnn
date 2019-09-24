import os
import torch
from maskrcnn_benchmark.utils.c2_model_loading import load_c2_format
from maskrcnn_benchmark.config import cfg
import argparse


def removekey(d, listofkeys):
    r = dict(d)
    for key in listofkeys:
        print('key: {} is removed'.format(key))
        r.pop(key)
    return r


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--input",
        default="",
        metavar="FILE",
        help="path to pretrained network",
        type=str,
    )
    parser.add_argument(
        "--cfg",
        default="",
        metavar="FILE",
        help="path to yaml cfg file",
        type=str,
    )
    parser.add_argument(
        "--output",
        default="",
        metavar="FILE",
        help="path to save trimmed network",
        type=str,
    )

    args = parser.parse_args()

    DETECTRON_PATH = os.path.expanduser(args.input)
    print('detectron path: {}'.format(DETECTRON_PATH))

    keys_removed = [
        'cls_score.bias',
        'cls_score.weight',
        'bbox_pred.bias',
        'bbox_pred.weight',
        'mask_fcn_logits.weight',
        'mask_fcn_logits.bias'
    ]

    cfg.merge_from_file(args.cfg)
    _d = load_c2_format(cfg, DETECTRON_PATH)
    newdict = _d

    newdict['model'] = removekey(_d['model'], keys_removed)
    torch.save(newdict, args.output)
    print('saved to {}.'.format(args.output))


if __name__ == "__main__":
    main()
