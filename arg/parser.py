# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# modified by sailfish009

import pathlib
import argparse
import cv2


def get_args():
    args = 




# def get_args(desc=""):
#     parser = argparse.ArgumentParser(description=desc)
#     cwd = pathlib.Path(__file__).parent.absolute()
#     parser.add_argument(
#         "--config-file",
#         # default= str(cwd) + "/configs/default/human_protein.yaml",
#         ### previous default config file
#         default= str(cwd) + "/configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml",
#         metavar="FILE",
#         help="path to config file",
#     )
#     parser.add_argument(
#         "--confidence-threshold",
#         type=float,
#         default=0.7,
#         help="Minimum score for the prediction to be shown",
#     )
#     parser.add_argument(
#         "--min-image-size",
#         type=int,
#         default=224,
#         help="Smallest size of the image to feed to the model. "
#             "Model was trained with 800, which gives best results",
#     )
#     parser.add_argument(
#         "--show-mask-heatmaps",
#         dest="show_mask_heatmaps",
#         help="Show a heatmap probability for the top masks-per-dim masks",
#         action="store_true",
#     )
#     parser.add_argument(
#         "--masks-per-dim",
#         type=int,
#         default=2,
#         help="Number of heatmaps per dimension to show",
#     )
#     parser.add_argument(
#         "opts",
#         help="Modify model config options using the command-line",
#         default=None,
#         nargs=argparse.REMAINDER,
#     )
# 
#     return parser.parse_args()
