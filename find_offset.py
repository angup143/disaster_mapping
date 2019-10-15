"""
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_template_matching/py_template_matching.html
Good for images of intermediate steps
Usage: python find_offset.py --image_inlay <path to image inlay, used for saving the outputs> --pre_img <path to pre disaster road mask> --post_img <path to post disaster mask>
"""
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from skimage import measure
import argparse


def read_image(img_path):
    img = cv2.imread(img_path, 0)
    return img


def dilate_image(img, kernel_size=5, num_iterations=6):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    new_img = cv2.dilate(img, kernel, iterations=num_iterations)
    return new_img


def erode_image(img, kernel_size=5, num_iterations=6):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    new_img = cv2.erode(img, kernel, iterations=num_iterations)
    return new_img


def open_image(img, kernel_size=7, num_iterations=6):
    new_image = erode_image(img, kernel_size, num_iterations)
    new_image = dilate_image(new_image, kernel_size, num_iterations)
    return new_image


def convert_gray_to_colour(image):
    coloured_img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    coloured_img[image == 0, :] = (255, 255, 255)
    coloured_img[image == 29, :] = (0, 0, 255)
    coloured_img[image == 225, :] = (0, 255, 0)
    return coloured_img
