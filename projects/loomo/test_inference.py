import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import PIL
from PIL import Image

import torch
import torch.nn.functional as F

import cv2
import socket
import sys
import numpy
import struct
import binascii

import argparse

model = torch.hub.load('ultralytics/yolov5', 'custom', path='best3.pt')

img=cv2.imread('img.jpg')

results = model(img)
if(results.xyxy[0].nelement()!=0):
	x1 = results.xyxy[0][0][0].cpu().detach().numpy()
	y1 = results.xyxy[0][0][1].cpu().detach().numpy()
	x2 = results.xyxy[0][0][2].cpu().detach().numpy()
	y2 = results.xyxy[0][0][3].cpu().detach().numpy()

w = int(abs(x1-x2))
h = int(abs(y1-y2))
x = x1+w/2
y = y1-h/2

pred_bboxes = [x, y, w, h]
pred_y_label = 1

print(pred_bboxes, pred_y_label)
