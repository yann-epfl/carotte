import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import PIL
from PIL import Image

import torch
import torch.nn.functional as F
import scipy.linalg
import cv2

from torchreid.utils import FeatureExtractor
import math

chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}

def cosine_sim(A,B):
    cosine = np.dot(A,B)/(norm(A)*norm(B))
    return cosine

def euclid_dist(x1,y1,x2,y2):
    dist = math.sqrt((x1-x2)**2+(y1-y2)**2)
    return dist

class KalmanFilter(object):
    """
    A simple Kalman filter for tracking bounding boxes in image space.
    The 8-dimensional state space
        x, y, a, h, vx, vy, va, vh
    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.
    Object motion follows a constant velocity model. The bounding box location
    (x, y, a, h) is taken as direct observation of the state space (linear
    observation model).
    """

    def __init__(self):
        ndim, dt = 4, 1.

        # Create Kalman filter model matrices.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement):
        """Create track from unassociated measurement.
        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h) with center position (x, y),
            aspect ratio a, and height h.
        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[0],   # the center point x
            2 * self._std_weight_position * measurement[1],   # the center point y
            1 * measurement[2],                               # the ratio of width/height
            2 * self._std_weight_position * measurement[3],   # the height
            10 * self._std_weight_velocity * measurement[0],
            10 * self._std_weight_velocity * measurement[1],
            0.1 * measurement[2],
            10 * self._std_weight_velocity * measurement[3]]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """Run Kalman filter prediction step.
        Parameters
        ----------
        mean : ndarray
            The 8 dimensional mean vector of the object state at the previous
            time step.
        covariance : ndarray
            The 8x8 dimensional covariance matrix of the object state at the
            previous time step.
        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.
        """
        std_pos = [
            self._std_weight_position * mean[0],
            self._std_weight_position * mean[1],
            1 * mean[2],
            self._std_weight_position * mean[3]]
        std_vel = [
            self._std_weight_velocity * mean[0],
            self._std_weight_velocity * mean[1],
            0.1 * mean[2],
            self._std_weight_velocity * mean[3]]

        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        return mean, covariance

    def project(self, mean, covariance):
        """Project state distribution to measurement space.
        Parameters
        ----------
        mean : ndarray
            The state's mean vector (8 dimensional array).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.
        """
        std = [
            self._std_weight_position * mean[0],
            self._std_weight_position * mean[1],
            0.1 * mean[2],
            self._std_weight_position * mean[3]]
        innovation_cov = np.diag(np.square(std))
        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement):
        """Run Kalman filter correction step.
        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        measurement : ndarray
            The 4 dimensional measurement vector (x, y, a, h), where (x, y)
            is the center position, a the aspect ratio, and h the height of the
            bounding box.
        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.
        """
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements,
                        only_position=False):
        """Compute gating distance between state distribution and measurements.
        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.
        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional).
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.
        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.
        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        cholesky_factor = np.linalg.cholesky(covariance)
        d = measurements - mean
        z = scipy.linalg.solve_triangular(
            cholesky_factor, d.T, lower=True, check_finite=False,
            overwrite_b=True)
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output, n_c):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.box = torch.nn.Linear(n_hidden, n_output-1)   # output layer
        self.logit = torch.nn.Linear(n_hidden, 1)
        
        self.conv1 = torch.nn.Sequential(         # input shape (3, 80, 60)
            torch.nn.Conv2d(
                in_channels = n_c,            # input height
                out_channels = 8,             # n_filters
                kernel_size = 5,              # filter size
                stride = 2,                   # filter movement/step
                padding = 0,                  
            ),                              
            torch.nn.ReLU(),                      # activation
            #torch.nn.MaxPool2d(kernel_size = 2),    
        )
        self.conv2 = torch.nn.Sequential(       
            torch.nn.Conv2d(in_channels = 8, 
                            out_channels = 16, 
                            kernel_size = 5, 
                            stride = 2, 
                            padding = 0),      
            torch.nn.ReLU(),                      # activation
            #torch.nn.MaxPool2d(2),                
        )
        
        self.conv3 = torch.nn.Sequential(       
            torch.nn.Conv2d(in_channels = 16, 
                            out_channels = 8, 
                            kernel_size = 1, 
                            stride = 1, 
                            padding = 0),      
            torch.nn.ReLU(),                      # activation
            #torch.nn.MaxPool2d(2),                
        )
    def forward(self, x):
        feat = self.conv1(x)
        feat = self.conv2(feat)
        feat = self.conv3(feat)
        feat = feat.view(feat.size(0), -1)
        x2 = F.relu(self.hidden(feat))      # activation function for hidden layer
        
        out_box = F.relu(self.box(x2))            # linear output
        out_logit = torch.sigmoid(self.logit(x2))
        
        return out_box, out_logit
        
class Detector(object):
    """docstring for Detector"""
    def __init__(self):
        super(Detector, self).__init__()
        # TODO: MEAN & STD
        #self.mean = [[[[0.5548078,  0.56693329, 0.53457436]]]] 
        #self.std = [[[[0.26367019, 0.26617227, 0.25692861]]]]
        #self.img_size = 100 
        #self.img_size_w = 80
        #self.img_size_h = 60
        #self.min_object_size = 10
        #self.max_object_size = 40 
        #self.num_objects = 1
        #self.num_channels = 3
        #self.model = Net(n_feature = 1632, n_hidden = 128, n_output = 5, n_c = 3)     # define the network

        self.model_best = torch.hub.load('ultralytics/yolov5', 'custom', path='good_logo.pt')
        self.model_p = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        self.init = 0
        self.kf = KalmanFilter()
        self.lost_count = 0
        self.lost = 0

        self.extractor = FeatureExtractor(
            model_name='osnet_x1_0',
            model_path='a/b/c/model.pth.tar',
            device='cuda'
        )

        self.mean = 0
        self.cov = []
        self.measurement = []


    def load(self, PATH):
        # self.model = torch.load(PATH)
        # self.model.eval()

        self.model.load_state_dict(torch.load(PATH))
        self.model.eval()

    def forward(self, img):   
        

        if not(self.init):
            results = self.model_best(img)
            if(results.xyxy[0].nelement()!=0):
                x1 = results.xyxy[0][0][0].cpu().detach().numpy()
                y1 = results.xyxy[0][0][1].cpu().detach().numpy()
                x2 = results.xyxy[0][0][2].cpu().detach().numpy()
                y2 = results.xyxy[0][0][3].cpu().detach().numpy()

                x_star=int((abs(x2-x1)/2)+x1)
                y_star=int((abs(y2-y1)/2)+y1)

                results_p = self.model_p(img)
                for i in range(int(results_p.xyxy[0].nelement()/6)): #for on the number of person detected
                  if (results_p.xyxy[0][i][4]>0.6) and (results_p.xyxy[0][i][5]==0): #if class person and confidence over 60%
                    x1 = int(results_p.xyxy[0][i][0].cpu().detach().numpy())
                    y1 = int(results_p.xyxy[0][i][1].cpu().detach().numpy())
                    x2 = int(results_p.xyxy[0][i][2].cpu().detach().numpy())
                    y2 = int(results_p.xyxy[0][i][3].cpu().detach().numpy())

                    w = int(abs(x1-x2))
                    h = int(abs(y1-y2))
                    x = x1+w/2
                    y = y1+h/2
                    a = w/h

                    if (x_star in range(x1,x2) and (y_star in range(y1,y2))):
                        #bbox_array = cv2.rectangle(bbox_array,(x1,y1),(x1+w,y1+h),(255,0,0),2)

                        #bbox_array[:,:,3] = (bbox_array.max(axis = 2) > 0 ).astype(int) * 255

                        #img_cropped = img.crop((x1,y1,x2,y2)) #(left, top, right, bottom)
                        #cv2_imshow(img_cropped)
                        #features = self.extractor(img_cropped)
                        #features_init = features.cpu().numpy()[0]
                        #print(features_init.cpu().numpy()[0])

                        #initiate Kalman filter
                        self.measurement = [x, y, a, h]
                        self.mean, self.cov = self.kf.initiate(self.measurement)
                        self.init = 1

                        bbox = [x, y, w, h]
                        label = [1]
                        return bbox, label

        elif self.init:
            reid_measurement_found = 0
            kalman_measurement_found = 0
            mean_pred, cov_pred = self.kf.predict(self.mean, self.cov)

            #calc pred bbox parameters
            x_pred = mean_pred[0]
            y_pred = mean_pred[1]
            a_pred = mean_pred[2]
            h_pred = mean_pred[3]
            w_pred = a_pred*h_pred
            x1_pred = int(x_pred-w_pred/2)
            y1_pred = int(y_pred-h_pred/2)
            x2_pred = int(x_pred+w_pred/2)
            y2_pred = int(y_pred+h_pred/2)

            #add pred bbox
            #bbox_array = cv2.rectangle(bbox_array,(x1_pred,y1_pred),(x2_pred,y2_pred),(0,0,255),2)
            #bbox_array[:,:,3] = (bbox_array.max(axis = 2) > 0 ).astype(int) * 255

            results_p = self.model_p(img)
            for i in range(int(results_p.xyxy[0].nelement()/6)):
                if (results_p.xyxy[0][i][4]>0.6) and (results_p.xyxy[0][i][5]==0):
                    x1 = int(results_p.xyxy[0][i][0].cpu().detach().numpy())
                    y1 = int(results_p.xyxy[0][i][1].cpu().detach().numpy())
                    x2 = int(results_p.xyxy[0][i][2].cpu().detach().numpy())
                    y2 = int(results_p.xyxy[0][i][3].cpu().detach().numpy())

                    w = int(abs(x1-x2))
                    h = int(abs(y1-y2))
                    x = x1+w/2
                    y = y1+h/2
                    a = w/h

                    #img_cropped = img[y1:y2,x1:x2]
                    #features = self.extractor(img_cropped)
                    #features_new = features.cpu().numpy()[0]
                    #similarity = cosine_sim(features_init,features_new)
                    #print(i)
                    #print(similarity)
                    #cv2_imshow(img_cropped)
                    #if (similarity>0.80): #and (euclid_dist(x,y,x_pred,y_pred)<300):

                        #measurement = [x, y, a, h]
                        #reid_measurement_found = 1
                        #self.lost_count = 0

                        #bbox_array = cv2.rectangle(bbox_array,(x1,y1),(x1+w,y1+h),(255,0,0),2)
                        #bbox_array[:,:,3] = (bbox_array.max(axis = 2) > 0 ).astype(int) * 255
              
                elif(euclid_dist(x,y,x_pred,y_pred)<100) and not(self.lost):

                    self.measurement = [x, y, a, h]
                    kalman_measurement_found = 1
                    self.lost_count = 0

                    #bbox_array = cv2.rectangle(bbox_array,(x1,y1),(x1+w,y1+h),(255,255,0),2)
                    #bbox_array[:,:,3] = (bbox_array.max(axis = 2) > 0 ).astype(int) * 255

            if(not(kalman_measurement_found)):
                self.lost_count += 1

            if(self.lost_count>=5):
                self.lost = 1
            else:
                self.lost = 0

            self.mean, self.cov = self.kf.update(mean_pred, cov_pred, self.measurement)
            #calc update bbox parameters
            x = self.mean[0]
            y = self.mean[1]
            a = self.mean[2]
            h = self.mean[3]
            w = a*h
            x1 = int(x-w/2)
            y1 = int(y-h/2)
            x2 = int(x+w/2)
            y2 = int(y+h/2)

            #add update bbox
            if (not(self.lost)):
                bbox = [x, y, w, h]
                label = [1]
                return bbox, label

        # if no person detected or lost
        bbox = [1, 1, 1, 1]
        label = [0]       #label < 0.5 stop the robpt
        return bbox, label