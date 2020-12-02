from detector.detector import *
from PIL import Image
import numpy as np
import cv2
import os
from .get_nets import PNet, RNet, ONet

class MTCNNFaceDetector():
    """
    This class load the MTCNN network and perform face detection.
    
    Attributes:
        model_path: path to the MTCNN weights files
    """
    def __init__(self):
    # LOAD MODELS
        self.pnet = PNet()
        self.rnet = RNet()
        self.onet = ONet()
        self.onet.eval()



    def detect_face(self, image, minsize=20, threshold=0.7, factor=0.709, use_auto_downscaling=True, min_face_area=25*25):
        image = Image.fromarray(image)
        if use_auto_downscaling:
            image, scale_factor = self.auto_downscale(image)

        faces, pnts =detect_faces(image,
                                  self.pnet,
                                  self.rnet,
                                  self.onet,
                 min_face_size=minsize,
                 thresholds=[0.6, 0.7, threshold],
                 nms_thresholds=[0.7, 0.7, 0.7],
                 factor = factor)
        
        faces = self.process_mtcnn_bbox(faces, np.array(image).shape)
        faces, pnts = self.remove_small_faces(faces, pnts, min_face_area)
        
        if use_auto_downscaling:
            faces = self.calibrate_coord(faces, scale_factor)
            pnts = self.calibrate_landmarks(pnts, scale_factor)
        return faces, pnts
    
    def auto_downscale(self, image):
        image = np.array(image)
        if self.is_higher_than_1080p(image):
            scale_factor = 4
            resized_image = cv2.resize(image, 
                                       (image.shape[1]//scale_factor, 
                                        image.shape[0]//scale_factor))
        elif self.is_higher_than_720p(image):
            scale_factor = 3
            resized_image = cv2.resize(image, 
                                       (image.shape[1]//scale_factor, 
                                        image.shape[0]//scale_factor))
        elif self.is_higher_than_480p(image):
            scale_factor = 2
            resized_image = cv2.resize(image, 
                                       (image.shape[1]//scale_factor, 
                                        image.shape[0]//scale_factor))
        else:
            scale_factor = 1
            resized_image = image.copy()
            
        resized_image = Image.fromarray(resized_image)
        return resized_image, scale_factor
    
    @staticmethod
    def is_higher_than_480p(x):
        return (x.shape[0] * x.shape[1]) >= (858*480)

    @staticmethod
    def is_higher_than_720p(x):
        return (x.shape[0] * x.shape[1]) >= (1280*720)

    @staticmethod
    def is_higher_than_1080p(x):
        return (x.shape[0] * x.shape[1]) >= (1920*1080)

    @staticmethod
    def process_mtcnn_bbox(bboxes, im_shape):
        # output bbox coordinate of MTCNN is (y0, x0, y1, x1)
        # Here we process the bbox coord. to a square bbox with ordering (x0, y1, x1, y0)
        for i, bbox in enumerate(bboxes):
            y0, x0, y1, x1 = bboxes[i,0:4]
            w = int(y1 - y0)
            h = int(x1 - x0)
            length = (w + h)/2
            center = (int((x1+x0)/2),int((y1+y0)/2))
            new_x0 = np.max([0, (center[0]-length//2)])#.astype(np.int32)
            new_x1 = np.min([im_shape[0], (center[0]+length//2)])#.astype(np.int32)
            new_y0 = np.max([0, (center[1]-length//2)])#.astype(np.int32)
            new_y1 = np.min([im_shape[1], (center[1]+length//2)])#.astype(np.int32)
            bboxes[i,0:4] = new_x0, new_y1, new_x1, new_y0
        return bboxes
    
    @staticmethod
    def calibrate_coord(faces, scale_factor):
        for i, (x0, y1, x1, y0, _) in enumerate(faces):
            faces[i] = (x0*scale_factor, y1*scale_factor, 
                        x1*scale_factor, y0*scale_factor, _)
        return faces

    @staticmethod
    def calibrate_landmarks(pnts, scale_factor):
        # pnts is a numpy array
        return np.array([xy * scale_factor for xy in pnts])
            
    @staticmethod
    def remove_small_faces(faces, pnts, min_area=25*25):
        def compute_area(face_coord):
            x0, y1, x1, y0, _ = face_coord
            area = np.abs((x1 - x0) * (y1 - y0))
            return area
            
        new_faces = []
        new_pnts = []
        # faces has shape (num_faces, coord), and pnts has shape (coord, num_faces)
        for face,pnt in zip(faces, pnts.transpose()):
            if compute_area(face) >= min_area:
                new_faces.append(face)
                new_pnts.append(pnt)
        new_faces = np.array(new_faces)
        new_pnts = np.array(new_pnts).transpose()
        return new_faces, new_pnts