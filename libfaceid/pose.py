from enum import Enum
import cv2
import numpy as np
import dlib # for FacePoseEstimatorModels.DLIB68





color_blue   = (255,0,0)
color_green  = (0,255,0)
color_red    = (0,0,255)
color_yellow = (0,255,255)
color_pink   = (255,0,255)
color_white  = (255,255,255)
color_white  = (0,0,0)



class FacePoseEstimatorModels(Enum):

    DLIB68 = 0
    DEFAULT = DLIB68


class FacePoseEstimator():

    def __init__(self, model=FacePoseEstimatorModels.DEFAULT, path=None):
        self.path = path
        self.detector = None
        self.connection = None
        if model == FacePoseEstimatorModels.DLIB68:
            self.detector = dlib.shape_predictor(self.path + 'shape_predictor_68_face_landmarks.dat')
            self.connection = {(0,1), (1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (7,8), (8,9), (9,10),
              (10,11), (11,12), (12,13), (13,14), (14,15), (15,16), 
              (16,26), (26,25), (25,24), (24,23), (23,22), 
              (22,21), (21,20), (20,19), (19,18), (18,17), (17,0),
              (36,37), (37,38), (38,39), (39,40), (40,41), (41,36), # left eye
              (42,43), (43,44), (44,45), (45,46), (46,47), (47,42), # right eye
              (48,49), (49,50), (50,51), (51,52), (52,53), (53,54), (54,55), (55,56), (56,57), (57,58), (58,59), (59,48), # outer lip
              (60,61), (61,62), (62,63), (63,64), (64,65), (65,66), (66,67), (67,60), # inner lip
              (31,32), (32,33), (33,34), (34,35), (27,28), (28,29), (29,30), (30,33), (27,31), (27,35), # nose
              }


    def detect(self, frame, face):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        (x, y, w, h) = face
        rect = dlib.rectangle(int(x), int(y), int(x+w), int(y+h))
        shape = self.shape_to_np(self.detector(frame_gray, rect))
        return shape


    def apply(self, frame, shape):
        j = 0
        for (x, y) in shape:
            if j >= 36 and j <= 47:
                cv2.circle(frame, (x, y), 3, color_green, -1)
            elif j >= 48 and j <= 67:
                cv2.circle(frame, (x, y), 3, color_pink, -1)
            elif j >= 27 and j <= 35:
                cv2.circle(frame, (x, y), 3, color_red, -1)
            else:
                cv2.circle(frame, (x, y), 3, color_yellow, -1)
            j += 1
        for conn in self.connection:
            cv2.line(frame, (shape[conn[0]][0], shape[conn[0]][1]), (shape[conn[1]][0], shape[conn[1]][1]), color_yellow, 1)


    # private function
    def shape_to_np(self, shape, dtype="int"):
        coords = np.zeros((shape.num_parts, 2), dtype=dtype)
        for i in range(0, shape.num_parts):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords

