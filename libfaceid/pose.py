from enum import Enum
import cv2
import numpy as np
import dlib # for FacePoseEstimatorModels.DLIB68





color_green  = (0,255,0)
color_blue   = (255,0,0)
color_red    = (0,0,255)
color_yellow = (0,255,255)
color_pink   = (255,0,255)
color_white  = (255,255,255)
color_black  = (0,0,0)



class FacePoseEstimatorModels(Enum):

    DLIB68 = 0
    DEFAULT = DLIB68


class FacePoseEstimatorColor(Enum):

    GREEN  = 0
    BLUE   = 1
    RED    = 2
    YELLOW = 3
    PINK   = 4
    WHITE  = 5
    BLACK  = 6
    DEFAULT = GREEN


class FacePoseEstimatorOverlay(Enum):

    ORIG   = 0
    OZ     = 1
    INT    = 2
    INTOZ  = 3
    DEFAULT = ORIG


class FacePoseEstimator():

    def __init__(self, model=FacePoseEstimatorModels.DEFAULT, path=None, overlay=FacePoseEstimatorOverlay.DEFAULT, color=FacePoseEstimatorColor.DEFAULT):
        self._base = None
        colors = [color_green, color_blue, color_red, color_yellow, color_pink, color_white, color_black]
        if model == FacePoseEstimatorModels.DLIB68:
            self._base = FacePoseEstimator_DLIB68(path, overlay, colors[color.value])

    def detect(self, frame, face):
        return self._base.detect(frame, face)

    def add_overlay(self, frame, shape):
        return self._base.add_overlay(frame, shape)


class FacePoseEstimator_DLIB68():

    def __init__(self, path, overlay, color):
        self._overlay = overlay
        self._color = color
        self._detector = dlib.shape_predictor(path + 'shape_predictor_68_face_landmarks.dat')
        if self._overlay == FacePoseEstimatorOverlay.OZ:
            self._connection = {(0,1), (1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (7,8), (8,9), (9,10), (10,11), (11,12), (12,13), (13,14), (14,15), (15,16), 
                (16,26), (26,25), (25,24), (24,23), (23,22), 
                (22,21), (21,20), (20,19), (19,18), (18,17), (17,0),
                (36,37), (37,38), (38,39), (39,40), (40,41), (41,36), # left eye
                (42,43), (43,44), (44,45), (45,46), (46,47), (47,42), # right eye
                (48,49), (49,50), (50,51), (51,52), (52,53), (53,54), (54,55), (55,56), (56,57), (57,58), (58,59), (59,48), # outer lip
                (60,61), (61,62), (62,63), (63,64), (64,65), (65,66), (66,67), (67,60), # inner lip
                (31,32), (32,33), (33,34), (34,35), (27,28), (28,29), (29,30), (30,33), (27,31), (27,35), # nose
            }
        elif self._overlay == FacePoseEstimatorOverlay.ORIG:
            self._connection = {(0,1), (1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (7,8), (8,9), (9,10), (10,11), (11,12), (12,13), (13,14), (14,15), (15,16), 
                (26,25), (25,24), (24,23), (23,22), 
                (21,20), (20,19), (19,18), (18,17),
                (36,37), (37,38), (38,39), (39,40), (40,41), (41,36), # left eye
                (42,43), (43,44), (44,45), (45,46), (46,47), (47,42), # right eye
                (48,49), (49,50), (50,51), (51,52), (52,53), (53,54), (54,55), (55,56), (56,57), (57,58), (58,59), (59,48), # outer lip
                (60,61), (61,62), (62,63), (63,64), (64,65), (65,66), (66,67), (67,60), # inner lip
                (31,32), (32,33), (33,34), (34,35), (27,28), (28,29), (29,30), (30,31), (30,35) # nose
            }
        elif self._overlay == FacePoseEstimatorOverlay.INT or self._overlay == FacePoseEstimatorOverlay.INTOZ:
            self._connection = {(1,4), (4,7), (7,9), (9,12), (12, 15), (1, 36), (15,45), (1, 31), (15,45), (15,35), (39, 42),
                (36,37), (37,38), (38,39), (39,40), (40,41), (41,36), # left eye
                (42,43), (43,44), (44,45), (45,46), (46,47), (47,42), # right eye
                (48,51), (51,54), (54,57), (57,48), (48, 4), (54,12), # outer lip
                (30,31), (31,35), (35,30), (30,39), (30,42), (31,48), (35,54), # nose
                (48, 7), (7, 57), (57, 9), (9,54),  (31, 39), (35, 42), (31, 36), (35, 45), (31, 51), (51, 35), (48, 1), (54,15),
            }

    def detect(self, frame, face):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        (x, y, w, h) = face
        rect = dlib.rectangle(int(x), int(y), int(x+w), int(y+h))
        shape = self.shape_to_np(self._detector(frame_gray, rect))
        return shape

    def add_overlay(self, frame, shape):
        if self._overlay == FacePoseEstimatorOverlay.OZ:
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
            for conn in self._connection:
                cv2.line(frame, (shape[conn[0]][0], shape[conn[0]][1]), (shape[conn[1]][0], shape[conn[1]][1]), color_yellow, 1)
        elif self._overlay == FacePoseEstimatorOverlay.ORIG:
            for conn in self._connection:
                cv2.line(frame, (shape[conn[0]][0], shape[conn[0]][1]), (shape[conn[1]][0], shape[conn[1]][1]), self._color, 1)
        elif self._overlay == FacePoseEstimatorOverlay.INT:
#            cv2.polylines(frame, [np.array(self._connection.values, np.int32)], True, color_white, thickness=3)
            for conn in self._connection:
                cv2.circle(frame, (shape[conn[0]][0], shape[conn[0]][1]), 2, self._color, -1, cv2.LINE_AA)
                cv2.line(frame, (shape[conn[0]][0], shape[conn[0]][1]), (shape[conn[1]][0], shape[conn[1]][1]), self._color, 1, cv2.LINE_AA)
        elif self._overlay == FacePoseEstimatorOverlay.INTOZ:
            for conn in self._connection:
                cv2.line(frame, (shape[conn[0]][0], shape[conn[0]][1]), (shape[conn[1]][0], shape[conn[1]][1]), color_yellow, 1)

    # private function
    def shape_to_np(self, shape, dtype="int"):
        coords = np.zeros((shape.num_parts, 2), dtype=dtype)
        for i in range(0, shape.num_parts):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords
