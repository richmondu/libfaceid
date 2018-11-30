import numpy as np
from enum import Enum
import cv2
import dlib                                # for FaceLivenessDetectorModels.EYEBLINKING
from scipy.spatial import distance as dist # for FaceLivenessDetectorModels.EYEBLINKING





class FaceLivenessDetectorModels(Enum):

    EYEBLINKING = 0


class FaceLiveness():

    def __init__(self, model=FaceLivenessDetectorModels.EYEBLINKING, path=None):
        self.model = model
        self.path = path
        self.detector = None


    def initialize(self):
        if self.model == FaceLivenessDetectorModels.EYEBLINKING:
            self.detector = dlib.shape_predictor(self.path + 'shape_predictor_68_face_landmarks.dat')


    def detect(self, frame, face, total_eye_blinks, eye_counter):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        (x, y, w, h) = face
        rect = dlib.rectangle(int(x), int(y), int(x+w), int(y+h))
        shape = self.shape_to_np(self.detector(frame_gray, rect))
        ear = (self.eye_aspect_ratio(shape[42:48]) + self.eye_aspect_ratio(shape[36:42])) / 2.0
        if ear < 0.3:
            print("less than eye threshold {:.2f}".format(ear))
            eye_counter += 1
        else:
            #print("more than eye threshold {:.2f}".format(ear))
            if eye_counter >= 1:
                total_eye_blinks += 1
            eye_counter = 0
        return total_eye_blinks, eye_counter


    # private function
    def eye_aspect_ratio(self, eye):
        return (dist.euclidean(eye[1], eye[5]) + dist.euclidean(eye[2], eye[4])) / (2.0 * dist.euclidean(eye[0], eye[3]))


    # private function
    def shape_to_np(self, shape, dtype="int"):
        coords = np.zeros((shape.num_parts, 2), dtype=dtype)
        for i in range(0, shape.num_parts):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords

