import numpy as np
from enum import Enum
import cv2
from imutils import face_utils
import dlib # for FaceLivenessDetectorModels.EYESBLINK_MOUTHOPEN
#from scipy.spatial import distance as dist # for FaceLivenessDetectorModels.EYESBLINK_MOUTHOPEN





class FaceLivenessModels(Enum):

    EYESBLINK_MOUTHOPEN = 0
    DEFAULT             = EYESBLINK_MOUTHOPEN


class FaceLiveness:

    def __init__(self, model=FaceLivenessModels.DEFAULT, path=None):
        if model == FaceLivenessModels.EYESBLINK_MOUTHOPEN:
            self._base = FaceLiveness_EYESBLINK_MOUTHOPEN(path)

    def is_eyes_close(self, frame, face):
        return self._base.is_eyes_close(frame, face)

    def is_mouth_open(self, frame, face):
        return self._base.is_mouth_open(frame, face)

    def detect(self, frame, face, total_eye_blinks, eye_counter):
        return self._base.detect(frame, face, total_eye_blinks, eye_counter)

    def set_eye_threshold(self, threshold):
        self._base.set_eye_threshold(threshold)

    def get_eye_threshold(self):
        return self._base.get_eye_threshold()

    def set_mouth_threshold(self, threshold):
        self._base.set_mouth_threshold(threshold)

    def get_mouth_threshold(self):
        return self._base.get_mouth_threshold()


class FaceLiveness_EYESBLINK_MOUTHOPEN:

    _ear_threshold = 0.3 # eye aspect ratio (ear); less than this value, means eyes is close
    _mar_threshold = 0.3 # mouth aspect ratio (mar); more than this value, means mouth is open
    _ear_consecutive_frames = 1


    def __init__(self, path):
        # use dlib 68-point facial landmark
        self._detector = dlib.shape_predictor(path + 'shape_predictor_68_face_landmarks.dat')
        (self._leye_start, self._leye_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self._reye_start, self._reye_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        try:
            (self._mouth_start, self._mouth_end) = face_utils.FACIAL_LANDMARKS_IDXS["inner_mouth"]
        except:
            (self._mouth_start, self._mouth_end) = (60, 68)


    def is_eyes_close(self, frame, face):
        shape = self.get_shape(frame, face)
        # get the average eye aspect ratio (ear) of left eye and right eye
        average_ear = (self.eye_aspect_ratio(shape[self._leye_start:self._leye_end]) + self.eye_aspect_ratio(shape[self._reye_start:self._reye_end])) / 2.0
        return (average_ear < self._ear_threshold), average_ear

    def set_eye_threshold(self, threshold):
        self._ear_threshold = threshold

    def get_eye_threshold(self):
        return self._ear_threshold


    def is_mouth_open(self, frame, face):
        shape = self.get_shape(frame, face)
        # get the mouth aspect ratio (mar) of inner mouth
        mar = self.mouth_aspect_ratio(shape[self._mouth_start:self._mouth_end])
        return (mar > self._mar_threshold), mar

    def set_mouth_threshold(self, threshold):
        self._mar_threshold = threshold

    def get_mouth_threshold(self):
        return self._mar_threshold


    def detect(self, frame, face, total_eye_blinks, eye_counter):
        eyes_close = is_eyes_close(self, frame, face)
        if eyes_close:
            print("less than eye threshold {:.2f}".format(ear))
            eye_counter += 1
        else:
            if eye_counter >= self._ear_consecutive_frames:
                total_eye_blinks += 1
            eye_counter = 0
        return total_eye_blinks, eye_counter


    # private function
    def mouth_aspect_ratio(self, mouth):
        # (|m1-m7|+|m2-m6|+|m3-m5|) / (2|m0-m4|)
        # np.linalg.norm is faster than dist.euclidean
        return (np.linalg.norm(mouth[1]-mouth[7]) + np.linalg.norm(mouth[2]-mouth[6]) + np.linalg.norm(mouth[3]-mouth[5])) / (2.0 * np.linalg.norm(mouth[0]-mouth[4]))
        #return (dist.euclidean(mouth[1],mouth[7]) + dist.euclidean(mouth[2],mouth[6]) + dist.euclidean(mouth[3],mouth[5])) / (2.0 * dist.euclidean(mouth[0],mouth[4]))

    # private function
    def eye_aspect_ratio(self, eye):
        # https://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf
        # (|e1-e5|+|e2-e4|) / (2|e0-e3|)
        # np.linalg.norm is faster than dist.euclidean
        return (np.linalg.norm(eye[1]-eye[5]) + np.linalg.norm(eye[2]-eye[4])) / (2.0 * np.linalg.norm(eye[0]-eye[3]))
        #return (dist.euclidean(eye[1],eye[5]) + dist.euclidean(eye[2],eye[4])) / (2.0 * dist.euclidean(eye[0],eye[3]))

    # private function
    def get_shape(self, frame, face):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        (x, y, w, h) = face
        rect = dlib.rectangle(int(x), int(y), int(x+w), int(y+h))
        shape = self._detector(frame_gray, rect)
        coords = np.zeros((shape.num_parts, 2), dtype="int")
        for i in range(0, shape.num_parts):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords

