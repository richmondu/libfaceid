from enum import Enum
import cv2





class FaceAgeEstimatorModels(Enum):

    CV2CAFFE = 0
    DEFAULT = CV2CAFFE


class FaceAgeEstimator:

    def __init__(self, model=FaceAgeEstimatorModels.DEFAULT, path=None):
        self._base = None
        if model == FaceAgeEstimatorModels.CV2CAFFE:
            self._base = FaceAgeEstimator_CV2CAFFE(path)

    def estimate(self, frame, face_image):
        return self._base.estimate(frame, face_image)


class FaceAgeEstimator_CV2CAFFE:

    def __init__(self, path):
        self._mean_values = (78.4263377603, 87.7689143744, 114.895847746)
        self._classifier = cv2.dnn.readNetFromCaffe(path + 'age_deploy.prototxt', path + 'age_net.caffemodel')
        self._selection = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']

    def estimate(self, frame, face_image):
        blob = cv2.dnn.blobFromImage(face_image, 1, (227, 227), self._mean_values, swapRB=False)
        self._classifier.setInput(blob)
        prediction = self._classifier.forward()
        return self._selection[prediction[0].argmax()]

