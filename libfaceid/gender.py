from enum import Enum
import cv2





class FaceGenderEstimatorModels(Enum):

    CV2CAFFE            = 0


class FaceGenderEstimator:

    def __init__(self, model=FaceGenderEstimatorModels.CV2CAFFE, path=None):
        self._base = None
        if model == FaceGenderEstimatorModels.CV2CAFFE:
            self._base = FaceGenderEstimator_CV2CAFFE(path)

    def estimate(self, frame, face_image):
        return self._base.estimate(frame, face_image)


class FaceGenderEstimator_CV2CAFFE:

    def __init__(self, path):
        self._mean_values = (78.4263377603, 87.7689143744, 114.895847746)
        self._classifier = cv2.dnn.readNetFromCaffe(path + 'gender_deploy.prototxt', path + 'gender_net.caffemodel')
        self._selection = ['Male', 'Female']

    def estimate(self, frame, face_image):
        blob = cv2.dnn.blobFromImage(face_image, 1, (227, 227), self._mean_values, swapRB=False)
        self._classifier.setInput(blob)
        prediction = self._classifier.forward()
        return self._selection[prediction[0].argmax()]

