from enum import Enum
import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image





class FaceEmotionEstimatorModels(Enum):

    KERAS = 0
    DEFAULT = KERAS


class FaceEmotionEstimator:

    def __init__(self, model=FaceEmotionEstimatorModels.DEFAULT, path=None):
        self._base = None
        if model == FaceEmotionEstimatorModels.KERAS:
            self._base = FaceEmotionEstimator_KERAS(path)

    def estimate(self, frame, face_image):
        return self._base.estimate(frame, face_image)


class FaceEmotionEstimator_KERAS:

    def __init__(self, path):
        self._classifier = model_from_json(open(path + 'emotion_deploy.json', "r").read())
        self._classifier.load_weights(path + 'emotion_net.h5')
        self._selection = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    def estimate(self, frame, face_image):
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        face_image = cv2.resize(face_image, (48, 48))
        img_pixels = np.expand_dims(image.img_to_array(face_image), axis = 0) / 255
        predictions = self._classifier.predict(img_pixels)
        max_index = np.argmax(predictions[0])
        return self._selection[max_index]

