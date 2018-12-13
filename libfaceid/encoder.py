import os
import numpy as np
from enum import Enum
import cv2                     # for FaceEncoderModels.LBPH, FaceEncoderModels.OPENFACE
import pickle                  # for FaceEncoderModels.OPENFACE and FaceEncoderModels.DLIBRESNET
from imutils import paths      # for FaceEncoderModels.LBPH
from sklearn.preprocessing import LabelEncoder # for FaceEncoderModels
import dlib                    # for FaceEncoderModels.DLIBRESNET
from libfaceid.classifier import FaceClassifierModels, FaceClassifier
import tensorflow as tf        # for FaceDetector_FACENET
import facenet.src.facenet as facenet # for FaceDetector_FACENET
import math # for FaceDetector_FACENET
from scipy import misc # for FaceDetector_FACENET





OUTPUT_LBPH_CLASSIFIER     = 'lbph.yml'
OUTPUT_LBPH_LABELER        = 'lbph_le.pickle'

INPUT_OPENFACE_MODEL       = 'openface_nn4.small2.v1.t7'
OUTPUT_OPENFACE_CLASSIFIER = 'openface_re.pickle'
OUTPUT_OPENFACE_LABELER    = 'openface_le.pickle'

INPUT_DLIBRESNET_MODEL       = 'dlib_face_recognition_resnet_model_v1.dat'
INPUT_DLIBRESNET_MODEL2      = 'shape_predictor_5_face_landmarks.dat'
OUTPUT_DLIBRESNET_CLASSIFIER = 'dlib_re.pickle'
OUTPUT_DLIBRESNET_LABELER    = 'dlib_le.pickle'

INPUT_FACENET_MODEL          = 'facenet_20180402-114759'
OUTPUT_FACENET_CLASSIFIER    = 'facenet_re.pickle'
OUTPUT_FACENET_LABELER       = 'facenet_le.pickle'


class FaceEncoderModels(Enum):

    LBPH                = 0    # [ML] LBPH Local Binary Patterns Histograms
    OPENFACE            = 1    # [DL] OpenCV OpenFace
    DLIBRESNET          = 2    # [DL] DLIB ResNet
    FACENET             = 3    # [DL] FaceNet implementation by David Sandberg
    # VGGFACE1_VGG16    = 4    # Refer to models\others\vggface_recognition
    # VGGFACE2_RESNET50 = 5    # Refer to models\others\vggface_recognition
    DEFAULT = LBPH


class FaceEncoder():

    def __init__(self, model=FaceEncoderModels.DEFAULT, path=None, path_training=None, training=False):
        self._base = None
        if model == FaceEncoderModels.LBPH:
            self._base = FaceEncoder_LBPH(path, path_training, training)
        elif model == FaceEncoderModels.OPENFACE:
            self._base = FaceEncoder_OPENFACE(path, path_training, training)
        elif model == FaceEncoderModels.DLIBRESNET:
            self._base = FaceEncoder_DLIBRESNET(path, path_training, training)
        elif model == FaceEncoderModels.FACENET:
            self._base = FaceEncoder_FACENET(path, path_training, training)

    def identify(self, frame, face_rect):
        try:
            return self._base.identify(frame, face_rect)
        except:
            return "Unknown", 0

    def train(self, face_detector, path_dataset, verify=False, classifier=FaceClassifierModels.LINEAR_SVM):
        self._base.train(face_detector, path_dataset, verify, classifier)


class FaceEncoder_Utils():

    def save_training(self, classifier, knownNames, knownEmbeddings, output_clf, output_le):
        #print(len(knownNames))
        #print(len(knownEmbeddings))
        #print("[INFO] Number of classes = {}".format(knownNames))

        le = LabelEncoder()
        labels = le.fit_transform(knownNames)
        #print(le.classes_)
        #print(labels)

        clf = FaceClassifier(classifier)
        clf.fit(knownEmbeddings, labels)

        f = open(output_clf, "wb")
        f.write(pickle.dumps(clf))
        f.close()

        f = open(output_le, "wb")
        f.write(pickle.dumps(le))
        f.close()


class FaceEncoder_LBPH():

    def __init__(self, path=None, path_training=None, training=False):
        self.path_training = path_training
        self.clf = None
        self.embedder = None
        self.label_encoder = None
        self.shaper = None

        self.clf = cv2.face.LBPHFaceRecognizer_create()
        if training == False:
            self.clf.read(self.path_training + OUTPUT_LBPH_CLASSIFIER)
            self.label_encoder = pickle.loads(open(self.path_training + OUTPUT_LBPH_LABELER, "rb").read())
            #print(self.label_encoder.classes_)

    def identify(self, frame, face_rect):
        face_id = "Unknown"
        confidence = 99.99
        (x, y, w, h) = face_rect
        frame_gray = frame[y:y+h, x:x+w]
        face = cv2.cvtColor(frame_gray, cv2.COLOR_BGR2GRAY)
        id, confidence = self.clf.predict(face)
        if confidence > 99.99: 
            confidence = 99.99
        face_id = self.label_encoder.classes_[id]
        return face_id, confidence

    def train(self, face_detector, path_dataset, verify, classifier):
        imagePaths = sorted(list(paths.list_images(path_dataset)))
        faceSamples=[]
        ids = []
        knownNames = []

        id = -1
        for (i, imagePath) in enumerate(imagePaths):
            frame = cv2.imread(imagePath, cv2.IMREAD_COLOR)
            name = imagePath.split(os.path.sep)[-2]
            try:
                id = knownNames.index(name)
            except:
                id = id + 1
            #print("name=%s id=%d" % (name, id))

            # FACE DETECTION
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detect(frame)
            for (index, face) in enumerate(faces):
                (x, y, w, h) = face
                faceSamples.append(frame_gray[y:y+h,x:x+w])
                knownNames.append(name)
                ids.append(id)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 1)
                break

            if verify and len(faces) != 1:
                print("\n [INFO] Image {} has {} faces ".format(imagePath, len(faces)))
                cv2.imshow('frame', frame)
                cv2.waitKey(1000)
        #print(ids)
        #print(knownNames)

        self.clf.train(faceSamples, np.array(ids))
        self.clf.write(self.path_training + OUTPUT_LBPH_CLASSIFIER)

        le = LabelEncoder()
        labels = le.fit_transform(knownNames)
        #print(le.classes_)
        #print(labels)
        
        f = open(self.path_training + OUTPUT_LBPH_LABELER, "wb")
        f.write(pickle.dumps(le))
        f.close()


class FaceEncoder_OPENFACE():

    def __init__(self, path=None, path_training=None, training=False):
        self.path_training = path_training
        self.clf = None
        self.embedder = None
        self.label_encoder = None
        self.shaper = None

        self.embedder = cv2.dnn.readNetFromTorch(path + INPUT_OPENFACE_MODEL)
        if training == False:
            self.clf = pickle.loads(open(self.path_training + OUTPUT_OPENFACE_CLASSIFIER, "rb").read())
            self.label_encoder = pickle.loads(open(self.path_training + OUTPUT_OPENFACE_LABELER, "rb").read())
            #print(self.label_encoder.classes_)

    def identify(self, frame, face_rect):
        face_id = "Unknown"
        confidence = 99.99
        vec = self.encode(frame, face_rect)
        predictions_face = self.clf.predict(vec)[0]
        id = np.argmax(predictions_face)
        confidence = predictions_face[id] * 100
        face_id = self.label_encoder.classes_[id]
        return face_id, confidence

    def encode(self, frame, face_rect):
        (x, y, w, h) = face_rect
        face = frame[y:y+h, x:x+w]
        faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
        self.embedder.setInput(faceBlob)
        return self.embedder.forward()

    def train(self, face_detector, path_dataset, verify, classifier):
        knownEmbeddings = []
        knownNames = []
        total = 0

        imagePaths = sorted(list(paths.list_images(path_dataset)))
        for (j, imagePath) in enumerate(imagePaths):
            name = imagePath.split(os.path.sep)[-2]
            frame = cv2.imread(imagePath, cv2.IMREAD_COLOR)
            faces = face_detector.detect(frame)
            for face in faces:
                vec = self.encode(frame, face)
                knownNames.append(name)
                knownEmbeddings.append(vec.flatten())
                total += 1

        FaceEncoder_Utils().save_training(classifier, knownNames, knownEmbeddings, 
            self.path_training + OUTPUT_OPENFACE_CLASSIFIER, 
            self.path_training + OUTPUT_OPENFACE_LABELER)


class FaceEncoder_DLIBRESNET():


    def __init__(self, path=None, path_training=None, training=False):
        self.path_training = path_training
        self.clf = None
        self.embedder = None
        self.label_encoder = None
        self.shaper = None

        self.embedder = dlib.face_recognition_model_v1(path + INPUT_DLIBRESNET_MODEL)
        self.shaper = dlib.shape_predictor(path + INPUT_DLIBRESNET_MODEL2)
        if training == False:
            self.clf = pickle.loads(open(self.path_training + OUTPUT_DLIBRESNET_CLASSIFIER, "rb").read())
            self.label_encoder = pickle.loads(open(self.path_training + OUTPUT_DLIBRESNET_LABELER, "rb").read())
            #print(self.label_encoder.classes_)

    def identify(self, frame, face_rect):
        face_id = "Unknown"
        confidence = 99.99
        vec = self.encode(frame, face_rect)
        predictions_face = self.clf.predict(vec)[0]
        #print(predictions_face)
        id = np.argmax(predictions_face)
        confidence = predictions_face[id] * 100
        face_id = self.label_encoder.classes_[id]
        return face_id, confidence

    def encode(self, frame, face_rect):
        (x, y, w, h) = face_rect
        rect = dlib.rectangle(x, y, x+w, y+h)
        frame_rgb = frame[:, :, ::-1]
        shape = self.shaper(frame_rgb, rect)
        vec = self.embedder.compute_face_descriptor(frame_rgb, shape)
        return np.array([vec])

    def train(self, face_detector, path_dataset, verify, classifier):
        knownEmbeddings = []
        knownNames = []
        total = 0

        imagePaths = sorted(list(paths.list_images(path_dataset)))
        for (j, imagePath) in enumerate(imagePaths):
            name = imagePath.split(os.path.sep)[-2]
            frame = cv2.imread(imagePath, cv2.IMREAD_COLOR)
            faces = face_detector.detect(frame)
            for face in faces:
                vec = self.encode(frame, face)
                knownNames.append(name)
                knownEmbeddings.append(vec.flatten())
                total += 1

        FaceEncoder_Utils().save_training(classifier, knownNames, knownEmbeddings, 
            self.path_training + OUTPUT_DLIBRESNET_CLASSIFIER, 
            self.path_training + OUTPUT_DLIBRESNET_LABELER)


class FaceEncoder_FACENET():

    _face_crop_size=160

    def __init__(self, path=None, path_training=None, training=False):
        self.path_training = path_training
        self._sess = tf.Session()
        with self._sess.as_default():
            facenet.load_model(path + INPUT_FACENET_MODEL)
        if training == False:
            self.clf = pickle.loads(open(self.path_training + OUTPUT_FACENET_CLASSIFIER, "rb").read())
            self.label_encoder = pickle.loads(open(self.path_training + OUTPUT_FACENET_LABELER, "rb").read())
            #print(self.label_encoder.classes_)

    def identify(self, frame, face_rect):
        vec = self.encode(frame, face_rect)
        predictions_face = self.clf.predict([vec])[0]
        id = np.argmax(predictions_face)
        confidence = predictions_face[id] * 100
        face_id = self.label_encoder.classes_[id]
        return face_id, confidence

    def encode(self, frame, face_rect):
        (x, y, w, h) = face_rect
        face = misc.imresize(frame[y:y+h, x:x+w, :], (self._face_crop_size, self._face_crop_size), interp='bilinear')
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        prewhiten_face = facenet.prewhiten(face)
        feed_dict = {images_placeholder: [prewhiten_face], phase_train_placeholder: False}
        return self._sess.run(embeddings, feed_dict=feed_dict)[0]

    def train(self, face_detector, path_dataset, verify, classifier):
        knownEmbeddings = []
        knownNames = []
        total = 0

        imagePaths = sorted(list(paths.list_images(path_dataset)))
        for (j, imagePath) in enumerate(imagePaths):
            name = imagePath.split(os.path.sep)[-2]
            frame = cv2.imread(imagePath, cv2.IMREAD_COLOR)
            faces = face_detector.detect(frame)
            for face in faces:
                vec = self.encode(frame, face)
                knownNames.append(name)
                knownEmbeddings.append(vec.flatten())
                total += 1

        FaceEncoder_Utils().save_training(classifier, knownNames, knownEmbeddings, 
            self.path_training + OUTPUT_FACENET_CLASSIFIER, 
            self.path_training + OUTPUT_FACENET_LABELER)

