import numpy as np
from enum import Enum
import cv2





class FaceDetectorModels(Enum):

    HAARCASCADE         = 0    # [ML] OpenCV Haar Cascade Classifier
    DLIBHOG             = 1    # [ML] DLIB HOG - Histogram of Oriented Gradients
    DLIBCNN             = 2    # [DL] DLIB CNN // Slow without GPU.
    SSDRESNET           = 3    # [DL] OpenCV SSD with ResNet-10
    MTCNN               = 4    # [DL] Tensorflow Multi-task Cascaded CNN (MTCNN)
    FACENET             = 5    # [DL] Tensorflow FaceNet's Multi-task Cascaded CNN (MTCNN)
    DEFAULT = HAARCASCADE


class FaceDetector:

    def __init__(self, model=FaceDetectorModels.DEFAULT, path=None, optimize=False, minfacesize=20):
        if optimize:
            minfacesize = max(180, minfacesize)
        else:
            minfacesize = minfacesize
        self._base = None

        if model == FaceDetectorModels.HAARCASCADE:
            self._base = FaceDetector_HAARCASCADE(path, optimize, minfacesize)
        elif model == FaceDetectorModels.DLIBHOG:
            self._base = FaceDetector_DLIBHOG(path, optimize, minfacesize)
        elif model == FaceDetectorModels.DLIBCNN:
            self._base = FaceDetector_DLIBCNN(path, optimize, minfacesize)
        elif model == FaceDetectorModels.SSDRESNET:
            self._base = FaceDetector_SSDRESNET(path, optimize, minfacesize)
        elif model == FaceDetectorModels.MTCNN:
            self._base = FaceDetector_MTCNN(path, optimize, minfacesize)
        elif model == FaceDetectorModels.FACENET:
            self._base = FaceDetector_FACENET(path, optimize, minfacesize)

    def detect(self, frame):
        return self._base.detect(frame)


class FaceDetector_HAARCASCADE:

    def __init__(self, path, optimize, minfacesize):
        self._optimize = optimize
        self._minfacesize = minfacesize
        self._detector = cv2.CascadeClassifier(path + 'haarcascade_frontalface_default.xml')

    def detect(self, frame):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.equalizeHist(frame_gray)
        return self._detector.detectMultiScale(frame_gray, 1.1, 5, minSize=(self._minfacesize, self._minfacesize))


class FaceDetector_DLIBHOG:

    def __init__(self, path, optimize, minfacesize):
        import dlib # lazy loading
        self._optimize = optimize
        self._minfacesize = minfacesize
        self._detector = dlib.get_frontal_face_detector()

    def detect(self, frame):
        frame_rgb = frame[:, :, ::-1]
        faces = self._detector(frame_rgb, 0)
        faces_updated = []
        for face in faces:
            (x, y, w, h) = (face.left(), face.top(), face.right()-face.left(), face.bottom()-face.top())
            faces_updated.append((x, y, w, h))
        return faces_updated


class FaceDetector_DLIBCNN:

    def __init__(self, path, optimize, minfacesize):
        import dlib # lazy loading
        self._optimize = optimize
        self._minfacesize = minfacesize
        self._detector = dlib.cnn_face_detection_model_v1(path + 'mmod_human_face_detector.dat')

    def detect(self, frame):
        frame_rgb = frame[:, :, ::-1]
        faces = self._detector(frame_rgb, 0)
        faces_updated = []
        for face in faces:
            (x, y, w, h) = (face.rect.left(), face.rect.top(), face.rect.right()-face.rect.left(), face.rect.bottom()-face.rect.top())
            faces_updated.append((x, y, w, h))
        return faces_updated


class FaceDetector_SSDRESNET:

    def __init__(self, path, optimize, minfacesize):
        self._optimize = optimize
        self._minfacesize = minfacesize
        self._detector = cv2.dnn.readNetFromCaffe(path + 'deploy.prototxt', path + 'res10_300x300_ssd_iter_140000.caffemodel')

    def detect(self, frame):
        if self._optimize:
            imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (150,150)), 1.0, (50, 50), (104.0, 177.0, 123.0), swapRB=False, crop=False)
        else:
            imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
        self._detector.setInput(imageBlob)
        faces = self._detector.forward()
        faces_filtered = []
        for index in range(faces.shape[2]):
            confidence = faces[0, 0, index, 2]
            if confidence > 0.5:
                box = faces[0, 0, index, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (x, y, x2, y2) = box.astype("int")
                (x, y, w, h) = (x, y, x2-x, y2-y)
                faces_filtered.append((x, y, w, h))
        return faces_filtered


class FaceDetector_MTCNN:

    def __init__(self, path, optimize, minfacesize):
        from mtcnn.mtcnn import MTCNN # lazy loading
        self._optimize = optimize
        self._minfacesize = minfacesize
        self._detector = MTCNN(min_face_size = minfacesize)

    def detect(self, frame):
        faces = self._detector.detect_faces(frame)
        faces_updated = []
        for face in faces:
            boxd = face['box']
            (x, y, w, h) = (boxd[0], boxd[1], boxd[2], boxd[3])
            faces_updated.append((x, y, w, h))
        return faces_updated


class FaceDetector_FACENET:

    # TODO: Add border and alignment
    _threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    _factor = 0.709  # scale factor

    def __init__(self, path, optimize, minfacesize):
        import tensorflow as tf                         # lazy loading
        import facenet.src.align.detect_face as facenet # lazy loading
        self._optimize = optimize
        self._minfacesize = minfacesize
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                self._pnet, self._rnet, self._onet = facenet.create_mtcnn(sess, None)

    def detect(self, frame):
        import facenet.src.align.detect_face as facenet # lazy loading
        faces, _ = facenet.detect_face(frame, self._minfacesize, 
            self._pnet, self._rnet, self._onet, self._threshold, self._factor)
        faces_updated = []
        for face in faces:
            face = face.astype("int")
            (x, y, w, h) = (max(face[0], 0), max(face[1],0), 
                min(face[2],frame.shape[1])-max(face[0],0), 
                min(face[3],frame.shape[0])-max(face[1],0) )
            faces_updated.append((x, y, w, h))
        return faces_updated
