libfaceid

    Python library for facial recognition developed for prototyping of PanL-based Facial Recognition solutions.


Facial Recognition

    libfaceid is a python library for facial recognition.
    It simplifies development of facial recognition systems by providing different models for detection and identification/encoding.
    Multiple models for detection and identification/encoding are supported.
    The models are seamlessly integrated so that user can mix and match detection models with identification/encoder models.
    Each model differs in speed, accuracy, memory requirements and 3rd-party library dependencies.
    This enables users to choose the appropriate model for their specific use cases and system requirements.
    The library and example applications have been tested on Windows 7 and Raspberry Pi 3B+.


Supported Models:

    Face Detector Models
        - Haar Cascade Classifier via OpenCV
        - Histogram of Oriented Gradients (HOG) via DLIB
        - Deep Neural Network via DLIB
        - Single Shot Detector with ResNet-10 via OpenCV
        - Multi-task Cascaded CNN (MTCNN) via Tensorflow

    Face Encoder/Embedder for Identification
        - Local Binary Patterns Histograms (LBPH) via OpenCV
        - OpenFace via OpenCV
        - ResNet via DLIB

    Other models are provided as stand-alone applications. Integration to libfaceid can be done in the future.
        - VGG-Face (VGG-16) via Keras
        - VGG-Face2 (ResNet-50) via Keras
        - FaceNet (Inception ResNet v1) via Tensorflow


Classes:

    detector.py - module for Face Detector models
        - class FaceDetectorModels
            FaceDetectorModels.HAARCASCADE
            FaceDetectorModels.DLIBHOG
            FaceDetectorModels.DLIBCNN
            FaceDetectorModels.SSDRESNET
            FaceDetectorModels.MTCNN
        - class FaceDetector
            initialize()
            detect(frame) - returs a list of face_rect

    encoder.py - module for Face Encoder/Embedder models
        - class FaceEncoderModels
            FaceEncoderModels.LBPH
            FaceEncoderModels.OPENFACE
            FaceEncoderModels.DLIBRESNET
        - class FaceEncoder
            initialize()
            identify(frame, face_rect) - returns the name of the face
            train() - outputs files containing encoding of the dataset of provided images with face

    liveness.py - module for Face Liveness Detector models
        - class FaceLivenessDetectorModels
        - class FaceLiveness
            initialize()
            detect()


Usage:

    Pre-requisites:

        - Install Python 3 and Python PIP

        - Install the required Python PIP packages 
            pip install -r requirements.txt

        - Add the dataset of images under the datasets directory
            Example:
            datasets/rico - contain .jpeg images of person name rico
            datasets/coni - contain .jpeg images of person named coni 
            ...
            datasets/xyz - contain .jpeg images of person named xyz 


    Training models with dataset of images:

        from libfaceid.detector import FaceDetectorModels, FaceDetector
        from libfaceid.encoder  import FaceEncoderModels, FaceEncoder

        INPUT_DIR_DATASET         = "datasets"
        INPUT_DIR_MODEL_DETECTION = "models/detection/"
        INPUT_DIR_MODEL_ENCODING  = "models/encoding/"
        INPUT_DIR_MODEL_TRAINING  = "models/training/"

        face_detector = FaceDetector(model=model_detector, path=INPUT_DIR_MODEL_DETECTION)
        face_detector.initialize()
        face_encoder = FaceEncoder(model=model_encoder, path=INPUT_DIR_MODEL_ENCODING, path_training=INPUT_DIR_MODEL_TRAINING, training=True)
        face_encoder.initialize()
        face_encoder.train(face_detector, path_dataset=INPUT_DIR_DATASET, verify=verify)


    Face Recognition on images:

        import cv2
        from libfaceid.detector import FaceDetectorModels, FaceDetector
        from libfaceid.encoder  import FaceEncoderModels, FaceEncoder

        INPUT_DIR_MODEL_DETECTION = "models/detection/"
        INPUT_DIR_MODEL_ENCODING  = "models/encoding/"
        INPUT_DIR_MODEL_TRAINING  = "models/training/"

        image = cv2.VideoCapture(imagePath)
        face_detector = FaceDetector(model=FaceDetectorModels.HAARCASCADE, path=INPUT_DIR_MODEL_DETECTION)
        face_detector.initialize()
        face_encoder = FaceEncoder(model=FaceEncoderModels.LBPH, path=INPUT_DIR_MODEL_ENCODING, path_training=INPUT_DIR_MODEL_TRAINING, training=False)
        face_encoder.initialize()

        frame = image.read()
        faces = face_detector.detect(frame)
        for (index, face) in enumerate(faces):
            (x, y, w, h) = face
            face_id, confidence = face_encoder.identify(frame, (x, y, w, h))
            label_face(frame, (x, y, w, h), face_id, confidence)
        cv2.imshow(window_name, frame)
        cv2.waitKey(5000)

        image.release()
        cv2.destroyAllWindows()


    Real-Time Face Recognition:

        import cv2
        from libfaceid.detector import FaceDetectorModels, FaceDetector
        from libfaceid.encoder  import FaceEncoderModels, FaceEncoder

        INPUT_DIR_MODEL_DETECTION = "models/detection/"
        INPUT_DIR_MODEL_ENCODING  = "models/encoding/"
        INPUT_DIR_MODEL_TRAINING  = "models/training/"

        camera = cv2.VideoCapture(webcam_index)
        face_detector = FaceDetector(model=FaceDetectorModels.HAARCASCADE, path=INPUT_DIR_MODEL_DETECTION)
        face_detector.initialize()
        face_encoder = FaceEncoder(model=FaceEncoderModels.LBPH, path=INPUT_DIR_MODEL_ENCODING, path_training=INPUT_DIR_MODEL_TRAINING, training=False)
        face_encoder.initialize()

        while True:
            frame = camera.read()
            faces = face_detector.detect(frame)
            for (index, face) in enumerate(faces):
                (x, y, w, h) = face
                face_id, confidence = face_encoder.identify(frame, (x, y, w, h))
                label_face(frame, (x, y, w, h), face_id, confidence)
            cv2.imshow(window_name, frame)
            cv2.waitKey(1)

        camera.release()
        cv2.destroyAllWindows()


    Examples:

        detector models:   0-HAARCASCADE, 1-DLIBHOG, 2-DLIBCNN, 3-SSDRESNET, 4-MTCNN
        encoder models:    0-LBPH, 1-OPENFACE, 2-DLIBRESNET
        camera resolution: 0-QVGA, 1-VGA, 2-HD, 3-FULLHD

        facial_recognition_training.py
            Usage: python facial_recognition_training.py --detector 0 --encoder 0

        facial_recognition_testing_image.py
            Usage: python facial_recognition_testing_image.py --detector 0 --encoder 0 --image datasets/rico/1.jpg

        facial_recognition_testing_webcam.py
            Usage: python facial_recognition_testing_webcam.py --detector 0 --encoder 0 --webcam 0 --resolution 2

