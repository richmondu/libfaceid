# libfaceid

    libfaceid is a python library for facial recognition that seamlessly integrates multiple face detection and face recognition models.
    It simplifies development of facial recognition systems by providing different models for detection and identification/encoding.
    Multiple models for detection and encoding/embedding including classification models are supported.
    The models are seamlessly integrated so that user can mix and match detection models with identification/encoder models.
    Each model differs in speed, accuracy, memory requirements and 3rd-party library dependencies.
    This enables users to easily experiment with various solutions appropriate for their specific use cases and system requirements.
    The library and example applications have been tested on Windows 7 and Raspberry Pi 3B+.


### Supported Models:

    Face Detector models for detecting face locations
        - Haar Cascade Classifier via OpenCV
        - Histogram of Oriented Gradients (HOG) via DLIB
        - Deep Neural Network via DLIB 
        - Single Shot Detector with ResNet-10 via OpenCV
        - Multi-task Cascaded CNN (MTCNN) via Tensorflow

    Face Encoder models for generating face embeddings
        - Local Binary Patterns Histograms (LBPH) via OpenCV
        - OpenFace via OpenCV
        - ResNet via DLIB

    Face Classifier models for face classification based on face embeddings
        - Naïve Bayes
        - Linear SVM
        - RVF SVM
        - Nearest Neighbors
        - Decision Tree
        - Random Forest
        - Neural Net
        - Adaboost
        - QDA
        
    Other models can be integrated to libfaceid in the future.
        - VGG-Face (VGG-16, ResNet-50) via Keras
        - FaceNet (Inception ResNet v1) via Tensorflow
        - OpenFace via Torch


### Usage:

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
        face_encoder = FaceEncoder(model=model_encoder, path=INPUT_DIR_MODEL_ENCODING, path_training=INPUT_DIR_MODEL_TRAINING, training=True)
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
        face_encoder = FaceEncoder(model=FaceEncoderModels.LBPH, path=INPUT_DIR_MODEL_ENCODING, path_training=INPUT_DIR_MODEL_TRAINING, training=False)

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
        face_encoder = FaceEncoder(model=FaceEncoderModels.LBPH, path=INPUT_DIR_MODEL_ENCODING, path_training=INPUT_DIR_MODEL_TRAINING, training=False)

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

        detector models:       0-HAARCASCADE, 1-DLIBHOG, 2-DLIBCNN, 3-SSDRESNET, 4-MTCNN
        encoder models:        0-LBPH, 1-OPENFACE, 2-DLIBRESNET
        classifier algorithms: 0-NAIVE_BAYES, 1-LINEAR_SVM, 2-RBF_SVM, 3-NEAREST_NEIGHBORS, 4-DECISION_TREE, 
                               5-RANDOM_FOREST, 6-NEURAL_NET, 7-ADABOOST, 8-QDA
        camera resolution:     0-QVGA, 1-VGA, 2-HD, 3-FULLHD

        1. facial_recognition_training.py
            Usage: python facial_recognition_training.py --detector 0 --encoder 0 --classifier 0
        2. facial_recognition_testing_image.py
            Usage: python facial_recognition_testing_image.py --detector 0 --encoder 0 --image datasets/rico/1.jpg
        3. facial_recognition_testing_webcam.py
            Usage: python facial_recognition_testing_webcam.py --detector 0 --encoder 0 --webcam 0 --resolution 2



### Links to valuable resoures:

        Special thanks to these guys for sharing their work and helping me understand Face Recognition.
        1. OpenCV by Adrian Rosebrock <https://www.pyimagesearch.com/>
        2. Dlib by Davis King <https://github.com/davisking/dlib>
        3. Face Recognition by Adam Geitgey <https://github.com/ageitgey/face_recognition>
        4. FaceNet by David Sandberg <https://github.com/davidsandberg/facenet>
        5. OpenFace <https://github.com/richmondu/openface>      
        6. VGG-Face <https://github.com/rcmalli/keras-vggface>
