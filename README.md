# libfaceid, a Face Recognition library for everybody

<p>
    <b> FaceRecognition Made Easy.</b> libfaceid is a Python library for facial recognition that seamlessly integrates multiple face detection and face recognition models. 
</p>
<p>
    <b> From Zero to Hero.</b> Learn the basics of Face Recognition and experiment with different models.
    libfaceid enables beginners to learn various models and simplifies prototyping of facial recognition solutions by providing a comprehensive list of models to choose from.
    Multiple models for detection and encoding/embedding including classification models are supported from the basic models (Haar Cascades + LBPH) to the more advanced models (MTCNN + FaceNet).
    The models are seamlessly integrated so that user can mix and match models. Each detector model has been made compatible with each embedding model to abstract you from the differences.
    Each model differs in speed, accuracy, memory requirements and 3rd-party library dependencies.
    This enables users to easily experiment with various solutions appropriate for their specific use cases and system requirements.
</p>
<p>
    <b> Awesome Design.</b> The library is designed so that it is easy to use, modular and robust.
    Selection of model is done via the constructors while the expose function is simply detect() or estimate() making usage very easy.
    The files are organized into modules so it is very intuitive to understand and debug.
    The robust design allows supporting new models in the future to be very straightforward.
</p> 
<p>
    <b> Extra Cool Features.</b> The library contains models for predicting your age, gender, emotion and facial landmarks.
    It also contains TTS text-to-speech (speech synthesizer) and STT speech-to-text (speech recognition) models for voice-enabled and voice-activated capabilities.
    Voice-enabled feature allows system to speak your name after recognizing your face.
    Voice-activated feature allows system to listen for a specified word or phrase to trigger the system to do something (wake-word/trigger-word/hotword detection).
    Web app is also supported for some test applications using Flask so you would be able to view the video capture remotely on another computer in the same network via a web browser. 
</p>


![](https://github.com/richmondu/libfaceid/blob/master/templates/teaser/libfaceid.jpg)
![](https://github.com/richmondu/libfaceid/blob/master/templates/teaser/libfaceid2.jpg)
![](https://github.com/richmondu/libfaceid/blob/master/templates/teaser/libfaceid3.jpg)
![](https://github.com/richmondu/libfaceid/blob/master/templates/teaser/libfaceid4.jpg)
![](https://github.com/richmondu/libfaceid/blob/master/templates/teaser/libfaceid5.jpg)


# News:

| Date | Milestones |
| --- | --- |
| 2018, Dec 29 | Integrated [Colorspace histogram](https://github.com/ee09115/spoofing_detection) for face liveness detection |
| 2018, Dec 26 | Integrated Google Cloud's STT speech-to-text (speech recognition) for voice-activated capability |
| 2018, Dec 19 | Integrated Google's [Tacotron](https://github.com/keithito/tacotron) TTS text-to-speech (speech synthesis) for voice-enabled capability |
| 2018, Dec 13 | Integrated Google's [FaceNet](https://github.com/davidsandberg/facenet) face embedding |
| 2018, Nov 30 | Committed libfaceid to Github |


# Background:

<p>
With Apple incorporating face recognition technology in iPhone X last year, 2017 
and with China implementing nation-wide wide-spread surveillance for social credit system in a grand scale, 
Face Recognition has become one of the most popular technologies where Deep Learning is used. 
Face recognition is used for identity authentication, access control, passport verification in airports, 
law enforcement, forensic investigations, social media platforms, disease diagnosis, police surveillance, 
casino watchlists and many more.
</p>

<p>
Modern state of the art Face Recognition solutions leverages graphics processor technologies, GPU, 
which has dramatically improved over the decades. (In particular, Nvidia released the CUDA framework which allowed C and C++ applications to utilize the GPU for massive parallel computing.)
It utilizes Deep Learning (aka Neural Networks) which requires GPU power to perform massive compute operations in parallel. 
Deep Learning is one approach to Artificial Intelligence that simulates how the brain functions by teaching software through examples, several examples (big data), instead of harcoding the logic rules and decision trees in the software. 
(One important contribution in Deep Learning is the creation of ImageNet dataset. It pioneered the creation of millions of images, a big data collection of images that were labelled and classified to teach computer for image classifications.) 
Neural networks are basically layers of nodes where each nodes are connected to nodes in the next layer feeding information. 
Deepnets are very deep neural networks with several layers made possible using GPU compute power. 
Many neural networks topologies exists such as Convolutional Neural Networks (CNN) architecture 
which particulary applies to Computer Vision, from image classification to face recognition.
</p>


# Introduction:

<p>
    
A facial recognition system is a technology capable of identifying or verifying a person from a digital image or a video frame from a video source. At a minimum, a simple real-time facial recognition system is composed of the following pipeline:

0. <b>Face Enrollment.</b> Registering faces to a database which includes pre-computing the face embeddings and training a classifier on top of the face embeddings of registered individuals. 
1. <b>Face Capture.</b> Reading a frame image from a camera source.
2. <b>Face Detection.</b> Detecting faces in a frame image.
3. <b>Face Encoding/Embedding.</b> Generating a mathematical representation of each face (coined as embedding) in the frame image.
4. <b>Face Identification.</b> Infering each face embedding in an image with face embeddings of known people in a database.

More complex systems include features such as <b>Face Liveness Detection</b> (to counter spoofing attacks via photo, video or 3d mask), face alignment, <b>face augmentation</b> (to increase the number of dataset of images) and face verification (to confirm prediction by comparing cosine similarity or euclidean distance with each database embedding).
</p>


# Problem:

<p>
libfaceid democratizes learning Face Recognition. Popular models such as FaceNet and OpenFace are not straightforward to use and don't provide easy-to-follow guidelines on how to install and setup. So far, dlib has been the best in terms of documentation and usage but it is slow on CPU and has too many abstractions (abstracts OpenCV as well). Simple models such as OpenCV is good but too basic and lacks documentation of the parameter settings, on classification algorithms and end-to-end pipeline. Pyimagesearch has been great having several tutorials with easy to understand explanations but not much emphasis on model comparisons and seems to aim to sell books so intentions to help the community are not so pure after all (I hate the fact that you need to wait for 2 marketing emails to arrive just to download the source code for the tutorials. But I love the fact that he replies to all questions in the threads). With all this said, I've learned a lot from all these resources so I'm sure you will learn a lot too. 

libfaceid was created to somehow address these problems and fill-in the gaps from these resources. It seamlessly integrates multiple models for each step of the pipeline enabling anybody specially beginners in Computer Vision and Deep Learning to easily learn and experiment with a comprehensive face recognition end-to-end pipeline models. No strings attached. Once you have experimented will all the models and have chosen specific models for your specific use-case and system requirements, you can explore the more advanced models like FaceNet.

</p>


# Design:

<p>
libfaceid is designed so that it is easy to use, modular and robust. Selection of model is done via the constructors while the expose function is simply detect() or estimate() making usage very easy. The files are organized into modules so it is very intuitive to understand and debug. The robust design allows supporting new models in the future to be very straightforward.

Only pretrained models will be supported. Transfer learning is the practice of applying a pretrained model (that is trained on a very large dataset) to a new dataset. It basically means that it is able to generalize models from one dataset to another when it has been trained on a very large dataset, such that it is 'experienced' enough to generalize the learnings to new environment to new datasets. It is one of the major factors in the explosion of popularity in Computer Vision, not only for face recognition but most specially for object detection. And just recently, mid-2018 this year, transfer learning has been making good advances to Natural Language Processing ( [BERT by Google](https://github.com/google-research/bert) and [ELMo by Allen Institute](https://allennlp.org/elmo) ). Transfer learning is really useful and it is the main goal that the community working on Reinforcement Learning wants to achieve for robotics.
</p>


# Features:

Having several dataset of images per person is not possible for some use cases of Face Recognition. So finding the appropriate model for that balances accuracy and speed on target hardware platform (CPU, GPU, embedded system) is necessary. The trinity of AI is Data, Algorithms and Compute. libfaceid allows selecting each model/algorithm in the pipeline.

libfaceid library supports several models for each step of the Face Recognition pipeline. Some models are faster while some models are more accurate. You can mix and match the models for your specific use-case, hardware platform and system requirements. 

### Face Detection models for detecting face locations
- [Haar Cascade Classifier via OpenCV](https://github.com/opencv/opencv/blob/master/samples/python/facedetect.py)
- [Histogram of Oriented Gradients (HOG) via DLIB](http://dlib.net/face_detector.py.html)
- [Deep Neural Network via DLIB](http://dlib.net/cnn_face_detector.py.html)
- [Single Shot Detector with ResNet-10 via OpenCV](https://github.com/opencv/opencv/blob/3.4.0/samples/dnn/resnet_ssd_face_python.py)
- [Multi-task Cascaded CNN (MTCNN) via Tensorflow](https://github.com/ipazc/mtcnn/blob/master/tests/test_mtcnn.py)
- [FaceNet MTCNN via Tensorflow](https://github.com/davidsandberg/facenet)

### Face Encoding models for generating face embeddings on detected faces
- [Local Binary Patterns Histograms (LBPH) via OpenCV](https://www.python36.com/face-recognition-using-opencv-part-3/)
- [OpenFace via OpenCV](https://www.pyimagesearch.com/2018/09/24/opencv-face-recognition/)
- [ResNet-34 via DLIB](http://dlib.net/face_recognition.py.html)
- [FaceNet (Inception ResNet v1) via Tensorflow](https://github.com/davidsandberg/facenet)
- [VGG-Face (VGG-16, ResNet-50) via Keras](https://github.com/rcmalli/keras-vggface) - TODO
- [OpenFace via Torch and Lua](https://github.com/cmusatyalab/openface) - TODO

### Classification algorithms for Face Identification using face embeddings
- [Naïve Bayes](https://www.analyticsvidhya.com/blog/2017/09/naive-bayes-explained/)
- Linear SVM
- RVF SVM
- Nearest Neighbors
- Decision Tree
- Random Forest
- Neural Net
- Adaboost
- QDA

### Additional models (bonus features for PR): 
- TTS Text-To-Speech <b>(speech synthesis)</b> models for voice-enabled capability
    - [PyTTSX3](https://pypi.org/project/pyttsx3/)
    - [Tacotron](https://github.com/keithito/tacotron)
    - [gTTS](https://pypi.org/project/gTTS/)
- STT Speech-To-Text <b>(speech recognition)</b> models for voice-activated capability
    - [GoogleCloud](https://pypi.org/project/SpeechRecognition/)
    - [Wit.ai](https://wit.ai/)
    - [Houndify](https://www.houndify.com/)
    - PocketSphinx - TODO
    - Snoyboy - TODO
    - Precise - TODO
- Face Liveness detection models for preventing face spoofing attacks
    - [Eye aspect ratio](https://www.pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib/)
    - [Mouth aspect ratio](https://github.com/mauckc/mouth-open)
    - [Colorspace histogram concatenation](https://github.com/ee09115/spoofing_detection)
- Face Pose estimator models for predicting face landmarks <b>(face landmark detection)</b>
- Face Age estimator models for predicting age <b>(age detection)</b>
- Face Gender estimator models for predicting gender <b>(gender detection)</b>
- Face Emotion estimator models for predicting facial expression <b>(emotion detection)</b>


# Compatibility:

<p>
The library and example applications have been tested on Raspberry Pi 3B+ (Python 3.5.3) and Windows 7 (Python 3.6.6)
using <b>OpenCV</b> 3.4.3.18, <b>Tensorflow</b> 1.8.0 and <b>Keras</b> 2.0.8. 
For complete dependencies, refer to requirements.txt. 
Tested with built-in laptop camera and with a Logitech C922 Full-HD USB webcam.

I encountered DLL issue with OpenCV 3.4.3.18 on my Windows 7 laptop. 
If you encounter such issue, use OpenCV 3.4.1.15 or 3.3.1.11 instead.
Also note that opencv-python and opencv-contrib-python must always have the same version.
</p>


# Usage:

### Installation:

        1. Install Python 3 and Python PIP
           Use Python 3.5.3 for Raspberry Pi 3B+ and Python 3.6.6 for Windows
        2. Install the required Python PIP package dependencies using requirements.txt
           pip install -r requirements.txt

           This will install the following dependencies below:
           opencv-python==3.4.3.18
           opencv-contrib-python==3.4.3.18
           numpy==1.15.4
           imutils==0.5.1
           dlib==19.16.0
           scipy==1.1.0
           scikit-learn==0.20.0
           mtcnn==0.0.8
           tensorflow==1.8.0
           keras==2.0.8
           h5py==2.8.0
           facenet==1.0.3
           flask==1.0.2

        3. Optional: Install the required Python PIP package dependencies for speech synthesizer and speech recognition for voice capability 
           pip install -r requirements_with_voicecapability.txt

           This will install additional dependencies below:
           playsound==1.2.2
           inflect==0.2.5
           librosa==0.5.1
           unidecode==0.4.20
           pyttsx3==2.7
           pypiwin32==223
           gtts==2.0.3
           speech_recognition==3.8.1


### Quickstart (Dummy Guide):

        1. Add your dataset
           ex. datasets/person1/1.jpg, datasets/person2/1.jpg
        2. Train your model with your dataset
           Update facial_recognition_training.bat to specify your chosen models
           Run facial_recognition_training.bat
        3. Test your model
           Update facial_recognition_testing_image.bat to specify your chosen models
           Run facial_recognition_testing_image.bat


### Folder structure:

        libfaceid
        |
        |   facial_estimation_poseagegenderemotion_webcam.py
        |   facial_recognition.py
        |   facial_recognition_testing_image.py
        |   facial_recognition_testing_webcam.py
        |   facial_recognition_testing_webcam_livenessdetection.py
        |   facial_recognition_testing_webcam_voiceenabled.py
        |   facial_recognition_testing_webcam_voiceenabled_voiceactivated.py
        |   facial_recognition_training.py
        |   requirements.txt
        |   requirements_with_voicecapability.txt
        |   
        +---libfaceid
        |   |   age.py
        |   |   classifier.py
        |   |   detector.py
        |   |   emotion.py
        |   |   encoder.py
        |   |   gender.py
        |   |   liveness.py
        |   |   pose.py
        |   |   speech_synthesizer.py
        |   |   speech_recognizer.py
        |   |   __init__.py
        |   |   
        |   \---tacotron
        |           
        +---models
        |   +---detection
        |   |       deploy.prototxt
        |   |       haarcascade_frontalface_default.xml
        |   |       mmod_human_face_detector.dat
        |   |       res10_300x300_ssd_iter_140000.caffemodel
        |   |       
        |   +---encoding
        |   |       dlib_face_recognition_resnet_model_v1.dat
        |   |       facenet_20180402-114759.pb
        |   |       openface_nn4.small2.v1.t7
        |   |       shape_predictor_5_face_landmarks.dat
        |   |           
        |   +---estimation
        |   |       age_deploy.prototxt
        |   |       age_net.caffemodel
        |   |       emotion_deploy.json
        |   |       emotion_net.h5
        |   |       gender_deploy.prototxt
        |   |       gender_net.caffemodel
        |   |       shape_predictor_68_face_landmarks.dat
        |   |       shape_predictor_68_face_landmarks.jpg
        |   |               
        |   +---synthesis
        |   |   \---tacotron-20180906
        |   |           model.ckpt.data-00000-of-00001
        |   |           model.ckpt.index
        |   |           
        |   \---training // This is generated during training (ex. facial_recognition_training.py)
        |           dlib_le.pickle
        |           dlib_re.pickle
        |           facenet_le.pickle
        |           facenet_re.pickle
        |           lbph.yml
        |           lbph_le.pickle
        |           openface_le.pickle
        |           openface_re.pickle
        |
        +---audiosets // This is generated during training (ex. facial_recognition_training.py)
        |       Person1.wav or Person1.mp3
        |       Person2.wav or Person2.mp3
        |       Person3.wav or Person3.mp3
        |       
        +---datasets // This is generated by user
        |   +---Person1
        |   |       1.jpg
        |   |       2.jpg
        |   |       ...
        |   |       X.jpg
        |   |       
        |   +---Person2
        |   |       1.jpg
        |   |       2.jpg
        |   |       ...
        |   |       X.jpg
        |   |       
        |   \---Person3
        |           1.jpg
        |           2.jpg
        |           ...
        |           X.jpg
        |           
        \---templates


### Pre-requisites:

        1. Add the dataset of images under the datasets directory
           The datasets folder should be in the same location as the test applications.
           Having more images per person makes accuracy much better.
           If only 1 image is possible, then do data augmentation.
             Example:
             datasets/Person1 - contain images of person name Person1
             datasets/Person2 - contain images of person named Person2 
             ...
             datasets/PersonX - contain images of person named PersonX 
        2. Train the model using the datasets. 
           Can use facial_recognition_training.py
           Make sure the models used for training is the same for actual testing for better accuracy.


### Examples:

        detector models:           0-HAARCASCADE, 1-DLIBHOG, 2-DLIBCNN, 3-SSDRESNET, 4-MTCNN, 5-FACENET
        encoder models:            0-LBPH, 1-OPENFACE, 2-DLIBRESNET, 3-FACENET
        classifier algorithms:     0-NAIVE_BAYES, 1-LINEAR_SVM, 2-RBF_SVM, 3-NEAREST_NEIGHBORS, 4-DECISION_TREE, 5-RANDOM_FOREST, 6-NEURAL_NET, 7-ADABOOST, 8-QDA
        liveness models:           0-EYESBLINK_MOUTHOPEN, 1-COLORSPACE_YCRCBLUV
        speech synthesizer models: 0-TTSX3, 1-TACOTRON, 2-GOOGLECLOUD
        speech recognition models: 0-GOOGLECLOUD, 1-WITAI, 2-HOUNDIFY
        camera resolution:         0-QVGA, 1-VGA, 2-HD, 3-FULLHD

        1. Training with datasets
            Usage: python facial_recognition_training.py --detector 0 --encoder 0 --classifier 0
            Usage: python facial_recognition_training.py --detector 0 --encoder 0 --classifier 0 --setsynthesizer True --synthesizer 0

        2. Testing with images
            Usage: python facial_recognition_testing_image.py --detector 0 --encoder 0 --image datasets/rico/1.jpg

        3. Testing with a webcam
            Usage: python facial_recognition_testing_webcam.py --detector 0 --encoder 0 --webcam 0 --resolution 0
            Usage: python facial_recognition_testing_webcam_flask.py
                   Then open browser and type http://127.0.0.1:5000 or http://ip_address:5000
                
        4. Testing with a webcam with anti-spoofing attacks
            Usage: python facial_recognition_testing_webcam_livenessdetection.py --detector 0 --encoder 0 --liveness 0 --webcam 0 --resolution 0

        5. Testing with voice-control
            Usage: python facial_recognition_testing_webcam_voiceenabled.py --detector 0 --encoder 0 --speech_synthesizer 0 --webcam 0 
            Usage: python facial_recognition_testing_webcam_voiceenabled_voiceactivated.py --detector 0 --encoder 0 --speech_synthesizer 0 --speech_recognition 0 --webcam 0 --resolution 0

        6. Testing age/gender/emotion detection
            Usage: python facial_estimation_poseagegenderemotion_webcam.py --detector 0 --webcam 0 --resolution 0
            Usage: python facial_estimation_poseagegenderemotion_webcam_flask.py
                   Then open browser and type http://127.0.0.1:5000 or http://ip_address:5000


### Training models with dataset of images:

        from libfaceid.detector import FaceDetectorModels, FaceDetector
        from libfaceid.encoder  import FaceEncoderModels, FaceEncoder
        from libfaceid.classifier  import FaceClassifierModels

        INPUT_DIR_DATASET         = "datasets"
        INPUT_DIR_MODEL_DETECTION = "models/detection/"
        INPUT_DIR_MODEL_ENCODING  = "models/encoding/"
        INPUT_DIR_MODEL_TRAINING  = "models/training/"

        face_detector = FaceDetector(model=FaceDetectorModels.DEFAULT, path=INPUT_DIR_MODEL_DETECTION)
        face_encoder = FaceEncoder(model=FaceEncoderModels.DEFAULT, path=INPUT_DIR_MODEL_ENCODING, path_training=INPUT_DIR_MODEL_TRAINING, training=True)
        face_encoder.train(face_detector, path_dataset=INPUT_DIR_DATASET, verify=verify, classifier=FaceClassifierModels.NAIVE_BAYES)

        // generate audio samples for image datasets using text to speech synthesizer
        OUTPUT_DIR_AUDIOSET       = "audiosets/"
        INPUT_DIR_MODEL_SYNTHESIS = "models/synthesis/"
        from libfaceid.speech_synthesizer import SpeechSynthesizerModels, SpeechSynthesizer
        speech_synthesizer = SpeechSynthesizer(model=SpeechSynthesizerModels.DEFAULT, path=INPUT_DIR_MODEL_SYNTHESIS, path_output=OUTPUT_DIR_AUDIOSET)
        speech_synthesizer.synthesize_datasets(INPUT_DIR_DATASET)


### Face Recognition on images:

        import cv2
        from libfaceid.detector import FaceDetectorModels, FaceDetector
        from libfaceid.encoder  import FaceEncoderModels, FaceEncoder

        INPUT_DIR_MODEL_DETECTION = "models/detection/"
        INPUT_DIR_MODEL_ENCODING  = "models/encoding/"
        INPUT_DIR_MODEL_TRAINING  = "models/training/"

        image = cv2.VideoCapture(imagePath)
        face_detector = FaceDetector(model=FaceDetectorModels.DEFAULT, path=INPUT_DIR_MODEL_DETECTION)
        face_encoder = FaceEncoder(model=FaceEncoderModels.DEFAULT, path=INPUT_DIR_MODEL_ENCODING, path_training=INPUT_DIR_MODEL_TRAINING, training=False)

        frame = image.read()
        faces = face_detector.detect(frame)
        for (index, face) in enumerate(faces):
            face_id, confidence = face_encoder.identify(frame, face)
            label_face(frame, face, face_id, confidence)
        cv2.imshow(window_name, frame)
        cv2.waitKey(5000)

        image.release()
        cv2.destroyAllWindows()


### Basic Real-Time Face Recognition (w/a webcam):

        import cv2
        from libfaceid.detector import FaceDetectorModels, FaceDetector
        from libfaceid.encoder  import FaceEncoderModels, FaceEncoder

        INPUT_DIR_MODEL_DETECTION = "models/detection/"
        INPUT_DIR_MODEL_ENCODING  = "models/encoding/"
        INPUT_DIR_MODEL_TRAINING  = "models/training/"

        camera = cv2.VideoCapture(webcam_index)
        face_detector = FaceDetector(model=FaceDetectorModels.DEFAULT, path=INPUT_DIR_MODEL_DETECTION)
        face_encoder = FaceEncoder(model=FaceEncoderModels.DEFAULT, path=INPUT_DIR_MODEL_ENCODING, path_training=INPUT_DIR_MODEL_TRAINING, training=False)

        while True:
            frame = camera.read()
            faces = face_detector.detect(frame)
            for (index, face) in enumerate(faces):
                face_id, confidence = face_encoder.identify(frame, face)
                label_face(frame, face, face_id, confidence)
            cv2.imshow(window_name, frame)
            cv2.waitKey(1)

        camera.release()
        cv2.destroyAllWindows()


### Real-Time Face Recognition With Liveness Detection (w/a webcam):

        import cv2
        from libfaceid.detector import FaceDetectorModels, FaceDetector
        from libfaceid.encoder  import FaceEncoderModels, FaceEncoder
        from libfaceid.liveness    import FaceLivenessModels, FaceLiveness

        INPUT_DIR_MODEL_DETECTION  = "models/detection/"
        INPUT_DIR_MODEL_ENCODING   = "models/encoding/"
        INPUT_DIR_MODEL_TRAINING   = "models/training/"
        INPUT_DIR_MODEL_ESTIMATION = "models/estimation/"
        INPUT_DIR_MODEL_LIVENESS   = "models/liveness/"

        camera = cv2.VideoCapture(webcam_index)
        face_detector = FaceDetector(model=FaceDetectorModels.DEFAULT, path=INPUT_DIR_MODEL_DETECTION)
        face_encoder = FaceEncoder(model=FaceEncoderModels.DEFAULT, path=INPUT_DIR_MODEL_ENCODING, path_training=INPUT_DIR_MODEL_TRAINING, training=False)
        face_liveness = FaceLiveness(model=model_liveness, path=INPUT_DIR_MODEL_ESTIMATION)
        face_liveness2 = FaceLiveness(model=FaceLivenessModels.COLORSPACE_YCRCBLUV, path=INPUT_DIR_MODEL_LIVENESS)

        while True:
            frame = camera.read()
            faces = face_detector.detect(frame)
            for (index, face) in enumerate(faces):

                // Check if eyes are close and if mouth is open
                eyes_close, eyes_ratio = face_liveness.is_eyes_close(frame, face)
                mouth_open, mouth_ratio = face_liveness.is_mouth_open(frame, face)

                // Detect if frame is a print attack or replay attack based on colorspace
                is_fake_print  = face_liveness2.is_fake(frame, face)
                is_fake_replay = face_liveness2.is_fake(frame, face, flag=1)

                // Identify face only if it is not fake and eyes are open and mouth is close
                if is_fake_print or is_fake_replay:
                    face_id, confidence = ("Fake", None)
                elif not eyes_close and not mouth_open:
                    face_id, confidence = face_encoder.identify(frame, face)

                label_face(frame, face, face_id, confidence)

            // Monitor eye blinking and mouth opening for liveness detection
            total_eye_blinks, eye_counter = monitor_eye_blinking(eyes_close, eyes_ratio, total_eye_blinks, eye_counter, eye_continuous_close)
            total_mouth_opens, mouth_counter = monitor_mouth_opening(mouth_open, mouth_ratio, total_mouth_opens, mouth_counter, mouth_continuous_open)

            cv2.imshow(window_name, frame)
            cv2.waitKey(1)

        camera.release()
        cv2.destroyAllWindows()


### Voice-Enabled Real-Time Face Recognition (w/a webcam):

        import cv2
        from libfaceid.detector import FaceDetectorModels, FaceDetector
        from libfaceid.encoder  import FaceEncoderModels, FaceEncoder
        from libfaceid.speech_synthesizer import SpeechSynthesizerModels, SpeechSynthesizer

        INPUT_DIR_MODEL_DETECTION = "models/detection/"
        INPUT_DIR_MODEL_ENCODING  = "models/encoding/"
        INPUT_DIR_MODEL_TRAINING  = "models/training/"
        INPUT_DIR_AUDIOSET        = "audiosets"

        camera = cv2.VideoCapture(webcam_index)
        face_detector = FaceDetector(model=FaceDetectorModels.DEFAULT, path=INPUT_DIR_MODEL_DETECTION)
        face_encoder = FaceEncoder(model=FaceEncoderModels.DEFAULT, path=INPUT_DIR_MODEL_ENCODING, path_training=INPUT_DIR_MODEL_TRAINING, training=False)
        speech_synthesizer = SpeechSynthesizer(model=SpeechSynthesizerModels.DEFAULT, path=None, path_output=None, training=False)

        frame_count = 0
        while True:
            frame = camera.read()
            faces = face_detector.detect(frame)
            for (index, face) in enumerate(faces):
                face_id, confidence = face_encoder.identify(frame, face)
                label_face(frame, face, face_id, confidence)
                if (frame_count % 120 == 0):
                    // Speak the person's name
                    speech_synthesizer.playaudio(INPUT_DIR_AUDIOSET, face_id, block=False)
            cv2.imshow(window_name, frame)
            cv2.waitKey(1)
            frame_count += 1

        camera.release()
        cv2.destroyAllWindows()


### Voice-Activated and Voice-Enabled Real-Time Face Recognition (w/a webcam):

        import cv2
        from libfaceid.detector import FaceDetectorModels, FaceDetector
        from libfaceid.encoder  import FaceEncoderModels, FaceEncoder
        from libfaceid.speech_synthesizer import SpeechSynthesizerModels, SpeechSynthesizer
        from libfaceid.speech_recognizer  import SpeechRecognizerModels,  SpeechRecognizer

        trigger_word_detected = False
        def speech_recognizer_callback(word):
            print("Trigger word detected! '{}'".format(word))
            trigger_word_detected = True

        INPUT_DIR_MODEL_DETECTION = "models/detection/"
        INPUT_DIR_MODEL_ENCODING  = "models/encoding/"
        INPUT_DIR_MODEL_TRAINING  = "models/training/"
        INPUT_DIR_AUDIOSET        = "audiosets"

        camera = cv2.VideoCapture(webcam_index)
        face_detector = FaceDetector(model=FaceDetectorModels.DEFAULT, path=INPUT_DIR_MODEL_DETECTION)
        face_encoder  = FaceEncoder(model=FaceEncoderModels.DEFAULT, path=INPUT_DIR_MODEL_ENCODING, path_training=INPUT_DIR_MODEL_TRAINING, training=False)
        speech_synthesizer = SpeechSynthesizer(model=SpeechSynthesizerModels.DEFAULT, path=None, path_output=None, training=False)
        speech_recognizer  = SpeechRecognizer(model=SpeechRecognizerModels.DEFAULT, path=None)

        // Wait for trigger word/wake word/hot word before starting face recognition
        TRIGGER_WORDS = ["Hey Google", "Alexa", "Activate", "Open Sesame"]
        print("\nWaiting for a trigger word: {}".format(TRIGGER_WORDS))
        speech_recognizer.start(TRIGGER_WORDS, speech_recognizer_callback)
        while (trigger_word_detected == False):
            time.sleep(1)
        speech_recognizer.stop()

        // Start face recognition
        frame_count = 0
        while True:
            frame = camera.read()
            faces = face_detector.detect(frame)
            for (index, face) in enumerate(faces):
                face_id, confidence = face_encoder.identify(frame, face)
                label_face(frame, face, face_id, confidence)
                if (frame_count % 120 == 0):
                    // Speak the person's name
                    speech_synthesizer.playaudio(INPUT_DIR_AUDIOSET, face_id, block=False)
            cv2.imshow(window_name, frame)
            cv2.waitKey(1)
            frame_count += 1

        camera.release()
        cv2.destroyAllWindows()


### Real-Time Face Pose/Age/Gender/Emotion Estimation (w/a webcam):

        import cv2
        from libfaceid.detector import FaceDetectorModels, FaceDetector
        from libfaceid.pose import FacePoseEstimatorModels, FacePoseEstimator
        from libfaceid.age import FaceAgeEstimatorModels, FaceAgeEstimator
        from libfaceid.gender import FaceGenderEstimatorModels, FaceGenderEstimator
        from libfaceid.emotion import FaceEmotionEstimatorModels, FaceEmotionEstimator

        INPUT_DIR_MODEL_DETECTION       = "models/detection/"
        INPUT_DIR_MODEL_ENCODING        = "models/encoding/"
        INPUT_DIR_MODEL_TRAINING        = "models/training/"
        INPUT_DIR_MODEL_ESTIMATION      = "models/estimation/"

        camera = cv2.VideoCapture(webcam_index)
        face_detector = FaceDetector(model=FaceDetectorModels.DEFAULT, path=INPUT_DIR_MODEL_DETECTION)
        face_pose_estimator = FacePoseEstimator(model=FacePoseEstimatorModels.DEFAULT, path=INPUT_DIR_MODEL_ESTIMATION)
        face_age_estimator = FaceAgeEstimator(model=FaceAgeEstimatorModels.DEFAULT, path=INPUT_DIR_MODEL_ESTIMATION)
        face_gender_estimator = FaceGenderEstimator(model=FaceGenderEstimatorModels.DEFAULT, path=INPUT_DIR_MODEL_ESTIMATION)
        face_emotion_estimator = FaceEmotionEstimator(model=FaceEmotionEstimatorModels.DEFAULT, path=INPUT_DIR_MODEL_ESTIMATION)

        while True:
            frame = camera.read()
            faces = face_detector.detect(frame)
            for (index, face) in enumerate(faces):
                age = face_age_estimator.estimate(frame, face_image)
                gender = face_gender_estimator.estimate(frame, face_image)
                emotion = face_emotion_estimator.estimate(frame, face_image)
                shape = face_pose_estimator.detect(frame, face)
                face_pose_estimator.add_overlay(frame, shape)
                label_face(age, gender, emotion)
            cv2.imshow(window_name, frame)
            cv2.waitKey(1)

        camera.release()
        cv2.destroyAllWindows()



# Case Study - Face Recognition for Identity Authentication:

One of the use cases of face recognition is for security identity authentication.
This is a convenience feature to authenticate with system using one's face instead of inputting passcode or scanning fingerprint. Passcode is often limited by the maximum number of digits allowed while fingerprint scanning often has problems with wet fingers or dry skin. Face authentication offers a more reliable and secure way to authenticate.

When used for identity authentication, face recognition specifications will differ a lot from general face recognition systems like Facebook's automated tagging and Google's search engine; it will be more like Apple's Face ID in IPhone X. Below are guidelines for drafting specifications for your face recognition solution. Note that [Apple's Face ID technology](https://support.apple.com/en-us/HT208109) will be used as the primary baseline in this case study of identity authentication use case of face recognition. Refer to this [Apple's Face ID white paper](https://www.apple.com/business/site/docs/FaceID_Security_Guide.pdf) for more information.


### Face Enrollment

- Should support dynamic enrollment of faces. Tied up with the maximum number of users the existing system supports.
- Should ask user to move/rotate face (in a circular motion) in order to capture different angles of the face. This gives the system enough flexbility to recognize you at different face angles.
- IPhone X Face ID face enrollment is done twice for some reason. It is possible that the first scan is for liveness detection only.
- How many images should be captured? We can store as much image as possible for better accuracy but memory footprint is the limiting factor. Estimate based on size of 1 picture and the maximum number of users.
- For security purposes and memory related efficiency, images used during enrollment should not be saved. 
Only the mathematical representations (128-dimensional vector) of the face should be used.


### Face Capture

- Camera will be about 1 foot away from user (Apple Face ID: 10-20 inches).
- Camera resolution will depend on display panel size and display resolutions. QVGA size is acceptable for embedded solutions. 
- Take into consideration a bad lighting and extremely dark situation. Should camera have a good flash/LED to emit some light. Iphone X has an infrared light to better perform on dark settings.


### Face Detection

- Only 1 face per frame is detected.
- Face is expected to be within a certain location (inside a fixed box or circular region).
- Detection of faces will be triggered by a user action - clicking some button. (Not automatic detection).
- Face alignment may not be helpful as users can be enforced or directed to have his face inside a fixed box or circular region so face is already expected to be aligned for the most cases. But if adding this feature does not affect speed performance, then face alignment ahould be added if possible.
- Should verify if face is alive via anti-spoofing techniques against picture-based attacks, video-based attacks and 3D mask attacks. Two popular example of liveness detection is detecting eye blinking and mouth opening. 


### Face Encoding/Embedding

- Speed is not a big factor. Face embedding and face identification can take 3-5 seconds.
- Accuracy is critically important. False match rate should be low as much as possible. 
- Can do multiple predictions and get the highest count. Or apply different models for predictions for double checking.


### Face Identification

- Recognize only when eyes are not closed and mouth is not open
- Images per person should at least be 50 images. Increase the number of images per person by cropping images with different face backgound margin, slight rotations, flipping and scaling.
- Classification model should consider the maximum number of users to support. For example, SVM is known to be good for less than 100k classes/persons only.
- Should support unknown identification by setting a threshold on the best prediction. If best prediction is too low, then consider as Unknown.
- Set the number of consecutive failed attempts allowed before disabling face recognition feature. Should fallback to passcode authentication if identification encounters trouble recognizing people.
- Images used for successful scan should be added to the existing dataset images during face enrollment making it adaptive and updated so that a person can be recognized with better accuracy in the future even with natural changes in the face appearance (hairstyle, mustache, pimples, etc.)

In addition to these guidelines, the face recognition solution should provide a way to disable/enable this feature as well as resetting the stored datasets during face enrollment.



# Case Study - Face Recognition for Home/Office/Hotel Greeting System:

One of the use cases of face recognition is for greeting system used in smart homes, office and hotels.
To enable voice capability feature, we use text-to-speech synthesis to dynamically create audio files given some input text. 

### Speech Synthesis

Speech synthesis is the artificial simulation of human speech by a computer device.
It is mostly used for translating text into audio to make the system voice-enabled.
Products such as Apple's Siri, Microsoft's Cortana, Amazon Echo and Google Assistant uses speech synthesis.
A good speech synthesizer is one that produces accurate outputs that naturally sounds like a real human in near real-time.
State-of-the-art speech synthesis includes [Deepmind's WaveNet](https://deepmind.com/blog/wavenet-generative-model-raw-audio/) 
and [Google's Tacotron](https://www.isca-speech.org/archive/Interspeech_2017/abstracts/1452.html).

Speech Synthesis can be used for some use-cases of Face Recognition to enable voice capability feature.
One example is to greet user as he approaches the terminal or kiosk system.
Given some input text, the speech synthesizer can generate an audio which can be played upon recognizing a face.
For example, upon detecting person arrival, it can be set to say 'Hello PersonX, welcome back...'. 
Upon departure, it can be set to say 'Goodbye PersonX, see you again soon...'.
It can be used in smart homes, office lobbies, luxury hotel rooms, and modern airports. 

### Face Enrollment

- For each person who registers/enrolls to the system, create an audio file "PersonX.wav" for some input text such as "Hello PersonX".
  
### Face Identification

- When a person is identified to be part of the database, we play the corresponding audio file "PersonX.wav". 



# Performance Optimizations:

Speed and accuracy is often a trade-off. Performance can be optimized depending on your specific use-case and system requirements. Some models are optimized for speed while others are optimized for accuracy. Be sure to test all the provided models to determine the appropriate model for your specific use-case, target platform (CPU, GPU or embedded) and specific requirements. Below are additional suggestions to optimize performance.

### Speed
- Reduce the frame size for face detection.
- Perform face recognition every X frames only
- Use threading in reading camera source frames or in processing the camera frames.
- Update the library and configure the parameters directly.

### Accuracy
- Add more datasets if possible (ex. do data augmentation). More images per person will often result to higher accuracy.
- Add face alignment if faces in the datasets are not aligned or when faces may be unaligned in actual deployment.
- Update the library and configure the parameters directly.



# References:

Below are links to valuable resoures. Special thanks to all of these guys for sharing their work on Face Recognition. Without them, learning Face Recognition would be difficult.

### Codes
- [OpenCV tutorials by Adrian Rosebrock](https://www.pyimagesearch.com/2018/09/24/opencv-face-recognition/)
- [Dlib by Davis King](https://github.com/davisking/dlib)
- [Face Recognition (Dlib wrapper) by Adam Geitgey](https://github.com/ageitgey/face_recognition)
- [FaceNet implementation by David Sandberg](https://github.com/davidsandberg/facenet)
- [OpenFace (FaceNet implementation) by Satyanarayanan](https://github.com/cmusatyalab/openface)
- [VGG-Face implementation by Refik Can Malli](https://github.com/rcmalli/keras-vggface)

Google and Facebook have access to large database of pictures being the best search engine and social media platform, respectively. Below are the face recognition models they have designed for their own system. Be sure to take time to read these papers for better understanding of high-quality face recognition models. 

### Papers
- [FaceNet paper by Google](https://arxiv.org/pdf/1503.03832.pdf)
- [DeepFace paper by Facebook](https://research.fb.com/wp-content/uploads/2016/11/deepface-closing-the-gap-to-human-level-performance-in-face-verification.pdf)



# Contribute:

Have a good idea for improving libfaceid? Please message me in [twitter](https://twitter.com/richmond_umagat).
If libfaceid has helped you in learning or prototyping face recognition system, please be kind enough to give this repository a 'Star'.
