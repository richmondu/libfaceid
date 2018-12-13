import sys
import argparse
import cv2
import datetime
from libfaceid.detector import FaceDetectorModels, FaceDetector
from libfaceid.encoder  import FaceEncoderModels, FaceEncoder
from libfaceid.pose import FacePoseEstimatorModels, FacePoseEstimator
from libfaceid.age import FaceAgeEstimatorModels, FaceAgeEstimator
from libfaceid.gender import FaceGenderEstimatorModels, FaceGenderEstimator
from libfaceid.emotion import FaceEmotionEstimatorModels, FaceEmotionEstimator



# Set the window name
WINDOW_NAME = "Facial_Recognition"

# Set the input directories
INPUT_DIR_DATASET               = "datasets"
INPUT_DIR_MODEL_DETECTION       = "models/detection/"
INPUT_DIR_MODEL_ENCODING        = "models/encoding/"
INPUT_DIR_MODEL_TRAINING        = "models/training/"
INPUT_DIR_MODEL_ESTIMATION      = "models/estimation/"

# Set width and height
RESOLUTION_QVGA   = (320, 240)
RESOLUTION_VGA    = (640, 480)
RESOLUTION_HD     = (1280, 720)
RESOLUTION_FULLHD = (1920, 1080)



def cam_init(cam_index, width, height): 
    cap = cv2.VideoCapture(cam_index)
    if sys.version_info < (3, 0):
        cap.set(cv2.cv.CV_CAP_PROP_FPS, 30)
        cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH,  width)
        cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, height)
    else:
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def label_face(frame, face_rect, face_id, confidence):
    (x, y, w, h) = face_rect
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 1)
    if face_id is not None:
        cv2.putText(frame, "{} {:.2f}%".format(face_id, confidence), 
            (x+5,y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def process_facedetection(model_detector, model_poseestimator, model_ageestimator, model_genderestimator, model_emotionestimator, cam_resolution, cam_index):

    # Initialize the camera
    camera = cam_init(cam_index, cam_resolution[0], cam_resolution[1])

    try:
        # Initialize face detection
        face_detector = FaceDetector(model=model_detector, path=INPUT_DIR_MODEL_DETECTION)#, optimize=True)
        # Initialize face pose/age/gender estimation
        face_pose_estimator = FacePoseEstimator(model=model_poseestimator, path=INPUT_DIR_MODEL_ESTIMATION)
        face_age_estimator = FaceAgeEstimator(model=model_ageestimator, path=INPUT_DIR_MODEL_ESTIMATION)
        face_gender_estimator = FaceGenderEstimator(model=model_genderestimator, path=INPUT_DIR_MODEL_ESTIMATION)
        face_emotion_estimator = FaceEmotionEstimator(model=model_emotionestimator, path=INPUT_DIR_MODEL_ESTIMATION)
    except:
        print("Warning, check if models and trained dataset models exists!")
    (age, gender, emotion) = (None, None, None)


    while (True):

        # Capture frame from webcam
        ret, frame = camera.read()
        if frame is None:
            print("Error, check if camera is connected!")
            break


        # Detect and identify faces in the frame
        faces = face_detector.detect(frame)
        for (index, face) in enumerate(faces):
            (x, y, w, h) = face

            # Detect age, gender, emotion
            face_image = frame[y:y+h, h:h+w]
            age = face_age_estimator.estimate(frame, face_image)
            gender = face_gender_estimator.estimate(frame, face_image)
            emotion = face_emotion_estimator.estimate(frame, face_image)

            # Detect and draw face pose locations
            shape = face_pose_estimator.detect(frame, face)
            face_pose_estimator.add_overlay(frame, shape)

            # Display age, gender, emotion
            if True: # Added condition to easily disable text
                cv2.putText(frame, "Age: {}".format(age), 
                    (x, y-45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, "Gender: {}".format(gender), 
                    (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, "Emotion: {}".format(emotion), 
                    (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


        # Display the resulting frame
        cv2.imshow(WINDOW_NAME, frame)

        # Check for user actions
        keyPressed = cv2.waitKey(1) & 0xFF
        if keyPressed == 27: # ESC
            break
        elif keyPressed == 13: # Enter
            cv2.imwrite(WINDOW_NAME + "_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg", frame);


    # Release the camera
    camera.release()
    cv2.destroyAllWindows()


def run(cam_index, cam_resolution):

    detector=FaceDetectorModels.HAARCASCADE
#    detector=FaceDetectorModels.DLIBHOG
#    detector=FaceDetectorModels.DLIBCNN
#    detector=FaceDetectorModels.SSDRESNET
#    detector=FaceDetectorModels.MTCNN
#    detector=FaceDetectorModels.FACENET

    encoder=FaceEncoderModels.LBPH
#    encoder=FaceEncoderModels.OPENFACE
#    encoder=FaceEncoderModels.DLIBRESNET
#    encoder=FaceEncoderModels.FACENET

    poseestimator    = FacePoseEstimatorModels.DLIB68
    ageestimator     = FaceAgeEstimatorModels.CV2CAFFE
    genderestimator  = FaceGenderEstimatorModels.CV2CAFFE
    emotionestimator = FaceEmotionEstimatorModels.KERAS

    process_facedetection(
        detector, 
        poseestimator, 
        ageestimator, 
        genderestimator,
        emotionestimator,
        cam_resolution, 
        cam_index)


def main(args):
    if sys.version_info < (3, 0):
        print("Error: Python2 is slow. Use Python3 for max performance.")
        return

    cam_index = int(args.webcam)
    resolutions = [ RESOLUTION_QVGA, RESOLUTION_VGA, RESOLUTION_HD, RESOLUTION_FULLHD ]
    try:
        cam_resolution = resolutions[int(args.resolution)]
    except:
        cam_resolution = RESOLUTION_QVGA

    if args.detector:
        try:
            detector = FaceDetectorModels(int(args.detector))
            print( "Parameters: {}".format(detector) )
            process_facedetection(
                detector, 
                FacePoseEstimatorModels.DEFAULT, 
                FaceAgeEstimatorModels.DEFAULT, 
                FaceGenderEstimatorModels.DEFAULT,
                FaceEmotionEstimatorModels.DEFAULT,
                cam_resolution, 
                cam_index)
        except:
            print( "Invalid parameter" )
        return
    run(cam_index, cam_resolution)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--detector', required=False,
        help='Detector model to use. Options: 0-HAARCASCADE, 1-DLIBHOG, 2-DLIBCNN, 3-SSDRESNET, 4-MTCNN, 5-FACENET')
    parser.add_argument('--webcam', required=False, default=0, 
        help='Camera index to use. Default is 0. Assume only 1 camera connected.)')
    parser.add_argument('--resolution', required=False, default=0,
        help='Camera resolution to use. Default is 0. Options: 0-QVGA, 1-VGA, 2-HD, 3-FULLHD')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
