import sys
import argparse
import cv2
from libfaceid.detector import FaceDetectorModels, FaceDetector
from libfaceid.encoder  import FaceEncoderModels, FaceEncoder
from libfaceid.pose import FacePoseEstimatorModels, FacePoseEstimator
from libfaceid.age import FaceAgeEstimatorModels, FaceAgeEstimator
from libfaceid.gender import FaceGenderEstimatorModels, FaceGenderEstimator
from libfaceid.emotion import FaceEmotionEstimatorModels, FaceEmotionEstimator


# Use flask for web app
from flask import Flask, render_template, Response
app = Flask(__name__)


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


def process_facedetection():

    cam_index = 0
    cam_resolution = RESOLUTION_QVGA
    model_detector=FaceDetectorModels.HAARCASCADE
#    model_detector=FaceDetectorModels.DLIBHOG
#    model_detector=FaceDetectorModels.DLIBCNN
#    model_detector=FaceDetectorModels.SSDRESNET
#    model_detector=FaceDetectorModels.MTCNN
#    model_detector=FaceDetectorModels.FACENET

    model_poseestimator=FacePoseEstimatorModels.DEFAULT
    model_ageestimator=FaceAgeEstimatorModels.DEFAULT
    model_genderestimator=FaceGenderEstimatorModels.DEFAULT
    model_emotionestimator=FaceEmotionEstimatorModels.DEFAULT


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
            cv2.putText(frame, "Age: {}".format(age), 
                (x, y-45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, "Gender: {}".format(gender), 
                (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, "Emotion: {}".format(emotion), 
                (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


        # Display updated frame to web app
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', frame)[1].tobytes() + b'\r\n\r\n')

    # Release the camera
    camera.release()
    cv2.destroyAllWindows()


# Initialize for web app
@app.route('/')
def index():
    return render_template('web_app_flask.html')

# Entry point for web app
@app.route('/video_viewer')
def video_viewer():
    return Response(process_facedetection(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    print("\n\nNote: Open browser and type http://127.0.0.1:5000/ or http://ip_address:5000/ \n\n")
    # Run flask for web app
    app.run(host='0.0.0.0', threaded=True, debug=True)
