import sys
import argparse
import cv2
from time import time
from libfaceid.detector    import FaceDetectorModels, FaceDetector
from libfaceid.encoder     import FaceEncoderModels, FaceEncoder
from libfaceid.liveness    import FaceLivenessModels, FaceLiveness



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


def monitor_eye_blinking(eyes_close, eyes_ratio, total_eye_blinks, eye_counter, eye_continuous_close):
    if eyes_close:
        print("eye less than threshold {:.2f}".format(eyes_ratio))
        eye_counter += 1
    else:
        if eye_counter >= eye_continuous_close:
            total_eye_blinks += 1
            print("eye:{:.2f} blinks:{}".format(eyes_ratio, total_eye_blinks))
        eye_counter = 0
    return total_eye_blinks, eye_counter


def monitor_mouth_opening(mouth_open, mouth_ratio, total_mouth_opens, mouth_counter, mouth_continuous_open):
    if mouth_open:
        print("mouth more than threshold {:.2f}".format(mouth_ratio))
        mouth_counter += 1
    else:
        if mouth_counter >= mouth_continuous_open:
            total_mouth_opens += 1
            print("mouth:{:.2f} opens:{}".format(mouth_ratio, total_mouth_opens))
        mouth_counter = 0
    return total_mouth_opens, mouth_counter


# process_livenessdetection is supposed to run before process_facerecognition
def process_livenessdetection(model_detector, model_recognizer, model_liveness, cam_index, cam_resolution):

    # Initialize the camera
    camera = cam_init(cam_index, cam_resolution[0], cam_resolution[1])

    try:
        # Initialize face detection
        face_detector = FaceDetector(model=model_detector, path=INPUT_DIR_MODEL_DETECTION)

        # Initialize face recognizer
        face_encoder = FaceEncoder(model=model_recognizer, path=INPUT_DIR_MODEL_ENCODING, path_training=INPUT_DIR_MODEL_TRAINING, training=False)

        # Initialize face liveness detection
        face_liveness = FaceLiveness(model=model_liveness, path=INPUT_DIR_MODEL_ESTIMATION)

    except:
        print("Error, check if models and trained dataset models exists!")
        return

    face_id, confidence = (None, 0)

    eyes_close, eyes_ratio = (False, 0)
    total_eye_blinks, eye_counter, eye_continuous_close = (0, 0, 3)
    mouth_open, mouth_ratio = (False, 0)
    total_mouth_opens, mouth_counter, mouth_continuous_open = (0, 0, 3)

    time_start = time()
    time_elapsed = 0
    frame_count = 0
    identified_unique_faces = {} # dictionary
    runtime = 10 # monitor for 10 seconds only


    print("Note: this will run for {} seconds only".format(runtime))
    while (time_elapsed < runtime): 

        # Capture frame from webcam
        ret, frame = camera.read()
        if frame is None:
            print("Error, check if camera is connected!")
            break


        # Detect and identify faces in the frame
        # Indentify face based on trained dataset (note: should run facial_recognition_training.py)
        faces = face_detector.detect(frame)
        for (index, face) in enumerate(faces):
            (x, y, w, h) = face

            # Identify face only if eyes are not closed
            eyes_close, eyes_ratio = face_liveness.is_eyes_close(frame, face)
            mouth_open, mouth_ratio = face_liveness.is_mouth_open(frame, face)
            #print("mouth_open={}, mouth_ratio={:.2f}".format(mouth_open, mouth_ratio))
            if not eyes_close and not mouth_open:
                face_id, confidence = face_encoder.identify(frame, (x, y, w, h))
                if face_id not in identified_unique_faces:
                    identified_unique_faces[face_id] = 1
                else:
                    identified_unique_faces[face_id] += 1
            # Set text and bounding box on face
            label_face(frame, (x, y, w, h), face_id, confidence)

            # Process 1 face only
            break


        # Monitor eye blinking and mouth opening for liveness detection
        total_eye_blinks, eye_counter = monitor_eye_blinking(eyes_close, eyes_ratio, total_eye_blinks, eye_counter, eye_continuous_close)
        total_mouth_opens, mouth_counter = monitor_mouth_opening(mouth_open, mouth_ratio, total_mouth_opens, mouth_counter, mouth_continuous_open)


        # Update frame count
        frame_count += 1
        time_elapsed = time()-time_start

        # Display updated frame
        cv2.imshow(WINDOW_NAME, frame)

        # Check for user actions
        if cv2.waitKey(1) & 0xFF == 27: # ESC
            break


    print("Note: this will run for {} seconds only".format(runtime))

    # Check the counters
    time_elapsed = int(time()-time_start)
    print("\n")
    print("Face Liveness Data:")
    print("time_elapsed      = {}".format(time_elapsed))
    print("frame_count       = {}".format(frame_count))
    print("total_eye_blinks  = {}".format(total_eye_blinks))
    print("total_mouth_opens = {}".format(total_mouth_opens))
    print("identified_unique_faces = {}".format(identified_unique_faces))
    print("TODO: determine if face is alive using this data.")
    print("\n")

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

    liveness=FaceLivenessModels.EYESBLINK_MOUTHOPEN

    process_livenessdetection(detector, encoder, liveness, cam_index, cam_resolution)


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

    if args.detector and args.encoder and args.liveness:
        try:
            detector = FaceDetectorModels(int(args.detector))
            encoder  = FaceEncoderModels(int(args.encoder))
            liveness = FaceLivenessModels(int(args.liveness))
            print( "Parameters: {} {} {}".format(detector, encoder, liveness) )
            process_livenessdetection(detector, encoder, liveness, cam_index, cam_resolution)
        except:
            print( "Invalid parameter" )
        return
    run(cam_index, cam_resolution)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--detector', required=False, default=0, 
        help='Detector model to use. Options: 0-HAARCASCADE, 1-DLIBHOG, 2-DLIBCNN, 3-SSDRESNET, 4-MTCNN, 5-FACENET')
    parser.add_argument('--encoder', required=False, default=0, 
        help='Encoder model to use. Options: 0-LBPH, 1-OPENFACE, 2-DLIBRESNET, 3-FACENET')
    parser.add_argument('--liveness', required=False, default=0, 
        help='Liveness detection model to use. Options: 0-EYESBLINK_MOUTHOPEN')
    parser.add_argument('--webcam', required=False, default=0, 
        help='Camera index to use. Default is 0. Assume only 1 camera connected.)')
    parser.add_argument('--resolution', required=False, default=0,
        help='Camera resolution to use. Default is 0. Options: 0-QVGA, 1-VGA, 2-HD, 3-FULLHD')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
