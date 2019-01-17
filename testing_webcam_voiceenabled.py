import sys
import argparse
import cv2
from libfaceid.detector import FaceDetectorModels, FaceDetector
from libfaceid.encoder  import FaceEncoderModels, FaceEncoder
from libfaceid.speech_synthesizer import SpeechSynthesizerModels, SpeechSynthesizer



# Set the window name
WINDOW_NAME = "Facial_Recognition"

# Set the input directories
INPUT_DIR_DATASET               = "datasets"
INPUT_DIR_MODEL_DETECTION       = "models/detection/"
INPUT_DIR_MODEL_ENCODING        = "models/encoding/"
INPUT_DIR_MODEL_TRAINING        = "models/training/"
INPUT_DIR_MODEL_ESTIMATION      = "models/estimation/"
INPUT_DIR_AUDIOSET              = "audiosets"

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


def process_facerecognition(model_detector, model_recognizer, model_speech_synthesizer, cam_index, cam_resolution):

    # Initialize the camera
    camera = cam_init(cam_index, cam_resolution[0], cam_resolution[1])

    try:
        # Initialize face detection
        face_detector = FaceDetector(model=model_detector, path=INPUT_DIR_MODEL_DETECTION)

        # Initialize face recognizer
        face_encoder = FaceEncoder(model=model_recognizer, path=INPUT_DIR_MODEL_ENCODING, path_training=INPUT_DIR_MODEL_TRAINING, training=False)

        # Initialize text-to-speech synthesizer
        speech_synthesizer = SpeechSynthesizer(model=model_speech_synthesizer, path=None, path_output=None, training=False)
    except:
        #face_encoder = None
        print("Warning, check if models and trained dataset models exists!")
    face_id, confidence = (None, 0)


    frame_count = 0
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
            # Indentify face based on trained dataset (note: should run facial_recognition_training.py)
            if face_encoder is not None:
                face_id, confidence = face_encoder.identify(frame, (x, y, w, h))
            # Set text and bounding box on face
            label_face(frame, (x, y, w, h), face_id, confidence)

            # Play audio file corresponding to the recognized name 
            if (frame_count % 30 == 0):
                if len(faces) == 1 and (face_id is not None) and (face_id != "Unknown"):
                    speech_synthesizer.playaudio(INPUT_DIR_AUDIOSET, face_id, block=False)

            # Process 1 face only
            break


        # Display updated frame
        cv2.imshow(WINDOW_NAME, frame)

        # Check for user actions
        if cv2.waitKey(1) & 0xFF == 27: # ESC
            break

        frame_count += 1

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

    speech_synthesizer=SpeechSynthesizerModels.TTSX3
#    speech_synthesizer=SpeechSynthesizerModels.TACOTRON
#    speech_synthesizer=SpeechSynthesizerModels.GOOGLECLOUD

    process_facerecognition(detector, encoder, speech_synthesizer, cam_index, cam_resolution)


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

    if args.detector and args.encoder and args.speech_synthesizer:
        try:
            detector = FaceDetectorModels(int(args.detector))
            encoder = FaceEncoderModels(int(args.encoder))
            speech_synthesizer = SpeechSynthesizerModels(int(args.speech_synthesizer))
            print( "Parameters: {} {} {}".format(detector, encoder, speech_synthesizer) )
            process_facerecognition(detector, encoder, speech_synthesizer, cam_index, cam_resolution)
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
    parser.add_argument('--speech_synthesizer', required=False, default=0, 
        help='Speech synthesizer model to use. Options: 0-TTSX3, 1-TACOTRON, 2-GOOGLECLOUD')
    parser.add_argument('--webcam', required=False, default=0, 
        help='Camera index to use. Default is 0. Assume only 1 camera connected.)')
    parser.add_argument('--resolution', required=False, default=0,
        help='Camera resolution to use. Default is 0. Options: 0-QVGA, 1-VGA, 2-HD, 3-FULLHD')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
