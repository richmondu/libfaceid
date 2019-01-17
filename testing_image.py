import sys
import argparse
import cv2
from libfaceid.detector import FaceDetectorModels, FaceDetector
from libfaceid.encoder  import FaceEncoderModels, FaceEncoder



# Set the window name
WINDOW_NAME = "Facial_Recognition"

# Set the input directories
INPUT_DIR_DATASET         = "datasets"
INPUT_DIR_MODEL_DETECTION = "models/detection/"
INPUT_DIR_MODEL_ENCODING  = "models/encoding/"
INPUT_DIR_MODEL_TRAINING  = "models/training/"
INPUT_DIR_MODEL           = "models/"



def label_face(frame, face_rect, face_id, confidence):
    (x, y, w, h) = face_rect
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 1)
    if face_id is not None:
        cv2.putText(frame, "{} {:.2f}%".format(face_id, confidence), 
            (x+5,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def process_facerecognition(model_detector, model_recognizer, image):

    # Initialize the camera
    image = cv2.VideoCapture(image)

    # Initialize face detection
    face_detector = FaceDetector(model=model_detector, path=INPUT_DIR_MODEL_DETECTION)

    # Initialize face recognizer
    try:
        face_encoder = FaceEncoder(model=model_recognizer, path=INPUT_DIR_MODEL_ENCODING, path_training=INPUT_DIR_MODEL_TRAINING, training=False)
    except:
        face_encoder = None
        print("Warning, check if models and trained dataset models exists!")
    face_id, confidence = (None, 0)


    # Capture frame-by-frame
    ret, frame = image.read()
    if ret == 0:
        print("Unexpected error! " + image)
        return


    # Detect faces in the image
    faces = face_detector.detect(frame)
    for (index, face) in enumerate(faces):
        (x, y, w, h) = face
        # Indentify face based on trained dataset (note: should run facial_recognition_training.py)
        if face_encoder is not None:
            face_id, confidence = face_encoder.identify(frame, (x, y, w, h))
        # Set text and bounding box on face
        label_face(frame, (x, y, w, h), face_id, confidence)


    # Display the resulting frame
    cv2.imshow(WINDOW_NAME, frame)
    cv2.waitKey(5000)

    # Release the image
    image.release()
    cv2.destroyAllWindows()


def run(image):
    detector=FaceDetectorModels.HAARCASCADE
#    detector=FaceDetectorModels.DLIBHOG
#    detector=FaceDetectorModels.DLIBCNN
#    detector=FaceDetectorModels.SSDRESNET
#    detector=FaceDetectorModels.MTCNN
#    detector=FaceDetectorModels.MTCNN

    encoder=FaceEncoderModels.LBPH
#    encoder=FaceEncoderModels.OPENFACE
#    encoder=FaceEncoderModels.DLIBRESNET
#    encoder=FaceEncoderModels.FACENET

    # check face recognition
    process_facerecognition(detector, encoder, image)


def main(args):
    if sys.version_info < (3, 0):
        print("Error: Python2 is slow. Use Python3 for max performance.")
        return
    if args.detector and args.encoder:
        try:
            detector = FaceDetectorModels(int(args.detector))
            encoder = FaceEncoderModels(int(args.encoder))
            print( "Parameters: {} {}".format(detector, encoder) )
            process_facerecognition(detector, encoder, args.image)
        except:
            print( "Invalid parameter" )
        return
    run(args.image)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--detector', required=False,
        help='Detector model to use. Options: 0-HAARCASCADE, 1-DLIBHOG, 2-DLIBCNN, 3-SSDRESNET, 4-MTCNN, 5-FACENET')
    parser.add_argument('--encoder', required=False,
        help='Encoder model to use. Options: 0-LBPH, 1-OPENFACE, 2-DLIBRESNET, 3-FACENET')
    parser.add_argument('--image', required=True, 
        help='Image to process.')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
