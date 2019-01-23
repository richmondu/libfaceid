import sys
import argparse
import cv2
import numpy as np
import os
import datetime
from libfaceid.detector import FaceDetectorModels, FaceDetector
from libfaceid.encoder  import FaceEncoderModels, FaceEncoder
from libfaceid.classifier import FaceClassifierModels



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


def ensure_directory(file_path):
    directory = os.path.dirname("./" + file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def label_face(frame, face_rect, face_id=None, confidence=0):
    (x, y, w, h) = face_rect
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 1)
    if face_id is not None:
        cv2.putText(frame, "{} {:.2f}%".format(face_id, confidence), 
            (x+5,y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def save_video(saveVideo, out, resolution, filename):
    if saveVideo == True:
        print("video recording ended!")
        out.release()
        out = None
        saveVideo = False
    else:
        print("video recording started...")
        print("Press space key to stop recording!")
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        (h, w) = resolution
        out = cv2.VideoWriter(filename, fourcc, 12, (w, h))
        saveVideo = True
    return saveVideo, out


def process_faceenrollment(model_detector, cam_index, cam_resolution):

    # Initialize the camera
    camera = cam_init(cam_index, cam_resolution[0], cam_resolution[1])

    try:
        # Initialize face detection
        face_detector = FaceDetector(model=model_detector, path=INPUT_DIR_MODEL_DETECTION)
    except:
        print("Warning, check if models and trained dataset models exists!")

    print("")
    print("Press SPACEBAR to record video or ENTER to capture picture!")
    print("Make sure that your face is inside the circular region!")
    print("")

    saveVideo = False
    out = None
    color_recording = (255,255,255)


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
            #print("{} {} {} {}".format(x,y,w,h))

            if saveVideo and len(faces) == 1:
                out.write(frame)

            # Set text and bounding box on face
            label_face(frame, (x, y, w, h))

            # Process 1 face only
            break


        mask = np.full((frame.shape[0], frame.shape[1]), 0, dtype=np.uint8)  # mask is only
        cv2.circle(mask, (int(cam_resolution[0]/2),int(cam_resolution[1]/2)), 115, (255,255,255), -1, cv2.LINE_AA)
        fg = cv2.bitwise_or(frame, frame, mask=mask)
        cv2.circle(fg, (int(cam_resolution[0]/2),int(cam_resolution[1]/2)), 115, color_recording, 1, cv2.LINE_AA)

        # Display updated frame
        cv2.imshow(WINDOW_NAME, fg)

        # Check for user actions
        keyPressed = cv2.waitKey(1) & 0xFF
        if keyPressed == 27: # ESC to exit
            break
        elif keyPressed == 32: # Space to save video
            saveVideo, out = save_video(saveVideo, out, frame.shape[:2], WINDOW_NAME + ".avi")
            if out is not None:
                color_recording = (0, 255, 0)
            else:
                color_recording = (0, 0, 0)
                break
        elif keyPressed == 13: # Enter to capture picture
            cv2.imwrite(WINDOW_NAME + "_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg", frame);


    # Release the camera
    camera.release()
    cv2.destroyAllWindows()


def video_to_images(model_detector, dir, name, one_image_only=False):

    ensure_directory(dir + "/" + name + "/")

    try:
        video = cv2.VideoCapture(WINDOW_NAME + ".avi")
        if video is None:
            return
    except:
        return

    try:
        # Initialize face detection
        face_detector = FaceDetector(model=model_detector, path=INPUT_DIR_MODEL_DETECTION)
    except:
        print("Warning, check if models and trained dataset models exists!")

    i = 1
    while (True):

        ret, frame = video.read()
        if frame is None:
            break

        faces = face_detector.detect(frame)
        if len(faces) == 1:
            cv2.imwrite("{}/{}/{}.jpg".format(dir, name, i), frame);
            i += 1
            if one_image_only:
                break
            
        #cv2.imshow(WINDOW_NAME, frame)
        #cv2.waitKey(1)

    video.release()
    cv2.destroyAllWindows()



def run(cam_index, cam_resolution, name):
#    detector=FaceDetectorModels.HAARCASCADE
#    detector=FaceDetectorModels.DLIBHOG
#    detector=FaceDetectorModels.DLIBCNN
#    detector=FaceDetectorModels.SSDRESNET
    detector=FaceDetectorModels.MTCNN
#    detector=FaceDetectorModels.FACENET

    process_faceenrollment(detector, cam_index, cam_resolution)

    print("")
    print("Processing of video recording started...")
    video_to_images(detector, "x" + INPUT_DIR_DATASET, name)
    video_to_images(detector, INPUT_DIR_DATASET, name, one_image_only=True)
    print("Processing of video recording completed!")
    print("Make sure to train the new datasets before testing!")
    print("")


def main(args):
    if sys.version_info < (3, 0):
        print("Error: Python2 is slow. Use Python3 for max performance.")
        return

    cam_index = int(args.webcam)
    resolutions = [ RESOLUTION_QVGA, RESOLUTION_VGA, RESOLUTION_HD, RESOLUTION_FULLHD ]
    try:
        cam_resolution = resolutions[int(args.resolution)]
    except:
        cam_resolution = RESOLUTION_VGA

    if args.detector and args.name:
        try:
            detector = FaceDetectorModels(int(args.detector))
            name = str(args.name)
            print( "Parameters: {}".format(detector))

            process_faceenrollment(detector, cam_index, cam_resolution)

            print("")
            print("Processing of video recording started...")
            video_to_images(detector, "x" + INPUT_DIR_DATASET, name)
            video_to_images(detector, INPUT_DIR_DATASET, name, one_image_only=True)
            print("Processing of video recording completed!")
            print("Make sure to train the new datasets before testing!")
            print("")
        except:
            print( "Invalid parameter" )
        return
    run(cam_index, cam_resolution, str(args.name))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--detector', required=False, default=0,
        help='Detector model to use. Options: 0-HAARCASCADE, 1-DLIBHOG, 2-DLIBCNN, 3-SSDRESNET, 4-MTCNN, 5-FACENET')
    parser.add_argument('--webcam', required=False, default=0, 
        help='Camera index to use. Default is 0. Assume only 1 camera connected.)')
    parser.add_argument('--resolution', required=False, default=0,
        help='Camera resolution to use. Default is 0. Options: 0-QVGA, 1-VGA, 2-HD, 3-FULLHD')
    parser.add_argument('--name', required=False, default="Unknown",
        help='Name of person to enroll')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
