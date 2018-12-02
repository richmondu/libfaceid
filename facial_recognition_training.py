import os
import sys
import argparse
from libfaceid.detector import FaceDetectorModels, FaceDetector
from libfaceid.encoder  import FaceEncoderModels, FaceEncoder
from libfaceid.classifier  import FaceClassifierModels



INPUT_DIR_DATASET         = "datasets"
INPUT_DIR_MODEL_DETECTION = "models/detection/"
INPUT_DIR_MODEL_ENCODING  = "models/encoding/"
INPUT_DIR_MODEL_TRAINING  = "models/training/"


def ensure_directory(file_path):
    directory = os.path.dirname("./" + file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def train_recognition(model_detector, model_encoder, model_classifier, verify):

    ensure_directory(INPUT_DIR_DATASET)
    ensure_directory(INPUT_DIR_MODEL_TRAINING)
    face_detector = FaceDetector(model=model_detector, path=INPUT_DIR_MODEL_DETECTION)
    face_encoder = FaceEncoder(model=model_encoder, path=INPUT_DIR_MODEL_ENCODING, path_training=INPUT_DIR_MODEL_TRAINING, training=True)
    face_encoder.train(face_detector, path_dataset=INPUT_DIR_DATASET, verify=verify, classifier=model_classifier)

def run():
    detector=FaceDetectorModels.HAARCASCADE
#    detector=FaceDetectorModels.DLIBHOG
#    detector=FaceDetectorModels.DLIBCNN
#    detector=FaceDetectorModels.SSDRESNET
#    detector=FaceDetectorModels.MTCNN

    encoder=FaceEncoderModels.LBPH
#    encoder=FaceEncoderModels.OPENFACE
#    encoder=FaceEncoderModels.DLIBRESNET

    classifier=FaceClassifierModels.NAIVE_BAYES
#    classifier=FaceClassifierModels.LINEAR_SVM
#    classifier=FaceClassifierModels.RBF_SVM
#    classifier=FaceClassifierModels.NEAREST_NEIGHBORS
#    classifier=FaceClassifierModels.DECISION_TREE
#    classifier=FaceClassifierModels.RANDOM_FOREST
#    classifier=FaceClassifierModels.NEURAL_NET
#    classifier=FaceClassifierModels.ADABOOST
#    classifier=FaceClassifierModels.QDA

    train_recognition(detector, encoder, classifier, True)

def main(args):
    if args.detector and args.encoder:
        try:
            detector = FaceDetectorModels(int(args.detector))
            encoder = FaceEncoderModels(int(args.encoder))
            classifier = FaceEncoderModels(int(args.classifier))
            print( "Parameters: {} {} {}".format(detector, encoder, classifier) )
            train_recognition(detector, encoder, classifier, True)
            print( "Training completed!" )
        except:
            print( "Invalid parameter" )
        return
    run()

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--detector', required=False,
        help='Detector model to use. Options: 0-HAARCASCADE, 1-DLIBHOG, 2-DLIBCNN, 3-SSDRESNET, 4-MTCNN.')
    parser.add_argument('--encoder', required=False,
        help='Encoder model to use. Options: 0-LBPH, 1-OPENFACE, 2-DLIBRESNET.')
    parser.add_argument('--classifier', required=False,
        help='Classifier algorithm to use. Options: 0-NAIVE_BAYES, 1-LINEAR_SVM, 2-RBF_SVM, 3-NEAREST_NEIGHBORS, 4-DECISION_TREE, 5-RANDOM_FOREST, 6-NEURAL_NET, 7-ADABOOST, 8-QDA.')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
