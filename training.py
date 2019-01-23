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
OUTPUT_DIR_AUDIOSET       = "audiosets/"
INPUT_DIR_MODEL_SYNTHESIS = "models/synthesis/"


def ensure_directory(file_path):
    directory = os.path.dirname("./" + file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_dataset_names(file_path):
    for (_d, names, _f) in os.walk(file_path):
        return names
    return None

def train_recognition(model_detector, model_encoder, model_classifier, verify):

    ensure_directory(INPUT_DIR_DATASET)

    print("")
    names = get_dataset_names(INPUT_DIR_DATASET)
    if names is not None:
        print("Names " + str(names))
        for name in names:
            for (_d, _n, files) in os.walk(INPUT_DIR_DATASET + "/" + name):
                print(name + ": " + str(files))
    print("")

    ensure_directory(INPUT_DIR_MODEL_TRAINING)
    face_detector = FaceDetector(model=model_detector, path=INPUT_DIR_MODEL_DETECTION)
    face_encoder = FaceEncoder(model=model_encoder, path=INPUT_DIR_MODEL_ENCODING, path_training=INPUT_DIR_MODEL_TRAINING, training=True)
    face_encoder.train(face_detector, path_dataset=INPUT_DIR_DATASET, verify=verify, classifier=model_classifier)
    #print("train_recognition completed")
    
# generate audio samples for image datasets using text to speech synthesizer
def train_audiosets(model_speech_synthesizer):

    ensure_directory(OUTPUT_DIR_AUDIOSET)
    from libfaceid.speech_synthesizer import SpeechSynthesizer # lazy loading
    speech_synthesizer = SpeechSynthesizer(model=model_speech_synthesizer, path=INPUT_DIR_MODEL_SYNTHESIS, path_output=OUTPUT_DIR_AUDIOSET)
    speech_synthesizer.synthesize_datasets(INPUT_DIR_DATASET)
    #speech_synthesizer.synthesize_name("libfaceid")
    #speech_synthesizer.synthesize("Hello World", "World.wav")


def run():
#    detector=FaceDetectorModels.HAARCASCADE
#    detector=FaceDetectorModels.DLIBHOG
#    detector=FaceDetectorModels.DLIBCNN
#    detector=FaceDetectorModels.SSDRESNET
    detector=FaceDetectorModels.MTCNN
#    detector=FaceDetectorModels.FACENET

    encoder=FaceEncoderModels.LBPH
#    encoder=FaceEncoderModels.OPENFACE
#    encoder=FaceEncoderModels.DLIBRESNET
#    encoder=FaceEncoderModels.FACENET

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
    print( "\nImage dataset training completed!" )

    # generate audio samples for image datasets using text to speech synthesizer
    if True: # Set true to enable generation of audio for each person in datasets folder 
        from libfaceid.speech_synthesizer import SpeechSynthesizerModels # lazy loading
        speech_synthesizer = SpeechSynthesizerModels.TTSX3
        #speech_synthesizer = SpeechSynthesizerModels.TACOTRON
        #speech_synthesizer = SpeechSynthesizerModels.GOOGLECLOUD
        train_audiosets(speech_synthesizer)
        print( "Audio samples created!" )


def main(args):
    if args.detector and args.encoder:
        try:
            detector   = FaceDetectorModels(int(args.detector))
            encoder    = FaceEncoderModels(int(args.encoder))
            classifier = FaceClassifierModels(int(args.classifier))
            print( "Parameters: {} {} {}".format(detector, encoder, classifier) )

            train_recognition(detector, encoder, classifier, True)
            print( "\nImage dataset training completed!" )

            # generate audio samples for image datasets using text to speech synthesizer
            if args.set_speech_synthesizer:
                from libfaceid.speech_synthesizer import SpeechSynthesizerModels # lazy loading
                speech_synthesizer= SpeechSynthesizerModels(int(args.speech_synthesizer))
                #print( "Parameters: {}".format(speech_synthesizer) )
                train_audiosets(speech_synthesizer)
                print( "Audio samples created!" )
        except:
            print( "Invalid parameter" )
        return
    run()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--detector', required=False,
        help='Detector model to use. Options: 0-HAARCASCADE, 1-DLIBHOG, 2-DLIBCNN, 3-SSDRESNET, 4-MTCNN, 5-FACENET')
    parser.add_argument('--encoder', required=False,
        help='Encoder model to use. Options: 0-LBPH, 1-OPENFACE, 2-DLIBRESNET, 3-FACENET')
    parser.add_argument('--classifier', required=False, default=0,
        help='Classifier algorithm to use. Options: 0-NAIVE_BAYES, 1-LINEAR_SVM, 2-RBF_SVM, 3-NEAREST_NEIGHBORS, 4-DECISION_TREE, 5-RANDOM_FOREST, 6-NEURAL_NET, 7-ADABOOST, 8-QDA.')
    parser.add_argument('--set_speech_synthesizer', required=False, default=False,
        help='Use text to speech synthesizier.')
    parser.add_argument('--speech_synthesizer', required=False, default=0,
        help='Speech synthesizier algorithm to use. Options: 0-TTSX3, 1-TACOTRON, 2-GOOGLECLOUD')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
