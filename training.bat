:: detector models:       0-HAARCASCADE, 1-DLIBHOG, 2-DLIBCNN, 3-SSDRESNET, 4-MTCNN, 5-FACENET
:: encoder models:        0-LBPH, 1-OPENFACE, 2-DLIBRESNET, 3-FACENET
:: classifier algorithms: 0-NAIVE_BAYES, 1-LINEAR_SVM, 2-RBF_SVM, 3-NEAREST_NEIGHBORS, 4-DECISION_TREE, 5-RANDOM_FOREST, 6-NEURAL_NET, 7-ADABOOST, 8-QDA
:: synthesizer models:    0-TTSX3, 1-TACOTRON, 2-GOOGLECLOUD

python training.py -h
::python training.py --detector 0 --encoder 0 --classifier 0
python training.py --detector 4 --encoder 1 --classifier 1 --set_speech_synthesizer True --speech_synthesizer 0
pause
