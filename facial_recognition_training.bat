:: detector models:       0-HAARCASCADE, 1-DLIBHOG, 2-DLIBCNN, 3-SSDRESNET, 4-MTCNN
:: encoder models:        0-LBPH, 1-OPENFACE, 2-DLIBRESNET
:: classifier algorithms: 0-NAIVE_BAYES, 1-LINEAR_SVM, 2-RBF_SVM, 3-NEAREST_NEIGHBORS, 4-DECISION_TREE, 5-RANDOM_FOREST, 6-NEURAL_NET, 7-ADABOOST, 8-QDA

python facial_recognition_training.py -h
python facial_recognition_training.py --detector 0 --encoder 0 --classifier 0
pause
