:: detector models: 0-HAARCASCADE, 1-DLIBHOG, 2-DLIBCNN, 3-SSDRESNET, 4-MTCNN
:: encoder models:  0-LBPH, 1-OPENFACE, 2-DLIBRESNET

python facial_recognition_testing_image.py -h
python facial_recognition_testing_image.py --detector 0 --encoder 0 --image datasets/rico/1.jpg

pause
