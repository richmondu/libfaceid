:: detector models:   0-HAARCASCADE, 1-DLIBHOG, 2-DLIBCNN, 3-SSDRESNET, 4-MTCNN
:: encoder models:    0-LBPH, 1-OPENFACE, 2-DLIBRESNET
:: camera resolution: 0-QVGA, 1-VGA, 2-HD, 3-FULLHD

python facial_recognition_testing_webcam.py -h
python facial_recognition_testing_webcam.py --detector 0 --encoder 0 --webcam 0 --resolution 0

pause
