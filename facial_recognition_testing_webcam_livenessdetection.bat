:: detector models:   0-HAARCASCADE, 1-DLIBHOG, 2-DLIBCNN, 3-SSDRESNET, 4-MTCNN, 5-FACENET
:: encoder models:    0-LBPH, 1-OPENFACE, 2-DLIBRESNET, 3-FACENET
:: liveness models:   0-EYESBLINK_MOUTHOPEN
:: camera resolution: 0-QVGA, 1-VGA, 2-HD, 3-FULLHD

python facial_recognition_testing_webcam_livenessdetection.py -h
python facial_recognition_testing_webcam_livenessdetection.py --detector 0 --encoder 0 --liveness 0 --webcam 0 --resolution 0

pause
