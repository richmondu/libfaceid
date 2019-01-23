:: detector models:           0-HAARCASCADE, 1-DLIBHOG, 2-DLIBCNN, 3-SSDRESNET, 4-MTCNN, 5-FACENET
:: encoder models:            0-LBPH, 1-OPENFACE, 2-DLIBRESNET, 3-FACENET
:: speech synthesizer models: 0-TTSX3, 1-TACOTRON, 2-GOOGLECLOUD
:: speech recognizer models:  0-GOOGLECLOUD, 1-WITAI, 2-HOUNDIFY
:: camera resolution:         0-QVGA, 1-VGA, 2-HD, 3-FULLHD

python testing_webcam_voiceenabled_voiceactivated.py -h
python testing_webcam_voiceenabled_voiceactivated.py --detector 4 --encoder 0 --speech_synthesizer 0 --speech_recognizer 0 --webcam 0 --resolution 0

pause
