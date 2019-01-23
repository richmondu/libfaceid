:: detector models: 0-HAARCASCADE, 1-DLIBHOG, 2-DLIBCNN, 3-SSDRESNET, 4-MTCNN, 5-FACENET
:: encoder models:  0-LBPH, 1-OPENFACE, 2-DLIBRESNET, 3-FACENET

python testing_image.py -h
python testing_image.py --detector 4 --encoder 0 --image datasets/Richmond/1.jpg

pause
