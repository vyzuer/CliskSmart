import numpy as np
import cv2, cv
import matplotlib.pyplot as plt
import os, sys
from skimage import io


def detect_face(img, visualise=False, dump=False, dump_path='', padding=0):

    face_cascade_path = '/home/vyzuer/opencv-2.4.9/data/haarcascades/haarcascade_frontalface_alt.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_h, img_w, img_d = img.shape
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, flags = cv.CV_HAAR_SCALE_IMAGE)
    
    frames = []
    # find the body frames
    if len(faces) > 0:
        frames = np.zeros(shape=(faces.shape), dtype=int)

    i = 0
    
    for (x,y,w,h) in faces:
        # find the full body frame
        x_0 = x - w - padding
        y_0 = y - h/2 - padding

        x_1 = x_0 + 3*w + 2*padding
        y_1 = y_0 + 9*h + 2*padding

        # boundary check
        if x_0 < 0:
            x_0 = 0
        if y_0 < 0:
            y_0 = 0
        if x_1 > img_w-1:
            x_1 = img_w-1
        if y_1 > img_h-1:
            y_1 = img_h-1

        frames[i] = x_0, y_0, x_1, y_1

        i += 1

        if visualise:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.rectangle(img,(x_0,y_0),(x_1, y_1), (0,0,255), 2)

    if visualise:
        plt.imshow(img)
        plt.show()
        plt.close('all')

    if dump and dump_path:
            cv2.imwrite(dump_path, img)

    return faces, frames

if __name__ == '__main__':

    if len(sys.argv) != 2:
        print "Usage : image_source"
        sys.exit(0)

    img_src = sys.argv[1]

    img = io.imread(img_src)

    detect_face(img, visualise=True)

