
import cv2
import numpy as np
import IMP_SD as imp


if __name__ == '__main__':
    # L = imp.file_name('E:\\OCR_DO\\OCR_PRO\\TrainNumber\\StandardRandom2432\\0')
    imgs,labels = imp.get_pos_samples('E:\\OCR_DO\\OCR_PRO\\TrainNumber\\StandardRandom2432\\0','C:\\Users\\SLJ\\Desktop\\ImageSave\\')
    for img in imgs:
        cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
        cv2.imshow('Image', img)
        cv2.waitKey(0)
        curHist = imp.computeHOGs(img)
        print(curHist)
        debg = 3


    another = 23