
import cv2
import numpy as np
import random
import IMP_SD


if __name__ == "__main__":
    img = cv2.imread('C:\\Users\\sunlj4\\Desktop\\OCR_Project_HOG+SVM\\958.bmp', cv2.IMREAD_GRAYSCALE)
    x = 560
    y = 432
    height = 216
    width = 800
    ROI = img[x:x+height, y:y+width]

    constValue = 8
    blockSize = 13
    maxVal = 255
    adaptiveMethod = 0
    thresholdType = 1
    auto_thre_Image = cv2.adaptiveThreshold(ROI, maxVal, adaptiveMethod, thresholdType, blockSize, constValue)
    thre_img_inv = cv2.bitwise_not(auto_thre_Image)


    cv2.namedWindow("SHOW", cv2.WINDOW_NORMAL)
    cv2.imshow("SHOW", auto_thre_Image)
    cv2.waitKey(0)



    # L_file = IMP_SD.file_name("C:\\Users\\SLJ\\Desktop\\OCR_Project\\sample_F")
    # i = 0
    # save_path = "C:\\Users\\SLJ\\Desktop\\sample_FF\\"
    # for cur_file_name in L_file:
    #     img = cv2.imread(cur_file_name, cv2.IMREAD_GRAYSCALE)
    #     img_resize = cv2.resize(img, (24, 32))
    #     res, img_resize_thre = cv2.threshold(img_resize, 100, 255, cv2.THRESH_OTSU)
    #     save_path_full = save_path + str(i) +  ".bmp"
    #     cv2.imwrite(save_path_full, img_resize_thre)
    #     i += 1
    #########################################################################################

    # img = cv2.imread('C:\\Users\\SLJ\\Desktop\\OCR_Project\\number.png', cv2.IMREAD_COLOR)
    # gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # res, thre_image = cv2.threshold(gray_image, 100, 255, cv2.THRESH_OTSU)
    # thre_image_inv = cv2.bitwise_not(thre_image)
    # print(gray_image.shape)
    #
    # height = 25
    # width = 20
    # i = 0
    # write_path = 'C:\\Users\\SLJ\\Desktop\\OCR_Project\\sample\\'
    # while(i < 10000):
    #     RandomRow = random.randint(0, 75)
    #     RandomCol = random.randint(0, 268)
    #     if((RandomRow + height) < 75 and (RandomCol + width) < 268):
    #         image_cut = thre_image_inv[RandomRow:RandomRow+height,RandomCol:RandomCol+width]
    #         # cv2.imshow('image_cut', image_cut)
    #         # cv2.waitKey(0)
    #         save_path = write_path + str(i) + '.bmp'
    #         cv2.imwrite(save_path, image_cut)
    #     i += 1
    #
    #
    # cv2.namedWindow('ImageShow', cv2.WINDOW_NORMAL)
    # cv2.imshow('ImageShow', thre_image_inv)
    # cv2.waitKey(0)


