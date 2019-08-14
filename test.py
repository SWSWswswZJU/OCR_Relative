import cv2
import numpy as np
import general_imp_func as imp
from matplotlib import pyplot as plt

if __name__ == '__main__':


    path = 'C:\\Users\\sunlj4\Desktop\\comacRecog\\PicProcess\\danju'
    img_path_List = imp.GetFileName_FromPath(path)
    for cur_img_path in img_path_List:
        cur_img = cv2.imread(cur_img_path, cv2.IMREAD_GRAYSCALE)
        plt.imshow(cur_img, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        plt.show()


    path = 'E:\\PyCharmProject\\PyOCRProject\\W1234'
    img_path_List = imp.GetFileName_FromPath(path)
    for cur_img_path in img_path_List:
        cur_img = cv2.imread(cur_img_path, cv2.IMREAD_GRAYSCALE)

        # 选择ROI
        x = 460
        y = 432
        height = 216
        width = 800
        ROI = cur_img[x:x + height, y:y + width]
        # 用自适应阈值方法
        constValue = 8
        blockSize = 13
        maxVal = 255
        adaptiveMethod = 0
        thresholdType = 1
        auto_thre_Image = cv2.adaptiveThreshold(ROI, maxVal, adaptiveMethod, thresholdType, blockSize, constValue)
        thre_img_inv = cv2.bitwise_not(auto_thre_Image)


        cv2.imshow('thre_img_inv', thre_img_inv)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        SHOW_IMAGE = ROI
        #  findContours
        # img_contours, contours, hierarchy = cv2.findContours(auto_thre_Image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # index = 0
        # for cur_contours in contours:
        #     x, y, w, h = cv2.boundingRect(cur_contours)
        #     if (w / h) > 0.5 and (w / h) < 2 and w*h > 100:
        #         img_cut = ROI[x:x + h, y:y + w]
        #         # cv2.imshow('img_cut', img_cut)
        #         # cv2.waitKey(0)
        #         img_cut_resize = cv2.resize(img_cut, (24, 32))
        #         cur_path_save = 'C:/Users/sunlj4/Desktop/Sample/random_sample/' + str(index) + '.bmp'
        #         cv2.imwrite(cur_path_save, img_cut_resize)
        #         cv2.rectangle(SHOW_IMAGE, (x, y), (x+w, y+h), (0,255,0), 2)
        #         index += 1
        #
        # cv2.namedWindow('ImageShow', cv2.WINDOW_NORMAL)
        # cv2.imshow('ImageShow', SHOW_IMAGE)
        # cv2.imshow('auto_thre_Image',auto_thre_Image)
        # cv2.waitKey(0)

        SHOW_IMAGE = ROI

        #  findContours
        img_contours, contours, hierarchy = cv2.findContours(auto_thre_Image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        index = 0
        for cur_contours in contours:
            x, y, w, h = cv2.boundingRect(cur_contours)
            if (w / h) > 0.5 and (w / h) < 2 and w*h > 100:
                img_cut = ROI[x:x + h, y:y + w]
                # cv2.imshow('img_cut', img_cut)
                # cv2.waitKey(0)
                img_cut_resize = cv2.resize(img_cut, (24, 32))
                cur_path_save = 'C:/Users/sunlj4/Desktop/Sample/random_sample/' + str(index) + '.bmp'
                cv2.imwrite(cur_path_save, img_cut_resize)
                cv2.rectangle(SHOW_IMAGE, (x, y), (x+w, y+h), (0,255,0), 2)
                index += 1

        cv2.namedWindow('ImageShow', cv2.WINDOW_NORMAL)
        cv2.imshow('ImageShow', SHOW_IMAGE)
        cv2.imshow('auto_thre_Image',auto_thre_Image)
        cv2.waitKey(0)






    AA = 3



