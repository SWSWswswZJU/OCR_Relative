import cv2
import numpy as np
from general_imp_func import get_samples

# ========================
# 在指定路径中计算HOG特征
# ========================
def computeHOGs(pos_elements_path):
    gray_image_list = get_samples(pos_elements_path)
    hist_list = []
    for cur_gray_image in gray_image_list:
        winSize = (24,32)
        blockSize = (8,8)
        blockStride = (4,4)
        cellSize = (4,4)
        nbins = 9
        hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
        # compute
        winStride = (4,4)
        padding = (0,0)
        hist = hog.compute(cur_gray_image, winStride, padding)
        hist_list.append(hist)
    count = len(hist_list)
    return count, hist_list

def get_svm_detector(svm):
    sv = svm.getSupportVectors()
    rho, _, _ = svm.getDecisionFunction(0)
    sv = np.transpose(sv)
    return np.append(sv, [[-rho]], 0)


if __name__ == '__main__':
    # 第一步计算HOG特征
    gradien_list = []
    labels = []
    hard_neg_list = []

    # 正样本以及label导入
    #  pos_num, gradien_list_pos = computeHOGs('C:\\Users\\SLJ\\Desktop\\OCR_Project\\sample_R')
    pos_num, gradien_list_pos = computeHOGs('C:\\Users\\sunlj4\\Desktop\\OCR_Project_HOG+SVM\\sample_RR')
    [labels.append(+1) for _ in range(pos_num)]
    # 负样本以及label导入
    neg_num , gradien_list_neg = computeHOGs('C:\\Users\\sunlj4\\Desktop\\OCR_Project_HOG+SVM\\sample_FF')
    [labels.append(-1) for _ in range(neg_num)]
    gradien_list = gradien_list_pos + gradien_list_neg


    # 第二部：创建并且训练SVM
    svm = cv2.ml.SVM_create()
    svm.setCoef0(0)
    svm.setCoef0(0.0)
    svm.setDegree(3)
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 1000, 1e-3)
    svm.setTermCriteria(criteria)
    svm.setGamma(0)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setNu(0.5)
    svm.setP(0.1)  # for EPSILON_SVR, epsilon in loss function?
    svm.setC(0.01)  # From paper, soft classifier
    svm.setType(cv2.ml.SVM_EPS_SVR)  # C_SVC # EPSILON_SVR # may be also NU_SVR # do regression task
    svm.train(np.array(gradien_list), cv2.ml.ROW_SAMPLE, np.array(labels))

    # 第三部，加入识别错误样本，进行第二轮识别

    # 第四部：保存训练结果
    winSize = (24, 32)
    blockSize = (8, 8)
    blockStride = (4, 4)
    cellSize = (4, 4)
    nbins = 9
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    hog.setSVMDetector(get_svm_detector(svm))

    # 直接测试
    img = cv2.imread('C:\\Users\\sunlj4\\Desktop\\OCR_Project_HOG+SVM\\958.bmp', cv2.IMREAD_GRAYSCALE)
    # 选择ROI
    x = 560
    y = 432
    height = 216
    width = 800
    ROI = img[x:x + height, y:y + width]
    # 用自适应阈值方法
    constValue = 8
    blockSize = 13
    maxVal = 255
    adaptiveMethod = 0
    thresholdType = 1
    auto_thre_Image = cv2.adaptiveThreshold(ROI, maxVal, adaptiveMethod, thresholdType, blockSize, constValue)
    thre_img_inv = cv2.bitwise_not(auto_thre_Image)
    cv2.imshow('auto_thre_Image', auto_thre_Image)
    cv2.waitKey(0)
    # 寻找区域
    rects, wei = hog.detectMultiScale(auto_thre_Image, winStride = (4, 4), padding = (0, 0), scale = 1.05)
    for (x, y, w, h) in rects:
        cv2.rectangle(ROI, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.imshow('a', ROI)
    cv2.waitKey(0)

    debug = 9




