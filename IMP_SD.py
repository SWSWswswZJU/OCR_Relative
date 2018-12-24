
import cv2
import numpy as np
import os

def VerticalProjection(image):
    # 绘制垂直方向直方图
    image_rows = image.shape[0]
    image_cols = image.shape[1]
    # 记录直方图的数据
    vec_col = np.array(range(image_cols))
    for col in range(0, image_cols):
        cur_col = image[:,col]
        per_col255_count = 0
        for row in range(0, image_rows):
            cur_pix = cur_col[row]
            if cur_pix == 255:
                per_col255_count = per_col255_count + 1
        vec_col[col] = per_col255_count
    # 开始绘制垂直方向直方图
    image_vertical = np.zeros(image_rows,image_cols)
    cv2.namedWindow("ImageVertical",cv2.WINDOW_NORMAL)
    cv2.imshow("ImageVertical",image_vertical)
    cv2.waitKey(0)

    print(vec_col)

    AAAA = 3

def load_txt_info(dirname):
    f = open(dirname,'a')
    for i in range(9,30):
        content = '\nE:/OCR_DO/OCR_PRO/TrainNumber/StandardRandom2432/0/' + str(i) + '.bmp'
        f.writelines(content)
    f.close()

def file_name(file_dir):
    L=[]
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.bmp':
                L.append(os.path.join(root, file))
    return L

# 加载正样本
def get_samples(file_dir):
    imgs = []
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.bmp':
                L.append(os.path.join(root,file))
    for filename in L:
        #print('filename = ',filename)
        src = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
        imgs.append(src)
    return imgs

# 计算HOG特征
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


