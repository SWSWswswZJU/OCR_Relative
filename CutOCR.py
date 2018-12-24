
import cv2
import numpy as np
def test():
    img = cv2.imread('IMG_2808.JPG', cv2.IMREAD_COLOR)  # 读入图像
    # 转化为灰度图
    GrayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 将灰度图转化为二值图
    res, mask = cv2.threshold(GrayImage, 100, 255, cv2.THRESH_BINARY)
    # 获取图像的信息
    height, width, channel = img.shape
    # 获取图像的ROI
    ROI = GrayImage[500:1500, 1000:2000]
    GrayImage[100:1100, 500:1500] = ROI
    # 图像的相加操作
    GrayImage_ADD1 = cv2.add(GrayImage, GrayImage)
    GrayImage_ADD2 = GrayImage + GrayImage
    GrayImage_ADD3 = cv2.addWeighted(GrayImage, 0.3, GrayImage_ADD1, 0.7, 80)
    # 图像取反
    mask_inv = cv2.bitwise_not(mask)
    # 像素点操作
    for i in range(400, 600):
        mask_inv[i, :] = 255
    cv2.namedWindow("ImageShow", cv2.WINDOW_NORMAL)
    cv2.imshow("ImageShow", mask_inv)
    cv2.waitKey(0)

def VerticalProjection(binaryImage):
    per_pixel_value = 0

    # 获取每一列
    image_rows = binaryImage.shape[0]
    image_cols = binaryImage.shape[1]
    vec_col = np.zeros(image_cols)
    for col in range(0, image_cols):
        num_cur_col255 = 0
        cur_col = binaryImage[:, col]
        for row in range(0, image_rows):
            per_pixel_value = cur_col[row]
            if per_pixel_value == 255:
                num_cur_col255 = num_cur_col255 + 1
        vec_col[col] = num_cur_col255

    # 绘制垂直方向的直方图
    print(vec_col)
    img_vertical_projection = np.zeros((image_rows, image_cols))
    for j in range(0, image_cols):
        num = vec_col[j]
        if not num == 0:
            i = image_rows
            while num > 0:
                img_vertical_projection[i - 1, j] = 255
                num = num - 1
                i = i - 1

    # 一种较为优秀的分割方法



    cv2.namedWindow('srcImage', cv2.WINDOW_NORMAL)
    cv2.imshow('srcImage', binaryImage)
    cv2.namedWindow("ImageShow",cv2.WINDOW_NORMAL)
    cv2.imshow("ImageShow",img_vertical_projection)
    cv2.waitKey(0)





def GetStandardElements(srcImage):
    grayImage = cv2.cvtColor(srcImage, cv2.COLOR_BGR2GRAY)
    constValue = 8
    blockSize = 13
    maxVal = 255
    adaptiveMethod = 0
    thresholdType = 1
    auto_thre_Image = cv2.adaptiveThreshold(grayImage, maxVal, adaptiveMethod, thresholdType, blockSize, constValue)

    #res, threImage = cv2.threshold(grayImage, 0, 255, cv2.THRESH_OTSU)
    #thre_image_inv = cv2.bitwise_not(threImage)
    # 用垂直投影法分割字符
    VerticalProjection(auto_thre_Image)

    cv2.namedWindow("Image",cv2.WINDOW_NORMAL)
    cv2.imshow("Image", auto_thre_Image)
    cv2.waitKey(0)


    ABC = 5

if __name__ == '__main__':
    srcImage = cv2.imread('C:/Users/sunlj4/Desktop/111.PNG',cv2.IMREAD_COLOR)
    #srcImage = cv2.imread('C:/Users/sunlj4/Desktop/OCR_DO/OCR_PRO/SampleImageSave/7.JPG',cv2.IMREAD_COLOR)
    GetStandardElements(srcImage)

    AAA = 3