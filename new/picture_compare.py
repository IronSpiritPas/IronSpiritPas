import cv2
import imutils
from skimage.metrics import structural_similarity as compare_ssim

# 加载图像
imageA = cv2.imread("C:\\Users\\U\\Desktop\\images\\different_pictures\\1.png")
imageB = cv2.imread("C:\\Users\\U\\Desktop\\images\\different_pictures\\2.png")

# 转为灰度图像
grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

# 计算SSIM相似性指数
(score, diff) = compare_ssim(grayA, grayB, full=True)
diff = (diff * 255).astype("uint8")
print("SSIM:{}".format(score))

# 二值化差异图像
thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

# 查找轮廓,findContours 第一个是轮廓，第二个是层次结构
# 查找轮廓并忽略层次结构
contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]

# 如果找到轮廓，绘制矩形框
if len(contours) > 0:
    for c in contours:
        if len(c) > 0:  # 确保轮廓不是空的
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)
else:
    print("没有找到轮廓")

# 显示结果
cv2.imshow("Image A", imageA)
cv2.imshow("Image B", imageB)
cv2.imwrite("result_image.png", imageB)
cv2.waitKey(0)
cv2.destroyAllWindows()
