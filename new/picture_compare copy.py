import cv2
import imutils
from skimage.metrics import structural_similarity as compare_ssim

def process_videos(video_path1, video_path2):
    # 加载第一个视频
    cap1 = cv2.VideoCapture(video_path1)
    if not cap1.isOpened():
        print("无法打开第一个视频")
        return

    # 加载第二个视频
    cap2 = cv2.VideoCapture(video_path2)
    if not cap2.isOpened():
        print("无法打开第二个视频")
        return

    # 获取第一帧
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    if not ret1 or not ret2:
        print("无法读取视频帧")
        return

    # 转为灰度图像
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    while True:
        # 计算SSIM相似性指数
        (score, diff) = compare_ssim(gray1, gray2, full=True)
        diff = (diff * 255).astype("uint8")
        print("SSIM:{}".format(score))

        # 二值化差异图像
        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        # 查找轮廓，findContours 第一个是轮廓，第二个是层次结构
        # 查找轮廓并忽略层次结构
        contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 如果找到轮廓，绘制矩形框
        if len(contours) > 0:
            for c in contours:
                if len(c) > 0:  # 确保轮廓不是空的
                    x, y, w, h = cv2.boundingRect(c)
                    cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 0, 255), 2)
        else:
            print("没有找到轮廓")

        # 显示结果
        cv2.imshow("Frame 1", frame1)
        cv2.imshow("Frame 2", frame2)

        # 更新帧
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            break
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # 按q键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源并关闭窗口
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

# 调用函数处理两个视频
process_videos("C:\\Users\\U\\Videos\\Captures\\1.mp4", "C:\\Users\\U\\Videos\\Captures\\2.mp4")
