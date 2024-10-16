#1.摄像头识别，能正常框出人脸，没有名字
import cv2
import face_recognition

# 打开摄像头
video_capture = cv2.VideoCapture(0)

while True:
    # 从摄像头捕获一帧画面
    ret, frame = video_capture.read()

    # 将捕获的画面转换为RGB格式
    rgb_frame = frame[:, :, ::-1]

    # 使用face_recognition库检测人脸
    face_locations = face_recognition.face_locations(rgb_frame)

    # 在画面上画出人脸边框
    for (top, right, bottom, left) in face_locations:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
    


    # 显示带有人脸边框的画面
    cv2.imshow('Video', frame)

    # 按下'q'键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头资源并关闭窗口
video_capture.release()
cv2.destroyAllWindows()

