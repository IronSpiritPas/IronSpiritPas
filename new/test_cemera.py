#这种列表的问题就是暂时无法显示图像，但是可以显示label
import cv2
from ultralytics import YOLO
import supervision as sv

def yolo_deal(frame, model):
    results = model(source=frame, conf=0.5, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    return detections

def check_2():
    # 加载模型
    model = YOLO("yolov8x.pt")

    # 创建摄像头对象
    cap = cv2.VideoCapture(0)

    # 获取视频帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"{fps} <---------视频帧率")

    # 定义类别列表
    category_list = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
        "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
        "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    ]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = yolo_deal(frame, model)
        labels = [
            f"{category_list[class_id]} {confidence:.2f}"
            for class_id, confidence in zip(detections.class_id, detections.confidence)
            if class_id < len(category_list)##是这里，类别列表逻辑控制！
        ]

        for label, box in zip(labels, detections.xyxy):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=3)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            

        for label in labels:
            print(label)
        cv2.imshow('Video', frame)


    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    check_2()

# from ultralytics import YOLO
# import cv2
# import subprocess
# import supervision as sv


# def yolo_deal(frame, model):
#     results = model(source=frame, conf=0.5, verbose=False)[0]
#     detections = sv.Detections.from_ultralytics(results)
#     return detections

# def check_2():
#     # 加载模型
#     model = YOLO("C:\\Users\\U\\Desktop\\yolov10\\yolov10-main\\yolov10x.pt")

#     # 创建摄像头对象
#     cap = cv2.VideoCapture(0)

#     # 获取视频帧率
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     print(f"{fps} <---------视频帧率")

#     # 定义类别字典
#     category_dict = {
#         0: "person",
#         1: "bicycle",
#         2: "car",
#         3: "motorcycle",
#         4: "airplane",
#         5: "bus",
#         6: "train",
#         7: "truck",
#         8: "boat",
#         9: "traffic light",
#         10: "fire hydrant",
#         11: "stop sign",
#         12: "parking meter",
#         13: "bench",
#         14: "bird",
#         15: "cat",
#         16: "dog",
#         17: "horse",
#         18: "sheep",
#         19: "cow",
#         20: "elephant",
#         21: "bear",
#         22: "zebra",
#         23: "giraffe",
#         24: "backpack",
#         25: "umbrella",
#         26: "handbag",
#         27: "tie",
#         28: "suitcase",
#         29: "frisbee",
#         30: "skis",
#         31: "snowboard",
#         32: "sports ball",
#         33: "kite",
#         34: "baseball bat",
#         35: "baseball glove",
#         36: "skateboard",
#         37: "surfboard",
#         38: "tennis racket",
#         39: "bottle",
#         40: "wine glass",
#         41: "cup",
#         42: "fork",
#         43: "knife",
#         44: "spoon",
#         45: "bowl",
#         46: "banana",
#         47: "apple",
#         48: "sandwich",
#         49: "orange",
#         50: "broccoli",
#         51: "carrot",
#         52: "hot dog",
#         53: "pizza",
#         54: "donut",
#         55: "cake",
#         56: "chair",
#         57: "couch",
#         58: "potted plant",
#         59: "bed",
#         60: "dining table",
#         61: "toilet",
#         62: "tv",
#         63: "laptop",
#         64: "mouse",
#         65: "remote",
#         66: "keyboard",
#         67: "cell phone",
#         68: "microwave",
#         69: "oven",
#         70: "toaster",
#         71: "sink",
#         72: "refrigerator",
#         73: "book",
#         74: "clock",
#         75: "vase",
#         76: "scissors",
#         77: "teddy bear",
#         78: "hair drier",
#         79: "toothbrush"
#     }

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         detections = yolo_deal(frame, model)
#         labels = [
#             f"{category_dict[class_id]} {confidence:.2f}"
#             for class_id, confidence in zip(detections.class_id, detections.confidence)
#             if class_id in category_dict
#         ]


#         for label, box in zip(labels, detections.xyxy):
#             x1, y1, x2, y2 = map(int, box)
#             cv2.rectangle(frame, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=3)
#             cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            
#             if label == 'person':
#                 subprocess.run(['python', 'C:\\Users\\U\\Desktop\\yolov10\\face_recognition-master\\test3.py'])

#         cv2.imshow('Video', frame)
#         if cv2.waitKey(1) == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     check_2()