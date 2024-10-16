# -*- coding: UTF-8 -*-
import argparse
import time
import json
from pathlib import Path
import os
import cv2
import copy
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from ultralytics import YOLO
import subprocess
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.plate_rec import get_plate_result,allFilePath,init_model,cv_imread
from utils.general import check_img_size, non_max_suppression_face, scale_coords
from utils.torch_utils import select_device
import supervision as sv
import face_recognition
import pickle
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Define database model
Base = declarative_base()

class Face(Base):
    __tablename__ = 'face'
    id = Column(Integer, primary_key=True)
    name = Column(String(255))
    age = Column(Integer)

    def __repr__(self):
        return "Face(id:{}, name:{}, age:{})".format(self.id, self.name,self.age)

class SqliteSqlalchemy:
    def __init__(self):
        engine = create_engine('sqlite:///./sqlalchemy.db', echo=True)
        Base.metadata.create_all(engine, checkfirst=True)
        self.session = sessionmaker(bind=engine)()

    def get_faces(self):
        faces = self.session.query(Face).all()
        print("Fetched faces:", faces)  # Check fetched faces
        return faces

# Initialize the database
db = SqliteSqlalchemy()
faces = db.get_faces()

# Map face names to encodings
name_dict = {face.id: face.name for face in faces}

def load_face_encodings(face_model):
    with open(face_model, 'rb') as pkl_file:
        all_known_face_encodings = pickle.load(pkl_file)
    return all_known_face_encodings

def process_frame_for_face_recognition(img, all_known_face_encodings, name_dict):
    
    # 将帧缩小四分之一以加速人脸识别
    rgb_small_frame = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)

    t0=time.time()
    # 使用face_recognition库检测人脸
    face_locations = face_recognition.face_locations(rgb_small_frame)
    t1=time.time()
    print("检测人脸",t1-t0)
    if(len(face_locations)>0):
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        t2=time.time()
        print("查找这是谁",t2-t1)

        #在画面中显示人脸名称
        face_names = []
        for face_encoding in face_encodings:
            min_dis = 10000.0
            name = "Unknown"

            for i, known_face_encoding in enumerate(all_known_face_encodings):
                second_array = known_face_encoding[1][0]
                known_face_encoding = np.array(second_array)

                # 只处理形状为128的已知人脸编码
                if known_face_encoding.shape[0] == 128:
                    face_dis = face_recognition.face_distance([known_face_encoding], face_encoding)
                    if len(face_dis) > 0:
                        face_mean = np.mean(face_dis)
                        if face_mean < min_dis:
                            min_dis = face_mean
                            name = name_dict.get(i + 1, "Unknown")

            if min_dis > 0.63:
                name = "Unknown"

            face_names.append(name)
            t3=time.time()
            print("写人名",t3-t2)

        # 调整人脸位置到原始尺寸并绘制框和名字
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4  # 恢复到原始比例
            right *= 4
            bottom *= 4
            left *= 4

            # 绘制人脸框
            cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
            # 在框下绘制名字
            cv2.rectangle(img, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            cv2.putText(img, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)
            t4=time.time()
            print("画框",t4-t3)

    return img


# Define database model
Base = declarative_base()

class Face(Base):
    __tablename__ = 'face'
    id = Column(Integer, primary_key=True)
    name = Column(String(255))
    age = Column(Integer)

    def __repr__(self):
        return "Face(id:{}, name:{}, age:{})".format(self.id, self.name,self.age)



def load_model(weights, device):
    model = attempt_load(weights, map_location=device)  # load FP32 model
    return model

def load_yolo_model(weights, device):
    model = YOLO(weights)  # 使用 Ultralytics YOLO 加载模型
    return model


category_dict = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush"
}

def yolo_deal(frame, model):
    yolo_results = model(frame, conf=0.5)
    for result in yolo_results:
    # 解析结果并绘制框
        for detection in result.boxes.data:
            x1, y1, x2, y2, conf, cls = detection
            class_name = category_dict[int(cls)]  # 获取类别名称
            label = f'{class_name}: {conf:.2f}'  # 类别和置信度
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return yolo_results




if __name__ == '__main__':
    with open('C:\\Users\\U\\Desktop\\yolov10\\Chinese_license_plate_detection_recognition-main\\config.json', 'r') as f:
        config = json.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#这一句是python代码，不能放入json文件里，因为JSON文件主要用于存储数据，而不是执行代码。
    detect_model = load_model(config['detect_model'], device)
    plate_rec_model = init_model(device, config['rec_model'], is_color=config['is_color'])
    yolo_model = load_yolo_model(config['yolo_model'], device)
    # 加载人脸编码
    all_known_face_encodings = load_face_encodings(config['face_model'])

    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print("无法打开摄像头")
        exit()

    while True:
        ret, img = capture.read()
        if not ret:
            print("无法读取视频帧")
            break
        
        # YOLO 物体检测
        yolo_results = yolo_deal(img, yolo_model)

        # # 处理车牌识别
        # dict_list = detect_Recognition_plate(detect_model, img, device, plate_rec_model, 640, is_color=config['is_color'])

        #处理人脸识别
        img = process_frame_for_face_recognition(img, all_known_face_encodings, name_dict)

        # # 绘制检测结果
        # img = draw_result(img, dict_list, is_color=config['is_color'])

        cv2.imshow('Video', img)
        if cv2.waitKey(1) == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()
