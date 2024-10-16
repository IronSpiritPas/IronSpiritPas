import cv2
import numpy as np
import face_recognition
import pickle
# import json

# with open('C:\\Users\\U\\Desktop\\yolov10\\Chinese_license_plate_detection_recognition-main\\config.json', 'r') as f:
#         config = json.load(f)
# # 加载人脸编码

def load_face_encodings(face_model):
    with open(face_model, 'rb') as pkl_file:
        all_known_face_encodings = pickle.load(pkl_file)
    return all_known_face_encodings

# all_known_face_encodings = load_face_encodings(config['face_model'])

def process_frame_for_face_recognition(img, all_known_face_encodings, name_dict):
    # 将图像缩小四分之一加快处理速度
    rgb_small_frame = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    # 标注检测到的人脸
    face_names = []
    for face_encoding in face_encodings:
        name = "Unknown"
        # 在已知编码中查找匹配
        for i, known_face_encoding in enumerate(all_known_face_encodings):
            face_dis = face_recognition.face_distance([known_face_encoding], face_encoding)
            if face_dis < 0.6:
                name = name_dict.get(i+1, "Unknown")
        face_names.append(name)

    # 在原图像上绘制框和名字
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(img, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

    return img
