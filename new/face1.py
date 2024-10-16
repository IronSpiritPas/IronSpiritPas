#框出人脸以后，写名字
import argparse
import time
import cv2
import numpy as np
from ultralytics import YOLO
import face_recognition
import pickle
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker


# 定义数据库模型
Base = declarative_base()

class Face(Base):
    __tablename__ = 'face'
    id = Column(Integer, primary_key=True)
    name = Column(String(255))
    age = Column(Integer)

    def __repr__(self):
        return "Face(id:{}, name:{}, age:{})".format(self.id, self.name, self.age)

class SqliteSqlalchemy:
    def __init__(self):
        engine = create_engine('sqlite:///./sqlalchemy.db', echo=True)
        Base.metadata.create_all(engine, checkfirst=True)
        self.session = sessionmaker(bind=engine)()

    def get_faces(self):
        faces = self.session.query(Face).all()
        print("Fetched faces:", faces)  # 检查提取的人脸
        return faces

# 初始化数据库
db = SqliteSqlalchemy()
faces = db.get_faces()

# 映射人脸名称到编码
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



if __name__ == '__main__':
    # 解析参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--face_model', type=str, default='C:\\Users\\U\\Desktop\\yolov10\\Chinese_license_plate_detection_recognition-main\\z\\facedata.pkl')
    opt = parser.parse_args()

    # 加载人脸编码
    all_known_face_encodings = load_face_encodings(opt.face_model)


    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print("无法打开摄像头")
        exit()

    while True:
        ret, img = capture.read()
        if not ret:
            print("无法读取视频帧")
            break
        
        process_frame_for_face_recognition(img, all_known_face_encodings, name_dict)
        cv2.imshow('Video', img)
        if cv2.waitKey(1) == ord('q'):
            break

