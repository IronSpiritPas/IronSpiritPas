from main1 import FastAPI, UploadFile, File, HTTPException
import cv2
import numpy as np
import time
import tempfile
from utils.cv_puttext import cv2ImgAddText
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression_face, scale_coords
from utils.torch_utils import select_device
from utils.plate_rec import get_plate_result, init_model
from utils.double_plate_split_merge import get_split_merge
from ultralytics import YOLO
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




# 加载人脸模型
face_model_path = "C:\\Users\\U\\Desktop\\new\\face_recognition_api\\facedata.pkl"
all_known_face_encodings = load_face_encodings(face_model_path)


app = FastAPI()

@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    try:
        # 将上传的文件读取为字节
        file_bytes = await file.read()
        
        # 将字节数据转换为图像
        np_img = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        # 调用人脸识别处理函数
        annotated_img = process_frame_for_face_recognition(img, all_known_face_encodings, name_dict)

        # 将结果图像返回给客户端
        _, img_encoded = cv2.imencode('.jpg', annotated_img)
        return {
            "filename": file.filename,
            "annotated_image": img_encoded.tobytes()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

