import cv2
import numpy as np
import face_recognition
import pickle
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.responses import HTMLResponse
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import io

# FastAPI 应用初始化
app = FastAPI()

# 数据库模型
Base = declarative_base()

class Face(Base):
    __tablename__ = 'face'
    id = Column(Integer, primary_key=True)
    name = Column(String(255))
    age = Column(Integer)

# SQLite 数据库初始化
class SqliteSqlalchemy:
    def __init__(self):
        engine = create_engine('sqlite:///./sqlalchemy.db', echo=True)
        Base.metadata.create_all(engine, checkfirst=True)
        self.session = sessionmaker(bind=engine)()

    def get_faces(self):
        faces = self.session.query(Face).all()
        return faces

# 加载人脸数据库
db = SqliteSqlalchemy()
faces = db.get_faces()
name_dict = {face.id: face.name for face in faces}

# 加载人脸编码
def load_face_encodings(face_model):
    with open(face_model, 'rb') as pkl_file:
        all_known_face_encodings = pickle.load(pkl_file)
    return all_known_face_encodings

# 人脸识别处理逻辑
def process_frame_for_face_recognition(img, all_known_face_encodings, name_dict):
    rgb_small_frame = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        name = "Unknown"
        min_dis = 10000.0

        for i, known_face_encoding in enumerate(all_known_face_encodings):
            second_array = known_face_encoding[1][0]
            known_face_encoding = np.array(second_array)
            
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

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(img, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        cv2.putText(img, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

    return img

face_model_path = "C:\\Users\\U\\Desktop\\new\\face_recognition_api\\facedata.pkl"  # 请使用你实际的路径
all_known_face_encodings = load_face_encodings(face_model_path)

# 摄像头实时人脸识别
def video_stream():
    cap = cv2.VideoCapture(0)  # 0 是默认摄像头编号
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 处理当前帧的人脸识别
            annotated_frame = process_frame_for_face_recognition(frame, all_known_face_encodings, name_dict)

            # 将处理后的帧转换为 JPEG 格式
            _, jpeg = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = jpeg.tobytes()

            # 使用 StreamingResponse 持续输出视频帧
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        cap.release()

# 视频流接口
@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(video_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/")
async def main():
    content = """
    <body>
    <form action="/upload-image/" enctype="multipart/form-data" method="post">
    <input name="file" type="file">
    <input type="submit">
    </form>
    </body>
    """
    return HTMLResponse(content=content)


# FastAPI 应用启动入口
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
