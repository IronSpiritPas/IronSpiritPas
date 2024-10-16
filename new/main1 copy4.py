import cv2
import numpy as np
import face_recognition
import pickle
import base64
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

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
        print(name)
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

# 上传图片并处理人脸识别
@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="文件为空，无法读取上传的图像。")

        np_img = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="无法解码图像，可能是无效的图像文件。")
        
        annotated_img = process_frame_for_face_recognition(img, all_known_face_encodings, name_dict)
        print(4444444444)
        _, img_encoded = cv2.imencode('.jpg', annotated_img)
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')

        html_content = f"""
        <html>
        <body>
        <h3>处理后的图片:</h3>
        <img src="data:image/jpeg;base64,{img_base64}" alt="处理后的人脸识别图像">
        </body>
        </html>
        """
        return HTMLResponse(content=html_content)

    except Exception as e:
        print(f"处理图像时发生错误: {e}")
        raise HTTPException(status_code=500, detail=f"处理图像时发生错误: {str(e)}")


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
