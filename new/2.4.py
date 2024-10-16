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
import argparse
import time
import json
from pathlib import Path
import os
import cv2
import copy
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

    # 从json文件中读取参数
    with open('C:\\Users\\U\\Desktop\\new\\config.json', 'r') as f:
        config = json.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#这一句是python代码，不能放入json文件里，因为JSON文件主要用于存储数据，而不是执行代码。
    detect_model = load_model(config['detect_model'], device)
    plate_rec_model = init_model(device, config['rec_model'], is_color=config['is_color'])
    yolo_model = load_yolo_model(config['yolo_model'], device)
    # 加载人脸编码
    all_known_face_encodings = load_face_encodings(config['face_model'])

    cap = cv2.VideoCapture(0)  # 0 是默认摄像头编号
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 处理当前帧的人脸识别
            annotated_frame = process_frame_for_face_recognition(frame, all_known_face_encodings, name_dict)
            # 就剩这两个了
            # YOLO 物体检测
            yolo_results = yolo_deal(frame, yolo_model)

            # 处理车牌识别
            dict_list = detect_Recognition_plate(detect_model, frame, device, plate_rec_model, 640, is_color=config['is_color'])                                                                  

            # 绘制检测结果
            annotated_frame = draw_result(frame, dict_list, is_color=config['is_color'])
            #最后需要变成frame版本的


            # 将处理后的帧转换为 JPEG 格式
            _, jpeg = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = jpeg.tobytes()

            # 使用 StreamingResponse 持续输出视频帧
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        cap.release()

clors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255)]
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
    # cv2.imshow('original frame',frame)
    # detections = yolo_deal(frame, model)
    # detections = sv.Detections.from_ultralytics(results)
    # labels = [
    #         f"{category_dict[class_id]} {confidence:.2f}"
    #         for class_id, confidence in zip(detections.class_id, detections.confidence)
    #         if class_id in category_dict
    #     ]
    # for label, box in zip(labels, detections.xyxy):
    #         x1, y1, x2, y2 = map(int, box)
    #         cv2.rectangle(frame, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=3)
    #         cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    #         print(label)
    #         print(type(label))

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


def load_model(weights, device):
    model = attempt_load(weights, map_location=device)  # load FP32 model
    return model

def load_yolo_model(weights, device):
    model = YOLO(weights)  # 使用 Ultralytics YOLO 加载模型
    return model




def four_point_transform(image, pts):                       #透视变换得到车牌小图
    # rect = order_points(pts)
    rect = pts.astype('float32')
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def get_plate_rec_landmark(img, xyxy, conf, landmarks, class_num,device,plate_rec_model,is_color=False):  #获取车牌坐标以及四个角点坐标并获取车牌号
    h,w,c = img.shape
    result_dict={}
    tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness

    x1 = int(xyxy[0])
    y1 = int(xyxy[1])
    x2 = int(xyxy[2])
    y2 = int(xyxy[3])
    height=y2-y1
    landmarks_np=np.zeros((4,2))
    rect=[x1,y1,x2,y2]
    for i in range(4):
        point_x = int(landmarks[2 * i])
        point_y = int(landmarks[2 * i + 1])
        landmarks_np[i]=np.array([point_x,point_y])

    class_label= int(class_num)  #车牌的的类型0代表单牌，1代表双层车牌
    roi_img = four_point_transform(img,landmarks_np)   #透视变换得到车牌小图
    if class_label:        #判断是否是双层车牌，是双牌的话进行分割后然后拼接
        roi_img=get_split_merge(roi_img)
    if not is_color:
        plate_number,rec_prob = get_plate_result(roi_img,device,plate_rec_model,is_color=is_color)                 #对车牌小图进行识别
    else:
        plate_number,rec_prob,plate_color,color_conf=get_plate_result(roi_img,device,plate_rec_model,is_color=is_color) 
    # cv2.imwrite("roi.jpg",roi_img)
    result_dict['rect']=rect                      #车牌roi区域
    result_dict['detect_conf']=conf              #检测区域得分
    result_dict['landmarks']=landmarks_np.tolist() #车牌角点坐标
    result_dict['plate_no']=plate_number   #车牌号
    result_dict['rec_conf']=rec_prob   #每个字符的概率
    result_dict['roi_height']=roi_img.shape[0]  #车牌高度
    result_dict['plate_color']=""
    if is_color:
        result_dict['plate_color']=plate_color   #车牌颜色
        result_dict['color_conf']=color_conf    #颜色得分
    result_dict['plate_type']=class_label   #单双层 0单层 1双层
    
    return result_dict

def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):  #返回到原图坐标
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6]] -= pad[0]  # x padding
    coords[:, [1, 3, 5, 7]] -= pad[1]  # y padding
    coords[:, :8] /= gain
    #clip_coords(coords, img0_shape)
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    coords[:, 4].clamp_(0, img0_shape[1])  # x3
    coords[:, 5].clamp_(0, img0_shape[0])  # y3
    coords[:, 6].clamp_(0, img0_shape[1])  # x4
    coords[:, 7].clamp_(0, img0_shape[0])  # y4
    # coords[:, 8].clamp_(0, img0_shape[1])  # x5
    # coords[:, 9].clamp_(0, img0_shape[0])  # y5
    return coords

def detect_Recognition_plate(model, orgimg, device,plate_rec_model,img_size,is_color=False):#获取车牌信息
    # Load model
    # img_size = opt_img_size
    conf_thres = 0.3      #得分阈值
    iou_thres = 0.5       #nms的iou值   
    dict_list=[]
    # orgimg = cv2.imread(image_path)  # BGR
    img0 = copy.deepcopy(orgimg)
    assert orgimg is not None, 'Image Not Found ' 
    h0, w0 = orgimg.shape[:2]  # orig hw
    r = img_size / max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1  else cv2.INTER_LINEAR
        img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

    imgsz = check_img_size(img_size, s=model.stride.max())  # check img_size  

    img = letterbox(img0, new_shape=imgsz)[0]           #检测前处理，图片长宽变为32倍数，比如变为640X640
    # img =process_data(img0)
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416  图片的BGR排列转为RGB,然后将图片的H,W,C排列变为C,H,W排列

    # Run inference
    t0 = time.time()

    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    # t1 = time_synchronized()/
    pred = model(img)[0]
    # t2=time_synchronized()
    # print(f"infer time is {(t2-t1)*1000} ms")

    # Apply NMS
    pred = non_max_suppression_face(pred, conf_thres, iou_thres)

    # print('img.shape: ', img.shape)
    # print('orgimg.shape: ', orgimg.shape)

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orgimg.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class

            det[:, 5:13] = scale_coords_landmarks(img.shape[2:], det[:, 5:13], orgimg.shape).round()

            for j in range(det.size()[0]):
                xyxy = det[j, :4].view(-1).tolist()
                conf = det[j, 4].cpu().numpy()
                landmarks = det[j, 5:13].view(-1).tolist()
                class_num = det[j, 13].cpu().numpy()
                result_dict = get_plate_rec_landmark(orgimg, xyxy, conf, landmarks, class_num,device,plate_rec_model,is_color=is_color)
                dict_list.append(result_dict)
    return dict_list



def draw_result(orgimg,dict_list,is_color=False):   # 车牌结果画出来
    result_str =""
    for result in dict_list:
        rect_area = result['rect']
        
        x,y,w,h = rect_area[0],rect_area[1],rect_area[2]-rect_area[0],rect_area[3]-rect_area[1]
        padding_w = 0.05*w
        padding_h = 0.11*h
        rect_area[0]=max(0,int(x-padding_w))
        rect_area[1]=max(0,int(y-padding_h))
        rect_area[2]=min(orgimg.shape[1],int(rect_area[2]+padding_w))
        rect_area[3]=min(orgimg.shape[0],int(rect_area[3]+padding_h))

        height_area = result['roi_height']
        landmarks=result['landmarks']
        result_p = result['plate_no']
        if result['plate_type']==0:#单层
            result_p+=" "+result['plate_color']
        else:                             #双层
            result_p+=" "+result['plate_color']+"双层"
        result_str+=result_p+" "
        for i in range(4):  #关键点
            cv2.circle(orgimg, (int(landmarks[i][0]), int(landmarks[i][1])), 5, clors[i], -1)
        cv2.rectangle(orgimg,(rect_area[0],rect_area[1]),(rect_area[2],rect_area[3]),(0,0,255),2) #画框
        
        labelSize = cv2.getTextSize(result_p,cv2.FONT_HERSHEY_SIMPLEX,0.5,1) #获得字体的大小
        if rect_area[0]+labelSize[0][0]>orgimg.shape[1]:                 #防止显示的文字越界
            rect_area[0]=int(orgimg.shape[1]-labelSize[0][0])
        print(22222)
        orgimg=cv2.rectangle(orgimg,(rect_area[0],int(rect_area[1]-round(1.6*labelSize[0][1]))),(int(rect_area[0]+round(1.2*labelSize[0][0])),rect_area[1]+labelSize[1]),(255,255,255),cv2.FILLED)#画文字框,背景白色
        
        if len(result)>=1:
            print(22222)
            print(orgimg)
            print(result_p)
            
            orgimg=cv2ImgAddText(orgimg,result_p,rect_area[0],int(rect_area[1]-round(1.6*labelSize[0][1])),(0,0,0),21)
            print(5555)
            # orgimg=cv2ImgAddText(orgimg,result_p,rect_area[0]-height_area,rect_area[1]-height_area-10,(0,255,0),height_area)
               
    print(result_str)
    return orgimg







# 视频流接口
@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(video_stream(), media_type="multipart/x-mixed-replace; boundary=frame")    #返回这个类型的东西，运行的是里面的video_stream

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
