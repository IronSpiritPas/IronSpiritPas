import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, np.ndarray)):  #判断是否OpenCV图片类型
        print(3333)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    print(img)
    draw = ImageDraw.Draw(img)
    # print(draw)
    print(44444)
    print(textSize)
    print(type(text))
    font_path = "C:\\Windows\\Fonts"
    print(f"Loading font from: {font_path}")
    
    try:
        fontText = ImageFont.truetype("C:\\Windows\\Fonts\\simkai.ttf", textSize, encoding="utf-8")

    except OSError as e:
        print(f"Error loading font: {e}")
        return img  # 或者使用默认字体
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    # fontText = ImageFont.truetype(
    #     "C:\\Windows\\fonts\\platech.ttf", textSize, encoding="utf-8")
    print(55555555555)

    draw.text((left, top), text, textColor, font=fontText)#Embedded color supported only in RGB and RGBA modes
    # cv2.putText(img, text, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    print(cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR))
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    

if __name__ == '__main__':
    imgPath = "C:\\Users\\U\\Desktop\\images\\1.jpg"
    img = cv2.imread(imgPath)
    # img = orgimg
    saveImg = cv2ImgAddText(img, 'no', 50, 100, (255, 0, 0), 50)
    
    cv2.imshow('display',saveImg)
    # cv2.imwrite('save.jpg',saveImg)
    cv2.waitKey()