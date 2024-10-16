fastapi.py下载hf上的模型
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from transformers.generation import GenerationConfig

# # Note: The default behavior now has injection attack prevention off.
# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B-Chat", trust_remote_code=True)

# # use bf16
# # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()
# # use fp16
# # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="auto", trust_remote_code=True, fp16=True).eval()
# # use cpu only
# # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="cpu", trust_remote_code=True).eval()
# # use auto mode, automatically select precision based on the device.
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="auto", trust_remote_code=True).eval()

# # Specify hyperparameters for generation. But if you use transformers>=4.32.0, there is no need to do this.
# # model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-7B-Chat", trust_remote_code=True) # 可指定不同的生成长度、top_p等相关超参

# # 第一轮对话 1st dialogue turn
# response, history = model.chat(tokenizer, "你好", history=None)
# print(response)
# # 你好！很高兴为你提供帮助。

# # 第二轮对话 2nd dialogue turn
# response, history = model.chat(tokenizer, "给我讲一个年轻人奋斗创业最终取得成功的故事。", history=history)
# print(response)
# # 这是一个关于一个年轻人奋斗创业最终取得成功的故事。
# # 故事的主人公叫李明，他来自一个普通的家庭，父母都是普通的工人。从小，李明就立下了一个目标：要成为一名成功的企业家。
# # 为了实现这个目标，李明勤奋学习，考上了大学。在大学期间，他积极参加各种创业比赛，获得了不少奖项。他还利用课余时间去实习，积累了宝贵的经验。
# # 毕业后，李明决定开始自己的创业之路。他开始寻找投资机会，但多次都被拒绝了。然而，他并没有放弃。他继续努力，不断改进自己的创业计划，并寻找新的投资机会。
# # 最终，李明成功地获得了一笔投资，开始了自己的创业之路。他成立了一家科技公司，专注于开发新型软件。在他的领导下，公司迅速发展起来，成为了一家成功的科技企业。
# # 李明的成功并不是偶然的。他勤奋、坚韧、勇于冒险，不断学习和改进自己。他的成功也证明了，只要努力奋斗，任何人都有可能取得成功。

# # 第三轮对话 3rd dialogue turn
# response, history = model.chat(tokenizer, "给这个故事起一个标题", history=history)
# print(response)
# # 《奋斗创业：一个年轻人的成功之路》

##任务
将gradio模型改成fastapi x

改写fastapi（app.py），人脸识别部分
并且还是实时的检测

##问题
C:\Users\U\Desktop\new\face_recognition_api\app.py代码的导入包的问题  x
upload_file.html上传文件，提示找不到文件，app.py要接收文件
app_start.py是yolov10-mian的原版前端界面,可以用

test_cemera.py文件，包含两种category的内容列表。这种列表的问题就是暂时无法显示图像，但是可以显示label
app_start.py改好了，可以用 
as解决video无法上传，视频实时处理的问题，还需要用fastapi

赶快弄fastapi，main1.py
main1copy3 和maincopy4都可以用fastapi识别
maincopy5可以fastapi实时检测人脸,还需要加入其他的检测方法，命令是uvicorn main1:app --host 127.0.0.1 --port 8000，网址是127.0.0.1：8000/video_feed

2.3.1copy是完整的代码
试试2.3.1能不能修改参数img、frame，2.4模块集成的地方
# 摄像头实时人脸识别
def video_stream():
2.4.py修改好了，字体打印是空白
2.4.py完美解决，因为参数是frame，处理后的帧是annonted_frame


2024.10.15
找出两个图片的不同picture_compare.py，x
找两个视频picture_compare copy.py
检测一下视频
要看得懂卷积神经网络，还得去学习pytorch等。自己想做一下医学图像检测，就是先转为3D模型


