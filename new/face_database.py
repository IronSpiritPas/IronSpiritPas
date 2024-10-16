# 引入sqlalchemy依赖
from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import select

 
# 申明基类对象
Base = declarative_base()
 
 
# 定义face表实体对象
class Face(Base):
    #定义表名
    __tablename__ = 'face'
    id = Column(Integer, primary_key=True)
    name = Column(String(255))
    age = Column(Integer)

    def __repr__(self):
        return "Face(id:{}, name:{}, age:{})".format(self.id, self.name,self.age)

# 定义category表实体对象
class Category(Base):
    #定义表名
    __tablename__ = 'category'
    id = Column(Integer, primary_key=True)
    name = Column(String(255))

    def __repr__(self):
        return "Category(id:{}, name:{})".format(self.id, self.name)
 
 
class SqliteSqlalchemy(object):
    def __init__(self):
        # 创建Sqlite连接引擎
        engine = create_engine('sqlite:///./sqlalchemy.db', echo=True)
        # 创建表
        Base.metadata.create_all(engine, checkfirst=True)
        # 创建Sqlite的session连接对象
        self.session = sessionmaker(bind=engine)()
 
 
if __name__ == '__main__':
    # 初始化Sqlite数据库连接，获取数据库session连接
    session = SqliteSqlalchemy().session
 
    # 新增多条用户信息数据
    users_to_add = [
        Face(name='obama', age=59),
        Face(name='biden', age=82),
        Category(name='obama'),
        Category(name='biden'),
    ]
# category_dict = {
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

    session.add_all(users_to_add)
    session.commit()
 
    # 初始化Sqlite数据库连接，获取数据库session连接
    session = SqliteSqlalchemy().session

    # 查询所有用户信息
    query = select(Face)
    result = session.execute(query)
    # 打印所有用户信息
    for face in result:
        print(face)

    query = select(Category)
    result = session.execute(query)
    for category in result:
        print(category)

    # 关闭数据库session连接
    session.close()