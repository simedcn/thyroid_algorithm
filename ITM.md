Interpretable Thyroid model
data分为：
json文件（结节点集、边缘级别、钙化情况、形态、回声等）

模型分为：
```
A 整体信息（结节）
B 局部信息（形态、边界、回声）对应：
  1、边界模型：input(结节环)
  2、形态模型：input(结节中间是黑色的图像 仅保留图像信息)
  3、回声模型：两部分（结节图像区域与正常组织区域） 提取特征
上述三个模型用fine tune的形式训练参数 然后 选择模型的提取特征部分
D 结节位置
  性质
  纵横比
  将上述模型直接作为单维向量
```

数据组织情况：
```
tuber:
  train：
    class 0
    class 1
  test：
    class 0
    class 1

结节环:
  train：
    class 0
    class 1
  test：
    class 0
    class 1
    
 形态图像:
  train：
    class 0
    class 1
  test：
    class 0
    class 1

 正常组织:
  train：
    class 0
    class 1
  test：
    class 0
    class 1
 
all json files
```

命名规则：
```
完整报告图像：
ID.jpg(ID 包括了医院编号与病人编号)
超声图像：
ID_image.jpg
结节图像：
ID_tuber.jpg
正常组织图像：
ID_normal.jpg
形态图像:
ID_shape.jpg
edge图像：
ID_margin.jpg
json文件：ID.json
#cy 囊实性 position 结节位置 ratio纵横比
```
