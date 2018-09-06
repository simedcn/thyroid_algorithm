Algorithms for Thyroid Project
ITM.md 中包含数据组织情况以及数据具体的命名规则
Support_code中含有实现部分数据准备工作的文件：
00100710043.json 为示例标注信息
convert_dcm_jpg.py
	将医学dicom格式转换为jpg文件格式
data_augmentation.py
	通过翻转变换、随机修剪、添加噪声等方法进行数据增强
patch.py
	将图片切成小的patch进行训练


