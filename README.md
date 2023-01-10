# search_from_videos
基于Python的机器学习应用，针对监控视频完成行人轨迹搜索。通过一张目标图像，能自动从大量视频中搜索出包含目标的视频片段，并标记目标。此项目为本人本科毕业设计项目，引用请注明出处。

## 运行环境 
```
Python 3.6.2
TensorFlow-GPU 1.6.0
opencv-python
numpy 1.18.1
keras 2.2.0
scikit-learn
pillow
```
额外依赖项下载（由于Github上传文件大小限制，我所使用的模型文件上传到百度云，读者也可根据后续教程自己获得）：
```
文件名：yolo.h5
目标文件夹：search_from_videos\edg_code\model_data\
链接：https://pan.baidu.com/s/1_oCYDz3Gpcn-WEr6qcqtsA 
提取码：yimi

文件名：yolov3.weights
目标文件夹：search_from_videos\edg_code\
链接：https://pan.baidu.com/s/1mNMwqp_R2-0G586hG9nmSA 
提取码：zkvu 
```

## 适用平台
笔者在Windows 10上基于Visual Stdio Code开发，但并不代表此项目存在平台限制

## 项目算法简述
预处理过程(edg_code)，图见edg_algorithm_structure.png
```
1. 基于三帧差分法，结合阈值法，对数据量庞大的监控视频文件进行预处理，去除其中的无意义部分（无行人出现），分割视频得到大量的视频片段。
2. 对每个视频片段基于YOLO算法，进行行人识别 [此处算法以及模型来源详见引用]
3. 结合YOLO和Deep Sort进行行人轨迹追踪 [此处算法以及模型来源详见引用]
4. 基于Caffe算法对行人人脸进行识别并评分，缓存评分较高的人脸（存在最大数量限制）
5. 基于EigenFace/LBPHFace, 对前面缓存的人脸建立模型文件
6. 将行人轨迹缓存结果与人脸模型文件关联，即预处理结果
```
搜索过程(client)，图见client_algorithm_structure.png
```
1. 对输入图片基于caffe算法截取人脸部分
2. 基于EigenFace/LBPHFace, 将目标人脸与预处理得到的人脸模型比对，得到置信度
3. 通过对置信度排序，并截取适当比例的结果，作为搜索结果输出
```

## 运行说明
#### code for edg（执行过程耗时）
下面的main.py统一指代为 search_from_videos/edg_code/main.py
```
# 将待处理的单个/多个视频文件（支持mp4）放入main.py中row_path指定的目录中

python main.py      # 根据环境自动修正设置

# 程序将自动在main.py中video_path和save_path指定路径生成处理参数
```

#### code for client（运行过程快速）
下面的main.py统一指代为 search_from_videos/client/main.py
```
# 在main.py中image_path指定待搜索的目标人物的照片
# main.py中的pre_dict和result_dict指向为edg_code生成的对应目录video_path和save_path

python main.py      # 根据环境自动修正设置

# 程序会自动在main.py中search_result_path指定路径生成搜索结果
```

## 修改方向
本项目仅为原型机，尚未提供可视化调用接口。可以考虑采用B/S架构，将其升级为真实可用的saas应用

## 引用
```
YOLO算法: 
Redmon J , Divvala S , Girshick R , et al. You Only Look Once: Unified, Real Time Object Detection[C].The IEEE Conference on Computer Vision and Pattern Recognition (CVPR).2016 ,01.779-788

行人轨迹追踪：https://github.com/Qidian213/deep_sort_yolov3
```
