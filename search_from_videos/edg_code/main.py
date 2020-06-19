import os
from yolo import YOLO
from frames_subtraction import FSB
from processor import Processor
from face_model import Face_Model_Creator

def main():
    # 此处如果报错可改为绝对地址
    row_path = "search_from_videos/videos"
    video_path = "search_from_videos/pre_videos"
    save_path = "search_from_videos/result"

    # 使用三帧差分法去除视频中多余的空白
    for row_video in os.listdir(row_path):
        fsb = FSB(video_path=os.path.join(row_path, row_video), max_leisure=30, save_fps=15, save_dict=video_path)
        # fsb = FSB(video_path=None, max_leisure=30, save_fps=15, save_dict=video_path)
        fsb.run()
        print(row_video,"finished")

    # 生成追踪数据库
    video_list = os.listdir(video_path)
    for video in video_list:
        process = Processor(YOLO(), video_path=os.path.join(video_path, video), max_face_sample_count=30, caffe_confidence=0.6, save_path=save_path, show_process=True)

        process.run()
        print(video, "finished")

    # 建立人脸训练模型
    model_creator = Face_Model_Creator(dict=save_path)
    model_creator.run()

if __name__ == '__main__':
    main()
    print("all work finished...")