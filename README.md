# God's Eye - A YOLO (Darknet) + Tensorflow implementation

Uses Python3 + Tensorflow 1.4

## Requirements

- [Tensorflow](https://www.tensorflow.org/)
- [Python 3](https://www.python.org/)
- Numpy
- OpenCV (3.4.X to support RTSP)
- Pillow (pip install Pillow)
- dLib (IMPORTANT: install from Source only)
- face_recognition (pip install face_recognition)

## Demo Video

[![Watch the video]](https://www.youtube.com/watch?v=89ardoKIjKc)


## How to run

- Dowload the pretrained model from [here](https://drive.google.com/file/d/1vW0wEo-1Db4IaVPDgMtMa2_NeyvAc04U/view?usp=sharing) and place in the model directory in the root of this project

- Create a directory POI in the root of this project and place images of the person that you wish to track (accepted formats are *.jpg, *.png)


python inference_video_face.py <args>
  
- --mode (Video / Camera / IP)
- --file_path (Optional : Video File Path. Required if mode = Video only)
- --uname (Optional: Requirded only if mode is IP for streaming from IP camera. Username for RTSP)
- --secret (Optional: Requirded only if mode is IP for streaming from IP camera. Password for RTSP)
- --addr (Optional: Requirded only if mode is IP for streaming from IP camera. ip address of camera



