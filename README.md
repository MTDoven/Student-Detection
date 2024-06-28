# Student Status Detection
An AI algorithm for detecting and logging the status of a student in classrooms.

## Environment
This project can run on Nvidia GPU or CPU without modifying the code.
````
cudatookit==12.1
pytorch==2.3.0 
ultralytics==8.2.45
````
(You can also run it without installing "ultralytics". 
This project has contained the code we used in ultralytics)

## Usage
Simply pass the URL to main.py as a parameter and run it.  
````
python main.py your_rtmp_url_or_file_path
# example-1: python main.py rtmp://172.0.0.1:1935/live/8888
# example-2: python main.py ../test_video.mp4
````

## Solution
Our overall idea is divided into three steps. 
**The first step** is to intercept the faces in the video, 
**the second step** is to make binary classification judgment 
on the intercepted faces, and **the third step** is 
to accumulate the judgment results in time to judge 
the state of the students.

**Dataset:** In the first step of face state detection we used 
the [Wider-Face](http://shuoyang1213.me/WIDERFACE/) dataset for face detection.
And [CAS-PEAL](http://www.jdl.link/peal/home.htm) dataset is used in the second step
of head pose classification.
Dataset structure:
````
dataset
├─ClassifierData
│  ├─train
│  │  ├─DOWN
│  │  └─UP
│  └─val
│      ├─DOWN
│      └─UP
└─DetectionData
    ├─train
    │  ├─images
    │  └─labels
    └─val
        ├─images
        └─labels
````

**Live-Streaming**: First, a server is built using [Nginx](https://nginx.org/), and then the video is encoded 
and encapsulated by ffmpeg and pushed to the server through the RTMP 
network protocol.

**Detection:** In step of object detection, we use [Yolov9c](https://github.com/ultralytics/ultralytics) model for 
single object detection. And we do not distinguish head pose at this time.  
For training detection model, run ``python trainDetection.py``.

**Classification:** We used the backbone of [Yolov8m](https://github.com/ultralytics/ultralytics) for classification. 
During this, we first resized and filled the captured images to 192*192. 
And in this step, all the faces captured in a picture were formed into 
a batch for batch inference.  
For training classification model, run ``python trainClassifier.py``.

**Cluster:** To locate the faces, we cluster the detected faces 
in the first 500 frames and take the cluster center as the face reference
position, and number each reference position as the number of the person. 
When a new frame comes in, we divide the detected faces to the 
corresponding cluster centers. Such an operation ensures ID invariance 
and is robust to face occlusion in a few frames. But the disadvantage 
is that the position of the person in the camera cannot change too much.

**Why not use object tracking?** Using multi-target tracking can often 
ensure the invariance of the target ID, but once the target is lost 
for a long time, such as someone who goes to the restroom and returned 
to the original seat. The ID of the target detection algorithm will change, 
resulting in inconsistent state information.

