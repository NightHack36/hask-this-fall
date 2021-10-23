# MirITeam
## Hask this fall 2.0

&nbsp;

The service is a streaming server with a desktop client for analysis
preparation of the examinee by monitoring the examination.
With the help of computer vision, video analytics and analysis of executable
computer processes are monitored for user actions
during the exam.
The service monitors student parameters
(head position, gaze direction, pupil position),
computer processes to check the operation of additional resources,
interception of clicks and scrolling.
Стек: Python, Django, OpenCV, OpenVINO, MobileNet, Scipy, psutil, pynput.

&nbsp;



#### Directories:

`. / desktop` contains:
 - `blut.py` - analysis of bluetooth connections
 - `check.py` - analysis of the Internet connection and current processes
 - `tk.py` - analysis of mouse events


 `. / server` contains:
 - `/ streamingproject` - django app that broadcasts a video stream
 - `/ videoProccesing` - image processor

&nbsp;

 
#### Install

OpenVINO:
https://docs.openvinotoolkit.org/latest/index.html
&nbsp;

&nbsp;

#### Project connection:
```


pip install -r requirements.txt
```

&nbsp;

#### Launch of the project:

Активация **OpenVINO**:
 - Linux: 
 ```
 source /opt/intel/openvino/bin/setupvars.sh
```
 - Windows: 
```
cd C:\Program Files (x86)\IntelSWTools\openvino\bin\
setupvars.bat
```


Server launch and video streaming along the way `http://127.0.0.1:8000/stream/screen/` 
```
python manage.py runserver
```


&nbsp;

&nbsp;
