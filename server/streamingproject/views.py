from django.shortcuts import render
from django.http import StreamingHttpResponse, HttpResponseServerError , HttpResponse
from django.core import serializers
from django.views.decorators import gzip
import cv2
import os
import sys
import json
from videoProccesing.openvino_processing import ImageOpenVINOPreprocessing
from FaсeDetector.faseclass import Fase
from FaсeDetector.main_errors import errors , img ,  writevid
from FaсeDetector.faseclass import drow
from FaсeDetector.start_end import startend

ImgProcessOpenVINO = ImageOpenVINOPreprocessing()

FaceDetection = Fase()

Img = img()
Vid = writevid()
Errors = errors(Vid)
Drow = drow()
StEn = startend(Drow)
text = ''
blut = ''
click = 0
programs = ''
glaza = 0
rez ='good stydent'
gl = ''
videoplay =''
v = False

class VideoCamera(object):

    def __init__(self, path):
        self.video = cv2.VideoCapture(0)
        self.score = 0
        self.chet = 0
        camx, camy = [(1920, 1080), (1280, 720), (800, 600), (480, 480)][1]  # Set camera resolution [1]=1280,720
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, camx)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, camy)

        self.ROOT_DIR = os.path.abspath("")
        print(self.ROOT_DIR)

        sys.path.append(self.ROOT_DIR)

        self.codec = cv2.VideoWriter_fourcc(*'DIVX')
        print("\nVizualize:")
        

    def __del__(self):
        self.video.read()

    # Обработчик фрейма
    def get_frame(self):
        ret, frame = self.video.read()
        if ret:
            frame1 = frame.copy()
            rez = frame
            if StEn.isit:
             if StEn.isStart == False and StEn.isEnd == False:
                mas = FaceDetection.main(frame)
                mas[0] = FaceDetection.vizual(mas[1],mas[0])
                mas[0] = StEn.start(mas[0],[350,150,670,500],mas[1])
                rez = mas[0]
             elif StEn.isStart == True and StEn.isEnd == False:
              
               Errors.frame()
               
               mas = FaceDetection.main(frame)
               if len(mas[1])>1:
                   Errors.twoperson()
               try:
                if mas[1][0]['name']=='unknown':
                    if self.chet == 50:
                      Errors.undetection()
                    else:
                        self.chet+=1
                else:
                    self.chet=0
               except:
                   True
               mas[0], isEyes = ImgProcessOpenVINO.main(mas[0])
               if isEyes == True:
                   Errors.eye()

               mas[0] = FaceDetection.vizual(mas[1],mas[0])
               
               mas[0] = Drow.drowimgs(mas[0],Img.imgs(Errors.mask))
               rez = mas[0]
               Vid.putframe(rez)

             elif StEn.isEnd == True and StEn.isStart == False:
                 mas = FaceDetection.main(frame)
                 mas[0]=StEn.end(mas[0],Errors)
                 rez = mas[0]
                 print(len(Vid.savevideos))
                 '''if retv(True):
                 rez = Vid.getfarme()
                 if rez==False:
                     rez==frame
                     retv(False)
                 else:
                     rez = frame'''
            else:
                 if retv(True):
                    rez , flag = Vid.getfarme(retname())
                   
                    if flag == False:
                      rez == frame
                      retv(False)
                 else:
                     rez = frame
            rez =cv2.resize(rez,(1920, 1080))
            try:
             jpeg = cv2.imencode('.jpg', rez)[1].tostring()
            except:
                jpeg = cv2.imencode('.jpg', frame1)[1].tostring()
            return jpeg

def retv(bol):
    global v
    if bol:
     return v
    else:
        v = False

def retname():
    global videoplay
    return videoplay

# Обработчик камеры
def gen(camera):
    while True:
        frame = camera.get_frame()
        yield(b'--frame\r\n'
        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def indexscreen(request):
    global text
    try:
        template = "index.html"
        
        return render(request,template)
    except HttpResponseServerError:
        print("aborted")

def startpage(request):
    global text
    try:
        template = "hello.html"
        
        return render(request,template)
    except HttpResponseServerError:
        print("aborted")
        
def profilepage(request):
    global text
    try:
        template = "profile__student.html"
        
        return render(request,template)
    except HttpResponseServerError:
        print("aborted")

def start(requests):
    global StEn
    global Errors
    global Vid
    if StEn.isit == False:
        StEn.isit=True
        StEn.isEnd=False
        StEn.isStart=False
        StEn.step = 0
        Vid = writevid()
        Errors = errors(Vid)

def playvid(request):
    global videoplay
    global v
    global Vid
    rec = str(request).split('/')
    print(rec)
    videoplay = int(rec[2][0])
    v = Vid.isplay(videoplay)
    #print(v)

def end(request):
    global StEn
    StEn.isEnd = True
    StEn.isStart =False
    #print('yes')

def changeline(request):
    global text
    global blut 
    global click 
    global programs 
    global glaza 
    global rez 
    global gl
    print('-------------------------------------------')
    #print(request)
    rec = str(request).split('/')
    mes = rec[2].replace('%20',' ')
    rec = mes
    text = mes
    if 'press' in rec:
        click = click+1
        if StEn.isStart == True: 
         Errors.mouse()
    elif 'start program:' in rec:
        prog = rec.split(':')
        prog=prog[1]
        programs=prog
        if StEn.isStart == True:
         Errors.prog()
    elif 'bluetuse detection:' in rec:
        prog = rec.split(':')
        prog=prog[1]
        blut=prog
        if StEn.isStart == True:
         Errors.devices()
    elif 'eye:' in rec:
        r = rec.split(':')
        rr =r[1]
        if gl!=rr:
         glaza=glaza+1
        
        gl = rr
        
    else:
        print('not info')
    if glaza > 5:
        rez='bad student'
    print(glaza)
    print('-------------------------------------------')
    
def getline(request):
    '''global text
    global blut 
    global click 
    global programs 
    global glaza 
    global rez 
    response_data = {}
    response_data['text'] = text
    response_data['blut'] = blut
    response_data['click'] = click
    response_data['programs'] = programs
    response_data['glaza'] = glaza
    response_data['rez'] = rez'''
    global Errors
    response_data = {}
    response_data['info'] = Errors.error

    return HttpResponse(json.dumps(response_data), content_type="application/json")
@gzip.gzip_page
def dynamic_stream(request, num=0,stream_path="0"):
   
    try:
        return StreamingHttpResponse(gen(VideoCamera(stream_path)), content_type="multipart/x-mixed-replace;boundary=frame")
    except:
        print("aborted");