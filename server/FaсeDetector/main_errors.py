
import os
import cv2
import datetime
import time
class errors(object):
    def __init__(self,vid):
        self.mask = 0
        self.error = []
        self.ids = 0
        self.vid = vid
        self.ersave = [0,0,0,0,0,0]
        self.types = ['a','b','c','d','e','f']
        self.num = 0

    def chmask(self,i):
        if i==0:
           self.mask=self.mask&47
        elif i==1:
           self.mask=self.mask&31
        elif i==2:
           self.mask=self.mask&62
        elif i==3:
           self.mask=self.mask&61
        elif i==4:
           self.mask=self.mask&55
        elif i==5:
           self.mask=self.mask&59
           
    def frame(self):
        self.num+=1
        print(len(self.vid.savevideos))
        for i in range(len(self.ersave)):
           
            if self.num - self.ersave[i] > 100:
                self.chmask(i)
            elif self.num - self.ersave[i] ==30 and self.ersave[i] > 0:
                st = {}
                today = datetime.datetime.today()
                st['type'] = i
                st['date'] = today.strftime("%Y-%m-%d")
                st['time']=today.strftime("%H.%M.%S")
                st['ids'] = self.ids
                self.error.append(st)
                self.vid.save(time.time(), self.ids)
                self.ids+=1


    def twoperson(self):
         self.ersave[0]=self.num
         self.mask = self.mask | 16
         
         
    def undetection(self):
        self.ersave[5]=self.num
        self.mask = self.mask | 4
     
    def prog(self):
        self.ersave[1]=self.num
        self.mask = self.mask | 32
       
    def devices(self):
        self.ersave[2]=self.num
        self.mask =self.mask | 1
       
    def eye(self):
        self.ersave[3]=self.num
        self.mask=self.mask | 2
       
    def mouse(self):
        self.ersave[4]=self.num
        self.mask=self.mask | 8
       
        
class img(object):
    def __init__(self):
       self.img ={}
       mas = os.listdir('ico')
       for i in range(len(mas)):
           self.img[str(mas[i])]=cv2.imread('ico/'+str(mas[i]),cv2.IMREAD_UNCHANGED)
    def imgs(self,mask):
        rez = []
        if mask & 1 ==1:
            rez.append(self.img['devaice.png'])
        else:
             rez.append(self.img['devaiceNo.png'])
        
        if mask & 2 ==2:
            rez.append(self.img['eye.png'])
        else:
             rez.append(self.img['eyeNo.png'])
        
        if mask & 4 ==4:
            rez.append(self.img['fase.png'])
        else:
             rez.append(self.img['faseNo.png'])
             
        
        if mask & 8 ==8:
            rez.append(self.img['mouse.png'])
        else:
             rez.append(self.img['mouseNo.png'])
             
             
        if mask & 16 ==16:
            rez.append(self.img['pipl.png'])
        else:
             rez.append(self.img['piplNo.png'])
        
        
        if mask & 32 ==32:
            rez.append(self.img['program.png'])
        else:
             rez.append(self.img['programNo.png'])
        
        return rez
             
class writevid(object):
    def __init__(self):
        self.savevideos = []
        self.frames =[]
        self.plays =0
        self.i = 0
        self.len = 0

    def putframe(self, frame):
        if len(self.frames) > 125:
            self.frames.remove(self.frames[0])
            self. frames.append(frame)
        else:
            self.frames.append(frame)

    def save(self, time, name):
          st = {}
          st['name'] = name
          st['time'] = time
          st['vid'] = self.frames.copy()
          self.savevideos.append(st)
          del(st)

    def isplay(self, name):
        print('////////========///////////')
        print(name)
        #print(len(self.savevideos[1]['vid']))
        for i in range(len(self.savevideos)):
            if self.savevideos[i]['name'] == name:
               
                self.plays == i
                self.i = 0
                self.len = len(self.savevideos[int(name)]['vid'])
                return True
            else:
                self.i = 0
                self.len = 0
        return False

    def canplay(self):
         if self.i < self.len-2:
             
             return True
         else:
             return False

    def getfarme(self,name):
        
        if self.canplay():
          self.i +=1
          return self.savevideos[name]['vid'][self.i] , True
        else:
            print('falsegg')
            return [] , False
         
        
        