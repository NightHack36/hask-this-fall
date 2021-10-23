
import cv2

class startend(object):
    def __init__(self,drow):
        self.isit = True
        self.isStart = False
        self.isEnd = False
        self.drow = drow
        self.img = {}
        self.img['3'] = cv2.imread('ico/3.png',cv2.IMREAD_UNCHANGED)
        self.img['2'] = cv2.imread('ico/2.png',cv2.IMREAD_UNCHANGED)
        self.img['1'] = cv2.imread('ico/1.png',cv2.IMREAD_UNCHANGED)
        self.img['gol'] = cv2.imread('ico/gol_en.png',cv2.IMREAD_UNCHANGED)
        self.img['end'] = cv2.imread('ico/end_en.png',cv2.IMREAD_UNCHANGED)
        self.step =0
        self.time = 0

    def start(self,frame,obl,box):
        self.time +=1
        if self.step == 0:
           frame = self.drow.drowimg(frame,self.img['gol'],115,20)
           frame = self.drow.drowzone(frame,obl[0],obl[1],obl[2],obl[3])
        elif self.step ==1:
           frame = self.drow.drowimg(frame,self.img['3'],350,100)
        elif self.step ==2:
            frame = self.drow.drowimg(frame,self.img['2'],350,100)
        else:
            frame = self.drow.drowimg(frame,self.img['1'],350,100)
        
        if len(box)==1:
         if obl[0]<box[0]['kord'][0] and obl[1] < box[0]['kord'][1] and obl[2] > box[0]['kord'][2] and obl[3] > box[0]['kord'][3] and self.step ==0:
            self.step =1
            self.time = 0
         elif self.step ==1 and int(self.time / 5) > 1:
            self.step =2
            self.time = 0
         elif self.step ==2 and int(self.time / 5) > 1:
            self.step =3
            self.time = 0
         else:
            if self.step ==3 and int(self.time / 5) > 1:
                self.isStart = True
                
                self.time = 0
        return frame

    def end(self,frame,er):
        types ={}
        types['0']  = 0    
        types['1']  = 0 
        types['2']  = 0 
        types['3']  = 0 
        types['4']  = 0 
        types['5']  = 0 
        for i in range(len(er.error)):
            types[str(er.error[i]['type'])]+=1
       
        self.time +=1
        frame = self.drow.drowzone(frame,2,2,3,3)
        frame = self.drow.drowimg(frame,self.img['end'],100,100)
        cv2.putText(frame, 'twoperson: ' + str(types['0']), (200, 350),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, 'programs: ' + str(types['1']), (200, 380),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, 'devises: ' + str(types['2']), (200, 410),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, 'eyes: ' + str(types['3']), (200, 440),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, 'mouse: ' + str(types['4']), (200, 470),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, 'undetection: ' + str(types['5']), (200, 500),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if int(self.time / 60) > 1:
            self.isit = False
        return frame
        
        