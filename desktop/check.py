import psutil # pip install psutil
import os
import requests as re
import time
import bluetooth
from pynput import mouse
import threading
exam = True
class blut(object):
    def __init__(self):
     self.time = 5
     self.name = ['Galaxy A6+','Galaxy S10']
 
    
    def skan(self):
        devices = bluetooth.discover_devices(duration=3, lookup_names=True,
                                            flush_cache=True, lookup_class=False)
        return devices
    def check(self):
        d = self.skan()
        for i in range(len(d)):
            #print(d[i][1])
            if d[i][1] in self.name:
                print('bluetuse устройство:' + str(d[i][1]))
                message('bluetuse detection:'+str(d[i][1]))
                time.sleep(1)
                return [1,'bluetuse detection:',d[i][1]]
        return [0,'bluetuse detection:']
    
class internet(object):
    def __init__(self):
     self.myhost = 'http://127.0.0.1:8000/stream/screen/'
     self.chechost = 'https://yandex.ru/'
    def internet_check(self,url):
     try:
        r = re.get(url)
        if r.status_code==200:
            return True
        else:
            return False
     except:
        return False
        
    def check(self):
        x = self.internet_check(self.myhost)
        y = self.internet_check(self.chechost)
        if x and y:
            print('good internet')
            return [0,'internet control: ','good internet']
        elif y:
            print('no conection server')
            return [1,'internet control: ','no conection server']
        else:
            print('no internet')
            return [2,'internet control: ','no internet']
class program(object):
    def __init__(self):
        self.proc_name = ['obs64.exe']
        self.prog = {}
        self.clean()
    def clean(self):
       for i in range(len(self.proc_name)):
          self.prog[self.proc_name[i]]=0
    def check(self):
        mas = [0, 'start program:']
        send =''
        self.prog = {}
        for proc in psutil.process_iter():
           for i in self.proc_name:
              if proc.name() == i:
                 #
                 self.prog[i]=1
                 #os.system("taskkill /f /im "+i)
              
        for i in self.proc_name:
            try:
             if self.prog[i]==1:
                print ("Process {}  started".format(i))
                mas.append(i)
                mas[0]=1
            except:
                mas[0]=0
        if mas[0] == 1:
            send =''
            send='start program:'
            mas.remove(mas[0])
            mas.remove(mas[0])
            for i in range(len(mas)):
                send=send+'+'+ str(mas[i])
        message(send)
        time.sleep(1)
        return [0]
    
            
class kontroler(object):
    def __init__(self):
        self.internet= internet()
        self.program=program()
        self.bluetuse = blut()
    def send(self,info):
        re.get()
    def start(self):
        i =self.internet.check()
        p = self.program.check()
        b = self.bluetuse.check()
        print(i)
        print(p)
        print(b)
        
        return [i,p,b]
def message(send):
    try:
     re.get('http://127.0.0.1:8000/mes/'+str(send))  
     print(send)
    except:
        print('ошибка отправки')
class main(object):
   def __init__(self):
    
    self.k = kontroler()
   def start(self):
         while onexam():
              #send =''
              rez= self.k.start()
              '''for i in rez:
                  if i[0]:
                      send=send + ' ' + str(i[1]) + str(i[2])
                      i.remove(i[0])
                      i.remove(i[0])
                      i.remove(i[0])
                      for j in range(len(i)):
                          send=send + ' ' + str(i[j])
              if send:
                  message(send)
              time.sleep(10)'''
  
   def run(self):
        t1 = threading.Thread(target=self.start())
        #t2 = threading.Thread(target=self.mouse())
        t1.start()
        #t2.sart()
        t1.join()
        #t2.join()

        
def onexam():
   global exam
   return exam  
def chenge():
   global exam
   exam = not exam          
def on_move(x, y):
    if not onexam():
        # Stop listener
        return False
def on_click(x, y, button, pressed):
    print('{0} at {1}'.format(
        'Pressed' if pressed else 'Released',
        (x, y)))
    message('clik mouse detected')
    if not onexam():
        # Stop listener
        return False
def on_scroll(x, y, dx, dy):
    print('Scrolled {0} at {1}'.format(
        'down' if dy < 0 else 'up',
        (x, y)))
    message('scroll mouse detected')
    if not onexam():
        # Stop listener
        return False
# Collect events until released
m = main()
m.run()





   
 
 

            
