import requests as re
from pynput import mouse
stop = False
def message(send):
    try:
     re.get('http://127.0.0.1:8000/mes/'+str(send))  
     print(send)
    except:
        print('ошибка отправки')
    
def on_click(x, y, button, pressed):
    print('{0} at {1}'.format(
        'Pressed' if pressed else 'Released',
        (x, y)))
    if pressed:
     message('press')
    
   
def on_scroll(x, y, dx, dy):
    print('Scrolled {0} at {1}'.format(
        'down' if dy < 0 else 'up',
        (x, y)))
# Collect events until released
with mouse.Listener(
    
    on_click=on_click,
    on_scroll=on_scroll) as listener:
    listener.join()

 
