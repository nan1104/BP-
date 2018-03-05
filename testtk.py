# encoding=utf-8
from tkinter import *
from tkinter import ttk
from PIL import *
from PIL import ImageGrab
import BP
import test_loadimage as tl
from tkinter import scrolledtext
from tkinter import Menu
from tkinter import Spinbox
from tkinter import messagebox as mBox

k=10

canvas_width = 500
canvas_height = 500


def callback():
    #屏幕size=1536*864
    im0 = ImageGrab.grab((90,80,590,330))
    #im0.show()
    im0.save('showimage/show.jpg')
    tl.show()
    W=BP.grab('weights2.txt')
    b = BP.grab('biases2.txt')
    M = [256, 25, 10]
    data=tl.show_loadtxt(16,'showdata')
    y=BP.show(M , W , b , data )
    #print(y)
    max=0
    for i in range(len(y)):
        if y[i]>max:
            max=y[i]
            k=i
    print(k)
    a.set(k)
    master.update()

def paint( event ):
   python_green = "#476042"
   x1, y1 = ( event.x - 1 ), ( event.y - 1 )
   x2, y2 = ( event.x + 1 ), ( event.y + 1 )
   w.create_oval( x1, y1, x2, y2, fill = python_green )

master = Tk()
master.title( "Painting using Ovals" )
a=StringVar()
#fm1 = Frame(master, bg='red', width=500, height=500)
w = Canvas(master,bg='white',width=canvas_width,height=canvas_height)
label=ttk.Label(master,text='The Number Is:').place(x=5,y=3)
label=ttk.Label(master,textvariable=a).place(x=100,y=3)
la=ttk.Button(master, text="start",width=10,command=callback).pack()
ttk.Button(master, text="clear",width=10,command=(lambda x=ALL:w.delete(x))).pack(side='bottom')
#one = Label(master,tex,compound='left',width = 30,height = 2).pack()
#expand = YES, fill = BOTH
w.pack()
w.bind( "<B1-Motion>", paint )

message = Label( master, text = "Press and Drag the mouse to draw" )
message.pack( side = BOTTOM )

mainloop()