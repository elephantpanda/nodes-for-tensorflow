import random
import os
import math
import sys
import numpy as np
import Tkinter as tk
import tensorflow as tf
import PIL.Image
import PIL.ImageDraw
import PIL.ImageTk
#import PIL.ImageFont
#import aggdraw
import json
import inspect
import about

#font = PIL.ImageFont.truetype("Arialbd.ttf",30)

#Create list of tensorflow functions
tfFunctions = []
for name in dir(tf):
    obj = getattr(tf, name)
    if inspect.isfunction(obj) and not name[0].istitle():       
        tfFunctions.append(name)

tfNNFunctions = []
for name in dir(tf.nn):
    obj = getattr(tf.nn, name)
    if inspect.isfunction(obj) and not name[0].istitle():       
        tfNNFunctions.append(name)

window = tk.Tk()
WIDTH = window.winfo_screenwidth()
HEIGHT = window.winfo_screenheight()

sess = tf.Session()

window.title("Visual Programming with Tensorflow")
fullscreen=not True
if fullscreen:
    WIDTH = window.winfo_screenwidth()
    HEIGHT = window.winfo_screenheight()
    window.overrideredirect(1)
    #window.attributes("-fullscreen", True)
window.geometry(str(WIDTH)+"x"+str(HEIGHT)+"+0+0")
window.configure(background="gray")


img = PIL.Image.new("RGB", (WIDTH, HEIGHT), "black")
#dc2 = aggdraw.Draw(img)
dc = PIL.ImageDraw.Draw(img)


#pen = aggdraw.Pen("white", 1)
#brush= aggdraw.Brush("red")

def updateImage():
    label.image = PIL.ImageTk.PhotoImage(img) 
    label.configure(image = label.image) 

class Input:
    node=0
    def __init__(self, node, output):
        self.node = node
        self.output = output
    value=0

def drawBezier( dc, points ):
    (x1,y1,x2,y2,x3,y3,x4,y4) = points
    steps = 50
    X=x1
    Y=y2
    for t in range(1,steps+1):
        a = t*1.0/steps
        b = (1-a)
        x = b*b*b*x1 + 3*b*b*a*x2 + 3*b*a*a*x3 + a*a*a*x4
        y = b*b*b*y1 + 3*b*b*a*y2 + 3*b*a*a*y3 + a*a*a*y4
        dc.line((X,Y,x,y),fill=(255,255,255))
        X=x
        Y=y


class Node:
    x=50
    y=50
    height = 50
    width = 100
    titleHeight=13
    drag = False
    name = "Image"
    inputs = [0,0]
    inputNames = ["a","b","c","d","e","f","g","h"]
    outputs= []
    value = 0
    type=""
    showvalue = True
    issetup =  True

    def __init__(self):
        self.inputs = [0,0]
        self.outputs = [Input(self,0)]

    def setup(self):
        self.issetup = True
        for n in self.inputs:
            if n!=0:
                if n.node!=0 and not n.node.issetup:
                    n.node.setup()

    def draw(self,dc):
        self.drawBackground(dc)
        if(self.showvalue):
            self.showValue(dc)
        self.drawForeground(dc)

    def calc(self):
        z=0

    def drawBackground(self,dc):
        dc.rectangle((self.x-1,self.y,self.x+self.width,self.y+self.titleHeight),outline=(0,128,255),fill=(0,0,128))
        title=self.name
        if self.value!=0 and self.type!="list":
            title += " "+str(self.value.get_shape())
        dc.text((self.x+5,self.y+1),title)
        dc.rectangle((self.x-1,self.y+self.titleHeight,self.x+self.width,self.y+self.height),
        outline=(0,128,255),fill=(0,0,255))

    circCenter=[]
    circInputCenter=[]

    def drawForeground(self,dc):
        circleSize=6
        spacing = 16
        self.circInputCenter=[]
        for n in range(0,len(self.inputs)):
            y=self.y+self.titleHeight + spacing * n + 5
            x=self.x+5
            self.circInputCenter.append( (x+circleSize/2,y+circleSize/2)  )
            dc.ellipse((x,y ,x+circleSize,y+circleSize),fill=(255,255,0))
            if n<len(self.inputNames):
                dc.text((x+10,y-3),self.inputNames[n],fill=(255,255,255))

            #dc2.ellipse((x,y ,x+circleSize,y+circleSize),pen,brush)
            if self.inputs[n]!=0:
                node2 = self.inputs[n].node;
                X = node2.x + node2.width - 5 - circleSize
                Y = node2.y + node2.titleHeight + 5 + spacing * self.inputs[n].output
                #dc.line((x+circleSize/2, y+circleSize/2 , X +circleSize/2,Y + circleSize/2 ) ,fill=(255,255,0) )
                away=50
                drawBezier(dc,
                    (x+circleSize/2, y+circleSize/2,
                    x+circleSize/2-away, y+circleSize/2,
                    X +circleSize/2+away,Y + circleSize/2,
                    X +circleSize/2,Y + circleSize/2)
                    )
        #dc2.flush() 

        self.circCenter=[]
        for n in range(0,len(self.outputs)):
            y=self.y+self.titleHeight + spacing * n + 5
            x=self.x+self.width-5-circleSize
            self.circCenter.append((x+circleSize/2,y+circleSize/2)  )
            dc.ellipse((x,y ,x+circleSize,y+circleSize),fill=(255,255,0))
        

    def inside(self,(px,py)):
        return (px>self.x and py>self.y and px<self.x+self.width and py<self.y+self.height)

    def insideOutput(self,(px,py)):
        global dragStartPos
        for n in range(0,len(self.circCenter)):
            (cx,cy) = self.circCenter[n]
            if (px-cx)*(px-cx)+(py-cy)*(py-cy) < 8*8:
                dragStartPos = (cx,cy)
                return n
        return -1

    def insideInput(self,(px,py)):
        for n in range(0,len(self.circInputCenter)):
            (cx,cy) = self.circInputCenter[n]
            if (px-cx)*(px-cx)+(py-cy)*(py-cy) < 8*8:
                dragStartPos = (cx,cy)
                return n
        return -1

    def showValue(self,dc):
        if self.value==0:
            return
        if self.type=="list":
            dc.text((self.x+20,self.y+10+self.titleHeight),str(self.value))
            return
        shape = self.value.get_shape()
        array = sess.run(self.value)
        #self.name = str(shape)
        if len(shape)>=2:
            if(shape[0]+self.titleHeight>self.height):
                 self.height=int(shape[0])+self.titleHeight
            if(shape[1]>self.width):
                 self.width=int(shape[1])

        if len(shape)==2:
            if shape[0]<=4 and shape[1]<=4:
                for y in range(0, shape[0]):
                    dc.text((self.x+20,self.y+10+self.titleHeight+y*10),str(array[y]))
            else:
                if array.dtype == np.complex128 or array.dtype==np.complex64:
                    data = np.reshape(array,(shape[0],shape[1],1))
                    data = np.concatenate( (
                        (np.angle(data)*255/np.pi/2).astype(np.uint8),
                        #np.full((shape[0],shape[1],1),255).astype(np.uint8),
                        (np.clip(np.abs(data),0,1) *255).astype(np.uint8),
                        (np.clip(1.0/np.abs(data),0,1) *255*1).astype(np.uint8)
                        ) ,axis=2 )
                    image = PIL.Image.fromarray(data, mode="HSV")
                    img.paste(image,(self.x,self.y+self.titleHeight))
                else:
                    data = np.reshape(array*255,(shape[0],shape[1])).astype(np.uint8)
                    image = PIL.Image.fromarray(data, mode="L")
                    img.paste(image,(self.x,self.y+self.titleHeight))
        elif len(shape)==3:
            if shape[2]==3: #RGB
                data = np.reshape(array*255,(shape[0],shape[1],shape[2])).astype(np.uint8)
                image = PIL.Image.fromarray(data, mode="RGB")
                img.paste(image,(self.x,self.y+self.titleHeight))
            elif shape[2]==2: #?
                iscomplex=1 #(doesnt do anything)
        elif len(shape)<=1:
            dc.text((self.x+20,self.y+10+self.titleHeight),str(array))

dragStartPos = (0,0)


class ConstantNode(Node):
    value = 0
    height = 50
    width = 100
    type="constant"
    name="Constant"
    inputs = []
    val = 0
    def __init__(self):
        Node.__init__(self)
        self.inputs=[]
    def setup(self):
        Node.setup(self)
        self.value = tf.constant(self.val)
        self.outputs[0].value=self.value

class ListNode(Node):
    value = 0
    height = 50
    width = 100
    type="list"
    name="List"
    inputs = []
    val = 0
    def __init__(self):
        Node.__init__(self)
        self.inputs=[]
    def setup(self):
        Node.setup(self)
        self.value = self.val
        self.outputs[0].value=self.value

class VariableNode(Node):
    value = 0
    height = 50
    width = 100
    val = 0
    name="Variable"
    def __init__(self):
        Node.__init__(self)
        self.inputs=[]
    def setup(self):
        Node.setup(self)
        self.value = tf.Variable(self.val)
        sess.run(tf.initialize_variables([self.value]))
        self.outputs[0].value=self.value

class RandomNode(Node):
    value = 0
    height = 50
    width = 100
    name="Random"
    def __init__(self):
        Node.__init__(self)
        self.inputs=[]
    def calc(self):
        self.value = tf.constant(random.uniform(-1.0,1.0).astype(np.float32))
        self.outputs[0].value= self.value

class Random3x3(Node):
    value = 0
    height = 50
    width = 100
    name="Random 3x3 Matrix"
    op=0
    def __init__(self):
        Node.__init__(self)
        self.inputs=[]
    def setup(self):
        Node.setup(self)
        self.value = tf.Variable(np.random.rand(3,3).astype(np.float32))
        self.outputs[0].value= self.value
        self.op = tf.assign(self.value,tf.cast(tf.random_uniform([3,3],-0.5,0.5),tf.float32))
    def calc(self):
        sess.run(self.op)


class Random3(Node):
    value = 0
    height = 50
    width = 100
    name="Random 3Vector"
    op = 0
    def __init__(self):
        Node.__init__(self)
        self.inputs=[]
    def setup(self):
        Node.setup(self)
        self.value = tf.Variable(np.random.rand(3).astype(np.float32)-0.5 )
        self.outputs[0].value = self.value 
        self.op = tf.assign(self.value,tf.cast(tf.random_uniform([3],-0.5,0.5),tf.float32))

    def calc(self):
        sess.run(self.op)


class DotNode(Node):
    name="A.B"
    height = 50
    width = 100
    inputs = [0,0]
    showvalue = not False
    def setup(self):
        Node.setup(self)
        if self.inputs[0]!=0 and self.inputs[1]!=0:
            A = self.inputs[0].value
            B = self.inputs[1].value
            shape1 = A.get_shape()
            shape2 = B.get_shape()
            #contract last indices
            if len(shape1)>0 and len(shape2)>0:
                a = len(shape1)-1
                b = len(shape2)-1
                self.value = tf.reduce_sum( A * B, a )
                self.outputs[0].value = self.value

#General node for any tensorflow function
class FunctionNode(Node):
    name="x"
    height = 50
    width = 100
    inputs = [0,0]
    showvalue = not False
    args = []
    func = 0
    def __init__(self,func,args):
        Node.__init__(self)
        self.func=func
        self.args=args
        self.inputs = [0] * len(args)
        self.height = 16 * len(args) + self.titleHeight
        self.name = func.__name__
        for a in range(0,len(self.args)):
            if self.args[a]!=0:
                self.inputs[a] = self.args[a].outputs[0]

    def setup(self):
        Node.setup(self)
        fargs=[]
        for a in range(0,len(self.inputs)):
            if self.inputs[a]==0:
                break
            fargs.append(self.inputs[a].value)
        try:
            self.value = self.func(*fargs)
            self.outputs[0].value = self.value
        except Exception as e:
            infoLabel.configure(text=str(e))

#Matrix multiplication for each element in a grid
class MatMultNode(Node):
    name="AB"
    height = 50
    width = 100
    inputs = [0,0]
    showvalue = not False
    def setup(self):
        Node.setup(self)
        if(self.inputs[0]!=0 and self.inputs[1]!=0):
            a = len(self.inputs[0].value.get_shape())-1
            b = len(self.inputs[1].value.get_shape())-2
            self.value =tf.tensordot( self.inputs[0].value , self.inputs[1].value , [[a],[b]])
            self.outputs[0].value = self.value

class WatchNode(Node):
    name="watch"
    inputs = [0]
    def setup(self):
        Node.setup(self)
        if(self.inputs[0]!=0):
            self.value = self.inputs[0].value
            self.outputs[0] = self.value

ar = np.zeros([256,256])
for x in range(0,256):
    for y in range(0,256):
        ar[y][x] = ((x+y)%256)/255.0

W=128
H=128
ar3 = np.zeros([H,W,3]).astype(np.float32)
for x in range(0,W):
    for y in range(0,H):
        ar3[y][x][0] = x*2.0/W-1
        ar3[y][x][1] = y*2.0/H-1


nodes=[]

def defaultNodes():
    global nodes

    X = ConstantNode()
    X.val = ar3

    M = Random3x3()

    MX = MatMultNode()
    MX.inputs[0]=X.outputs[0]
    MX.inputs[1]=M.outputs[0]
    MX.x=500

    XMX = DotNode()
    XMX.inputs[0]=MX.outputs[0]
    XMX.inputs[1]=X.outputs[0]
    XMX.x=800
    XMX.y=300

    B = Random3()

    BX = DotNode()
    BX.inputs[0]=X.outputs[0]
    BX.inputs[1]=B.outputs[0]
    BX.x=800


    XMX_BX = FunctionNode(tf.add,[BX,XMX])
    XMX_BX.x = 1000

    W = WatchNode()
    W.inputs[0]=XMX_BX.outputs[0]

    nodes = [X,M,MX,XMX,B,BX,XMX_BX]

def defaultNodes2():
    global nodes
    img1 = PIL.Image.open("abc.png").convert("RGB") 
    ar2 = np.array(img1) /256.0

    c1 = ConstantNode()
    c1.value = tf.constant(ar3)
    c2 = ConstantNode()
    c2.value = tf.constant(ar3)

    m = Random3x3()

    u = Random3()

    t=MatMultNode()
    t.inputs[0] = c1.outputs[0]
    t.inputs[1] = m.outputs[0]

    a = DotNode()
    a.inputs[0] = t.outputs[0]
    a.inputs[1] = u.outputs[0]

    w=WatchNode()
    w.inputs[0]=a.outputs[0]

    nodes = [c1,m,t,u,a,w]

def setupNodes():
    for n in nodes:
        n.issetup = False
    for n in nodes:
        if not n.issetup:
            n.setup()

    sess.run(tf.global_variables_initializer())

defaultNodes()
setupNodes()

draggingObject = 0
draggingOutput = -1
lastPos = (0,0)


############# make menu bar ##############

menubar = tk.Frame(window)

nodeButtons= [tf.add,tf.multiply,tf.assign]  #tf.reduce_sum,

DERIVE="[astype]"
typeButtons = [DERIVE,  "int32", "int64", "float16", "float32","float64" , "complex64", "complex128" ,"string"]

def buttonPressed(nb):
    args = inspect.getargspec(nb)
    numArgs = len(args[0]) - len(args.defaults)
    print(str(inspect.getargspec(nb)[0]))
    n = FunctionNode(nb,np.zeros(numArgs-0))
    n.inputNames = inspect.getargspec(nb)[0]
    n.setup()
    nodes.append(n)

for nb in nodeButtons:
    button1 = tk.Button(menubar, text=nb.__name__, command = lambda x=nb:buttonPressed(x) )
    button1.pack(side="left")

def listbuttonPressed():
    b= ListNode()
    v = inputVar.get()
    tv = typeVar.get()
    if tv == "string":
        val=v
    else:
        val = eval(v)
        if tv!=DERIVE:
            t = getattr(np,tv)
            val = np.array(val).astype(t)
    b.val=val
    b.setup()
    nodes.append(b)


def cbuttonPressed():
    b = ConstantNode()
    v = inputVar.get()
    tv = typeVar.get()
    if tv == "string":
        val=v
    else:
        val = eval(v)
        if tv!=DERIVE:
            t = getattr(np,tv)
            val = np.array(val).astype(t)
    b.val=val
    b.setup()
    nodes.append(b)

def vbuttonPressed():
    b = VariableNode()
    v = inputVar.get()
    tv = typeVar.get()
    if tv == "string":
        val=v
    else:
        val = eval(v)
        if tv!=DERIVE:
            t = getattr(np,tv)
            val = np.array(val).astype(t)
    b.val=val
    b.setup()
    nodes.append(b)

def abuttonPressed():
    W=128
    H=128
    ar3 = np.zeros([H,W]).astype(np.complex64)
    for x in range(0,W):
        for y in range(0,H):
            ar3[y][x] =  x*2.0/W-1 + (y*2.0/H-1)*1j
    b = ConstantNode()
    b.val = ar3
    b.setup()
    nodes.append(b)

cbutton = tk.Button(menubar, text="constant", command = cbuttonPressed )
cbutton.pack(side="left")   

vbutton = tk.Button(menubar, text="variable", command = vbuttonPressed )
vbutton.pack(side="left")   

listbutton = tk.Button(menubar, text="list", command = listbuttonPressed )
listbutton.pack(side="left") 

argandbutton = tk.Button(menubar, text="argand", command = abuttonPressed )
argandbutton.pack(side="left")  

inputVar = tk.StringVar()
inputVar.set("[1.0, 2.0], [3.0, 4.0]")
input = tk.Entry(menubar, textvariable=inputVar)
input.pack(side="left")

typeVar = tk.StringVar(window)
typeVar.set(typeButtons[0]) # default value

typelist = tk.OptionMenu(menubar, typeVar, *typeButtons )
typelist.pack(side="left")

optionVar = tk.StringVar(window)
optionVar.set(tfFunctions[0]) # default value

def optionChosen(event):
    buttonPressed(getattr(tf,optionVar.get()))

funclist = tk.OptionMenu(menubar, optionVar, *tfFunctions, command=optionChosen)
funclist.pack(side="left")

optionVar2 = tk.StringVar(window)
optionVar2.set("conv2d") # default value

def optionChosen2(event):
    buttonPressed(getattr(tf.nn,optionVar2.get()))

funclist2 = tk.OptionMenu(menubar, optionVar2, *tfNNFunctions, command=optionChosen2)
funclist2.pack(side="left")






def clearbuttonPressed():
    global nodes
    nodes = []
    resetbuttonPressed()

clrbutton = tk.Button(menubar, text="clear", command = clearbuttonPressed )
clrbutton.pack(side="left") 


def resetbuttonPressed():
    global nodes, sess
    sess.close()
    tf.reset_default_graph()
    for n in nodes:
        n.value = 0
        n.outputs[0].value=0
    sess = tf.Session()
    setupNodes()

    
"""
resetbutton = tk.Button(menubar, text="reset", command = resetbuttonPressed )
resetbutton.pack(side="left") 
"""




def infoPressed():
    graph_def = tf.get_default_graph().as_graph_def()
    infoLabel.configure(text="nodes="+str(len(graph_def.node)))
    for node in graph_def.node:
        print(node.name+" : "+node.op)

infobutton = tk.Button(menubar, text="graph info", command = infoPressed )
infobutton.pack(side="left") 

def aboutbuttonPressed():
    about.AboutDialog(window)

aboutbutton = tk.Button(menubar, text="about", command = aboutbuttonPressed )
aboutbutton.pack(side="left") 

menubar.pack(fill="x")

statusbar = tk.Frame(window)
infoLabel =tk.Label(statusbar,text="info")
infoLabel.pack(side="left")

statusbar.pack(fill="x")

label = tk.Label(window, borderwidth=0)
label.pack(fill="x")




def getMousePos():
    return (
        window.winfo_pointerx() - label.winfo_rootx(),
        window.winfo_pointery() - label.winfo_rooty()
    )

def leftMouseUp(event):
    global draggingObject, draggingOutput
    pos = getMousePos()
    if draggingOutput !=-1:
        for b in nodes:
            o =  b.insideInput(pos)
            if o!=-1:
                #print ("inputs " +str(o)+ " of "+str(b)+"to output "+str(draggingOutput)+" of "+str(draggingObject))
                b.inputs[o] = draggingObject.outputs[draggingOutput]
                resetbuttonPressed()
                #b.setup()

    draggingObject = 0
    draggingOutput = -1
    pos = getMousePos()


def leftMouseDown(event):
    global draggingObject, draggingOutput
    draggingObject = 0
    draggingOutput = -1
    pos = getMousePos()
    for b in nodes:
        if b.inside(pos):
            #dc.rectangle((0,0,650,650),fill=(0,255,0))
            draggingObject=b
            #if draggingObject.type=="constant":
            infoLabel.configure(text=str(draggingObject.value))
            #    inputVar.set(draggingObject.val)
        o = b.insideOutput(pos)
        if o!=-1:
            draggingOutput = o
        

    updateImage()
    window.update()

def doStuff():
    global lastPos
    dc.rectangle((0,0,WIDTH,HEIGHT),fill=(0,0,0))
    (x0,y0) = lastPos
    (x,y) = getMousePos()

    if draggingObject !=0 and draggingOutput ==-1:
        #dc.rectangle((0,0,650,650),fill=(0,255,0))
        draggingObject.x += (x-x0)
        draggingObject.y += (y-y0)

    if draggingOutput!=-1:
        (x0,y0) = dragStartPos
        drawBezier(dc,
        (x0,y0,
        x0+50,y0,
        x-50,y,
        x,y)  
        )

    for b in nodes:
        b.calc()
        b.draw(dc)

    lastPos = (x,y)


label.bind("<Button-1>",leftMouseDown)
label.bind("<ButtonRelease-1>",leftMouseUp)

def update():
    doStuff()
    updateImage()
    window.update()
    window.after(1,update)

def outputGraph():
    graph_def = tf.get_default_graph().as_graph_def()
    for node in graph_def.node:
        print(node.name+" : "+node.op)
        for input in node.input:
            print("\t"+input)

update()

window.mainloop()