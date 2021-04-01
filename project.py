import numpy as np
import winsound
import imutils
import scipy.spatial.distance 
import time
import cv2
from time import sleep
from tkinter import *
import tkinter.messagebox


root=Tk()
root.geometry('520x590')
frame = Frame(root, relief=RIDGE, borderwidth=2)
frame.pack(fill=BOTH,expand=1)
root.title('Social Distance Detection')
frame.config(background='light blue')
label = Label(frame, text="Social Distance Detection",bg='light blue',font=('Times 35 bold'))
label.pack(side=TOP)
filename = PhotoImage(file="F:\project\demo2.png")
background_label = Label(frame,image=filename)
background_label.pack(side=TOP)



def help():
   help(cv2)

def Contri():
   tkinter.messagebox.showinfo("Contributors","\n1. Sharaf\n2. Naveen \n3. Himaz \n4. Rilah\n5. Anuk")


def anotherWin():
   tkinter.messagebox.showinfo("About",'Social Distance detection version v1.0\n Made Using\n-OpenCV\n-YOLO\n-Tkinter\n In Python 3')
                                    
   

menu = Menu(root)
root.config(menu=menu)

subm1 = Menu(menu)
menu.add_cascade(label="Tools",menu=subm1)
subm1.add_command(label="Open CV Docs",command=help)

subm2 = Menu(menu)
menu.add_cascade(label="About",menu=subm2)
subm2.add_command(label="Social Distance Detection",command=anotherWin)
subm2.add_command(label="Contributors",command=Contri)



def Exit():
    exit()

  
def cam():
    vs =cv2.VideoCapture(0)
    while True:
        ret, frame=vs.read()
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF ==ord('q'):
            break
    vs.release()
    cv2.destroyAllWindows()

def rec():
    vs =cv2.VideoCapture(0)
    fourcc=cv2.VideoWriter_fourcc(*'XVID') 
    op=cv2.VideoWriter('Sample1.avi',fourcc,11.0,(640,480))
    while True:
        ret,frame=vs.read()
        cv2.imshow('frame',frame)
        op.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    op.release()
    vs.release()
    cv2.destroyAllWindows()   

def det():
   
    net = cv2.dnn.readNetFromDarknet("yolov3.cfg","yolov3.weights")
    classes = []
    with open("coco.names","r") as f:
        classes = [line.strip() for line in f.readlines()]
    layername = net.getLayerNames()
    outputlayers = [layername[i[0]-1] for i in net.getUnconnectedOutLayers()]

    vs = cv2.VideoCapture(0)

    while True:
        ret, frame = vs.read()
        frame = imutils.resize(frame, width=590)
        height, width, channels = frame.shape

        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416,416), (0,0,0), swapRB=True, crop=False)

        net.setInput(blob)
        outs = net.forward(outputlayers)

        boxes = []
        confidences = []
        centers = []
        red = set()


        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and class_id==classes.index("person"):
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    centers.append((center_x,center_y))
                
                    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

                    if len(indexes) >= 2 :
                        for i in indexes.flatten:
                            distance = scipy.spatial.distance.cdist(centers, centers, metric="euclidean")

                            for r in range(0, distance.shape[0]):
                                for j in range(r+1, distance.shape[1]):
                                    if distance[r,j] < 40 :
                                        red.add(r)
                                        red.add(j)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        for i in range(len(boxes)):
            if i in indexes:
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 2)        
                if i in red:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0,0,255), 2)
                    for i in range (0, 3) :
                        winsound.PlaySound('socialdistance.wav', winsound.SND_FILENAME)
                        sleep(5)


        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            vs.release()
            cv2.destroyAllWindows()

def detandrec():

    net = cv2.dnn.readNetFromDarknet("yolov3.cfg","yolov3.weights")
    classes = []
    with open("coco.names","r") as f:
       classes = [line.strip() for line in f.readlines()]
    layername = net.getLayerNames()
    outputlayers = [layername[i[0]-1] for i in net.getUnconnectedOutLayers()]
    vs = cv2.VideoCapture(0)
    fourcc=cv2.VideoWriter_fourcc(*'XVID') 
    op=cv2.VideoWriter('Sample2.avi',fourcc,9.0,(640,480))

    while True:
        ret, frame = vs.read()
        frame = imutils.resize(frame, width=700)
        height, width, channels = frame.shape

        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416,416), (0,0,0), swapRB=True, crop=False)

        net.setInput(blob)
        outs = net.forward(outputlayers)

        boxes = []
        confidences = []
        centers = []
        red = set()


        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and class_id==classes.index("person"):
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x,y,w,h])
                    confidences.append(float(confidence))
                    centers.append((center_x,center_y))
                
                    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

                    if len(indexes) >= 2 :
                        for i in indexes.flatten:
                            distance = scipy.spatial.distance.cdist(centers, centers, metric="euclidean")

                            for r in range(0, distance.shape[0]):
                                for j in range(r+1, distance.shape[1]):
                                    if distance[r,j] < 40 :
                                        red.add(r)
                                        red.add(j)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        for i in range(len(boxes)):
            if i in indexes:
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                        
                if i in red:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0,0,255), 2)
                    for i in range (0, 3) :
                        winsound.PlaySound('socialdistance.wav', winsound.SND_FILENAME)
                        sleep(5)


        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            op.release()
            vs.release()
            cv2.destroyAllWindows()

   
but1=Button(frame,padx=5,pady=5,width=40,bg='white',fg='black',relief=GROOVE,command=cam,text='Open Camera',font=('helvetica 15 bold'))
but1.place(x=5,y=104)

but2=Button(frame,padx=5,pady=5,width=40,bg='white',fg='black',relief=GROOVE,command=rec,text='Open Camera & Record',font=('helvetica 15 bold'))
but2.place(x=5,y=176)

but3=Button(frame,padx=5,pady=5,width=40,bg='white',fg='black',relief=GROOVE,command=det,text='Open Camera & Detect',font=('helvetica 15 bold'))
but3.place(x=5,y=250)

but4=Button(frame,padx=5,pady=5,width=40,bg='white',fg='black',relief=GROOVE,command=detandrec,text='Detect & Record',font=('helvetica 15 bold'))
but4.place(x=5,y=322)

but5=Button(frame,padx=5,pady=5,width=5,bg='white',fg='black',relief=GROOVE,text='EXIT',command=Exit,font=('helvetica 15 bold'))
but5.place(x=210,y=478)


root.mainloop()

