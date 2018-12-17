from numpy import *
import cv2
import imageio

lena= cv2.imread('./multimedia/lena.jpg')
nobel = cv2.imread('./multimedia/nobel.jpg')
nobel1 = cv2.imread('./multimedia/nobel.jpg')

radius = range(0,300,10)
for i in radius:
    for x in range(0,407):
        for y in range(0,407):
            if(((x-204)*(x-204)+(y-204)*(y-204))<i*i):
                nobel1[x,y,0]=lena[x,y,0]
                nobel1[x,y,1]=lena[x,y,1]
                nobel1[x,y,2]=lena[x,y,2]
    cv2.imwrite("./frames/res"+str(i)+".jpg",nobel1)

frames=[]
for i in radius:
    frames.append(imageio.imread("./frames/res"+str(i)+".jpg"))
imageio.mimsave("change.gif",frames,"GIF",duration=0.1)
