#! /usr/bin/env python
import Tkinter
import tkMessageBox
from Tkinter import *
import cv2 
import numpy as np
import openpyxl
from os import listdir
from os.path import isfile, join

class Window(Frame):

    def __init__(self, master=None):
        Frame.__init__(self, master)                 
        self.master = master
        self.init_window()

    def init_window(self):

        self.master.title("GUI")

        self.pack(fill=BOTH, expand=1)

        Blur = Button(self, text="Blur Images", command = self.blurimages)
        Colourdet = Button(self, text = "Colour Detection", command = self.colourdetection)

        Blur.place(x=50, y=100)
        Colourdet.place(x = 200, y = 100)

    def blurimages(self):
		tkMessageBox.showinfo( "Blur Images", "Please wait while blurring")

		tshirt_cascade = cv2.CascadeClassifier('cascade.xml')
		tshirt12_cascade = cv2.CascadeClassifier('cascade12.xml')

		mypath = 'Images'
		onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]

		for n in range(0, len(onlyfiles)):
		  filename = onlyfiles[n]
		  img = cv2.imread(join(mypath,filename))
		  shape = img.shape
		  img = cv2.resize(img,(400,400))
		  height = img.shape[0] 
		  width = img.shape[1]

		  mask = np.zeros(img.shape[:2],np.uint8)

		  bgdModel = np.zeros((1,65),np.float64)
		  fgdModel = np.zeros((1,65),np.float64)

		  tshirts = tshirt_cascade.detectMultiScale(img,minSize = (150,150))
		  if (tshirts == ()):
		      tshirts = tshirt12_cascade.detectMultiScale(img,minSize = (200,200))
		  if (tshirts ==()):
		      tshirts = [[50,50,250,250]]

		  rect = (tshirts[0][0]-10,tshirts[0][1]+10,tshirts[0][0]+tshirts[0][2]+10,tshirts[0][1]+tshirts[0][3]+10)

		  cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
		   
		  mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
		  grab = img*mask2[:,:,np.newaxis]
		  grab = np.array(grab,dtype = 'uint8') 

		  zero = np.zeros(400*400*3).reshape(400,400,3)
		  if (grab==zero).all():
		    continue

		  blur = img - grab
		  
		  mask = blur*255
		  mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
		  out = cv2.distanceTransform(mask, cv2.DIST_L2, 3)

		  def kernel(s,i,j):
		    sa = s
		    sc = s
		    a = i-s/2
		    b = i+s/2
		    c = j-s/2
		    d = j+s/2
		    if a<0:
		      a = 0
		      sa = len(range(a,b))
		    if b>400:
		      b=400
		      sa = len(range(a,b))
		    if c<0:
		      c=0
		      sc = len(range(c,d))
		    if d>400:
		      d=400
		      sc = len(range(c,d))
		    kernel  = np.ones(sa*sc*3).reshape(sa,sc,3)
		    kernel = np.array(kernel,dtype = "float32")
		    
		    return (img[a:b,c:d] + kernel/(s*s)).mean(axis = (0,1))

		  chan = np.zeros((height*width)*3).reshape(height,width,3)
		  chan = np.array(chan,dtype = "uint8")

		  for y in range(0,height):
		    for x in range(0,width):

		      if out[y,x] == 0.0:
		        chan[y,x] = img[y,x]
		        continue

		      if out[y,x]<10:
		        chan[y,x] = img[y,x]
		        continue

		      if out[y,x]<20:
		        chan[y,x] = kernel(2,y,x)
		        continue

		      if out[y,x]<30:
		        chan[y,x] = kernel(4,y,x)
		        continue

		      if out[y,x]<40:
		        chan[y,x] = kernel(6,y,x)
		        continue

		      if out[y,x]<50:
		        chan[y,x] = kernel(8,y,x)
		        continue

		      if out[y,x]<100:
		        chan[y,x] = kernel(10,y,x)
		        continue

		      if out[y,x]<120:
		        chan[y,x] = kernel(12,y,x)
		        continue

		      else:
		        chan[y,x] = kernel(14,y,x)

		  if shape[0]>3000 or shape[1]>3000:
		    chan = cv2.resize(chan,(shape[1]/6,shape[0]/6))

		  elif shape[0]>2000 or shape[1]>2000:
		    chan = cv2.resize(chan,(shape[1]/4,shape[0]/4))

		  elif shape[0]>1000 or shape[1]>1000:
		    chan = cv2.resize(chan,(shape[1]/2,shape[0]/2))
		  else:
		    chan = cv2.resize(chan,(shape[1],shape[0]))

		  cv2.imwrite(join('Blurred Images',filename),chan)

		tkMessageBox.showinfo( "Blur Images", "Images saved in folder 'Blurred Images'")

    def colourdetection(self):
		tkMessageBox.showinfo("Colour Detection","Please wait for a while")

		wb  = openpyxl.Workbook()
		sheet = wb.active
		sheet.title = 'Colour Detection'
		sheet['A' + str(1)] = 'Filename'
		sheet['B' + str(1)] = 'Colour'
		row = 2

		def colour(image,x,y,w,h):
		    X = np.array([x,y,w,h],dtype = 'int')
		    xa,xy,p = image.shape
		    crop = np.zeros((w*h)*3).reshape(w,h,3)
		    crop = np.array(crop,dtype = "uint8")
		    for i in range(x,x+w):
		        for j in range(y,y+h):
		            crop[i-x,j-y] = image[i,j]
		            image[i,j] = (0,0,0)

		    back = np.zeros(x*y*3).reshape(x,y,3)
		    back = np.array(back,dtype = "uint8")
		    for i in range(0,x):
		        for j in range(0,y):
		            back[i,j] = image[i,j]
		    B = np.zeros(12)
		    B = np.array(B,dtype = "int")

		    COL = np.zeros(12)
		    COL = np.array(COL,dtype = 'int')

		    hsvim = cv2.cvtColor(back,cv2.COLOR_BGR2HSV)
		    for i in range(0,x):
		        for j in range(0,y):
		            hsv = hsvim[i,j]
		            code,c = colourcode(hsv)
		            if c == 'black':
		                COL[0] = COL[0] +1
		            elif c == 'blue':
		                COL[1] = COL[1] +1
		            elif c == 'red':
		                COL[2] = COL[2] +1
		            elif c == 'green':
		                COL[3] = COL[3] +1
		            elif c == 'white':
		                COL[4] = COL[4] +1
		            elif c == 'yellow':
		                COL[5] = COL[5] +1
		            elif c == 'pink':
		                COL[6] = COL[6] +1
		            elif c == 'purple':
		                COL[7] = COL[7] +1
		            elif c == 'orange':
		                COL[8] = COL[8] +1
		            elif c == 'Cream':
		                COL[9] = COL[9] +1
		            elif c == 'Grey':
		                COL[10] = COL[10] +1
		            else:
		                COL[11] = COL[11] +1

		    highestback = 0

		    for i in range(0,12):
		        if (COL[i]>=COL[highestback]):
		            highestback = i

		    COL = np.zeros(12)
		    COL = np.array(COL,dtype = 'int')
		    COLor = np.array(COL,dtype = 'str')
		    
		    COLor[0] = '#000000'
		    COLor[1] = '#0000FF'
		    COLor[2] = '#FF0000'
		    COLor[3] = '#00FF00'
		    COLor[4] = '#FFFFFF'
		    COLor[5] = '#FFFF00'
		    COLor[6] = '#FF33FF'
		    COLor[7] = '#800080'
		    COLor[8] = '#FF8000'
		    COLor[9] = '#FFFFCC'
		    COLor[10] = '#808080'
		    COLor[11] = '#A0522D'
		    hsvimg = cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
		    for i in range(x,x+w):
		        for j in range(y,y+h):
		            hsv = hsvimg[i-int(x),j-int(y)]
		            code,c = colourcode(hsv)
		            if c == 'black':
		                COL[0] = COL[0] +1
		            elif c == 'blue':
		                COL[1] = COL[1] +1
		            elif c == 'red':
		                COL[2] = COL[2] +1
		            elif c == 'green':
		                COL[3] = COL[3] +1
		            elif c == 'white':
		                COL[4] = COL[4] +1
		            elif c == 'yellow':
		                COL[5] = COL[5] +1
		            elif c == 'pink':
		                COL[6] = COL[6] +1
		            elif c == 'purple':
		                COL[7] = COL[7] +1
		            elif c == 'orange':
		                COL[8] = COL[8] +1
		            elif c == 'Cream':
		                COL[9] = COL[9] +1
		            elif c == 'Grey':
		                COL[10] = COL[10] +1
		            else:
		                COL[11] = COL[11] +1

		    highest = 0

		    for i in range(0,12):
		        if (COL[i]>=COL[highest]):
		            highest = i

		    if highest==0:
		        highest2 = 1
		    else:
		        highest2 = 0

		    if highest2==highestback:
		        highest2 = highest2 +1

		    for i in range(0,12):    
		        if (COL[i]>=COL[highest2] and COL[i]<COL[highest] and i!=highestback):
		            highest2 = i

		    if COL[highest2]<(COL[highest]/2 - COL[highest]/12):
		        return COLor[highest],'NULL'
		    else:
		        return COLor[highest],COLor[highest2]
		    
		def colourcode(hsv):    
		    h = hsv[0]
		    s = hsv[1]
		    v = hsv[2]

		    black = '#000000' # 0 0 0 v<10
		    blue = '#0000FF'# 255 0 0 h-162:268 (h-264:270,s>70,v>70)
		    red = '#FF0000' #0 0 255 h-0:40, 360:344
		    green = '#00FF00' # 0 255 0 h-74:162
		    white = '#FFFFFF' # 255 255 255 v >87 s < 11, v-83:87 s<7
		    yellow = '#FFFF00' # 0 255 255 h-40:74
		    pink = '#FF33FF' #255 51 255 294:344
		    purple = '#800080' #128 0 128 h-270:294
		    orange = '#FF8000' #255 128 0 h-16:40
		    cream = '#FFFFCC' #255 255 204  
		    gray = '#808080' # 128 128 128  s - 0-17 v - 20-70, s - 10 v - 70-83, s - 7:12 v - 83:85, s -7:10 v - 85-87 
		    brown = '#A0522D' # 160 82 45
		    if((s<43 and v>.13*255 and v<.70*255) or (s<.10*255 and v>=.70*255 and v<.75*255)): #or (s>=7*2.55 and s<12*2.55 and v>=83*2.55 and v<85*2.55) or (s>=7*2.55 and s<10*2.55 and v>=85*2.55 and v<87*2.55)):
		        colorcode = gray
		        cname = 'Grey'
		    elif(s<=40*2.55 and v>87*2.55 and h>30/2 and h<75/2):
		        colorcode = cream
		        cname = 'Cream'
		    elif((v>=87*2.55 and v<90 and s<11*2.55) or (v>=90 and v<95 and s<15*2.55) or (v>=95 and s<20*2.55) or (v>=75*2.55 and v<87*2.55 and s<7*2.55)):
		        colorcode = white
		        cname = 'white'
		    elif(v>=13*2.55 and v<55*2.55 and h>=356/2) or (v>=13*2.55 and v<55*2.55 and h>=12/2 and h<46/2) or (v>=13*2.55 and v<45*2.55 and h<12/2):
		        colorcode = brown
		        cname = 'brown'
		    elif((h>=294/2 and h<320/2 and v>=80*2.55) or (h>=320/2 and h<350/2 and v>=70*2.55) or (h>=350/2 and h<354/2 and s<=85*2.55 and v>=70*2.55) or (h>354/2 and v>=70*2.55 and s<72*2.55) or (h<10/2 and s<=65*2.55 and v>70*2.55)):
		        colorcode = pink
		        cname = 'pink'
		    elif((h>=270/2 and h<294/2 and v>=13*2.55) or (h>=266/2 and h<270/2 and s<=60*2.55 and v>=13*2.55) or (h>=294/2 and h<330/2 and v<80*2.55) or (h>=330/2 and h<344/2 and v<70*2.55 and s<75*2.55)):
		        colorcode = purple
		        cname = 'purple'
		    elif (h>=162/2 and h<270/2 and v>13*2.55):
		        colorcode = blue
		        cname = 'blue'
		    elif((h<6/2 and v>=13*2.55) or (h>=344/2 and h<360/2 and v>=13*2.55) or (h>=330/2 and h<344/2 and v<70*2.55 and s>75*2.55)):
		        colorcode = red
		        cname = 'red'
		    elif(h>=74/2 and h<162/2 and v>=13*2.55) or (h<=72/2 and h>=64/2 and v<=90) or (h>=58 and h<64 and v<=70):
		        colorcode = green
		        cname = 'green'
		    elif(h>=36/2 and h<74/2 and v>=13*2.55):
		        colorcode = yellow
		        cname = 'yellow'
		    elif(h>=6/2 and h<36/2 and v>=13*2.55):
		        colorcode = orange
		        cname = 'orange'
		    else:
		        colorcode = black
		        cname = 'black'
		    return colorcode,cname

		tshirt_cascade = cv2.CascadeClassifier('cascade.xml')
		tshirt12_cascade = cv2.CascadeClassifier('cascade12.xml')
		mypath = 'Images'

		onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]

		for n in range(0, len(onlyfiles)):

		    filename = onlyfiles[n]
		    img = cv2.imread(join(mypath,filename))
		    img = cv2.resize(img,(400,400))
		    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		    tshirts = tshirt_cascade.detectMultiScale(gray,minSize = (150,150))
		    if (tshirts == ()):
		        tshirts = tshirt12_cascade.detectMultiScale(gray,minSize = (200,200))
		    if (tshirts ==()):
		        tshirts = [[100,100,200,200]]

		    for (x,y,w,h) in tshirts:
		        cv2.rectangle(img,(x,y),(x+w,y+h),255,1)
		        
		        hexcolour,hexcolour2 = colour(img,y,x,h,w)
		        sheet['A' + str(row)] = filename
		        sheet['B' + str(row)] = hexcolour
		        sheet['C' + str(row)] = hexcolour2
		        row = row + 1
		        break

		wb.save('colours.xlsx')

		tkMessageBox.showinfo( "Colour Detection","Colour detection done, hex colours saved in 'colours.xlsx'")

root = Tk()
root.geometry("400x300")

app = Window(root)
root.mainloop()  