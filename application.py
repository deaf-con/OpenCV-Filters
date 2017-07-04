
# coding: utf-8

# In[1]:


from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2


# In[2]:


def nothing(x):
    pass

def inside(r, q):
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh

def draw_detections(img, rects, thickness = 1):
    for x, y, w, h in rects:
        # so we slightly shrink the rectangles to get a nicer output.
        pad_w, pad_h = int(0.15*w), int(0.05*h)
        cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), thickness)


# In[3]:


newx = 0
newy = 0
newy_c = 0
secs = 0

objects = []
bloby = []

n_channels = 5


# In[4]:


cap = cv2.VideoCapture("/home/sunspot/Desktop/see/jack/input videos/_demo1.mp4")
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) 

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter("output.avi", fourcc, 20.0,(width, height))

font = cv2.FONT_HERSHEY_SIMPLEX


# In[5]:


cv2.namedWindow('Options')
cv2.moveWindow('Options',0,0)
cv2.namedWindow('Original')
cv2.moveWindow('Original',0,480)


# In[6]:


cv2.createTrackbar('Select Video to Display','Options',0,4,nothing)
cv2.createTrackbar('Record Video','Options',0,1,nothing)
cv2.createTrackbar('Select Channel to Record','Options',5,n_channels,nothing)
cv2.createTrackbar('Wait Time','Options',1,20,nothing)
cv2.createTrackbar('Time Delay','Options',18,60,nothing)


# In[7]:


for n in range(1,n_channels+1):
    cv2.namedWindow('Channel '+str(n))
    cv2.moveWindow('Channel '+str(n),480*n,0)
    cv2.createTrackbar('Grey Scale','Channel '+str(n),0,1,nothing)
    cv2.createTrackbar('Invert Colors','Channel '+str(n),0,1,nothing)
    cv2.createTrackbar('Gauss','Channel '+str(n),0,10,nothing)
    cv2.createTrackbar('Threshold','Channel '+str(n),0,255,nothing)
    cv2.createTrackbar('Erode','Channel '+str(n),0,50,nothing)
    cv2.createTrackbar('Dilate','Channel '+str(n),0,50,nothing)
    cv2.createTrackbar('Opening','Channel '+str(n),0,50,nothing)
    cv2.createTrackbar('Closing','Channel '+str(n),0,50,nothing)
    cv2.createTrackbar('Canny Edge Detector','Channel '+str(n),0,200,nothing)
    cv2.createTrackbar('Canny Parameter 2','Channel '+str(n),0,200,nothing)
    cv2.createTrackbar('Blob Detector','Channel '+str(n),0,1,nothing)
    cv2.createTrackbar('Feature Detector','Channel '+str(n),0,1,nothing)
    cv2.createTrackbar('Calculate Frequency','Channel '+str(n),0,1,nothing)


# In[8]:


while(1):
    #file_select = cv2.getTrackbarPos('Select Video File', 'Options')
    #if file_select == 0:
    #    filestr = '_demo0'
    #elif file_select == 1:
    #    filestr = '_demo1'
    #elif file_select == 2:
    #    filestr = '_demo2'
    #elif file_select == 3:
    #    filestr = 'test2'
    #elif file_ps/1.0select == 4:
    #    filestr = 'pumpjack rider pro'
        
#----------------------------------------------------------------------------frame
    #cap = cv2.VideoCapture("/home/sunspot/Desktop/see/jack/input videos/"+filestr+".mp4")

    
    ret,frame = cap.read()
    frame_count = cap.get(cv2.CAP_PROP_POS_FRAMES) 
    wait_time = cv2.getTrackbarPos('Wait Time','Options')
#----------------------------------------------------------------------------    
    
    if ret:
        og = frame
        cv2.imshow('Original',og)
        
        for n in range(1,n_channels+1):

            #----------------------------------------------------------------
            #Grey Scale------------------------------------------------------
            bw = cv2.getTrackbarPos('Grey Scale', 'Channel '+str(n))
            if bw > 0:
                if len(frame.shape) == 3:
                    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                elif len(frame.shape) == 2:
                    frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)

            #----------------------------------------------------------------
            #Invert Colors---------------------------------------------------
            iv = cv2.getTrackbarPos('Invert Colors', 'Channel '+str(n))
            if iv > 0:
                frame = cv2.bitwise_not(frame)

            #----------------------------------------------------------------
            #Gaussian blur---------------------------------------------------
            gs = cv2.getTrackbarPos('Gauss', 'Channel '+str(n))
            if gs > 0:
                frame = cv2.GaussianBlur(frame,(2*gs-1,2*gs-1),0)

            #----------------------------------------------------------------
            #Threshold-------------------------------------------------------
            th = cv2.getTrackbarPos('Threshold', 'Channel '+str(n)) 
            if th > 0:
                ret, frame = cv2.threshold(frame, th, 255, cv2.THRESH_BINARY)

            #----------------------------------------------------------------
            #Erode-----------------------------------------------------------
            er = cv2.getTrackbarPos('Erode', 'Channel '+str(n))
            if er > 0:
                kernel = np.ones((er,er),np.uint8)
                frame = cv2.erode(frame, kernel, iterations = 1)

            #---------------------------------------------------------------- 
            #Dilate----------------------------------------------------------
            dl = cv2.getTrackbarPos('Dilate', 'Channel '+str(n))
            if dl > 0:
                kernel = np.ones((dl,dl),np.uint8)
                frame = cv2.dilate(frame, kernel, iterations = 1)

            #----------------------------------------------------------------
            #Opening---------------------------------------------------------
            op = cv2.getTrackbarPos('Opening', 'Channel '+str(n))
            if op > 0:
                kernel = np.ones((op,op),np.uint8)
                frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)

            #----------------------------------------------------------------
            #Closing---------------------------------------------------------
            cl = cv2.getTrackbarPos('Closing', 'Channel '+str(n))
            if cl > 0:
                kernel = np.ones((cl,cl),np.uint8)
                frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)

            #----------------------------------------------------------------
            #Edge Detection--------------------------------------------------
            eg = cv2.getTrackbarPos('Canny Edge Detector', 'Channel '+str(n)) 
            if eg > 0:
                edgepar2 = cv2.getTrackbarPos('Canny Parameter 2', 'Channel '+str(n))
                frame = cv2.Canny(frame, eg, edgepar2)

            #----------------------------------------------------------------
            #Blob Detection--------------------------------------------------
            bl = cv2.getTrackbarPos('Blob Detector', 'Channel '+str(n))
            if bl > 0:
                params = cv2.SimpleBlobDetector_Params()
                detector = cv2.SimpleBlobDetector_create(params)
                keypoints = detector.detect(frame)

                frame = cv2.drawKeypoints(frame,keypoints, np.array([]),(255,255,255))#, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

                if len(keypoints) == 1:
                    #--------------------------------------------------------
                    #Calc Frequency------------------------------------------
                    x = int(keypoints[0].pt[0])
                    y = int(keypoints[0].pt[1])

                    freq = cv2.getTrackbarPos('Calculate Frequency','Channel '+str(n))
                    if freq > 0:
                        delay = cv2.getTrackbarPos('Time Delay','Options')+1
                        bloby.append(y)

                        if len(bloby) > delay:
                            bloby.pop(0)

                        if bloby[0] < bloby[len(bloby)-1]:
                            #frame = cv2.rectangle(frame, (x+30, y+50), (x-30, y-50), (0, 0, 255), thickness = 2)
                            frame = cv2.rectangle(og, (x+30, y+60), (x-30, y-60), (0, 0, 255), thickness = 2)
                            
                        elif bloby[0] > bloby[len(bloby)-1]:
                            #frame = cv2.rectangle(frame, (x+30, y+50), (x-30, y-50), (0, 255, 0), thickness = 2)
                            frame = cv2.rectangle(og, (x+30, y+60), (x-30, y-60), (0, 255, 0), thickness = 2)
                        else:
                            #frame = cv2.rectangle(frame, (x+30, y+50), (x-30, y-50), (255, 255, 255), thickness = 2)
                            frame = cv2.rectangle(og, (x+30, y+60), (x-30, y-60), (255, 255, 255), thickness = 2)

            #----------------------------------------------------------------
            #Feature Detection-----------------------------------------------
            ft = cv2.getTrackbarPos('Feature Detector', 'Channel '+str(n)) 
            if ft > 0:
                detector = cv2.FastFeatureDetector_create(threshold=25)
                keypoints = detector.detect(frame, None)

                frame = cv2.drawKeypoints(frame,keypoints, np.array([]),(255,255,255))
            
            #----------------------------------------------------------------
            #Title Card------------------------------------------------------
            sel = cv2.getTrackbarPos('Select Video to Display','Options')
            if sel > 0:
                cv2.putText(frame,"Tracking Pump Jacks",(100,100),font,0.7,(0,0,0),3)
                cv2.putText(frame,"Tracking Pump Jacks",(100,100),font,0.7,(255,255,255),2)
            if sel > 1:
                cv2.putText(frame,"with Computer Vision",(103,150),font,0.7,(0,0,0),3)
                cv2.putText(frame,"with Computer Vision",(103,150),font,0.7,(255,255,255),2)
            if sel > 2:
                cv2.putText(frame,"Test 1",(170,200),font,0.7,(0,0,0),3)
                cv2.putText(frame,"Test 1",(170,200),font,0.7,(255,255,255),2)
                
            #----------------------------------------------------------------
            #Output Video----------------------------------------------------
            ot = cv2.getTrackbarPos('Record Video','Options')
            if (ot > 0):
                chn = cv2.getTrackbarPos('Select Channel to Record','Options')
                if n == chn:
                    out.write(frame)
     
            #----------------------------------------------------------------
            #----------------------------------------------------------------
     
            
            cv2.imshow('Channel '+str(n),frame)
            

    else:
        print('resetting')
        cap.set(cv2.CAP_PROP_POS_FRAMES,0)

    if cv2.waitKey(wait_time) & 0xFF == ord('q'):
        break 

cap.release()
out.release()
cv2.destroyAllWindows()


# In[9]:


print cv2.CAP_PROP_FPS


# In[ ]:




