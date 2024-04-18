from ultralytics import YOLO
import cv2
import winsound #für Audioausgabe
import os #für Audioausgabe via MP3-Dateien
import time #für Sleep während Audioausgabe
import numpy as np
import math
import pandas as pd

dct = {}

cap = cv2.VideoCapture(0)
# Load custom trained YOLOv8 model
model = YOLO("balls/model-balls.pt")
#runs/detect/train3/weights/best.pt
#face/model-face.pt
#balls/model-balls.pt
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

show_boxes = True
a=0
i=0
j=0
B1_x=0
B1_y=0
B2_x=0
B2_y=0
M_x=0
M_y=0
M_length=0
S_x=0
S_y=0
l=1
marker_radius=30 # Radius der markierten Punkte
marker_distance=30 # Textabstand zu markierten Punkten
marker_thickness=2 # Schriftdicke der markierten Punkte
list_balls=[]
index=1
xlist_lines=[]
ylist_lines=[]

"""
#Beginn der manuellen Definition der Bahnbreiten bei unterschiedlichen Bahnbreiten (getestet)
n=int(input("Anzahl der Bahnen:"))
pool_width_mm=int(input("Poolbeckenbreite in Millimeter?"))
n_width_list_px=[]
if n>1:
    n_width=int(input("Alle Bahnen gleicher Abstand? 0=NEIN; 1=JA: "))
    if n_width==0:
        for index in range(1,n+1):
            n_width=int(input("Abstand von Bahn"+str(index)+" in Millimeter: "))
            factor=n_width*(l/pool_width_mm)
            n_width_px=n_width*factor
            n_width_list_px.append(n_width_px) #Liste mit Bahnbreiten in Pixeln, zur Berechnung der Punkte auf dem Beckenrand
    else:
        n_width=pool_width_mm/n
        factor=n_width*(l/pool_width_mm)
        n_width_px=n_width*factor
        for index in range(1,n+1):
            n_width_list_px.append(n_width_px) #Liste mit Bahnbreiten in Pixeln, zur Berechnung der Punkte auf dem Beckenrand        


#einzelne Bahnpunkte auf Beckenrandgerade finden (ungetestet)
k_r=(B2_y-B1_y)/(B2_x-B1_x)

d_new=B1_y-(k_new*B1_x)
# Ausschlussbereich bis oberen, seitlichen Beckenrand von Track n definiert n(x)<k_n*x+d_n, zuvor l(x)
"""


n=int(input("Anzahl der Bahnen:"))

out = cv2.VideoWriter('output1.mp4', fourcc, fps,(frame_width,frame_height),True )
while(cap.isOpened()):
    ret,frame = cap.read()
    if ret == True:
        swimmer_found = False
        results = model(frame, imgsz=640, stream=True, verbose=False)
        balls=0
        
        
        for result in results:
                
            for box in result.boxes.cpu().numpy():
                balls+=1 #gibt Bällen eine ID
                if show_boxes:
                    a+=1

                    cls = int(box.cls[0])
                    name=result.names[int(box.cls[0])]
                    r = box.xyxy[0].astype(int)
                    c= box.conf[0] #confidence of detection
                    
                    if c>0.3:
                        x=r[0]
                        y=r[1]
                        dct['list_%s' % i] = [x,y]
                        act=np.mean([x,y])
                        pos=np.mean(list(dct.values()))

                        list_balls.append(balls)

                        print(a, balls, act, pos)

                        if len(list_balls)==10 and B1_x==0 and index<2:
                            if act>(pos+10) or act<(pos-10) or np.mean(list_balls)>1.2:
                                print("Halten Sie den Ball ruhig oder entfernen Sie andere ähnliche Gegenstände")
                                a=0
                                list_balls=[]
                            else:
                                B1_x=x
                                B1_y=y
                                print("Position B1 erkannt: x=",B1_x," y=",B1_y)
                                winsound.Beep(500, 3000)
                                
                                
                                
                                time.sleep(10)
                                a=0
                                list_balls=[]
                                x=r[0]
                                y=r[1]
                                dct['list_%s' % i] = [x,y]
                                act=np.mean([x,y])
                        
                        if len(list_balls)==10 and B1_x!=0 and len(xlist_lines)<n:
                            while index<=n:
                                if len(list_balls)==10 and B1_x!=0 and B2_x==0: #für nBahnen
                                    if act>(pos+10) or act<(pos-10):
                                        a=0
                                        list_balls=[]
                                    else:
                                        xlist_lines.append(x)
                                        ylist_lines.append(y)
                                        print("Bahn", index, "erkannt: x=",xlist_lines[index-1]," y=",ylist_lines[index-1])
                                        winsound.Beep(500, 3000)
                                        
                                        
                                        #Kreis zeichnen (Mittelpunkt, Radius, BRG-Farbe, Linienstärke, wenn neg. Farbausfüllung des Kreises)
                                        cv2.circle(frame,(xlist_lines[index-1],ylist_lines[index-1]), marker_radius, (0,255,0), marker_thickness)
                                        #Text einfügen (Text, Koordinate links oben, font, Schriftgröße, BRG-Farbe, Schriftdicke)
                                        font = cv2.FONT_HERSHEY_SIMPLEX
                                        cv2.putText(frame,'Bahn '+str(index),(xlist_lines[index-1]+marker_distance,ylist_lines[index-1]+marker_distance), font, 1,(0,0,0),marker_thickness)
                                        
                                        
                                        
                                        time.sleep(15)
                                        a=0
                                        list_balls=[]
                                        x=r[0]
                                        y=r[1]
                                        dct['list_%s' % i] = [x,y]
                                        act=np.mean([x,y])
                                        pos=np.mean(list(dct.values()))
                                        index+=1
                                        break
                                        

                        elif len(list_balls)==10 and B1_x!=0 and B2_x==0 and len(xlist_lines)==n:
                            if act>(pos+10) or act<(pos-10):
                                a=0
                                list_balls=[]
                            else:
                                B2_x=x
                                B2_y=y
                                print("Position B2 erkannt: x=",B2_x," y=",B2_y)
                                winsound.Beep(500, 3000)
                                                    
                                               
                                time.sleep(15)
                                a=0
                                list_balls=[]
                                x=r[0]
                                y=r[1]
                                dct['list_%s' % i] = [x,y]
                                act=np.mean([x,y])
                                pos=np.mean(list(dct.values()))

                        elif len(list_balls)==10 and B1_x!=0 and B2_x!=0:
                            if act>(pos+10) or act<(pos-10):
                                a=0
                                list_balls=[]
                            else:
                                M_x=x
                                M_y=y
                                print("Position M erkannt: x=",M_x," y=",M_y)
                                M_length=math.sqrt( math.pow(B2_x-M_x,2) + math.pow(B2_y-M_y,2) ) 
                                print('xList: ',xlist_lines,'yList: ',ylist_lines) 
                                print("Signallinienlänge in Pixel: ",M_length)
                                
                                winsound.Beep(500, 3000)
                                
                                k_r=(B2_y-B1_y)/(B2_x-B1_x)
                                e=(B2_x-B1_x)*(B2_x-B1_x)
                                f=(B2_y-B1_y)*(B2_y-B1_y)
                                l=np.sqrt(e+f) #Schwimmbeckenbreite in Pixel

                                S_x=M_x-l #Pkt. S befindet sich auf Höhe von B1
                                S_y=M_y-l*k_r
                                
                                k_t=(B2_y-S_y)/(B2_x-S_x)
                            
                                while (j<index-1):
                                    #cv2.line(image, start_point, end_point, color in BGR, thickness in PX)
                                    cv2.line(frame,(int(xlist_lines[j]),int(ylist_lines[j])),(int(xlist_lines[j]*10),int(ylist_lines[j]*10+k_t)),(255,0,0),2) #Bahnlinien
                                    #Text einfügen (Text, Koordinate links oben, font, Schriftgröße, BRG-Farbe, Schriftdicke)
                                    cv2.putText(frame,'Bahn '+str(j+1),(xlist_lines[j],ylist_lines[j]), font, 0.5,(0,255,0),marker_thickness) #Bahnkennzeichnungen

                                    j+=1

                                
                        elif B1_x!=0 and B2_x!=0 and M_x!=0:
                                                  
                                raise SystemExit("Konfiguration erfolgreich!")

                    #Rechteck über Schwimmer
                        cv2.rectangle(frame, r[:2], r[2:], (255, 255, 255), 2)
                        
                    #Punkt B1 Anzeige
                        #Kreis zeichnen (Mittelpunkt, Radius, BRG-Farbe, Linienstärke, wenn neg. Farbausfüllung des Kreises)
                        cv2.circle(frame,(B1_x,B1_y), marker_radius, (255,255,255), marker_thickness)
                        
                        #Text einfügen (Text, Koordinate links oben, font, Schriftgröße, BRG-Farbe, Schriftdicke)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(frame,'B1',(B1_x+marker_distance,B1_y+marker_distance), font, 1,(0,0,0),marker_thickness)

                    #Punkt B2 Anzeige
                        cv2.circle(frame,(B2_x,B2_y), marker_radius, (255,255,255), marker_thickness)
                        
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(frame,'B2',(B2_x+marker_distance,B2_y+marker_distance), font, 1,(0,0,0),marker_thickness)                    

                    #Punkt M Anzeige
                        cv2.circle(frame,(M_x,M_y), marker_radius, (255,255,255), marker_thickness)
                        
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(frame,'M',(M_x+marker_distance,M_y+marker_distance), font, 1,(0,0,0),marker_thickness)
                                
                        
                    #Beckenrandlinie
                        #cv2.line(image, start_point, end_point, color in BGR, thickness in PX)
                        cv2.line(frame,(B1_x,B1_y),(B2_x,B2_y),(255,0,0),2) #Beckenrandlinie
                        
                       

                        cv2.line(frame,(M_x,M_y),(int(S_x),int(S_y)),(0,255,0),2) #Signallinie

                       
                        

                if cls == 0:
                    
                    swimmer_found = True
                    

        if swimmer_found:
            
            out.write(frame)
        
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
# for openCV - When everything done, release the capture
cap.release()
out.release()
cv2.destroyAllWindows()