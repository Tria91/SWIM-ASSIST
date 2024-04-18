from ultralytics import YOLO
import cv2
import winsound #für Audioausgabe
import os #für Audioausgabe via MP3-Dateien
import time #für Sleep während Audioausgabe
import numpy as np
import math
import pandas as pd

path='C:\\Users\\offic\\Desktop\\Schwimmen\\Musik\\' #Pfad für MP3-Dateien

cap = cv2.VideoCapture(0)
# Load custom trained YOLOv8 model
model = YOLO("runs/detect/train3/weights/best.pt")
#runs/detect/train3/weights/best.pt
#face/model-face.pt
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

show_boxes = True

#für Multi-Feedback
z=0
j=0
xlist_lines_center = []
ylist_lines_center = []
swimmer_difference_to_track_list = []

xlist_lines.insert(0, B1_x)   #insert B1_x
ylist_lines.insert(0, B1_y)   #insert B1_y

#für Geschwindigkeitserkennung
xlist_last_position = [2000]*n
ylist_last_position = [2000]*n
list_last_time = [time.time()]*n
list_swimmer_act_speed=[0]*n
list_need_time = [4]*n # 4 Sekunden Abstand zur Initialisierung aller Bahnen



while (z<len(xlist_lines)-1):
    xlist_lines_center.append(xlist_lines[z]+((xlist_lines[z]-xlist_lines[z+1])/2))
    ylist_lines_center.append(ylist_lines[z]+((ylist_lines[z+1]-ylist_lines[z])/2))
    z+=1

out = cv2.VideoWriter('output1.mp4', fourcc, fps,(frame_width,frame_height),True )
while(cap.isOpened()):
    ret,frame = cap.read()
    if ret == True:
        swimmer_found = False
        results = model(frame, imgsz=640, stream=True, verbose=False)
        
        for result in results:
            
            for box in result.boxes.cpu().numpy():
                
                
                if show_boxes:
                    name=result.names[int(box.cls[0])]
                    r = box.xyxy[0].astype(int)
                    c= box.conf[0] #confidence of detection
                    if c>0.3:
                        x=r[0]
                        y=r[1]
                        
                        print(c, x, y)
                        
                        #print(xlist_lines_center) #Liste der x-Koordinaten der Bahnmittelpunkte
                        
                    """
                    if r[0]<320: #Musikausgabe wenn Person sich links von der Mitte (320Pixel) aus befindet
                        winsound.Beep(500, 500) #CopyPaste - erster Wert=Frequenz - Zweiter Dauer in ms
                    """
                    #Rechteck über Schwimmer
                    cv2.rectangle(frame, r[:2], r[2:], (255, 255, 255), 2) 
                    
                    
                #Fuer einen Schwimmer im gesamten Becken:
                    #Ausschlussbereich bis hor. Beckenrad definiert durch r(x)<k_r*x+d_r
                    d_r=B1_y-(k_r*B1_x)
                    
                    # Ausschlussbereich bis oberen, seitlichen Beckenrand von Track n definiert n(x)<k_n*x+d_n, zuvor l(x)
                    k_n=(M_y-B2_y)/(M_x-B2_x)
                    d_n=B2_y-(k_n*B2_x)
                    
                    # Ausschlussbereich bis unteren, seitlichen Beckenrand von Track 1 definiert t0(x)>k_n*x+d_t0
                    d_t0=B1_y-(k_n*B1_x)
                    
                    #Signalbereich im Wasser definiert s(x)<k_r*x+d_s
                    d_s=M_y-(k_r*M_x)
                    
                    """
                    #Single-Feedback (= ein Schwimmer im gesamten Becken)
                    #von links unten bis rechts oben: unterer seitlicher Beckenrand, hor. Beckenrand, oberer Beckenrand, Signalbereich
                    if y<k_n*x+d_t0 and y>k_r*x+d_r and y>k_n*x+d_n and y<k_r*x+d_s:
                        winsound.Beep(500, 3000) #erster Wert=Frequenz - Zweiter Dauer in ms
                    """
                    
                    #für Multifeedback
                    while (j<len(xlist_lines_center)):
                        swimmer_difference_to_track_list.append( math.sqrt( math.pow(x-xlist_lines_center[j],2) + math.pow(y-ylist_lines_center[j],2) ) )    
                        j+=1
                    
                    swimmer_track_nr=swimmer_difference_to_track_list.index(min(swimmer_difference_to_track_list))+1
                    print("Schwimmer Bahn-Nr.", swimmer_track_nr)
                    list_swimmer_act_speed[swimmer_track_nr-1]=(math.sqrt( math.pow(x-xlist_last_position[swimmer_track_nr-1],2) + math.pow(y-xlist_last_position[swimmer_track_nr-1],2) ) ) / (time.time() - list_last_time[swimmer_track_nr-1]) #-1, da ja Liste bei Index 0 beginnt
                    print("Aktuelle Geschwindigkeitsliste in Pixel/s", list_swimmer_act_speed)
                    list_need_time[swimmer_track_nr-1]=swimmer_difference_to_track_list[swimmer_track_nr-1] / list_swimmer_act_speed[swimmer_track_nr-1]
                    print("Voraussichtliche Ankunftszeitliste in Sekunden ", list_need_time)
                    
                    
                    #Multi-Feedback (vorher If-Anwendung löschen)
                    if y<k_n*x+d_t0 and y>k_r*x+d_r and y>k_n*x+d_n:
                        if swimmer_difference_to_track_list[swimmer_track_nr-1]<M_length or list_need_time[swimmer_track_nr-1]<2:
                            winsound.Beep(500, 3000) #erster Wert=Frequenz - Zweiter Dauer in ms
                            print("Warnung! Schwimmer: ",swimmer_track_nr)
                    
                    
                    
                cls = int(box.cls[0])
                
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