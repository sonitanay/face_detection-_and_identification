import cv2
import numpy as np
import face_recognition as fr
import os 
import time 

def updateMainArr():
    img_dir = os.listdir("./saved_images/")

    for filename in img_dir:
        comp_img_arr.append(fr.load_image_file("./saved_images/"+filename.lower()))

    for i in range(len(comp_img_arr)):
        comp_enc_arr.append(fr.face_encodings(comp_img_arr[i])[0])
    
    return img_dir

img_dir = []
comp_img_arr = []
comp_enc_arr = []
face_cascade = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)
runtime = 0

img_dir = updateMainArr()

""" print(len(comp_img_arr))
print(len(comp_enc_arr))
tanay_img = fr.load_image_file("./saved_images/tanay.jpg")
tanay_face_enc = fr.face_encodings(tanay_img)[0]
print(tanay_face_enc)
 """
while True:

    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.resize(frame_gray, (0,0), fx = .25, fy = .25)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_cascade.detectMultiScale(frame_gray, 1.2, 5)
    face_loc = []
    face_enc = []
    cmp = []

    for (x,y,w,h) in faces:
        face_loc.append([y*4, (x+w)*4, (y+h)*4, x*4])
    
    face_enc = fr.face_encodings(frame_rgb, face_loc)

    for (y,x1,y1, x) in face_loc:
        for face in face_enc:
            
            try:
                face_cmp = fr.compare_faces(comp_enc_arr, face, tolerance=0.4)
            except:
                pass
            if True in face_cmp:
                val = face_cmp.index(True)
                name = img_dir[val]
                print(name)
                cv2.putText(frame, name.upper() ,(x, y1+15),cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255,0), 2 )
            else:
                print("unknown")
                face_roi = frame[y:y1, x:x1]
                cv2.imshow("current face", face_roi)
                if runtime >= 5:
                    response = input("do you want to give a name for this face ? (y/n) : ")
                    if response == 'y':
                        response = input("enter a valid name that can be displayed : ")
                        cv2.imwrite("./saved_images/"+response+".jpg", face_roi)
                        img_dir = updateMainArr()
                    runtime = 0
                cv2.putText(frame,"unknown".upper() ,(x, y1+15),cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255,0), 2 )
        cv2.rectangle(frame, (x, y), (x1, y1), (0,255,0), 3)
    runtime += 1


    cv2.imshow("video", frame)
    if cv2.waitKey(30) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()