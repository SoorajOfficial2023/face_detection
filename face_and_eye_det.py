import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

cap = cv2.VideoCapture(0)

def detect_face_with_eyes(frame):
    gray_image = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image,1.1,5,minSize=(40,40))
    
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),4)
        roi_gray = gray_image[y:y+h,x:x+w]
        roi_color = frame[y:y+h,x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        for (ex,ey,ew,eh) in eyes:
            center = (int(ex+ew/2),int(ey+eh/2))
            radius = int(min(ew,eh)/2)
            cv2.circle(roi_color,center,radius,(0,0,255),2)
    return frame

while True:
    ret,frame = cap.read()
    if ret is False:
        break
    frame_with_faces = detect_face_with_eyes(frame)
    cv2.imshow("frame",frame_with_faces)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
    
    