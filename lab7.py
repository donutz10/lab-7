import cv2

#loading face cascade
face_cascade = cv2.CascadeClassifier('haar_face.xml')
#video
video = cv2.VideoCapture('video.mp4')
#Font
Font = cv2.FONT_HERSHEY_SIMPLEX 
#saving purposes
frame_width = int(video.get(3))
frame_height = int(video.get(4))
#saving size and importing frame and width
size = (frame_width, frame_height)
# to output the result
result = cv2.VideoWriter('FaceDetect.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         12, size)

while True:
    frame, vid = video.read()
    # detecting the faces
    gray = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY) 
    # read faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    #draw rectangle and text
    for (x, y, w, h) in faces:  
        cv2.rectangle(vid, (x, y), (x+w, y+h), (0, 255, 0), 1)
        cv2.putText(vid, 'Face', (x,y-10), Font, 0.5, (255,0,0), 2)
    # showing the video frame
    cv2.imshow('Video', vid)
    #takes result of video and writes to FaceDetect.avi
    result.write(vid)
    # 30 frames per second
    cv2.waitKey(30)

result.release()