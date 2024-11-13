import cv2
import numpy as np
import os
import face_recognition
from datetime import datetime

path = 'images'
images = []
classnames = []
mylist = os.listdir(path)
print(mylist)

for cl in mylist:
    curImg = cv2.imread(f'{path}/{cl}') 
    images.append(curImg)
    classnames.append(os.path.splitext(cl)[0])  # Store names without file extension
print(classnames)

def findEncodings(images):
    encodelist = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for face_recognition
        encode = face_recognition.face_encodings(img)[0]  # Get the face encoding
        encodelist.append(encode)
    return encodelist

def markAttendence(name):
    # Check if the attendance file exists
    if not os.path.exists('Attendence.csv'):
        with open('Attendence.csv', 'w') as f:
            f.writelines('Name,Time\n')  # Write header if file is created

    with open('Attendence.csv', 'r+') as f:
        mydatalist = f.readlines()
        namelist = []
        for line in mydatalist:
            entry = line.split(',')
            namelist.append(entry[0])  # Extract names from CSV file
        if name not in namelist:
            now = datetime.now()
            dtstring = now.strftime('%H:%M:%S')  # Format the time
            f.writelines(f'\n{name},{dtstring}')  # Add the name and time to the CSV

encodelistknown = findEncodings(images)
print('Encoding complete')

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgs = cv2.resize(img, (0, 0), None, 0.25, 0.25)  # Resize to speed up processing
    imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # Detect faces in the frame
    facecurrentFrame = face_recognition.face_locations(imgs)
    encodecurrentFrame = face_recognition.face_encodings(imgs, facecurrentFrame)

    for encodeface, faceloc in zip(encodecurrentFrame, facecurrentFrame):
        matches = face_recognition.compare_faces(encodelistknown, encodeface)  # Compare faces
        facedis = face_recognition.face_distance(encodelistknown, encodeface)  # Calculate face distances

        matchIndex = np.argmin(facedis)  # Get the index of the closest match
        if matches[matchIndex]:  # If a match is found
            name = classnames[matchIndex].upper()  # Get the name of the matched person

            y1, x2, y2, x1 = faceloc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4  # Scale face locations back to original size
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw a rectangle around the face
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)  # Draw background for name
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)  # Put name text

            markAttendence(name)  # Mark the attendance

    cv2.imshow('Webcam', img)  # Show the frame
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit if 'q' is pressed
        break

cap.release()
cv2.destroyAllWindows()
