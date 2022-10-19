import face_recognition
import cv2
import numpy as np
import gspread

#Create Some Class!
class Person:

    def __init__(self, name, age, id, jobs, income, married):
        self.name = name
        self.age = age
        self.jobs = jobs
        self.income = income
        self.married = married

gc = gspread.service_account(filename='skilful-bliss-364906-e3a82a50429c.json')

#Load all data! (Preload)
sh = gc.open("Face Identify Charana")
ws = sh.get_worksheet(0)

namelist = ws.col_values(2)
namelist.pop(0)

ListOfPerson = {}

for x in namelist:
    ListOfPerson[x] = Person(ws.cell(ws.find(x).row,ws.find(x).col-1).value,ws.cell(ws.find(x).row,ws.find(x).col+1).value,x,ws.cell(ws.find(x).row,ws.find(x).col+2).value,ws.cell(ws.find(x).row,ws.find(x).col+3).value,ws.cell(ws.find(x).row,ws.find(x).col+4).value)
#
video_capture = cv2.VideoCapture(0)

charana_image = face_recognition.load_image_file("myphotro.png")
charana_face_encoding = face_recognition.face_encodings(charana_image)[0]

tomas_image = face_recognition.load_image_file("Thomas_Eddy.jpg")
tomas_face_encoding = face_recognition.face_encodings(tomas_image)[0]

known_face_encodings = [
    charana_face_encoding,
    tomas_face_encoding
]
#เราจะใช้ ID แทนชื่อเพื่อง่ายต่อการระบุตัวตน
known_face_names = [
    "41852",
    "41853"
]

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    ret, frame = video_capture.read()

    if process_this_frame:
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        rgb_small_frame = small_frame[:, :, ::-1]
        
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        cv2.rectangle(frame, (left, bottom), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        id = "ID : "+name
        cv2.putText(frame, id, (left + 150, top + 40), font, 0.7, (0, 0, 255), 1)
        if name != "Unknown":
            realname = "Name : "+ListOfPerson[name].name
            age = "Age : "+ListOfPerson[name].age
            job = "Jobs : "+ListOfPerson[name].jobs
            income = "Income : "+ListOfPerson[name].income
            married = "Married : "+ListOfPerson[name].married
            cv2.putText(frame, realname, (left + 150, top + 20), font, 0.7, (0, 0, 255), 1)
            cv2.putText(frame, age, (left + 150, top + 60), font, 0.7, (0, 0, 255), 1)
            cv2.putText(frame, job, (left + 150, top + 80), font, 0.7, (0, 0, 255), 1)
            cv2.putText(frame, income, (left + 150, top + 100), font, 0.7, (0, 0, 255), 1)
            cv2.putText(frame, married, (left + 150, top + 120), font, 0.7, (0, 0, 255), 1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()