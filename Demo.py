import cv2
import numpy as np
# import time
import os
import mediapipe as mp
import HandTracking as htm
# import pyttsx3
import datetime

# import turtle
# import speech_recognition as sr


'''engine = pyttsx3.init("sapi5")
voices = engine.getProperty("voices")
print(voices[0].id)
print(voices)
engine.setProperty("voice", voices[0].id)

def speak(audio):

    engine.say(audio)
    engine.runAndWait()

def wish():

    hour = int(datetime.datetime.now().hour)

    if hour>=0 and hour<12:
        speak("Good Morning.")
    elif hour>=12 and hour<17:
        speak("Good Afternoon.")
    else:
        speak("Good Evening.")

    speak("Which figure do you want to draw.")

def takeCommand():
    wish()

    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        r.pause_threshold = 0.8
        audio = r.listen(source)

    try:
        print("Recognizing....")
        query =  r.recognize_google(audio,language='en-in')
        print(f" User said : {query}\n")

    except Exception as e:
        print(e)
        print("Please!Say that again ")
        return 'None'

    return query
'''

########################
brushThickness = 15
eraserThickness = 50

########################
folderPath = "Img"
myList = os.listdir(folderPath)
# print(myList)

overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')  # it is our complete path to run image in loop
    overlayList.append(image)  # image is stored in overlayList.

# print(len(overlayList)) #it print the number of images in folder.

# Now to run Webcam
header = overlayList[0]
drawColor = (255, 0, 255)
cap = cv2.VideoCapture(0)
cap.set(3, 1080)
cap.set(4, 720)

detector = htm.handDetector(detectionCon=0.85)

xp, yp = 0, 0  # xp and yp means previous points
imgCanvas = np.zeros((720, 1280, 3), np.uint8)  # Original Value 720, 1280, 3

# cv2.namedWindow("Paint", cv2.WINDOW_NORMAL)

while True:
    # 1.Import image
    success, img = cap.read()  # cap.read() returns two values one is boolean which is stored in success and another is tuple which is stored in img.
    img = cv2.flip(img, 1)

    # 2.Find hand landmarks
    img = detector.findhands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        # print(lmList)

        # tip of index and middle finger
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        # 3.Check which fingers are up
        fingers = detector.fingersUp()
        # print(fingers)

        # 4.Selection mode - two fingers are up
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            print("Selection mode")

            # Checking for the click
            if y1 < 50:  # here 125 is the value in header.
                if 150 < x1 < 200:
                    header = overlayList[0]
                    drawColor = (255, 0, 255)
                elif 250 < x1 < 300:
                    header = overlayList[1]
                    drawColor = (255, 0, 0)
                elif 350 < x1 < 450:
                    header = overlayList[2]
                    drawColor = (0, 255, 0)
                elif 500 < x1 < 600:
                    header = overlayList[3]
                    drawColor = (0, 0, 255)
                elif 650 < x1 < 750:
                    header = overlayList[4]
                    drawColor = (0, 0, 0)

                elif 800 < x1 < 900:
                    header = overlayList[5]
                    '''query = takeCommand().lower()
                    code = cv2.waitKey(1)
                    if 'square' in query:
                         cv2.rectangle(imgCanvas, (150, 150), (330, 330), drawColor, 3, 1)
                         cv2.rectangle(img, (150, 150), (330, 330), drawColor, 3, 1)
 
                    elif 'rectangle' in query:
                        cv2.rectangle(imgCanvas, (170, 500), (550, 350), drawColor, 3, 1)
                        cv2.rectangle(img, (170, 500), (550, 350), drawColor, 3, 1)
 
                    elif 'circle' in query:
                        cv2.circle(imgCanvas, (750, 250), 100, drawColor, 3)
                        cv2.circle(img, (750, 250), 100, drawColor, 3)'''

                elif 950 < x1 < 1050:
                    header = overlayList[6]

            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

        # 5.If Drawing mode - index finger is up
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            print("Drawing mode")
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)

            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

            xp, yp = x1, y1

    imgGrey = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGrey, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    # Setting the header image on cam gui.(this is a menu bar)
    img[0:118, 0:1080] = header  # 0:118 represent height of image and 0:1079 represent width of image
    imgCanvas[0:118, 0:1080] = header

    # img = cv2.addWeighted(img,0.5,imgCanvas,0.5,0)

    cv2.imshow('Image', img)
    cv2.imshow('ImgCanvas', imgCanvas)
    # cv2.imshow('ImgInverse', imgInv)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()


