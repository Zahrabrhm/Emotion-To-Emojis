import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import emoji
from PIL import Image, ImageFont, ImageDraw


model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

model.load_weights('model.h5')

    # prevents openCL usage and unnecessary logging messages
cv2.ocl.setUseOpenCL(False)

# Dictionary which assigns each label an emotion (alphabetical order)
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Start the webcam feed
# Start the webcam feed
cap = cv2.VideoCapture(0)
emoji_font = ImageFont.truetype("C:/Windows/Fonts/seguisym.ttf", 40)

while True:
    # Find haar cascade to draw bounding box around face
    ret, frame = cap.read()
    if not ret:
        break

    facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        prediction = model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        
        if maxindex == 0:
            emoji_tion = '\U0001F621'  # Unicode for :angry_face:
        elif maxindex == 1:
            emoji_tion = '\U0001F922'  # Unicode for :nauseated_face:
        elif maxindex == 2:
            emoji_tion = '\U0001F628'  # Unicode for :fearful_face:
        elif maxindex == 3:
            emoji_tion = '\U0001F642'  # Unicode for :slightly_smiling_face:
        elif maxindex == 4:
            emoji_tion = '\U0001F610'  # Unicode for :neutral_face:
        elif maxindex == 5:
            emoji_tion = '\U0001F641'  # Unicode for :slightly_frowning_face:
        elif maxindex == 6:
            emoji_tion = '\U0001F632'  # Unicode for :astonished_face:

        cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        emoji_image = Image.new('RGBA', (100, 100), (255, 255, 255, 0))  # Create an RGB image without an alpha channel
        d = ImageDraw.Draw(emoji_image)
        d.text((0, 0), emoji_tion, font=emoji_font, fill=(255, 192, 203, 255))

        # Convert PIL image to numpy array
        emoji_np = np.array(emoji_image)

        # Split the emoji into RGB and alpha channels
        emoji_rgb = emoji_np[:, :, :3]
        emoji_alpha = emoji_np[:, :, 3]

        ## Calculate the coordinates of the top, bottom, left, and right of the emoji
        top = max(y-40, 0)
        bottom = min(y+60, frame.shape[0])
        left = max(x+20, 0)
        right = min(x+120, frame.shape[1])

        # Extract the area of the frame where the emoji will be placed
        frame_area = frame[top:bottom, left:right]

        # Resize the emoji to match the size of the frame area
        emoji_resized = cv2.resize(emoji_rgb, (right-left, bottom-top))
        alpha_resized = cv2.resize(emoji_alpha, (right-left, bottom-top))

        # Calculate the inverse of the alpha mask
        inverse_alpha = 255 - alpha_resized

        # Use the alpha and inverse alpha masks to calculate the new values for the frame area
        new_area = (emoji_resized * (alpha_resized[..., None] / 255)).astype(np.uint8) + (frame_area * (inverse_alpha[..., None] / 255)).astype(np.uint8)

        # Place the new area onto the frame
        frame[top:bottom, left:right] = new_area



    cv2.imshow('Video', cv2.resize(frame, (1600, 960), interpolation=cv2.INTER_CUBIC))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()