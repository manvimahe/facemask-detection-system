import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import smtplib

# Load the model and the face detection classifier
model = load_model('face_detection.h5')
face_det_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

text_dict = {0: 'Mask not worn Correctly or no mask', 1: 'Mask not worn Correctly or no mask', 2: 'Mask On'}
rect_color_dict = {0: (0, 0, 255), 1: (0, 0, 255), 2: (0, 255, 0)}

def send_email(subject, text):
    message = f"Subject:{subject}\n\n{text}"
    try:
        mail = smtplib.SMTP('smtp.gmail.com', 587)
        mail.ehlo()
        mail.starttls()
        mail.login('your_email', 'mail_id_password')
        mail.sendmail('your_email', 'recievers_mail', message)
        mail.close()
    except Exception as e:
        st.error(f"Error sending email: {e}")

def detect_and_display():
    vid_source = cv2.VideoCapture(0)
    stframe = st.empty()

    while True:
        ret, img = vid_source.read()
        grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_det_classifier.detectMultiScale(grayscale_img, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = grayscale_img[y:y+w, x:x+w]
            resized_img = cv2.resize(face_img, (112, 112))
            normalized_img = resized_img / 255.0
            reshaped_img = np.reshape(normalized_img, (1, 112, 112, 1))
            result = model.predict(reshaped_img)

            label = np.argmax(result, axis=1)[0]

            cv2.rectangle(img, (x, y), (x+w, y+h), rect_color_dict[label], 2)
            cv2.rectangle(img, (x, y-40), (x+w, y), rect_color_dict[label], -1)
            cv2.putText(img, text_dict[label], (x, y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 0, 0), 2)

            if label == 0:
                st.warning("Access denied")
                send_email("Alert", "No Mask Detected. Access Denied.")

        stframe.image(img, channels="BGR")

        if cv2.waitKey(1) & 0xFF == 27:
            break

    vid_source.release()
    cv2.destroyAllWindows()

st.title("Face Mask Detection")

if st.button("Start Detection"):
    detect_and_display()
