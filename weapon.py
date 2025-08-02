import cv2
import numpy as np
import smtplib
import ssl
import os
import time
from email.message import EmailMessage
from playsound import playsound

# --- Email Configuration ---
EMAIL_SENDER = "your_email@gmail.com"
EMAIL_PASSWORD = "your_app_password"  # Use Gmail App Password
EMAIL_RECEIVER = "receiver_email@gmail.com"

# Send Email Function
def send_email_alert(image_path):
    msg = EmailMessage()
    msg['Subject'] = 'Weapon Detected Alert!'
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECEIVER
    msg.set_content('Alert: A weapon has been detected. See attached image.')

    with open(image_path, 'rb') as img:
        msg.add_attachment(img.read(), maintype='image', subtype='jpeg', filename='alert.jpg')

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
        smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
        smtp.send_message(msg)
        print("✅ Email sent with alert image.")

# --- Load YOLO ---
net = cv2.dnn.readNet("yolov3_training_2000.weights", "yolov3_testing.cfg")
classes = ["Weapon"]
output_layer_names = net.getUnconnectedOutLayersNames()
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# --- Webcam Setup ---
cap = cv2.VideoCapture(0)
last_alert_time = 0  # Timestamp for last alert
alert_cooldown = 1  # Seconds between alerts

while True:
    ret, img = cap.read()
    if not ret:
        print("❌ Error: Frame not captured.")
        break

    height, width, _ = img.shape

    # YOLO Input
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layer_names)

    # Detection Parsing
    class_ids, confidences, boxes = [], [], []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    weapon_detected = False

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            weapon_detected = True

    # Trigger Alert if needed
    current_time = time.time()
    if weapon_detected and (current_time - last_alert_time > alert_cooldown):
        print("[ALERT] Weapon Detected!")

        alert_img_path = "alert.jpg"
        cv2.imwrite(alert_img_path, img)

        # Play sound
        try:
            playsound("alarm.wav")
        except Exception as e:
            print("Sound Error:", e)

        # Send Email
        send_email_alert(alert_img_path)

        last_alert_time = current_time  # Update alert timestamp

    # Display Feed
    cv2.imshow("Weapon Detection", img)
    if cv2.waitKey(1) == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
