🔫 Real-Time Weapon Detection using Python & YOLO  

This project detects weapons (e.g., Guns, Knives) in "real-time" using a YOLO (You Only Look Once), object detection model and Python.  
Whenever a weapon is detected, it **plays an alarm sound** 🔔 and "sends an email (png file)alert " 📧 with the captured frame attached.

---

 🚀 Features
- Real-time detection via Laptop Camera (Webcam) 
- Detects "Weapons like Guns & Knives"
- Plays "alarm sound " when a weapon is detected 
- Sends "email alerts with attached image" of the detection
- Uses "YOLOv3 custom-trained model"
- Built using "Python + OpenCV + DNN"

---
 🛠️ Requirements
Make sure you have the following installed:

- Python 3.8+  
- Virtual Environment (`venv`)  
- Required Libraries:
  ```bash
  pip install opencv-python numpy playsound smtplib ssl

Real-Time-Weapon-Detection/
│
├── yolov3_training_2000.weights   # Pre-trained YOLO model
├── yolov3_testing.cfg             # YOLO configuration file
├── alarm.wav                      # Custom alarm sound
├── weapon_detection.py            # Main detection script
├── README.md                      # This file
└── requirements.txt               # (Optional) Dependencies list

🖥️ How to Run

1️⃣ Create Virtual Environment

python -m venv venv

Activate it:

On Windows:

venv\Scripts\activate

2️⃣ Install Dependencies

pip install opencv-python numpy playsound

3️⃣ Download YOLO Model

Place your trained YOLO model files in the project folder:

yolov3_training_2000.weights

yolov3_testing.cfg

4️⃣ Run the Program

python weapon_detection.py

"Press ESC to stop the camera."

📧 Email Alerts

Configure your Gmail in weapon_detection.py:

EMAIL_SENDER = "your_email@gmail.com"
EMAIL_PASSWORD = "your_app_password"  # Use Gmail App Password
EMAIL_RECEIVER = "receiver_email@gmail.com"

"Enable App Passwords in Gmail (required for login)."

🎯 Example Output

When a weapon is detected:

Frame shows bounding box around weapon.

Alarm sound plays.

Email sent with alert.jpg attached.

📌 Notes

Detection is based on YOLOv3 custom model.

You can add more labels (e.g., Fight, Sword) by training the model again.

👨‍💻 Author

Developed by [Mathiyalayan]

📧 Contact: [mathiy379@gmail.com]
