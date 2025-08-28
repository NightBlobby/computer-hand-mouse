# 🖐️ Computer Hand Mouse  
**Control your computer using hand gestures with OpenCV and Mediapipe.**

---

## 📌 Overview  
**Computer Hand Mouse** is a Python-based project that allows users to **control the mouse cursor using hand gestures**. This is achieved through **computer vision techniques** leveraging **OpenCV** and **MediaPipe**, providing a seamless and interactive experience without the need for additional hardware.

---

## ✅ Features  
✔️ **Hand Gesture Detection** – Uses real-time hand tracking to recognize finger positions.  
✔️ **Mouse Control** – Move the cursor with your hand in front of the webcam.  
✔️ **Gesture-Based Clicks** – Perform left-click, right-click, and drag actions using specific gestures.  
✔️ **Lightweight & Easy to Use** – Minimal dependencies and quick setup.  
✔️ **Customizable** – Extend functionality by adding your own gestures.

---

## 🛠️ Technologies Used  
- **Python 3.x**  
- **OpenCV** – For real-time image processing.  
- **MediaPipe** – For accurate hand tracking and gesture detection.  
- **PyAutoGUI** – To control mouse and screen interactions.  

---

## 📂 Project Structure  
```
computer-hand-mouse/
│
├── hand_mouse.py # Main script for gesture control
├── requirements.txt # Required dependencies
└── README.md # Project documentation
```

---

## ⚡ Installation & Setup 

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/NightBlobby/computer-hand-mouse.git
cd computer-hand-mouse
```
### 2️⃣ Install the imports
```
pip install opencv-python mediapipe pyautogui
```
### 3️⃣ Run the Program
```
python hand_mouse.py
```

---
## 🧠 How It Works

- Captures video feed from your webcam.

- Detects hand landmarks using MediaPipe Hand Detection.

- Maps hand movement to your screen coordinates.

- Interprets gestures for actions like click, drag, scroll.

---

## ✋ Supported Gestures  
| Gesture             | Action        |
|---------------------|-------------|
| Index Finger Up     | Move Mouse  |
| Index + Thumb Pinch | Left Click  |
| Two Fingers Up      | Right Click |
| Closed Fist         | Drag/Scroll |

---

## 🚀 Future Improvements  
- ✅ Add **gesture customization** via config file.  
- ✅ Improve **accuracy in low light conditions**.  
- ✅ Add **multi-hand support**.  
- ✅ Implement **scroll gestures**.  

---

## 👨‍💻 Author  
**Blobby (NightBlobby)**  
[GitHub Profile](https://github.com/NightBlobby)  

---

## 📜 License  
This project is licensed under the **MIT License** – feel free to use and modify.  
