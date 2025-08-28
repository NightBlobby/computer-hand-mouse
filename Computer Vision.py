import time
import math
import cv2
import mediapipe as mp
from collections import deque
import sys
import numpy as np

# ---------------- CONFIG ----------------
BUFFER_LEN = 5          # Number of past points for smoothing
# ----------------------------------------

# Global MediaPipe components
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

def check_dependencies():
    """Check if all required dependencies are available"""
    try:
        import cv2
        import mediapipe as mp
        return True
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install required packages:")
        print("pip install opencv-python mediapipe")
        return False

def check_camera():
    """Check if camera is available and working"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera. Please check if:")
        print("1. Camera is connected")
        print("2. Camera is not being used by another application")
        print("3. Camera permissions are granted")
        return False
    
    # Test if we can read a frame
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Error: Cannot read from camera")
        return False
    
    return True

def check_mediapipe_version():
    """Check MediaPipe version and provide compatibility info"""
    try:
        import mediapipe as mp
        version = mp.__version__
        print(f"MediaPipe version: {version}")
        
        # Check if version is compatible
        if version.startswith('0.8') or version.startswith('0.9'):
            print("MediaPipe version should be compatible")
        else:
            print(f"Warning: MediaPipe version {version} might have compatibility issues")
            print("Consider updating to a stable version: pip install mediapipe==0.10.0")
        
        return True
    except Exception as e:
        print(f"Error checking MediaPipe version: {e}")
        return False

def initialize_mediapipe():
    """Initialize MediaPipe components with error handling"""
    # Make sure these are imported somewhere:
    # import mediapipe as mp
    # mp_hands = mp.solutions.hands
    # mp_face_mesh = mp.solutions.face_mesh
    # mp_drawing = mp.solutions.drawing_utils
    # mp_styles = mp.solutions.drawing_styles

    check_mediapipe_version()

    try:
        # Hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        # Face mesh
        face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        return hands, face_mesh, mp_drawing, mp_styles

    except Exception as e:
        print(f"Error initializing MediaPipe: {e}")
        print("Trying to initialize without face mesh...")

        try:
            hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7
            )
            return hands, None, mp_drawing, None

        except Exception as e2:
            print(f"Failed to initialize even basic MediaPipe: {e2}")
            return None, None, None, None


def count_fingers(hand_landmarks, frame_shape, handedness=None):
    """Count extended fingers and return the count with improved accuracy"""
    h, w, _ = frame_shape
    
    # Define finger tip and pip (middle joint) indices for each finger
    finger_tips = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky tips
    finger_pips = [3, 6, 10, 14, 18]  # thumb, index, middle, ring, pinky middle joints
    
    extended_fingers = 0
    finger_states = []  # Track which fingers are extended
    
    for i in range(len(finger_tips)):
        tip = hand_landmarks.landmark[finger_tips[i]]
        pip = hand_landmarks.landmark[finger_pips[i]]
        
        # For thumb, check horizontal position based on handedness with improved accuracy
        if i == 0:  # thumb
            if handedness == "Left":
                # For left hand, thumb is extended when tip.x > pip.x
                # Add some tolerance for better detection
                if tip.x > pip.x - 0.02:  # thumb extended with tolerance
                    extended_fingers += 1
                    finger_states.append(True)
                else:
                    finger_states.append(False)
            else:
                # For right hand, thumb is extended when tip.x < pip.x
                # Add some tolerance for better detection
                if tip.x < pip.x + 0.02:  # thumb extended with tolerance
                    extended_fingers += 1
                    finger_states.append(True)
                else:
                    finger_states.append(False)
        else:  # other fingers, check vertical position with improved accuracy
            # Add tolerance for better finger detection
            if tip.y < pip.y - 0.01:  # finger extended with tolerance
                extended_fingers += 1
                finger_states.append(True)
            else:
                finger_states.append(False)
    
    return extended_fingers, finger_states

def detect_gesture(hand_landmarks, frame_shape, handedness=None, hand_idx=0):
    """Detect specific gestures and return gesture name"""
    h, w, _ = frame_shape
    
    # Get finger states
    finger_count, finger_states = count_fingers(hand_landmarks, frame_shape, handedness)
    
    # Get hand position (center of palm)
    palm_center = hand_landmarks.landmark[9]  # Middle finger MCP
    palm_x, palm_y = int(palm_center.x * w), int(palm_center.y * h)
    
    # Enhanced gesture detection logic with better accuracy
    if finger_count == 5:  # All fingers extended - waving gesture
        return "HI"
    elif finger_count == 2 and finger_states[1] and finger_states[4]:  # Index and pinky finger - yo sign
        return "YOYO"
    elif finger_count == 2 and finger_states[1] and finger_states[2]:  # Index and middle finger - peace sign
        return "PEACE"
    elif finger_count == 3 and finger_states[0] and finger_states[1] and finger_states[4]:  # Thumb, index, and pinky - rock sign
        return "ROCK"
    elif finger_count == 1 and finger_states[0]:  # Only thumb
        return "THUMBS_UP"
    elif finger_count == 1 and finger_states[1]:  # Only index
        return "POINTING"
    elif finger_count == 4 and not finger_states[2]:  # All except middle finger
        return "LOSER"
    elif finger_count == 0:  # Fist
        return "FIST"
    else:
        return None

def detect_thumbs_down(hand_landmarks, frame_shape, handedness=None):
    """Detect thumbs down gesture (thumb pointing down using y-coordinates)"""
    h, w, _ = frame_shape
    
    # Get thumb tip and base positions
    thumb_tip = hand_landmarks.landmark[4]
    thumb_base = hand_landmarks.landmark[2]
    
    # Check if only thumb is extended and pointing down
    finger_count, finger_states = count_fingers(hand_landmarks, frame_shape, handedness)
    
    if finger_count == 1 and finger_states[0]:  # Only thumb extended
        # Check thumb direction: tip below base (y is downward in image coordinates)
        if thumb_tip.y > thumb_base.y + 0.05:  # Add tolerance
            return True
    return False

def detect_namasthe(hand_res, frame_shape):
    """Detect Namasthe gesture: both palms and all fingertips are close together, and most fingers are extended."""
    from collections import Counter
    if not hand_res.multi_hand_landmarks:
        return False
    
    h, w, _ = frame_shape
    
    # Only detect Namasthe with exactly 2 hands
    if len(hand_res.multi_hand_landmarks) == 2:
        hand1 = hand_res.multi_hand_landmarks[0]
        hand2 = hand_res.multi_hand_landmarks[1]
        
        # Get palm centers
        palm1 = hand1.landmark[9]  # Middle finger MCP
        palm2 = hand2.landmark[9]
        x1, y1 = int(palm1.x * w), int(palm1.y * h)
        x2, y2 = int(palm2.x * w), int(palm2.y * h)
        palm_dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        
        # Get fingertip distances
        tip_ids = [8, 12, 16, 20]  # Index, middle, ring, pinky
        tip_dists = []
        for tid in tip_ids:
            t1 = hand1.landmark[tid]
            t2 = hand2.landmark[tid]
            tx1, ty1 = int(t1.x * w), int(t1.y * h)
            tx2, ty2 = int(t2.x * w), int(t2.y * h)
            tip_dists.append(math.sqrt((tx1 - tx2) ** 2 + (ty1 - ty2) ** 2))
        avg_tip_dist = sum(tip_dists) / len(tip_dists)
        
        # Check finger extension
        def extended_count(hand):
            _, states = count_fingers(hand, frame_shape)
            return Counter(states)[True]
        
        ext1 = extended_count(hand1)
        ext2 = extended_count(hand2)
        
        # Enhanced Namasthe detection criteria:
        # 1. Palms should be close (within 80 pixels)
        # 2. Average fingertip distance should be small (within 50 pixels)
        # 3. At least 3 fingers extended on both hands
        # 4. Hands should be roughly at the same height (vertical alignment)
        height_diff = abs(y1 - y2)
        
        if (palm_dist < 80 and 
            avg_tip_dist < 50 and 
            ext1 >= 3 and ext2 >= 3 and
            height_diff < 100):  # Hands should be at similar height
            return 'NAMASTHE'
    
    return False

def detect_hands_near_face(hand_res, face_res, frame_shape):
    """Detect if both hands are near the face (for thank you gesture)"""
    if not hand_res.multi_hand_landmarks or len(hand_res.multi_hand_landmarks) < 2:
        return False
    
    if not face_res.multi_face_landmarks:
        return False
    
    h, w, _ = frame_shape
    
    # Get face center
    face_landmarks = face_res.multi_face_landmarks[0]
    face_center = face_landmarks.landmark[10]  # Nose tip
    face_x, face_y = int(face_center.x * w), int(face_center.y * h)
    
    # Check if both hands are near face
    hands_near_face = 0
    for hand_landmarks in hand_res.multi_hand_landmarks:
        palm_center = hand_landmarks.landmark[9]  # Middle finger MCP
        palm_x, palm_y = int(palm_center.x * w), int(palm_center.y * h)
        
        # Calculate distance to face
        distance = math.sqrt((palm_x - face_x)**2 + (palm_y - face_y)**2)
        
        # If hand is within 200 pixels of face
        if distance < 200:
            hands_near_face += 1
    
    return hands_near_face >= 2

def draw_enhanced_face_mesh(frame, face_landmarks, mp_drawing, mp_styles):
    """Draw enhanced face mesh with multiple styles and better visualization"""
    h, w, _ = frame.shape
    
    # Draw the main face mesh with enhanced styling
    mp_drawing.draw_landmarks(
        frame, face_landmarks,
        mp_face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1),  # Cyan mesh
        connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style()
    )
    
    # Draw face contours with enhanced styling
    mp_drawing.draw_landmarks(
        frame, face_landmarks,
        mp_face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),  # Red contours
        connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style()
    )
    
    # Draw iris details with enhanced styling
    mp_drawing.draw_landmarks(
        frame, face_landmarks,
        mp_face_mesh.FACEMESH_IRISES,
        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),  # Green irises
        connection_drawing_spec=mp_styles.get_default_face_mesh_iris_connections_style()
    )
    
    # Enhanced face feature highlights
    # Eyes with better positioning
    left_eye_center = face_landmarks.landmark[33]  # Left eye center
    right_eye_center = face_landmarks.landmark[263]  # Right eye center
    
    left_eye_x, left_eye_y = int(left_eye_center.x * w), int(left_eye_center.y * h)
    right_eye_x, right_eye_y = int(right_eye_center.x * w), int(right_eye_center.y * h)
    
    # Draw enhanced eye highlights with glow effect
    cv2.circle(frame, (left_eye_x, left_eye_y), 5, (255, 255, 255), -1)  # White center
    cv2.circle(frame, (left_eye_x, left_eye_y), 8, (255, 255, 255), 2)   # White glow
    cv2.circle(frame, (right_eye_x, right_eye_y), 5, (255, 255, 255), -1)
    cv2.circle(frame, (right_eye_x, right_eye_y), 8, (255, 255, 255), 2)
    
    # Enhanced nose tip
    nose_tip = face_landmarks.landmark[4]
    nose_x, nose_y = int(nose_tip.x * w), int(nose_tip.y * h)
    cv2.circle(frame, (nose_x, nose_y), 4, (255, 255, 0), -1)  # Yellow nose tip
    cv2.circle(frame, (nose_x, nose_y), 7, (255, 255, 0), 2)   # Yellow glow
    
    # Enhanced mouth corners
    left_mouth = face_landmarks.landmark[61]
    right_mouth = face_landmarks.landmark[291]
    
    left_mouth_x, left_mouth_y = int(left_mouth.x * w), int(left_mouth.y * h)
    right_mouth_x, right_mouth_y = int(right_mouth.x * w), int(right_mouth.y * h)
    
    cv2.circle(frame, (left_mouth_x, left_mouth_y), 3, (255, 0, 255), -1)  # Magenta mouth corners
    cv2.circle(frame, (left_mouth_x, left_mouth_y), 6, (255, 0, 255), 2)   # Magenta glow
    cv2.circle(frame, (right_mouth_x, right_mouth_y), 3, (255, 0, 255), -1)
    cv2.circle(frame, (right_mouth_x, right_mouth_y), 6, (255, 0, 255), 2)
    
    # Add eyebrow highlights
    left_eyebrow = face_landmarks.landmark[70]  # Left eyebrow center
    right_eyebrow = face_landmarks.landmark[300]  # Right eyebrow center
    
    left_eyebrow_x, left_eyebrow_y = int(left_eyebrow.x * w), int(left_eyebrow.y * h)
    right_eyebrow_x, right_eyebrow_y = int(right_eyebrow.x * w), int(right_eyebrow.y * h)
    
    cv2.circle(frame, (left_eyebrow_x, left_eyebrow_y), 2, (0, 255, 255), -1)  # Cyan eyebrow
    cv2.circle(frame, (right_eyebrow_x, right_eyebrow_y), 2, (0, 255, 255), -1)
    
    # Add face mesh status indicator
    cv2.putText(frame, "Face Mesh Active", (w - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, (0, 255, 255), 2)

import cv2

def draw_enhanced_hand_mesh(
    frame,
    hand_landmarks,
    mp_drawing,
    mp_hands,
    hand_idx: int = 0,
    handedness: str = None
):
    """
    Draws an enhanced hand mesh with color-coded fingers, palm connections,
    and optional handedness labels.

    Args:
        frame (np.ndarray): The BGR image frame (OpenCV).
        hand_landmarks: MediaPipe Hand landmarks object.
        mp_drawing: MediaPipe drawing utilities.
        mp_hands: MediaPipe hands module (for connections).
        hand_idx (int, optional): Index of the current hand (0 or 1). Defaults to 0.
        handedness (str, optional): 'Left' or 'Right'. Defaults to None.
    """
    h, w, _ = frame.shape

    # Colors for each hand (Blue for first hand, Green for second)
    hand_colors = [(255, 0, 0), (0, 255, 0)]
    main_color = hand_colors[hand_idx % len(hand_colors)]

    # Draw basic MediaPipe connections first
    mp_drawing.draw_landmarks(
        frame,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=main_color, thickness=2, circle_radius=3),
        mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2)
    )

    # Finger colors and tips
    finger_colors = [
        (255, 0, 0),   # Thumb
        (0, 255, 0),   # Index
        (0, 0, 255),   # Middle
        (255, 255, 0), # Ring
        (255, 0, 255)  # Pinky
    ]
    finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
    finger_tips = [4, 8, 12, 16, 20]

    for idx, tip_id in enumerate(finger_tips):
        color = finger_colors[idx]

        # Define points for each finger
        if idx == 0:  # Thumb
            finger_points = [4, 3, 2, 1, 0]
        else:
            # Index: [8,7,6,5], Middle: [12,11,10,9], etc.
            base_id = (idx * 4) + 1
            finger_points = [tip_id, tip_id - 1, tip_id - 2, tip_id - 3, base_id]

        # Draw lines & points for this finger
        for i in range(len(finger_points) - 1):
            pt1 = hand_landmarks.landmark[finger_points[i]]
            pt2 = hand_landmarks.landmark[finger_points[i + 1]]

            x1, y1 = int(pt1.x * w), int(pt1.y * h)
            x2, y2 = int(pt2.x * w), int(pt2.y * h)

            cv2.line(frame, (x1, y1), (x2, y2), color, 3)
            cv2.circle(frame, (x1, y1), 4, color, -1)

        # Label finger tip
        tip = hand_landmarks.landmark[tip_id]
        tip_x, tip_y = int(tip.x * w), int(tip.y * h)
        cv2.putText(frame, finger_names[idx], (tip_x + 8, tip_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Palm connections
    palm_points = [0, 5, 9, 13, 17, 0]
    for i in range(len(palm_points) - 1):
        pt1 = hand_landmarks.landmark[palm_points[i]]
        pt2 = hand_landmarks.landmark[palm_points[i + 1]]
        x1, y1 = int(pt1.x * w), int(pt1.y * h)
        x2, y2 = int(pt2.x * w), int(pt2.y * h)
        cv2.line(frame, (x1, y1), (x2, y2), main_color, 2)

    # Display handedness label (Left/Right)
    if handedness:
        wrist = hand_landmarks.landmark[0]
        wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)
        cv2.putText(frame, handedness.upper(), (wrist_x - 20, wrist_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, main_color, 2)

def main():
    # Check dependencies first
    check_dependencies()

    # Check camera availability
    if not check_camera():
        return

    # Initialize MediaPipe
    hands, face_mesh, mp_drawing, mp_styles = initialize_mediapipe()
    if hands is None:
        return

    cap = None
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Cannot open camera")
            return

        print("Enhanced Dual Hand Tracking with Advanced Gestures Started!")
        print("Press 'q' to quit")
        print("Gestures:")
        print("- Wave (all fingers): HI")
        print("- Index + Pinky: YOYO")
        print("- Peace sign (index + middle): PEACE")
        print("- Rock on (thumb + index + pinky): ROCK")
        print("- Thumbs up (thumb only): THUMBS_UP")
        print("- Thumbs down (thumb pointing down): THUMBS_DOWN")
        print("- Pointing (index only): POINTING")
        print("- Loser sign (all except middle): LOSER")
        print("- Fist (no fingers): FIST")
        print("- Both hands near face: THANK YOU")
        print("- Both palms together: NAMASTHE")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Cannot read frame from camera")
                break

            frame = cv2.flip(frame, 1)  # Flip horizontally
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # --- Enhanced Face mesh ---
            face_res = None
            try:
                if face_mesh is not None:
                    face_res = face_mesh.process(rgb)
                    if face_res.multi_face_landmarks:
                        for face_landmarks in face_res.multi_face_landmarks:
                            draw_enhanced_face_mesh(frame, face_landmarks, mp_drawing, mp_styles)
            except Exception as e:
                print(f"Face mesh error: {e}")

            # --- Hand detection ---
            try:
                hand_res = hands.process(rgb)
                total_fingers = 0
                hands_detected = 0
                gestures_detected = []

                if hand_res.multi_hand_landmarks:
                    hands_detected = len(hand_res.multi_hand_landmarks)

                    for hand_idx, hand_landmarks in enumerate(hand_res.multi_hand_landmarks):
                        handedness = None
                        if hand_res.multi_handedness:
                            handedness = hand_res.multi_handedness[hand_idx].classification[0].label

                        # Draw hand mesh
                        draw_enhanced_hand_mesh(frame, hand_landmarks, mp_drawing, mp_hands, hand_idx, handedness)

                        # Count fingers
                        finger_count, finger_states = count_fingers(hand_landmarks, frame.shape, handedness)
                        total_fingers += finger_count

                        # Detect gesture
                        gesture = detect_gesture(hand_landmarks, frame.shape, handedness, hand_idx)
                        if detect_thumbs_down(hand_landmarks, frame.shape, handedness):
                            gesture = "THUMBS_DOWN"

                        if gesture:
                            gestures_detected.append(gesture)

                        # Display hand info
                        hand_label = f"Hand {hand_idx + 1} ({handedness})" if handedness else f"Hand {hand_idx + 1}"
                        hand_text = f"{hand_label}: {finger_count} fingers"
                        y_position = 30 + (hand_idx * 40)
                        color = (255, 0, 0) if hand_idx == 0 else (0, 255, 0)
                        cv2.putText(frame, hand_text, (10, y_position),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                        if gesture:
                            gesture_text = f"Hand {hand_idx + 1}: {gesture}"
                            cv2.putText(frame, gesture_text, (10, y_position + 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                    # Hands near face
                    if face_res and face_res.multi_face_landmarks:
                        if detect_hands_near_face(hand_res, face_res, frame.shape):
                            gestures_detected.append("THANK_YOU")
                            cv2.putText(frame, "THANK YOU - Hands near face", (10, h - 110),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

                    # Namasthe gesture
                    if detect_namasthe(hand_res, frame.shape):
                        gestures_detected.append("NAMASTHE")
                        cv2.putText(frame, "NAMASTHE - Palms together", (10, h - 140),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 3)

                    # Display totals
                    total_text = f"Total: {hands_detected} hands, {total_fingers} fingers"
                    cv2.putText(frame, total_text, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                    if total_fingers > 0:
                        color = (0, 255, 0) if total_fingers <= 5 else (0, 255, 255) if total_fingers <= 10 else (0, 0, 255)
                        cv2.putText(frame, f"Total Fingers: {total_fingers}", (10, h - 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

                    if gestures_detected:
                        gesture_display = " | ".join(gestures_detected)
                        cv2.putText(frame, f"Gestures: {gesture_display}", (10, h - 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
                else:
                    cv2.putText(frame, "No hands detected", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            except Exception as e:
                print(f"Hand detection error: {e}")

            cv2.imshow("Enhanced Dual Hand Tracking with Advanced Gestures", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        print("Enhanced Dual Hand Tracking with Advanced Gestures Stopped")


if __name__ == "__main__":
    main()
