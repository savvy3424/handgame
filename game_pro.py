import cv2
import mediapipe as mp
import random
import time
import numpy as np
import os

WIDTH, HEIGHT = 1280, 720
LIFETIME = 8.0  
SCORE = 0
INPUT_MODE = "WEBCAM"

cursor_pos = [WIDTH // 2, HEIGHT // 2]
last_shoot_time = 0

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

def load_resource(name, size, color):
    base_path = os.path.dirname(__file__)
    path = os.path.join(base_path, name)
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                chunk = f.read()
            arr = np.frombuffer(chunk, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
            if img is not None:
                if img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                return cv2.resize(img, (size, size))
        except:
            pass
    img = np.zeros((size, size, 4), dtype="uint8")
    cv2.circle(img, (size//2, size//2), size//2 - 5, color, -1)
    return img

def draw_overlay(bg, overlay, x, y):
    h, w = overlay.shape[:2]
    if y + h > HEIGHT or x + w > WIDTH or y < 0 or x < 0:
        return bg
    alpha = overlay[:, :, 3:] / 255.0
    bg[y:y+h, x:x+w] = (1.0 - alpha) * bg[y:y+h, x:x+w] + alpha * overlay[:, :, :3]
    return bg

def mouse_event(event, x, y, flags, param):
    global cursor_pos
    if INPUT_MODE == "MOUSE":
        cursor_pos[0], cursor_pos[1] = x, y
        if event == cv2.EVENT_LBUTTONDOWN:
            process_shot(x, y)

def process_shot(px, py):
    global SCORE, last_shoot_time
    now = time.time()
    if now - last_shoot_time > 0.3:
        for i, t in enumerate(targets):
            size = 130 if t[2] else 100
            if t[0] < px < t[0] + size and t[1] < py < t[1] + size:
                if t[2]: SCORE += 100
                else: SCORE -= 200
                targets[i] = spawn_target(t[2])
                last_shoot_time = now
                return True
    return False

def spawn_target(is_enemy):
    return [random.randint(50, WIDTH-150), random.randint(50, HEIGHT-150), is_enemy, time.time()]

enemy_img = load_resource('astronaut.png', 130, (0, 0, 255, 255))
ally_img = load_resource('satellite.png', 100, (0, 255, 0, 255))

targets = [spawn_target(True)] + [spawn_target(False) for _ in range(3)]

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cv2.namedWindow('Cyber Hunter')
cv2.setMouseCallback('Cyber Hunter', mouse_event)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (WIDTH, HEIGHT))
    now = time.time()

    cam_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if INPUT_MODE == "WEBCAM":
        results = hands.process(cam_rgb)
        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0].landmark[8]
            cursor_pos[0], cursor_pos[1] = int(lm.x * WIDTH), int(lm.y * HEIGHT)
            process_shot(cursor_pos[0], cursor_pos[1])

    for i in range(len(targets)):
        if now - targets[i][3] > LIFETIME:
            targets[i] = spawn_target(targets[i][2])
        
        t = targets[i]
        img = enemy_img if t[2] else ally_img
        frame = draw_overlay(frame, img, t[0], t[1])
        
        pct = max(0, (LIFETIME - (now - t[3])) / LIFETIME)
        cv2.line(frame, (t[0], t[1]-10), (t[0] + int(100*pct), t[1]-10), (0, 255, 255), 2)

    cv2.circle(frame, (cursor_pos[0], cursor_pos[1]), 22, (0, 255, 136), 2)
    cv2.circle(frame, (cursor_pos[0], cursor_pos[1]), 2, (255, 255, 255), -1)

    # UI Overlay
    cv2.putText(frame, f"SCORE: {SCORE}", (30, 60), 0, 1.3, (0, 255, 136), 2)
    cv2.putText(frame, f"MODE: {INPUT_MODE}", (30, 100), 0, 0.7, (255, 255, 255), 1)
    cv2.putText(frame, "[M] Switch Mode", (30, 130), 0, 0.6, (200, 200, 200), 1)
    cv2.putText(frame, "[ESC] Exit Game", (30, 155), 0, 0.6, (200, 200, 200), 1)
    
    # Author tag (Top Right)
    cv2.putText(frame, "Created by Savvy", (WIDTH - 160, 30), 0, 0.5, (120, 120, 120), 1)

    cv2.imshow('Cyber Hunter', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == 27: break
    elif key == ord('m'):
        INPUT_MODE = "MOUSE" if INPUT_MODE == "WEBCAM" else "WEBCAM"

cap.release()
cv2.destroyAllWindows()