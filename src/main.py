
import time
from pathlib import Path

import cv2
import mediapipe as mp


MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "hand_landmarker.task"
)




import math


def fingers_up(hand):
    """Identifica a extensão dos dedos usando os pontos da mão."""

    tips = [8, 12, 16, 20]
    joints = [6, 10, 14, 18]

    fingers = [
        hand[tip].y < hand[joint].y
        for tip, joint in zip(tips, joints)
    ]

    # Usa a distância entre os pontos do polegar,
    # em vez de depender somente da coordenada X.
    def distance(a, b):
        return math.hypot(
            hand[a].x - hand[b].x,
            hand[a].y - hand[b].y,
        )

    thumb_extended = (
        distance(4, 2) > distance(3, 2) * 1.2
    )

    return [thumb_extended] + fingers




def recognize_gesture(hand):
    thumb, index, middle, ring, pinky = fingers_up(hand)

    # Polegar apontando para cima e demais dedos dobrados.
    thumb_pointing_up = (
        hand[4].y < hand[3].y
        and hand[3].y < hand[2].y
    )

    other_fingers_folded = not any(
        [index, middle, ring, pinky]
    )

    if thumb and other_fingers_folded and thumb_pointing_up:
        return "Polegar para cima"

    if all([thumb, index, middle, ring, pinky]):
        return "Mao aberta"

    if not any([thumb, index, middle, ring, pinky]):
        return "Punho fechado"

    if index and middle and not ring and not pinky:
        return "Sinal de paz"

    return "Gesto nao identificado"



def draw_hand(frame, landmarks):
    height, width, _ = frame.shape
    points = [
        (int(point.x * width), int(point.y * height))
        for point in landmarks
    ]

    connections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS

    for connection in connections:
        start = connection.start
        end = connection.end
        cv2.line(
            frame,
            points[start],
            points[end],
            (0, 255, 0),
            2,
        )

    for x, y in points:
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)


def main():
    if not MODEL_PATH.exists():
        print(f"Modelo nao encontrado: {MODEL_PATH}")
        return

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
    )

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Erro: nao foi possivel abrir a webcam.")
        return

    start_time = time.monotonic()
    print("Reconhecimento de gestos iniciado. Pressione Q para sair.")

    try:
        with HandLandmarker.create_from_options(options) as landmarker:
            while True:
                success, frame = camera.read()

                if not success:
                    print("Erro ao capturar imagem.")
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb,
                )

                timestamp = int(
                    (time.monotonic() - start_time) * 1000
                )

                result = landmarker.detect_for_video(image, timestamp)

                if result.hand_landmarks:
                    for hand in result.hand_landmarks:
                        draw_hand(frame, hand)
                        gesture = recognize_gesture(hand)

                        cv2.putText(
                            frame,
                            gesture,
                            (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2,
                        )
                else:
                    cv2.putText(
                        frame,
                        "Nenhuma mao detectada",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                    )

                cv2.imshow("Hand Gesture Control", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
