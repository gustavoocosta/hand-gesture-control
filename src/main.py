
import math
import time
from pathlib import Path

import cv2
import mediapipe as mp
import pyautogui

from libras import ReconhecedorLibras


MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "hand_landmarker.task"
)

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.01

SMOOTHING = 0.20

# Configuracoes da pinca
PINCH_START_DISTANCE = 0.035
PINCH_RELEASE_DISTANCE = 0.055
PINCH_HOLD_SECONDS = 0.12
CLICK_COOLDOWN = 0.5
PINCH_RELEASE_FRAMES = 3


def distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def fingers_up(hand):
    tips = [8, 12, 16, 20]
    joints = [6, 10, 14, 18]

    fingers = [
        hand[tip].y < hand[joint].y
        for tip, joint in zip(tips, joints)
    ]

    thumb_extended = (
        distance(hand[4], hand[2])
        > distance(hand[3], hand[2]) * 1.2
    )

    return [thumb_extended] + fingers


def recognize_gesture(hand):
    thumb, index, middle, ring, pinky = fingers_up(hand)

    thumb_pointing_up = (
        hand[4].y < hand[3].y
        and hand[3].y < hand[2].y
    )

    if (
        thumb
        and not any([index, middle, ring, pinky])
        and thumb_pointing_up
    ):
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

    connections = (
        mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
    )

    for connection in connections:
        cv2.line(
            frame,
            points[connection.start],
            points[connection.end],
            (0, 255, 0),
            2,
        )

    for x, y in points:
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)


def main():
    if not MODEL_PATH.exists():
        print(f"Modelo nao encontrado: {MODEL_PATH}")
        return

    screen_width, screen_height = pyautogui.size()

    previous_x = screen_width / 2
    previous_y = screen_height / 2

    pinching = False
    pinch_started_at = None
    last_click_time = 0.0
    pinch_release_frames = 0

    reconhecedor = ReconhecedorLibras()

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,
    )

    camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)

    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    camera.set(cv2.CAP_PROP_FPS, 30)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print(
        "Resolucao da webcam:",
        int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "x",
        int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )

    if not camera.isOpened():
        print("Erro: nao foi possivel abrir a webcam.")
        return

    start_time = time.monotonic()

    print("Controle do mouse iniciado.")
    print("Mova o indicador para controlar o cursor.")
    print("Junte e segure a pinca; solte para clicar.")
    print("Numeros de 0 a 5: reconhecimento experimental.")
    print("Pressione Q na janela da webcam para sair.")
    print("CUIDADO: o canto superior esquerdo ativa o failsafe.")

    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(
            options
        ) as landmarker:
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

                result = landmarker.detect_for_video(
                    image,
                    timestamp,
                )

                if result.hand_landmarks:
                    hand = result.hand_landmarks[0]
                    draw_hand(frame, hand)

                    configuracao, movimento = (
                        reconhecedor.atualizar(hand)
                    )

                    index_tip = hand[8]
                    thumb_tip = hand[4]

                    target_x = (
                        max(0, min(1, 1 - index_tip.x))
                        * (screen_width - 1)
                    )
                    target_y = (
                        max(0, min(1, index_tip.y))
                        * (screen_height - 1)
                    )

                    current_x = (
                        previous_x
                        + (target_x - previous_x) * SMOOTHING
                    )
                    current_y = (
                        previous_y
                        + (target_y - previous_y) * SMOOTHING
                    )

                    pyautogui.moveTo(
                        int(current_x),
                        int(current_y),
                    )

                    previous_x = current_x
                    previous_y = current_y

                    # Detecta a aproximacao e a soltura da pinca.
                    now = time.monotonic()
                    pinch_distance = distance(
                        thumb_tip,
                        index_tip,
                    )

                    if not pinching:
                        is_pinching = (
                            pinch_distance < PINCH_START_DISTANCE
                        )

                        if is_pinching:
                            pinch_started_at = now
                            pinch_release_frames = 0
                            pinching = True

                    else:
                        if pinch_distance >= PINCH_RELEASE_DISTANCE:
                            pinch_release_frames += 1
                        else:
                            pinch_release_frames = 0

                        # Exige tres frames consecutivos para confirmar
                        # que a pinca foi solta.
                        if (
                            pinch_release_frames
                            >= PINCH_RELEASE_FRAMES
                        ):
                            if (
                                pinch_started_at is not None
                                and now - pinch_started_at
                                >= PINCH_HOLD_SECONDS
                                and now - last_click_time
                                >= CLICK_COOLDOWN
                            ):
                                pyautogui.click()
                                last_click_time = now

                            pinching = False
                            pinch_started_at = None
                            pinch_release_frames = 0

                    if pinching:
                        cv2.putText(
                            frame,
                            "Pinca detectada",
                            (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 255),
                            2,
                        )

                    gesture = recognize_gesture(hand)

                    cv2.putText(
                        frame,
                        f"Gesto: {gesture}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

                    cv2.putText(
                        frame,
                        "Indicador: mover | Pinca: clicar",
                        (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        2,
                    )

                    cv2.putText(
                        frame,
                        f"Numero/Gesto: {configuracao}",
                        (20, 160),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 255),
                        2,
                    )

                    cv2.putText(
                        frame,
                        f"Movimento: {movimento}",
                        (20, 195),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 200, 0),
                        2,
                    )

                else:
                    pinching = False
                    pinch_started_at = None
                    pinch_release_frames = 0

                    cv2.putText(
                        frame,
                        "Nenhuma mao detectada",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )

                cv2.namedWindow(
                    "Hand Gesture Control",
                    cv2.WINDOW_NORMAL,
                )
                cv2.resizeWindow(
                    "Hand Gesture Control",
                    1280,
                    720,
                )
                cv2.imshow("Hand Gesture Control", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
