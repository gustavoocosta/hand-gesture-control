
import time
from pathlib import Path

import cv2
import mediapipe as mp


MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]


def draw_hand(frame, landmarks):
    height, width, _ = frame.shape
    points = []

    for landmark in landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        points.append((x, y))

    for start, end in HAND_CONNECTIONS:
        cv2.line(
            frame,
            points[start],
            points[end],
            (0, 255, 0),
            2,
        )

    for x, y in points:
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)


def main():
    if not MODEL_PATH.exists():
        print(f"Modelo não encontrado: {MODEL_PATH}")
        return

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
    )

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Erro: não foi possível abrir a webcam.")
        return

    print("Detecção de mãos iniciada.")
    print("Pressione Q para sair.")

    start_time = time.monotonic()

    try:
        with HandLandmarker.create_from_options(options) as landmarker:
            while True:
                success, frame = camera.read()

                if not success:
                    print("Erro ao capturar imagem da webcam.")
                    break

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame,
                )

                timestamp_ms = int(
                    (time.monotonic() - start_time) * 1000
                )

                result = landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms,
                )

                if result.hand_landmarks:
                    for landmarks in result.hand_landmarks:
                        draw_hand(frame, landmarks)

                    message = (
                        f"Maos detectadas: {len(result.hand_landmarks)}"
                    )
                else:
                    message = "Nenhuma mao detectada"

                cv2.putText(
                    frame,
                    message,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
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
