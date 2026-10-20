
import time
from pathlib import Path

import cv2
import mediapipe as mp


MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "gesture_recognizer.task"
)

CAMERA_INDEX = 1
WINDOW_NAME = "Teste de Reconhecimento de Gestos"

TRADUCOES = {
    "Open_Palm": "Mao aberta",
    "Closed_Fist": "Punho fechado",
    "Pointing_Up": "Indicador apontando para cima",
    "Thumb_Up": "Polegar para cima",
    "Thumb_Down": "Polegar para baixo",
    "Victory": "Sinal de paz",
    "ILoveYou": "Gesto I Love You",
    "None": "Gesto nao identificado",
}


def main():
    if not MODEL_PATH.exists():
        print(f"Modelo nao encontrado: {MODEL_PATH}")
        return

    camera = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_DSHOW,
    )

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    camera.set(cv2.CAP_PROP_FPS, 30)

    if not camera.isOpened():
        print(f"Nao foi possivel abrir a camera {CAMERA_INDEX}.")
        return

    options = mp.tasks.vision.GestureRecognizerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,
    )

    inicio = time.monotonic()
    ultimo_timestamp = -1

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    try:
        with mp.tasks.vision.GestureRecognizer.create_from_options(
            options
        ) as recognizer:

            print("Reconhecimento de gestos iniciado.")
            print("Pressione Q na janela para encerrar.")

            while True:
                success, frame = camera.read()

                if not success:
                    print("Erro ao capturar imagem da webcam.")
                    break

                frame = cv2.flip(frame, 1)

                rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB,
                )

                image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb,
                )

                timestamp = int(
                    (time.monotonic() - inicio) * 1000
                )

                # Garante timestamps crescentes para o modo VIDEO.
                timestamp = max(timestamp, ultimo_timestamp + 1)
                ultimo_timestamp = timestamp

                result = recognizer.recognize_for_video(
                    image,
                    timestamp,
                )

                nome_gesto = "Nenhuma mao detectada"
                confianca = 0.0

                if result.gestures:
                    categorias = result.gestures[0]

                    if categorias:
                        categoria = categorias[0]
                        nome_gesto = TRADUCOES.get(
                            categoria.category_name,
                            categoria.category_name,
                        )
                        confianca = categoria.score

                texto = f"Gesto: {nome_gesto}"

                cv2.putText(
                    frame,
                    texto,
                    (25, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )

                if result.gestures and categorias:
                    cv2.putText(
                        frame,
                        f"Confianca: {confianca:.0%}",
                        (25, 85),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                    )

                if result.hand_landmarks:
                    altura, largura, _ = frame.shape

                    for hand in result.hand_landmarks:
                        for ponto in hand:
                            x = int(ponto.x * largura)
                            y = int(ponto.y * altura)

                            cv2.circle(
                                frame,
                                (x, y),
                                4,
                                (0, 0, 255),
                                -1,
                            )

                cv2.imshow(WINDOW_NAME, frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
