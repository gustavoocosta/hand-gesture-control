
import cv2
import mediapipe as mp

from libras import ReconhecedorLibras


CAMERA_INDEX = 1
MODEL_PATH = "models/hand_landmarker.task"
WINDOW_NAME = "Teste de Numeros"


def desenhar_mao(frame, mao):
    altura, largura, _ = frame.shape

    conexoes = (
        mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
    )

    for conexao in conexoes:
        inicio = mao[conexao.start]
        fim = mao[conexao.end]

        x1, y1 = int(inicio.x * largura), int(inicio.y * altura)
        x2, y2 = int(fim.x * largura), int(fim.y * altura)

        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    for ponto in mao:
        x = int(ponto.x * largura)
        y = int(ponto.y * altura)
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)


def main():
    camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not camera.isOpened():
        print("Nao foi possivel abrir a webcam.")
        return

    opcoes = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=MODEL_PATH
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,
    )

    reconhecedor = ReconhecedorLibras()
    timestamp = 0

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(
            opcoes
        ) as landmarker:
            print("Teste iniciado. Pressione Q para sair.")

            while True:
                sucesso, frame = camera.read()

                if not sucesso:
                    print("Erro ao capturar imagem.")
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                imagem = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb,
                )

                timestamp += 1
                resultado = landmarker.detect_for_video(
                    imagem, timestamp
                )

                if resultado.hand_landmarks:
                    mao = resultado.hand_landmarks[0]

                    dedos = reconhecedor.dedos_estendidos(mao)
                    numero = reconhecedor.reconhecer_configuracao(mao)

                    nomes = [
                        "Polegar", "Indicador", "Medio",
                        "Anelar", "Minimo"
                    ]

                    estado = " | ".join(
                        f"{nome}: {'ABERTO' if aberto else 'FECHADO'}"
                        for nome, aberto in zip(nomes, dedos)
                    )

                    desenhar_mao(frame, mao)

                    cv2.putText(
                        frame, numero, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0), 2
                    )

                    cv2.putText(
                        frame, estado, (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                        (0, 255, 255), 2
                    )
                else:
                    cv2.putText(
                        frame, "Nenhuma mao detectada", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 0, 255), 2
                    )

                cv2.imshow(WINDOW_NAME, frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
