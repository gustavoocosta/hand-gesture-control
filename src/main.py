import cv2


def main():
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Erro: não foi possível abrir a webcam.")
        return

    print("Webcam iniciada")
    print("Pressione Q para sair")

    while True:
        success, frame = camera.read()

        if not success:
            print("Erro ao capturar imagem da webcam.")
            break

        # Espelha a imagem como uma câmera frontal.
        frame = cv2.flip(frame, 1)

        cv2.putText(
            frame,
            "Hand Gesture Control",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Hand Gesture Control", frame)

        # Pressione Q para fechar.
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()