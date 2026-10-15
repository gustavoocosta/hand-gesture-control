
from collections import deque
import time


class ReconhecedorLibras:
    def __init__(self, duracao_historico=1.5):
        self.historico = deque()
        self.duracao_historico = duracao_historico

    def atualizar(self, hand):
        agora = time.monotonic()

        # Guarda a posição normalizada do pulso
        pulso = hand[0]
        self.historico.append(
            (agora, pulso.x, pulso.y)
        )

        # Remove posições antigas
        while (
            self.historico
            and agora - self.historico[0][0]
            > self.duracao_historico
        ):
            self.historico.popleft()

        return self.analisar_movimento()

    def analisar_movimento(self):
        if len(self.historico) < 5:
            return "Coletando movimento..."

        _, x_inicial, y_inicial = self.historico[0]
        _, x_atual, y_atual = self.historico[-1]

        deslocamento_x = x_atual - x_inicial
        deslocamento_y = y_atual - y_inicial

        limite = 0.12

        if abs(deslocamento_x) > limite:
            return (
                "Movimento horizontal para a direita"
                if deslocamento_x > 0
                else "Movimento horizontal para a esquerda"
            )

        if abs(deslocamento_y) > limite:
            return (
                "Movimento para baixo"
                if deslocamento_y > 0
                else "Movimento para cima"
            )

        return "Sem movimento significativo"
