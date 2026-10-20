
from collections import deque, Counter
import time
import math


class ReconhecedorLibras:
    def __init__(self, duracao_historico=1.5):
        self.historico = deque()
        self.duracao_historico = duracao_historico
        self.historico_configuracoes = deque(maxlen=7)

    @staticmethod
    def distancia(a, b):
        return math.hypot(a.x - b.x, a.y - b.y)

    def dedos_estendidos(self, hand):
        """
        Estima quais dedos estao estendidos usando a
        posicao das pontas em relacao as articulacoes.
        Pressupoe a palma aproximadamente voltada para a camera.
        """
        pontas = [8, 12, 16, 20]
        articulacoes = [6, 10, 14, 18]

        dedos = []

        for ponta, articulacao in zip(pontas, articulacoes):
            dedos.append(
                hand[ponta].y < hand[articulacao].y
            )

        # Compara a distancia do polegar em relacao a palma.
        # Funciona melhor com a palma voltada para a camera.
        polegar = (
            self.distancia(hand[4], hand[5])
            > self.distancia(hand[3], hand[5]) * 1.1
        )

        return [polegar] + dedos

    def reconhecer_configuracao(self, hand):
        polegar, indicador, medio, anelar, minimo = (
            self.dedos_estendidos(hand)
        )

        # Reconhece os dedos principais sem depender
        # da quantidade total isoladamente.
        if indicador and not medio and not anelar and not minimo:
            if polegar:
                return "Configuracao nao identificada"
            return "Numero 1"

        if indicador and medio and not anelar and not minimo:
            if not polegar:
                return "Numero 2"

        if indicador and medio and anelar and not minimo:
            if not polegar:
                return "Numero 3"

        if indicador and medio and anelar and minimo:
            if not polegar:
                return "Numero 4"

        if polegar and indicador and medio and anelar and minimo:
            return "Numero 5"

        if not polegar and not indicador and not medio and not anelar and not minimo:
            return "Punho fechado - numero 0"

        return "Configuracao nao identificada"

    def atualizar(self, hand):
        agora = time.monotonic()
        pulso = hand[0]

        self.historico.append((agora, pulso.x, pulso.y))

        while (
            self.historico
            and agora - self.historico[0][0]
            > self.duracao_historico
        ):
            self.historico.popleft()

        configuracao_atual = self.reconhecer_configuracao(hand)
        self.historico_configuracoes.append(configuracao_atual)

        contagem = Counter(self.historico_configuracoes)
        configuracao, quantidade = contagem.most_common(1)[0]

        if (
            quantidade < 4
            or configuracao == "Configuracao nao identificada"
        ):
            configuracao = "Estabilizando reconhecimento..."

        movimento = self.analisar_movimento(hand)

        return configuracao, movimento
    def analisar_movimento(self, hand):
        if len(self.historico) < 5:
            return "Coletando movimento..."

        _, x_inicial, y_inicial = self.historico[0]
        _, x_atual, y_atual = self.historico[-1]

        dx = x_atual - x_inicial
        dy = y_atual - y_inicial
        limite = 0.12

        if abs(dx) > limite:
            return (
                "Movimento horizontal para a direita"
                if dx > 0
                else "Movimento horizontal para a esquerda"
            )

        if abs(dy) > limite:
            return "Movimento para baixo" if dy > 0 else "Movimento para cima"

        return "Sem movimento significativo"
