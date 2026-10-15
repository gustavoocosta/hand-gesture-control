# Hand Gesture Control

Sistema de controle do cursor do mouse por meio de gestos das mãos, utilizando Python, OpenCV, MediaPipe e PyAutoGUI. O projeto utiliza a webcam para detectar os movimentos da mão em tempo real, reconhecer gestos e permitir a movimentação do cursor e a execução de cliques sem a necessidade de utilizar um mouse físico.

## Sobre o projeto

O Hand Gesture Control foi desenvolvido com o objetivo de explorar visão computacional, reconhecimento de gestos e automação da interação com o computador. A aplicação processa as imagens capturadas pela webcam, identifica os pontos de referência da mão e utiliza a posição do dedo indicador para controlar o cursor.

O projeto também identifica o movimento de pinça entre o polegar e o indicador para executar cliques do mouse.

## Funcionalidades

- Captura de imagens em tempo real por meio da webcam.
- Detecção e rastreamento de mãos utilizando MediaPipe.
- Identificação dos pontos de referência da mão.
- Reconhecimento de gestos básicos.
- Controle da posição do cursor por meio do dedo indicador.
- Execução de cliques com o movimento de pinça entre o polegar e o indicador.
- Suavização do movimento do cursor para melhorar a experiência de uso.
- Exibição dos pontos da mão e das informações de reconhecimento na janela da webcam.

## Tecnologias utilizadas

- **Python:** linguagem principal do projeto.
- **OpenCV:** captura, processamento e exibição de imagens.
- **MediaPipe:** detecção da mão e identificação dos pontos de referência.
- **NumPy:** suporte ao processamento de dados e imagens por meio das dependências do projeto.
- **PyAutoGUI:** movimentação do cursor e execução de cliques.
- **Git e GitHub:** versionamento e gerenciamento do código-fonte.

## Pré-requisitos

Antes de executar o projeto, verifique se você possui:

- Python instalado.
- Git instalado.
- Webcam funcional.
- VS Code ou outro editor de código de sua preferência.

O projeto foi desenvolvido e testado em ambiente Windows. O acesso à webcam precisa estar habilitado para o aplicativo utilizado.

## Estrutura do projeto

```text
hand-gesture-control/
├── models/
│   └── hand_landmarker.task
├── src/
│   └── main.py
├── .gitignore
├── README.md
└── requirements.txt
```

O arquivo `hand_landmarker.task` contém o modelo utilizado pelo MediaPipe para detectar e identificar os pontos de referência da mão.

## Instalação

### 1. Clone o repositório

```bash
git clone URL_DO_REPOSITORIO
```

Substitua `URL_DO_REPOSITORIO` pela URL real do repositório no GitHub.

### 2. Acesse a pasta do projeto

```bash
cd hand-gesture-control
```

### 3. Crie um ambiente virtual

```bash
python -m venv .venv
```

### 4. Ative o ambiente virtual

No Windows, utilizando o Git Bash:

```bash
source .venv/Scripts/activate
```

### 5. Instale as dependências

```bash
python -m pip install -r requirements.txt
```

### 6. Verifique o modelo

Confirme que o arquivo abaixo existe:

```text
models/hand_landmarker.task
```

Se o arquivo não estiver disponível no repositório, será necessário obtê-lo e colocá-lo nesse caminho antes de executar a aplicação.

## Como executar

Com o ambiente virtual ativado, execute:

```bash
python src/main.py
```

A aplicação abrirá uma janela com a imagem da webcam.

### Como utilizar

1. Posicione a mão diante da webcam.
2. Mova o dedo indicador para controlar o cursor.
3. Aproxime o polegar do indicador para formar uma pinça.
4. Afaste os dedos para concluir o gesto de clique.
5. Pressione a tecla `Q` na janela da webcam para encerrar a aplicação.

O controle depende da detecção adequada da mão, da iluminação e do posicionamento diante da câmera.

## Gestos reconhecidos

A aplicação contém lógica para identificar alguns gestos básicos:

| Gesto | Descrição |
|---|---|
| Polegar para cima | Identificado pela posição do polegar e dos demais dedos. |
| Mão aberta | Identificada quando os dedos estão estendidos. |
| Punho fechado | Identificado quando os dedos estão recolhidos. |
| Sinal de paz | Identificado pela extensão dos dedos indicador e médio. |
| Gesto não identificado | Exibido quando a configuração da mão não corresponde às regras implementadas. |

O controle do cursor utiliza o dedo indicador, enquanto o clique é acionado pelo movimento de pinça entre o polegar e o indicador.

## Segurança e limitações

- A aplicação precisa de acesso à webcam para funcionar.
- O desempenho depende da iluminação, da posição da mão e da qualidade da câmera.
- Os gestos são identificados por regras geométricas e podem exigir ajustes conforme as condições de uso.
- O controle do mouse pode interagir com outros aplicativos e elementos da área de trabalho.
- O PyAutoGUI possui um mecanismo de segurança que pode interromper a execução quando o cursor é levado ao canto superior esquerdo da tela.

Recomenda-se testar o sistema em um ambiente controlado antes de utilizá-lo em tarefas importantes.

## Possíveis melhorias

- Implementar um mecanismo para ativar e desativar o controle por gestos.
- Adicionar gestos para clique duplo, rolagem e arrastar e soltar.
- Melhorar a calibração da área de movimentação do cursor.
- Ajustar a suavização e a sensibilidade dos movimentos.
- Criar testes automatizados para as funções de reconhecimento de gestos.
- Separar a captura de vídeo, o reconhecimento de gestos e o controle do mouse em módulos independentes.
- Desenvolver uma interface gráfica para configuração dos comandos.
- Adicionar demonstrações em vídeo ou GIF ao README.

## Aprendizados

Este projeto explora conceitos de:

- Visão computacional.
- Processamento de imagens em tempo real.
- Detecção de pontos de referência da mão.
- Reconhecimento de gestos.
- Automação de mouse.
- Integração de bibliotecas Python.
- Controle de versão com Git.

## Autor

**Gustavo Costa**

- GitHub: [gustavocosta](https://github.com/gustavocosta)
- Portfólio: [gustavocosta.github.io](https://gustavocosta.github.io)
- LinkedIn: [linkedin.com/in/gustavocosta](https://www.linkedin.com/in/gustavocosta/)

## Licença

Este projeto ainda não possui uma licença definida. A licença poderá ser adicionada futuramente conforme a intenção de distribuição e reutilização do código.