### Vetorização para criptografia de imagens sensíveis
---

## Grupo:

| Nome | RA |
|------|-----|
| Jhonatan Frossard | RA: 200304 |
| João Victor Athayde Grilo | RA: 210491 |
| Julio Cesar Bonow Manoel | RA: 210375 |
| Rafael Henrique Ramos | RA: 210432 |
| Rafael Rocha Leite | RA: 222469 |
| Rickelme Gabriel Dias | RA: 224276 |

## Objetivos

Este trabalho tem como foco principal implementar, com NumPy, o processamento vetorizado em um sistema de criptografia para imagens sensíveis, de modo a comparar a pipeline de processamento com e sem o uso de vetorização.


## Contextualização

Este projeto surgiu como um ponto de melhoria ao trabalho de conclusão de curso intitulado "Computação Quântica e Criptografia Pós-Quântica Aplicadas à Segurança de Dados Sensíveis".

Nesse contexto, o TCC propõe implementar um sistema de proteção de imagens sensíveis que combina três módulos principais.

1. Geradores Quânticos de Números Aleatórios (QRNG)
2. Sistema de cifragem baseados em operações de XOR e mapa caótico logístico.
3. Encapsulamento de chaves com o algoritmo pós-quântico CRYSTAL-Kyber 512.


<p align="center">
  <b>Figura 1 - Pipeline completa.</b>
</p>

![Figura 1](/imagens_relatorio/Sistema%20de%20Criptografia%20para%20Imagens.png)

<p align="center">
  <b>Fonte: Elaborado pelo autor.</b>
</p>

Dentro desse contexto, os processos que serão convertidos para utilizar vetorização são a aplicação do XOR entre a matriz QRNG e a imagem original e a ordenação dos pixels por meio do mapa logístico caótico.

## Recursos Utilizados

### Bibliotecas Principais

```
numpy>=1.24.0          # Processamento vetorizado e operações matriciais
opencv-python>=4.8.0   # Leitura e manipulação de imagens
matplotlib>=3.7.0      # Visualização de resultados
```

### Arquivo requirements.txt

```txt
numpy>=1.24.0
opencv-python>=4.8.0
matplotlib>=3.7.0
```

## 🛠️ Instalação e Configuração

### Opção 1: Ambiente Conda (Recomendado)

1. **Criar ambiente virtual:**
```bash
conda create -n vector python=3.12
```

2. **Ativar o ambiente:**
```bash
conda activate vector
```

3. **Instalar dependências:**
```bash
pip install -r requirements.txt
```

### Opção 2: Ambiente Python Virtual (venv)

1. **Criar ambiente virtual:**
```bash
python -m venv vector_env
```

2. **Ativar o ambiente:**
   - **Linux/Mac:**
   ```bash
   source vector_env/bin/activate
   ```
   - **Windows:**
   ```bash
   vector_env\Scripts\activate
   ```

3. **Instalar dependências:**
```bash
pip install -r requirements.txt
```

### Estrutura de Diretórios

Antes de executar, certifique-se de que a estrutura de diretórios está correta:

```
projeto/
├── processamento_vetorizado.py
├── requirements.txt
├── README.md
├── inputs_images/
│   └── image_input_400.jpg    # Imagem de entrada
│   └── image_input_1280.jpg    # Imagem de entrada
└── matrices/
    └── matriz_qrng_400.png # Matriz QRNG
    └── matriz_qrng_1280_001.png # Matriz QRNG
```

## Como executar

Com o ambiente criado (ou com as dependências instaladas no ambiente atual), basta executar o script em Python processamento_vetorizado.py, pois os caminhos das imagens já estão pré-configurados para facilitar os testes.

É importante ressaltar que a imagem de entrada e a matriz QRNG utilizadas foram geradas previamente a partir de computadores quânticos. Rodar esse tipo de circuito diretamente exigiria a instalação de diversas outras bibliotecas e acesso à infraestrutura da IBM, portanto uma das matrizes geradas já foi salva e está pronta para uso.

A única configuração que exige atenção é o caminho da imagem, definido no início do código:

```
#Caminho para a imagem de exemplo 
CAMINHO_IMAGEM = "inputs_images/image_input_1280.jpg"

#Caminho para a imagem da matriz QRNG
CAMINHO_QRNG   = "matrices/matriz_qrng_1280_001.png"


TAMANHO_CORTE = None  #Tamanho do recorte da imagem
```

## Explicação

Desse modo, o script desenvolvido tem como objetivo comparar duas abordagens de implementação para um módulo de criptografia de imagens: uma versão não vetorizada, que utiliza laços for em Python para percorrer pixels e canais, e uma versão vetorizada, que explora operações sobre arrays usando NumPy.

O código foi organizado de forma modular, separando as funções de XOR e de permutação caótica, cada uma com suas versões vetorizadas e não vetorizadas.

Abaixo está a função de XOR vetorizada:

```
def aplicar_xor_com_qrng_vectorizado(imagem_processada, matriz_qrng):
    """
    Aplica XOR entre a imagem e a matriz QRNG usando vetorização NumPy.
    Funciona tanto para imagens em tons de cinza quanto RGB.
    """
    if imagem_processada.shape[:2] != matriz_qrng.shape:
        raise ValueError("A imagem e a matriz QRNG devem ter o mesmo tamanho")

    imagem = imagem_processada.astype(np.uint8)
    qrng = matriz_qrng.astype(np.uint8)

    if imagem.ndim == 3 and imagem.shape[2] == 3:
        qrng_expandida = qrng[..., None]  # (H, W, 1)
        xor_resultado = np.bitwise_xor(imagem, qrng_expandida)
    else:
        xor_resultado = np.bitwise_xor(imagem, qrng)

    return xor_resultado
```


Primeiramente, antes de aplicar a operação de XOR, essa função verifica se a imagem e a matriz QRNG possuem o mesmo tamanho e também garantem que estejam no formato **uint8**, sendo esse o formato mais usual de píxels. Em seguida, ela aplica a operação de XOR pixel a pixel na imagem, porém vale a pena ressaltar que, ao invés de percorrer todos os pixels com um laço **for**, são usadas duas técnicas principais, a primeira dela é o **broadcasting** sobre os três canais de cor, para que os shapes sejam compativels em caso de imagens RGB, e a função **np.bitwise_xor**, que é aplicada diretamente sobre os arrays da imagem e da matriz QRNG.

Dessa forma, o XOR é realizado em todos os elementos de uma única vez, em código otimizado em C, sem a necessidade de laços explícitos em Python, o que reduz significativamente o tempo de processamento.

Para comparação, o código a seguir apresenta a implementação não vetorizada do XOR, utilizando laços de repetição:

```
def aplicar_xor_com_qrng(imagem_processada, matriz_qrng):
    """
    Versão não vetorizada: uso de laços em pixels e canais.
    """
    if imagem_processada.shape[:2] != matriz_qrng.shape:
        raise ValueError("A imagem e a matriz QRNG devem ter o mesmo tamanho")

    imagem = imagem_processada.astype(np.uint8)
    qrng = matriz_qrng.astype(np.uint8)

    altura, largura = qrng.shape

    if imagem.ndim == 3 and imagem.shape[2] == 3:
        xor_resultado = np.empty_like(imagem, dtype=np.uint8)
        for i in range(altura):
            for j in range(largura):
                for c in range(3):
                    xor_resultado[i, j, c] = imagem[i, j, c] ^ qrng[i, j]
    else:
        xor_resultado = np.empty_like(imagem, dtype=np.uint8)
        for i in range(altura):
            for j in range(largura):
                xor_resultado[i, j] = imagem[i, j] ^ qrng[i, j]

    return xor_resultado
```

Após a aplicação do XOR, a próxima etapa é a implementação do mapa logístico caótico, que é responsável por embaralhar os pixels da imagem com base em uma sequência numérica altamente sensível às condições iniciais (parâmetros **r e x0**).

```

def logistic_map_vectorizado(size, r=3.999, x0=0.98892455322743):
    """
    Gera a sequência do mapa logístico com array pré-alocado.
    """
    x = x0
    seq = np.empty(size, dtype=np.float64)
    for i in range(size):
        x = r * x * (1 - x)
        seq[i] = x
    return seq


def aplicar_lm_vectorizado(imagem):
    """
    Aplica o mapa logístico para permutar os pixels da imagem de forma vetorizada.
    """
    h, w = imagem.shape[:2]
    total = h * w

    chaos_seq = logistic_map_vectorizado(total)
    indices = np.argsort(chaos_seq)

    if imagem.ndim == 3:
        C = imagem.shape[2]
        img_flat = imagem.reshape(-1, C)      # (total, C)
        img_perm = img_flat[indices, :]       # reordena linhas
        imagem_caotica = img_perm.reshape(h, w, C)
    else:
        img_flat = imagem.reshape(-1)
        img_perm = img_flat[indices]
        imagem_caotica = img_perm.reshape(h, w)

    return imagem_caotica, chaos_seq
```

No código, a função **logistic_map_vectorizado** recebe o tamanho total da imagem, isto é, o número de pixels, e gera uma sequência caótica (chaos_seq) com esse mesmo tamanho, usando a recorrência do mapa logístico e armazenando os valores em um array NumPy pré-alocado.

Em seguida, a função **aplicar_lm_vectorizado** usa essa sequência para permutar os pixels da imagem de forma vetorizada. Primeiro, ela calcula o total de pixels com base na altura e largura e gera chaos_seq chamando o mapa logístico. Depois, utiliza **np.argsort(chaos_seq)** para obter o vetor de índices que permite reordenar os pixels de acordo com a ordem crescente da sequência caótica. A imagem é achatada com reshape e, por fim, reindexada de uma vez só com **img_flat[indices]**, aplicando a permutação em bloco. Dessa forma, o embaralhamento dos pixels é feito utilizando indexação avançada em NumPy, o que caracteriza o uso de processamento vetorizado nessa etapa.

A função pipeline foi criada para auxiliar na execução do experimento com dois casos de uso: uma imagem de tamanho 400×400 e outra de 1280×1280 pixels, permitindo comparar o ganho de desempenho em diferentes escalas.


<p align="center">
  <b>Figura 2 - Imagem aplicada com XOR e LM (Mapa Logistico).</b>
</p>

![Figura 2 - Cifra](/imagens_relatorio/imagem_cifrada_peq.png)

<p align="center">
  <b>Fonte: Elaborado pelo autor.</b>
</p>

## Metodologia

O projeto utiliza a biblioteca `time` do Python para medir com precisão:
- Tempo de execução do XOR vetorizado vs. não vetorizado
- Tempo de execução do mapa logístico vetorizado vs. não vetorizado
- Tempo total do pipeline completo

E desse modo, alguns fatores podem influenciar nisso, tais qual:

1. **Tamanho da Imagem**: Imagens maiores amplificam a diferença de performance
2. **Número de Canais**: RGB (3 canais) vs. escala de cinza (1 canal)
3. **Hardware**: CPU com suporte a instruções SIMD (AVX, AVX2) tem maior ganho
4. **Memória Cache**: Imagens que cabem em cache L3 têm melhor performance


## Resultados

Resultados da imagem recortada de tamanho de 400x400

![Figura 3 - Resultados da Imagem 400x400](/imagens_relatorio/Processamento_imagem_pequeno.png)

Para imagem de 400x400
| Etapa          | Tempo não vetorizado (s) | Tempo vetorizado (s) |
| -------------- | ------------------------ | -------------------- |
| XOR            | 0.1937                   | 0.0014               |
| Mapa logístico | 0.2878                   | 0.0310               |
| **Total**      | **0.4815**               | **0.0324**           |

Para imagem de 1280x1280

![Figura 4 - Resultados da Imagem 1280x1280](/imagens_relatorio/Processamento_imagem_grande.png)


| Etapa          | Tempo não vetorizado (s) | Tempo vetorizado (s) |
| -------------- | ------------------------ | -------------------- |
| XOR            | 1.9280                   | 0.0134               |
| Mapa logístico | 3.8287                   | 0.3153               |
| **Total**      | **5.7567**               | **0.3287**           |

Foi possível observar que, em ambos os cenários, a versão vetorizada apresentou uma redução de tempo bastante significativa em relação à implementação não vetorizada. 

Na imagem de 400×400, o tempo total das etapas analisadas caiu de aproximadamente 0,48 s para 0,03 s, o que representa um ganho em torno de 15 vezes mais rápido. 

Já na imagem de 1280×1280, o tempo passou de cerca de 5,76 s para 0,33 s, resultando em um ganho de aproximadamente 17 vezes mais rápido na execução. 

Além disso, o módulo de XOR, por ser uma operação puramente elemento a elemento, foi o que mais se beneficiou da vetorização, chegando a ser cerca de 100 vezes mais rápido em comparação com a versão não vetorizada.

## Considerações Finais

Do ponto de vista de criptografia e programação distribuída, esses resultados reforçam a importância de se pensar em desempenho desde o nível de implementação. Sistemas de segurança que lidam com imagens de alta resolução ou grandes volumes de dados não podem depender de rotinas puramente sequenciais, sob risco de se tornarem impraticáveis em produção. 

A vetorização com NumPy mostrou-se uma solução relativamente simples de aplicar, mas com impacto direto em desempenho e escalabilidade, justamente um dos pontos de limitação do TCC original.

No cenário do TCC completo, esse efeito se torna ainda mais relevante: o tempo total da pipeline de criptografia, que inicialmente era de aproximadamente 226 segundos, foi reduzido para algo em torno de 178 segundos apenas com a aplicação de técnicas de vetorização em partes do processo. 

Embora isso ainda não resolva totalmente o problema de desempenho, já torna o sistema mais viável e abre espaço para extensões futuras, como o uso de GPU (por meio de bibliotecas compatíveis com a API do NumPy) ou a paralelização em múltiplos nós, juntamente com a parte para a geração da matriz QRNG.

Assim, este trabalho não só melhora a performance da solução proposta no TCC original, como também mostra que otimizações baseadas em processamento vetorizado são um passo importante na construção de sistemas de criptografia mais modernos, eficientes e preparados para ambientes distribuídos.