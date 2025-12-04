import time
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Caso 1: imagem maior (sem recorte)
CAMINHO_IMAGEM_GRANDE = "inputs_images/image_input_1280.jpg"
CAMINHO_QRNG_GRANDE = "matrices/matriz_qrng_1280_001.png"
TAMANHO_CORTE_GRANDE = None  # sem recorte

# Caso 2: imagem com centro recortado
CAMINHO_IMAGEM_REC = "inputs_images/image_input_400.jpg"       
CAMINHO_QRNG_REC = "matrices/matriz_qrng_400.png"        
TAMANHO_CORTE_REC = 400    # recorte central 400x400 para ajustar ao tamanho da matriz

#----------------------------------------
#Funções da criptografia de imagens vetorizadas
#----------------------------------------

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

#----------------------------------------
#Funções não vetorizadas
#----------------------------------------

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


def logistic_map(size, r=3.999, x0=0.98892455322743):
    """
    Versão não vetorizada do mapa logístico usando lista + append.
    """
    x = x0
    seq = []
    for _ in range(size):
        x = r * x * (1 - x)
        seq.append(x)
    return np.array(seq, dtype=np.float64)


def aplicar_lm(imagem):
    """
    Aplica o mapa logístico permutando a imagem usando laços explícitos.
    """
    h, w = imagem.shape[:2]
    total = h * w

    chaos_seq = logistic_map(total)

    # Gera os índices ordenados "na mão" (sem np.argsort)
    indices = list(range(total))
    indices.sort(key=lambda i: chaos_seq[i])

    imagem_caotica = np.empty_like(imagem)

    if imagem.ndim == 3:
        C = imagem.shape[2]
        for novo_idx, antigo_idx in enumerate(indices):
            new_i = novo_idx // w
            new_j = novo_idx % w
            old_i = antigo_idx // w
            old_j = antigo_idx % w

            for c in range(C):
                imagem_caotica[new_i, new_j, c] = imagem[old_i, old_j, c]

    else:
        for novo_idx, antigo_idx in enumerate(indices):
            new_i = novo_idx // w
            new_j = novo_idx % w
            old_i = antigo_idx // w
            old_j = antigo_idx % w

            imagem_caotica[new_i, new_j] = imagem[old_i, old_j]

    return imagem_caotica, chaos_seq

#----------------------------------------
#Funções de Imagens
#----------------------------------------
def cortar_centro(imagem: np.ndarray, tamanho: int) -> np.ndarray | None:
    """
    Recorta um quadrado central da imagem com o tamanho especificado.
    """
    if imagem is None:
        print("Erro: Nenhuma imagem fornecida para cortar.")
        return None

    h, w = imagem.shape[:2]
    if h < tamanho or w < tamanho:
        print(f"Erro: Imagem ({w}x{h}) muito pequena para {tamanho}x{tamanho}.")
        return None

    centro_x, centro_y = w // 2, h // 2
    metade_tamanho = tamanho // 2

    y1 = centro_y - metade_tamanho
    y2 = centro_y + metade_tamanho
    x1 = centro_x - metade_tamanho
    x2 = centro_x + metade_tamanho

    return imagem[y1:y2, x1:x2]


def mostrar_imagem(imagem_input: np.ndarray, titulo: str) -> None:
    """
    Mostra uma única imagem usando Matplotlib.
    Converte de BGR (OpenCV) para RGB (Matplotlib).
    """
    if imagem_input is not None:
        imagem_rgb = cv2.cvtColor(imagem_input, cv2.COLOR_BGR2RGB)
        plt.imshow(imagem_rgb)
        plt.title(titulo)
        plt.axis("off")
        plt.show()
    else:
        print("Erro: Tentativa de mostrar uma imagem que é nula (None).")


def mostrar_lado_a_lado(img1: np.ndarray, img2: np.ndarray,
                        titulo1: str = "Imagem 1", titulo2: str = "Imagem 2") -> None:
    """Mostra duas imagens lado a lado para comparação."""
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    axs[0].set_title(titulo1)
    axs[0].axis('off')

    axs[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    axs[1].set_title(titulo2)
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()


# ----------------------------------------
# Função para rodar todos os caso (pipeline de imagem + matriz)
# ----------------------------------------

def pipeline(nome_caso: str, caminho_img: str, caminho_qrng: str, tamanho_crop: int | None = None):
    print(f"\n{nome_caso}")

    # Carrega a imagem normal em BGR (colorida)
    imagem = cv2.imread(caminho_img, cv2.IMREAD_COLOR)
    if imagem is None:
        print(f"Erro ao carregar imagem: {caminho_img}")
        return

    # Carrega a matriz QRNG em escala de cinza (2D)
    matriz_qrng = cv2.imread(caminho_qrng, cv2.IMREAD_GRAYSCALE)
    if matriz_qrng is None:
        print(f"Erro ao carregar imagem QRNG: {caminho_qrng}")
        return

    print("Tamanho da imagem original:", imagem.shape)
    print("Tamanho da QRNG original:", matriz_qrng.shape)

    # Crop opcional
    if tamanho_crop is not None:
        imagem_proc = cortar_centro(imagem, tamanho_crop)
        matriz_proc = cortar_centro(matriz_qrng, tamanho_crop)
    else:
        imagem_proc = imagem
        matriz_proc = matriz_qrng

    if imagem_proc.shape[:2] != matriz_proc.shape:
        print("Erro: tamanhos diferentes após corte.")
        print("Imagem:", imagem_proc.shape, "QRNG:", matriz_proc.shape)
        return

    print("Tamanho da imagem após processamento:", imagem_proc.shape)
    print("Tamanho da QRNG após processamento:", matriz_proc.shape)

    mostrar_imagem(imagem_proc, f"Imagem de entrada - {nome_caso}")

    tempo_start = time.time()

    # XOR vetorizado vs não vetorizado
    t0 = time.time()
    imagem_xor_vec = aplicar_xor_com_qrng_vectorizado(imagem_proc, matriz_proc)
    t1 = time.time()
    imagem_xor_nao = aplicar_xor_com_qrng(imagem_proc, matriz_proc)
    t2 = time.time()

    print(f"[{nome_caso}] Tempo XOR vetorizado:     {t1 - t0:.4f} s")
    print(f"[{nome_caso}] Tempo XOR não vetorizado: {t2 - t1:.4f} s")

    # Logistic map vetorizado vs não vetorizado
    imagem_caotica_vec, chaos_seq_vec = aplicar_lm_vectorizado(imagem_xor_vec)
    t3 = time.time()
    imagem_caotica_nao, chaos_seq_nao = aplicar_lm(imagem_xor_nao)
    t4 = time.time()

    print(f"[{nome_caso}] Tempo LM vetorizado:      {t3 - t2:.4f} s")
    print(f"[{nome_caso}] Tempo LM não vetorizado:  {t4 - t3:.4f} s")

    tempo_end = time.time()
    print(f"[{nome_caso}] Pipeline concluído em {tempo_end - tempo_start:.2f} s\n")

    # Visualização
    mostrar_lado_a_lado(imagem_xor_vec, imagem_caotica_vec,
                        f"XOR vetorizado - {nome_caso}",
                        f"LM vetorizado - {nome_caso}")
    
    #mostrar_lado_a_lado(imagem_xor_nao, imagem_caotica_nao,
    #                    f"XOR não vetorizado - {nome_caso}",
    #                    f"LM não vetorizado - {nome_caso}")

#----------------------------------------
#Execução
#----------------------------------------
def main():
    # Caso 1: imagem maior
    pipeline("Imagem maior",
               CAMINHO_IMAGEM_GRANDE,
               CAMINHO_QRNG_GRANDE,
               tamanho_crop=TAMANHO_CORTE_GRANDE)

    # Caso 2: imagem menor
    pipeline("Imagem centro recortado",
               CAMINHO_IMAGEM_REC,
               CAMINHO_QRNG_REC,
               tamanho_crop=TAMANHO_CORTE_REC)


if __name__ == "__main__":
    main()
