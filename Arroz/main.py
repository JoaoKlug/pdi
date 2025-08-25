#===============================================================================
# Exemplo: segmentação de uma imagem em escala de cinza.
#-------------------------------------------------------------------------------
# Autor: Bogdan T. Nassu
# Universidade Tecnológica Federal do Paraná
#===============================================================================

import sys
import timeit
import numpy as np
import cv2

#===============================================================================

INPUT_IMAGE =  'arroz.bmp'
NEGATIVO = False
THRESHOLD = 0.65
ALTURA_MIN = 15
LARGURA_MIN = 15
N_PIXELS_MIN = 15
ARROZ = 1
FUNDO = 0
PILHA = True
JANELA = 5
#===============================================================================

def binariza (img, threshold):
    ''' Binarização simples por limiarização.

Parâmetros: img: imagem de entrada. Se tiver mais que 1 canal, binariza cada
              canal independentemente.
            threshold: limiar.
            
Valor de retorno: versão binarizada da img_in.'''

    img = filtro_media(img, JANELA)
    return np.where(img > threshold, ARROZ, FUNDO).astype(np.float32)
#-------------------------------------------------------------------------------

def rotula (img, largura_min, altura_min, n_pixels_min):
    '''Rotulagem usando flood fill. Marca os objetos da imagem com os valores
[0.1,0.2,etc].

Parâmetros: img: imagem de entrada E saída.
            largura_min: descarta componentes com largura menor que esta.
            altura_min: descarta componentes com altura menor que esta.
            n_pixels_min: descarta componentes com menos pixels que isso.

Valor de retorno: uma lista, onde cada item é um vetor associativo (dictionary)
com os seguintes campos:

'label': rótulo do componente.
'n_pixels': número de pixels do componente.
'T', 'L', 'B', 'R': coordenadas do retângulo envolvente de um componente conexo,
respectivamente: topo, esquerda, baixo e direita.'''

    # Itera sobre cada pixel da imagem
    altura = img.shape[0]
    largura = img.shape[1]
    label = 0.01
    componentes = []

    flood_fill_func = flood_fill_pilha if PILHA else flood_fill_recursivo

    for y in range(altura):
        for x in range(largura):
            # Se achou um pixel de um objeto que ainda não foi rotulado.
            if img[y][x][0] == ARROZ:
                # Inicia o flood fill para encontrar o componente inteiro.
                componente_achado = flood_fill_func(img, label, x, y)

                # Verifica se o componente atende aos critérios de tamanho.
                if verficar_componente(componente_achado, largura_min, altura_min, n_pixels_min):
                    componentes.append(componente_achado)

                label += 0.01
    return componentes
#-------------------------------------------------------------------------------

def verficar_componente(componente, largura_min, altura_min, n_pixels_min):

    altura_obj = componente['B'] - componente['T'] + 1
    largura_obj = componente['R'] - componente['L'] + 1

    if(componente['n_pixels'] < n_pixels_min
       or altura_obj < altura_min
       or largura_obj < largura_min):
        return False

    return True
#-------------------------------------------------------------------------------

def flood_fill_recursivo(img, label, x, y): # x: coluna (largura), y: linha (altura)

    # Marca o pixel atual com o rótulo para não visitá-lo novamente.
    img[y][x] = label

    altura = img.shape[0]
    largura = img.shape[1]
    # Inicializa um componente "vazio" para ser preenchido pelos retornos da recursão.
    # T/L são inicializados com valores altos, B/R com valores baixos.
    componente = {'label': label, 'n_pixels': 0, 'T': altura, 'L': largura, 'B': -1, 'R': -1}

    # Chamadas recursivas para os 4 vizinhos, corrigindo o acesso e os limites.
    if(x > 0 and img[y][x-1][0] == ARROZ): # Vizinho da ESQUERDA
        componente = atualizar_componente_recursivo(componente, flood_fill_recursivo(img, label, x-1, y))

    if(y > 0 and img[y-1][x][0] == ARROZ): # Vizinho de CIMA
        componente = atualizar_componente_recursivo(componente,flood_fill_recursivo(img, label, x, y-1))

    if(x < largura-1 and img[y][x+1][0] == ARROZ): # Vizinho da DIREITA
        componente = atualizar_componente_recursivo(componente,flood_fill_recursivo(img, label, x+1, y))

    if(y < altura-1 and img[y+1][x][0] == ARROZ): # Vizinho de BAIXO
        componente = atualizar_componente_recursivo(componente, flood_fill_recursivo(img, label, x, y+1))

    # Após processar os vizinhos, adiciona os dados do pixel ATUAL ao componente.
    componente['n_pixels'] += 1
    componente['T'] = min(componente['T'], y)
    componente['L'] = min(componente['L'], x)
    componente['B'] = max(componente['B'], y)
    componente['R'] = max(componente['R'], x)

    return componente
#-------------------------------------------------------------------------------

def atualizar_componente_recursivo(componente, vizinho):
    
    componente['n_pixels'] += vizinho['n_pixels']
    componente['T'] = min(componente['T'], vizinho['T'])
    componente['L'] = min(componente['L'], vizinho['L'])
    componente['B'] = max(componente['B'], vizinho['B'])
    componente['R'] = max(componente['R'], vizinho['R'])

    return componente
#-------------------------------------------------------------------------------

def flood_fill_pilha(img, label, x, y):

    altura = img.shape[0]
    largura = img.shape[1]
    componente = {'label': label, 'n_pixels': 0, 'T': altura, 'L': largura, 'B': -1, 'R': -1}

    pilha = [(x,y)]
    img[y][x] = label
    atualizar_componente_pilha(componente, x, y)

    while(pilha):
        
        x, y = pilha.pop()

        if(x > 0 and img[y][x-1][0] == ARROZ): # Vizinho da ESQUERDA
            pilha.append((x-1, y))
            img[y][x-1] = label
            atualizar_componente_pilha(componente, x-1, y)
        if(y > 0 and img[y-1][x][0] == ARROZ): # Vizinho de CIMA
            pilha.append((x, y-1))
            img[y-1][x] = label
            atualizar_componente_pilha(componente, x, y-1)
        if(x < largura-1 and img[y][x+1][0] == ARROZ): # Vizinho da DIREITA
            pilha.append((x+1, y))
            img[y][x+1] = label
            atualizar_componente_pilha(componente, x+1, y)
        if(y < altura-1 and img[y+1][x][0] == ARROZ): # Vizinho de BAIXO
            pilha.append((x, y+1))
            img[y+1][x] = label
            atualizar_componente_pilha(componente, x, y+1)

    return componente
#-------------------------------------------------------------------------------

def atualizar_componente_pilha(componente, novo_x, novo_y):
    componente['n_pixels'] += 1
    componente['T'] = min(componente['T'], novo_y)
    componente['L'] = min(componente['L'], novo_x)
    componente['B'] = max(componente['B'], novo_y)
    componente['R'] = max(componente['R'], novo_x)

    return componente  
#-------------------------------------------------------------------------------

def filtro_media(img, janela):
    """
    Aplica um filtro de média usando um algoritmo separável e otimizado com janelas deslizantes.
    Primeiro, calcula a soma horizontal em um buffer e, em seguida, a soma vertical sobre esse buffer.

    Parâmetros:
        img: imagem de entrada (NumPy array).
        janela: tamanho da janela do filtro (deve ser um número ímpar).

    Valor de retorno:
        imagem filtrada (NumPy array).
    """

    # Obtém as dimensões da imagem.
    altura = img.shape[0]
    largura = img.shape[1]

    # Cria um buffer para armazenar os resultados intermediários (somas horizontais).
    buffer = img.copy()
    # Cria uma imagem final para armazenar o resultado.
    img_final = img.copy()

    # Calcula o raio da janela (metade do tamanho).
    w_janela = janela // 2

    # --- PASSE 1: SOMA HORIZONTAL ---
    # Itera sobre cada linha da imagem.
    for y in range (altura):
        
        # Calcula a soma da primeira janela da linha.
        soma = 0
        for x in range (janela):
            soma += img[y][x]
        # Armazena a primeira soma na posição central da janela no buffer.
        buffer[y][w_janela] = soma
        
        # Desliza a janela pelo resto da linha.
        for x in range (w_janela + 1, largura - w_janela):
            primeiro = x - w_janela - 1
            ultimo = x + w_janela
            soma -= img[y][primeiro]
            soma += img[y][ultimo]
            # Armazena a nova soma na posição atual no buffer.
            buffer [y][x] = soma


    # --- PASSE 2: SOMA VERTICAL E CÁLCULO DA MÉDIA ---
    # Itera sobre cada coluna da imagem.
    for x in range (largura):

        # Calcula a soma da primeira janela da coluna, usando os valores do buffer.
        soma = 0
        for y in range (janela):
            soma += buffer[y][x]
        # Calcula a média final e armazena na imagem de saída.
        img_final[w_janela][x] = soma / (janela * janela)

        # Desliza a janela pelo resto da coluna.
        for y in range (w_janela + 1, altura - w_janela):
            primeiro = y - w_janela - 1
            ultimo = y + w_janela
            soma -= buffer[primeiro][x]
            soma += buffer[ultimo][x]
            # Calcula a média final e armazena na imagem de saída.
            img_final[y][x] = soma / (janela * janela)

    # Retorna a imagem com o filtro de média aplicado.
    return img_final
#===============================================================================

def main ():

     # Abre a imagem em escala de cinza.
    img = cv2.imread (INPUT_IMAGE, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print ('Erro abrindo a imagem.\n')
        sys.exit ()

    # É uma boa prática manter o shape com 3 valores, independente da imagem ser
    # colorida ou não. Também já convertemos para float32.
    img = img.reshape ((img.shape [0], img.shape [1], 1))
    img = img.astype (np.float32) / 255

    # Mantém uma cópia colorida para desenhar a saída.
    img_out = cv2.cvtColor (img, cv2.COLOR_GRAY2BGR)

    # Segmenta a imagem.
    if NEGATIVO:
        img = 1 - img
    img = binariza (img, THRESHOLD)
    cv2.imshow ('01 - binarizada', img)
    cv2.imwrite ('01 - binarizada.png', (img*255).astype(np.uint8))

    start_time = timeit.default_timer ()
    componentes = rotula (img, LARGURA_MIN, ALTURA_MIN, N_PIXELS_MIN)
    n_componentes = len (componentes)
    print ('Tempo: %f' % (timeit.default_timer () - start_time))
    print ('%d componentes detectados.' % n_componentes)

    # Mostra os objetos encontrados.
    for c in componentes:
        cv2.rectangle (img_out, (c ['L'], c ['T']), (c ['R'], c ['B']), (0,0,1))

    cv2.imshow ('02 - out', img_out)
    cv2.imwrite ('02 - out.png', (img_out*255).astype(np.uint8))
    cv2.waitKey ()
    cv2.destroyAllWindows ()


if __name__ == '__main__':
    main ()

#===============================================================================
