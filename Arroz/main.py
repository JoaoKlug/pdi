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

# TODO: ajuste estes parâmetros!
NEGATIVO = False
THRESHOLD = 0.8
ALTURA_MIN = 10
LARGURA_MIN = 10
N_PIXELS_MIN = 10

#===============================================================================

def binariza (img, threshold):
    ''' Binarização simples por limiarização.

Parâmetros: img: imagem de entrada. Se tiver mais que 1 canal, binariza cada
              canal independentemente.
            threshold: limiar.
            
Valor de retorno: versão binarizada da img_in.'''

    '''
    canais = img.shape[2]
    linhas = img.shape[0]
    colunas = img.shape[1]

    for k in range(canais):
        for i in range(linhas):
            for j in range(colunas):
                if img[i][j][k] > threshold:
                    img[i][j][k] = 1
                else:
                    img[i][j][k] = 0

    return img'''

    return np.where(img > threshold, 1, 0).astype(np.float32)

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

    # TODO: escreva esta função.
    # Use a abordagem com flood fill recursivo.
    
    # Garante que a recursão não exceda o limite do Python para imagens grandes.
    
    # Itera sobre cada pixel da imagem

    linhas = img.shape[0]
    colunas = img.shape[1]
    label = 0.01
    componentes = []
    aux = []

    for i in range(linhas):
        for j in range(colunas):
            if img[i][j][0] == 1:
                flood_fill(img, label, i, j)
                aux = achar_componente(img, label)
                if aux != None:
                    componentes.append(aux)
                label += 0.01

    return componentes

def achar_componente (img, label):
    
    coords = np.where(img == label)

    n_pixels = len(coords[0])

    # T (Topo) = índice mínimo da linha
    T = np.min(coords[0])
    # B (Base) = índice máximo da linha
    B = np.max(coords[0])
    # L (Esquerda) = índice mínimo da coluna
    L = np.min(coords[1])
    # R (Direita) = índice máximo da coluna
    R = np.max(coords[1])

    if((B-T) < ALTURA_MIN or (R-L) < LARGURA_MIN or n_pixels < N_PIXELS_MIN):
        return None

    return {'label': label, 'n_pixels': n_pixels, 'T': T, 'L': L, 'B': B, 'R': R}
            
def flood_fill(img, label, x, y):

    img[x][y] = label

    if(x > 0 and img[x-1][y][0] == 1):
        flood_fill(img, label, x-1, y)
    if(y > 0 and img[x][y-1][0] == 1):
        flood_fill(img, label, x, y-1)
    if(x < img.shape[0]-1 and img[x+1][y][0] == 1):
        flood_fill(img, label, x+1, y)
    if(y < img.shape[1]-1 and img[x][y+1][0] == 1):
        flood_fill(img, label, x, y+1)

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
    cv2.imwrite ('01 - binarizada.png', img*255)

    start_time = timeit.default_timer ()
    componentes = rotula (img, LARGURA_MIN, ALTURA_MIN, N_PIXELS_MIN)
    n_componentes = len (componentes)
    print ('Tempo: %f' % (timeit.default_timer () - start_time))
    print ('%d componentes detectados.' % n_componentes)

    # Mostra os objetos encontrados.
    for c in componentes:
        cv2.rectangle (img_out, (c ['L'], c ['T']), (c ['R'], c ['B']), (0,0,1))

    cv2.imshow ('02 - out', img_out)
    cv2.imwrite ('02 - out.png', img_out*255)
    cv2.waitKey ()
    cv2.destroyAllWindows ()


if __name__ == '__main__':
    main ()

#===============================================================================
