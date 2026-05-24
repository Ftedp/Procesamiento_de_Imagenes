import cv2
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# EJERCICIO 2 - Análisis automático de partidas de TA-TE-TI
# =============================================================================
#
# PASO A PASO:
#
# 1. DETECCIÓN DE BORDES (con Canny)
#    Se aplica GaussianBlur para reducir el ruido y luego Canny para obtener
#    una imagen binaria con los bordes de la imagen.
#
# 2. DETECCIÓN DE LÍNEAS (HoughLines)
#    Sobre la imagen de bordes se aplica la transformada de Hough para detectar
#    las líneas del tablero en coordenadas polares (rho, theta).
#    Se separan en horizontales (theta ≈ 90°) y verticales (theta ≈ 0°).
#    Las líneas duplicadas (por grosor) se agrupan por proximidad de rho.
#    Resultado: 2 líneas horizontales + 2 líneas verticales = 4 líneas reales.
#
# 3. EXTRACCIÓN DE CELDAS
#    Con las 4 líneas detectadas se definen los bordes del tablero y se
#    recortan las 9 celdas mediante slicing de la imagen original.
#
# 4. CLASIFICACIÓN DE CELDAS
#    Cada celda se clasifica en Círculo, Cruz o Vacío:
#    - Círculo: se detecta con HoughCircles.
#    - Cruz: se aplican filtros diagonales (+45° y -45°). Si la respuesta
#            es fuerte en ambos → hay una cruz.
#    - Vacío: no se detectó ni círculo ni cruz.
#
# 5. DETECCIÓN DE GANADOR
#    Se analizan las 8 combinaciones posibles (3 filas, 3 columnas, 2 diagonales)
#    para determinar si algún jugador alineó 3 figuras iguales.
#
# =============================================================================


def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=True, ticks=False):
    if new_fig:
        plt.figure()
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    if not ticks:
        plt.xticks([]), plt.yticks([])
    if colorbar:
        plt.colorbar()
    if new_fig:
        plt.show(block=blocking)


def agrupar_lineas(rhos, gap=10):
    """Agrupa rhos cercanos (diferencia < gap) y devuelve la media de cada grupo.
    
    Parámetros:
        rhos: array de valores rho ordenados.
        gap:  distancia máxima en píxeles para considerar dos líneas como duplicadas.
    
    Retorna:
        Array con un rho representativo por cada grupo.
    """
    grupos, grupo_actual = [], [rhos[0]]
    for r in rhos[1:]:
        if r - grupo_actual[-1] < gap:
            grupo_actual.append(r)
        else:
            grupos.append(np.mean(grupo_actual))
            grupo_actual = [r]
    grupos.append(np.mean(grupo_actual))
    return np.array(grupos)


def detectar_lineas(img):
    """Detecta las 4 líneas del tablero (2 horizontales + 2 verticales).
    
    Aplica GaussianBlur + Canny para obtener bordes, luego HoughLines para
    detectar líneas en coordenadas polares. Separa horizontales de verticales
    por el ángulo theta y agrupa las duplicadas por proximidad de rho.
    
    Parámetros:
        img: imagen BGR del tablero de ta-te-ti.
    
    Retorna:
        bordes_h: array con los 4 límites verticales [0, y1, y2, alto].
        bordes_v: array con los 4 límites horizontales [0, x1, x2, ancho].
    """
    blur  = cv2.GaussianBlur(img, (3,3), 0)
    canny = cv2.Canny(blur, threshold1=80, threshold2=120)

    lines  = cv2.HoughLines(canny, rho=1, theta=np.pi/180, threshold=250)
    thetas = lines[:,0,1] * 180 / np.pi

    horizontales = lines[thetas > 45]
    verticales   = lines[thetas <= 45]

    rhos_h = agrupar_lineas(np.sort(horizontales[:,0,0]))
    rhos_v = agrupar_lineas(np.sort(np.abs(verticales[:,0,0])))

    bordes_h = np.array([0, rhos_h[0], rhos_h[1], img.shape[0]], dtype=int)
    bordes_v = np.array([0, rhos_v[0], rhos_v[1], img.shape[1]], dtype=int)

    return bordes_h, bordes_v


def clasificar_celda(celda):
    """Clasifica una celda del tablero como Círculo, Cruz o Vacío.
    
    Primero intenta detectar un círculo con HoughCircles. Si no encuentra,
    aplica filtros diagonales para detectar una cruz. Si ninguno da resultado,
    la celda se considera vacía.
    
    Parámetros:
        celda: recorte BGR de una celda del tablero.
    
    Retorna:
        String: 'Círculo', 'Cruz' o 'Vacío'.
    """
    w1 = np.array([[-1,-1,2],[-1,2,-1],[2,-1,-1]])  # filtro diagonal +45°
    w2 = np.array([[2,-1,-1],[-1,2,-1],[-1,-1,2]])  # filtro diagonal -45°

    celda_gray = cv2.cvtColor(celda, cv2.COLOR_BGR2GRAY)
    celda_blur = cv2.GaussianBlur(celda_gray, (5,5), 1)

    circles = cv2.HoughCircles(celda_blur, method=cv2.HOUGH_GRADIENT, dp=1, minDist=50, param2=30)
    if circles is not None:
        return "Círculo"

    f1 = np.abs(cv2.filter2D(celda_gray, cv2.CV_64F, w1))
    f2 = np.abs(cv2.filter2D(celda_gray, cv2.CV_64F, w2))
    if f1.max() > 500 and f2.max() > 500:
        return "Cruz"

    return "Vacío"


def analizar_tablero(img):
    """Recorre las 9 celdas del tablero y clasifica cada una.
    
    Parámetros:
        img: imagen BGR del tablero de ta-te-ti.
    
    Retorna:
        grilla: lista 3x3 con las etiquetas ('Círculo', 'Cruz' o 'Vacío').
    """
    bordes_h, bordes_v = detectar_lineas(img)

    grilla = []
    for i in range(3):
        fila = []
        for j in range(3):
            y1, y2 = bordes_h[i], bordes_h[i+1]
            x1, x2 = bordes_v[j], bordes_v[j+1]
            celda    = img[y1:y2, x1:x2]
            etiqueta = clasificar_celda(celda)
            fila.append(etiqueta)
        grilla.append(fila)

    return grilla


def detectar_ganador(grilla):
    """Determina si hay un ganador en la partida de ta-te-ti.
    
    Verifica las 8 combinaciones ganadoras posibles: 3 filas, 3 columnas
    y 2 diagonales. Un jugador gana si tiene 3 figuras iguales en línea.
    
    Parámetros:
        grilla: lista 3x3 con las etiquetas de cada celda.
    
    Retorna:
        String con el ganador ('Ganador: Círculo' / 'Ganador: Cruz')
        o 'Sin ganador' si la partida no terminó o es empate.
    """
    for i in range(3):
        if grilla[i][0] == grilla[i][1] == grilla[i][2] and grilla[i][0] != "Vacío":
            return f"Ganador: {grilla[i][0]}"
        if grilla[0][i] == grilla[1][i] == grilla[2][i] and grilla[0][i] != "Vacío":
            return f"Ganador: {grilla[0][i]}"

    if grilla[0][0] == grilla[1][1] == grilla[2][2] and grilla[0][0] != "Vacío":
        return f"Ganador: {grilla[0][0]}"
    if grilla[0][2] == grilla[1][1] == grilla[2][0] and grilla[0][2] != "Vacío":
        return f"Ganador: {grilla[0][2]}"

    return "Sin ganador"


# =============================================================================
if __name__ == "__main__":
    for k in range(1, 10):
        img     = cv2.imread(f"Unidad_3/Ejercicios/tateti_{k}.png")
        grilla  = analizar_tablero(img)
        resultado = detectar_ganador(grilla)
        print(f"\ntateti_{k}: {resultado}")
        for fila in grilla:
            print(fila)