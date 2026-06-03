import cv2
import numpy as np
import matplotlib.pyplot as plt


#-----------------------------------------------------------------------------------
#----------------------------------IMG----------------------------------------------
#-----------------------------------------------------------------------------------

img = cv2.imread('img_12.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print(f"Shape: {img.shape}")
print(f"Alto (h): {img.shape[0]} px")
print(f"Ancho (w): {img.shape[1]} px")
print(f"Área total: {img.shape[0] * img.shape[1]} px²")

plt.figure()
plt.subplot(121), plt.imshow(img_rgb), plt.title('Original'), plt.axis('off')
plt.subplot(122), plt.imshow(img_gray, cmap='gray'), plt.title('Grises'), plt.axis('off')
plt.show(block=False)

# Redimensionamos manteniendo proporción
h, w = img.shape[:2]
escala = 800 / max(h, w)
img = cv2.resize(img, (int(w * escala), int(h * escala)), interpolation=cv2.INTER_AREA)

# Recalculamos grises sobre la imagen YA redimensionada
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print(f"Shape redimensionada: {img.shape}")

# Blur suave para no amplificar ruido con el sharpening
img_suave = cv2.GaussianBlur(img_gray, (3, 3), 0)

# Sharpening: realza bordes restando el laplaciano local
kernel_sharp = np.array([[ 0, -1,  0],
                         [-1,  5, -1],
                         [ 0, -1,  0]], dtype=np.float32)
img_sharp = cv2.filter2D(img_suave, -1, kernel_sharp)

# Blur mayor antes de Canny para suprimir ruido fino post-sharpening
img_blur = cv2.GaussianBlur(img_sharp, (5, 5), 0)
edges = cv2.Canny(img_blur, 100, 200)

plt.figure()
plt.subplot(131), plt.imshow(img_gray, cmap='gray'), plt.title('Grises'), plt.axis('off')
plt.subplot(132), plt.imshow(img_sharp, cmap='gray'), plt.title('Sharpening'), plt.axis('off')
plt.subplot(133), plt.imshow(edges, cmap='gray'), plt.title('Canny'), plt.axis('off')
plt.show(block=False)

# Buscamos todos los contornos en la imagen de bordes
contornos, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"Total contornos encontrados: {len(contornos)}")

img_h, img_w = img.shape[:2]
img_area = img_h * img_w

# Mostramos todos los bounding boxes para ver qué hay
candidatos_crudos = []
for cnt in contornos:
    x, y, w, h = cv2.boundingRect(cnt)
    if h == 0:
        continue
    ratio = w / h
    area_rel = (w * h) / img_area
    candidatos_crudos.append((x, y, w, h, ratio, area_rel, cnt))

print(f"Candidatos con h>0: {len(candidatos_crudos)}")
print(f"\nTop 10 por área relativa:")
for c in sorted(candidatos_crudos, key=lambda c: c[5], reverse=True)[:10]:
    print(f"  ratio={c[4]:.2f}  area_rel={c[5]:.5f}  pos=({c[0]},{c[1]})  size={c[2]}x{c[3]}")

# Filtro por ratio y área relativa derivados de las dimensiones reales de la patente
RATIO_MIN = 2.0
RATIO_MAX = 4.5
AREA_REL_MIN = 0.005
AREA_REL_MAX = 0.05

candidatos_filtrados = []
for x, y, w, h, ratio, area_rel, cnt in candidatos_crudos:
    if RATIO_MIN < ratio < RATIO_MAX and AREA_REL_MIN < area_rel < AREA_REL_MAX:
        candidatos_filtrados.append((x, y, w, h, ratio, area_rel, cnt))

print(f"Candidatos tras filtro ratio+área: {len(candidatos_filtrados)}")
for c in sorted(candidatos_filtrados, key=lambda c: c[5], reverse=True):
    print(f"  ratio={c[4]:.2f}  area_rel={c[5]:.5f}  pos=({c[0]},{c[1]})  size={c[2]}x{c[3]}")

# Dibujamos cada candidato con su número de identificación
img_candidatos = img_rgb.copy()
candidatos_sorted = sorted(candidatos_filtrados, key=lambda c: c[5], reverse=True)

for i, (x, y, w, h, ratio, area_rel, cnt) in enumerate(candidatos_sorted):
    cv2.rectangle(img_candidatos, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.putText(img_candidatos, str(i), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

plt.figure(figsize=(8, 10))
plt.imshow(img_candidatos)
plt.title(f'Candidatos filtrados ({len(candidatos_filtrados)})')
plt.axis('off')
plt.show(block=False)

RATIO_IDEAL = 3.08
UMBRAL_BLANCO = 200
UMBRAL_NEGRO = 50
UMBRAL_SOLAP_BLANCO = 0.35
UMBRAL_SOLAP_NEGRO = 0.10
UMBRAL_SOLIDEZ_MAR = 0.80

# Inicializamos el puntaje de cada candidato en 0
puntajes = [0] * len(candidatos_sorted)

#---------------------------Seleccion por ratio---------------------------------
RATIO_IDEAL = 3.08
UMBRAL_ANGULO = 15

for i, (x, y, w, h, ratio, area_rel, cnt) in enumerate(candidatos_sorted):
    rect = cv2.minAreaRect(cnt)
    _, (rw, rh), angulo = rect
    if abs(angulo) > UMBRAL_ANGULO:
        ratio_usado = max(rw, rh) / min(rw, rh) if min(rw, rh) > 0 else 0
    else:
        ratio_usado = ratio
    dist_ratio = abs(ratio_usado - RATIO_IDEAL)
    if dist_ratio < 0.5:
        puntajes[i] += 1
    print(f"{i:>3} {ratio_usado:>12.2f} {angulo:>8.2f} {dist_ratio:>11.2f} {puntajes[i]:>7}")

#---------------------------Discriminacion por blanco Y negro---------------------------
UMBRAL_BLANCO = 200
UMBRAL_SOLAP_BLANCO = 0.35
UMBRAL_NEGRO = 100
UMBRAL_SOLAP_NEGRO = 0.10

mascara_blanco = (img_gray > UMBRAL_BLANCO).astype(np.uint8)
mascara_negro  = (img_gray < UMBRAL_NEGRO).astype(np.uint8)

for i, (x, y, w, h, ratio, area_rel, cnt) in enumerate(candidatos_sorted):
    roi_blanco = mascara_blanco[y:y+h, x:x+w]
    roi_negro  = mascara_negro[y:y+h, x:x+w]
    solap_blanco = roi_blanco.sum() / (w * h)
    solap_negro  = roi_negro.sum() / (w * h)
    if solap_blanco > UMBRAL_SOLAP_BLANCO and solap_negro > UMBRAL_SOLAP_NEGRO:
        puntajes[i] += 1
    print(f"{i:>3} {solap_blanco:>13.3f} {solap_negro:>12.3f} {puntajes[i]:>7}")
    

#---------------------------Discriminacion por contornos de caracteres---------------------------
print(f"\n{'ID':>3} {'contornos_car':>14} {'puntos':>7}")
print("-" * 30)

for i, (x, y, w, h, ratio, area_rel, cnt) in enumerate(candidatos_sorted):
    roi = img_gray[y:y+h, x:x+w]
    roi_area = w * h

    roi_bin = cv2.adaptiveThreshold(roi, 255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV,
                                    11, 2)

    contornos_roi, _ = cv2.findContours(roi_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    n_car = 0
    for c in contornos_roi:
        xc, yc, wc, hc = cv2.boundingRect(c)
        if hc == 0 or wc == 0:
            continue
        area_rel_c = (wc * hc) / roi_area
        ratio_c = hc / wc
        if 1.2 < ratio_c < 2.5 and 0.04 < area_rel_c < 0.20:
            n_car += 1

    if 3 <= n_car <= 7:
        puntajes[i] += 1

    print(f"{i:>3} {n_car:>14} {puntajes[i]:>7}")


#---------------------------Seleccion del ganador---------------------------
# En caso de empate, gana el candidato con mayor y (más abajo en la imagen)
puntaje_max = max(puntajes)
ganadores = [i for i, p in enumerate(puntajes) if p == puntaje_max]

if len(ganadores) == 1:
    idx_ganador = ganadores[0]
else:
    # Desempate: el más abajo en la imagen (mayor y)
    idx_ganador = max(ganadores, key=lambda i: candidatos_sorted[i][1])

print(f"\nPuntajes finales:")
for i, p in enumerate(puntajes):
    marca = " <-- GANADOR" if i == idx_ganador else ""
    print(f"  ID {i}: {p} puntos{marca}")

x, y, w, h, ratio, area_rel, cnt = candidatos_sorted[idx_ganador]
print(f"\nGanador: ID {idx_ganador}, pos=({x},{y}), size={w}x{h}, ratio={ratio:.2f}")

#---------------------------Crop del ganador---------------------------
x, y, w, h, ratio, area_rel, cnt = candidatos_sorted[idx_ganador]
crop_ganador = img_rgb[y:y+h, x:x+w]

plt.figure()
plt.imshow(crop_ganador)
plt.title(f'Ganador ID {idx_ganador} - ratio={ratio:.2f} - puntos={puntaje_max}')
plt.axis('off')
plt.show(block=False)

#=====================================================================================
#=====================================================================================
# SI -> 1,2,3,4,8,10,12. count -> 7
# NO -> 5,6,7,9,11. count -> 5
#Imagen 5 hay que solucionar que la patente no entra directamente en los candidatos
#por ende ninguna discriminacion funciona.
#Imagen 6 pasa lo mismo.
#Imagen 7 falla porque candidato 0(falso positivo) esta mas bajo que candidato 2(patente).
# Ajustar parametro
#Imagen 11 falla ratio y contornos (solo tiene dos, pero si bajo mas el parametro introdzco ruido)
#Falta identificar cada letra de la patente.
#=====================================================================================

    # Tomamos el crop del ganador actual
x, y, w, h, ratio, area_rel, cnt = candidatos_sorted[idx_ganador]
crop = img_rgb[y:y+h, x:x+w]
crop_gray = img_gray[y:y+h, x:x+w]

# Binarizamos localmente
crop_bin = cv2.adaptiveThreshold(crop_gray, 255,
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV,
                                  11, 2)

# Buscamos contornos
contornos_crop, _ = cv2.findContours(crop_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Contornos encontrados en ganador: {len(contornos_crop)}")

roi_h, roi_w = crop_gray.shape[:2]
roi_area = roi_h * roi_w

caracteres_validos = []
crop_debug2 = crop.copy()

for c in contornos_crop:
    xc, yc, wc, hc = cv2.boundingRect(c)
    if hc == 0 or wc == 0:
        continue
    area_rel = (wc * hc) / roi_area
    ratio_c = hc / wc  # más alto que ancho → ratio > 1
    if 1.2 < ratio_c < 2.5 and 0.04 < area_rel < 0.20:
        caracteres_validos.append((xc, yc, wc, hc))
        cv2.rectangle(crop_debug2, (xc, yc), (xc+wc, yc+hc), (0, 255, 0), 1)

print(f"Contornos con forma de carácter: {len(caracteres_validos)}")

plt.figure()
plt.imshow(crop_debug2)
plt.title(f'Caracteres válidos ({len(caracteres_validos)})')
plt.axis('off')
plt.show(block=False)