import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Cargo imagen ---
img = cv2.imread('pills.png')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print(f"Shape: {img.shape}")
print(f"dtype: {img.dtype}")

# --- Perfil vertical: intensidad promedio por fila ---
perfil = np.mean(img_gray, axis=1)  # Para cada fila, promedio de todas las columnas

plt.figure(figsize=(10, 4))
plt.subplot(121), plt.imshow(img_rgb), plt.title('Imagen original'), plt.axis('off')
plt.subplot(122), plt.plot(perfil), plt.title('Perfil de intensidad por fila')
plt.xlabel('Fila'), plt.ylabel('Intensidad promedio')
plt.grid(True)
plt.tight_layout()
plt.show()


# --- Detecto límites de la ROI automáticamente ---
perfil_suavizado = np.convolve(perfil, np.ones(20)/20, mode='same')  # Suavizo para evitar falsos picos

#Identificamos la diferencia entre cada elemento y el anterior
derivada = np.diff(perfil_suavizado, prepend=perfil_suavizado[0])

plt.figure(figsize=(10, 4))
plt.subplot(121), plt.plot(perfil_suavizado), plt.title('Perfil suavizado'), plt.grid(True)
plt.subplot(122), plt.plot(derivada), plt.title('Derivada del perfil'), plt.grid(True)
plt.tight_layout()
plt.show()

# El límite superior es la caída más grande en la primera mitad
mitad = len(derivada) // 2
# Límite superior: caída más grande en la primera mitad (metal → cinta)
fila_superior = np.argmin(derivada[:mitad])

# Límite inferior: SUBIDA más grande en la segunda mitad (cinta → metal)
fila_inferior = mitad + np.argmax(derivada[mitad:])

print(f"Límite superior detectado: fila {fila_superior}")
print(f"Límite inferior detectado: fila {fila_inferior}")

# --- Dibujo las líneas sobre la imagen para verificar ---
img_verificacion = img_rgb.copy()
cv2.line(img_verificacion, (0, fila_superior), (img_rgb.shape[1], fila_superior), (255, 0, 0), 3)
cv2.line(img_verificacion, (0, fila_inferior), (img_rgb.shape[1], fila_inferior), (255, 0, 0), 3)

plt.figure()
plt.imshow(img_verificacion)
plt.title(f'ROI detectada: filas {fila_superior} a {fila_inferior}')
plt.axis('off')
plt.show()

# --- Recorto la ROI ---
roi = img_rgb[fila_superior:fila_inferior, :]
roi_gray = img_gray[fila_superior:fila_inferior, :]

plt.figure()
plt.imshow(roi)
plt.title(f'ROI recortada: shape {roi.shape}')
plt.axis('off')
plt.show()

# --- Convierto ROI a HSV ---
roi_hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
H, S, V = cv2.split(roi_hsv)

plt.figure()
ax1 = plt.subplot(221); plt.imshow(roi), plt.title('ROI original'), plt.axis('off')
plt.subplot(222, sharex=ax1, sharey=ax1), plt.imshow(H, cmap='gray'), plt.title('Canal H'), plt.axis('off')
plt.subplot(223, sharex=ax1, sharey=ax1), plt.imshow(S, cmap='gray'), plt.title('Canal S'), plt.axis('off')
plt.subplot(224, sharex=ax1, sharey=ax1), plt.imshow(V, cmap='gray'), plt.title('Canal V'), plt.axis('off')
plt.show()

# --- Histograma del canal V ---
hist_v, bins_v = np.histogram(V.flatten(), 256, [0, 256])

plt.figure()
plt.plot(bins_v[:-1], hist_v)
plt.title('Histograma canal V')
plt.xlabel('Intensidad')
plt.ylabel('Cantidad de píxeles')
plt.grid(True)
plt.show()

# --- Umbralizo canal V en el valle entre los dos picos ---
umbral = 110
_, mask = cv2.threshold(V, umbral, 255, cv2.THRESH_BINARY)

plt.figure()
ax1 = plt.subplot(121), plt.imshow(roi), plt.title('ROI original'), plt.axis('off')
plt.subplot(122), plt.imshow(mask, cmap='gray'), plt.title(f'Máscara (umbral={umbral})'), plt.axis('off')
plt.show()

# --- Morfología: cierre para rellenar huecos pequeños ---
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# --- Busco contornos ---
contornos, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Contornos encontrados: {len(contornos)}")

# --- Dibujo contornos sobre la ROI ---
roi_contornos = roi.copy()
cv2.drawContours(roi_contornos, contornos, -1, (0, 255, 0), 2)

plt.figure()
plt.imshow(roi_contornos)
plt.title(f'Contornos detectados: {len(contornos)}')
plt.axis('off')
plt.show()

# --- Analizo áreas de los contornos ---
areas = [cv2.contourArea(c) for c in contornos]
areas_ordenadas = sorted(areas, reverse=True)

print("Las 10 áreas más grandes:")
for i, a in enumerate(areas_ordenadas[:10]):
    print(f"  {i+1}: {a:.0f} px²")

print(f"\nÁrea mínima: {min(areas):.0f} px²")
print(f"Área máxima: {max(areas):.0f} px²")
print(f"Área promedio: {np.mean(areas):.0f} px²")

# --- Distribución de áreas ---
plt.figure()
plt.hist(areas, bins=50)
plt.title('Distribución de áreas de contornos')
plt.xlabel('Área (px²)')
plt.ylabel('Cantidad de contornos')
plt.grid(True)
plt.show()

# Cuántos contornos tienen área > 500, > 1000, > 2000?
for umbral_area in [500, 1000, 1500, 2000]:
    n = sum(1 for a in areas if a > umbral_area)
    print(f"Contornos con área > {umbral_area}: {n}")

area_roi = roi.shape[0] * roi.shape[1]
print(f"Área total de la ROI: {area_roi} px²")
print(f"1500 px² representa el {100*1500/area_roi:.4f}% del área de la ROI")

# --- Filtro por área mínima Y relación de aspecto ---
area_minima = 0.0014 * area_roi
contornos_filtrados = []

for c in contornos:
    area = cv2.contourArea(c)
    if area < area_minima:
        continue
    x, y, w, h = cv2.boundingRect(c)
    aspect_ratio = w / h  # ancho / alto
    if aspect_ratio > 10:  # si es 10 veces más ancho que alto → es el borde, no una pastilla
        continue
    contornos_filtrados.append(c)

print(f"Contornos después del filtro: {len(contornos_filtrados)}")

roi_contornos = roi.copy()
cv2.drawContours(roi_contornos, contornos_filtrados, -1, (0, 255, 0), 2)
plt.figure()
plt.imshow(roi_contornos)
plt.title(f'Contornos filtrados: {len(contornos_filtrados)}')
plt.axis('off')
plt.show()


# --- Extraigo color promedio de cada pastilla en LAB ---
# Usamos LAB porque las distancias de color son más uniformes que en HSV
roi_lab = cv2.cvtColor(roi, cv2.COLOR_RGB2LAB)

colores_promedio = []
for c in contornos_filtrados:
    # Creo una máscara solo para esta pastilla
    mask_pastilla = np.zeros(roi.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask_pastilla, [c], -1, 255, -1)  # -1 rellena el contorno
    
    # Calculo el color promedio dentro de la máscara
    color_medio = cv2.mean(roi_lab, mask=mask_pastilla)[:3]  # L, A, B
    colores_promedio.append(color_medio)

colores_promedio = np.float32(colores_promedio)
print(f"Shape de colores_promedio: {colores_promedio.shape}")
print("Primeros 5 colores (L, A, B):")
for i in range(5):
    print(f"  Pastilla {i}: L={colores_promedio[i,0]:.1f}, A={colores_promedio[i,1]:.1f}, B={colores_promedio[i,2]:.1f}")

# --- Visualizo los colores promedio en espacio LAB ---
plt.figure()
ax = plt.subplot(111, projection='3d')
# Normalizo para mostrar el color real en el scatter
colores_rgb_norm = []
for lab in colores_promedio:
    # Convierto cada color LAB a RGB para visualizarlo
    pixel_lab = np.uint8([[lab]])
    pixel_rgb = cv2.cvtColor(pixel_lab, cv2.COLOR_LAB2RGB)
    colores_rgb_norm.append(pixel_rgb[0,0] / 255.0)

ax.scatter(colores_promedio[:,1], colores_promedio[:,2], colores_promedio[:,0],
           c=colores_rgb_norm, s=100)
ax.set_xlabel('A'), ax.set_ylabel('B'), ax.set_zlabel('L')
plt.title('Colores promedio de cada pastilla en LAB')
plt.show()


# --- K-means con K=4 sobre los colores promedio en LAB ---
K = 4
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ret, labels, centers = cv2.kmeans(colores_promedio, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

print(f"Labels asignados a cada pastilla:")
print(labels.flatten())

# Cuántas pastillas hay en cada grupo
for k in range(K):
    n = np.sum(labels.flatten() == k)
    print(f"Grupo {k}: {n} pastillas")

# --- Visualizo el color central de cada grupo ---
plt.figure(figsize=(8, 2))
for k in range(K):
    centro_lab = np.uint8([[centers[k]]])
    centro_rgb = cv2.cvtColor(centro_lab, cv2.COLOR_LAB2RGB)
    color_norm = centro_rgb[0, 0] / 255.0
    cantidad = np.sum(labels.flatten() == k)
    
    plt.subplot(1, K, k+1)
    plt.imshow([[color_norm]])
    plt.title(f'Grupo {k}\n{cantidad} pastillas')
    plt.axis('off')

plt.suptitle('Color central de cada grupo K-means')
plt.tight_layout()
plt.show()

# --- Analizo circularidad de las pastillas blancas (grupo 2) ---
indices_blancas = np.where(labels.flatten() == 2)[0]

for i in indices_blancas:
    c = contornos_filtrados[i]
    area = cv2.contourArea(c)
    perimetro = cv2.arcLength(c, True)
    circularidad = 4 * np.pi * area / (perimetro ** 2)
    print(f"  Pastilla {i}: circularidad = {circularidad:.3f}")

# --- Visualizo pastillas blancas con su circularidad ---
roi_circ = roi.copy()
for i in indices_blancas:
    c = contornos_filtrados[i]
    area = cv2.contourArea(c)
    perimetro = cv2.arcLength(c, True)
    circularidad = 4 * np.pi * area / (perimetro ** 2)
    
    # Centro del contorno
    M = cv2.moments(c)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    
    cv2.putText(roi_circ, f"{circularidad:.2f}", (cx-25, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

plt.figure()
plt.imshow(roi_circ)
plt.title('Circularidad de pastillas blancas')
plt.axis('off')
plt.show()

# --- K-means sobre circularidad de las blancas ---
circularidades = []
for i in indices_blancas:
    c = contornos_filtrados[i]
    area = cv2.contourArea(c)
    perimetro = cv2.arcLength(c, True)
    circularidad = 4 * np.pi * area / (perimetro ** 2)
    circularidades.append([circularidad])  # lista de listas porque kmeans espera array 2D

circularidades = np.float32(circularidades)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ret, labels_circ, centers_circ = cv2.kmeans(circularidades, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

print(f"Centros de los 2 grupos: {centers_circ.flatten()}")
print(f"Grupo 0: {np.sum(labels_circ.flatten()==0)} pastillas")
print(f"Grupo 1: {np.sum(labels_circ.flatten()==1)} pastillas")

# --- Visualizo separación redondas vs cuadradas ---
roi_formas = roi.copy()

# Identifico cuál centro es el más bajo (cuadradas)
idx_cuadradas = 0 if centers_circ[0] < centers_circ[1] else 1
idx_redondas = 1 - idx_cuadradas

for j, i in enumerate(indices_blancas):
    c = contornos_filtrados[i]
    M = cv2.moments(c)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    
    if labels_circ[j] == idx_cuadradas:
        color = (255, 0, 0)   # rojo = cuadradas
        texto = "C"
    else:
        color = (0, 200, 0)   # verde = redondas
        texto = "R"
    
    cv2.putText(roi_formas, texto, (cx-8, cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

plt.figure()
plt.imshow(roi_formas)
plt.title('Blancas: R=redonda (verde), C=cuadrada (rojo)')
plt.axis('off')
plt.show()

# --- Asigno tipo y ID a cada pastilla ---
conteos = {'BR': 0, 'BC': 0, 'AP': 0, 'RR': 0, 'AzC': 0}

# Identifico qué grupo K-means de color corresponde a cada tipo
# (según lo que vimos en la visualización de colores)
# Grupo 0: amarillas, Grupo 1: rosadas, Grupo 2: blancas, Grupo 3: azules
idx_cuadradas = 0 if centers_circ[0] < centers_circ[1] else 1

etiquetas = []  # lista con (contorno, tipo, id)

j_blanca = 0  # contador para recorrer labels_circ
for i, c in enumerate(contornos_filtrados):
    grupo_color = labels.flatten()[i]
    
    if grupo_color == 0:
        tipo = 'AP'
    elif grupo_color == 1:
        tipo = 'RR'
    elif grupo_color == 3:
        tipo = 'AzC'
    else:  # grupo 2: blancas, separamos por forma
        if labels_circ[j_blanca] == idx_cuadradas:
            tipo = 'BC'
        else:
            tipo = 'BR'
        j_blanca += 1
    
    conteos[tipo] += 1
    etiquetas.append((c, tipo, conteos[tipo]))

# --- Reporte por consola ---
print("=== RESULTADOS DE CLASIFICACIÓN ===")
for tipo, n in conteos.items():
    print(f"  {tipo}: {n} pastillas")
print(f"  TOTAL: {sum(conteos.values())} pastillas")

# --- Genero imagen final con etiquetas ---
img_resultado = roi.copy()

for c, tipo, id_pastilla in etiquetas:
    # Centro del contorno
    M = cv2.moments(c)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    
    # Dibujo contorno
    cv2.drawContours(img_resultado, [c], -1, (0, 255, 0), 2)
    
    # Etiqueta
    etiqueta = f"{tipo}{id_pastilla}"
    cv2.putText(img_resultado, etiqueta, (cx-20, cy+5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

plt.figure(figsize=(14, 6))
plt.imshow(img_resultado)
plt.title('Resultado final')
plt.axis('off')
plt.show()