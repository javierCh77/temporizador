import cv2
import numpy as np
import time

# Definir las 6 áreas de detección (3 arriba, 3 abajo)
ROIS = [
    (75, 50, 100, 150),    # Área superior izquierda
    (275, 50, 100, 150),   # Área superior central
    (475, 50, 100, 150),   # Área superior derecha
    (75, 250, 100, 150),   # Área inferior izquierda
    (275, 250, 100, 150),  # Área inferior central
    (475, 250, 100, 150)   # Área inferior derecha
]

# Estados de temporizador para cada área (None si no está corriendo, tiempo final si está corriendo)
temporizadores = [None] * len(ROIS)

def detectar_tonalidad_negra(frame, roi):
    """Detecta áreas negras dentro de la región especificada."""
    x, y, w, h = roi
    frame_roi = frame[y:y+h, x:x+w]

    # Convertir la imagen a escala de grises
    gris = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)

    # Definir el rango de negro en escala de grises
    _, umbral = cv2.threshold(gris, 50, 255, cv2.THRESH_BINARY_INV)

    # Contar el número de píxeles negros
    area_negra = np.sum(umbral == 255)

    # Ajustar el umbral según el tamaño del área para considerar como detección
    umbral_area = (w * h) * 0.10  # 10% del área de la ROI

    if area_negra > umbral_area:
        # Dibujar el área detectada en la ROI
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return True
    return False

def mostrar_temporizador(frame, tiempo_restante, roi):
    """Muestra el tiempo restante en la ROI correspondiente."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y, w, h = roi
    cv2.putText(frame, f"Tiempo: {tiempo_restante}s", (x, y - 10), font, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

def manejar_temporizadores(frame, roi_index):
    """Maneja el temporizador para la ROI específica."""
    if temporizadores[roi_index] is None:
        # Si el temporizador aún no ha sido iniciado, lo inicializamos
        temporizadores[roi_index] = time.time() + 50  # Iniciar temporizador de 10 segundos

    tiempo_restante = int(temporizadores[roi_index] - time.time())

    if tiempo_restante <= 0:
        print(f"¡Tiempo terminado para el área {roi_index + 1}! Retirar objeto.")
        temporizadores[roi_index] = None  # Reiniciar el temporizador cuando el tiempo termine
    else:
        mostrar_temporizador(frame, tiempo_restante, ROIS[roi_index])

def main():
    global cap
    cap = cv2.VideoCapture(0)  # Iniciar la cámara

    while True:
        ret, frame = cap.read()  # Leer un frame
        objeto_detectado_en_areas = [False] * len(ROIS)  # Lista de detección para cada ROI

        # Dibujar las 6 áreas de detección en la imagen
        for i, roi in enumerate(ROIS):
            x, y, w, h = roi
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Dibujar el rectángulo ROI

            # Detectar tonalidad negra en el área
            if detectar_tonalidad_negra(frame, roi):
                objeto_detectado_en_areas[i] = True

        # Procesar los temporizadores para cada área
        for i, detectado in enumerate(objeto_detectado_en_areas):
            if detectado:
                manejar_temporizadores(frame, i)  # Manejar temporizador si se detecta
            else:
                # Reiniciar el temporizador si el objeto ya no está
                if temporizadores[i] is not None:
                    temporizadores[i] = None  # Reiniciar el temporizador al perder la detección

        # Mostrar la imagen con las ROIs y temporizadores
        cv2.imshow('Detección de tonalidad negra en ROIs', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Salir con 'q'
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
