from ultralytics import YOLO
import cv2
import os
import numpy as np
import tensorflow as tf
from deep_sort_realtime.deepsort_tracker import DeepSort

# Cargar el modelo preentrenado YOLOv11

modeloDeep = tf.keras.models.load_model('modelo (2).keras')
modeloObjetos = YOLO("best.pt")
print("Modelo cargado correctamente")


video_path = "Un exmarine desarma a dos atracadores en la cola de una gasolinera en cuestión de segundos.mp4"

def preprocesar_imagen(frame):
    IMG_SIZE = 112
    """Convierte la imagen al formato esperado por el modelo."""
    if frame is None or frame.size == 0:
        return None
    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))  # Redimensionar
    frame = frame.astype(np.float32) / 255.0  # Normalizar
    return frame

def mostrar_video_con_yolo(video_path):
    SEQUENCE_LENGTH = 25 
    cap = cv2.VideoCapture(video_path)
    frames = []  # Lista para almacenar los frames de video
    contador = 0

    while cap.isOpened():
        success, frame = cap.read()
        if success:
            # Aplicar YOLO para detección de objetos
            results = modeloObjetos.track(frame, persist=True, conf=0.6)
            annotated_frame = results[0].plot()

            #Usar los frames de video para el modelo de Deep Learning
            # Si tenemos suficientes frames, predecir
            if len(frames) >= SEQUENCE_LENGTH:
                input_batch = np.expand_dims(np.array(frames), axis=0)
                predicciones = modeloDeep.predict(input_batch)
                predicciones = np.argmax(predicciones, axis=1)
                if predicciones[0] == 1:
                    contador += 1
                else:
                    contador -=1
                frames = []  # Reiniciar la lista de frames después de la predicción
            else:
                # Preprocesar la imagen y agregarla a la lista de frames
                frame_preprocesada = preprocesar_imagen(frame)
                if frame_preprocesada is not None:
                    frames.append(frame_preprocesada)
                    
            if contador > 0:
                    cv2.putText(annotated_frame, "ADVERTENCIA", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("YOLOv11", annotated_frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()
mostrar_video_con_yolo(video_path)

        

