from ultralytics import YOLO
import cv2
import os

# Cargar el modelo preentrenado YOLOv8
modelo = YOLO("yolov8n.pt")
print("Modelo cargado correctamente")

# Ruta de la carpeta con videos
carpeta_videos = r"C:\Users\braya\Documents\modeloVideosPersonas" # Cambia esta ruta a tu carpeta con videos
output_carpeta = r"C:\Users\braya\Documents\modeloEntrenados"  # Carpeta para guardar los videos procesados
os.makedirs(output_carpeta, exist_ok=True)  # Crear carpeta si no existe

# Obtener la lista de videos en la carpeta
videos = [os.path.join(carpeta_videos, f) for f in os.listdir(carpeta_videos) if f.endswith(('.mp4', '.avi', '.mov'))]
print("Videos encontrados:", videos)

if not videos:
    print("No se encontraron videos en la carpeta.")
    exit()  # Salir si no hay videos

for video_path in videos:
    # Abrir el video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"No se pudo abrir el video {video_path}")
        continue
    else:
        print(f"Abriendo video: {video_path}")

    output_path = os.path.join(output_carpeta, f"output_{os.path.basename(video_path)}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Procesar los fotogramas
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("No se pudo leer el fotograma.")
            break
        
        print("Procesando fotograma...")

        # Realizar detección de personas
        resultados = modelo(frame)

        # Dibujar los bounding boxes en la imagen
        for resultado in resultados[0].boxes:  # Asegúrate de acceder al primer resultado
            x1, y1, x2, y2 = map(int, resultado.xyxy[0])  # Coordenadas de la caja
            conf = resultado.conf[0].item()  # Confianza de detección
            cls = int(resultado.cls[0].item())  # Clase detectada

            # Solo dibujar si es una persona (clase 0 en COCO dataset)
            if cls == 0:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'Persona: {conf:.2f}', (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Mostrar el fotograma procesado
        cv2.imshow("Video con Detecciones", frame)

        # Esperar 1 ms para cerrar la ventana con 'q' y continuar con el siguiente fotograma
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Cerrando ventana de visualización.")
            break

        # Guardar el fotograma procesado
        out.write(frame)

    # Liberar recursos después de procesar el video
    cap.release()
    out.release()

# Cerrar todas las ventanas de OpenCV al finalizar
cv2.destroyAllWindows()

print("Procesamiento completado.")
