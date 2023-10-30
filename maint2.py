import cv2
import os
import tkinter as tk
from PIL import Image, ImageTk



# Función para el seguimiento de sueño del bebé
def track_sleep(frame, label):
    # Aquí se realizaría la detección de movimiento y análisis de postura del bebé
    # Agrega tu lógica de detección de patrones de sueño aquí

    # Ejemplo básico de detección de movimiento y postura
    cv2.putText(frame, 'Sleep Tracking', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    label.config(text="Sleep Tracking")

def detect_emotions(frame, face_cascade, smile_cascade, emotion_cascade, label):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20)
        for (ex, ey, ew, eh) in smiles:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

        emotions = emotion_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (ex, ey, ew, eh) in emotions:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), 2)

            if ew > 30 and eh > 30:
                cv2.putText(roi_color, 'Emotion Detected', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                label.config(text="Emotion Detected")

def main():
    # Obtén la ruta completa a los archivos haarcascade_frontalface_default.xml, haarcascade_smile.xml y haarcascade_eye.xml en el directorio actual
    current_directory = os.path.dirname(os.path.abspath(__file__))
    face_cascade = cv2.CascadeClassifier(os.path.join(current_directory, 'haarcascade_frontalface_default.xml'))
    smile_cascade = cv2.CascadeClassifier(os.path.join(current_directory, 'haarcascade_smile.xml'))
    emotion_cascade = cv2.CascadeClassifier(os.path.join(current_directory, 'haarcascade_eye.xml'))

    # Configuración de Tkinter
    root = tk.Tk()
    root.bind('<Escape>', lambda e: root.quit())

    label = tk.Label(root, text="")
    label.pack()

    def start_detection():
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            detect_emotions(frame, face_cascade, smile_cascade, emotion_cascade, label)
            track_sleep(frame, label)
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            label.imgtk = imgtk
            label.config(image=imgtk)

            root.update()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def stop_detection():
        cv2.destroyAllWindows()
        # Aquí iría el código para detener la detección de emociones
        pass

    start_button = tk.Button(root, text="Iniciar detección", command=start_detection)
    start_button.pack()

    stop_button = tk.Button(root, text="Detener detección", command=stop_detection)
    stop_button.pack()

    root.mainloop()

if __name__ == "__main__":
    main()


