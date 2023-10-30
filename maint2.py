import cv2
import os
import tkinter as tk
from PIL import Image, ImageTk

def detect_smile(frame, face_cascade, smile_cascade, label):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    smile_detected = False
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20)

        if len(smiles) > 0:
            smile_detected = True
            cv2.putText(roi_color, 'Smile Detected', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        for (ex, ey, ew, eh) in smiles:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    if smile_detected:
        label.config(text="Sonrisa detectada")
    else:
        label.config(text="")

def main():
    # Obtén la ruta completa a los archivos haarcascade_frontalface_default.xml y haarca

    current_directory = os.path.dirname(os.path.abspath(__file__))
    face_cascade = cv2.CascadeClassifier(os.path.join(current_directory, 'haarcascade_frontalface_default.xml'))
    smile_cascade = cv2.CascadeClassifier(os.path.join(current_directory, 'haarcascade_smile.xml'))

    # Configuración de Tkinter
    root = tk.Tk()
    root.bind('<Escape>', lambda e: root.quit())
    label = tk.Label(root, text="")
    label.pack()

    # Abre la cámara
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        detect_smile(frame, face_cascade, smile_cascade, label)

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

if __name__ == "__main__":
    main()

