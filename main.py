import cv2
import os
def main():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    cascade_path = os.path.join(current_directory, 'haarcascade_frontalface_default.xml')

    # Carga el clasificador para detectar rostros
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # Abre la cámara
    cap = cv2.VideoCapture(0)

    while True:
        # Captura frame por frame
        ret, frame = cap.read()

        # Convierte el frame a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detecta rostros en el frame
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Dibuja un rectángulo alrededor de cada rostro detectado
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Muestra el frame resultante
        cv2.imshow('Video', frame)

        # Detén el ciclo cuando se presione la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libera la cámara y destruye las ventanas abiertas
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
