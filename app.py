import cv2
import numpy as np
import wyzecam


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.label = QLabel()
        self.setCentralWidget(self.label)

        # Crea una barra de herramientas
        self.toolbar = QToolBar()
        self.addToolBar(self.toolbar)

        # Crea un botón para iniciar la detección de movimiento
        self.action_start = QAction("Iniciar", self)
        self.action_start.triggered.connect(self.start_detection)
        self.toolbar.addAction(self.action_start)

        # Crea un botón para detener la detección de movimiento
        self.action_stop = QAction("Detener", self)
        self.action_stop.triggered.connect(self.stop_detection)
        self.toolbar.addAction(self.action_stop)

        # Muestra la ventana
        self.show()

    def start_detection(self):
        # Inicia la detección de movimiento
        self.cap = wyzecam.WyzeCam("YOUR_CAMERA_ID", "YOUR_PASSWORD")
        while True:
            # Captura un marco de la cámara
            ret, frame = self.cap.read()

            # Detecta movimiento en el marco
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]
            contours, hierarchy = cv2.findContours(
                threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Encuentra los ojos
            eyes = []
            for contour in contours:
                if cv2.contourArea(contour) > 100:
                    eyes.append(contour)

            # Determina si los ojos están abiertos
            is_open = False
            for eye in eyes:
                # Calcula el centro del ojo
                (x, y), radius = cv2.minEnclosingCircle(eye)

                # Calcula el área del ojo
                area = cv2.contourArea(eye)

                # Determina si el ojo está abierto
                is_open = (area / (math.pi * radius ** 2)) > 0.5

                # Muestra el ojo
                cv2.circle(frame, (x, y), radius, (0, 255, 0), 2)

            # Muestra el marco en la etiqueta
            self.label.setPixmap(QPixmap.fromImage(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

            # Si los ojos están abiertos, muestra una notificación
            if is_open:
                print("Los ojos están abiertos!")

    def stop_detection(self):
        # Detiene la detección de movimiento
        self.cap.release()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
