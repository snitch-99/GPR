import sys
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QLabel,
    QHBoxLayout, QVBoxLayout, QFileDialog, QProgressBar
)
from PyQt5.QtCore import Qt, QTimer, QPoint
from PyQt5.QtGui import QPainter, QPen


# --- CSV Loader ---
class CSVLoader:
    def __init__(self):
        self.timestamps = []
        self.latitudes = []
        self.longitudes = []
        self.total_points = 0

    def prompt_and_load_csv(self):
        file_path, _ = QFileDialog.getOpenFileName(None, "Select CSV File", "", "CSV Files (*.csv)")
        if not file_path:
            return False
        return self.load_csv(file_path)

    def load_csv(self, file_path):
        try:
            df = pd.read_csv(file_path)
            if df.shape[1] < 3:
                print("CSV must have at least 3 columns (timestamp, latitude, longitude)")
                return False
            self.timestamps = df.iloc[:, 0].values
            self.latitudes = df.iloc[:, 1].values
            self.longitudes = df.iloc[:, 2].values
            self.total_points = len(self.latitudes)
            print(f"CSVLoader: Loaded {self.total_points} points.")
            return True
        except Exception as e:
            print("CSVLoader Error:", e)
            return False


# --- Custom MapCanvas with trace support ---
class MapCanvas(QLabel):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("""
            background-color: lightblue;
            border: 2px solid black;
        """)
        self.setAlignment(Qt.AlignCenter)
        self.setFixedSize(700, 600)
        self.trace_points = []

    def paintEvent(self, event):
        super().paintEvent(event)
        if len(self.trace_points) < 2:
            return
        painter = QPainter(self)
        pen = QPen(Qt.red, 2)
        painter.setPen(pen)
        for i in range(1, len(self.trace_points)):
            painter.drawLine(self.trace_points[i - 1], self.trace_points[i])


# --- Movement Control ---
class MovementControl:
    def __init__(self, map_widget: MapCanvas, ui_callback=None):
        self.map_widget    = map_widget
        self.ui_callback   = ui_callback
        self.current_index = 0
        self.dot           = None
        self.timestamps    = []
        self.latitudes     = []
        self.longitudes    = []
        self.total_points  = 0
        self._stop_flag    = False

    def load_data(self, timestamps, latitudes, longitudes):
        self.timestamps              = timestamps
        self.latitudes               = latitudes
        self.longitudes              = longitudes
        self.total_points            = len(latitudes)
        self.current_index           = 0
        self.map_widget.trace_points = []
        print("MovementControl: Data loaded.")
        self.spawn_dot(latitudes[0], longitudes[0])

    def start(self):
        if self.total_points == 0:
            return
        self._stop_flag = False
        if self.ui_callback:
            self.ui_callback("start")
        self.current_index = 0
        self.update()

    def update(self):
        if self._stop_flag or self.current_index >= self.total_points:
            self.stop()
            return

        lat = self.latitudes[self.current_index]
        lon = self.longitudes[self.current_index]
        self.spawn_dot(lat, lon)

        if self.ui_callback:
            percent = int((self.current_index / self.total_points) * 100)
            self.ui_callback("progress", percent)

        if self.current_index < self.total_points - 1:
            t_curr = float(self.timestamps[self.current_index])
            t_next = float(self.timestamps[self.current_index + 1])
            delta_ms = max(int((t_next - t_curr) * 1000), 1)
            self.current_index += 1
            QTimer.singleShot(delta_ms, self.update)
        else:
            self.current_index += 1
            self.stop()

    def stop(self):
        self._stop_flag = True
        if self.ui_callback:
            self.ui_callback("stop")
        print("Animation stopped.")

    def latlon_to_pixels(self, lat, lon):
        lat_min = min(self.latitudes)
        lat_max = max(self.latitudes)
        lon_min = min(self.longitudes)
        lon_max = max(self.longitudes)

        width = self.map_widget.width()
        height = self.map_widget.height()

        x = int((lon - lon_min) / (lon_max - lon_min + 1e-9) * width)
        y = int((1 - (lat - lat_min) / (lat_max - lat_min + 1e-9)) * height)
        return x, y

    def spawn_dot(self, lat, lon):
        x, y = self.latlon_to_pixels(lat, lon)
        self.map_widget.trace_points.append(QPoint(x, y))
        self.map_widget.update()

        if self.dot:
            self.dot.deleteLater()

        self.dot = QLabel(self.map_widget)
        self.dot.setStyleSheet("background-color: red; border-radius: 5px;")
        self.dot.setFixedSize(10, 10)
        self.dot.move(x - 5, y - 5)
        self.dot.show()


# --- Main GUI Window ---
class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Boat Simulation")
        self.setGeometry(100, 100, 1000, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        main_layout = QHBoxLayout()
        self.central_widget.setLayout(main_layout)

        self.map_area = MapCanvas()

        self.button_panel = QWidget()
        button_layout = QVBoxLayout()
        self.button_panel.setLayout(button_layout)

        self.select_csv_btn = QPushButton("Select CSV")
        self.start_btn = QPushButton("Start")
        self.stop_btn = QPushButton("Stop")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)

        button_layout.addWidget(self.select_csv_btn)
        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.stop_btn)
        button_layout.addWidget(self.progress_bar)
        button_layout.addStretch()

        main_layout.addWidget(self.map_area)
        main_layout.addWidget(self.button_panel)

        self.loader = CSVLoader()
        self.movement = MovementControl(self.map_area, self.handle_ui_update)

        self.select_csv_btn.clicked.connect(self.select_csv)
        self.start_btn.clicked.connect(self.movement.start)
        self.stop_btn.clicked.connect(self.movement.stop)

    def select_csv(self):
        if self.loader.prompt_and_load_csv():
            self.movement.load_data(
                self.loader.timestamps,
                self.loader.latitudes,
                self.loader.longitudes
            )

    def handle_ui_update(self, command, value=None):
        if command == "start":
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
        elif command == "stop":
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.progress_bar.setValue(100)
        elif command == "progress" and value is not None:
            self.progress_bar.setValue(value)


# --- Run App ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())
