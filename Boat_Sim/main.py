import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QHBoxLayout, QVBoxLayout, QPushButton, QLabel,
    QFileDialog
)
from PyQt5.QtCore import Qt, QTimer
import pandas as pd

class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dot Tracker GUI")
        self.setGeometry(100, 100, 1000, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Main layout
        main_layout = QHBoxLayout()
        self.central_widget.setLayout(main_layout)

        # Light blue map placeholder
        self.map_area = QLabel("")  # FIXED: this was missing before
        self.map_area.setStyleSheet("""
            background-color: lightblue;
            border: 2px solid black;
            font-size: 16px;
            font-weight: bold;
        """)
        self.map_area.setAlignment(Qt.AlignCenter)
        self.map_area.setFixedSize(700, 600)

        # Button panel
        self.button_panel = QWidget()
        button_layout = QVBoxLayout()
        self.button_panel.setLayout(button_layout)

        self.select_csv_btn = QPushButton("Select CSV")
        self.start_btn = QPushButton("Start")
        self.stop_btn = QPushButton("Stop")
        button_layout.addWidget(self.select_csv_btn)
        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.stop_btn)
        button_layout.addStretch()

        # Add widgets to layout
        main_layout.addWidget(self.map_area)
        main_layout.addWidget(self.button_panel)

        # Button connections
        self.select_csv_btn.clicked.connect(self.select_csv)
        self.start_btn.clicked.connect(self.start_animation)
        self.stop_btn.clicked.connect(self.stop_animation)

        # Timer setup
        self.timer = QTimer()
        self.timer.timeout.connect(self.move_dot)
        self.current_index = 0

    def select_csv(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select CSV File", "", "CSV Files (*.csv)")
        if not file_path:
            return

        try:
            df = pd.read_csv(file_path)
            if df.shape[1] < 3:
                print("CSV must have at least 3 columns (timestamp, latitude, longitude)")
                return

            self.latitudes = df.iloc[:, 1].values
            self.longitudes = df.iloc[:, 2].values
            self.total_points = len(self.latitudes)
            print(f"Loaded {self.total_points} points.")
            print("First lat/lon:", self.latitudes[0], self.longitudes[0])

            self.spawn_dot(self.latitudes[0], self.longitudes[0])
        except Exception as e:
            print("Error loading CSV:", e)

    def latlon_to_pixels(self, lat, lon):
        lat_min = min(self.latitudes)
        lat_max = max(self.latitudes)
        lon_min = min(self.longitudes)
        lon_max = max(self.longitudes)

        width = self.map_area.width()
        height = self.map_area.height()

        x = int((lon - lon_min) / (lon_max - lon_min + 1e-9) * width)
        y = int((1 - (lat - lat_min) / (lat_max - lat_min + 1e-9)) * height)

        return x, y

    def spawn_dot(self, lat, lon):
        if hasattr(self, 'dot'):
            self.dot.deleteLater()

        x, y = self.latlon_to_pixels(lat, lon)

        self.dot = QLabel(self.map_area)
        self.dot.setStyleSheet("background-color: red; border-radius: 5px;")
        self.dot.setFixedSize(10, 10)
        self.dot.move(x - 5, y - 5)
        self.dot.show()

    def start_animation(self):
        if not hasattr(self, 'latitudes') or self.total_points == 0:
            print("Please load a valid CSV first.")
            return

        self.current_index = 0
        self.timer.start(5)  # in milliseconds

    def stop_animation(self):
        self.timer.stop()
        print("Animation stopped.")

    def move_dot(self):
        if self.current_index >= self.total_points:
            self.timer.stop()
            print("Reached end of path.")
            return

        lat = self.latitudes[self.current_index]
        lon = self.longitudes[self.current_index]
        self.spawn_dot(lat, lon)
        self.current_index += 1

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())
