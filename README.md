# Real-time Loadcell & RPM Viewer

## Overview
This program is designed to measure, visualize, and log data from 2 loadcells and an encoder motor (RPM) in real-time. It's built for lab experiments, providing a user-friendly interface to manage device connections, monitor data, and save experiment logs.

## Key Features
- **Manual Connection Control**: Connect to and disconnect from your serial device manually.
- **Persistent Port Name**: The last used serial port is automatically saved and reloaded.
- **Live Data Visualization**: Displays data from 2 loadcells and motor RPM on separate, real-time graphs.
- **Current Value & Hover Display**: Shows the latest numerical values and provides detailed data points on mouse hover.
- **Experiment Naming**: Assign a name to each experiment, which is then included in the saved filename for easy identification.
- **Data Logging**: Use the "Start"/"Stop" button to log specific segments of the data stream.
- **Live Log Viewer**: A scrollable log viewer at the bottom of the window shows raw data coming from the serial port.
- **CSV Export**: Logged data is saved to a CSV file containing timestamps, loadcell values, and RPM.
- **Mock Mode**: Test the application's full functionality without any physical hardware.

---

## Installation

1. **Python 3.x 설치**
2. **가상환경(venv) 생성 및 활성화**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Windows는 venv\Scripts\activate
   ```
3. **필수 패키지 설치**
   ```bash
   pip install -r requirements.txt
   ```

---

## How to Use

1.  **Set Serial Port & Connect**:
    -   Run the program (`python main.py`).
    -   At the top-left, enter the serial port name for your device (e.g., `COM3` on Windows, or `/dev/tty.usbmodemXXXX` on macOS).
    -   Click the **"Connect"** button. The button will turn red ("Disconnect") and the log viewer at the bottom will confirm the connection.

2.  **Name Your Experiment (Optional)**:
    -   In the "Experiment:" text box at the top, give your current test a name (e.g., `15_knots`).

3.  **Start Logging**:
    -   Click the green **"Start"** button to begin logging data. The status window will show "Logging...".

4.  **Stop Logging & Save**:
    -   Click the red **"Stop"** button. The logging will stop, and the data will be saved to a CSV file.
    -   The filename will automatically include the date, time, and the experiment name you entered.
    -   Example Filename: `measurement_20240608_153000_15_knots.csv`

5.  **Disconnect**:
    -   When you're finished, click the **"Disconnect"** button.

---

## Data Format Example

- **로드셀 데이터**: `[LoadCell] Cell 1: -57 | Cell 2: 7`
- **모터 데이터**: `[Motor] RPM:1203, PWM: 122`
- **저장 CSV**: `rel_time,cell1,cell2,rpm`

---

## Other Notes
- The entire UI is in English to prevent font compatibility issues.
- If the graph doesn't appear or the connection fails, double-check your serial port name and device connection.
- Additional features or customizations can be requested. 