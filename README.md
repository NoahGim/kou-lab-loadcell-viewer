# Real-time Loadcell & RPM Viewer

## Overview
This program is designed to measure, visualize, and log data from up to 4 loadcells and an encoder motor (RPM) in real-time. It provides a user-friendly interface to manage device connections, calibrate sensors, monitor data, and save experiment logs.

## Key Features
- **Flexible Sensor Configuration**: Supports up to 4 loadcell channels.
- **Live Data Visualization**: Displays data from each loadcell and motor RPM on separate, real-time graphs.
- **Integrated Magnus Force & RPM Graph**: A dedicated graph visualizes the total Magnus force (calculated from all loadcells) and RPM on a shared timeline with dual Y-axes.
- **Data Logging with Magnus Force**: Use the "Start"/"Stop" button to log specific segments of the data stream. Logged data includes individual cell weights, RPM, and the calculated Magnus force.
- **Sensor Calibration**: A dedicated calibration window allows you to set scale and offset values for each loadcell channel to convert raw data into grams. Calibration settings are saved and automatically reloaded.
- **Manual Connection Control**: Connect to and disconnect from your serial device manually.
- **Live Log Viewer**: A scrollable log viewer at the bottom of the window shows raw data coming from the serial port or application status messages.
- **Mock Mode**: Test the application's full functionality without any physical hardware.
- **Persistent Settings**: The serial port and calibration parameters are saved to `config.json` and `calibration.json` respectively, and are loaded on startup.

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
    -   At the top-left, enter the serial port name for your device (e.g., `COM3` on Windows, or `/dev/tty.usbmodemXXXX` on macOS). To use the mock mode, enter `mock`.
    -   Click the **"Connect"** button. The button will turn red ("Disconnect") and the log viewer at the bottom will confirm the connection.

2.  **Calibrate Sensors (Recommended)**:
    - After connecting, click the **"Calibrate"** button on the right-side panel.
    - In the new window, you can input known weights to automatically calculate the `scale` or manually enter `scale` and `offset` values for each channel.
    - Click **"Save & Close"**. The new parameters will be applied immediately and saved in `calibration.json` for future sessions.

3.  **Name Your Experiment (Optional)**:
    -   In the "Experiment:" text box at the top, give your current test a name (e.g., `15_knots`).

4.  **Start Logging**:
    -   Click the green **"Start"** button to begin logging data. The status window will show "Logging...".

5.  **Stop Logging & Save**:
    -   Click the red **"Stop"** button. The logging will stop, and the data will be saved to a CSV file.
    -   The filename will automatically include the date, time, and the experiment name you entered.
    -   Example Filename: `measurement_20240608_153000_15_knots.csv`

6.  **Disconnect**:
    -   When you're finished, click the **"Disconnect"** button.

---

## Data Format Example

- **로드셀 데이터**: `[LoadCell_input] Cell 1: -57 | Cell 4: 100 ...` (Raw data example)
- **모터 데이터**: `[Motor] RPM:1203, PWM: 122`
- **저장 CSV 컬럼**: `rel_time,cell1,cell2,cell3,cell4,rpm,magnus_force`

---

## Configuration Files

- **`config.json`**: Stores the last used serial port name.
- **`calibration.json`**: Stores the scale and offset for each loadcell channel. You can manually edit this file, but it's recommended to use the in-app calibration window.

---

## Other Notes
- The entire UI is in English to prevent font compatibility issues.
- If the graph doesn't appear or the connection fails, double-check your serial port name and device connection.
- Additional features or customizations can be requested. 

## calibration factor

calibration.json
```json
{
  "cell_1": 3192.13630406291,
  "cell_2": 3003.59778597786,
  "cell_3": 3375.276752767528,
  "cell_4": 3341.5366972477063
}
```

