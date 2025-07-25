import serial
import threading
import time
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from utils.parser import parse_loadcell, parse_motor
import random
import pandas as pd
from matplotlib.widgets import Button, TextBox
import datetime
import matplotlib
import numpy as np
import matplotlib.font_manager as fm
import json
import os
from matplotlib.gridspec import GridSpec
from scipy.stats import linregress
from calibration_window import CalibrationWindow

# 시리얼 포트 설정 (포트명은 환경에 맞게 수정)
SERIAL_PORT = '/dev/tty.usbserial-1130'  # macOS 예시, Windows는 'COM3' 등
BAUDRATE = 9600

# --- 설정 상수 ---
MAX_CELLS = 4

# 데이터 버퍼
MAXLEN = 500
loadcell_data = deque(maxlen=MAXLEN)
rpm_data = deque(maxlen=MAXLEN)
log_view_data = [] # Deque가 아닌 일반 리스트로 변경하여 모든 로그 저장
log_view_offset = 0 # 로그 뷰어 스크롤 위치
LOG_VIEW_LINES = 5 # 로그 뷰어에 표시할 줄 수

# 로깅 관련 변수
data_logging = False
log_buffer = []  # 각 원소: (timestamp, cell1, cell2, rpm, magnus_force)
log_start_time = None  # 측정 시작 시각
last_logged_cell_values = [np.nan] * MAX_CELLS # 로깅 시 사용될 마지막 셀 값

# 전역 마이너스 부호 설정 (유지하는 것이 좋음)
matplotlib.rcParams['axes.unicode_minus'] = False

# 상태 변수
is_connected = False
serial_thread = None
stop_thread = threading.Event()
ser = None # 시리얼 객체

# 캘리브레이션 파라미터 (단순화된 구조: scale 값만 저장)
calibration_params = {f"cell_{i+1}": 1.0 for i in range(MAX_CELLS)}

# --- 설정 관리 ---
def save_config(port):
    with open('config.json', 'w') as f:
        json.dump({'serial_port': port}, f)

def load_config():
    if os.path.exists('config.json'):
        with open('config.json', 'r') as f:
            return json.load(f).get('serial_port', '')
    return ''

def load_calibration_params():
    global calibration_params
    try:
        if os.path.exists('calibration.json'):
            with open('calibration.json', 'r') as f:
                params = json.load(f)
                # 단순화된 JSON 구조에 맞게 파라미터 로드
                for i in range(MAX_CELLS):
                    key = f"cell_{i+1}"
                    if key in params:
                        # offset 없이 scale 값만 직접 할당
                        calibration_params[key] = float(params[key])
            print("[Info] Calibration parameters loaded.")
    except Exception as e:
        print(f"[Error] Failed to load calibration file: {e}")

def save_calibration_params(params):
    global calibration_params, ser, is_connected
    calibration_params = params
    try:
        with open('calibration.json', 'w') as f:
            json.dump(params, f, indent=2)
        print("[Info] Calibration parameters saved.")
        
        # Arduino에 변경된 calibration 값 즉시 전송
        if ser and is_connected:
            print("[Info] Sending updated calibration factors to device...")
            for i in range(MAX_CELLS):
                key = f"cell_{i+1}"
                if key in params:
                    factor = params[key]
                    command = f"[LoadCell_{i+1}] cal:{factor}\n"
                    ser.write(command.encode('utf-8'))
                    time.sleep(1) # 각 명령 전송 후 짧은 딜레이
            print("[Info] Updated calibration factors sent.")
            
    except Exception as e:
        print(f"[Error] Failed to save or send calibration file: {e}")

# --- 시리얼 통신 스레드 ---
def serial_reader(port, baud, stop_event):
    global ser, is_connected, log_view_offset, log_buffer, rpm_data, last_logged_cell_values
    try:
        ser = serial.Serial(port, baud, timeout=1) # 타임아웃 1초로 명확하게 설정
        
        # "Initialize complete" 핸드셰이크
        init_complete = False
        log_view_data.append("[Info] Waiting for device to initialize...")
        start_time = time.time()
        while time.time() - start_time < 20: # 5초 타임아웃
            line = ser.readline().decode(errors='ignore').strip()
            if line:
                log_view_data.append(line)
            if "Initialize complete" in line:
                is_connected = True
                init_complete = True
                log_view_data.append(f"[Info] Connected to {port}")
                break
        
        if not init_complete:
            log_view_data.append("[Error] Device initialization timeout.")
            is_connected = False
            if ser: ser.close()
            return
            
        # 연결 성공 후, calibration.json 값 전송 및 tare 실행
        log_view_data.append("[Info] Sending calibration and taring...")
        for i in range(1, MAX_CELLS + 1):
            key = f"cell_{i}"
            factor = calibration_params[key]
            # 1. Calibration 값 전송
            cal_command = f"[LoadCell_{i}] cal:{factor}\n"
            ser.write(cal_command.encode('utf-8'))
            time.sleep(1)
            # 2. Tare 명령 전송
            tare_command = f"[LoadCell_{i}] tare\n"
            ser.write(tare_command.encode('utf-8'))
            time.sleep(1)
        log_view_data.append("[Info] Initial setup sent.")

    except Exception as e:
        log_view_data.append(f"[Error] Failed to connect: {e}")
        is_connected = False
        return

    while not stop_event.is_set():
        try:
            line = ser.readline().decode(errors='ignore').strip()
            if line:
                # 사용자가 로그 맨 아래를 보고 있을 때만 자동 스크롤
                is_at_bottom = (log_view_offset >= len(log_view_data) - LOG_VIEW_LINES)
                log_view_data.append(line)
                if is_at_bottom:
                    log_view_offset = len(log_view_data) - LOG_VIEW_LINES
                lc = parse_loadcell(line)
                if lc:
                    loadcell_data.append(lc)
                    
                    if data_logging:
                        # Arduino에서 보정된 값이 오므로, Python에서 추가 계산 불필요
                        for cell_index, value in lc:
                            if 0 <= cell_index < MAX_CELLS:
                                last_logged_cell_values[cell_index] = value
                        
                        # Calculate Magnus force from the most recent values of all cells
                        valid_weights = [w for w in last_logged_cell_values if np.isfinite(w)]
                        magnus_force = (sum(valid_weights) / 1000.0) * 9.8 if valid_weights else np.nan

                        current_rpm = rpm_data[-1] if rpm_data else np.nan
                        log_buffer.append((time.time(), *last_logged_cell_values, current_rpm, magnus_force))
                    continue
                motor = parse_motor(line)
                if motor:
                    rpm_data.append(motor[0])
        except Exception:
            log_view_data.append("[Error] Serial port disconnected.")
            is_connected = False
            break
    
    if ser:
        ser.close()
    is_connected = False
    log_view_data.append("[Info] Disconnected.")

def mock_reader(stop_event):
    global is_connected, log_view_offset, log_buffer, rpm_data, last_logged_cell_values
    is_connected = True
    log_view_data.append("[Info] Mock mode connected.")
    t = 0
    while not stop_event.is_set():
        # RPM 목 데이터 생성
        mock_rpm = 1500 + 500 * random.uniform(-1, 1)
        rpm_data.append(mock_rpm)

        # 로드셀 mock 데이터 생성 (1~4개 채널 랜덤, 인덱스도 랜덤)
        num_cells = random.randint(1, MAX_CELLS)
        cell_indices = sorted(random.sample(range(MAX_CELLS), num_cells))
        cells = [(i, int(10000 + 3000 * random.uniform(-1, 1))) for i in cell_indices]
        loadcell_data.append(cells)
        
        if data_logging:
            # Mock 데이터도 이미 보정되었다고 가정하고 직접 값 사용
            for cell_index, value in cells:
                if 0 <= cell_index < MAX_CELLS:
                    last_logged_cell_values[cell_index] = value

            # Calculate Magnus force from the most recent values of all cells
            valid_weights = [w for w in last_logged_cell_values if np.isfinite(w)]
            magnus_force = (sum(valid_weights) / 1000.0) * 9.8 if valid_weights else np.nan
            
            log_buffer.append((time.time(), *last_logged_cell_values, rpm_data[-1], magnus_force))
        
        # 로그 메시지를 동적으로 생성하여, 셀 개수에 상관없이 동작하도록 수정
        cell_log_parts = [f"C{i+1}: {cells[i][1]}" for i in range(len(cells))]
        log_message = f"[Mock] {', '.join(cell_log_parts)}, RPM: {rpm_data[-1]:.0f}"

        # 로그 추가 (자동 스크롤 로직 추가)
        is_at_bottom = (log_view_offset >= len(log_view_data) - LOG_VIEW_LINES)
        log_view_data.append(log_message)
        if is_at_bottom:
            log_view_offset = len(log_view_data) - LOG_VIEW_LINES
        t += 1
        time.sleep(0.1)
    is_connected = False
    log_view_data.append("[Info] Mock mode disconnected.")

def main():
    global SERIAL_PORT, stop_thread, serial_thread, data_logging, log_buffer, log_start_time, is_connected, log_view_offset, calibration_params

    load_calibration_params() # 프로그램 시작 시 캘리브레이션 값 로드

    log_view_offset = 0  # Make log_view_offset a local variable in main's scope

    # --- UI 설정 및 GridSpec 레이아웃 ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(14, 9)) # 가로 폭을 넓혀 공간 확보
    # 전체 그리드: [상단 컨트롤], [그래프+측면패널], [하단 로그]
    gs = GridSpec(3, 1, figure=fig, height_ratios=[0.7, 8, 2.5], hspace=0.35)

    # --- 상단 컨트롤 패널 (그룹화 적용) ---
    gs_top = gs[0].subgridspec(1, 2, wspace=0.1, width_ratios=[1, 1.8])

    # 그룹 1: 연결
    connect_group = gs_top[0].subgridspec(1, 2, wspace=0.1, width_ratios=[2, 1])
    ax_port = fig.add_subplot(connect_group[0])
    ax_connect = fig.add_subplot(connect_group[1])

    # 그룹 2: 실험 및 상태
    exp_group = gs_top[1].subgridspec(1, 3, wspace=0.15, width_ratios=[2, 1, 3])
    ax_exp_name = fig.add_subplot(exp_group[0])
    ax_start_stop = fig.add_subplot(exp_group[1])
    ax_status = fig.add_subplot(exp_group[2])

    # --- 메인 영역 (그래프 + 우측 현재값 패널) ---
    gs_main_area = gs[1].subgridspec(1, 2, wspace=0.15, width_ratios=[8, 1.5])

    # 그래프 영역 (2x2 그리드에서 3x2로 변경)
    gs_graphs = gs_main_area[0, 0].subgridspec(3, 2, hspace=0.4, wspace=0.2, height_ratios=[1, 1, 1.2])
    ax1 = fig.add_subplot(gs_graphs[0, 0])
    ax2 = fig.add_subplot(gs_graphs[0, 1])
    ax3 = fig.add_subplot(gs_graphs[1, 0])
    ax4 = fig.add_subplot(gs_graphs[1, 1])
    cell_axes = [ax1, ax2, ax3, ax4]

    # 매그너스 힘 & RPM을 위한 새 그래프 영역 추가
    ax_magnus_rpm = fig.add_subplot(gs_graphs[2, :]) # 맨 아래 행 전체 사용
    ax_rpm = ax_magnus_rpm.twinx() # Y축 공유
    
    # 우측 현재값 패널 (최대 셀 개수 + 매그너스 + RPM + 버튼)
    gs_side = gs_main_area[0, 1].subgridspec(MAX_CELLS + 4, 1, hspace=0.6, height_ratios=[1]*(MAX_CELLS+2) + [0.8, 0.8])
    ax_vals = [fig.add_subplot(gs_side[i]) for i in range(MAX_CELLS + 2)]
    ax_calibrate_btn = fig.add_subplot(gs_side[MAX_CELLS + 2])
    ax_tare_all_btn = fig.add_subplot(gs_side[MAX_CELLS + 3]) # Tare All 버튼용 축

    # 현재값 패널의 축/눈금 숨기기
    for ax_val in ax_vals:
        ax_val.set_facecolor('whitesmoke')
        for spine in ax_val.spines.values(): spine.set_visible(False)
        ax_val.set_xticks([]); ax_val.set_yticks([])

    # --- 하단 로그 뷰어 (위치 수정) ---
    gs_bottom = gs[2].subgridspec(1, 1)
    ax_log = fig.add_subplot(gs_bottom[0])
    
    # --- 위젯 생성 ---
    port_box = TextBox(ax_port, 'Port:', initial=load_config())
    btn_connect = Button(ax_connect, 'Connect', color='lightgoldenrodyellow')
    exp_box = TextBox(ax_exp_name, ' | ', initial='Test 1')
    btn_start_stop = Button(ax_start_stop, 'Start', color='lightgray')
    status_box = TextBox(ax_status, '', initial='Ready. Set port and connect.')
    status_box.set_active(False)
    log_viewer = TextBox(ax_log, '', initial='--- Serial Log ---') # ax_log로 재할당
    log_viewer.set_active(False)
    btn_calibrate = Button(ax_calibrate_btn, 'Calibrate', color='skyblue')
    btn_tare_all = Button(ax_tare_all_btn, 'Tare All', color='lightyellow') # Tare All 버튼 생성
    
    # --- 그래프 설정 (2x2 그리드에 맞게 최적화) ---
    fig.suptitle('Real-time Loadcell & RPM Monitoring', fontsize=18, weight='bold')
    lines = []
    for i, ax in enumerate(cell_axes):
        ax.set_title(f'Cell {i+1}', fontsize=10)
        line, = ax.plot([], [], linewidth=1.5)
        lines.append(line)
        ax.set_visible(False) # 초기에 숨기기

    # 새 그래프(매그너스/RPM) 라인 추가
    line_magnus, = ax_magnus_rpm.plot([], [], 'r-', label='Magnus Force (N)')
    line_rpm, = ax_rpm.plot([], [], 'b-', label='RPM')

    # 모든 그래프에 Y축 레이블을 표시하도록 수정
    for ax in cell_axes:
        ax.set_ylabel('Weight (g)', fontsize=9)

    # X축 레이블은 아래쪽 행(Cell 3, 4)에만 표시 -> 새 그래프로 이동
    cell_axes[2].tick_params(axis='x', labelbottom=False)
    cell_axes[3].tick_params(axis='x', labelbottom=False)
    
    # 새 그래프 축 설정
    ax_magnus_rpm.set_xlabel('Time (samples)', fontsize=9)
    ax_magnus_rpm.set_ylabel('Magnus Force (N)', color='r', fontsize=9)
    ax_rpm.set_ylabel('RPM', color='b', fontsize=9)
    ax_magnus_rpm.tick_params(axis='y', labelcolor='r')
    ax_rpm.tick_params(axis='y', labelcolor='b')
    ax_rpm.set_ylim(0, 3000) # RPM Y축 범위 고정
    ax_magnus_rpm.grid(True, linestyle='--', which='both', alpha=0.6)
    # 레전드(범례) 추가
    lines_for_legend = [line_magnus, line_rpm]
    labels_for_legend = [l.get_label() for l in lines_for_legend]
    ax_rpm.legend(lines_for_legend, labels_for_legend, loc='upper left')

    # 위쪽 행의 불필요한 X축 눈금 레이블만 숨기기
    cell_axes[0].tick_params(axis='x', labelbottom=False)
    cell_axes[1].tick_params(axis='x', labelbottom=False)

    # --- 현재값 텍스트 및 마우스 호버 기능 복구 ---
    val_texts = []
    for i in range(MAX_CELLS):
        text = ax_vals[i].text(0.5, 0.5, f'Cell {i+1}:\n---', ha='center', va='center', fontsize=12, color=lines[i].get_color())
        val_texts.append(text)
        ax_vals[i].set_visible(False) # 초기에 숨기기
    
    # Magnus 값
    magnus_text = ax_vals[MAX_CELLS].text(0.5, 0.5, 'Magnus (N):\n---', ha='center', va='center', fontsize=12, color='r')
    val_texts.append(magnus_text)
    ax_vals[MAX_CELLS].set_visible(False) # 초기에 숨기기

    # RPM 값
    rpm_text = ax_vals[MAX_CELLS + 1].text(0.5, 0.5, 'RPM:\n---', ha='center', va='center', fontsize=12, color='b')
    val_texts.append(rpm_text)

    annotations = [ax.annotate("", xy=(0,0), xytext=(20,20), textcoords="offset points",
                            bbox=dict(boxstyle="round", fc="w"), arrowprops=dict(arrowstyle="->")) for ax in cell_axes]

    def update_annot(ax, annot, line, ind):
        x, y = line.get_data()
        annot.xy = (x[ind["ind"][0]], y[ind["ind"][0]])
        annot.set_text(f"{y[ind['ind'][0]]:.2f} g")
        annot.get_bbox_patch().set_facecolor(line.get_color()); annot.get_bbox_patch().set_alpha(0.7)

    def hover(event):
        # 모든 어노테이션을 순회하며 마우스 위치 확인
        for i, ax in enumerate(cell_axes):
            if event.inaxes == ax:
                line = lines[i]
                cont, ind = line.contains(event)
                if cont:
                    update_annot(ax, annotations[i], line, ind)
                    annotations[i].set_visible(True)
                    fig.canvas.draw_idle()
                    return # 하나를 찾으면 종료
        
        # 그래프 영역 밖이거나 데이터 포인트가 아니면 모두 숨김
        for annot in annotations:
            if annot.get_visible():
                annot.set_visible(False)
                fig.canvas.draw_idle()

    # --- 버튼 상태 업데이트 함수 ---
    def update_start_stop_button():
        if is_connected:
            if data_logging:
                btn_start_stop.color = 'salmon'; btn_start_stop.label.set_text('Stop')
            else:
                btn_start_stop.color = 'lightgreen'; btn_start_stop.label.set_text('Start')
        else:
            btn_start_stop.color = 'lightgray'; btn_start_stop.label.set_text('Start')
        
        btn_start_stop.ax.set_facecolor(btn_start_stop.color)
        fig.canvas.draw_idle()

    # --- 이벤트 핸들러 ---
    def on_connect_toggle(event):
        global serial_thread, stop_thread, is_connected, SERIAL_PORT
        if not is_connected:
            port_text = port_box.text.strip()
            stop_thread.clear()

            if port_text.lower() == 'mock':
                target_func = mock_reader
                args = (stop_thread,)
                log_view_data.append("[Info] Starting mock mode...")
            else:
                SERIAL_PORT = port_text
                save_config(SERIAL_PORT)
                target_func = serial_reader
                args = (SERIAL_PORT, BAUDRATE, stop_thread)
            
            serial_thread = threading.Thread(target=target_func, args=args, daemon=True)
            serial_thread.start()
        else:
            stop_thread.set()
            if serial_thread:
                serial_thread.join(timeout=1)
        update_start_stop_button() # 연결 상태 변경 후 버튼 즉시 업데이트

    def on_start_stop_toggle(event):
        if not is_connected: return # 연결 안됐으면 무시
        global data_logging, log_buffer, log_start_time, last_logged_cell_values
        data_logging = not data_logging # 상태 토글
        if data_logging:
            print("[Start] Data logging initiated.")
            log_buffer = []
            last_logged_cell_values = [np.nan] * MAX_CELLS
            data_logging = True
            log_start_time = time.time()
            # 버튼 클릭 시 상태 메시지를 즉시 업데이트
            status_box.set_val(f'Logging... (Elapsed: 00:00)')
        else:
            print("[Stop] Data logging terminated.")
            data_logging = False
            elapsed = int(time.time() - log_start_time) if log_start_time else 0
            if log_buffer:
                columns = ['timestamp'] + [f'cell{i+1}' for i in range(MAX_CELLS)] + ['rpm', 'magnus_force']
                df = pd.DataFrame(log_buffer, columns=columns)
                t0 = df['timestamp'].iloc[0]
                df['rel_time'] = df['timestamp'] - t0
                now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'measurement_{now}_{exp_box.text.replace(" ", "_").strip()}.csv' if exp_box.text.strip() else f'measurement_{now}.csv'
                
                save_columns = ['rel_time'] + [f'cell{i+1}' for i in range(MAX_CELLS)] + ['rpm', 'magnus_force']
                df[save_columns].to_csv(filename, index=False)
                print(f"[Saved] Data saved to {filename}.")
                status_box.set_val(f'Saved ({elapsed//60:02d}:{elapsed%60:02d})')
            else:
                print("[Save Failed] No data to save.")
                status_box.set_val('Save Failed')
            log_start_time = None
        update_start_stop_button() # 버튼 즉시 업데이트

    def on_tare_all_click(event):
        if ser and is_connected:
            log_view_data.append("[Info] Taring all sensors...")
            for i in range(1, MAX_CELLS + 1):
                tare_command = f"[LoadCell_{i}] tare\n"
                ser.write(tare_command.encode('utf-8'))
                time.sleep(0.05)
            log_view_data.append("[Info] Tare commands sent.")
        else:
            status_box.set_val("Not connected. Cannot send tare commands.")

    def on_close(event):
        stop_thread.set()
        if serial_thread and serial_thread.is_alive():
            serial_thread.join(timeout=1)

    def on_scroll(event):
        global log_view_offset
        if event.inaxes != ax_log: return

        max_offset = max(0, len(log_view_data) - LOG_VIEW_LINES)

        if event.button == 'down': # 스크롤 다운 -> 이전 로그 (위로)
            log_view_offset = max(0, log_view_offset - 1)
        elif event.button == 'up': # 스크롤 업 -> 최신 로그 (아래로)
            log_view_offset = min(max_offset, log_view_offset + 1)

        start = max(0, log_view_offset)
        end = start + LOG_VIEW_LINES
        log_viewer.set_val('\n'.join(log_view_data[start:end]))
        fig.canvas.draw_idle()

    def on_calibrate_click(event):
        global stop_thread
        if not is_connected:
            status_box.set_val("Connect device before calibration.")
            return

        # 캘리브레이션 창이 열려있는 동안 메인 스레드(데이터 수신)는 계속 동작하도록 수정
        # (기존의 연결 중지/재시작 로직 제거)
        
        # 캘리브레이션 창 열기. 시리얼 객체를 전달하여 창 내에서 tare 등의 명령을 보낼 수 있게 함.
        # 참고: 이 변경으로 인해 calibration_window.py의 생성자도 수정이 필요할 수 있습니다.
        cal_window = CalibrationWindow(ser, loadcell_data, calibration_params, save_calibration_params)
        cal_window.show() # blocking

        # 캘리브레이션 창이 닫힌 후, 특별히 할 작업은 없음.
        # save_calibration_params 콜백이 새 값을 저장하고 Arduino에 전송하는 역할까지 담당함.
        log_view_data.append("[Info] Calibration window closed.")

    btn_connect.on_clicked(on_connect_toggle)
    btn_start_stop.on_clicked(on_start_stop_toggle)
    btn_tare_all.on_clicked(on_tare_all_click) # Tare All 버튼 이벤트 연결
    btn_calibrate.on_clicked(on_calibrate_click)
    fig.canvas.mpl_connect('close_event', on_close)
    fig.canvas.mpl_connect('scroll_event', on_scroll)

    # UI 업데이트 함수들
    def update_status(msg):
        status_box.set_val(msg)
        plt.draw()

    # 애니메이션
    def animate(frame):
        global is_connected, log_view_data, calibration_params, log_view_offset
        # 상태 버튼 업데이트
        if is_connected:
            btn_connect.label.set_text('Disconnect')
            btn_connect.color = 'lightcoral'
        else:
            btn_connect.label.set_text('Connect')
            btn_connect.color = 'lightgoldenrodyellow'
        btn_connect.ax.set_facecolor(btn_connect.color)

        # 로그 뷰어 업데이트
        start = max(0, log_view_offset)
        end = start + LOG_VIEW_LINES
        log_viewer.set_val('\n'.join(log_view_data[start:end]))
        
        # 프레임 시작 시 모든 셀 그래프/값을 숨김
        for i in range(MAX_CELLS):
            cell_axes[i].set_visible(False)
            ax_vals[i].set_visible(False)
            val_texts[i].set_text(f'Cell {i+1}:\n---')
        
        ax_vals[MAX_CELLS].set_visible(False)
        val_texts[MAX_CELLS].set_text('Magnus (N):\n---')

        if loadcell_data:
            # 각 셀 별로 그래프 데이터를 구축하고 그림
            for cell_idx in range(MAX_CELLS):
                history_x = []
                history_y = []
                # 전체 데이터 버퍼를 순회하며 해당 셀의 데이터만 추출
                for time_step, packet in enumerate(loadcell_data):
                    for p_idx, p_val in packet:
                        if p_idx == cell_idx:
                            history_x.append(time_step)
                            # Arduino에서 보정된 값이므로 추가 계산 없이 바로 사용
                            history_y.append(p_val)
                
                # 데이터가 존재하면 그래프를 그리고 값을 업데이트
                if history_y:
                    lines[cell_idx].set_data(history_x, history_y)
                    cell_axes[cell_idx].set_xlim(0, len(loadcell_data))
                    
                    # Y축 범위 자동 조절 로직 수정 (NaN, Inf 값 방지)
                    finite_history_y = [v for v in history_y if np.isfinite(v)]
                    
                    if finite_history_y:
                        data_min, data_max = min(finite_history_y), max(finite_history_y)
                        margin = (data_max - data_min) * 0.1
                        if margin < 10: # 데이터 변화가 거의 없을 경우 최소 마진 보장
                            margin = 10
                        cell_axes[cell_idx].set_ylim(data_min - margin, data_max + margin)

                        val_texts[cell_idx].set_text(f'Cell {cell_idx+1}:\n{history_y[-1]:.3f} g')
                        
                        # 해당 셀의 그래프와 값 표시를 활성화
                        cell_axes[cell_idx].set_visible(True)
                        ax_vals[cell_idx].set_visible(True)
            
            # --- 매그너스 힘 및 RPM 그래프 업데이트 ---
            magnus_x = list(range(len(loadcell_data)))
            
            # 1. 모든 셀의 보정된 무게를 (타임스텝, 셀) 매트릭스로 구성
            # Arduino에서 보정된 값이므로 Python에서 추가 계산 불필요
            calibrated_weights = [[np.nan] * MAX_CELLS for _ in range(len(loadcell_data))]
            for time_step, packet in enumerate(loadcell_data):
                for p_idx, p_val in packet:
                    if 0 <= p_idx < MAX_CELLS:
                        calibrated_weights[time_step][p_idx] = p_val
            
            # 2. 누락된 데이터를 이전 값으로 채우기 (Forward Fill)
            for cell_idx in range(MAX_CELLS):
                last_val = np.nan
                for time_step in range(len(loadcell_data)):
                    if not np.isnan(calibrated_weights[time_step][cell_idx]):
                        last_val = calibrated_weights[time_step][cell_idx]
                    else:
                        calibrated_weights[time_step][cell_idx] = last_val
            
            # 3. 매그너스 힘 계산
            magnus_force_y = []
            for time_step in range(len(loadcell_data)):
                valid_weights = [w for w in calibrated_weights[time_step] if np.isfinite(w)]
                if not valid_weights:
                    magnus_force_y.append(np.nan)
                    continue
                total_g = sum(valid_weights)
                magnus_force = (total_g / 1000.0) * 9.8
                magnus_force_y.append(magnus_force)
            
            # 4. 매그너스 힘 그래프 업데이트
            line_magnus.set_data(magnus_x, magnus_force_y)
            finite_magnus_y = [v for v in magnus_force_y if np.isfinite(v)]
            if finite_magnus_y:
                magnus_min, magnus_max = min(finite_magnus_y), max(finite_magnus_y)
                margin = (magnus_max - magnus_min) * 0.1 if magnus_max > magnus_min else 1
                ax_magnus_rpm.set_ylim(magnus_min - margin, magnus_max + margin)
                val_texts[MAX_CELLS].set_text(f'Magnus (N):\n{finite_magnus_y[-1]:.2f}')
                ax_vals[MAX_CELLS].set_visible(True)

        # RPM 값 업데이트
        if rpm_data:
            val_texts[MAX_CELLS + 1].set_text(f'RPM:\n{rpm_data[-1]:.0f}')
            line_rpm.set_data(range(len(rpm_data)), list(rpm_data))
        else:
            val_texts[MAX_CELLS + 1].set_text('RPM:\n---')
            line_rpm.set_data([], [])

        # X축 범위 동기화
        if loadcell_data or rpm_data:
            max_len = max(len(loadcell_data), len(rpm_data))
            ax_magnus_rpm.set_xlim(0, max_len)
            ax_rpm.set_xlim(0, max_len)

        # 측정 중이면 경과 시간 실시간 표시
        if data_logging and log_start_time:
            elapsed = int(time.time() - log_start_time)
            update_status(f'Logging... (Elapsed: {elapsed//60:02d}:{elapsed%60:02d})')

        return lines + [line_magnus, line_rpm]
    
    fig.canvas.mpl_connect("motion_notify_event", hover)
    ani = FuncAnimation(fig, animate, interval=100, blit=False)
    plt.show()

if __name__ == '__main__':
    main() 