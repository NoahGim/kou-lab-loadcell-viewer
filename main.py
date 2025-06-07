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

# 시리얼 포트 설정 (포트명은 환경에 맞게 수정)
SERIAL_PORT = '/dev/tty.usbserial-1130'  # macOS 예시, Windows는 'COM3' 등
BAUDRATE = 9600

CELL_1_CAL = 10000
CELL_2_CAL = 12000
# CELL_3_CAL = 12000
# CELL_4_CAL = 12000

# mock 데이터 사용 여부
# USE_MOCK = True  # 이 줄을 삭제하고, 아래 로직으로 대체합니다.

# 데이터 버퍼
MAXLEN = 100
loadcell_data = deque(maxlen=MAXLEN)
rpm_data = deque(maxlen=MAXLEN)
log_view_data = [] # Deque가 아닌 일반 리스트로 변경하여 모든 로그 저장
log_view_offset = 0 # 로그 뷰어 스크롤 위치
LOG_VIEW_LINES = 10 # 로그 뷰어에 표시할 줄 수

# 로깅 관련 변수
data_logging = False
log_buffer = []  # 각 원소: (timestamp, cell1, cell2, rpm)
log_start_time = None  # 측정 시작 시각

# 전역 마이너스 부호 설정 (유지하는 것이 좋음)
matplotlib.rcParams['axes.unicode_minus'] = False

# 상태 변수
is_connected = False
serial_thread = None
stop_thread = threading.Event()
ser = None # 시리얼 객체

# --- 설정 관리 ---
def save_config(port):
    with open('config.json', 'w') as f:
        json.dump({'serial_port': port}, f)

def load_config():
    if os.path.exists('config.json'):
        with open('config.json', 'r') as f:
            return json.load(f).get('serial_port', '')
    return ''

# --- 시리얼 통신 스레드 ---
def serial_reader(port, baud, stop_event):
    global ser, is_connected, log_view_offset
    try:
        ser = serial.Serial(port, baud, timeout=0.1)
        is_connected = True
        log_view_data.append(f"[Info] Connected to {port}")
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
                    if data_logging and rpm_data:
                        log_buffer.append((time.time(), *lc, rpm_data[-1]))
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
    global is_connected, log_view_offset
    is_connected = True
    log_view_data.append("[Info] Mock mode connected.")
    t = 0
    while not stop_event.is_set():
        # 로드셀 mock 데이터 생성 (2개 채널)
        cell1 = int(10000 + 2000 * random.uniform(-1, 1))
        cell2 = int(12000 + 2000 * random.uniform(-1, 1))
        loadcell_data.append([cell1, cell2])
        # 모터 mock 데이터 생성
        rpm = int(1000 + 200 * random.uniform(-1, 1))
        rpm_data.append(rpm)
        if data_logging:
            log_buffer.append((time.time(), cell1, cell2, rpm))
        # 로그 추가 (자동 스크롤 로직 추가)
        is_at_bottom = (log_view_offset >= len(log_view_data) - LOG_VIEW_LINES)
        log_view_data.append(f"[Mock] C1: {cell1}, C2: {cell2}, RPM: {rpm}")
        if is_at_bottom:
            log_view_offset = len(log_view_data) - LOG_VIEW_LINES
        t += 1
        time.sleep(0.1)
    is_connected = False
    log_view_data.append("[Info] Mock mode disconnected.")

def main():
    global SERIAL_PORT, stop_thread, serial_thread, data_logging, log_buffer, log_start_time, is_connected, log_view_offset

    log_view_offset = 0  # Make log_view_offset a local variable in main's scope

    # --- UI 설정 및 GridSpec 레이아웃 ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(14, 9)) # 가로 폭을 넓혀 공간 확보
    # 전체 그리드: [상단 컨트롤], [그래프+측면패널], [하단 로그]
    gs = GridSpec(3, 1, figure=fig, height_ratios=[1.5, 8, 2], hspace=0.45)

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

    # 그래프 영역
    gs_graphs = gs_main_area[0, 0].subgridspec(2, 1, hspace=0.3)
    ax1 = fig.add_subplot(gs_graphs[0])
    ax2 = fig.add_subplot(gs_graphs[1])

    # 우측 현재값 패널
    gs_side = gs_main_area[0, 1].subgridspec(3, 1, hspace=0.5)
    ax_val1 = fig.add_subplot(gs_side[0])
    ax_val2 = fig.add_subplot(gs_side[1])
    ax_val3 = fig.add_subplot(gs_side[2])

    # 현재값 패널의 축/눈금 숨기기
    for ax_val in [ax_val1, ax_val2, ax_val3]:
        ax_val.set_facecolor('whitesmoke')
        for spine in ax_val.spines.values(): spine.set_visible(False)
        ax_val.set_xticks([]); ax_val.set_yticks([])

    # --- 하단 로그 뷰어 ---
    gs_bottom = gs[2].subgridspec(1, 1)
    ax_log = fig.add_subplot(gs_bottom[0])
    
    # --- 위젯 생성 ---
    port_box = TextBox(ax_port, 'Port:', initial=load_config())
    btn_connect = Button(ax_connect, 'Connect', color='lightgoldenrodyellow')
    exp_box = TextBox(ax_exp_name, ' | ', initial='Test 1')
    btn_start_stop = Button(ax_start_stop, 'Start', color='lightgray')
    status_box = TextBox(ax_status, '', initial='Ready. Set port and connect.')
    status_box.set_active(False)
    log_viewer = TextBox(ax_log, '', initial='--- Serial Log ---')
    log_viewer.set_active(False)
    
    # --- 그래프 설정 ---
    fig.suptitle('Real-time Loadcell & RPM Monitoring', fontsize=18, weight='bold')
    ax1.set_title('Loadcell Values'); ax2.set_title('Motor RPM')
    lines = [ax1.plot([], [], label=f'Cell{i+1}', linewidth=1.5)[0] for i in range(2)]
    ax1.set_ylabel('Value'); ax1.legend(loc='upper left')
    rpm_line, = ax2.plot([], [], color='crimson', label='RPM', linewidth=1.5)
    ax2.set_ylabel('RPM'); ax2.set_xlabel('Time (samples)'); ax2.legend(loc='upper left')

    # --- 현재값 텍스트 및 마우스 호버 기능 복구 ---
    val1_text = ax_val1.text(0.5, 0.5, 'Cell 1:\n---', ha='center', va='center', fontsize=14, color=lines[0].get_color())
    val2_text = ax_val2.text(0.5, 0.5, 'Cell 2:\n---', ha='center', va='center', fontsize=14, color=lines[1].get_color())
    val3_text = ax_val3.text(0.5, 0.5, 'RPM:\n---', ha='center', va='center', fontsize=14, color=rpm_line.get_color())

    annotations = [ax.annotate("", xy=(0,0), xytext=(20,20), textcoords="offset points",
                            bbox=dict(boxstyle="round", fc="w"), arrowprops=dict(arrowstyle="->")) for ax in [ax1, ax2]]

    def update_annot(ax, annot, line, ind):
        x, y = line.get_data()
        annot.xy = (x[ind["ind"][0]], y[ind["ind"][0]])
        annot.set_text(f"{y[ind['ind'][0]]:.0f}")
        annot.get_bbox_patch().set_facecolor(line.get_color()); annot.get_bbox_patch().set_alpha(0.7)

    def hover(event):
        for i, (ax, line_list) in enumerate([(ax1, lines), (ax2, [rpm_line])]):
            if event.inaxes == ax:
                for line in line_list:
                    cont, ind = line.contains(event)
                    if cont:
                        update_annot(ax, annotations[i], line, ind)
                        annotations[i].set_visible(True); fig.canvas.draw_idle()
                        return
        for annot in annotations:
            if annot.get_visible():
                annot.set_visible(False); fig.canvas.draw_idle()

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
        global data_logging, log_buffer, log_start_time
        data_logging = not data_logging # 상태 토글
        if data_logging:
            print("[Start] Data logging initiated.")
            log_buffer = []
            data_logging = True
            log_start_time = time.time()
            # 버튼 클릭 시 상태 메시지를 즉시 업데이트
            status_box.set_val(f'Logging... (Elapsed: 00:00)')
        else:
            print("[Stop] Data logging terminated.")
            data_logging = False
            elapsed = int(time.time() - log_start_time) if log_start_time else 0
            if log_buffer:
                # 실험명을 파일명에 추가
                exp_name = exp_box.text.replace(' ', '_').strip()
                df = pd.DataFrame(log_buffer, columns=['timestamp', 'cell1', 'cell2', 'rpm'])
                t0 = df['timestamp'].iloc[0]
                df['rel_time'] = df['timestamp'] - t0
                now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'measurement_{now}_{exp_name}.csv' if exp_name else f'measurement_{now}.csv'
                df[['rel_time', 'cell1', 'cell2', 'rpm']].to_csv(filename, index=False)
                print(f"[Saved] Data saved to {filename}.")
                status_box.set_val(f'Saved ({elapsed//60:02d}:{elapsed%60:02d})')
            else:
                print("[Save Failed] No data to save.")
                status_box.set_val('Save Failed')
            log_start_time = None
        update_start_stop_button() # 버튼 즉시 업데이트

    def on_close(event):
        stop_thread.set()
        if serial_thread and serial_thread.is_alive():
            serial_thread.join(timeout=1)

    def on_scroll(event, offset_container=[log_view_offset]):
        if event.inaxes != ax_log: return
        
        max_offset = max(0, len(log_view_data) - LOG_VIEW_LINES)
        
        if event.button == 'up':
            offset_container[0] = max(0, offset_container[0] - 1)
        elif event.button == 'down':
            offset_container[0] = min(max_offset, offset_container[0] + 1)
        
        start = max(0, offset_container[0])
        end = start + LOG_VIEW_LINES
        log_viewer.set_val('\n'.join(log_view_data[start:end]))
        fig.canvas.draw_idle()

    btn_connect.on_clicked(on_connect_toggle)
    btn_start_stop.on_clicked(on_start_stop_toggle)
    fig.canvas.mpl_connect('close_event', on_close)
    fig.canvas.mpl_connect('scroll_event', on_scroll)

    # UI 업데이트 함수들
    def update_status(msg):
        status_box.set_val(msg)
        plt.draw()

    # 애니메이션
    def animate(frame, offset_container=[log_view_offset]):
        # 상태 버튼 업데이트
        if is_connected:
            btn_connect.label.set_text('Disconnect')
            btn_connect.color = 'lightcoral'
        else:
            btn_connect.label.set_text('Connect')
            btn_connect.color = 'lightgoldenrodyellow'
        btn_connect.ax.set_facecolor(btn_connect.color)

        # 로그 뷰어 업데이트
        start = max(0, offset_container[0])
        end = start + LOG_VIEW_LINES
        log_viewer.set_val('\n'.join(log_view_data[start:end]))
        
        # 그래프 및 현재값 업데이트
        if loadcell_data:
            data = np.array(list(loadcell_data)); latest_vals = data[-1]
            for i in range(2): lines[i].set_data(range(len(data)), data[:, i])
            ax1.set_xlim(0, len(data)); ax1.set_ylim(data.min()*0.9, data.max()*1.1)
            val1_text.set_text(f'Cell 1:\n{latest_vals[0]:.0f}'); val2_text.set_text(f'Cell 2:\n{latest_vals[1]:.0f}')
        else:
            val1_text.set_text('Cell 1:\n---'); val2_text.set_text('Cell 2:\n---')

        if rpm_data:
            data = list(rpm_data)
            rpm_line.set_data(range(len(data)), data)
            ax2.set_xlim(0, len(data)); ax2.set_ylim(min(data)*0.9, max(data)*1.1)
            val3_text.set_text(f'RPM:\n{data[-1]:.0f}')
        else:
            val3_text.set_text('RPM:\n---')

        # 측정 중이면 경과 시간 실시간 표시
        if data_logging and log_start_time:
            elapsed = int(time.time() - log_start_time)
            update_status(f'Logging... (Elapsed: {elapsed//60:02d}:{elapsed%60:02d})')
        return lines + [rpm_line]
    
    fig.canvas.mpl_connect("motion_notify_event", hover)
    ani = FuncAnimation(fig, animate, interval=100, blit=False)
    plt.show()

if __name__ == '__main__':
    main() 