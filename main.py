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

# 시리얼 포트 설정 (포트명은 환경에 맞게 수정)
SERIAL_PORT = '/dev/tty.usbserial-1130'  # macOS 예시, Windows는 'COM3' 등
BAUDRATE = 9600

CELL_1_CAL = 10000
CELL_2_CAL = 12000
# CELL_3_CAL = 12000
# CELL_4_CAL = 12000

# mock 데이터 사용 여부
USE_MOCK = True  # True면 mock 데이터, False면 실제 시리얼 사용

# 데이터 버퍼 (최근 100개 데이터만 저장)
MAXLEN = 100
loadcell_data = deque(maxlen=MAXLEN)  # 각 원소: [cell1, cell2]
rpm_data = deque(maxlen=MAXLEN)        # 각 원소: rpm

# 로깅 관련 변수
data_logging = False
log_buffer = []  # 각 원소: (timestamp, cell1, cell2, rpm)
log_start_time = None  # 측정 시작 시각

# 전역 마이너스 부호 설정 (유지하는 것이 좋음)
matplotlib.rcParams['axes.unicode_minus'] = False

def serial_reader():
    global data_logging, log_buffer
    try:
        ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
        print(f"시리얼 포트 연결: {SERIAL_PORT}")
    except Exception as e:
        print(f"시리얼 포트 연결 실패: {e}")
        return
    while True:
        try:
            line = ser.readline().decode(errors='ignore').strip()
            if not line:
                continue
            lc = parse_loadcell(line)
            if lc:
                loadcell_data.append(lc)
                if data_logging:
                    # rpm_data가 비어있을 수 있으니 마지막 값 사용
                    rpm = rpm_data[-1] if rpm_data else 0
                    log_buffer.append((time.time(), *lc, rpm))
                continue
            motor = parse_motor(line)
            if motor:
                rpm, pwm = motor
                rpm_data.append(rpm)
        except Exception as e:
            print(f"시리얼 읽기 오류: {e}")
            continue

def mock_reader():
    global data_logging, log_buffer
    t = 0
    while True:
        # 로드셀 mock 데이터 생성 (2개 채널)
        cell1 = int(10000 + 2000 * random.uniform(-1, 1))
        cell2 = int(12000 + 2000 * random.uniform(-1, 1))
        loadcell_data.append([cell1, cell2])
        # 모터 mock 데이터 생성
        rpm = int(1000 + 200 * random.uniform(-1, 1))
        rpm_data.append(rpm)
        if data_logging:
            log_buffer.append((time.time(), cell1, cell2, rpm))
        t += 1
        time.sleep(0.1)

def main():
    global data_logging, log_buffer, log_start_time
    # mock/실제 시리얼 선택
    if USE_MOCK:
        t = threading.Thread(target=mock_reader, daemon=True)
        print("[Mock Mode] Testing with mock data.")
    else:
        t = threading.Thread(target=serial_reader, daemon=True)
    t.start()

    # --- UI 설정 ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle('Real-time Loadcell & RPM Monitoring', fontsize=18, weight='bold')

    # 그래프 1: 로드셀
    ax1.set_title('Loadcell Values', fontsize=12)
    lines = [ax1.plot([], [], label=f'Cell{i+1}', linewidth=1.5)[0] for i in range(2)]
    ax1.set_ylabel('LoadCell Value')
    ax1.legend(loc='upper left')

    # 그래프 2: RPM
    ax2.set_title('Motor RPM', fontsize=12)
    rpm_line, = ax2.plot([], [], color='crimson', label='RPM', linewidth=1.5)
    ax2.set_ylabel('RPM')
    ax2.set_xlabel('Time (samples)')
    ax2.legend(loc='upper left')

    # --- 레이아웃 및 컨트롤 설정 ---
    # 컨트롤과 제목, 현재값 표시를 위한 공간을 남기고 레이아웃 최적화
    fig.tight_layout(rect=[0, 0.1, 0.9, 0.95])
    
    # 컨트롤 (실험명 입력, 버튼, 상태창)
    ax_exp_name = plt.axes([0.1, 0.02, 0.18, 0.06])
    experiment_box = TextBox(ax_exp_name, 'Experiment:', initial='Test 1')

    ax_start_stop = plt.axes([0.3, 0.02, 0.18, 0.06])
    btn_start_stop = Button(ax_start_stop, 'Start', color='lightgreen', hovercolor='palegreen')

    ax_status = plt.axes([0.5, 0.02, 0.45, 0.06])
    status_box = TextBox(ax_status, '', initial='Ready')
    status_box.ax.set_facecolor('whitesmoke')
    status_box.set_active(False)

    # 현재값 표시용 텍스트
    val_texts = []
    for i in range(2):
        val_texts.append(fig.text(0.91, 0.8 - i*0.05, '', fontsize=12, color=lines[i].get_color()))
    val_texts.append(fig.text(0.91, 0.4, '', fontsize=12, color=rpm_line.get_color())) # RPM 값 표시

    # 마우스 호버용 어노테이션
    annot1 = ax1.annotate("", xy=(0,0), xytext=(20,20), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w", alpha=0.7),
                        arrowprops=dict(arrowstyle="->"))
    annot2 = ax2.annotate("", xy=(0,0), xytext=(20,20), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w", alpha=0.7),
                        arrowprops=dict(arrowstyle="->"))
    annot1.set_visible(False)
    annot2.set_visible(False)

    def update_annot(ax, annot, line, ind):
        x, y = line.get_data()
        annot.xy = (x[ind["ind"][0]], y[ind["ind"][0]])
        annot.set_text(f"{y[ind['ind'][0]]:.0f}")
        annot.get_bbox_patch().set_facecolor(line.get_color())

    def hover(event):
        for ax, annot, line_list in [(ax1, annot1, lines), (ax2, annot2, [rpm_line])]:
            if event.inaxes == ax:
                for line in line_list:
                    cont, ind = line.contains(event)
                    if cont:
                        update_annot(ax, annot, line, ind)
                        annot.set_visible(True)
                        fig.canvas.draw_idle()
                        return
        
        # 마우스가 그래프 밖으로 나가면 모든 어노테이션 숨기기
        for annot in [annot1, annot2]:
            if annot.get_visible():
                annot.set_visible(False)
                fig.canvas.draw_idle()

    # UI 업데이트 함수들
    def update_button_state():
        if data_logging:
            btn_start_stop.label.set_text('Stop')
            btn_start_stop.color = 'salmon'
            btn_start_stop.hovercolor = 'darksalmon'
        else:
            btn_start_stop.label.set_text('Start')
            btn_start_stop.color = 'lightgreen'
            btn_start_stop.hovercolor = 'palegreen'
        btn_start_stop.ax.set_facecolor(btn_start_stop.color)
        plt.draw()

    def update_status(msg):
        status_box.set_val(msg)
        plt.draw()

    # 이벤트 핸들러
    def on_toggle(event):
        global data_logging, log_buffer, log_start_time
        if not data_logging:
            print("[Start] Data logging initiated.")
            log_buffer = []
            data_logging = True
            log_start_time = time.time()
            # 버튼 클릭 시 상태 메시지를 즉시 업데이트
            update_status(f'Logging... (Elapsed: 00:00)')
        else:
            print("[Stop] Data logging terminated.")
            data_logging = False
            elapsed = int(time.time() - log_start_time) if log_start_time else 0
            if log_buffer:
                # 실험명을 파일명에 추가
                exp_name = experiment_box.text.replace(' ', '_').strip()
                df = pd.DataFrame(log_buffer, columns=['timestamp', 'cell1', 'cell2', 'rpm'])
                t0 = df['timestamp'].iloc[0]
                df['rel_time'] = df['timestamp'] - t0
                now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'measurement_{now}_{exp_name}.csv' if exp_name else f'measurement_{now}.csv'
                df[['rel_time', 'cell1', 'cell2', 'rpm']].to_csv(filename, index=False)
                print(f"[Saved] Data saved to {filename}.")
                update_status(f'Saved ({elapsed//60:02d}:{elapsed%60:02d})')
            else:
                print("[Save Failed] No data to save.")
                update_status('Save Failed')
            log_start_time = None
        update_button_state()

    btn_start_stop.on_clicked(on_toggle)
    update_button_state()

    # 애니메이션
    def animate(frame):
        # 로드셀 (2개 채널에 맞게 수정)
        if loadcell_data:
            data = np.array(list(loadcell_data))
            for i in range(2):
                lines[i].set_data(range(len(data)), data[:, i])
            ax1.set_xlim(0, max(len(data), 10))
            if data.size > 0:
                ax1.set_ylim(data.min() - 100, data.max() + 100)
                # 현재값 텍스트 업데이트
                latest_vals = loadcell_data[-1]
                for i in range(2):
                    val_texts[i].set_text(f'Cell {i+1}: {latest_vals[i]}')
        
        # RPM
        if rpm_data:
            data = list(rpm_data)
            rpm_line.set_data(range(len(data)), data)
            ax2.set_xlim(0, max(len(data), 10))
            ax2.set_ylim(min(data)-50, max(data)+50)
            # RPM 현재값 텍스트 업데이트
            val_texts[2].set_text(f'RPM: {rpm_data[-1]}')
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