import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox, RadioButtons
import numpy as np
from scipy.stats import linregress
import time
import platform
import matplotlib.font_manager as fm
from matplotlib.animation import FuncAnimation

# 한글 폰트 설정
def set_korean_font():
    system_name = platform.system()
    if system_name == 'Darwin': # macOS
        font_name = 'AppleGothic'
    elif system_name == 'Windows': # Windows
        font_name = 'Malgun Gothic'
    else: # Linux
        # 기본적으로 나눔고딕을 시도, 없으면 시스템 기본 폰트 사용
        if 'NanumGothic' in [f.name for f in fm.fontManager.ttflist]:
            font_name = 'NanumGothic'
        else:
            font_name = None
    
    if font_name:
        plt.rc('font', family=font_name)
    # 마이너스 부호 깨짐 방지
    plt.rcParams['axes.unicode_minus'] = False

set_korean_font()

class CalibrationWindow:
    def __init__(self, ser, loadcell_data_deque, initial_params, save_callback):
        self.ser = ser
        self.loadcell_data = loadcell_data_deque
        self.cal_params = initial_params.copy()
        self.save_callback = save_callback

        self.fig, self.ax = plt.subplots(figsize=(8, 7))
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        
        # OS 호환성을 고려한 창 제목 설정
        try:
            self.fig.canvas.manager.set_window_title('Calibration Helper')
        except AttributeError:
            pass # 일부 백엔드에서는 지원하지 않으므로, 오류 없이 넘어감

        self.selected_cell_idx = 0
        self.measured_points = {i: [] for i in range(4)}
        self.is_reading_live = False
        self.current_gram_value = 0
        self.last_r_squared = {i: None for i in range(4)}

        self.setup_widgets()
        self.update_display()
        
        # 실시간 raw 값 업데이트를 위한 애니메이션 (plt. 접두사 제거)
        self.animation = FuncAnimation(self.fig, self.update_live_values, interval=200)

    def setup_widgets(self):
        self.ax.text(0.05, 0.95, "1. Select Cell, Tare, and Add Points", transform=self.ax.transAxes, fontsize=12, weight='bold')
        
        # 셀 선택 라디오 버튼 및 Tare 버튼
        ax_radio = plt.axes([0.05, 0.8, 0.2, 0.15])
        self.radio_cell = RadioButtons(ax_radio, ('Cell 1', 'Cell 2', 'Cell 3', 'Cell 4'))
        self.radio_cell.on_clicked(self.on_cell_select)
        
        ax_tare = plt.axes([0.25, 0.8, 0.1, 0.05])
        self.btn_tare = Button(ax_tare, 'Initialize Cell')
        self.btn_tare.on_clicked(self.on_initialize_cell)

        # 무게 입력
        ax_weight = plt.axes([0.4, 0.85, 0.15, 0.05])
        self.text_known_weight = TextBox(ax_weight, "Known W (g):", initial="100.0")

        # 측정 시작/추가 버튼
        ax_read = plt.axes([0.6, 0.85, 0.2, 0.05])
        self.btn_read = Button(ax_read, 'Start Reading Live')
        self.btn_read.on_clicked(self.on_read_toggle)

        # 현재 raw 값 표시
        self.ax.text(0.8, 0.875, "Live Value (g):", transform=self.ax.transAxes)
        self.live_raw_text = self.ax.text(0.9, 0.875, "---", transform=self.ax.transAxes, color='red', weight='bold')
        
        # 포인트 테이블
        self.ax.text(0.05, 0.7, "Measured Points:", transform=self.ax.transAxes)
        self.table_text = self.ax.text(0.05, 0.68, "No points added yet.", transform=self.ax.transAxes, va='top', family='monospace')
        
        ax_clear = plt.axes([0.75, 0.7, 0.2, 0.05])
        self.btn_clear = Button(ax_clear, 'Clear Points')
        self.btn_clear.on_clicked(self.on_clear_points)

        self.ax.axhline(0.55, color='gray', linestyle='--')
        self.ax.text(0.05, 0.5, "2. Calculate & Save", transform=self.ax.transAxes, fontsize=12, weight='bold')

        # 계산 버튼
        ax_calc = plt.axes([0.05, 0.4, 0.4, 0.07])
        self.btn_calc = Button(ax_calc, 'Calculate Scale Factor from Points')
        self.btn_calc.on_clicked(self.on_calculate)
        
        # 결과 표시
        self.ax.text(0.05, 0.3, "Result:", transform=self.ax.transAxes)
        self.result_text = self.ax.text(0.05, 0.28, "Scale: ---\nR-squared: ---", transform=self.ax.transAxes, va='top', family='monospace')

        # 테스트
        self.ax.text(0.5, 0.3, "Live Test (g):", transform=self.ax.transAxes)
        self.live_gram_text = self.ax.text(0.75, 0.3, "---", transform=self.ax.transAxes, color='blue', weight='bold', fontsize=14)

        # 저장/완료 버튼
        ax_save = plt.axes([0.2, 0.1, 0.25, 0.07])
        self.btn_save = Button(ax_save, 'Save to File')
        self.btn_save.on_clicked(self.on_save)
        
        ax_close = plt.axes([0.55, 0.1, 0.25, 0.07])
        self.btn_close = Button(ax_close, 'Finish')
        self.btn_close.on_clicked(self.on_close)

        self.ax.axis('off')

    def update_display(self):
        # 포인트 테이블 업데이트
        points = self.measured_points[self.selected_cell_idx]
        if not points:
            self.table_text.set_text("No points added yet.")
        else:
            table_str = "Known (g) | Est. Raw\n" + "-"*25 + "\n"
            table_str += "\n".join([f"{p[0]:<9.2f} | {p[1]:.2f}" for p in points])
            self.table_text.set_text(table_str)

        # 버튼 상태 업데이트
        self.btn_calc.set_active(len(points) >= 2)
        
        # 결과 텍스트 업데이트
        key = f"cell_{self.selected_cell_idx+1}"
        s = self.cal_params.get(key, '---')
        r2 = self.last_r_squared[self.selected_cell_idx]
        
        scale_str = f"Scale: {s:.4f}" if isinstance(s, float) else f"Scale: {s}"
        r2_str = f"R-squared: {r2:.6f}" if isinstance(r2, float) else "R-squared: ---"
        self.result_text.set_text(f"{scale_str}\n{r2_str}")

        self.fig.canvas.draw_idle()

    def on_cell_select(self, label):
        self.selected_cell_idx = int(label.replace('Cell ', '')) - 1
        self.is_reading_live = False
        self.btn_read.label.set_text('Start Reading Live')
        self.btn_read.color = 'lightgoldenrodyellow'
        self.update_display()

    def on_initialize_cell(self, event):
        if self.ser and self.ser.is_open:
            cell_num = self.selected_cell_idx + 1
            
            # initialize calibration scale
            command_cal = f"[LoadCell_{cell_num}] init\n"
            self.ser.write(command_cal.encode('utf-8'))
            print(f"Sent initialize command to Cell {cell_num}")
        else:
            print("Serial port not connected. Cannot send tare command.")

    def on_read_toggle(self, event):
        if not self.is_reading_live:
            self.is_reading_live = True
            self.btn_read.label.set_text('Add This Point')
            self.btn_read.color = 'lightgreen'
        else: # Add point
            known_weight = float(self.text_known_weight.text)
            
            # [Fix] Arduino에서 이미 보정된 gram값이 오므로, 현재 scale을 곱해서 원래의 raw 값으로 역산.
            # 이 역산된 raw 값으로 새로운 scale을 계산해야 함.
            key = f"cell_{self.selected_cell_idx+1}"
            current_scale = self.cal_params.get(key, 1.0)
            # estimated_raw_value = self.current_gram_value * current_scale
            estimated_raw_value = self.current_gram_value
            self.measured_points[self.selected_cell_idx].append((known_weight, estimated_raw_value))
            self.is_reading_live = False
            self.btn_read.label.set_text('Start Reading Live')
            self.btn_read.color = 'lightgoldenrodyellow'
            self.update_display()
            
    def on_clear_points(self, event):
        self.measured_points[self.selected_cell_idx] = []
        self.last_r_squared[self.selected_cell_idx] = None
        self.update_display()

    def on_calculate(self, event):
        points = self.measured_points[self.selected_cell_idx]
        if len(points) < 2: return

        known_weights = np.array([p[0] for p in points])
        raw_values = np.array([p[1] for p in points])

        # raw_values = scale * known_weights (offset은 tare로 처리됨)
        # y = a*x 형태의 회귀. y/x의 평균으로 scale 추정
        # 단, 0g 지점이 포함될 수 있으므로 일반적인 linregress 사용이 더 안정적
        # y = a*x + b 에서 b는 0에 가까워야 함.
        res = linregress(known_weights, raw_values)
        
        scale, r_value = res.slope, res.rvalue
        
        key = f"cell_{self.selected_cell_idx+1}"
        self.cal_params[key] = scale
        self.last_r_squared[self.selected_cell_idx] = r_value**2

        self.update_display()

    def update_live_values(self, frame):
        # 가장 최근 데이터 패킷에서 선택된 셀의 raw 값 찾기
        gram_val = None
        if self.loadcell_data:
            latest_packet = self.loadcell_data[-1]
            for idx, val in latest_packet:
                if idx == self.selected_cell_idx:
                    gram_val = val
                    break
        
        if gram_val is not None:
            self.current_gram_value = gram_val
            self.live_raw_text.set_text(f"{gram_val:.2f} g")
            
            # 라이브 테스트 (offset 제거)
            key = f"cell_{self.selected_cell_idx+1}"
            s = self.cal_params[key]
            if s != 0:
                gram_val_test = self.current_gram_value
                self.live_gram_text.set_text(f"{gram_val_test:.2f} g")
        else:
            self.live_raw_text.set_text("---")
            self.live_gram_text.set_text("---")

    def on_save(self, event):
        self.save_callback(self.cal_params)
        self.btn_save.color = 'palegreen'
        self.fig.canvas.draw_idle()

    def on_close(self, event):
        plt.close(self.fig)

    def show(self):
        plt.show(block=True) 