from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.properties import NumericProperty, StringProperty, ObjectProperty
from kivy.uix.popup import Popup
from kivy.clock import Clock
from kivy_garden.graph import Graph, LinePlot, ContourPlot
from kivy.config import Config
Config.set('graphics', 'width', '1200')
Config.set('graphics', 'height', '600')
from kivy.core.window import Window
from sigmakokicommander import SC101GCommander
from CircularProgressBar import CircularProgressBar
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import os
if os.name == 'nt':
    from pyAndorSDK2 import atmcd, atmcd_codes, atmcd_errors
else:  # Macで開発する際エラーが出ないようにする
    atmcd = atmcd_codes = atmcd_errors = None
import numpy as np
from dataloader import RamanHDFWriter, DataLoader
import datetime
import time
import serial
import threading
from ConfigLoader import ConfigLoader
from utils import subtract_baseline, generate_fake_data


def classify_key(keycode):
    key = keycode[1]
    if key in ['a', 'left']:
        axis = 1
        direction = '-'
    elif key in ['w', 'up']:
        axis = 2
        direction = '+'
    elif key in ['d', 'right']:
        axis = 1
        direction = '+'
    elif key in ['s', 'down']:
        axis = 2
        direction = '-'
    else:
        return None, None
    return axis, direction


# データの保存先を指定するダイアログ
class SaveDialogContent(FloatLayout):
    save = ObjectProperty(None)
    text_input = ObjectProperty(None)
    cancel = ObjectProperty(None)
    folder = StringProperty('')


# 保存していない状態で次の測定を始めてもよいか確認するダイアログ
class YesNoDialogContent(FloatLayout):
    message = ObjectProperty(None)
    yes = ObjectProperty(None)
    cancel = ObjectProperty(None)


# エラーを表示するダイアログ
class ErrorDialogContent(FloatLayout):
    message = ObjectProperty(None)
    ok = ObjectProperty(None)


class RSDriver(BoxLayout):
    current_temperature = NumericProperty(0)
    start_pos = ObjectProperty(np.zeros(2), force_dispatch=True)
    current_pos = ObjectProperty(np.zeros(2), force_dispatch=True)
    goal_pos = ObjectProperty(np.zeros(2), force_dispatch=True)
    jog_speed = NumericProperty(1)
    progress_value_acquire = NumericProperty(0)
    progress_value_scan = NumericProperty(0)
    # 露光時間
    integration = ObjectProperty(30)
    # 積算回数
    accumulation = ObjectProperty(3)
    # 測定間隔(距離）
    pixel_size = ObjectProperty(1.0)
    # スペクトル範囲
    line_x_range_1 = ObjectProperty(-209)
    line_x_range_2 = ObjectProperty(2102)
    line_y_range_1 = ObjectProperty(0)
    line_y_range_2 = ObjectProperty(10000)
    # マッピング範囲
    map_range_1 = ObjectProperty(1570)
    map_range_2 = ObjectProperty(1610)
    msg_important = StringProperty('Please initialize the detector.')
    msg_general = StringProperty('')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Window.bind(on_request_close=self.quit)

        self.folder = './'

        self.cl = ConfigLoader('./config.json')
        if self.cl.mode == 'RELEASE':
            self.folder = self.cl.folder
            if not os.path.exists(self.folder):
                os.mkdir(self.folder)

        self.xdata = DataLoader('./default_xdata.asc').spec_dict['./default_xdata.asc'].xdata
        self.ydata = np.ones([1, 1, 1, len(self.xdata)])
        self.coord_x = np.array([])
        self.coord_y = np.array([])
        self.last_ij = (0, 0)

        # 測定開始可能かどうかのフラグ
        self.validate_state_dict = {
            'temperature': False,
            'integration': True,
            'accumulation': True,
            'pixel_size': True,
            'not_busy': True,
        }

        self.ids.button_acquire.disabled = True
        self.ids.button_scan.disabled = True
        self.ids.button_save.disabled = True
        self.ids.button_save_as_hdf5.disabled = True

        self.saved_previous = True
        self.save_dialog = Popup(
            title="Save file",
            content=SaveDialogContent(
                save=self.save,
                cancel=lambda: self.save_dialog.dismiss(),
                folder=self.folder),
            size_hint=(0.9, 0.9)
        )
        self.save_as_hdf5_dialog = Popup(
            title="Save file as HDF5",
            content=SaveDialogContent(
                save=self.save_as_hdf5,
                cancel=lambda: self.save_as_hdf5_dialog.dismiss(),
                folder=self.folder),
            size_hint=(0.9, 0.9)
        )
        self.error_dialog = Popup(
            title="Error",
            content=ErrorDialogContent(
                message='Check the condition again.\nThe start position must be smaller than the goal position.',
                ok=lambda: self.error_dialog.dismiss(),
            ),
            size_hint=(0.4, 0.4)
        )

        self.open_ports()
        Clock.schedule_interval(self.ask_position, self.cl.dt)
        self.create_and_start_thread_position()

        self.input_widgets = [
            self.ids.input_pos_x,
            self.ids.input_pos_y,
            self.ids.input_integration,
            self.ids.input_accumulation,
            self.ids.input_pixel_size,
            self.ids.button_set_start,
            self.ids.button_set_goal,
            self.ids.button_go,
            self.ids.button_acquire,
            self.ids.button_scan,
            self.ids.button_save,
            self.ids.button_save_as_hdf5,
            self.ids.move_top,
            self.ids.move_bottom,
            self.ids.move_left,
            self.ids.move_right,
        ]

        # キーボード入力も受け付けるために必要
        self.is_moving = {
            1: {
                '+': False,
                '-': False,
            },
            2: {
                '+': False,
                '-': False,
            }
        }
        self.move_widgets = {
            1: {
                '+': self.ids.move_right,
                '-': self.ids.move_left,
            },
            2: {
                '+': self.ids.move_top,
                '-': self.ids.move_bottom,
            }
        }
        self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
        self._keyboard.bind(on_key_down=self._on_keyboard_down)
        self._keyboard.bind(on_key_up=self._on_keyboard_up)

        self.create_graph()

    def create_graph(self):
        # for spectrum
        self.graph_line = Graph(
            xlabel='Pixel number', ylabel='Counts',
            xmin=float(self.xdata.min()), xmax=float(self.xdata.max()), ymin=0, ymax=2000,
            x_ticks_major=500, x_ticks_minor=5,
            y_ticks_major=500, y_ticks_minor=5,
            x_grid_label=True, y_grid_label=True,
        )
        self.ids.graph_line.add_widget(self.graph_line)
        self.lineplot = LinePlot(color=[0, 1, 0, 1], line_width=1)
        self.graph_line.add_plot(self.lineplot)
        self.lineplot.points = [(i, i) for i in range(1024)]

        # for mapping
        self.graph_contour = Graph(
            xlabel='x [um]', ylabel='y [um]',
            xmin=0, xmax=10, ymin=0, ymax=10,
            x_ticks_major=5, y_ticks_major=5,
            x_grid_label=True, y_grid_label=True,
        )
        self.ids.graph_contour.add_widget(self.graph_contour)
        self.contourplot = ContourPlot()
        self.graph_contour.add_plot(self.contourplot)
        self.contourplot.xrange = (0, 10)
        self.contourplot.yrange = (0, 10)
        self.contourplot.data = np.arange(0, 10).reshape([10, 1]) * np.ones([1, 10])
        self.contourplot.draw()

    def open_ports(self):
        # 各装置との接続を開く
        if self.cl.mode == 'RELEASE':
            self.sdk = atmcd()
            self.ser = serial.Serial(self.cl.port, self.cl.baudrate, timeout=0.5, write_timeout=0)
            self.com = SC101GCommander(self.ser)
        elif self.cl.mode == 'DEBUG':
            self.sdk = None
            self.ser = None
            self.com = SC101GCommander()
        else:
            raise ValueError('Error with config.json. mode must be DEBUG or RELEASE.')

    def create_and_start_thread_position(self):
        # 画面が停止しないよう、座標の更新は別スレッドを立てて行う
        self.thread_pos = threading.Thread(target=self.update_position)
        self.thread_pos.daemon = True
        self.thread_pos.start()

    def create_and_start_thread_initialization(self):
        # 画面が停止しないよう、初期化と冷却は別スレッドを立てて行う
        self.thread_initialization = threading.Thread(target=self.initialize)
        self.thread_initialization.daemon = True
        self.thread_initialization.start()

    def create_and_start_thread_acquisition(self):
        # 画面が停止しないよう、acquireは別スレッドを立てて行う
        self.prepare_acquisition()
        self.thread_acq = threading.Thread(target=self.acquire)
        self.thread_acq.daemon = True
        self.thread_acq.start()

    def create_and_start_thread_scan(self):
        # 画面が停止しないよう、スキャンは別スレッドを立てて行う
        ok = self.prepare_scan()
        if not ok:
            return
        self.thread_scan = threading.Thread(target=self.scan)
        self.thread_scan.daemon = True
        self.thread_scan.start()

    def disable_inputs(self):
        # 測定中はボタンを押させない
        for widget in self.input_widgets:
            widget.disabled = True

    def activate_inputs(self):
        # 測定終了後操作を許可
        for widget in self.input_widgets:
            widget.disabled = False

    def initialize(self):
        # 初期化
        self.msg_important = 'Initializing...'
        self.ids.button_initialize.disabled = True
        if self.cl.mode == 'RELEASE':
            if self.sdk.Initialize('') == atmcd_errors.Error_Codes.DRV_SUCCESS:
                self.ids.button_initialize.disabled = True
                self.msg_important = 'Successfully initialized. Now cooling...'
                self.sdk.SetTemperature(self.cl.temperature)
                self.sdk.CoolerON()
            else:
                self.msg_important = 'Initialization failed.'
                self.ids.button_initialize.disabled = False
                return
        elif self.cl.mode == 'DEBUG':
            self.ids.button_initialize.disabled = True

        if self.cl.mode == 'RELEASE':
            ret, self.size_xdata, ypixels = self.sdk.GetDetector()
            self.sdk.handle_return(ret)
        elif self.cl.mode == 'DEBUG':
            self.size_xdata = 1024

        self.update_temperature()

    def update_temperature(self):
        # detectorの温度を監視し、規定の温度に下がっていれば準備OK
        if self.cl.mode == 'RELEASE':
            while True:
                ret, temperature = self.sdk.GetTemperature()
                if ret == atmcd_errors.Error_Codes.DRV_TEMP_STABILIZED:
                    break
                self.current_temperature = temperature
                time.sleep(self.cl.dt)
        elif self.cl.mode == 'DEBUG':
            self.current_temperature = self.cl.temperature
        self.msg_important = 'Cooling finished.'
        self.validate_state_dict['temperature'] = True
        self.check_if_ready()

    def update_graph_line(self, ydata):
        # スペクトルを表示
        self.lineplot.points = [(x, y) for x, y in zip(self.xdata, ydata)]
        self.graph_line.xmin = float(self.line_x_range_1)
        self.graph_line.xmax = float(self.line_x_range_2)
        self.graph_line.ymin = float(self.line_y_range_1)
        self.graph_line.ymax = float(self.line_y_range_2)

    def update_graph_contour(self):
        if len(self.ydata.shape) != 4:
            return
        # マップを表示
        map_data = self.ydata.sum(axis=2)
        graph_size_x, graph_size_y = self.ids.graph_contour.size
        map_size_x, map_size_y = self.goal_pos - self.start_pos
        xy_ratio_graph = graph_size_x / graph_size_y
        xy_ratio_map = map_size_x / map_size_y
        if xy_ratio_graph > xy_ratio_map:
            # グラフの方が横長
            map_size_x = map_size_y * xy_ratio_graph
        else:
            # グラフの方が縦長
            map_size_y = map_size_x / xy_ratio_graph
        self.graph_contour.xmin = float(self.start_pos[0])
        self.graph_contour.ymin = float(self.start_pos[1])
        self.graph_contour.xmax = float(self.start_pos[0] + map_size_x)
        self.graph_contour.ymax = float(self.start_pos[1] + map_size_y)
        self.graph_contour.x_ticks_major = self.pixel_size
        self.graph_contour.y_ticks_major = self.pixel_size
        self.contourplot.xrange = (self.start_pos[0], self.goal_pos[0])
        self.contourplot.yrange = (self.start_pos[1], self.goal_pos[1])
        signal_to_baseline = subtract_baseline(self.xdata, map_data, self.map_range_1, self.map_range_2)
        if signal_to_baseline is None:
            return
        self.contourplot.data = signal_to_baseline.T.reshape(self.ydata.shape[:2])

    def show_map_in_matplotlib(self) -> None:
        extent_mapping = (
            self.start_pos[0], self.goal_pos[0] + self.pixel_size,
            self.start_pos[1], self.goal_pos[1] + self.pixel_size)
        map_data = self.ydata.mean(axis=2).transpose(1, 0, 2)
        data = subtract_baseline(self.xdata, map_data, self.map_range_1, self.map_range_2)
        if data is None:
            return
        vmin = data.min()
        vmax = data.max()
        plt.imshow(data, alpha=0.9, extent=extent_mapping, origin='lower', cmap='hot', norm=Normalize(vmin=vmin, vmax=vmax))
        plt.show()

    def ask_position(self, dt):
        if self.ids.toggle_sync.state == 'down':
            self.com.get_position()

    def update_position(self):
        # 別スレッド内で動き続ける
        # シリアル通信の受信をすべて請け負う
        # 基本的に座標の情報以外は不要なためスルー
        while True:
            if self.cl.mode == 'RELEASE':
                msg = self.com.recv()
            elif self.cl.mode == 'DEBUG':
                msg = '0,0'
                time.sleep(1)
            try:
                pos_list = list(map(lambda x: int(x) * self.com.um_per_pulse, msg.split(',')))
                self.current_pos = np.array(pos_list)
            except ValueError:
                print(f'could not parse response: {msg}')
                continue

    def go(self, x, y):
        try:
            pos = list(map(float, [x, y]))
        except ValueError:
            self.msg_general = 'invalid value.'
            return

        if self.cl.mode == 'RELEASE':
            self.com.move_absolute(pos)
        elif self.cl.mode == 'DEBUG':
            self.current_pos = np.array(pos)

    def set_param(self, name, val, dtype, validate):
        # 各種パラメータの設定関数の一般化
        self.ids.button_acquire.disabled = True
        self.ids.button_scan.disabled = True
        self.validate_state_dict[name] = False

        # 有効な数値か確認
        try:
            val_casted = dtype(val)
        except ValueError:
            self.msg_general = f'Invalid value at {name}.'
            return

        # 数値の範囲を確認
        if validate(val_casted):
            self.msg_general = f'Set {name}.'
            exec(f'self.{name} = {val_casted}')
        else:
            self.msg_general = f'Invalid value at {name}.'
            return

        if name not in self.validate_state_dict:
            raise KeyError(f'key: {name} does not exist in validate_state_dict')
        self.validate_state_dict[name] = True
        self.check_if_ready()

    def check_if_ready(self):
        # 全ての値が正しければacquireボタンとscanボタンを使えるように
        ok = all(self.validate_state_dict.values())

        if ok:
            self.ids.button_acquire.disabled = False
            self.ids.button_scan.disabled = False
        else:
            self.ids.button_acquire.disabled = True
            self.ids.button_scan.disabled = True

    def set_integration(self, val):
        self.set_param(
            name='integration',
            val=val,
            dtype=float,
            validate=lambda x: 0.03 <= x <= 120  # なんとなく120秒を上限に．宇宙線の量を考えると妥当か？
        )

    def set_accumulation(self, val):
        self.set_param(
            name='accumulation',
            val=val,
            dtype=int,
            validate=lambda x: x >= 1
        )

    def set_pixel_size(self, val):
        self.set_param(
            name='pixel_size',
            val=val,
            dtype=float,
            validate=lambda x: x > 0.1
        )

    def set_line_x_range(self, line_x_range_1: str, line_x_range_2: str):
        self.set_param(
            name='line_x_range_1',
            val=line_x_range_1,
            dtype=float,
            validate=lambda x: True
        )
        self.set_param(
            name='line_x_range_2',
            val=line_x_range_2,
            dtype=float,
            validate=lambda x: True
        )
        if self.line_x_range_1 < self.line_x_range_2:
            last_ydata = self.ydata.sum(axis=2)[self.last_ij[0], self.last_ij[1], :]
            self.update_graph_line(last_ydata)

    def set_line_y_range(self, line_y_range_1: str, line_y_range_2: str):
        self.set_param(
            name='line_y_range_1',
            val=line_y_range_1,
            dtype=float,
            validate=lambda x: True
        )
        self.set_param(
            name='line_y_range_2',
            val=line_y_range_2,
            dtype=float,
            validate=lambda x: True
        )
        if self.line_y_range_1 < self.line_y_range_2:
            last_ydata = self.ydata.sum(axis=2)[self.last_ij[0], self.last_ij[1], :]
            self.update_graph_line(last_ydata)

    def set_map_range(self, map_range_1: str, map_range_2: str):
        self.set_param(
            name='map_range_1',
            val=map_range_1,
            dtype=float,
            validate=lambda x: True
        )
        self.set_param(
            name='map_range_2',
            val=map_range_2,
            dtype=float,
            validate=lambda x: True
        )
        if self.map_range_1 < self.map_range_2:
            self.update_graph_contour()

    def create_yesno_dialog(self, title, message, yes_func):
        self.dialog = Popup(
            title=title,
            content=YesNoDialogContent(
                message=message,
                yes=yes_func,
                cancel=lambda: self.dialog.dismiss()),
            size_hint=(0.4, 0.4)
        )
        self.dialog.open()

    def acquire_clicked(self):
        if not self.saved_previous:
            # 直前のデータが保存されていない場合確認ダイアログを出す
            self.create_yesno_dialog(
                title='Save previous data?',
                message='Previous data is not saved.\nOK?',
                yes_func=self.confirm_acquire_condition,
            )
            return
        self.confirm_acquire_condition()

    def scan_clicked(self):
        if self.start_pos[0] >= self.goal_pos[0] or self.start_pos[1] >= self.goal_pos[1]:
            self.error_dialog.open()
            return
        if not self.saved_previous:
            # 直前のデータが保存されていない場合確認ダイアログを出す
            self.create_yesno_dialog(
                title='Save previous data?',
                message='Previous data is not saved.\nOK?',
                yes_func=self.confirm_scan_condition,
            )
            return
        self.confirm_scan_condition()

    def get_condition_str_acquire(self):
        return 'Integration: {} sec\nAccumulation: {}'.format(self.integration, self.accumulation)

    def get_condition_str_scan(self):
        arr_x = np.arange(self.start_pos[0], self.goal_pos[0], self.pixel_size)
        arr_y = np.arange(self.start_pos[1], self.goal_pos[1], self.pixel_size)
        duration = np.ceil((arr_x.shape[0] * arr_y.shape[0]) * (self.integration * self.accumulation) / 60)
        return 'Integration: {} sec\nAccumulation: {}\nPixel size: {} um\nDuration: {} min'.format(
            self.integration, self.accumulation, self.pixel_size, duration
        )

    def confirm_acquire_condition(self):
        # 保存の確認ダイアログが開いていたら閉じる
        if not self.saved_previous:
            self.dialog.dismiss()
        # 露光時間・積算回数の確認ダイアログ
        self.create_yesno_dialog(
            title='Start acquire?',
            message=self.get_condition_str_acquire(),
            yes_func=self.start_acquire,
        )

    def confirm_scan_condition(self):
        # 保存の確認ダイアログが開いていたら閉じる
        if not self.saved_previous:
            self.dialog.dismiss()
        # 露光時間・積算回数・測定距離間隔・測定個所の数の確認ダイアログ
        self.create_yesno_dialog(
            title='Start scan?',
            message=self.get_condition_str_scan(),
            yes_func=self.start_scan,
        )

    def start_acquire(self):
        self.dialog.dismiss()
        # ポップアップウィンドウでyesと答えるとacquire開始
        self.create_and_start_thread_acquisition()

    def start_scan(self):
        self.dialog.dismiss()
        # ポップアップウィンドウでyesと答えるとスキャン開始
        self.create_and_start_thread_scan()

    def prepare_acquisition(self):
        # for GUI
        self.lineplot.points = []
        self.disable_inputs()
        self.validate_state_dict['not_busy'] = False
        self.ids.progress_acquire.max = self.accumulation
        self.progress_value_acquire = 0
        # for instruments
        if self.cl.mode == 'RELEASE':
            self.sdk.handle_return(self.sdk.SetAcquisitionMode(atmcd_codes.Acquisition_Mode.SINGLE_SCAN))
            self.sdk.handle_return(self.sdk.SetReadMode(atmcd_codes.Read_Mode.FULL_VERTICAL_BINNING))
            self.sdk.handle_return(self.sdk.SetTriggerMode(atmcd_codes.Trigger_Mode.INTERNAL))
            self.sdk.handle_return(self.sdk.SetExposureTime(self.integration))
            self.sdk.handle_return(self.sdk.PrepareAcquisition())
        elif self.cl.mode == 'DEBUG':
            print('prepare acquisition')

        self.ydata = np.zeros([1, 1, self.accumulation, self.size_xdata])
        # set_line_x_range, set_line_y_rangeで使う
        self.last_ij = (0, 0)

    def acquire(self, during_scan=False, i=0, j=0):
        ydata = np.zeros([self.accumulation, self.size_xdata])
        for k in range(self.accumulation):
            if not during_scan:
                self.msg_important = f'Acquisition {k + 1} of {self.accumulation}...'
            if self.cl.mode == 'RELEASE':
                self.sdk.handle_return(self.sdk.StartAcquisition())
                self.sdk.handle_return(self.sdk.WaitForAcquisition())
                ret, spec, first, last = self.sdk.GetImages16(1, 1, self.size_xdata)
                ydata[k] = np.array([spec])
                self.sdk.handle_return(ret)
            elif self.cl.mode == 'DEBUG':
                time.sleep(self.integration)
                print(f'acquired {k + 1}')
                spec = generate_fake_data(self.size_xdata)
                ydata[k] = np.array([spec])
            self.update_graph_line(ydata.sum(axis=0))  # show accumulated spectrum

            self.progress_value_acquire = k + 1

            self.ydata[i, j] = ydata  # line graphのx, y range変更時に使うために保存

        if not during_scan:  # finalize acquisition
            self.coord_x = np.array([self.current_pos[0]])
            self.coord_y = np.array([self.current_pos[1]])
            self.activate_inputs()
            self.msg_important = 'Acquisition finished.'
            self.validate_state_dict['not_busy'] = True
            self.saved_previous = False
            self.ids.button_save.disabled = False
            self.ids.button_save_as_hdf5.disabled = False

    def prepare_scan(self):
        # for GUI
        black = np.zeros([1024, 1024])
        black[0, 0] = 1
        self.contourplot.data = black
        self.disable_inputs()
        # for instruments
        self.prepare_acquisition()
        self.com.set_speed_max()
        self.com.set_acceleration_max()
        self.com.move_absolute(self.start_pos)
        distance = np.max(self.current_pos - self.start_pos)
        time.sleep(distance / self.com.max_speed + 1)

        # 座標計算
        arr_x = np.arange(self.start_pos[0], self.goal_pos[0], self.pixel_size)
        arr_y = np.arange(self.start_pos[1], self.goal_pos[1], self.pixel_size)
        self.goal_pos = np.array([arr_x[-1], arr_y[-1]])
        self.msg_general = (f'{arr_x.shape[0]} x {arr_y.shape[0]} = {arr_x.shape[0] * arr_y.shape[0]} points. '
                            f'The goal was set at {self.goal_pos}')
        self.coord_x, self.coord_y = np.meshgrid(arr_x, arr_y, indexing='ij')
        if self.coord_x.shape[0] == 0:
            self.error_dialog.open()
            self.activate_inputs()
            self.check_if_ready()
            return False
        self.ids.progress_scan.max = self.coord_x.shape[0] * self.coord_x.shape[1]

        # データ格納用numpy配列用意
        self.ydata = np.zeros([*self.coord_x.shape, self.accumulation, self.size_xdata])

        return True

    def scan(self):
        num_pos = self.coord_x.shape[0] * self.coord_x.shape[1]
        num_done = 0
        for i, (col_x, col_y) in enumerate(zip(self.coord_x, self.coord_y)):
            for j, coord in enumerate(zip(col_x, col_y)):
                time_left = np.ceil((num_pos - num_done) * (self.integration * self.accumulation) / 60)
                self.msg_important = f'Scan {num_done + 1} of {num_pos}... {time_left} minutes left.'

                if self.cl.mode == 'RELEASE':
                    self.com.move_absolute(coord)
                    distance = np.linalg.norm(np.array(coord) - self.current_pos)
                    time.sleep(distance / self.com.max_speed + 1)
                elif self.cl.mode == 'DEBUG':
                    self.current_pos = coord

                self.acquire(during_scan=True, i=i, j=j)
                self.update_graph_contour()

                self.last_ij = (i, j)
                num_done += 1
                self.progress_value_scan = num_done

        # 終了処理
        self.activate_inputs()
        self.validate_state_dict['not_busy'] = True
        self.saved_previous = False
        self.ids.button_save.disabled = False
        self.ids.button_save_as_hdf5.disabled = False
        self.msg_important = 'Scan finished.'

    def save(self, folder, filename):
        filename = os.path.basename(filename)
        if '.txt' not in filename:
            filename += '.txt'

        with open(os.path.join(folder, filename), 'w') as f:
            now = datetime.datetime.now()
            f.write(f'# time: {now.strftime("%Y-%m-%d-%H-%M")}\n')
            f.write(f'# integration: {self.integration}\n')
            f.write(f'# accumulation: {self.accumulation}\n')
            f.write(f'# pixel_size: {self.pixel_size}\n')
            f.write(f'# shape: {",".join(map(str, self.coord_x.shape))}\n')
            f.write(f'pos_x,{",".join(np.ravel(self.coord_x).astype(str))}\n')
            f.write(f'pos_y,{",".join(np.ravel(self.coord_y).astype(str))}\n')
            for col in self.ydata:
                for accumulated_y in col:
                    for y in accumulated_y:
                        f.write(','.join(y.astype(str)) + '\n')

        self.finalize_save_dialog(folder, filename)

    def save_as_hdf5(self, folder, filename):
        filename = os.path.basename(filename)
        if '.hdf5' not in filename:
            filename += '.hdf5'

        writer = RamanHDFWriter(os.path.join(folder, filename))
        writer.create_attr('time', datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"))
        writer.create_attr('integration', self.integration)
        writer.create_attr('accumulation', self.accumulation)
        writer.create_attr('pixel_size', self.pixel_size)
        writer.create_attr('shape', self.coord_x.shape)
        writer.create_attr('x_start', self.start_pos[0])
        writer.create_attr('y_start', self.start_pos[1])
        writer.create_attr('x_pad', self.pixel_size)
        writer.create_attr('y_pad', self.pixel_size)
        writer.create_attr('x_span', self.goal_pos[0] - self.start_pos[0])
        writer.create_attr('y_span', self.goal_pos[1] - self.start_pos[1])
        writer.create_dataset('pos_x', self.coord_x)
        writer.create_dataset('pos_y', self.coord_y)
        writer.create_dataset('xdata', self.xdata)
        writer.create_dataset('spectra', self.ydata)
        writer.close()

        self.finalize_save_dialog(folder, filename)

    def finalize_save_dialog(self, folder, filename):
        self.save_as_hdf5_dialog.dismiss()
        self.saved_previous = True
        self.save_dialog.folder = folder
        self.save_as_hdf5_dialog.folder = folder
        self.msg_important = f'Successfully saved to "{os.path.join(folder, filename)}".'

    def start_jogging(self, axis, direction):
        if not self.is_moving[axis][direction]:
            self.is_moving[axis][direction] = True
            self.move_widgets[axis][direction].state = 'down'
            if axis == 1:  # x軸を反転
                direction = '+' if direction == '-' else '-'
            self.com.jog(axis, direction, self.jog_speed)

    def stop_jogging(self, axis, direction):
        if self.is_moving[axis][direction]:
            self.is_moving[axis][direction] = False
            self.move_widgets[axis][direction].state = 'normal'
            self.com.stop(axis)

    def _keyboard_closed(self):
        self._keyboard.unbind(on_key_down=self._on_keyboard_down)
        self._keyboard = None

    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
        axis, direction = classify_key(keycode)
        if axis is None:
            return
        if self.validate_state_dict['not_busy'] is False:
            self.msg_general = 'Busy. Wait until the end.'
            return
        self.start_jogging(axis, direction)

    def _on_keyboard_up(self, keyboard, keycode):
        axis, direction = classify_key(keycode)
        if axis is None:
            return
        if self.validate_state_dict['not_busy'] is False:
            self.msg_general = 'Busy. Wait until the end.'
            return
        self.stop_jogging(axis, direction)

    def quit(self, instance):
        if self.cl.mode == 'RELEASE':
            self.sdk.ShutDown()
            self.ser.close()


class RSApp(App):
    def build(self):
        self.driver = RSDriver()
        return self.driver


if __name__ == '__main__':
    RSApp().run()
