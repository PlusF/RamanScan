from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.properties import NumericProperty, StringProperty, ObjectProperty
from kivy.uix.popup import Popup
from kivy_garden.graph import Graph, LinePlot, ContourPlot
from kivy.config import Config
Config.set('graphics', 'width', '1200')
Config.set('graphics', 'height', '600')
from kivy.core.window import Window
from CircularProgressBar import CircularProgressBar

import os
if os.name == 'nt':
    from pyAndorSDK2 import atmcd, atmcd_codes, atmcd_errors
else:  # Macで開発する際エラーが出ないようにする
    atmcd = atmcd_codes = atmcd_errors = None
import numpy as np
import datetime
import time
import serial
import threading
from ConfigLoader import ConfigLoader
from utils import remove_cosmic_ray


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


class RASDriver(BoxLayout):
    current_temperature = NumericProperty(0)
    start_pos = ObjectProperty(np.zeros(2), force_dispatch=True)
    current_pos = ObjectProperty(np.zeros(2), force_dispatch=True)
    goal_pos = ObjectProperty(np.zeros(2), force_dispatch=True)
    progress_value_acquire = NumericProperty(0)
    progress_value_scan = NumericProperty(0)
    # 露光時間
    integration = ObjectProperty(30)
    # 積算回数
    accumulation = ObjectProperty(3)
    # 測定間隔(距離）
    interval = ObjectProperty(50)
    # num_posを指定したときの実際の測定間隔
    actual_interval = ObjectProperty(50)
    # 測定する位置の数
    num_pos = ObjectProperty(10)
    # intervalを指定したときの実際の測定位置の数
    actual_num_pos = ObjectProperty(10)
    # 測定間のスリープ時間
    sleep_sec = ObjectProperty(0)
    msg = StringProperty('Please initialize the detector.')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Window.bind(on_request_close=self.quit)

        self.folder = './'

        self.cl = ConfigLoader('./config.json')
        if self.cl.mode == 'RELEASE':
            self.folder = self.cl.folder
            if not os.path.exists(self.folder):
                os.mkdir(self.folder)

        self.xdata = np.array([])
        self.ydata = np.array([])
        self.coord = np.array([])

        # 測定開始可能かどうかのフラグ
        self.validate_state_dict = {
            'temperature': False,
            'integration': True,
            'accumulation': True,
            'interval': True,
            'num_pos': True,
        }

        self.create_graph()

        self.ids.button_acquire.disabled = True
        self.ids.button_scan.disabled = True
        self.ids.button_save.disabled = True

        self.saved_previous = True
        self.save_dialog = Popup(
            title="Save file",
            content=SaveDialogContent(
                save=self.save,
                cancel=lambda: self.save_dialog.dismiss(),
                folder=self.folder),
            size_hint=(0.9, 0.9)
        )
        self.error_dialog = Popup(
            title="Error",
            content=ErrorDialogContent(
                message='Check the condition again.',
                ok=lambda: self.error_dialog.dismiss(),
            ),
            size_hint=(0.4, 0.4)
        )

        self.open_ports()
        self.create_and_start_thread_position()

    def create_graph(self):
        # for spectrum
        self.graph_line = Graph(
            xlabel='Pixel number', ylabel='Counts',
            xmin=0, xmax=1023, ymin=0, ymax=1023,
            x_ticks_major=100, x_ticks_minor=2, y_ticks_major=200,
            x_grid_label=True, y_grid_label=True,
        )
        self.ids.graph_line.add_widget(self.graph_line)
        self.lineplot = LinePlot(color=[0, 1, 0, 1], line_width=1)
        self.graph_line.add_plot(self.lineplot)
        self.lineplot.points = [(i, i) for i in range(1024)]

        # for mapping
        self.graph_contour = Graph(
            xlabel='Pixel number', ylabel='Position',
            xmin=0, xmax=1023, ymin=0, ymax=9,
            x_ticks_major=100, x_ticks_minor=2, y_ticks_major=5,
            x_grid_label=True, y_grid_label=True,
        )
        self.ids.graph_contour.add_widget(self.graph_contour)
        self.contourplot = ContourPlot()
        self.graph_contour.add_plot(self.contourplot)
        self.contourplot.xrange = (0, 1023)
        self.contourplot.yrange = (0, 9)
        self.contourplot.data = np.arange(0, 10).reshape([10, 1]) * np.ones([10, 1024])
        self.contourplot.draw()

    def open_ports(self):
        # 各装置との接続を開く
        if self.cl.mode == 'RELEASE':
            self.sdk = atmcd()
            self.ser = serial.Serial(self.cl.port, self.cl.baudrate, write_timeout=0)
            # self.hsc = HSC103Controller(self.ser)
        elif self.cl.mode == 'DEBUG':
            self.sdk = None
            self.ser = None
            # self.hsc = HSC103Controller(self.ser)
        else:
            raise ValueError('Error with config.json. mode must be DEBUG or RELEASE.')

    def create_and_start_thread_position(self):
        # 画面が停止しないよう、座標の更新は別スレッドを立てて行う
        self.thread_pos = threading.Thread(target=self.update_position)
        self.thread_pos.daemon = True
        self.thread_pos.start()

    def create_and_start_thread_cooling(self):
        # 画面が停止しないよう、温度の監視は別スレッドを立てて行う
        self.thread_cool = threading.Thread(target=self.update_temperature)
        self.thread_cool.daemon = True
        self.thread_cool.start()

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

    def set_interval_or_num_pos(self):
        # スキャンの前にintervalまたはnum_posの値が正しいか確認
        use_interval = self.ids.toggle_interval.state == 'down'
        use_num_pos = self.ids.toggle_num_pos.state == 'down'
        if use_interval:
            if self.interval == 0:
                self.msg = 'Check the num_pos value again.'
                self.error_dialog.open()
                return False
            # interval指定でscanするときは、num_posが2以上にならなければいけない
            self.actual_num_pos = int(np.linalg.norm(self.goal_pos - self.start_pos) // self.interval + 1)
            if self.actual_num_pos <= 1:
                self.msg = 'Check the interval value again.'
                self.error_dialog.open()
                return False
            self.actual_interval = self.interval
        elif use_num_pos:
            # num_pos指定の場合は同じ場所で何回も取得することを許す
            self.actual_num_pos = self.num_pos
            self.actual_interval = np.linalg.norm(self.goal_pos - self.start_pos) / self.actual_num_pos

        return True

    def disable_buttons(self):
        # 測定中はボタンを押させない
        self.ids.button_acquire.disabled = True
        self.ids.button_scan.disabled = True
        self.ids.button_set_start.disabled = True
        self.ids.button_set_goal.disabled = True
        self.ids.button_go.disabled = True

    def activate_buttons(self):
        # 測定終了後操作を許可
        self.ids.button_acquire.disabled = False
        self.ids.button_scan.disabled = False
        self.ids.button_set_start.disabled = False
        self.ids.button_set_goal.disabled = False
        self.ids.button_go.disabled = False

    def clear_things(self):
        self.ydata = np.empty([0, self.xpixels])
        self.coord = np.empty([0, 2])
        self.lineplot.points = []

    def initialize(self):
        # 初期化
        if self.cl.mode == 'RELEASE':
            if self.sdk.Initialize('') == atmcd_errors.Error_Codes.DRV_SUCCESS:
                self.ids.button_initialize.disabled = True
                self.msg = 'Successfully initialized. Now cooling...'
                self.sdk.SetTemperature(self.cl.temperature)
                self.sdk.CoolerON()
            else:
                self.msg = 'Initialization failed.'
                return
        elif self.cl.mode == 'DEBUG':
            self.ids.button_initialize.disabled = True
        self.create_and_start_thread_cooling()

        if self.cl.mode == 'RELEASE':
            ret, self.xpixels, ypixels = self.sdk.GetDetector()
            self.sdk.handle_return(ret)
        elif self.cl.mode == 'DEBUG':
            self.xpixels = 1024

    def update_temperature(self):
        # detectorの温度を監視し、規定の温度に下がっていれば
        if self.cl.mode == 'RELEASE':
            while True:
                ret, temperature = self.sdk.GetTemperature()
                if ret == atmcd_errors.Error_Codes.DRV_TEMP_STABILIZED:
                    break
                self.current_temperature = temperature
                time.sleep(self.cl.dt)
        elif self.cl.mode == 'DEBUG':
            self.current_temperature = self.cl.temperature
        self.msg = 'Cooling finished.'
        self.validate_state_dict['temperature'] = True
        self.check_if_ready()

    def update_graph_line(self):
        # スペクトルを表示
        # TODO: show the spectrum accumulated
        ydata = self.ydata
        if self.cl.cosmic_ray_removal:
            ydata = remove_cosmic_ray(ydata)

        self.xdata = np.arange(0, self.xpixels, 1)
        self.graph_line.xmin = float(np.min(self.xdata))
        self.graph_line.xmax = float(np.max(self.xdata))
        self.graph_line.ymin = float(np.min(ydata[-1]))
        self.graph_line.ymax = float(max(np.max(ydata[-1]), np.min(ydata[-1]) + 0.1))
        self.lineplot.points = [(x, y) for x, y in zip(self.xdata, ydata[-1])]

    def update_graph_contour(self):
        # マップを表示
        # TODO: show the spectrum accumulated
        map_data = self.ydata
        if self.cl.cosmic_ray_removal:
            map_data = remove_cosmic_ray(map_data)

        self.xdata = np.arange(0, self.xpixels, 1)
        self.graph_contour.xmax = self.xpixels - 1
        self.graph_contour.ymax = self.actual_num_pos * self.accumulation - 1
        self.contourplot.xrange = (0, self.xpixels - 1)
        self.contourplot.yrange = (0, len(self.ydata) - 1)

        # なぜかわからないが[::-1]しないと正しく表示されない
        self.contourplot.data = map_data.reshape(self.ydata.shape[::-1])

    def update_position(self):
        # 別スレッド内で動き続ける
        # たまに座標の更新に数秒~10数秒のラグがあるのはなぜ？
        while True:
            # self.hsc.get_position()
            # msg = self.hsc.recv()
            #
            # try:
            #     pos_list = list(map(lambda x: int(x) * self.hsc.um_per_pulse, msg.split(',')))
            # except ValueError:
            #     print(msg)
            #     time.sleep(self.cl.dt)
            #     continue
            #
            # self.current_pos = np.array(pos_list)
            time.sleep(self.cl.dt)

    def go(self, x, y):
        try:
            pos = np.array([x, y], float)
        except ValueError:
            self.msg = 'invalid value.'
            return

        if self.cl.mode == 'RELEASE':
            self.hsc.move_linear(pos - self.current_pos)
        elif self.cl.mode == 'DEBUG':
            self.current_pos = pos

    def set_param(self, name, val, dtype, validate):
        # 各種パラメータの設定関数の一般化
        self.ids.button_acquire.disabled = True
        self.ids.button_scan.disabled = True
        self.validate_state_dict[name] = False

        # 有効な数値か確認
        try:
            val_casted = dtype(val)
        except ValueError:
            self.msg = f'Invalid value at {name}.'
            return

        # 数値の範囲を確認
        if validate(val_casted):
            self.msg = f'Set {name}.'
            exec("self.%s = %d" % (name, val_casted))
        else:
            self.msg = f'Invalid value at {name}.'
            return

        if name not in self.validate_state_dict:
            raise KeyError(f'key: {name} does not exist in validate_state_dict')
        self.validate_state_dict[name] = True
        self.check_if_ready()

    def check_if_ready(self):
        # 全ての値が正しければacquireボタンとscanボタンを使えるように
        # interval指定駆動の時，intervalの値が正しい必要があるが，num_posは正しくなくても良い．逆もしかり．
        use_interval = self.ids.toggle_interval.state == 'down'
        use_num_pos = self.ids.toggle_num_pos.state == 'down'
        interval_ok = (not use_interval) or (use_interval and self.validate_state_dict['interval'])
        num_pos_ok = (not use_num_pos) or (use_num_pos and self.validate_state_dict['num_pos'])
        keys_else = [key for key in self.validate_state_dict.keys() if key not in ['interval', 'num_pos']]
        else_ok = all([self.validate_state_dict[key] for key in keys_else])

        if interval_ok and num_pos_ok and else_ok:
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

    def set_interval(self, val):
        self.ids.toggle_interval.state = 'down'
        self.apply_interval()
        self.set_param(
            name='interval',
            val=val,
            dtype=float,
            validate=lambda x: x > 0
        )

    def set_num_pos(self, val):
        self.ids.toggle_num_pos.state = 'down'
        self.apply_num_pos()
        self.set_param(
            name='num_pos',
            val=val,
            dtype=int,
            validate=lambda x: x > 0)

    def apply_interval(self):
        # interval側がオンの時はnum_pos側はオフ
        state = self.ids.toggle_interval.state
        if state == 'down':
            self.ids.toggle_num_pos.state = 'normal'
        elif state == 'normal':
            self.ids.toggle_num_pos.state = 'down'
        self.check_if_ready()

    def apply_num_pos(self):
        # num_pos側がオンの時はinterval側はオフ
        state = self.ids.toggle_num_pos.state
        if state == 'down':
            self.ids.toggle_interval.state = 'normal'
        elif state == 'normal':
            self.ids.toggle_interval.state = 'down'
        self.check_if_ready()

    def set_sleep_sec(self, val):
        self.set_param(
            name='sleep_sec',
            val=val,
            dtype=float,
            validate=lambda x: x >= 0
        )

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
            # self.unsaved_dialog_acquire.open()
            return
        self.confirm_acquire_condition()

    def scan_clicked(self):
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
        if self.ids.toggle_interval.state == 'down':
            return 'Integration: {} sec\nAccumulation: {}\nInterval: {} um'.format(
                self.integration, self.accumulation, self.interval
            )
        else:
            return 'Integration: {} sec\nAccumulation: {}\nNumber of positions: {}'.format(
                self.integration, self.accumulation, self.num_pos
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
        self.clear_things()
        self.disable_buttons()
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

    def acquire(self, during_scan=False):
        for i in range(self.accumulation):
            if self.cl.mode == 'RELEASE':
                self.sdk.handle_return(self.sdk.StartAcquisition())
                self.sdk.handle_return(self.sdk.WaitForAcquisition())
                ret, spec, first, last = self.sdk.GetImages16(1, 1, self.xpixels)
                self.ydata = np.append(self.ydata, np.array([spec]), axis=0)
                self.sdk.handle_return(ret)
            elif self.cl.mode == 'DEBUG':
                time.sleep(self.integration)
                print(f'acquired {i + 1}')
                spec = np.expand_dims(np.sin(np.linspace(-np.pi, np.pi, self.xpixels)), axis=0) * np.random.randint(1, 10)
                noise = np.random.random(self.xpixels) * 10
                cosmic_ray = np.zeros(self.xpixels)
                cosmic_ray[np.random.randint(0, self.xpixels)] = 100
                spec += noise + cosmic_ray
                self.ydata = np.append(self.ydata, spec, axis=0)
            self.coord = np.append(self.coord, self.current_pos.reshape([1, 2]), axis=0)
            self.update_graph_line()

            self.progress_value_acquire = i + 1

        if not during_scan:
            self.activate_buttons()
            self.msg = 'Acquisition finished.'
            self.saved_previous = False
            self.ids.button_save.disabled = False

    def prepare_scan(self):
        ok = self.set_interval_or_num_pos()
        if not ok:
            return False
        # for GUI
        self.clear_things()
        black = np.zeros([1024, 1024])
        black[0, 0] = 1
        self.contourplot.data = black
        self.ids.progress_scan.max = self.actual_num_pos
        self.disable_buttons()
        # for instruments
        self.prepare_acquisition()
        # self.hsc.set_speed_max()
        # self.hsc.move_abs(self.start_pos)
        # distance = np.max(self.current_pos - self.start_pos)
        # time.sleep(distance / self.hsc.max_speed + 1)

        return True

    def scan(self):
        for i in range(self.actual_num_pos):
            time_left = np.ceil((self.actual_num_pos - i) * (self.integration * self.accumulation + self.sleep_sec) / 60)
            self.msg = f'Acquisition {i + 1} of {self.actual_num_pos}... {time_left} minutes left. (interval: {self.actual_interval})'

            # 移動が必要なとき(同じ場所での測定では移動はいらない)
            if abs(self.actual_interval) > 0:
                point = self.start_pos + (self.goal_pos - self.start_pos) * i / (self.actual_num_pos - 1)
                if self.cl.mode == 'RELEASE':
                    # self.hsc.move_abs(point)
                    distance = np.max(self.current_pos - self.start_pos)
                    # time.sleep(distance / self.hsc.max_speed + 1)
                elif self.cl.mode == 'DEBUG':
                    self.current_pos = point

            self.acquire(during_scan=True)
            self.update_graph_contour()

            self.progress_value_scan = i + 1

            self.msg = f'Sleeping... ({i + 1}/{self.actual_num_pos}, {time_left} minutes left.)'
            time.sleep(self.sleep_sec)

        # 終了処理
        self.activate_buttons()
        self.saved_previous = False
        self.ids.button_save.disabled = False
        self.msg = 'Scan finished.'

    def save(self, path, filename):
        filename = os.path.basename(filename)
        if '.txt' not in filename:
            filename += '.txt'

        with open(os.path.join(path, filename), 'w') as f:
            now = datetime.datetime.now()
            f.write(f'# time: {now.strftime("%Y-%m-%d-%H-%M")}\n')
            f.write(f'# integration: {self.integration}\n')
            f.write(f'# accumulation: {self.accumulation}\n')
            use_interval = self.ids.toggle_interval.state == 'down'
            use_num_pos = self.ids.toggle_num_pos.state == 'down'
            f.write(f'# interval: {use_interval} {self.interval}\n')
            f.write(f'# num_pos: {use_num_pos} {self.num_pos}\n')
            f.write(f'pos_x,{",".join(self.coord[:, 0].astype(str))}\n')
            f.write(f'pos_y,{",".join(self.coord[:, 1].astype(str))}\n')
            for x, y in zip(self.xdata.astype(str), self.ydata.T.astype(str)):
                f.write(f'{x},{",".join(y)}\n')

        self.save_dialog.dismiss()
        self.saved_previous = True
        self.save_dialog.folder = path
        self.msg = f'Successfully saved to "{os.path.join(path, filename)}".'

    def quit(self, instance):
        if self.cl.mode == 'RELEASE':
            self.sdk.ShutDown()
            self.ser.close()


class RASApp(App):
    def build(self):
        self.driver = RASDriver()
        return self.driver


if __name__ == '__main__':
    RASApp().run()
