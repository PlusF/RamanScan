import json


class ConfigLoader:
    def __init__(self, filename):
        with open(filename, 'r') as f:
            config = json.load(f)
        self.mode = config['mode']
        self.dt = 1 / config['FPS']
        self.port = f'COM{config["PORT"]}'
        self.baudrate = int(config['BAUDRATE'])
        self.temperature = int(config['TEMPERATURE'])
        self.folder = config['FOLDER']
