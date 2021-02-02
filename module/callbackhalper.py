from module.config import Config
import sys, importlib


class CallbackHelper:
    def __init__(self, config: Config):
        self.config = config

    def get_callbacks(self) -> list:
        callback_list = []
        for t in self.config.CNN_TRAIN_CALLBACK:
            sys.stdout.write("\rCreating %s callback." % t)
            sys.stdout.flush()

            # 從文字import類別
            TrainCallback = None
            try:
                imp = importlib.import_module("module.callback." + t)
                TrainCallback = getattr(imp, 'TrainCallback')
            except Exception as ex:
                print("Can't load module.callback." + t)
                print("Process  terminated.")
                sys.exit()
            cb = TrainCallback(self.config)
            callback_list.append(cb.create_callback())

        return callback_list
