import pickle

from Phase1 import Constants


class Preferences:
    pref = {}

    @classmethod
    def load_pref(cls):
        try:
            with open(Constants.preferences_path, 'rb') as f:
                cls.pref = pickle.load(f)
        except FileNotFoundError:
            cls.pref = {Constants.pref_compression_type_key: Constants.VAR_BYTE_MODE}
            with open(Constants.preferences_path, 'wb') as f:
                pickle.dump(cls.pref, f, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def save_pref(cls):
        with open(Constants.preferences_path, 'wb') as f:
            pickle.dump(cls.pref, f, pickle.HIGHEST_PROTOCOL)