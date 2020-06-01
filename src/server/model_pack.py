import os
from pathlib import Path
from utils.util import read_json


class ModelPackLoader:
    def __init__(self):
        self.models = dict()
        self.config = None
        self.conf_path = ''

    def load_conf(self, config):
        self.conf_path = Path(config)
        self.config = read_json(config)
        if "models" not in self.config:
            return False
        for m in self.config["models"]:
            # check
            if not os.path.exists(m["path"]):
                if not (self.conf_path.parents[0] / m["path"]).exists():
                    print("Model", m["name"], "path", m["path"], "is not exist.")
                    # continue
                    m["path"] = str(self.conf_path.parents[0] / m["path"])

            self.models[m["name"]] = m

    def get_model(self, m_name):
        return self.models.get(m_name, None)
