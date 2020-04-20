import os


class ModelPackLoader:
    def __init__(self):
        self.models = dict()
        self.config = None

    def load_conf(self, config):
        self.config = config
        if "models" not in config:
            return False
        for m in config["models"]:
            # check
            if not os.path.exists(m["path"]):
                print("Model", m["name"], "path", m["path"], "is not exist.")
                continue

            self.models[m["name"]] = m

    def get_model(self, m_name):
        return self.models.get(m_name, None)
