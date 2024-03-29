from thrift import Thrift
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer

import sys
import os
from pathlib import Path

src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.append(src_dir)
sys.path.append(os.path.join(src_dir, "gen-py"))

from server import ServerMain, ModelPackLoader

from mapworld import MapWorldService
from mapworld.ttypes import *
from mwfrontend import MapWorldMain
from mwfrontend.ttypes import *


class MapWorldHandler:
    def __init__(self):
        self.srv = None
        self.mp_loader = None
        # For client
        self.transport = None
        self.protocol = None
        self.client = None

    def initialize(self, InitRequest):
        res = Response()
        try:
            if self.mp_loader is None:
                mpl = ModelPackLoader()
                if not os.path.exists(InitRequest.config_path):
                    res.code = -2
                    res.msg = "The configure file cannot be found."
                    return res
                mpl.load_conf(InitRequest.config_path)
                if InitRequest.fr_addr and InitRequest.fr_port:
                    # Also init client
                    self.transport = TSocket.TSocket(InitRequest.fr_addr, InitRequest.fr_port)
                    self.transport = TTransport.TBufferedTransport(self.transport)
                    self.protocol = TBinaryProtocol.TBinaryProtocol(self.transport)
                    self.client = MapWorldMain.Client(self.protocol)
                    self.transport.open()
                self.mp_loader = mpl
                print("Config", InitRequest.config_path, "has been loaded to server.")
                res.code = 0
            else:
                res.code = -1
                res.msg = "The server has already been initialized."
            return res
        except:
            res.code = -100
            res.msg = "Unknown error."
            return res

    def deinit(self):
        res = Response()
        try:
            if self.srv is not None:
                del self.srv
                res.code += 1

            if self.transport is not None:
                self.transport.close()
                res.code += 2

            if self.mp_loader is not None:
                del self.mp_loader
                res.code += 4
            return res
        except:
            res.code = -100
            res.msg = "Unknown error."
            return res

    def doPred(self, PredRequest):
        res = Response()
        try:
            if self.mp_loader is not None:
                model = self.mp_loader.get_model(PredRequest.model_name)
                if model is None:
                    res.code = -2
                    res.msg = "The request model cannot be found."
                    return res
                if PredRequest.tmp_opt_path:
                    path = Path(PredRequest.tmp_opt_path)
                    if not os.path.exists(path) or not os.path.isdir(path):
                        res.code = -4
                        res.msg = "The request tmp_opt_path cannot be accessed."
                        return res
                else:
                    path = Path(os.path.dirname(os.path.abspath(__file__))) / "tmp" / "results"

                self.srv = ServerMain(self.client, path, PredRequest.n_gpu_use)
                if hasattr(PredRequest, 'cyclegan_type') and PredRequest.cyclegan_type is not None:
                    if PredRequest.model_name.split("-")[1] != "CycleGAN":
                        del self.srv
                        res.code = -5
                        res.msg = "The cyclegan_type is set, but with no CycleGAN model selected."
                        return res
                    else:
                        self.srv.set_cyclegan_type(PredRequest.cyclegan_type)
                if PredRequest.prescale:
                    self.srv.set_prescale(PredRequest.prescale)
                if PredRequest.batch_size:
                    self.srv.set_batch_size(PredRequest.batch_size)
                res = self.srv.pred(PredRequest.imgs_path, PredRequest.imgs_meta, model, PredRequest.id)
            else:
                del self.srv
                res.code = -1
                res.msg = "The server is not init."
            return res
        except Exception as e:
            res.code = -100
            res.msg = str(e)
            return res

    def getTask(self):
        # deprecated
        res = Response()
        try:
            res.code = 0
            res.msg = "The task is not running."
            if self.srv is not None:
                if self.srv.pred_th is not None:
                    if self.srv.pred_th.is_alive():
                        res.code = 1
                        res.msg = "The task is running."
            return res
        except:
            res.code = -100
            res.msg = "Unknown error."
            return res


def main():
    handler = MapWorldHandler()
    processor = MapWorldService.Processor(handler)
    transport = TSocket.TServerSocket(host='127.0.0.1', port=8888)
    tfactory = TTransport.TBufferedTransportFactory()
    pfactory = TBinaryProtocol.TBinaryProtocolFactory()

    server = TServer.TThreadedServer(processor, transport, tfactory, pfactory)
    print('Server started.')
    server.serve()


if __name__ == "__main__":
    main()
