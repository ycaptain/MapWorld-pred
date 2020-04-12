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

from mapworld import MapWorldService
from mapworld.ttypes import *

from demo import DemoMain
from parse_config import ConfigParser


class ServerMain(DemoMain):
    def __init__(self, config, client):
        super().__init__(config)
        self.client = client


class MapWorldHandler:
    def __init__(self):
        self.srv = None
        # For client
        self.transport = None
        self.protocol = None
        self.client = None

    def initialize(self, InitRequest):
        if self.srv is None:
            config = ConfigParser(ConfigParser.from_file(InitRequest.config_path),
                                  Path(InitRequest.model_path),
                                  dict())
            if InitRequest.fr_addr and InitRequest.fr_port:
                # Also init client
                self.transport = TSocket.TSocket(InitRequest.fr_addr, InitRequest.fr_port)
                self.transport = TTransport.TBufferedTransport(self.transport)
                self.protocol = TBinaryProtocol.TBinaryProtocol(self.transport)
                self.client = MapWorldService.Client(self.protocol)
                self.transport.open()
            self.srv = ServerMain(config, self.client)

    def deinit(self):
        if self.srv is not None:
            del self.srv

        if self.transport is not None:
            self.transport.close()

    def doPred(self, PredRequest):
        imgs = PredRequest.imgs_path


def main():
    handler = MapWorldHandler()
    processor = MapWorldService.Processor(handler)
    transport = TSocket.TServerSocket(host='::1', port=77971)
    tfactory = TTransport.TBufferedTransportFactory()
    pfactory = TBinaryProtocol.TBinaryProtocolFactory()

    server = TServer.TThreadedServer(processor, transport, tfactory, pfactory)
    print('Starting the server...')
    server.serve()


if __name__ == "__main__":
    main()
