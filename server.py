from thrift import Thrift
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer

import sys
import os
src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.append(src_dir)
sys.path.append(os.path.join(src_dir, "gen-py"))

from mapworld import MapWorldService
from mapworld.ttypes import *


class MapWorldHandler:
    def __init__(self):
        self.log = {}


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
