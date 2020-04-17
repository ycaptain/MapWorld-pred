import unittest
import os
import sys

from thrift import Thrift
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(root_dir, "src")
sys.path.insert(0, src_dir)
sys.path.append(os.path.join(src_dir, "gen-py"))
# change cwd to root dir
os.chdir(root_dir)

from mapworld import MapWorldService
from mapworld.ttypes import *


class SrvTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Make socket
        cls.transport = TSocket.TSocket('::1', 77971)

        # Buffering is critical. Raw sockets are very slow
        cls.transport = TTransport.TBufferedTransport(cls.transport)

        # Wrap in a protocol
        cls.protocol = TBinaryProtocol.TBinaryProtocol(cls.transport)

        # Create a client to use the protocol encoder
        cls.client = MapWorldService.Client(cls.protocol)

    def test_init(self):
        self.transport.open()
        ir = InitRequest("saved/models/DeepLabTest/0331_043602/config.json",
                         "saved/models/DeepLabTest/0331_043602/model_best.pth")
        self.client.initialize(ir)
        self.transport.close()


if __name__ == '__main__':
    unittest.main()
