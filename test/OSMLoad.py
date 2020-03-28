import unittest
from pathlib import Path
import os, sys

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(root_dir, "src")
sys.path.insert(0, src_dir)
# change cwd to root dir
os.chdir(root_dir)


class OSMTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def test_load(self):
        pass


if __name__ == '__main__':
    unittest.main()
