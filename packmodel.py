import argparse
import os
import sys
import time
from pathlib import Path

src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, src_dir)

from utils.util import read_json, write_json


def ensure_exist(path):
    if os.path.isfile(path):
        return True
    else:
        print("No such file:" + str(path))
        return False


def main(args_parsed):
    o_path = Path(args_parsed.output)
    o_path.mkdir(parents=True, exist_ok=True)

    res = dict()
    res["name"] = args_parsed.name
    res["createTime"] = int(time.time())
    if args_parsed.describe is not None:
        res["description"] = args_parsed.describe
    mods = list()
    for l_mod in args_parsed.model:
        mod_cont = dict()
        conf = l_mod[0]
        path = l_mod[1]
        # if not ensure_exist(conf) or not ensure_exist(path):
        if not ensure_exist(conf):
            continue
        try:
            cf_content = read_json(conf)
            for sec in ("name", "arch", "metrics", "tester"):
                mod_cont[sec] = cf_content[sec]
            mod_cont["path"] = path
            # copy model & rename

            mods.append(mod_cont)
        except Exception as e:
            print(str(e))
            continue
    res["models"] = mods
    write_json(res, o_path / (args_parsed.name+".json"))


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Models package utility')
    args.add_argument('-n', '--name', default=None, type=str, required=True,
                      help='packed models name')
    args.add_argument('-o', '--output', default=None, type=str, required=True,
                      help='packed models output directory')
    args.add_argument('-d', '--describe', default=None, type=str,
                      help='packed models description')
    args.add_argument('-m', '--model', nargs='+', action='append', required=True,
                      help='<path to model config> <path to saved model>')
    main(args.parse_args())
