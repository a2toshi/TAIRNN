import datetime
import json

from .path_utils import *
from .print_utils import *


def print_args(args):
    """Print arguments"""
    if not isinstance(args, dict):
        args = vars(args)

    keys = args.keys()
    keys = sorted(keys)

    print("================================")
    for key in keys:
        print("{} : {}".format(key, args[key]))
    print("================================")


def save_args(args, filename):
    """Dump arguments as json file"""
    with open(filename, "w") as f:
        json.dump(vars(args), f, indent=4, sort_keys=True)


def check_args(args):
    """Check arguments"""

    if args.tag is None:
        tag = datetime.datetime.today().strftime("%Y%m%d_%H%M_%S")
        args.tag = tag
        print_info("Set tag = %s" % tag)

    # make log directory
    check_path(os.path.join(args.log_dir, args.tag), mkdir=True)

    # saves arguments into json file
    save_args(args, os.path.join(args.log_dir, args.tag, "args.json"))

    print_args(args)
    return args


def restore_args(filename):
    """Load argument file from file"""
    with open(filename, "r") as f:
        args = json.load(f)
    return args
