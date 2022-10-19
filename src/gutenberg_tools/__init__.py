"""
    __init__.py
    Created by Pierre Tessier
    10/17/22 - 4:22 PM
    Description:
    # Enter file description
 """
import argparse
import sys


def main(argv=None):
    parser = argparse.ArgumentParser()
    args = parser.parse_args(argv)
    pass


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
