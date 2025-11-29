"""
Entry point when running: python -m rde_solver
"""

import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        prog = "rde-solver",
        description="RDE Solver Command Line Interface"
    )

    parser.add_argument("--version", action="version", version="RDE Solver 0.1.0")
    parser.add_argument("--help", action="help", help="Show this help message and exit")
    return parser.parse_args()

def main():
    return 0

if __name__ == "__main__":
    main()