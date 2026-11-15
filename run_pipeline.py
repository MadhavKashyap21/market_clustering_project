"""Command-line entry point: python run_pipeline.py [config.yaml]"""
import sys

from src.pipeline import run_all

if __name__ == "__main__":
    run_all(sys.argv[1] if len(sys.argv) > 1 else "config.yaml")
