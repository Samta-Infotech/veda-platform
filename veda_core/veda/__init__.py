"""VEDA NL→SQL engine (package). Importing sets offline model mode."""
import os, sys, logging
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
logging.disable(logging.INFO)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
