#!/bin/bash

# Download VADER lexicon
python download_nltk.py

# Start the app
uvicorn main:app --host 0.0.0.0 --port 10000
