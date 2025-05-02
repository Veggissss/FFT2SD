#!/bin/bash
# Train+Eval for unix
echo "Creating venv"
python3 -m venv venv

echo "Activating venv"
source venv/bin/activate

echo "Installing dependencies"
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu118 --extra-index-url https://pypi.org/simple

echo "Fine tuning models"
python3 model_train.py

echo "Running evals"
python3 model_eval.py

echo "DONE!"
deactivate