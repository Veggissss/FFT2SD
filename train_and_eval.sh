#!/bin/bash
# Label+Train+Eval for unix
echo "Creating venv"
python3 -m venv venv

echo "Activating venv"
source venv/bin/activate

echo "Installing dependencies"
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu118 --extra-index-url https://pypi.org/simple

echo "Auto labeling data"
python3 model_auto_label.py

echo "Fine tuning models"
python3 model_train.py

echo "Running evals"
python3 model_eval.py

echo "DONE!"
deactivate