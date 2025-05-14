
:: Label+Train+Eval for windows
echo "Creating venv" &^
python -m venv venv &^

echo "Activating venv" &^
.\venv\Scripts\activate &^

echo "Installing dependencies" &^
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu118 --extra-index-url https://pypi.org/simple &^

echo "Auto labeling data" &^
python model_auto_label.py &^

echo "Fine tuning models" &^
python model_train.py &^

echo "Running evals" &^
python model_eval.py &^

echo "DONE!" &^
call deactivate