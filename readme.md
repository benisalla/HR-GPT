## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```


# train 
```bash
source .venv/Scripts/activate
python ./trainer.py

```


# tensorboard
```bash
source .venv/Scripts/activate
tensorboard --logdir=logs --reload_interval=5 --port=6006
```

# run the app
```bash
source .venv/Scripts/activate

```