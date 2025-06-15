
# num16-cnn

Projekt klasifikácie ručne písaných číslic (0–16) pomocou CNN v PyTorch.

## Zloženie
- `dataset.py` — Trieda Num16Dataset
- `models.py` — Dve CNN architektúry (CNN_A a CNN_B)
- `train.py` — Tréning + TensorBoard
- `main.py` — Spustenie modelu
- `requirements.txt` — Zoznam balíkov

## Spustenie

```bash
pip install -r requirements.txt
python main.py
```

## TensorBoard
```bash
tensorboard --logdir=runs
```

## Dáta
Repozitár musí obsahovať:
- `images.zip` rozbalený do `images/`
- `labels.csv` (súbor so štítkami)
