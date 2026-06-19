# network anomaly detection

random forest classifier on the NSL-KDD dataset. detects and classifies network intrusions into 5 categories.

- 0 = normal
- 1 = DoS
- 2 = Probe
- 3 = Privilege Escalation
- 4 = Unauthorized Access

## requirements

```
pip install -r requirements.txt
```

## dataset

download KDD+.txt from https://www.unb.ca/cic/datasets/nsl.html and drop it in data/

## usage

```
python src/train.py
```

outputs confusion matrices and saves the model to results/

## results

ran on 125,973 records with a 80/20 train-test split

| | accuracy | precision | recall | f1 |
|---|---|---|---|---|
| validation | 0.9982 | 0.9982 | 0.9982 | 0.9982 |
| test | 0.9981 | 0.9980 | 0.9981 | 0.9980 |

privilege escalation scores are lower because there are only 43 samples for that class in the whole dataset
