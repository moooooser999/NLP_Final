# NLP Final Project Task 2
## Training
```bash=
./train_task2.sh
```
## Predicting
```bash=
./predict_task2.sh
```

You can also use
```bash=
python3 train_task2.py -h
python3 predict_task2.py -h
```
to unsderstand and modify the arguments.
## Load Pre-trained Model
Set
```bash=
    --model_name_or_path EthanChen0418/task2-macbert-multi-class
```
in the bash files, or use
```python=
from transformers import AutoTokenizer
from auto_model_for_task2 import AutoModelForTask2MultiClass

model_name_or_path = "EthanChen0418/task2-macbert-multi-class"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForTask2MultiClass.from_pretrained(model_name_or_path)
```