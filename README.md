### Setup the environment

```
#setup a clean conda environment

conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

# install packages
pip3 install -r requirements.txtuer/roberta-base-finetuned-chinanews-chinese uer/roberta-base-finetuned-ifeng-chinese hfl/chinese-macbert-base \




```
### How to train the model 
```

# run this script for training
# if you want to change the argument
#see the arguments in train.py
bash run_train_multi_label.sh

```
### How to predict 
```
#run the following script to run prediction

bash run_predict.sh 

```
### How to predict with ensemble
```
# run the following script to predict with ensemble
bash run_ensemble.sh
```