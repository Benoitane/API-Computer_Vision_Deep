## Presentation

In this project, you can find :

- 6 python files (train.py, evaluate.py, server.py, client.py, utils_model.py and utils_preprocessing.py)

- 3 folders :
	- data : contains your images and the cdv
	- graphs : contains outposts exported as png 
	- save :  contains pytorch model .pt

- 1 txt file (requirements.txt)

## How to make it works

1) Put the downloaded folder where you want
2) From your terminal : cd PATH_TO_THE_FOLDER
3) From your terminal : pip install -r requirements.txt
4) you can now run train.py, evaluate.py and server.py from your terminal

All dependencies have been tested with virtual environment (command python3 -m venv testenv) 

## What's going on ?

- train.py : analyse the dataset, create split and assign observations in train/val/test folders in order to fine-tune a pertained model (mobilenet V2 as asked in instructions). Next, it exports model into save folder and 2 graphs (loss and accuracy) in graphs folder.
- evaluate.py : evaluate fine-tuned model on test set in order to appreciate performance with new observations
- server.py : flask server to send images for classification and also exports into graphs folder ranked predictions according to respective probabilities with plot of tested image. This server has been tested with client.py. Do not hesitate to check it.
-  utils_model.py and utils_preprocessing.py : used to store functions
