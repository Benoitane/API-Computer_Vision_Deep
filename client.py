import os
import requests
import random
import numpy as np

np.random.seed(2021)
random.seed(2021)

url = 'http://localhost:8080/classification_model'


labeltestdir = 'data/images/test/' + random.choice(['2885','1680','1320','280','1141']) + '/'
path_img = labeltestdir + random.choice([x for x in os.listdir(labeltestdir) if os.path.isfile(os.path.join(labeltestdir, x))])
print(path_img)
with open(path_img, 'rb') as img:
    name_img = os.path.basename(path_img)
    print(name_img)
    files = {'file': (path_img, img)}
    with requests.Session() as s:
        r = s.post(url, files=files)
