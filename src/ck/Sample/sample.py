
import numpy as np

f = open(r'datalab/2217/DatasetA_train_20180813/DatasetA_train_20180813/attributes_per_class.txt')
data_per_class = f.readlines()

data = []
for line in data_per_class:
    data.append(line.split())
print(data)










