import os
import numpy as np

files = os.listdir('localmaps')
path = os.path.join('localmaps', str(2) + '.png')
print(len(files))
print(path)
data = np.loadtxt('./trajectory/0.csv', delimiter=',').reshape(1, 60)
print(data.shape)
print(data)