import numpy as np


a=[[1,2,3],[1,2,6]]
b=np.zeros(len(a[0]))

for i in range(len(a[0])):
    b[i]=a[0][i]/a[1][i]
    print(i)

print(b)