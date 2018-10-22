import glob
import scipy.io
import cv2
import numpy as np
import random

pixel_level_files = glob.glob("./annotations/pixel-level/*.mat")

# Process label
import pandas as pd

mat = [x[0] for x in scipy.io.loadmat("label_list.mat")['label_list'][0]]
result = [["null",0,0,0],]

r = [51*x for x in range(0,6)]
g = [51*x for x in range(0,6)]
b = [51*x for x in range(0,6)]

random.shuffle(r)
random.shuffle(g)
random.shuffle(b)

u = 0
v = 0
c = 1
for x in mat[1:]:
    result.append([x,r[u],g[v],b[c]])

    if (c == 5 and v == 5):
        u += 1
        v = 0
        c = 0
    elif (c == 5):
        v += 1
        c = 0
    else:
        c += 1

df = pd.DataFrame(data={"name":[x[0] for x in result],
               "r": [x[1] for x in result],
               "g": [x[2] for x in result],
               "b": [x[3] for x in result]})

df.to_csv("class_dict.csv", header=True,columns=["name","r","g","b"], index=False)

for x in pixel_level_files:
    mat = scipy.io.loadmat(x)
    mat = np.array(mat['groundtruth'])

    image = np.zeros((mat.shape[0], mat.shape[1], 3))

    for uu in range(len(result)):
        image[np.where(mat == uu)] = (result[uu][1], result[uu][2], result[uu][3])

    cv2.imwrite("./real_mask/" + x.split("/")[-1].replace(".mat", ".png"), image)