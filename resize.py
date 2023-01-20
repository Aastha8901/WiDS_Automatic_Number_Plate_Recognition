import cv2
import numpy as np


resize_img = []
for i in range(248):
    
        u = cv2.imread("images/N"+str(i+1)+".jpeg")
        if u is not None:
            resize_img.append(cv2.resize(u,(224,224)))
        else:
            continue

normalize = np.array(resize_img)
print(normalize[0][1].shape)

b = normalize[0][0][0][0]
norm_img = []
for i in range(227):
    a=normalize[i]
    d=[]
    for j in range(224):
        for k in range(224):
            if(max(a[j][k]>b)):
                b = max(a[j][k])
    for j in range(224):
        c = []
        for k in range(224):
           c.append(np.divide(a[j][k],b))
        d.append(c)
    norm_img.append(d)

print(b)




