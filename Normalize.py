import cv2
import pandas as pd
#import resize

img_label = pd.read_csv("output.csv")
output = []
for i in range(224):
    img=cv2.imread("images/N"+str(i+1)+".jpeg")
    if img is not None:
        xmax = img_label.iloc[i]["xmax"]
        ymax = img_label.iloc[i]["xmin"]
        xmin = img_label.iloc[i]["ymax"]
        ymin = img_label.iloc[i]["ymin"]
        rows,col=img.shape[:2]
        xmax = xmax/(col-1.0)
        ymax = ymax/(rows-1.0)
        xmin = xmin/(col-1.0)
        ymin = ymin/(rows-1.0)
    else:
        continue
    lael_norm = (xmin,xmax,ymin,ymax)
    output.append(lael_norm)
