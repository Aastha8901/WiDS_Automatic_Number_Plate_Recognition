from sklearn.model_selection import train_test_split
import resize
import Normalize
import numpy as np

X = np.array(resize.norm_img, dtype=np.float32)
Y = np.array(Normalize.lael_norm, dtype=np.float32)

train_x, test_x,train_y,test_y = train_test_split(X, Y, train_size=0.8, random_state=0)
print(train_x.shape)
