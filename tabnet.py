# Offry: pip install pytorch-tabnet
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
import numpy as np

X_train = np.array([[1, 2, 3, 4], [1, 2, 3, 5], [1, 3, 5, 6], [2, 3, 4, 5]])
Y_train = np.array([1, 1, 4, 5])

X_valid = np.array([(3, 4, 6, 7), (1, 5, 6, 7)])
y_valid = np.array([1, 5])

X_test = np.array([(1, 2, 3, 5), (2, 4, 6, 7)])
clf = TabNetClassifier()  # TabNetRegressor()
clf.fit(X_train, Y_train, eval_set=[(X_valid, y_valid)])
preds = clf.predict(X_test)

print(preds)
