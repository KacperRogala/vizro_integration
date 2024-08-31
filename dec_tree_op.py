''' A file for modeling the selected dataset with a decision tree classifier model''' 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix 
import numpy as np
import pandas as pd 

from modeling.models.dec_tree.dec_tree_main import dec_tree
from data.built_in_data import dataset1 

X = dataset1.drop(['class'], axis=1)
y = dataset1["class"] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234) 

model = dec_tree.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Results of modeling 
result = pd.DataFrame({'class': np.array(y_test), 'class_pred':y_pred})
# print(result)
