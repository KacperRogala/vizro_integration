''' File for importing and optionally formatting built-in Python datasets. ''' 

from sklearn.datasets import load_wine 
import pandas as pd 

# preparation of the first data set - dataset1
wine_dataset = load_wine() 
dataset1 = pd.DataFrame(data=wine_dataset.data, columns=wine_dataset.feature_names)
dataset1["class"] = wine_dataset.target 
## check if the dataset has been converted correctly
## print(dataset1.head()) 
