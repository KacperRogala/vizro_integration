''' File storing functions for custom data processing. Processed data is visualized via Vizro dashboard''' 
import pandas as pd
from pandas.api.types import is_numeric_dtype

# A function that for each column of a dataset with a numeric data type returns 
# the number of objects whose value is not in the range (q1-1.5*IQR, q3+1.5*IQR)
def IQR_count(dataset):
    num_cols = []
    for column in dataset.columns:
        if is_numeric_dtype(dataset[column]):
            num_cols.append(column) 

    dataset_num = dataset.loc[:,num_cols] # dataset with only numerical columns 

    # IQR masks calculation 
    IQR_masks = {} 
    for column in dataset_num.columns:
        q1 = dataset_num[column].quantile(0.25)
        q3 = dataset_num[column].quantile(0.75)
        IQR = q3 - q1 
        IQR_mask = (dataset_num[column] > (q3 + 1.5*IQR)) | (dataset_num[column] < (q1 - 1.5*IQR))
        IQR_masks[column] = IQR_mask

    # Calculating the percentage of correct (non-outlier) values
    non_outliers = []
    for column in dataset_num.columns:
        percentage = dataset_num[~IQR_masks[column]].shape[0]/dataset_num.shape[0]
        percentage = round(100*percentage,2)
        non_outliers.append(percentage)

    # Putting results into pandas DataFrame 
    result = pd.DataFrame({"Column": dataset_num.columns, "Non-outliers count [%]": non_outliers})   

    return result 

# A function that calculates z-score for numeric columns 
def z_score_calc(dataset):
    num_cols = []
    for column in dataset.columns:
        if is_numeric_dtype(dataset[column]):
            num_cols.append(column) 
    
    dataset_num = dataset.loc[:,num_cols] # dataset with only numerical columns 

    # Calculating z-scores 
    for column in dataset_num.columns:
        dataset_num[column] = (dataset_num[column] - dataset_num[column].mean())/dataset_num[column].std(ddof=0)

    return dataset_num 
