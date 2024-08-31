''' File storing functions for basic data processing. Processed data is visualized via Vizro dashboard''' 
import pandas as pd


# A function that returns the head (first 5 rows) of a dataset 
def dataset_head(dataset):
    return dataset.head()

# A function that returns basic information about a dataset 
def dataset_info(dataset):
    dict_to_convert = {}
    dict_to_convert["Column"] = dataset.columns 
    dict_to_convert["Non-null Count"] = len(dataset)-dataset.isnull().sum().values

    dtypes = dataset.dtypes.values
    dtypes_str = [str(element) for element in dtypes]
    dict_to_convert["Dtype"] = dtypes_str

    return pd.DataFrame(dict_to_convert) 

# A function that returns the values ​​of quantities describing the numerical data of a dataset 
def dataset_describe(dataset):
    df =  dataset.describe().transpose()
    df["Column"] = df.index 

    columns = df.columns.to_list()
    columns = columns[-1:] + columns[:-1]
    df = df[columns] 
    return df 
