''' A file for creating a vizro dashboard. '''
import vizro.plotly.express as px
from vizro import Vizro
import vizro.models as vm 
from vizro.tables import dash_ag_grid 
from vizro.models.types import capture 
from sklearn.metrics import confusion_matrix 

from data.built_in_data import dataset1
from data_processing.basic import dataset_info
from data_processing.basic import dataset_describe
from data_processing.custom import IQR_count 
from dec_tree_op import result

# Loading data and performing selected operations on the data
dataset = dataset1 
data_info = dataset_info(dataset)
data_desc = dataset_describe(dataset)
data_iqr = IQR_count(dataset) 
model_result = result  

# Prepering of custom charts 
@capture('graph')
def conf_matrix_vis(data_frame, y_real, y_pred):
    real = data_frame.loc[:, y_real]
    pred = data_frame.loc[:, y_pred]
    cm = confusion_matrix(real, pred)
    fig = px.imshow(cm)
    return fig 

# Craeting a dashboard 
page1 = vm.Page(
    title = 'Dataset',
    components = [
        vm.AgGrid(figure=dash_ag_grid(data_frame=dataset)),
    ],
)

page2 = vm.Page(
    title = 'Basic dataset info',
    components = [
        vm.AgGrid(figure=dash_ag_grid(data_frame=data_info)),
    ],
)

page3 = vm.Page(
    title = 'Basic numeric data info',
    components = [
        vm.AgGrid(figure=dash_ag_grid(data_frame=data_desc)),
    ],
)

page4 = vm.Page(
    title = 'Outliers info',
    components = [
        vm.AgGrid(figure=dash_ag_grid(data_frame=data_iqr)),
    ],
)

page5 = vm.Page(
    title = 'Selected distributions',
    layout = vm.Layout(grid=[[0, 1]]),
    components = [
        vm.Graph(id="dist_alcohol", figure=px.histogram(dataset, x='alcohol')),
        vm.Graph(id="dist_malic", figure=px.histogram(dataset, x='malic_acid')),
    ],
    controls = [
        vm.Filter(column='class', targets=['dist_alcohol', 'dist_malic'], selector=vm.Dropdown()),
    ],
)

page6 = vm.Page(
    title = 'Model results',
    layout = vm.Layout(grid=[[0, 1]]),
    components = [
        vm.AgGrid(figure=dash_ag_grid(data_frame=model_result)),
        vm.Graph(id="conf_matrix", figure=conf_matrix_vis(y_real='class', y_pred='class_pred', data_frame=model_result)),
    ],
)

dashboard = vm.Dashboard(pages=[page1, page2, page3, page4, page5, page6]) 
Vizro().build(dashboard).run()
