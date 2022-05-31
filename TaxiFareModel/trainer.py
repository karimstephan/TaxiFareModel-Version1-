from TaxiFareModel.data import get_data,  clean_data
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
import pandas as pd

class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        # self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        # create distance pipeline
        dist_pipe = Pipeline([
    ('dist_trans', DistanceTransformer()),
    ('stdscaler', StandardScaler())
])

    # create time pipeline
        time_pipe = Pipeline([
    ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
    ('ohe', OneHotEncoder(handle_unknown='ignore'))
])

    # create preprocessing pipeline
        preproc_pipe = ColumnTransformer([
        ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
        ('time', time_pipe, ['pickup_datetime'])
    ], remainder="drop")


    # Add the model of your choice to the pipeline
        self.pipeline = Pipeline([
    ('preproc', preproc_pipe),
    ('linear_model', LinearRegression())
])


    def run(self):
        """set and train the pipeline"""
        self.set_pipeline()
        self.pipeline.fit(self.X, self.y)

    def evaluate(self,a,b):
        """evaluates the pipeline on df_test and return the RMSE"""
        return self.pipeline.score(a,b)


if __name__ == "__main__":
    # get data
    # clean data
    # set X and y
    # hold out
    # train
    # evaluate
    print('TODO')
