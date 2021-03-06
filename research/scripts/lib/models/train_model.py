from models.metrics import eval_metrics
from sklearn.linear_model import ElasticNet
import mlflow.sklearn
import os
import mlflow
from mlflow.sklearn import log_model
from mlflow.models.signature import infer_signature
from helpers.env import load_config
from sklearn.model_selection import train_test_split
import logging
import pandas as pd
import tempfile

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

CONFIG = load_config()
MODELS_PATH = CONFIG["FOLDERS"]["MODELS_DIR"]




def train_model(train_x,train_y,**kargs):
     lr = ElasticNet(**kargs)
     lr.fit(train_x, train_y)
     return lr



def data_split(data):
    train, test = train_test_split(data)
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]
    return train_x,test_x,train_y,test_y





def main_example():
    def get_data():
        csv_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
        try:
            df = pd.read_csv(csv_url, sep=";")
        except Exception as e:
            df = None
            logger.exception(
                "Unable to download training & test CSV, check your internet connection. Error: %s", e
            )
        return df
    data = get_data()
    train_x,test_x,train_y,test_y = data_split(data)
    model = train_model(train_x,train_y)
    preds = model.predict(test_x)
    test_x.insert(1,"preds",preds)
    metrics = eval_metrics(test_y,preds)
    mlrun = MlflowRun(model,metrics,x=train_x,y=train_y)
    mlrun.log_model_ml()
    add_dataframe_artifacts({"preds.csv":test_x})


def save_artifacts(dirname):
    def save_artifacts(func):
        def wrapper(*args,**kargs):
            with tempfile.TemporaryDirectory() as temp_dir:
                out_filedir = os.path.join(temp_dir, dirname)
                if not os.path.exists(out_filedir):
                    os.mkdir(out_filedir)
                _ = func(out_dir=out_filedir,*args,**kargs)
                mlflow.log_artifacts(out_filedir)
            return out_filedir
        return wrapper
    return save_artifacts



@save_artifacts("figures")
def add_plot_artifacts(figs_dic,out_dir=".",**kargs):
    for filename, fig in figs_dic.items():
        out_filedir = os.path.join(out_dir, filename)
        fig.savefig(out_filedir,**kargs)
    return True


@save_artifacts("outputs")
def add_dataframe_artifacts(df_dics,out_dir=".",**kargs):
    for filename, df in df_dics.items():
        out_filedir = os.path.join(out_dir, filename)
        df.to_csv(out_filedir,**kargs)
    return True


# def save_dict_in_files(func):
#     def wrapper(*args,**kargs):
#         for filename, art in figs_dic.items():
#             out_filedir = os.path.join(out_dir, filename)
#             func(out_filedir)
#         return True
#     return wrapper


class MlflowRun(object):
    def __init__(self, model,metrics,x,y,tags={},data_vers="vf"):
        self.model = model
        self.params = model.get_params()
        self.metrics = metrics
        self.tags = tags
        tags.update({"model_name": self.get_model_name(),
                                "data_vers":data_vers})
        self.run_uri = None
        self.signature = self.signature(x,y)
        self.artifacts = []
        self.experiment = "{0}-{1}".format(self.get_model_name(),data_vers)
        self.data_vers = data_vers
        self.temp_dir = tempfile.TemporaryDirectory()
        self.start()
    def add_artifacts(self,artifact):
        self.artifacts.append(artifact)
    def log_artifacts(self):
        for artifact in self.artifacts_funcs:
            mlflow.log_artifact(artifact)

    def get_model_name(self):
        return type(self.model).__name__
    def __repr__(self):
        return "{0}-{1}".format(self.get_model_name(),self.data_vers)

    def start(self):
        mlflow.set_tracking_uri("file:{0}/mlruns".format(MODELS_PATH))
        mlflow.set_experiment(self.experiment)
        self.run = mlflow.start_run()
    def end(self):
        temp_dir.cleanup()
        mlflow.end_run()

    def log_model_ml(self):
        mlflow.log_params(self.params)
        mlflow.log_metrics(self.metrics)
        mlflow.set_tags(self.tags)
        mlflow.log_metrics(self.metrics)
        log_model(self.model,"model",signature=self.signature)
        return self.run_uri
    def signature(self,x,y):
        return infer_signature(x,y)
