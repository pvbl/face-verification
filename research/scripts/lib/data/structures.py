import pandas as pd
from helpers.env import load_config
import os

CONFIG = load_config()
DATA_RAW_DIR = CONFIG["FOLDERS"]["DATA_RAW_DIR"]
DATA_INTERM_DIR = CONFIG["FOLDERS"]["DATA_INTERM_DIR"]
DATA_EXT_DIR = CONFIG["FOLDERS"]["DATA_EXT_DIR"]
DATA_PROCESSED_DIR = CONFIG["FOLDERS"]["DATA_PROCESSED_DIR"]



class FileProcess(object):
    def __init__(self,filename:str,step:str,datapath:str):
        self.filename = filename
        self.step = step
        self.datapath = os.path.join(datapath,filename)
    def __repr__(self):
        return "{0}".format(self.filename)
    def load(**kargs)->pd.DataFrame:
        return pd.read_csv(self.datapath,**kargs)
    def visualize(self):
        pass
    def clean(self):
        pass
    def preprocess(self):
        pass
    def save(self,output_filename,step="int",**kargs):
        if step=="int":
            output_filepath = os.path.join(DATA_INTERM_DIR,output_filename)
        elif step=="proc":
            output_filepath = os.path.join(DATA_PROCESSED_DIR,output_filename)
        pd.to_csv(output_filepath,**kargs)
        return 0
    def summary(self):
        pass


class RawProcess(FileProcess):
    def __init__(self,filename,step):
        super().__Init(filename,step,DATA_RAW_DIR)



class IntProcess(FileProcess):
    def __init__(self,filename,step):
        super().__Init(filename,step,DATA_INTERM_DIR)



class DataFlowProcess(object):
    def __init__(self,filename,step="raw"):
        assert step in ["raw","int","ext","proc"]
        self.file = filename
        self.step = step
    def __repr__(self):
        return "file {0} from step {1}".format(self.file,self.step)
    def __str__(self):
        return "{0}-{1}".format(self.file,self.step)
    def raw_to_int(self):
        pass
    def int_to_processed(self):
        pass
    def get_external(self):
        pass
    def get_raw(self):
        pass
    def raw_to_processed(self):
        interm_filename = self.process_raw()
        output = self.process_intermediate()
