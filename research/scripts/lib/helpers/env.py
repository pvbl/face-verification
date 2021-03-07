from dotenv import load_dotenv,find_dotenv
import configparser
from os import environ, path
from sys import version_info
import json
import jsonschema


def get_env_vars(dir_env=None, **kargs):
    """
    Return env vars found in .env file
    """
    if not dir_env:
        dir_env = find_dotenv()
    return load_dotenv(dotenv_path=dir_env,**kargs)

def find_working_directory_from_env():
    """
    Returns the folder where the .env is located, usually in working directory
    """
    return find_dotenv().rsplit("/",1)[0]

def add_abs_path_config(cf,dir_path,sector="FOLDERS"):
    for key in cf[sector].keys():
        cf[sector][key] = path.join(dir_path,cf[sector][key])
    return cf


def load_config(file_path="config.ini"):
    cf = configparser.ConfigParser()
    dir_path = find_working_directory_from_env()
    file_path = path.join(dir_path,file_path)
    cf.read(file_path)
    sector = "FOLDERS"
    if sector in cf.keys():
        cf = add_abs_path_config(cf,dir_path,sector=sector)
    return cf

def return_python_version():
    PYTHON_VERSION = "{major}.{minor}.{micro}".format(major=version_info.major,
                                                  minor=version_info.minor,
                                                  micro=version_info.micro)
    return PYTHON_VERSION





class ReadConfigRunModel(object):
    def __init__(self,CONFIG,CONFIG_SCHEMA):
        self.CONFIGNAME = CONFIG
        self.CONFIG_SCHEMANAME = CONFIG_SCHEMA
        self.CONFIG = self.get_config(self.CONFIGNAME)
        self.CONFIG_SCHEMA = self.get_config(self.CONFIG_SCHEMANAME)

    def __repr__(self):
        return "{0}".format(self.CONFIGNAME)

    def get_config(self,config_path):
        with open(config_path) as config_file:
            conf = json.load(config_file)
        return conf

    def validate(self,json_ex):
        jsonschema.validate(json_ex, schema=self.CONFIG_SCHEMA)

    def validate_auto(self):
        self.validate(self.CONFIG)
    def return_config(self):
        self.validate_auto()
        return self.CONFIG
    def return_schema(self):
        return self.CONFIG_SCHEMA
