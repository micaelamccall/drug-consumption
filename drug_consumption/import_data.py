import os
import sys
import pandas as pd
from urllib.request import urlretrieve
from settings import *

DATA_PATH = os.path.join(PROJ_ROOT_DIR, "data")
DATA_DOWNLOAD_ROOT = "https://archive.ics.uci.edu/ml/machine-learning-databases/00373/"
DATA_DOWNLOAD_URL = DATA_DOWNLOAD_ROOT + "drug_consumption.data"


def import_drug_data(data_url=DATA_DOWNLOAD_URL, data_path=DATA_PATH):
    if not os.path.isdir(data_path):
        os.makedirs(data_path)
    
    data_file = os.path.join(data_path, "drug_consumption.data")

    urlretrieve(data_url, data_file)


def load_drug_data():
    header_list=['ID','Age','Gender','Education','Country', 'Ethnicity', 'Nscore','Escore','Oscore','Ascore','Cscore','ImpulsiveScore','SS','Alcohol','Amphet','Amyl','Benzos','Caff','Cannabis','Choc','Coke','Crack','Ecstacy','Heroin','Ketamine','LegalH','LSD','Meth','Mushroom','Nicotine','Semeron','VSA']
    drug_df = pd.read_csv(os.path.join(DATA_PATH, "drug_consumption.data"), header=None, names=header_list)
    return drug_df





make_proj_module()
import_drug_data()
drug_df = load_drug_data()
