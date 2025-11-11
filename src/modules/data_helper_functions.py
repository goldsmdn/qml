#data_helper_functions
import csv
import numpy as np
import math

HOME_DIR = '..'
import sys
sys.path.append(HOME_DIR)

from config.config import ENCODING, C

def read_csv(file_path:str)->list[dict]:
    """read a CSV file and return as a list of dictionary items"""
    try:
        with open (file_path, encoding = ENCODING) as csvfile:
            reader = csv.DictReader(csvfile)
            data = [row for row in reader]
            return data

    except FileNotFoundError:
        print(f'Error: The file {file_path} was not found.')
    except PermissionError:
        print(f'Error: You do not have permission to read the file {file_path}.')
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def find_gamma_m(x:np.array, xm: np.array)-> float:
    diff = x - xm
    gamma_m = 1 - (np.dot(diff, diff)/C)
    return gamma_m

def normalise(x1:np.array, x2:np.array) -> tuple[np.array, np.array]:
    if len(x1) != len(x2):
        raise Exception(f'{len(x1)=} but {len(x2)=}')
    for i in range(len(x1)):
        l2_norm = math.sqrt(x1[i]**2 + x2[i]**2)
        x1[i] /= l2_norm
        x2[i] /= l2_norm
    return x1, x2
