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

def clean_and_print_data(data: list[dict]) -> tuple[list[str], list[float], list[float], list]:
    """Cleans and prints data loaded into a list of dictionaries from a CSV file."""
    labels, x1, x2, y = [], [], [], []

    for row in data:
        labels.append(row['passenger'])
        x1.append(float(row['price']))
        x2.append(float(row['cabin']))
        y_val = (row['survived'])
        if y_val:
            y.append(int(y_val))
        else:
            y.append(y_val)

    print('\nAfter loading the data into lists we have:\r')
    print(f'labels = {labels}')
    print(f'x1 = {x1}')
    print(f'x2 = {x2}')
    print(f'y: classification result  = {y}')

    return labels, x1, x2, y


def find_gamma_m(x:np.array, xm: np.array, c=1)-> float:
    """find gamma_m, an inverse similarity measure between x and xm"""
    diff = x - xm
    gamma_m = 1 - (np.dot(diff, diff)/c)
    return gamma_m

def normalise(x1:np.array, x2:np.array) -> tuple[np.array, np.array]:
    """Normalise two feature vectors x1 and x2"""
    if len(x1) != len(x2):
        raise Exception(f'{len(x1)=} but {len(x2)=}')
    for i in range(len(x1)):
        l2_norm = math.sqrt(x1[i]**2 + x2[i]**2)
        x1[i] /= l2_norm
        x2[i] /= l2_norm
    return x1, x2

def find_test_data(x1: list[float], x2: list[float], y: list) -> np.array:
    """Find the test data point where y is empty string"""
    for i, items in enumerate(y):
        if items == '':
            x = np.array([x1[i],x2[i]])
    if x.shape != (2,):
        raise Exception(f'x,shape should be (2,), is {x.shape}')
    print(f'The test point is {x}')
    return x

def find_norm(alpha: list[float]) -> list[float]:
    """find the norm a a"""
    alpha_norm = np.sqrt(np.dot(alpha, alpha))
    return alpha_norm

def pre_process_feature_vector(x1: list[float], x2: list[float], y:list) -> tuple[list[float], list[float], list[int]]:
    """Add an extra copy of the features of Passsenger 3, and tidy up y to be integers"""
    x1.append(x1[2])
    x2.append(x2[2])
    y = [y[v%2] for v in range(4)]
    print('\nAfter pre-processing feature vector:')
    print('Added extra copy of Passenger 3 and tidy up y to be integers')
    print(f'x1={[f'{v:.3f}' for v in x1]}')
    print(f'x2={[f'{v:.3f}' for v in x2]}')
    print(f'{y=} \r')
    return x1, x2, y

def prepare_quantum_feature_vector (x1:list[float], x2:list[float], y: list[float]) -> list[float]:
    """Prepare quantum feature vector by extending each 2D data point based on class label y"""
    alpha = []
    if len(x1) != len(x2):
        raise Exception(f'{len(x1)=} but {len(x2)=}')
    for i in range(len(x1)):
        # extend state to forth qubit
        if y[i] == 0:
            alpha.append((x1[i]))
            alpha.append(0)
            alpha.append(x2[i])
            alpha.append(0)
        elif y[i] == 1:
            alpha.append(0)
            alpha.append(x1[i])
            alpha.append(0)
            alpha.append(x2[i])
        else:
            raise Exception(f'y should be 0 or 1, not {y[i]}')
    print('\nAfter preparing quantum feature vector:')
    print(f'alpha={[f'{v:.3f}' for v in alpha]}')
    return(alpha)

def normalise_feature_vector(alpha: list[float]) -> list[float]:
    """Normalise the quantum feature vector alpha expressed as a list"""
    alpha_norm = []
    norm = float(find_norm(alpha))
    print(f'Norm before normalisation = {norm}')
    for items in alpha:
        alpha_norm.append(float(items)/norm)
    print('\nAfter normalisation:')
    print(f'alpha_norm={[f'{v:.3f}' for v in alpha_norm]}')
    return alpha_norm