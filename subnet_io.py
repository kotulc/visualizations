"""
Input and output operations and related helper functions
"""


import gzip
import json
import numpy
import os.path
import pickle


# Helper data path constants
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
SAVE_PATH = DIR_PATH + '\\data\\'


class DataDict(object):
    """A single dictionary for single and multi-indexed Pandas DataFrames"""
    def __init__(self, data_dir, buffer_size=None):
        os.makedirs(os.path.dirname(data_dir), exist_ok=True)
        self.data_dir = data_dir
        self.data_dict = {}
        self.buffer_idx = 0
        self.buffer_size = buffer_size

    def __len__(self):
        return self.buffer_idx

    def append(self, data_dict, dict_key, dict_value):
        """Append dict_value to the list at dict_key"""
        if dict_key not in data_dict:
            data_dict[dict_key] = []
        data_dict[dict_key].append(dict_value)
        if self.buffer_size is not None:
            if len(data_dict[dict_key]) > self.buffer_size:
                data_dict[dict_key].pop(0)
        if len(data_dict[dict_key]) > self.buffer_idx:
            self.buffer_idx = len(data_dict[dict_key])

    def append_dict(self, value_dict, data_dict=None):
        """Append each key-value pair from value_dict to data_dict"""
        if data_dict is None:
            data_dict = self.data_dict
        for k, v in value_dict.items():
            if isinstance(v, dict):
                if k not in data_dict:
                    data_dict[k] = {}
                self.append_dict(v, data_dict[k])
            else:
                self.append(data_dict, k, v)

    def rows_to_columns(self, data_dict=None):
        """Convert all 1-dimensional numpy arrays to column-wise format"""
        if data_dict is None:
            data_dict = self.data_dict
        for k, v in data_dict.items():
            if isinstance(v, list) and len(v) > 0:
                if isinstance(v[0], numpy.ndarray) and len(v[0].shape) == 1:
                    v = [a.reshape(1, a.shape[0]) for a in v]
                    v = numpy.concatenate(v, axis=0)
                    data_dict[k] = [v[:, i] for i in range(v.shape[1])]
            elif isinstance(v, dict):
                self.rows_to_columns(data_dict=v)

    def rows_to_matrix(self, data_dict=None):
        """Convert lists of 1-dimensional numpy arrays to matrices"""
        if data_dict is None:
            data_dict = self.data_dict
        for k, v in data_dict.items():
            if isinstance(v, list) and len(v) > 0:
                if isinstance(v[0], numpy.ndarray):
                    v = [a.reshape(1, *a.shape) for a in v]
                    data_dict[k] = numpy.concatenate(v, axis=0)
                elif isinstance(v[0], int) or isinstance(v[0], float):
                    data_dict[k] = numpy.asarray(v)
            elif isinstance(v, dict):
                self.rows_to_matrix(data_dict=v)

    def to_disk(self, file_name, pickle_file=True):
        if pickle_file:
            save_pickle(self.data_dir + file_name, self.data_dict)
        else:
            save_json(self.data_dir + file_name, self.data_dict)


class DataStore(object):
    """A collection of related data dicts in a nested directory"""
    def __init__(self, store_dir):
        os.makedirs(os.path.dirname(store_dir), exist_ok=True)
        self.store_dir = store_dir
        self.data_collections = {}
        self.data_dicts = {}

    def __getitem__(self, dict_id):
        return self.data_dict(dict_id)

    def append(self, data_dict, dict_id):
        """Append all key value pairs from data_dict to the store dict"""
        store_dict = self.data_dict(dict_id)
        for k, v in data_dict.items():
            if k not in store_dict:  # Initialize new list
                store_dict[k] = [v]
            else:  # Existing key, append to value list
                store_dict[k].append(v)

    def collection(self, c_id):
        """Return a child DataStore object at the sub directory c_id"""
        if c_id not in self.data_collections:
            store_path = self.store_dir + c_id + '\\'
            self.data_collections[c_id] = DataStore(store_path)
        return self.data_collections[c_id]

    def data_dict(self, dict_id):
        """Return a dict object associated with dict_id"""
        if dict_id not in self.data_dicts:
            self.data_dicts[dict_id] = {}
        return self.data_dicts[dict_id]

    def save_meta(self, meta_obj):
        """Save a meta descriptor object to the root directory"""
        save_json(self.store_dir + 'meta.json', meta_obj)

    def to_disk(self):
        """Save all dictionaries as DataFrame objects"""
        for ds in self.data_collections.values():
            ds.to_disk()
        for dict_id, dict_obj in self.data_dicts.items():
            df_path = self.store_dir + dict_id + '.pkl'
            pd.DataFrame(dict_obj).to_pickle(df_path)


def load_or_build(build_fcn, build_args, local_path, report=True):
    """Return file from file_path if found, else build with build_fcn"""
    file_path = SAVE_PATH + local_path
    if report:
        print(build_fcn.__name__.upper() + ': loading...', end='', flush=True)
    if file_path.endswith('.json'):
        file_obj = load_json(file_path)
        if file_obj is None:
            file_obj = build_fcn(*build_args)
            save_json(file_path, file_obj)
    else:  # Assume pickle file has been compressed
        file_obj = load_pickle(file_path, True)
        if file_obj is None:
            file_obj = build_fcn(*build_args)
            save_pickle(file_path, file_obj, True)
    if report:
        print('\r' + build_fcn.__name__.upper() + ': operation complete.')
    return file_obj


def load_json(file_str):
    if os.path.isfile(file_str):
        with open(file_str, 'r') as file_obj:
            json_obj = json.load(file_obj)
        return json_obj
    return None


def load_pickle(file_str, compressed=True):
    if os.path.isfile(file_str):
        if compressed:
            with gzip.open(file_str, 'rb') as file_obj:
                pkl_obj = pickle.load(file_obj)
        else:
            with open(file_str, 'rb') as file_obj:
                pkl_obj = pickle.load(file_obj)
        return pkl_obj
    return None


def load_text(file_str):
    if os.path.isfile(file_str):
        with open(file_str, 'r') as file_obj:
            lines = file_obj.readlines()
        return lines
    return None


def load_vectors(vect_path):
    if os.path.isfile(vect_path):
        vect_dict = {}
        with open(vect_path, 'r', encoding="utf8") as file_obj:
            for line in file_obj:
                split_line = line.split()
                vect_key = split_line[0]
                vect_value = numpy.array([float(val) for val in split_line[1:]])
                vect_dict[vect_key] = vect_value
        return vect_dict
    else:
        raise FileNotFoundError


def save_json(file_str, data_obj):
    os.makedirs(os.path.dirname(file_str), exist_ok=True)
    with open(file_str, 'w') as file_obj:
        json.dump(data_obj, file_obj)


def save_pickle(file_str, data_obj, compressed=True):
    os.makedirs(os.path.dirname(file_str), exist_ok=True)
    if compressed:
        with gzip.open(file_str, 'wb') as file_obj:
            pickle.dump(data_obj, file_obj, protocol=-1)
    else:
        with open(file_str, 'wb') as file_obj:
            pickle.dump(data_obj, file_obj, protocol=-1)