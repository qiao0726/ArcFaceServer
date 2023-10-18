import json

def save_dict_to_json(data, filename):
    """
    Save a dictionary to a JSON file.

    Parameters:
    - data (dict): The dictionary to save.
    - filename (str): The name of the file to save to.
    """
    with open(filename, 'w') as f:
        json.dump(data, f)
        

def load_dict_from_json(filename):
    """
    Load a dictionary from a JSON file.

    Parameters:
    - filename (str): The name of the file to load from.

    Returns:
    - dict: The loaded dictionary.
    """
    with open(filename, 'r') as f:
        data = json.load(f)
    return data
