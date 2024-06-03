import importlib
import json


def get_module_name(file_path):
    return file_path.split("/")[-1].split(".")[0]


def module_from_file(file_path):
    module_name = get_module_name(file_path)
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_function_from_file(file_path, function_name):
    module = module_from_file(file_path)
    function = getattr(module, function_name)
    return function


def try_get_function_from_file(path, name):
    if path is None or name is None:
        return None
    else:
        return get_function_from_file(file_path=path, function_name=name)


def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except:
        return False
