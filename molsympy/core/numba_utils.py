from numba.typed import List

def to_typed_list(py_list):
    typed = List()
    for arr in py_list:
        typed.append(arr)
    return typed
