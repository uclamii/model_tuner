import pickle
import joblib
# import numpy as np


def dumpObjects(file, filename, use_pickle=True):
    """Method to dump objects
    https://pythontips.com/2013/08/02/what-is-pickle-in-python/
    """
    if use_pickle:
        fileObject = open(filename, "wb")

        # Set the desired Numpy C API version
        np.lib.format.MAGIC_PREFIX = (0xD, 0x3)

        # this writes the object a to the
        # file named 'testfile'
        pickle.dump(file, fileObject)

        # here we close the fileObject
        fileObject.close()
    else:
        # Save the model using Joblib
        joblib.dump(file, filename)

    print("Object saved!")


def loadObjects(filename, use_pickle=True):
    """Method to load objects"""
    if use_pickle:
        # we open the file for reading
        fileObject = open(filename, "rb")
        # load the object from the file into var b
        print("Object loaded!")

        # Set the desired Numpy C API version
        np.lib.format.MAGIC_PREFIX = (0xD, 0x3)

        return pickle.load(fileObject)
    else:
        # Load the model from the file
        fileObject = joblib.load(filename)
        print("Object loaded!")
        return fileObject
