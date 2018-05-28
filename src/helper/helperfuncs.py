# this file will contain helper functions

import numpy as np


# this function will find the increment value of the string
def findIncr(string):
    period = string.rfind('.')
    number = []
    i = 1
    while string[period - i].isdigit():
        number.append(string[period - i])
        i += 1
    return int(''.join(number))