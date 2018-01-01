#!/usr/bin/python3

import sys
import numpy

import greet

edges = [(0,2), (2,3), (1,4), (4,5), (3,6)]

result = greet.get_components(7, edges)
print(type(result))
print(result)

