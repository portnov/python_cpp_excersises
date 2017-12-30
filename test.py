#!/usr/bin/python3

import sys
import numpy

import greet

array = numpy.random.rand(10, 3)

print(array)

#print(numpy.linalg.det(array))

result = greet.twice(array)
print(type(result))
print(result)

