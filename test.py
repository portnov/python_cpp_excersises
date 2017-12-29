#!/usr/bin/python3

import sys
import numpy

import greet

array = numpy.random.rand(4,4)

print(array)

print(numpy.linalg.det(array))

greet.determinant(array)

