#!/usr/bin/python3

import sys
import numpy
import timeit

import greet

def process(data):
    #vs_set = set()
    u_values = dict()

    for u, v in list(data):
        if v in u_values:
            u_values[v].add(u)
        else:
            u_values[v] = set([u])

    print("Size: {}".format(len(u_values)))
    #for key, values in u_values.items():
    #    for value in values:
    #        pass
            #print("Key: {} => {}".format(key, value))

array = numpy.random.rand(100000, 2)

def test_python():
    process(array)

def test_cpp():
    greet.process(array)

sys.stdout.flush()
print("Testing Python", file=sys.stderr)
print(timeit.timeit("test_python()", setup = "from __main__ import test_python", number=10), file=sys.stderr)

sys.stdout.flush()
print("Testing C++", file=sys.stderr)
print(timeit.timeit("test_cpp()", setup = "from __main__ import test_cpp", number=10), file=sys.stderr)

