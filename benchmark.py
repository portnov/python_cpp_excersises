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

def create_knots_numpy(pts):
    tmp = numpy.linalg.norm(pts[:-1] - pts[1:], axis=1)
    tknots = numpy.insert(tmp, 0, 0).cumsum()
    tknots = tknots / tknots[-1]
    return tknots

array = numpy.random.rand(100000, 3)

def test_python():
    print(create_knots_numpy(array))

def test_cpp():
    knots = greet.create_knots_euclidean(array)
    knots = numpy.insert(knots, 0, 0)
    print(knots)

sys.stdout.flush()
print("Testing C++", file=sys.stderr)
print(timeit.timeit("test_cpp()", setup = "from __main__ import test_cpp", number=1000), file=sys.stderr)

sys.stdout.flush()
print("Testing Python", file=sys.stderr)
print(timeit.timeit("test_python()", setup = "from __main__ import test_python", number=1000), file=sys.stderr)

