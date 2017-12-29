#!/usr/bin/python3

import numpy

import greet

a = numpy.arange(1.0, 13.0, 1.0)
b = a.reshape(6, 2)
print("B:", b)

greet.say_greeting("me")

x = greet.multiply(b, 4.0)
print("X:", x)

greet.process(x)
