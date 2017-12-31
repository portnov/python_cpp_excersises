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

def topo_sort(verts, edges):
    WHITE = 0
    GRAY = 1
    BLACK = 2

    edges_map = dict()
    for v1, v2 in edges:
        if v1 in edges_map:
            edges_map[v1].add(v2)
        else:
            edges_map[v1] = set([v2])

    marks = dict()
    result = []

    def dfw(idx):
        color = marks.get(idx, WHITE)
        if color == GRAY:
            raise Exception("Cycle detected")
        elif color == WHITE:
            marks[idx] = GRAY
            #print("Visiting", idx)
            next_idxs = edges_map.get(idx, set())
            for next_idx in next_idxs:
                dfw(next_idx)
            marks[idx] = BLACK
            if color != BLACK:
                result.insert(0, verts[idx])

    for start_idx in range(len(verts)):
        #print("Starting from", start_idx)
        dfw(start_idx)

    return result

array = numpy.random.rand(100000, 3)

N = 1000000
verts = [(x,x,x) for x in range(N)]
edges = [(i,i+2) for i in range(N) if i+2 < N]
edges.append((N-3, 0))

def test_python():
    try:
        print(topo_sort(verts, edges))
    except Exception as e:
        print(e)

def test_cpp():
    result = greet.topo_sort(len(verts), edges)
    print([verts[i] for i in result])

sys.stdout.flush()
print("Testing C++", file=sys.stderr)
print(timeit.timeit("test_cpp()", setup = "from __main__ import test_cpp", number=10), file=sys.stderr)

sys.stdout.flush()
print("Testing Python", file=sys.stderr)
print(timeit.timeit("test_python()", setup = "from __main__ import test_python", number=10), file=sys.stderr)
# 
