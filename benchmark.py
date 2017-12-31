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

##################
# Topological sort algorithm by R.Tarjan (see wikipedia)
# This works correctly for all graphs, and it is faster
# than the algorithm in topo_sort_for_profile.py.
# But this implementation is recursive, which actually 
# means that it is restricted with graphs of < 10000 vertices.
# To work correctly with larger graphs, it should be re-written
# without recursion.

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

#########
# Old topo_sort algorithm from topo_sort_for_profile.py
# It is not recursive, which is good;
# But it is:
# 1) very slow;
# 2) does not work properly for all graphs
#

def dodo(verts, edges, verts_o,k):
    for i in edges:
        if k in i:
            # this is awesome !!
            k = i[int(not i.index(k))]
            verts_o.append(verts[k])
            return k, i
    return False, False

def topo_sort_old(verts, edges):
    vout = []
    eout = []
    ed = 1
    edges_o = []
    verts_o = []
    k = 0
    while True:
        k, ed = dodo(verts, edges, verts_o,k)
        if ed:
            edges.remove(ed)
        if not ed:
            break
    edges_o = [[k,k+1] for k in range(len(verts_o)-1)]
    edges_o.append([0, len(verts_o)-1])
    eout.append(edges_o)
    vout.append(verts_o)
    return vout

array = numpy.random.rand(100000, 3)

#N = 10
N = 10000
verts = [(x,x,x) for x in range(N)]
edges = [(i,i+2) for i in range(N) if i+2 < N]
edges.append((N-3, 0))

def test_python():
    try:
        print(topo_sort_old(verts, edges))
    except Exception as e:
        print(e, file=sys.stderr)

def test_cpp():
    # Topological sort implemented in C++
    result = greet.topo_sort(len(verts), edges)
    # Result is just list of vertex indicies.
    print([verts[i] for i in reversed(result)])

def do_test():
    print(edges)
    test_python()
    test_cpp()

def do_benchmark():
    sys.stdout.flush()
    print("Testing C++", file=sys.stderr)
    print(timeit.timeit("test_cpp()", setup = "from __main__ import test_cpp", number=10), file=sys.stderr)

    sys.stdout.flush()
    print("Testing Python", file=sys.stderr)
    print(timeit.timeit("test_python()", setup = "from __main__ import test_python", number=10), file=sys.stderr)

#do_test()
do_benchmark()

