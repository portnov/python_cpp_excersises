#!/usr/bin/python3

import sys
import numpy
import timeit
import random
import collections

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

###################
# SeparateMesk from mesh_separate_mk2.py

def get_components(ve, pe):
    verts_out = []
    poly_edge_out = []
    vert_index = []
    poly_edge_index = []

    # build links
    node_links = collections.defaultdict(set)
    for edge_face in pe:
        for i in edge_face:
            node_links[i].update(edge_face)

    nodes = set(node_links.keys())
    n = nodes.pop()
    node_set_list = [set([n])]
    node_stack = collections.deque()
    node_stack_append = node_stack.append
    node_stack_pop = node_stack.pop
    node_set = node_set_list[-1]
    # find separate sets
    while nodes:
        for node in node_links[n]:
            if node not in node_set:
                node_stack_append(node)
        if not node_stack:  # new mesh part
            n = nodes.pop()
            node_set_list.append(set([n]))
            node_set = node_set_list[-1]
        else:
            while node_stack and n in node_set:
                n = node_stack_pop()
            nodes.discard(n)
            node_set.add(n)
    # create new meshes from sets, new_pe is the slow line.
    if len(node_set_list) > 1:
        node_set_list.sort(key=lambda x: min(x))
        for idx, node_set in enumerate(node_set_list):
            mesh_index = sorted(node_set)
            vert_dict = {j: i for i, j in enumerate(mesh_index)}
            new_vert = [ve[i] for i in mesh_index]
            new_pe = [[vert_dict[n] for n in fe]
                      for fe in pe
                      if fe[0] in node_set]

            verts_out.append(new_vert)
            poly_edge_out.append(new_pe)
            vert_index.append([idx for i in range(len(new_vert))])
            poly_edge_index.append([idx for face in new_pe])
    elif node_set_list:  # no reprocessing needed
        verts_out.append(ve)
        poly_edge_out.append(pe)
        vert_index.append([0 for i in range(len(ve))])
        poly_edge_index.append([0 for face in pe])

    return verts_out, poly_edge_out

array = numpy.random.rand(100000, 3)

N = 2*10000
M = 2*5000
# N = 100000
verts = [(x,x,x) for x in range(N)]
# edges = [(i,i+2) for i in range(N) if i+2 < N]
# edges.append((N-3, 0))

edges = [(random.randint(0,N-1), random.randint(0,N-1)) for i in range(M)]
edges = []
for i in range(M):
    u = random.randint(0, N-1)
    v = random.randint(0, N-1)
    while (v == u):
        v = random.randint(0, N-1)
    edges.append((u,v))

def test_python():
    try:
        #print(topo_sort(verts, edges))
        #print(topo_sort_old(verts, edges))
        vs, es = get_components(verts, edges)
        print("{} components:".format(len(vs)))
        for v in vs:
            print(v)
    except Exception as e:
        print(e, file=sys.stderr)

def test_topo_cpp():
    # Topological sort implemented in C++
    result = greet.topo_sort(len(verts), edges)
    # Result is just list of vertex indicies.
    print([verts[i] for i in reversed(result)])

def test_cpp():
    result = greet.get_components(len(verts), edges)
    print("{} components:".format(len(result)))
    for vert_indicies, es in result:
        vs = [verts[i] for i in vert_indicies]
        print(vs, " *** ", es)

def do_test():
    print(verts)
    print(edges)
    test_python()
    test_cpp()

def do_benchmark():
    NUMBER = 10

    sys.stdout.flush()
    print("Testing C++", file=sys.stderr)
    print(timeit.timeit("test_cpp()", setup = "from __main__ import test_cpp", number=NUMBER), file=sys.stderr)

    sys.stdout.flush()
    print("Testing Python", file=sys.stderr)
    print(timeit.timeit("test_python()", setup = "from __main__ import test_python", number=NUMBER), file=sys.stderr)

#do_test()
do_benchmark()

