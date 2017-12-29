all: greet.so map_test.o

greet.so: greet_binding.cpp
	g++ -O2 -Wno-ignored-attributes $< -I/usr/include/python3.6m -I/usr/include/eigen3 -lboost_python-py36 -lpython3.6m -o greet.so -shared -fPIC

%.o: %.cpp
	g++ -Wall $<
