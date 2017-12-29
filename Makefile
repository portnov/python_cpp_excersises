all: greet.so map_test.o

greet.so: greet_binding.cpp
	g++ -O2 $< -I/usr/include/python3.6m -lboost_python-py36 -lpython3.6m -o greet.so -shared -fPIC

%.o: %.cpp
	g++ -Wall $<
