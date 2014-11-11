all: main

%: %.cpp
	g++ -std=c++11 `pkg-config --cflags --libs opencv` -Wall -O2 -funroll-loops -march=native -o $@ $<

