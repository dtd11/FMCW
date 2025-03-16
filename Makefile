CXX = g++
CXXFLAGS = -Wall -Wextra -O2 -std=c++17
LIBS = -lfftw3 -lm -lboost_iostreams

all: test

test: fmcw.cpp
	$(CXX) $(CXXFLAGS) -o test fmcw.cpp $(INCLUDES) $(LIBS)

clean:
	rm -f test
