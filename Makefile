CXX        := g++
CXXFLAGS   := -Wall -Wextra -O2 -std=c++17 -Wno-unused-variable -Wno-unused-parameter

INCLUDES   := -I/home/dtd11/FMCW

LIBS := -lfftw3f -lfftw3 -lmatio -lboost_iostreams

test: fmcw.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) fmcw.cpp -o test $(LIBS)
	
clean:
	rm -f test
