CXX = g++
CXXFLAGS = -Wall -Wextra -O2 -std=c++11
LDFLAGS = -lfftw3 -lm

TARGET = test
SRCS = test.cpp

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRCS) $(LDFLAGS)

clean:
	rm -f $(TARGET)
