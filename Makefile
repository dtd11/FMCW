CC = gcc
CFLAGS = -Wall -Wextra -O2 -std=c99
LDFLAGS = -lfftw3 -lm

TARGET = test
SRCS = test.c

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRCS) $(LDFLAGS)

clean:
	rm -f $(TARGET)
