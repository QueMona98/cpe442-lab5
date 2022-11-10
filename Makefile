#compiler and linker options
CC = g++ 
LDFLAGS = $(shell pkg-config --libs opencv4)
CPPFLAGS = -Werror `pkg-config --cflags opencv4` -O0 -lpthread -g
SOURCES = $(wildcard *.cpp)
INCLUDES = $(wildcard *.hpp)
OBJECTS = $(SOURCES:.cpp=.o)
TARGET=output

all: $(TARGET)

$(TARGET):$(OBJECTS)
		$(CC) -o $@ $(CPPFLAGS) $^ $(LDFLAGS)

%.o0 : %.c $.cpp
		$(CC) -c $(CPPFLAGS) $<

clean:
		rm -f $(TARGET) $(OBJECTS)