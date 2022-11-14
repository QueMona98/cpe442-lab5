CC = g++
OUT = sobel.out 
LDFLAGS = -g
CFLAGS = -Werror -mfpu=neon -std=c++11 -O0 -lpthread
SRCS = ${wildcard*.cpp}

all: default

default:
    ${CC} ${CFLAGS} ${LDFLAGS} `pkg-config --flags 
    --libs opencv4` ${SRCS} - o ${OUT}

clean:
    rm -rf ${OUT} ${OUT}.dSYM

