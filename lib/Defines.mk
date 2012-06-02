include ../Defines.mk

DEBUG =
CFLAGS = $(DEBUG) -O3 -I$(INCLUDE_DIR)
CUINCFLAGS = -I/sw/keeneland/cuda/4.1/linux_binary/include -I$(INCLUDE_DIR)
CULDFLAGS = -fPIC -lcudart -lcuda -L/sw/keeneland/cuda/4.1/linux_binary/lib64
CCLDFLAGS = -lm -lrt

AR = ar
RANLIB = ranlib
LDFLAGS = 
CC = mpicc
CXX = mpic++
NVCC = nvcc

