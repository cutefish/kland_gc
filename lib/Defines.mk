include ../Defines.mk

DEBUG =
CFLAGS = $(DEBUG) -O3 -I$(INCLUDE_DIR)
CUINCFLAGS = -I$(CUDAINCPATH) -I$(INCLUDE_DIR)
CULDFLAGS = -fPIC -lcudart -lcuda -L$(CUDALIBPATH)
CCLDFLAGS = -lm -lrt -lmpi_cxx

AR = ar
RANLIB = ranlib
LDFLAGS = 
CC = mpicc
CXX = mpic++
NVCC = nvcc

