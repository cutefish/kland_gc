ROOT = $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

DEBUG = -g
CFLAGS = $(DEBUG) -Wall -O3 
LIBS = -lpthread -lm -lrt

AR = ar
RANLIB = ranlib
LDFLAGS = 
CC = gcc
CXX = g++
NVCC = nvcc

KLAND_GC = klgc
LIB_AUX = lib$(KLAND_GC)aux

LIB_DIR = $(ROOT)/lib
INCLUDE_DIR = $(ROOT)/include
BIN_DIR = $(ROOT)/bin
KLGC_DIR = $(ROOT)/user

LINKAGE = static
ifeq ($(LINKAGE),static)
LIB_TARGET = $(LIB_AUX).a
LIB_DEP = $(ROOT)/$(BIN_DIR)/$(LIB_TARGET)
endif
ifeq ($(LINKAGE),dynamic)
LIB_TARGET = $(LIB_AUX).so
LIB_DEP = $(ROOT)/$(BIN_DIR)/$(LIB_TARGET)
endif
