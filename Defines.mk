DEBUG = -g
CFLAGS = $(DEBUG) -Wall -O3 
LIBS = -lpthread -lm -lrt

AR = ar
RANLIB = ranlib
LDFLAGS = 
CC = gcc
CXX = g++

KLAND_GC = kland_gc
LIB_KLAND_GC = lib$(KLAND_GC)

LINKAGE = static
ifeq ($(LINKAGE),static)
TARGET = $(LIB_KLAND_GC).a
LIB_DEP = 
