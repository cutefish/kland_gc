ROOT = $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

LIB_DIR = $(ROOT)/lib
INCLUDE_DIR = $(ROOT)/include
BIN_DIR = $(ROOT)/bin

KLAND_GC = $(BIN_DIR)/kland_gc

