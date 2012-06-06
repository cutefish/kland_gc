include Defines.mk

MAIN_MODULES = cuda user
OBJ_DIR = obj

MAIN_CCSRC = $(foreach sdir, $(MAIN_MODULES), $(wildcard $(sdir)/*.cc))
MAIN_CUSRC = $(foreach sdir, $(MAIN_MODULES), $(wildcard $(sdir)/*.cu))
MAIN_OBJCC = $(patsubst %.cc, $(OBJ_DIR)/%.cco, $(notdir $(MAIN_CCSRC)))
MAIN_OBJCU = $(patsubst %.cu, $(OBJ_DIR)/%.cuo, $(notdir $(MAIN_CUSRC)))
MAIN_OBJ = $(MAIN_OBJCC) $(MAIN_OBJCU)

vpath %.cc $(MAIN_MODULES)
vpath %.cu $(MAIN_MODULES)

.PHONY: all main dispSac clean

all: main dispSac

main: $(MAIN_OBJ)
	$(MPICXX) $(CULDFLAGS) $(CCLDFLAGS) $^ -o $(KLAND_GC)

dispSac: check/dispSac.cc $(MAIN_OBJ)
	$(MPICXX) $(CFLAGS) check/dispSac.cc obj/sacio.cco -o $(BIN_DIR)/dispSac 

$(OBJ_DIR):
	@mkdir -p $(OBJ_DIR)

$(OBJ_DIR)/%.cco: %.cc | $(OBJ_DIR)
	$(MPICXX) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/%.cuo: %.cu | $(OBJ_DIR)
	$(NVCC) $(CUINCFLAGS) -c $< -o $@

clean:
	-rm -rf $(OBJ_DIR)/*
