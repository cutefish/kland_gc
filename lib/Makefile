include Defines.mk

MODULES = cuda user
OBJ_DIR = obj

CCSRC = $(foreach sdir, $(MODULES), $(wildcard $(sdir)/*.cc))
CUSRC = $(foreach sdir, $(MODULES), $(wildcard $(sdir)/*.cu))
OBJCC = $(patsubst %.cc, $(OBJ_DIR)/%.cco, $(notdir $(CCSRC)))
OBJCU = $(patsubst %.cu, $(OBJ_DIR)/%.cuo, $(notdir $(CUSRC)))
OBJ = $(OBJCC) $(OBJCU)

vpath %.cc $(MODULES)
vpath %.cu $(MODULES)

.PHONY: all goal clean

all: goal

goal: $(OBJ)
	$(CC) $(CULDFLAGS) $(CCLDFLAGS) $^ -o $(KLAND_GC)

$(OBJ_DIR):
	@mkdir -p $(OBJ_DIR)

$(OBJ_DIR)/%.cco: %.cc | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/%.cuo: %.cu | $(OBJ_DIR)
	$(NVCC) $(CUINCFLAGS) -c $< -o $@

clean:
	-rm -rf $(OBJ_DIR)/*
