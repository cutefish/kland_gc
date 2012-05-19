include ../Defines.mk

.PHONY: all objects

all: $(LIB_TARGET)

$(LIB_TARGET) : objects
	$(AR) cr $@ $(wildcard $(BIN_DIR)/*.o)
	$(RANLIB) $@
	mv $(LIB_TARGET) $(BIN_DIR)/$(LIB_TARGET)
	-rm -f $(BIN_DIR)/*.o

objects : 
	@mkdir -p $(BIN_DIR)
	@$(MAKE) -C support

