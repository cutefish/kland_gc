include Defines.mk

.PHONY: default all lib clean

default: all

all: $(LIB_TARGET) $(KLAND_GC)

lib: $(LIB_TARGET)

$(LIB_TARGET):
	@$(MAKE) -C $(LIB_DIR)

$(KLAND_GC): $(LIB_TARGET)
	@$(MAKE) -C $(KLGC_DIR)

clean:
	-rm -f $(BIN_DIR)/$(LIB_TARGET)
	-rm -f $(BIN_DIR)/$(KLAND_GC)
