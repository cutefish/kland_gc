include Defines.mk

.PHONY: all clean kland_gc

all: kland_gc

kland_gc:
	@$(MAKE) -C $(LIB_DIR)

clean:
	-rm -f $(KLAND_GC)
	@$(MAKE) -C $(LIB_DIR) clean
