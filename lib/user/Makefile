include $(ROOT)/Defines.mk

.PHONY: all

objects := $(patsubst %.cc, $(BIN_DIR)/%.o, $(wildcard *.cc))

all : $(objects)

$(BIN_DIR)/%.o : %.cc
		$(CC) $(CFLAGS) -c $< -o $@ -I$(INCLUDE_DIR)
