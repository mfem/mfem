# Use the MFEM build directory
MFEM_DIR ?= ../..
MFEM_BUILD_DIR ?= ../..
SRC = ./

CONFIG_MK = $(MFEM_BUILD_DIR)/config/config.mk

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

# Remove built-in rule
#%: %.cpp

exQPContactBlockTL: exQPContactBlockTL.o problems.o IPsolver.o $(MFEM_LIB_FILE)
	$(MFEM_CXX) $(MFEM_FLAGS) exQPContactBlockTL.o problems.o IPsolver.o -o $@ $(MFEM_LIBS)



exQPContactBlockTL.o: exQPContactBlockTL.cpp $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) -c $<

problems.o: problems.cpp $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) -c $< 

IPsolver.o: IPsolver.cpp $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) -c $<

# Generate an error message if the MFEM library is not built and exit
$(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

.PHONY: clean
clean:
	rm -f *.o exQPContactBlockTL


