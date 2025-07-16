MFEM_DIR = ../
CONFIG_MK = $(MFEM_DIR)/config/config.mk

include $(CONFIG_MK)

CXX = mpicxx
CPPFLAGS = -I$(MFEM_DIR) $(MFEM_TPLFLAGS) $(MFEM_PICFLAG)
CXXFLAGS = $(MFEM_CXXFLAGS)

LDFLAGS = $(MFEM_LDFLAGS)
LDLIBS = $(MFEM_LIBS)

#brute_force_test: brute_force_test.cpp test_utils.cpp #boundary_edge_dofs.cpp 
#	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(LDLIBS)

#clean:
#	rm -f brute_force_test

brute_force_test_internal_bdr: brute_force_test_internal_bdr.cpp test_utils.cpp #boundary_edge_dofs.cpp 
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(LDLIBS)

clean:
	rm -f brute_force_test_internal_bdr

.PHONY: clean