MFEM_DIR ?= ../..
MFEM_BUILD_DIR ?= ../..
CONFIG_MK = $(MFEM_BUILD_DIR)/config/config.mk

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

# Directory structure
LIB_DIR=./lib
SOURCE_DIR=src
INC_DIR=include
BUILD_DIR=build
APP_DIR=apps
DEPENDS=$(BUILD_DIR)/.depends

# dgpa shared library
LIB_NAME=dgpa
LIB=$(LIB_DIR)/lib$(LIB_NAME).a
LIB_SOURCES=$(wildcard $(SOURCE_DIR)/*.cpp)
LIB_OBJECTS=$(LIB_SOURCES:$(SOURCE_DIR)/%.cpp=$(BUILD_DIR)/$(LIB_DIR)/%.o)

# "apps" using the dgpa library
APP_SOURCES=$(wildcard $(APP_DIR)/*.cpp)
APP_OBJECTS=$(APP_SOURCES:$(APP_DIR)/%.cpp=$(BUILD_DIR)/$(APP_DIR)/%.o)
APPS=$(APP_SOURCES:%.cpp=%)

# Compiler configuration
CXXFLAGS= -std=c++11 ${MFEM_CXXFLAGS}
LFLGAS=${MFEM_LIBS} ${MFEM_EXT_LIBS}
INCFLAGS=-I$(SOURCE_DIR) -I${INC_DIR} ${MFEM_INCFLAGS}

.PHONY: all clean style
all: $(LIB) $(APPS)

# Build all the apps (the library is a dependency)
$(APPS): $(APP_DIR)/%: $(BUILD_DIR)/$(APP_DIR)/%.o $(LIB)
	$(MFEM_CXX) $(LFLGAS) -L$(LIB_DIR) -l$(LIB_NAME) -o $@ $<

$(BUILD_DIR)/$(APP_DIR)/%.o: makefile | $(BUILD_DIR)
	$(MFEM_CXX) -c $(CXXFLAGS) $(INCFLAGS) -o $@ $(APP_DIR)/$*.cpp

# Build the library
$(LIB): $(LIB_OBJECTS)
	$(AR) $(ARFLAGS) $(@) $(LIB_OBJECTS)

$(BUILD_DIR)/$(LIB_DIR)/%.o: makefile | $(BUILD_DIR)
	$(MFEM_CXX) -c $(CXXFLAGS) $(INCFLAGS) -o $@ $(SOURCE_DIR)/$*.cpp

# Use the compiler to determine dependencies on header files
# Some awk magic in the next target
# Prefix all lines matching the regexp /^.*\.o/ with build/ (to match e.g. "file.o:")
# and leave other lines alone
$(DEPENDS): $(APP_SOURCES) $(LIB_SOURCES) | $(BUILD_DIR)
	$(CXX) -std=c++11 $(INCFLAGS) -MM $(LIB_SOURCES) \
		| awk '/^.*\.o:/{ print "$(BUILD_DIR)/$(LIB_DIR)/" $$0; next } 1'  >$@
	$(CXX) -std=c++11 $(INCFLAGS) -MM $(APP_SOURCES) \
		| awk '/^.*\.o:/{ print "$(BUILD_DIR)/$(APP_DIR)/" $$0; next } 1'  >>$@

# Rebuild dependencies unless doing "make clean" or "make style"
ifneq ($(MAKECMDGOALS),clean)
ifneq ($(MAKECMDGOALS),style)
-include $(DEPENDS)
endif
endif

$(BUILD_DIR):
	mkdir -p $@/$(APP_DIR)
	mkdir -p $@/$(LIB_DIR)
	mkdir -p $(LIB_DIR)

clean:
	rm -rf $(APPS) $(LIB) $(BUILD_DIR)

FORMAT_FILES = $(foreach dir,$(SOURCE_DIR) $(INC_DIR) $(APP_DIR),"$(dir)/*.?pp")
ASTYLE = astyle --options=$(MFEM_DIR)/config/mfem.astylerc
style:
	@if ! $(ASTYLE) $(FORMAT_FILES) | grep Formatted; then\
	   echo "No source files were changed.";\
	fi
