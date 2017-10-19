#### Target ####
tgt := dbg

##### Paths ####
pwd = $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
obj = $(pwd)/obj

#### Object files ####
OBJ = $(obj)/$(tgt).o
OBJ += $(obj)/$(tgt).co
OBJ += $(obj)/dbgBackTrace.o
OBJ += $(obj)/dbgBackTraceData.o


#### Compilation variables ####
CXX = g++
CPU = $(shell echo $(shell getconf _NPROCESSORS_ONLN)*2|bc -l)
INC = -I.
FLG = -fPIC -Wall -g $(INC)
LIB = $(HOME)/lib/libbacktrace.so

#### Linking variables ####
AR      = ar
ARFLAGS = cruv
RANLIB  = ranlib

#### rule debug ####
rule_path = $(notdir $(patsubst %/,%,$(dir $<)))
rule_file = $(basename $(notdir $@))
rule_dumb = @echo -e $(tgt): $(rule_path)/$(rule_file)
rule_xterm = @echo -e \\033[32\;1m$(rule_path)\\033[m/\\033[\;1m$(rule_file)\\033[m
rule = $(rule_${TERM})

#### quiet ####
quiet := --quiet

#### PHONIES ####
.PHONY = clean cln all

#### 
all: | $(shell mkdir -p $(obj));$(rule)
	$(MAKE) $(quiet) -j $(CPU) lib$(tgt).so tst

#### Main target ####
lib$(tgt).so: $(OBJ);$(rule)
	$(CXX) -shared -Wl,-soname,libdbg.so -o $@ $(OBJ) $(LIB)
#	$(AR) $(ARFLAGS) $@ $(OBJ) $(LIB)
#	$(RANLIB) $@

#### tst ####
tst:tst.c lib$(tgt).so makefile
	$(CXX) -fno-inline-functions $(FLG) -o $@ $< -l$(tgt)

#### ./*.cpp #####
$(obj)/%.o: $(pwd)/%.cpp $(pwd)/%.hpp;$(rule)
	$(CXX) -c $(FLG) -o $@ $<

#### ./*.c #####
$(obj)/%.co: $(pwd)/%.c $(pwd)/%.h;$(rule)
	$(CXX) -c $(FLG) -o $@ $<

#### Clean ####
clean cln:;$(rule)
	rm -rf $(obj) lib$(tgt).so tst
