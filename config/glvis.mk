# GLVis library - From GLVis makefile

# Macro that searches for a file in a list of directories returning the first
# directory that contains the file.
# $(1) - the file to search for
# $(2) - list of directories to search
define find_dir
$(patsubst %/$(1),%,$(firstword $(wildcard $(foreach d,$(2),$(d)/$(1)))))
endef

# Macro to find the proper library sub-directory, 'lib64' or 'lib', given a
# tentative prefix and a library name. Returns empty path if prefix is empty,
# '/usr', or the library is not found.
# $(1) - the prefix to search, e.g. $(SDL_DIR)
# $(2) - library name without 'lib' prefix, e.g. 'SDL2'
define dir2lib
$(if $(filter-out /usr,$(1)),$(patsubst %/,%,$(dir $(firstword $(wildcard\
 $(1)/lib64/lib$(2).* $(1)/lib/lib$(2).*)))))
endef

BREW_PREFIX := $(if $(NOTMAC),,$(shell brew --prefix 2> /dev/null))

FREETYPE_SEARCH_PATHS = $(BREW_PREFIX) /usr /opt/X11
FREETYPE_SEARCH_FILE = include/freetype2/ft2build.h
FREETYPE_DIR = $(call find_dir,$(FREETYPE_SEARCH_FILE),$(FREETYPE_SEARCH_PATHS))
FREETYPE_LIB_DIR = $(call dir2lib,$(FREETYPE_DIR),freetype)
FREETYPE_LIBS = -lfreetype -lfontconfig

# If GLEW is in /usr, there's no need to add search paths
GLEW_SEARCH_PATHS = /usr/local $(BREW_PREFIX) $(abspath ../glew)
GLEW_SEARCH_FILE = include/GL/glew.h
GLEW_DIR ?= $(call find_dir,$(GLEW_SEARCH_FILE),$(GLEW_SEARCH_PATHS))
GLEW_LIB_DIR = $(call dir2lib,$(GLEW_DIR),GLEW)
GLEW_LIBS = -lGLEW

# If SDL is in /usr, there's no need to add search paths
SDL_SEARCH_PATHS := /usr/local $(BREW_PREFIX) $(abspath ../SDL2)
SDL_SEARCH_FILE = include/SDL2/SDL.h
SDL_DIR ?= $(call find_dir,$(SDL_SEARCH_FILE),$(SDL_SEARCH_PATHS))
SDL_LIB_DIR = $(call dir2lib,$(SDL_DIR),SDL2)
SDL_LIBS = -lSDL2

# If GLM is in /usr/include, there's no need to add search paths
GLM_SEARCH_PATHS = /usr/local/include \
 $(if $(BREW_PREFIX),$(BREW_PREFIX)/include) $(abspath ../glm)
GLM_SEARCH_FILE = glm/glm.hpp
GLM_DIR ?= $(call find_dir,$(GLM_SEARCH_FILE),$(GLM_SEARCH_PATHS))

# If OpenGL is in /usr, there's no need to add search paths
OPENGL_SEARCH_PATHS = /usr/local /opt/local
OPENGL_SEARCH_FILE = include/GL/gl.h
OPENGL_DIR ?= $(call find_dir,$(OPENGL_SEARCH_FILE),$(OPENGL_SEARCH_PATHS))
OPENGL_LIB_DIR = $(if $(NOTMAC),$(call dir2lib,$(OPENGL_DIR),GL))
OPENGL_LIBS = $(if $(NOTMAC),-lGL,-framework OpenGL -framework Cocoa)

# Regarding -DGLEW_NO_GLU, see https://github.com/nigels-com/glew/issues/192
GL_OPTS ?= $(if $(FREETYPE_DIR),-I$(FREETYPE_DIR)/include/freetype2) \
 $(if $(SDL_DIR),-I$(SDL_DIR)/include) \
 $(if $(GLEW_DIR),-I$(GLEW_DIR)/include) -DGLEW_NO_GLU \
 $(if $(GLM_DIR),-I$(GLM_DIR)) \
 $(if $(OPENGL_DIR),-I$(OPENGL_DIR)/include)

rpath=-Wl,-rpath,
GL_LIBS ?= $(if $(FREETYPE_LIB_DIR),-L$(FREETYPE_LIB_DIR)) \
 $(if $(SDL_LIB_DIR),-L$(SDL_LIB_DIR) $(rpath)$(SDL_LIB_DIR)) \
 $(if $(NOTMAC),$(if $(OPENGL_LIB_DIR),-L$(OPENGL_LIB_DIR) \
   $(rpath)$(OPENGL_LIB_DIR))) \
 $(if $(GLEW_LIB_DIR),-L$(GLEW_LIB_DIR) $(rpath)$(GLEW_LIB_DIR)) \
 $(FREETYPE_LIBS) $(SDL_LIBS) $(GLEW_LIBS) $(OPENGL_LIBS)

GLVIS_FLAGS += $(GL_OPTS)
GLVIS_LIBS  += $(GL_LIBS)

# Take screenshots internally with libtiff, libpng, or sdl2?
GLVIS_USE_LIBTIFF ?= NO
GLVIS_USE_LIBPNG  ?= YES
TIFF_OPTS = -DGLVIS_USE_LIBTIFF -I/sw/include
TIFF_LIBS = -L/sw/lib -ltiff
PNG_OPTS = -DGLVIS_USE_LIBPNG
PNG_LIBS = -lpng
ifeq ($(GLVIS_USE_LIBTIFF),YES)
   GLVIS_FLAGS += $(TIFF_OPTS)
   GLVIS_LIBS  += $(TIFF_LIBS)
else ifeq ($(GLVIS_USE_LIBPNG),YES)
   GLVIS_FLAGS += $(PNG_OPTS)
   GLVIS_LIBS  += $(PNG_LIBS)
else
   # no flag --> SDL screenshots
endif

# EGL headless rendering
GLVIS_USE_EGL ?= NO
EGL_OPTS = -DGLVIS_USE_EGL
EGL_LIBS = -lEGL
ifeq ($(GLVIS_USE_EGL),YES)
	GLVIS_FLAGS += $(EGL_OPTS)
	GLVIS_LIBS  += $(EGL_LIBS)
endif

# CGL headless rendering
GLVIS_USE_CGL ?= $(if $(NOTMAC),NO,YES)
CGL_OPTS = -DGLVIS_USE_CGL
ifeq ($(GLVIS_USE_CGL),YES)
	GLVIS_FLAGS += $(CGL_OPTS)
endif

PTHREAD_LIB = -lpthread
GLVIS_LIBS += $(PTHREAD_LIB)
GLVIS_LIBS += -lmfem
