# Copyright Jed Brown
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice, this
#   list of conditions and the following disclaimer in the documentation and/or
#   other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# See https://github.com/jedbrown/cmake-modules/

# ResolveCompilerPaths - this module defines two macros
#
# RESOLVE_LIBRARIES (XXX_LIBRARIES LINK_LINE)
#  This macro is intended to be used by FindXXX.cmake modules.
#  It parses a compiler link line and resolves all libraries
#  (-lfoo) using the library path contexts (-L/path) in scope.
#  The result in XXX_LIBRARIES is the list of fully resolved libs.
#  Example:
#
#    RESOLVE_LIBRARIES (FOO_LIBRARIES "-L/A -la -L/B -lb -lc -ld")
#
#  will be resolved to
#
#    FOO_LIBRARIES:STRING="/A/liba.so;/B/libb.so;/A/libc.so;/usr/lib/libd.so"
#
#  if the filesystem looks like
#
#    /A:       liba.so         libc.so
#    /B:       liba.so libb.so
#    /usr/lib: liba.so libb.so libc.so libd.so
#
#  and /usr/lib is a system directory.
#
#  Note: If RESOLVE_LIBRARIES() resolves a link line differently from
#  the native linker, there is a bug in this macro (please report it).
#
# RESOLVE_INCLUDES (XXX_INCLUDES INCLUDE_LINE)
#  This macro is intended to be used by FindXXX.cmake modules.
#  It parses a compile line and resolves all includes
#  (-I/path/to/include) to a list of directories.  Other flags are ignored.
#  Example:
#
#    RESOLVE_INCLUDES (FOO_INCLUDES "-I/A -DBAR='\"irrelevant -I/string here\"' -I/B")
#
#  will be resolved to
#
#    FOO_INCLUDES:STRING="/A;/B"
#
#  assuming both directories exist.
#  Note: as currently implemented, the -I/string will be picked up mistakenly (cry, cry)
include (CorrectWindowsPaths)

macro (RESOLVE_LIBRARIES LIBS LINK_LINE)
  string (REGEX MATCHALL "((-L|-l|-Wl)([^\" ]+|\"[^\"]+\")|[^\" ]+\\.(a|so|dll|lib))" _all_tokens "${LINK_LINE}")
  set (_libs_found "")
  set (_directory_list "")
  foreach (token ${_all_tokens})
    if (token MATCHES "-L([^\" ]+|\"[^\"]+\")")
      # If it's a library path, add it to the list
      string (REGEX REPLACE "^-L" "" token ${token})
      string (REGEX REPLACE "//" "/" token ${token})
      convert_cygwin_path(token)
      list (APPEND _directory_list ${token})
    elseif (token MATCHES "^(-l([^\" ]+|\"[^\"]+\")|[^\" ]+\\.(a|so|dll|lib))")
      # It's a library, resolve the path by looking in the list and then (by default) in system directories
      if (WIN32) #windows expects "libfoo", Linux expects "foo"
        string (REGEX REPLACE "^-l" "lib" token ${token})
      else (WIN32)
        string (REGEX REPLACE "^-l" "" token ${token})
      endif (WIN32)
      set (_root "")
      if (token MATCHES "^/")	# We have an absolute path
        #separate into a path and a library name:
        string (REGEX MATCH "[^/]*\\.(a|so|dll|lib)$" libname ${token})
        string (REGEX MATCH ".*[^${libname}$]" libpath ${token})
        convert_cygwin_path(libpath)
        set (_directory_list ${_directory_list} ${libpath})
        set (token ${libname})
      endif (token MATCHES "^/")
      set (_lib "NOTFOUND" CACHE FILEPATH "Cleared" FORCE)
      find_library (_lib ${token} HINTS ${_directory_list} ${_root})
      if (_lib)
	string (REPLACE "//" "/" _lib ${_lib})
        list (APPEND _libs_found ${_lib})
      else (_lib)
        message (STATUS "Unable to find library ${token}")
      endif (_lib)
    endif (token MATCHES "-L([^\" ]+|\"[^\"]+\")")
  endforeach (token)
  set (_lib "NOTFOUND" CACHE INTERNAL "Scratch variable" FORCE)
  # only the LAST occurrence of each library is required since there should be no circular dependencies
  if (_libs_found)
    list (REVERSE _libs_found)
    list (REMOVE_DUPLICATES _libs_found)
    list (REVERSE _libs_found)
  endif (_libs_found)
  set (${LIBS} "${_libs_found}")
endmacro (RESOLVE_LIBRARIES)

macro (RESOLVE_INCLUDES INCS COMPILE_LINE)
  string (REGEX MATCHALL "-I([^\" ]+|\"[^\"]+\")" _all_tokens "${COMPILE_LINE}")
  set (_incs_found "")
  foreach (token ${_all_tokens})
    string (REGEX REPLACE "^-I" "" token ${token})
    string (REGEX REPLACE "//" "/" token ${token})
    convert_cygwin_path(token)
    if (EXISTS ${token})
      list (APPEND _incs_found ${token})
    else (EXISTS ${token})
      message (STATUS "Include directory ${token} does not exist")
    endif (EXISTS ${token})
  endforeach (token)
  list (REMOVE_DUPLICATES _incs_found)
  set (${INCS} "${_incs_found}")
endmacro (RESOLVE_INCLUDES)
