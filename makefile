# Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at the
# Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the MFEM library. For more information and source code
# availability see http://mfem.googlecode.com.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.

CC     = g++
CCOPTS = -O3
DEBUG_OPTS = -g -DMFEM_DEBUG

# Internal mfem options
USE_MEMALLOC     = YES
USE_LAPACK       = NO

USE_MEMALLOC_NO  =
USE_MEMALLOC_YES = -DMFEM_USE_MEMALLOC
USE_MEMALLOC_DEF = $(USE_MEMALLOC_$(USE_MEMALLOC))

USE_LAPACK_NO  =
USE_LAPACK_YES = -DMFEM_USE_LAPACK
USE_LAPACK_DEF = $(USE_LAPACK_$(USE_LAPACK))

DEFS = $(USE_MEMALLOC_DEF) $(USE_LAPACK_DEF)

CCC = $(CC) $(CCOPTS) $(DEFS)

# Generate with 'echo general/*.cpp linalg/*.cpp mesh/*.cpp fem/*.cpp'
SOURCE_FILES = general/array.cpp general/error.cpp general/isockstream.cpp \
general/osockstream.cpp general/sort_pairs.cpp general/stable3d.cpp    \
general/table.cpp general/tic_toc.cpp linalg/bicgstab.cpp              \
linalg/cgsolver.cpp linalg/densemat.cpp linalg/gmres.cpp               \
linalg/matrix.cpp linalg/operator.cpp linalg/pcgsolver.cpp             \
linalg/sparsemat.cpp linalg/sparsesmoothers.cpp linalg/vector.cpp      \
mesh/element.cpp mesh/hexahedron.cpp mesh/mesh.cpp mesh/point.cpp      \
mesh/quadrilateral.cpp mesh/segment.cpp mesh/tetrahedron.cpp           \
mesh/triangle.cpp mesh/vertex.cpp fem/bilinearform.cpp                 \
fem/bilininteg.cpp fem/coefficient.cpp fem/eltrans.cpp fem/fe_coll.cpp \
fem/fe.cpp fem/fespace.cpp fem/geom.cpp fem/gridfunc.cpp               \
fem/intrules.cpp fem/linearform.cpp fem/lininteg.cpp
OBJECT_FILES = $(SOURCE_FILES:.cpp=.o)
# Generated with 'echo general/*.hpp linalg/*.hpp mesh/*.hpp fem/*.hpp'
HEADER_FILES = general/array.hpp general/error.hpp                     \
general/isockstream.hpp general/mem_alloc.hpp general/osockstream.hpp  \
general/sort_pairs.hpp general/stable3d.hpp general/table.hpp          \
general/tic_toc.hpp linalg/densemat.hpp linalg/linalg.hpp              \
linalg/matrix.hpp linalg/operator.hpp linalg/solvers.hpp               \
linalg/sparsemat.hpp linalg/sparsesmoothers.hpp linalg/vector.hpp      \
mesh/element.hpp mesh/hexahedron.hpp mesh/mesh_headers.hpp             \
mesh/mesh.hpp mesh/point.hpp mesh/quadrilateral.hpp mesh/segment.hpp   \
mesh/tetrahedron.hpp mesh/triangle.hpp mesh/vertex.hpp                 \
fem/bilinearform.hpp fem/bilininteg.hpp fem/coefficient.hpp            \
fem/eltrans.hpp fem/fe_coll.hpp fem/fe.hpp fem/fem.hpp fem/fespace.hpp \
fem/geom.hpp fem/gridfunc.hpp fem/intrules.hpp fem/linearform.hpp      \
fem/lininteg.hpp

.SUFFIXES: .cpp .o
.cpp.o:
	cd $(<D); $(CCC) -c $(<F)

lib:	libmfem.a mfem_defs.hpp

debug:
	make "CCOPTS=$(DEBUG_OPTS)"

$(OBJECT_FILES): $(HEADER_FILES)

libmfem.a: $(OBJECT_FILES)
	ar cruv libmfem.a $(OBJECT_FILES)
	ranlib libmfem.a

mfem_defs.hpp:
	@echo "Generating 'mfem_defs.hpp' ..."
	@echo "// Auto-generated file." > mfem_defs.hpp
	for i in $(CCC); do \
		case x$${i} in\
		   x-D*) echo -n "#define " >> mfem_defs.hpp;\
		         echo $${i} | sed -e 's/-D//' >> mfem_defs.hpp;;\
		esac; done

clean:
	rm -f */*.o */*~ *~ libmfem.a mfem_defs.hpp deps.mk
	cd examples; make clean
