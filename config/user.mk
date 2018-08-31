MFEM_USE_LAPACK      = YES
MFEM_USE_HIOP        = YES

# LAPACK library configuration
LAPACK_DIR = @MFEM_DIR@/../lapack-3.8.0
BLAS_DIR = @MFEM_DIR@/../blas-3.8.0
LAPACK_LIB = $(if $(NOTMAC),-L$(LAPACK_DIR) -llapack -L$(BLAS_DIR) -lblas -lgfortran,-framework Accelerate)
