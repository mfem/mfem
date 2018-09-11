#pragma once

#include <cmath>
#include <list>
#include <vector>
#include <utility>
#ifdef __NVCC__
#include "cuda_runtime.h"
#include "cuda.h"
#endif
// #include "ftc.hpp"

namespace mfem
{

namespace pa
{

#ifdef __NVCC__

using std::size_t;

class CudaKernel {
public:
    typedef std::list< std::pair<std::string, std::string> > DefineList;

    CudaKernel(const char* base_path, const char* kernel_file,
               const char* kernel_name, const DefineList& define_list,
               const std::vector<const char*>& include_names,
               const int num_opts = 0, const std::vector<const char*>& = {});

    int launch(const dim3 grid, const dim3 threads,
               const unsigned int shared_bytes, CUstream hstream,
               void* args[], void* extra[] = {}) const;

    int launch(const dim3 grid, const dim3 threads, void *args[]) const;

private:
    CUfunction cukernel;
};

#endif

}

}