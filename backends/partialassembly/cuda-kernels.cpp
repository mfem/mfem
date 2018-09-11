#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <cstring>

#include "../../general/error.hpp"

#ifdef __NVCC__
#include "cuda-kernels.hpp"
#include "cuda.h"
#include "nvrtc.h"
#endif


namespace mfem
{

namespace pa
{

#ifdef __NVCC__

// ------ Utility methods ------
namespace {

void nvrtcError(nvrtcProgram& prog, const char message[]) {
    size_t log_size;
    nvrtcGetProgramLogSize(prog, &log_size);
    char *compile_log = new char[log_size];
    nvrtcGetProgramLog(prog, compile_log);
    std::cerr << compile_log << std::endl;
    delete[] compile_log;
    std::cerr << message << std::endl;
}

void appendFile(const std::string& file_name, std::string& str) {
    std::ifstream t(file_name);

    MFEM_VERIFY(t.good(), file_name.c_str() );

    t.seekg(0, std::ios::end);
    str.reserve(str.size() + t.tellg());
    t.seekg(0, std::ios::beg);

    str.append(std::istreambuf_iterator<char>(t),
               std::istreambuf_iterator<char>());
}

}

CudaKernel::CudaKernel(const char* base_path, const char* kernel_file,
                       const char* kernel_name, const DefineList& define_list,
                       const std::vector<const char*>& include_names,
                       const int num_opts, const std::vector<const char*>& opts) {
    // Build the source file as a std::string
    // Add defines
    std::stringstream s;
    for (auto const& x : define_list)
        s << "#define " << x.first << " " << x.second << "\n";
    std::string source(s.str());

    {
        std::stringstream s;
        s << base_path << kernel_file;
        appendFile(s.str(), source);
    }

    std::vector<char*> headers(include_names.size());
    for (int i = 0; i < headers.size(); i++) {
        std::stringstream s;
        s << base_path << include_names[i];
        std::string tmp; 
        appendFile(s.str(), tmp);
        headers[i] = new char[tmp.size()];
        strcpy(headers[i], tmp.c_str());
    }

    nvrtcProgram prog;
    nvrtcResult rcode;
    rcode = nvrtcCreateProgram(&prog,
                               source.c_str(),
                               "kernel.cu",
                               headers.size(),
                               headers.data(),
                               include_names.data());
    std::cout << source << std::endl;
    if (rcode != NVRTC_SUCCESS) nvrtcError(prog, "Encountered error creating program");

    for (auto& c : headers) delete [] c;

    rcode = nvrtcCompileProgram(prog, num_opts, opts.data());
    if (rcode != NVRTC_SUCCESS) nvrtcError(prog, "Encountered error compiling program");

    CUresult rres;
    CUmodule Module;
    size_t ptxSize = 0;

    rcode = nvrtcGetPTXSize(prog, &ptxSize);
    if (rcode != NVRTC_SUCCESS) nvrtcError(prog, "Encountered error obtaining PTX size");
    char *ptx = new char[ptxSize];
    rcode = nvrtcGetPTX(prog, ptx);
    if (rcode != NVRTC_SUCCESS) nvrtcError(prog, "Encountered error getting PTX code");

    // Load the generated PTX and get a handle to the specific kernel
    rres = cuModuleLoadData(&Module, ptx);
    if (rcode != NVRTC_SUCCESS) nvrtcError(prog, "Encountered error loading module");

    rres = cuModuleGetFunction(&cukernel, Module, kernel_name);
    const char* errorString;
    cuGetErrorString(rres, &errorString);
    std::cout << "rres=" << rres << ", " << errorString << std::endl;
    if (rres != CUDA_SUCCESS) nvrtcError(prog, "Encountered error obtaining function from module");

    nvrtcDestroyProgram(&prog);

    // struct cudaFuncAttributes funcAttrib;
    // cudaFuncGetAttributes(&funcAttrib, cukernel);
    // std::cout << "register count = " << funcAttrib.numRegs << std::endl;

    delete [] ptx;
}

int CudaKernel::launch(const dim3 grid, const dim3 threads,
                       const unsigned int shared_bytes, CUstream hstream,
                       void* args[], void* extra[]) const {
    return  cuLaunchKernel(cukernel,
                           grid.x, grid.y, grid.z,          // grid dim
                           threads.x, threads.y, threads.z, // threads dim
                           shared_bytes, hstream,           // shared mem and stream
                           args,                            // arguments
                           extra);
}

int CudaKernel::launch(const dim3 grid, const dim3 threads, void *args[]) const {
    return launch(grid, threads, 0, 0, args);
}

#endif

}

}