#ifndef MFEM_STOKES_PACKAGE_TESTS_ADJOINT_HPP
#define MFEM_STOKES_PACKAGE_TESTS_ADJOINT_HPP

#include "stokes.hpp"
#include <string>

namespace mfem::flow
{

int RunStokesAdjointTests(const std::string &suite, const std::string &invalid = "");

}

#endif
