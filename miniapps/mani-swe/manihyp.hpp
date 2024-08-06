#include "mfem.hpp"

namespace mfem
{
Mesh sphericalMesh(const double r, Element::Type element_type,
                   const int level_serial, const int level_parallel=0, bool parallel=false);
} // end of namespace mfem
