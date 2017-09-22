#include "globalostream.hpp"

namespace mfem
{

WrappedOStream mout(&std::cout);
WrappedOStream merr(&std::cerr);

}