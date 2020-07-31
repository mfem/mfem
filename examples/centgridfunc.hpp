#include "mfem.hpp"


using namespace mfem;

/// A derived grid function class used for store information on the element center
class CentGridFunction : public mfem::GridFunction
{
public:
   CentGridFunction(mfem::FiniteElementSpace *f);

   virtual void ProjectCoefficient(mfem::VectorCoefficient &coeff);
   CentGridFunction &operator=(const Vector &v);
   CentGridFunction &operator=(double value);

   // mfem::HypreParVector *GetTrueDofs()
   // {
   //    mfem::HypreParVector *tv = new HypreParVector(comm,GlobalTrueVSize(),GetTrueDofOffsets()));
   // }

};
