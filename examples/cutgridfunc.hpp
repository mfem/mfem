#include "mfem.hpp"


using namespace mfem;

/// A derived grid function class used for store information on the element center
class CutGridFunction : public mfem::GridFunction
{
public:
   CutGridFunction(mfem::FiniteElementSpace *f);

   virtual void ProjectCoefficient(mfem::VectorCoefficient &coeff);
   CutGridFunction &operator=(const Vector &v);
   CutGridFunction &operator=(double value);

   // mfem::HypreParVector *GetTrueDofs()
   // {
   //    mfem::HypreParVector *tv = new HypreParVector(comm,GlobalTrueVSize(),GetTrueDofOffsets()));
   // }

};