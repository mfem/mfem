#ifndef MFEM_HYPSYS
#define MFEM_HYPSYS

#include "mfem.hpp"
#include "massmat.hpp"

using namespace std;
using namespace mfem;

class HyperbolicSystem : public TimeDependentOperator
{
public:
	double t;
	FiniteElementSpace *fes;

	HyperbolicSystem() : TimeDependentOperator(fes->GetVSize()) { };
   virtual ~HyperbolicSystem() { };

   virtual void PreprocessProblem(FiniteElementSpace *fes, GridFunction &u) = 0;
	
	//virtual void Mult(const Vector &x, Vector &y) const override {};
};

#endif
