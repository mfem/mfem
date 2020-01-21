#ifndef MFEM_HYPSYS
#define MFEM_HYPSYS

#include <fstream>
#include <iostream>
#include "mfem.hpp"
#include "lib/massmat.hpp"
#include "lib/dofs.hpp"

using namespace std;
using namespace mfem;

enum EvolutionScheme { Standard,
							  MCL };

struct Configuration
{
	int ProblemNum;
	int ConfigNum;
	int order;
	double tFinal;
	double dt;
	int odeSolverType;
	int VisSteps;
	EvolutionScheme scheme;
	int precision;
	Vector bbMin, bbMax;
};


class HyperbolicSystem : public TimeDependentOperator
{
public:
	double t;
	FiniteElementSpace *fes;
	Mesh *mesh;
	const IntegrationRule *IntRuleElem;
	const IntegrationRule *IntRuleFace;
	EvolutionScheme scheme;
	DenseMatrix ShapeEval;
	DenseTensor DShapeEval;
	DenseTensor ShapeEvalFace;
	DenseTensor ElemInt;
	DenseTensor BdrInt;
	GridFunction inflow;
	GridFunction u;
	const Vector LumpedMassMat;
	const MassMatrixDG *MassMat;
	const InverseMassMatrixDG *InvMassMat;
	const DofInfo dofs;
	
	bool SolutionKnown = true;
	bool SteadyState;
	
	int dim, nd, ne, nqe, nqf;
	mutable Array<int> vdofs;
	mutable Vector z, w, vec1, vec2, vec3, uElem, uEval, QuadWeightFace;
	mutable int DofInd, nbr;
   mutable Vector uNbr;

	HyperbolicSystem(FiniteElementSpace *fes_, DofInfo &dofs_, const Vector &LumpedMassMat_, Configuration &config);
	
   virtual ~HyperbolicSystem();
	
	virtual void EvaluateFlux(const Vector &u, DenseMatrix &f) const = 0;
	virtual void ComputeErrors(Array<double> &errors, double DomainSize) const = 0;
	virtual void WriteErrors(const Array<double> &errors) const = 0;
	
	void Mult(const Vector &x, Vector &y) const override;
	void EvolveStandard(const Vector &x, Vector &y) const;
	void EvolveMCL     (const Vector &x, Vector &y) const;
	
	double ConvergenceCheck(double dt, double tol);
};


const IntegrationRule* GetElementIntegrationRule(FiniteElementSpace *fes);

// Appropriate quadrature rule for faces, conforming with DGTraceIntegrator.
const IntegrationRule* GetFaceIntegrationRule(FiniteElementSpace *fes);

#endif
