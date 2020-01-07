#ifndef MFEM_HYPSYS
#define MFEM_HYPSYS

#include "mfem.hpp"
#include "massmat.hpp"
#include "dofs.hpp"

using namespace std;
using namespace mfem;

enum EvolutionScheme { Standard,
							  MCL };

struct Configuration
{
	int ProblemNum;
	int ConfigNum;
	int order;
	double tEnd;
	double dt;
	int odeSolverType;
	int VisSteps;
	EvolutionScheme scheme;
	Vector bbMin, bbMax;
};

const IntegrationRule* GetElementIntegrationRule(FiniteElementSpace *fes);

class HyperbolicSystem : public TimeDependentOperator
{
public:
	double t;
	FiniteElementSpace *fes;
	const IntegrationRule *IntRuleElem;
	EvolutionScheme scheme;
	DenseMatrix ShapeEval;
	DenseTensor DShapeEval;
	DenseTensor ElemInt;
	DenseTensor BdrInt;
	const MassMatrixDG *MassMat;
	const InverseMassMatrixDG *InvMassMat;
	const DofInfo *dofs;
	
	mutable Vector z;


	HyperbolicSystem(FiniteElementSpace *fes_, Configuration &config)
						: TimeDependentOperator(fes_->GetVSize()), fes(fes_),
						  scheme(config.scheme), z(fes_->GetVSize())
	{
		Mesh *mesh = fes->GetMesh();
		IntRuleElem = GetElementIntegrationRule(fes);
		const FiniteElement *el = fes->GetFE(0);
		const int nd = el->GetDof();
		const int nq = IntRuleElem->GetNPoints();
		const int dim = mesh->Dimension();
		ShapeEval.SetSize(nd,nq);
		DShapeEval.SetSize(nd,dim,nq);
		Vector shape(nd);
		DenseMatrix dshape(nd,dim);
		
		for (int k = 0; k < IntRuleElem->GetNPoints(); k++)
		{
			const IntegrationPoint &ip = IntRuleElem->IntPoint(k);
			el->CalcShape(ip, shape);
			el->CalcDShape(ip, dshape);
			ShapeEval.SetCol(k, shape);
			for (int i = 0; i < dim; i++)
			{
				DShapeEval(i) = dshape;
			}
		}
		
		MassMat = new MassMatrixDG(fes);
		InvMassMat = new InverseMassMatrixDG(MassMat);
		
		// The min and max bounds are represented as CG functions of the same order
		// as the solution, thus having 1:1 dof correspondence inside each element.
		H1_FECollection fecBounds(max(fes->GetOrder(0), 1), dim, BasisType::GaussLobatto);
		FiniteElementSpace fesBounds(mesh, &fecBounds);
		
		dofs = new DofInfo(fes, &fesBounds);
	};
	
   virtual ~HyperbolicSystem()
	{
		delete MassMat;
		delete InvMassMat;
		delete dofs;
	};

   virtual void PreprocessProblem(FiniteElementSpace *fes, GridFunction &u);
	
	void Mult(const Vector &x, Vector &y) const override;
	void EvolveStandard(const Vector &x, Vector &y) const;
	void EvolveMCL     (const Vector &x, Vector &y) const;
	
	void EvaluateSolution(const Vector &u, Vector &v, const int QuadNum) const;
	virtual void EvaluateFlux(const Vector &u, DenseMatrix &f) const = 0;
};

#endif
