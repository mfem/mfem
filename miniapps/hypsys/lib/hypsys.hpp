#ifndef MFEM_HYPSYS
#define MFEM_HYPSYS

#include "mfem.hpp"
#include "massmat.hpp"
#include "dofs.hpp"

using namespace std;
using namespace mfem;

void velaux(const Vector &x, Vector &v);

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
// Appropriate quadrature rule for faces according to DGTraceIntegrator.
const IntegrationRule* GetFaceIntegrationRule(FiniteElementSpace *fes);

class HyperbolicSystem : public TimeDependentOperator
{
public:
	double t;
	FiniteElementSpace *fes;
	const IntegrationRule *IntRuleElem;
	const IntegrationRule *IntRuleFace;
	EvolutionScheme scheme;
	DenseMatrix ShapeEval;
	DenseTensor DShapeEval;
	DenseTensor ShapeEvalFace;
	DenseTensor ElemInt;
	DenseTensor BdrInt;
	GridFunction inflow;
	const MassMatrixDG *MassMat;
	const InverseMassMatrixDG *InvMassMat;
	const DofInfo *dofs;
	
	mutable Vector z;

	HyperbolicSystem(FiniteElementSpace *fes_, Configuration &config)
						: TimeDependentOperator(fes_->GetVSize()), fes(fes_),
						  scheme(config.scheme), inflow(fes_), z(fes_->GetVSize())
	{
		Mesh *mesh = fes->GetMesh();
		IntRuleElem = GetElementIntegrationRule(fes);
		IntRuleFace = GetFaceIntegrationRule(fes);
		const FiniteElement *el = fes->GetFE(0);
		const int nd = el->GetDof();
		const int nqe = IntRuleElem->GetNPoints();
		const int nqf = IntRuleFace->GetNPoints();
		const int dim = mesh->Dimension();
		Array <int> bdrs, orientation;
		
		// The min and max bounds are represented as CG functions of the same order
		// as the solution, thus having 1:1 dof correspondence inside each element.
		H1_FECollection fecBounds(max(fes->GetOrder(0), 1), dim, BasisType::GaussLobatto);
		FiniteElementSpace fesBounds(mesh, &fecBounds);
		dofs = new DofInfo(fes, &fesBounds);
		
		ShapeEval.SetSize(nd,nqe);
		DShapeEval.SetSize(nd,dim,nqe);
		ShapeEvalFace.SetSize(dofs->NumBdrs, dofs->NumFaceDofs, nqf);
		Vector shape(nd);
		DenseMatrix dshape(nd,dim);
		
		for (int k = 0; k < nqe; k++)
		{
			const IntegrationPoint &ip = IntRuleElem->IntPoint(k);
			el->CalcShape(ip, shape);
			el->CalcDShape(ip, dshape);
			ShapeEval.SetCol(k, shape);
			DShapeEval(k) = dshape;
		}
		
		for (int k = 0; k < nqf; k++)
		{
			const IntegrationPoint &ip = IntRuleFace->IntPoint(k);
			if (dim==1)      { mesh->GetElementVertices(0, bdrs); }
			else if (dim==2) { mesh->GetElementEdges(0, bdrs, orientation); }
			else if (dim==3) { mesh->GetElementFaces(0, bdrs, orientation); }
			
			for (int i = 0; i < dofs->NumBdrs; i++)
			{
            FaceElementTransformations *facetrans = 
					mesh->GetFaceElementTransformations(bdrs[i]);
            
				IntegrationPoint eip;
				facetrans->Face->SetIntPoint(&ip);
				facetrans->Loc1.Transform(ip, eip);
				el->CalcShape(eip, shape);
				
				for (int j = 0; j < dofs->NumFaceDofs; j++)
				{
					ShapeEvalFace(i,j,k) = shape(dofs->BdrDofs(j,i));
				}
			}
		}
		
		MassMat = new MassMatrixDG(fes);
		InvMassMat = new InverseMassMatrixDG(MassMat);
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
	void EvaluateSolution(const Vector &u, Vector &v, const int QuadNum,
								 const int BdrNum) const;
	virtual void EvaluateFlux(const Vector &u, DenseMatrix &f) const = 0;
};

#endif
