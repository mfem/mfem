#ifndef MFEM_HYPSYS
#define MFEM_HYPSYS

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
	const MassMatrixDG *MassMat;
	const InverseMassMatrixDG *InvMassMat;
	const DofInfo *dofs;
	
	int dim, nd, ne, nqe, nqf;
	mutable Array<int> vdofs;
	mutable Vector vec1, vec2, vec3, uElem, uEval, QuadWeightFace;
	mutable Vector z;

	HyperbolicSystem(FiniteElementSpace *fes_, Configuration &config)
						: TimeDependentOperator(fes_->GetVSize()), fes(fes_),
						  scheme(config.scheme), inflow(fes_), z(fes_->GetVSize())
	{
		mesh = fes->GetMesh();
		IntRuleElem = GetElementIntegrationRule(fes);
		IntRuleFace = GetFaceIntegrationRule(fes);
		const FiniteElement *el = fes->GetFE(0);
		dim = mesh->Dimension();
		ne = mesh->GetNE();
		nd = el->GetDof();
		nqe = IntRuleElem->GetNPoints();
		nqf = IntRuleFace->GetNPoints();
		Array <int> bdrs, orientation;
		vec1.SetSize(dim);
		vec2.SetSize(nd);
		vec3.SetSize(nd);
		uElem.SetSize(nd);
		uEval.SetSize(1); // TODO Vector valued soultion.
		QuadWeightFace.SetSize(nqf);
		
		// The min and max bounds are represented as CG functions of the same order
		// as the solution, thus having 1:1 dof correspondence inside each element.
		H1_FECollection fecBounds(max(fes->GetFE(0)->GetOrder(), 1), dim,
										  BasisType::GaussLobatto);
		FiniteElementSpace fesBounds(mesh, &fecBounds);
		dofs =  new DofInfo(fes, &fesBounds);
		
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
			QuadWeightFace(k) = ip.weight;
			
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
		delete dofs;
		delete MassMat;
		delete InvMassMat;
	};

   virtual void PreprocessProblem(FiniteElementSpace *fes, GridFunction &u) = 0;
	virtual void PostprocessProblem(const GridFunction &u, Array<double> &errors) = 0;
	
	void Mult(const Vector &x, Vector &y) const override;
	void EvolveStandard(const Vector &x, Vector &y) const;
	void EvolveMCL     (const Vector &x, Vector &y) const;
	
	virtual void EvaluateFlux(const Vector &u, DenseMatrix &f) const = 0;
};

#endif
