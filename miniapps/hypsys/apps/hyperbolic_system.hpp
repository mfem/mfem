#ifndef HYPSYS_HYPERBOLIC_SYSTEM
#define HYPSYS_HYPERBOLIC_SYSTEM

#include <fstream>
#include <iostream>
#include "../../../mfem.hpp"
#include "../lib/tools.hpp"

using namespace std;
using namespace mfem;

struct Configuration
{
   int ProblemNum;
   int ConfigNum;
   int order;
   double tFinal;
   double dt;
   int odeSolverType;
   int VisSteps;
   int precision;
   Vector bbMin, bbMax;
};

class HyperbolicSystem
{
public:
   explicit HyperbolicSystem(FiniteElementSpace *fes_, BlockVector &u_block,
                             int NumEq_, Configuration &config_,
                             VectorFunctionCoefficient BdrCond_) : fes(fes_), inflow(fes_), u0(fes_,
                                      u_block), NumEq(NumEq_), BdrCond(BdrCond_)
   {
      l2_fec = new L2_FECollection(fes->GetFE(0)->GetOrder(),
                                   fes->GetMesh()->Dimension());
      l2_fes = new FiniteElementSpace(fes->GetMesh(), l2_fec, NumEq,
                                      Ordering::byNODES);
      l2_proj = new GridFunction(l2_fes);
   }

   virtual ~HyperbolicSystem()
   {
      delete l2_proj;
      delete l2_fes;
      delete l2_fec;
   }

   virtual void EvaluateFlux(const Vector &u, DenseMatrix &FluxEval,
                             int e, int k, int i = -1) const = 0;
   virtual double GetWaveSpeed(const Vector &u, const Vector n, int e, int k,
                               int i) const = 0;
   virtual void ComputeErrors(Array<double> &errors, const GridFunction &u,
                              double DomainSize, double t) const = 0;
   virtual void WriteErrors(const Array<double> &errors) const
   {
      ofstream file("errors.txt", ios_base::app);

      if (!file)
      {
         MFEM_ABORT("Error opening file.");
      }
      else
      {
         ostringstream strs;
         strs << errors[0] << " " << errors[1] << " " << errors[2] << "\n";
         string str = strs.str();
         file << str;
         file.close();
      }
   }

   // L2 projection for scalar problems.
   void L2_Projection(FunctionCoefficient fun, GridFunction &proj) const
   {
      l2_proj->ProjectCoefficient(fun);
      proj.ProjectGridFunction(*l2_proj);
   }

   // L2 projection for systems.
   void L2_Projection(VectorFunctionCoefficient fun, GridFunction &proj) const
   {
      l2_proj->ProjectCoefficient(fun);
      proj.ProjectGridFunction(*l2_proj);
   }

   // 0: L2 projection, 1: Nodal values as dofs (second order accurate for Bernstein)
   int ProjType;
   const int NumEq;

   FiniteElementSpace *fes;
   GridFunction inflow, u0; // TODO is this inflow used?

   // Auxiliary data needed for L2 projections
   L2_FECollection *l2_fec;
   FiniteElementSpace *l2_fes;
   GridFunction *l2_proj;

   mutable VectorFunctionCoefficient BdrCond;

   string ProblemName;
   bool SolutionKnown;
   bool SteadyState;
   bool TimeDepBC;

   bool FileOutput = true;
};

#endif
