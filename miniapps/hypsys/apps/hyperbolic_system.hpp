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
                             int NumEq_, Configuration &config_, VectorFunctionCoefficient b_) :
      fes(fes_), inflow(fes_), u0(fes_, u_block), NumEq(NumEq_), b(b_) { };
   virtual ~HyperbolicSystem() { };

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

   const int NumEq;

   FiniteElementSpace *fes;
   GridFunction inflow, u0;
   mutable VectorFunctionCoefficient b;

   bool SolutionKnown = true;
   bool FileOutput = true;
   bool SteadyState;
};

#endif
