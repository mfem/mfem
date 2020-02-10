#ifndef HYPSYS_HYPSYS
#define HYPSYS_HYPSYS

#include <fstream>
#include <iostream>
#include "mfem.hpp"
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
									  int NumEq_, Configuration &config_) :
      fes(fes_), inflow(fes_), u0(fes_, u_block), NumEq(NumEq_) { };
   ~HyperbolicSystem() { };

   virtual void EvaluateFlux(const Vector &u, DenseMatrix &FluxEval,
									  int e, int k, int i = -1) const = 0;
	virtual double GetWaveSpeed(const Vector &u, const Vector n, int e, int k, int i) const = 0;
   virtual void ComputeErrors(Array<double> &errors, double DomainSize,
                              const GridFunction &u) const = 0;
   virtual void WriteErrors(const Array<double> &errors) const = 0;

	const int NumEq;
	
   FiniteElementSpace *fes;
   GridFunction inflow, u0;

   bool SolutionKnown = true;
   bool FileOutput = false;
   bool SteadyState;
};

#endif
