#pragma once
#include "smoother-util.hpp"

using namespace std;
using namespace mfem;

class ElementSmoother: public Solver
{
private:
   int num_procs, myid;
   MPI_Comm comm;
   int nrelems;
   int dim;
   ParFiniteElementSpace * fes = nullptr;
   const Operator * Pr = nullptr;
   Coefficient * cf = nullptr;
   Array<int> eidx; // edge local index
   Array<int> edge_orient; // orientation of edges wrt the element
   Array<FDSolver *> elem_inv;
   Array<int> ess_tdof_list;
   Vector ovlp_count;
   Array<Array<int> * > dofmap;
   Array<ElementTPFunctionCoefficient *> tpcf;
public:
   ElementSmoother(ParFiniteElementSpace * fes_, Array<int> ess_bdr, Coefficient * cf_=nullptr);
   virtual void SetOperator(const Operator &op) { }
   virtual void Mult(const Vector &r, Vector &z) const;
   virtual void MultTranspose(const Vector &r, Vector &z) const { Mult(r,z); }
   virtual ~ElementSmoother(){};
};