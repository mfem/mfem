#include "local_stiffening_coefficient.hpp"

using namespace mfem;


//////////////////////////////////////////////////////
// ----- Abstract LocalStiffeningCoefficient
//////////////////////////////////////////////////////
LocalStiffeningCoefficient::LocalStiffeningCoefficient(ParFiniteElementSpace
                                                       *fes_, Array<int> &wall_attrs_)
   : fes(fes_), wall_attrs(wall_attrs_), pmesh(fes_->GetParMesh())
{
}

LocalStiffeningCoefficient::~LocalStiffeningCoefficient()
{
   // Base class destructor - nothing to clean up as fes is not owned
}

bool LocalStiffeningCoefficient::RequiresRemeshing(real_t threshold)
{
   int dim = pmesh->Dimension();

   // Iterate over each (local) element and find max skewness norm
   Vector loc_asprVals, loc_skewVals, loc_oriVals;
   real_t loc_norminf_skew, global_norminf_skew;
   Array<int> vdofs;
   DenseMatrix jacobian(dim);
   loc_norminf_skew = 0.0;
   for (int e = 0; e < pmesh->GetNE(); e++)
   {
      const FiniteElement *fe = fes->GetFE(e);
      const IntegrationRule &ir = fe->GetNodes();
      fes->GetElementVDofs(e, vdofs);
      for (int q = 0; q < ir.GetNPoints(); q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         pmesh->GetElementJacobian(e, jacobian, &ip);
         real_t sizeVal;
         Vector asprVals, skewVals, oriVals;
         pmesh->GetGeometricParametersFromJacobian(jacobian, sizeVal,
                                                   asprVals, skewVals, oriVals);

         loc_norminf_skew = std::max(loc_norminf_skew, skewVals.Normlinf());
      }
   }

   // Compute (global) max pmesh-skewness norm
   MPI_Allreduce(&loc_norminf_skew, &global_norminf_skew, 1, MFEM_MPI_REAL_T,
                 MPI_MAX, pmesh->GetComm());

   return global_norminf_skew > threshold;
}

//////////////////////////////////////////////////////
// ----- Derived JacobianBasedStiffeningCoefficient
//////////////////////////////////////////////////////

JacobianBasedStiffeningCoefficient::JacobianBasedStiffeningCoefficient(
   ParFiniteElementSpace *fes_, Array<int> &wall_attrs_, real_t stiffening_degree_)
   : LocalStiffeningCoefficient(fes_, wall_attrs_),
     stiffening_degree(stiffening_degree_)
{
}

JacobianBasedStiffeningCoefficient::~JacobianBasedStiffeningCoefficient()
{
   // Nothing to clean up
}

real_t JacobianBasedStiffeningCoefficient::Eval(ElementTransformation &T,
                                                const IntegrationPoint &ip)
{
   // Compute stiffening coefficient
   T.SetIntPoint(&ip);
   real_t weightJ = T.Weight();

   return std::pow(weightJ, -stiffening_degree);
}



//////////////////////////////////////////////////////
// ----- Derived JacobianBasedStiffeningCoefficient
//////////////////////////////////////////////////////

DistanceBasedStiffeningCoefficient::DistanceBasedStiffeningCoefficient(
   ParFiniteElementSpace *fes_, Array<int> &wall_attrs_, int distance_solver_type,
   bool filter_ls_)
   : LocalStiffeningCoefficient(fes_, wall_attrs_), filter_ls(filter_ls_)
{
   // Create Level-Set coefficient for wall attribute
   ls_gf.SetSpace(fes);
   ls_gf = 1.0;
   ConstantCoefficient zero_coeff(0.0);
   ls_gf.ProjectBdrCoefficient(zero_coeff, wall_attrs);
   ls_coeff.SetGridFunction(&ls_gf);

   const real_t dx = AvgElementSize(*pmesh);
   // CHECK: do we need filtering in this case?
   if (filter_ls)
   {
      real_t filter_weight = distance_solver_type == 2 ? 4 * dx : dx;
      PDEFilter filter(*pmesh, filter_weight);
      filter.Filter(ls_coeff, ls_gf);
   }

   // Create distance solver
   switch (distance_solver_type)
   {
      case 0: // HeatDistanceSolver
      {
         distance_solver = new HeatDistanceSolver(dx * dx);
         break;
      }
      case 1: // PLapDistanceSolver
      {
         const int p = 15;
         const int newton_iter = 50;
         distance_solver = new PLapDistanceSolver(p, newton_iter);
         break;
      }
      case 2: // NormalizationDistanceSolver
         distance_solver = new NormalizationDistanceSolver();
         break;
   }

   distance_gf.SetSpace(fes);
   dist_coeff.SetGridFunction(
      &distance_gf); // maps refined (LOR) element transformations back to the parent elements
   distance_solver->ComputeScalarDistance(ls_coeff, distance_gf);
}

DistanceBasedStiffeningCoefficient::~DistanceBasedStiffeningCoefficient()
{
   delete distance_solver;
}

void DistanceBasedStiffeningCoefficient::SetParameters(real_t Rmin, real_t Rmax,
                                                       real_t alpha, real_t Amax, real_t Amin, real_t stiffening_degree)
{
   this->Rmin = Rmin;
   this->Rmax = Rmax;
   this->Rmid = 0.5 * (Rmin + Rmax);
   this->alpha = alpha;
   this->Amax = Amax;
   this->Amin = Amin;
   this->stiffening_degree = stiffening_degree;
}


void DistanceBasedStiffeningCoefficient::Update()
{
   distance_gf.Update();
   ls_gf.Update();

   distance_solver->ComputeScalarDistance(ls_coeff, distance_gf);
}


real_t DistanceBasedStiffeningCoefficient::Eval(ElementTransformation &T,
                                                const IntegrationPoint &ip)
{
   // Get distance at the current point
   real_t distance = dist_coeff.Eval(T, ip);

   // Compute the sigmoid stiffening coefficient based on the distance (argument normalized by the zone width)
   real_t A = Amin + (Amax - Amin) / (1 + std::exp(alpha * (distance - Rmid) /
                                                   (Rmax - Rmin)));

   return std::pow(A, stiffening_degree);
}






















