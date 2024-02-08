
#include "mfem.hpp"
#include <string>
namespace mfem
{


enum ThermalProblem
{
   HeatSink
};

void GetThermalProblem(const ThermalProblem problem,
                       double &filter_radius, double &vol_fraction,
                       std::unique_ptr<Mesh> &mesh, std::unique_ptr<Coefficient> &vforce_cf,
                       Array2D<int> &ess_bdr, Array<int> &ess_bdr_filter,
                       std::string &prob_name, int ref_levels, int par_ref_levels=-1)
{
#ifndef MFEM_USE_MPI
   if (par_ref_levels >= 0)
   {
      MFEM_ABORT("Tried to refine in parallel in serial code.");
   }
#else
   if (!Mpi::IsInitialized() && par_ref_levels >= 0)
   {
      MFEM_ABORT("Tried to refine in parallel without initializing MPI");
   }
#endif
   mesh.reset(new Mesh);
   switch (problem)
   {
      case ThermalProblem::HeatSink:
      {
         if (filter_radius < 0) { filter_radius = 5e-02; }
         if (vol_fraction < 0) { vol_fraction = 0.35; }

         *mesh = Mesh::MakeCartesian2D(4, 4, mfem::Element::Type::QUADRILATERAL, true,
                                       1.0,
                                       1.0);
         mesh->MarkBoundary([](const Vector &x) {return std::fabs(x[0] - 0.5) < 0.25 && x[1] > 1-1e-03; },
         5);
         ess_bdr.SetSize(1, 5);
         ess_bdr_filter.SetSize(5);
         ess_bdr = 0; ess_bdr_filter = 0;
         ess_bdr(0, 4) = 1;
         const Vector center({2.9, 0.5});
         vforce_cf.reset(new ConstantCoefficient(1.0));
         prob_name = "HeatSink";
      } break;
      default:
         mfem_error("Undefined problem.");
   }
   mesh->SetAttributes();

   // 3. Refine the mesh->
   for (int lev = 0; lev < ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }
#ifdef MFEM_USE_MPI
   if (Mpi::IsInitialized() && par_ref_levels >= 0)
   {
      ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
      mesh->Clear();
      mesh.reset(pmesh);
      for (int lev=0; lev<par_ref_levels; lev++)
      {
         mesh->UniformRefinement();
      }
   }
#endif
}
}
