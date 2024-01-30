
#include "mfem.hpp"
#include <string>
namespace mfem
{


enum ElasticityProblem
{
   Cantilever,
   MBB,
   LBracket,
   Cantilever3,
   Torsion3
};

enum CompliantProblem
{
   ForceInverter,
   ForceInverter3
};

void GetElasticityProblem(const ElasticityProblem problem,
                          double &filter_radius, double &vol_fraction,
                          std::unique_ptr<Mesh> &mesh, std::unique_ptr<VectorCoefficient> &vforce_cf,
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
      case ElasticityProblem::Cantilever:
      {
         if (filter_radius < 0) { filter_radius = 5e-02; }
         if (vol_fraction < 0) { vol_fraction = 0.5; }

         *mesh = Mesh::MakeCartesian2D(3, 1, mfem::Element::Type::QUADRILATERAL, true,
                                       3.0,
                                       1.0);
         ess_bdr.SetSize(3, 4);
         ess_bdr_filter.SetSize(4);
         ess_bdr = 0; ess_bdr_filter = 0;
         ess_bdr(0, 3) = 1;
         const Vector center({2.9, 0.5});
         vforce_cf.reset(new VectorFunctionCoefficient(2, [center](const Vector &x,
                                                                   Vector &f)
         {
            f = 0.0;
            if (x.DistanceTo(center) < 0.05) { f(1) = -1.0; }
         }));
         prob_name = "Cantilever";
      } break;
      case ElasticityProblem::MBB:
      {
         if (filter_radius < 0) { filter_radius = 5e-02; }
         if (vol_fraction < 0) { vol_fraction = 0.5; }

         *mesh = Mesh::MakeCartesian2D(3, 1, mfem::Element::Type::QUADRILATERAL, true,
                                       3.0,
                                       1.0);
         ess_bdr.SetSize(3, 5);
         ess_bdr_filter.SetSize(5);
         ess_bdr = 0; ess_bdr_filter = 0;
         ess_bdr(1, 3) = 1; // left : y-roller -> x fixed
         ess_bdr(2, 4) = 1; // right-bottom : x-roller -> y fixed
         const Vector center({0.05, 0.95});
         vforce_cf.reset(new VectorFunctionCoefficient(2, [center](const Vector &x,
                                                                   Vector &f)
         {
            f = 0.0;
            if (x.DistanceTo(center) < 0.05) { f(1) = -1.0; }
         }));
         prob_name = "MBB";
      } break;
      case ElasticityProblem::LBracket:
      {
         if (filter_radius < 0) { filter_radius = 5e-02; }
         if (vol_fraction < 0) { vol_fraction = 0.5; }

         ref_levels--;
         *mesh = mesh->LoadFromFile("../../data/lbracket_square.mesh");
         ess_bdr.SetSize(3, 6);
         ess_bdr_filter.SetSize(6);
         ess_bdr = 0; ess_bdr_filter = 0;
         ess_bdr(0, 4) = 1;
         const Vector center({0.95, 0.35});
         vforce_cf.reset(new VectorFunctionCoefficient(2, [center](const Vector &x,
                                                                   Vector &f)
         {
            f = 0.0;
            if (x.DistanceTo(center) < 0.05) { f(1) = -1.0; }
         }));
         prob_name = "LBracket";
      } break;
      case ElasticityProblem::Cantilever3:
      {
         if (filter_radius < 0) { filter_radius = 5e-02; }
         if (vol_fraction < 0) { vol_fraction = 0.12; }
         // 1: bottom,
         // 2: front,
         // 3: right,
         // 4: back,
         // 5: left,
         // 6: top
         *mesh = Mesh::MakeCartesian3D(2, 1, 1, mfem::Element::Type::HEXAHEDRON, 2.0,
                                       1.0,
                                       1.0);
         ess_bdr.SetSize(4, 6);
         ess_bdr_filter.SetSize(6);
         ess_bdr = 0; ess_bdr_filter = 0;
         ess_bdr(0, 4) = 1;

         const Vector center({1.9, 0.1, 0.25});
         // force(0) = 0.0; force(1) = 0.0; force(2) = -1.0;
         vforce_cf.reset(new VectorFunctionCoefficient(3, [center](const Vector &x,
                                                                   Vector &f)
         {
            f = 0.0;
            Vector xx(x); xx(1) = center(1);
            if (xx.DistanceTo(center) < 0.05) { f(2) = -std::sin(M_PI*x(1)); }
         }));
         prob_name = "Cantilever3";
      } break;

      case ElasticityProblem::Torsion3:
      {
         if (filter_radius < 0) { filter_radius = 0.05; }
         if (vol_fraction < 0) { vol_fraction = 0.01; }

         // [1: bottom, 2: front, 3: right, 4: back, 5: left, 6: top]
         *mesh = Mesh::MakeCartesian3D(6, 5, 5, mfem::Element::Type::HEXAHEDRON, 1.2,
                                       1.0,
                                       1.0);
         ess_bdr.SetSize(4, 7);
         ess_bdr_filter.SetSize(7);
         ess_bdr = 0; ess_bdr_filter = 0;
         ess_bdr(0, 6) = 1;

         const Vector center({0.0, 0.5, 0.5});
         vforce_cf.reset(new VectorFunctionCoefficient(3, [center](const Vector &x,
                                                                   Vector &f)
         {
            Vector xx(x); xx(0) = 0.0;
            xx -= center;
            double d = xx.Norml2();
            if (x[0] > 1.0 && d < 0.2)
            {
               f[0] = 0.0;
               f[1] = -xx[2];
               f[2] = xx[1];
            }
            else
            {
               f = 0.0;
            }
         }));
         prob_name = "Torsion3";
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
   switch (problem)
   {
      case ElasticityProblem::MBB:
      {
         {
            mesh->MarkBoundary([](const Vector &x) {return ((x(0) > (3 - std::pow(2, -5))) && (x(1) < 1e-10)); },
            5);
         } break;
      }
      case ElasticityProblem::Torsion3:
      {
         {
            // left center: Dirichlet
            Vector center({0.0, 0.5, 0.5});
            mesh->MarkBoundary([center](const Vector &x) { return (center.DistanceTo(x) < 0.2); },
            7);
         } break;
      }
      default:
         break;
   }

}

void GetCompliantProblem(const CompliantProblem problem,
                          double &filter_radius, double &vol_fraction,
                          std::unique_ptr<Mesh> &mesh, std::unique_ptr<VectorCoefficient> &vforce_cf, std::unique_ptr<VectorCoefficient> &sforce_cf,
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
      case ElasticityProblem::Cantilever:
      {
         if (filter_radius < 0) { filter_radius = 5e-02; }
         if (vol_fraction < 0) { vol_fraction = 0.5; }

         *mesh = Mesh::MakeCartesian2D(3, 1, mfem::Element::Type::QUADRILATERAL, true,
                                       3.0,
                                       1.0);
         ess_bdr.SetSize(3, 4);
         ess_bdr_filter.SetSize(4);
         ess_bdr = 0; ess_bdr_filter = 0;
         ess_bdr(0, 3) = 1;
         const Vector center({2.9, 0.5});
         vforce_cf.reset(new VectorFunctionCoefficient(2, [center](const Vector &x,
                                                                   Vector &f)
         {
            f = 0.0;
            if (x.DistanceTo(center) < 0.05) { f(1) = -1.0; }
         }));
         prob_name = "Cantilever";
      } break;
      case ElasticityProblem::MBB:
      {
         if (filter_radius < 0) { filter_radius = 5e-02; }
         if (vol_fraction < 0) { vol_fraction = 0.5; }

         *mesh = Mesh::MakeCartesian2D(3, 1, mfem::Element::Type::QUADRILATERAL, true,
                                       3.0,
                                       1.0);
         ess_bdr.SetSize(3, 5);
         ess_bdr_filter.SetSize(5);
         ess_bdr = 0; ess_bdr_filter = 0;
         ess_bdr(1, 3) = 1; // left : y-roller -> x fixed
         ess_bdr(2, 4) = 1; // right-bottom : x-roller -> y fixed
         const Vector center({0.05, 0.95});
         vforce_cf.reset(new VectorFunctionCoefficient(2, [center](const Vector &x,
                                                                   Vector &f)
         {
            f = 0.0;
            if (x.DistanceTo(center) < 0.05) { f(1) = -1.0; }
         }));
         prob_name = "MBB";
      } break;
      case ElasticityProblem::LBracket:
      {
         if (filter_radius < 0) { filter_radius = 5e-02; }
         if (vol_fraction < 0) { vol_fraction = 0.5; }

         ref_levels--;
         *mesh = mesh->LoadFromFile("../../data/lbracket_square.mesh");
         ess_bdr.SetSize(3, 6);
         ess_bdr_filter.SetSize(6);
         ess_bdr = 0; ess_bdr_filter = 0;
         ess_bdr(0, 4) = 1;
         const Vector center({0.95, 0.35});
         vforce_cf.reset(new VectorFunctionCoefficient(2, [center](const Vector &x,
                                                                   Vector &f)
         {
            f = 0.0;
            if (x.DistanceTo(center) < 0.05) { f(1) = -1.0; }
         }));
         prob_name = "LBracket";
      } break;
      case ElasticityProblem::Cantilever3:
      {
         if (filter_radius < 0) { filter_radius = 5e-02; }
         if (vol_fraction < 0) { vol_fraction = 0.12; }
         // 1: bottom,
         // 2: front,
         // 3: right,
         // 4: back,
         // 5: left,
         // 6: top
         *mesh = Mesh::MakeCartesian3D(2, 1, 1, mfem::Element::Type::HEXAHEDRON, 2.0,
                                       1.0,
                                       1.0);
         ess_bdr.SetSize(4, 6);
         ess_bdr_filter.SetSize(6);
         ess_bdr = 0; ess_bdr_filter = 0;
         ess_bdr(0, 4) = 1;

         const Vector center({1.9, 0.1, 0.25});
         // force(0) = 0.0; force(1) = 0.0; force(2) = -1.0;
         vforce_cf.reset(new VectorFunctionCoefficient(3, [center](const Vector &x,
                                                                   Vector &f)
         {
            f = 0.0;
            Vector xx(x); xx(1) = center(1);
            if (xx.DistanceTo(center) < 0.05) { f(2) = -std::sin(M_PI*x(1)); }
         }));
         prob_name = "Cantilever3";
      } break;

      case ElasticityProblem::Torsion3:
      {
         if (filter_radius < 0) { filter_radius = 0.05; }
         if (vol_fraction < 0) { vol_fraction = 0.01; }

         // [1: bottom, 2: front, 3: right, 4: back, 5: left, 6: top]
         *mesh = Mesh::MakeCartesian3D(6, 5, 5, mfem::Element::Type::HEXAHEDRON, 1.2,
                                       1.0,
                                       1.0);
         ess_bdr.SetSize(4, 7);
         ess_bdr_filter.SetSize(7);
         ess_bdr = 0; ess_bdr_filter = 0;
         ess_bdr(0, 6) = 1;

         const Vector center({0.0, 0.5, 0.5});
         vforce_cf.reset(new VectorFunctionCoefficient(3, [center](const Vector &x,
                                                                   Vector &f)
         {
            Vector xx(x); xx(0) = 0.0;
            xx -= center;
            double d = xx.Norml2();
            if (x[0] > 1.0 && d < 0.2)
            {
               f[0] = 0.0;
               f[1] = -xx[2];
               f[2] = xx[1];
            }
            else
            {
               f = 0.0;
            }
         }));
         prob_name = "Torsion3";
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
   switch (problem)
   {
      case ElasticityProblem::MBB:
      {
         {
            mesh->MarkBoundary([](const Vector &x) {return ((x(0) > (3 - std::pow(2, -5))) && (x(1) < 1e-10)); },
            5);
         } break;
      }
      case ElasticityProblem::Torsion3:
      {
         {
            // left center: Dirichlet
            Vector center({0.0, 0.5, 0.5});
            mesh->MarkBoundary([center](const Vector &x) { return (center.DistanceTo(x) < 0.2); },
            7);
         } break;
      }
      default:
         break;
   }

}
}
