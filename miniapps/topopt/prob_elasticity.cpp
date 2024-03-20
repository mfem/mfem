#include "mfem.hpp"
#include "prob_elasticity.hpp"
namespace mfem
{
double DistanceToSegment(const Vector &p, const Vector &v, const Vector &w)
{
   const double l2 = v.DistanceSquaredTo(w);
   const double t = std::max(0.0, std::min(1.0,
                                           ((p*w) - (p*v) - (v*w) + (v*v))/l2));
   Vector projection(v);
   projection.Add(t, w);
   projection.Add(-t, v);
   return p.DistanceTo(projection);
}

void initialDesign(GridFunction& psi, Vector domain_center,
                   Array<Vector*> ports,
                   double target_volume, double domain_size, double lower, double upper,
                   bool toDensity)
{
   double weight = 0;
   double current_volume = 0;
   FunctionCoefficient dist([&domain_center, &ports](const Vector &x)
   {
      double d = infinity();
      for (int i=0; i<ports.Size(); i++) {d = std::min(d, DistanceToSegment(x, domain_center, *(ports[i])));}
      // double d = x.DistanceTo(domain_center);
      // for (int i=0; i<ports.Size(); i++) {d = std::min(d, x.DistanceTo(*(ports[i])));}
      return d;
   });
   psi.ProjectCoefficient(dist);
   double maxDist = psi.Max();
   MPI_Allreduce(MPI_IN_PLACE, &maxDist, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
   const double scale = upper - lower;
   psi.ApplyMap([maxDist, lower, upper](double x)
   {
      double d = lower + (upper - lower)*(1.0 - std::pow(x / maxDist, 0.3));
      return d;
   });
}


void uniformRefine(std::unique_ptr<Mesh>& mesh, int ser_ref_levels,
                   int par_ref_levels)
{
   for (int lev = 0; lev < ser_ref_levels; lev++)
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

void GetElasticityProblem(const ElasticityProblem problem,
                          double &filter_radius, double &vol_fraction,
                          std::unique_ptr<Mesh> &mesh, std::unique_ptr<VectorCoefficient> &vforce_cf,
                          Array2D<int> &ess_bdr, Array<int> &ess_bdr_filter,
                          std::string &prob_name, int ref_levels, int par_ref_levels)
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
         prob_name = "Cantilever";
         CantileverPreRefine(filter_radius, vol_fraction, mesh, ess_bdr, ess_bdr_filter,
                             vforce_cf);
         uniformRefine(mesh, ref_levels, par_ref_levels);
         CantileverPostRefine(ref_levels, par_ref_levels, mesh);
      } break;
      case ElasticityProblem::MBB:
      {
         prob_name = "MBB";
         MBBPreRefine(filter_radius, vol_fraction, mesh, ess_bdr, ess_bdr_filter,
                      vforce_cf);
         uniformRefine(mesh, ref_levels, par_ref_levels);
         MBBPostRefine(ref_levels, par_ref_levels, mesh);
      } break;
      case ElasticityProblem::Bridge:
      {
         prob_name = "Bridge";
         BridgePreRefine(filter_radius, vol_fraction, mesh, ess_bdr, ess_bdr_filter,
                         vforce_cf);
         uniformRefine(mesh, ref_levels, par_ref_levels);
         BridgePostRefine(ref_levels, par_ref_levels, mesh);
      } break;
      case ElasticityProblem::LBracket:
      {
         prob_name = "LBracket";
         LBracketPreRefine(filter_radius, vol_fraction, mesh, ess_bdr, ess_bdr_filter,
                           vforce_cf);
         uniformRefine(mesh, ref_levels, par_ref_levels);
         LBracketPostRefine(ref_levels, par_ref_levels, mesh);
      } break;
      case ElasticityProblem::Cantilever3:
      {
         prob_name = "Cantilever3";
         Cantilever3PreRefine(filter_radius, vol_fraction, mesh, ess_bdr, ess_bdr_filter,
                              vforce_cf);
         uniformRefine(mesh, ref_levels, par_ref_levels);
         Cantilever3PostRefine(ref_levels, par_ref_levels, mesh);
      } break;
      case ElasticityProblem::Torsion3:
      {
         prob_name = "Torsion3";
         Torsion3PreRefine(filter_radius, vol_fraction, mesh, ess_bdr, ess_bdr_filter,
                           vforce_cf);
         uniformRefine(mesh, ref_levels, par_ref_levels);
         Torsion3PostRefine(ref_levels, par_ref_levels, mesh);
      } break;
      case ElasticityProblem::MBB_selfloading:
      {
         prob_name = "MBB_selfloading";
         MBB_selfloadingPreRefine(filter_radius, vol_fraction, mesh, ess_bdr,
                                  ess_bdr_filter,
                                  vforce_cf);
         uniformRefine(mesh, ref_levels, par_ref_levels);
         MBB_selfloadingPostRefine(ref_levels, par_ref_levels, mesh);
      } break;
      case ElasticityProblem::Arch2:
      {
         prob_name = "Arch2";
         Arch2PreRefine(filter_radius, vol_fraction, mesh, ess_bdr, ess_bdr_filter,
                        vforce_cf);
         uniformRefine(mesh, ref_levels, par_ref_levels);
         Arch2PostRefine(ref_levels, par_ref_levels, mesh);
      } break;
      case ElasticityProblem::SelfLoading3:
      {
         prob_name = "SelfLoading3";
         SelfLoading3PreRefine(filter_radius, vol_fraction, mesh, ess_bdr,
                               ess_bdr_filter,
                               vforce_cf);
         uniformRefine(mesh, ref_levels, par_ref_levels);
         SelfLoading3PostRefine(ref_levels, par_ref_levels, mesh);
      } break;

      default:
         mfem_error("Undefined problem.");
   }
   mesh->SetAttributes();
}

void GetCompliantMechanismProblem(const CompliantMechanismProblem problem,
                                  double &filter_radius, double &vol_fraction,
                                  std::unique_ptr<Mesh> &mesh,
                                  double &k_in, double &k_out,
                                  Vector &d_in, Vector &d_out,
                                  std::unique_ptr<VectorCoefficient> &t_in,
                                  Array<int> &bdr_in, Array<int> &bdr_out,
                                  Array2D<int> &ess_bdr, Array<int> &ess_bdr_filter,
                                  std::string &prob_name, int ref_levels, int par_ref_levels)
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
      case CompliantMechanismProblem::ForceInverter:
      {
         prob_name = "ForceInverter";
         ForceInverterPreRefine(filter_radius, vol_fraction, mesh, ess_bdr,
                                ess_bdr_filter,
                                k_in, d_in, bdr_in, t_in, k_out, d_out, bdr_out);
         uniformRefine(mesh, ref_levels, par_ref_levels);
         ForceInverterPostRefine(ref_levels, par_ref_levels, mesh);
      } break;
      default:
         mfem_error("Undefined problem.");
         break;
   }
   mesh->SetAttributes();
}


void CantileverPreRefine(double &filter_radius, double &vol_fraction,
                         std::unique_ptr<Mesh> &mesh, Array2D<int> &ess_bdr, Array<int> &ess_bdr_filter,
                         std::unique_ptr<VectorCoefficient> &vforce_cf)
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
}
void CantileverPostRefine(int ser_ref_levels, int par_ref_levels,
                          std::unique_ptr<Mesh> &mesh) {}

void MBBPreRefine(double &filter_radius, double &vol_fraction,
                  std::unique_ptr<Mesh> &mesh, Array2D<int> &ess_bdr, Array<int> &ess_bdr_filter,
                  std::unique_ptr<VectorCoefficient> &vforce_cf)
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
}
void MBBPostRefine(int ser_ref_levels, int par_ref_levels,
                   std::unique_ptr<Mesh> &mesh)
{
   mesh->MarkBoundary([](const Vector &x) {return ((x(0) > (3 - std::pow(2, -5))) && (x(1) < 1e-10)); },
   5);
   mesh->SetAttributes();
}

void BridgePreRefine(double &filter_radius, double &vol_fraction,
                     std::unique_ptr<Mesh> &mesh, Array2D<int> &ess_bdr, Array<int> &ess_bdr_filter,
                     std::unique_ptr<VectorCoefficient> &vforce_cf)
{
   if (filter_radius < 0) { filter_radius = 0.05; }
   if (vol_fraction < 0) { vol_fraction = 0.3; }

   *mesh = Mesh::MakeCartesian2D(2, 1, mfem::Element::Type::QUADRILATERAL, true,
                                 2.0, 1.0);
   ess_bdr.SetSize(3, 5);
   ess_bdr_filter.SetSize(5);
   ess_bdr = 0; ess_bdr_filter = 0;
   ess_bdr(1, 3) = 1; // left : y-roller -> x fixed
   ess_bdr(0, 4) = 1; // right-bottom : pin support
   ess_bdr_filter[2] = 1;
   vforce_cf.reset(new VectorFunctionCoefficient(2, [](const Vector &x,
                                                       Vector &f)
   {
      f = 0.0;
      if (x[1] > 1.0 - std::pow(2, -5.0)) { f(1) = -1.0; }
   }));
}
void BridgePostRefine(int ser_ref_levels, int par_ref_levels,
                      std::unique_ptr<Mesh> &mesh)
{
   double h = std::pow(0.5,
                       ser_ref_levels + (par_ref_levels < 0 ? 0 : par_ref_levels));
   mesh->MarkBoundary([h](const Vector &x) {return ((x(0) > (2.0 - h)) && (x(1) < 1e-10)); },
   5);
   mesh->SetAttributes();
}

void LBracketPreRefine(double &filter_radius, double &vol_fraction,
                       std::unique_ptr<Mesh> &mesh, Array2D<int> &ess_bdr, Array<int> &ess_bdr_filter,
                       std::unique_ptr<VectorCoefficient> &vforce_cf)
{
   if (filter_radius < 0) { filter_radius = 5e-02; }
   if (vol_fraction < 0) { vol_fraction = 0.5; }

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
}
void LBracketPostRefine(int ser_ref_levels, int par_ref_levels,
                        std::unique_ptr<Mesh> &mesh) {}

void Cantilever3PreRefine(double &filter_radius, double &vol_fraction,
                          std::unique_ptr<Mesh> &mesh, Array2D<int> &ess_bdr, Array<int> &ess_bdr_filter,
                          std::unique_ptr<VectorCoefficient> &vforce_cf)
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
}
void Cantilever3PostRefine(int ser_ref_levels, int par_ref_levels,
                           std::unique_ptr<Mesh> &mesh) {}

void Torsion3PreRefine(double &filter_radius, double &vol_fraction,
                       std::unique_ptr<Mesh> &mesh, Array2D<int> &ess_bdr, Array<int> &ess_bdr_filter,
                       std::unique_ptr<VectorCoefficient> &vforce_cf)
{
   if (filter_radius < 0) { filter_radius = 0.05; }
   if (vol_fraction < 0) { vol_fraction = 0.1; }

   // [1: bottom, 2: front, 3: right, 4: back, 5: left, 6: top]
   *mesh = Mesh::MakeCartesian3D(6, 5, 5, mfem::Element::Type::HEXAHEDRON, 1.2,
                                 1.0,
                                 1.0);
   ess_bdr.SetSize(4, 8);
   ess_bdr_filter.SetSize(8);
   ess_bdr = 0; ess_bdr_filter = 0;
   ess_bdr(0, 6) = 1;
   ess_bdr_filter = -1; // all boundaries void
   ess_bdr_filter[6] = 0; // left circle is free
   ess_bdr_filter[7] = 0; // right circle is free
   const Vector center({0.0, 0.5, 0.5});
   vforce_cf.reset(new VectorFunctionCoefficient(3, [center](const Vector &x,
                                                             Vector &f)
   {
      if (x[0] > 1.0)
      {
         f[0] = 0.0;
         f[1] = -x[2];
         f[2] = x[1];
      }
      else
      {
         f = 0.0;
      }
   }));
}
void Torsion3PostRefine(int ser_ref_levels, int par_ref_levels,
                        std::unique_ptr<Mesh> &mesh)
{
   // left center: Dirichlet
   Vector center({0.0, 0.5, 0.5});
   mesh->MarkBoundary([center](const Vector &x) { return (center.DistanceTo(x) < 0.2); },
   7);
   // Right center: Torsion
   center[0] = 1.2;
   mesh->MarkBoundary([center](const Vector &x) { return (center.DistanceTo(x) < 0.2); },
   8);
   mesh->SetAttributes();
}

void MBB_selfloadingPreRefine(double &filter_radius, double &vol_fraction,
                              std::unique_ptr<Mesh> &mesh, Array2D<int> &ess_bdr, Array<int> &ess_bdr_filter,
                              std::unique_ptr<VectorCoefficient> &vforce_cf)
{
   if (filter_radius < 0) { filter_radius = 5e-02; }
   if (vol_fraction < 0) { vol_fraction = 0.1; }

   *mesh = Mesh::MakeCartesian2D(2, 1, mfem::Element::Type::QUADRILATERAL, true,
                                 2.0,
                                 1.0);
   ess_bdr.SetSize(3, 5);
   ess_bdr_filter.SetSize(5);
   ess_bdr = 0; ess_bdr_filter = 0;
   ess_bdr(1, 3) = 1; // left : y-roller -> x fixed
   ess_bdr(2, 4) = 1; // right-bottom : x-roller -> y fixed
   const Vector zero({0.0, 0.0});
   vforce_cf.reset(new VectorConstantCoefficient(zero));
}
void MBB_selfloadingPostRefine(int ser_ref_levels, int par_ref_levels,
                               std::unique_ptr<Mesh> &mesh)
{
   mesh->MarkBoundary([](const Vector &x) {return ((x(0) > (2 - std::pow(2, -5))) && (x(1) < 1e-10)); },
   5);
   mesh->SetAttributes();
}

void Arch2PreRefine(double &filter_radius, double &vol_fraction,
                    std::unique_ptr<Mesh> &mesh, Array2D<int> &ess_bdr, Array<int> &ess_bdr_filter,
                    std::unique_ptr<VectorCoefficient> &vforce_cf)
{

   if (filter_radius < 0) { filter_radius = 5e-02; }
   if (vol_fraction < 0) { vol_fraction = 0.1; }

   *mesh = Mesh::MakeCartesian2D(1, 1, mfem::Element::Type::QUADRILATERAL, true,
                                 1.0,
                                 1.0);
   ess_bdr.SetSize(3, 5);
   ess_bdr_filter.SetSize(5);
   ess_bdr = 0; ess_bdr_filter = 0;
   ess_bdr(1, 3) = 1; // left : y-roller -> x fixed
   ess_bdr(0, 4) = 1; // right-bottom : pin
   const Vector zero({0.0, 0.0});
   vforce_cf.reset(new VectorConstantCoefficient(zero));
}
void Arch2PostRefine(int ser_ref_levels, int par_ref_levels,
                     std::unique_ptr<Mesh> &mesh)
{
   mesh->MarkBoundary([](const Vector &x) {return ((x(0) > (1 - std::pow(2, -5))) && (x(1) < 1e-10)); },
   5);
   mesh->SetAttributes();
}

void SelfLoading3PreRefine(double &filter_radius, double &vol_fraction,
                           std::unique_ptr<Mesh> &mesh, Array2D<int> &ess_bdr, Array<int> &ess_bdr_filter,
                           std::unique_ptr<VectorCoefficient> &vforce_cf)
{
   if (filter_radius < 0) { filter_radius = 3e-02; }
   if (vol_fraction < 0) { vol_fraction = 0.07; }

   // [1: bottom,
   //  2: front,
   //  3: right,
   //  4: back,
   //  5: left,
   //  6: top
   //  7: (1,1,0)]
   *mesh = Mesh::MakeCartesian3D(2, 2, 1, mfem::Element::Type::HEXAHEDRON,
                                 2.0, 2.0, 1.0, false);
   ess_bdr.SetSize(4, 7);
   ess_bdr_filter.SetSize(7);
   ess_bdr = 0;
   ess_bdr(2, 2 - 1) = 1;// front - xz-roller plane
   ess_bdr(1, 5 - 1) = 1;// left - yz-roller plane
   ess_bdr(0, 6) = 1;// corner - pin
   ess_bdr_filter = 0;

   const Vector zero({0.0, 0.0, 0.0});
   vforce_cf.reset(new VectorConstantCoefficient(zero));
}
void SelfLoading3PostRefine(int ser_ref_levels, int par_ref_levels,
                            std::unique_ptr<Mesh> &mesh)
{
   // left center: Dirichlet
   const double lv = ser_ref_levels + std::max((double)par_ref_levels, 0.0);
   mesh->MarkBoundary([lv](const Vector &x) { return (x[0] > 2.0 - std::pow(0.5, lv)) && (x[1] > 2.0 - std::pow(0.5, lv)) && (x[2] < 1e-10); },
   7);
   mesh->SetAttributes();
}


void ForceInverterPreRefine(double &filter_radius, double &vol_fraction,
                            std::unique_ptr<Mesh> &mesh, Array2D<int> &ess_bdr, Array<int> &ess_bdr_filter,
                            double &k_in, Vector &d_in, Array<int> &bdr_in,
                            std::unique_ptr<VectorCoefficient> &t_in,
                            double &k_out, Vector &d_out, Array<int> &bdr_out)
{

   if (filter_radius < 0) { filter_radius = 2.5e-02; }
   if (vol_fraction < 0) { vol_fraction = 0.3; }

   *mesh = Mesh::MakeCartesian2D(2, 1, mfem::Element::Type::QUADRILATERAL, true,
                                 2.0,
                                 1.0);
   //                        X-Roller (3)
   //               ---------------------------------
   //  INPUT (6) -> |                               | <- output (5)
   //               -                               -
   //               |                               |
   //               |                               |
   //               -                               |
   //  FIXED (7)  X |                               |
   //               ---------------------------------
   ess_bdr.SetSize(3, 7);
   ess_bdr_filter.SetSize(7);
   ess_bdr = 0; ess_bdr_filter = 0;
   ess_bdr(2, 3 - 1) = 1; // Top - x-roller -> y direction fixed
   ess_bdr(0, 7 - 1) = 1; // Left Bottom - Fixed
   bdr_in.SetSize(7); bdr_out.SetSize(7);
   bdr_in = 0; bdr_out = 0;
   bdr_in[6 - 1] = 1;
   bdr_out[5 - 1] = 1;
   // ess_bdr_filter[5] = 1; ess_bdr_filter[6] = 1; ess_bdr_filter[7] = 1;


   k_in = 10.0; k_out = 10.0;
   Vector traction(2); traction[0] = 10.0; traction[1] = 0.0;
   t_in.reset(new VectorConstantCoefficient(traction));
   d_out.SetSize(2); d_out[0] = -1.0; d_out[1] = 0.0;
   d_in.SetSize(2); d_in[0] = -1.0; d_in[1] = 0.0;
}

void ForceInverterPostRefine(int ref_levels, int par_ref_levels,
                             std::unique_ptr<Mesh> &mesh)
{
   double h = std::pow(2.0, -(ref_levels + std::max(0.0, 0.0 + par_ref_levels)));
   mesh->MarkBoundary([h](const Vector &x)
   {
      return (x[0] > 2.0 - 0.1*h) && (x[1] > 1.0 - h); // Output - Right, Top h
   }, 5);
   mesh->MarkBoundary([h](const Vector &x)
   {
      return (x[0] < 0.0 + 0.1*h) && (x[1] > 1.0 - h); // Input - Left, Top h
   }, 6);
   mesh->MarkBoundary([h](const Vector &x)
   {
      return (x[0] < 0.0 + 0.1*h) && (x[1] < 0.0 + h); // Fixed - Left, Bottom h
   }, 7);
}
}
