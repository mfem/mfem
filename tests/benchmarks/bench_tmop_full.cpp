// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "bench.hpp"

#ifdef MFEM_USE_BENCHMARK

#include "mfem.hpp"
#include <iostream>
#include <fstream>
#include "../../miniapps/meshing/mesh-optimizer.hpp"

using namespace mfem;
using namespace std;

////////////////////////////////////////////////////////////////////////////////
static MPI_Session *mpi = nullptr;
static int config_dev_size = 4; // default 4 GPU per node

////////////////////////////////////////////////////////////////////////////////
struct TMOP_PMESH_OPTIMIZER
{
   int num_procs, myid;
   bool mpi_done = false;
   const int mesh_poly_deg;                   // -o 1
   const int rs_levels         = 0;           // -rs 1
   const int rp_levels         = 0;
   const double jitter         = 0.0;
   const int metric_id         = 303;         // -mid 303
   const int target_id         = 1;           // -tid 1
   const double lim_const      = 0.0;
   const double adapt_lim_const   = 0.0;
   const double surface_fit_const = 0.0;
   const int quad_type         = 1;           // -qt 1
   const int quad_order;                     // -qo 0
   int hexahedron_quad_points  = 0;
   const int solver_type       = 0;           // -st 0
   const int solver_iter       = 1;           // -ni 1
   const double solver_rtol    = 1e-10;
   const int solver_art_type   = 0;           // -art 0
   const int lin_solver        = 2;           // -ls 2
   const int max_lin_iter      = 20;          // -li 20
   const bool move_bnd         = true;        // -bnd
   const int combomet          = 0;
   const bool hradaptivity     = false;
   int h_metric_id             = -1;
   const bool normalization    = false;
   const bool visualization    = false;
   const int verbosity_level   = 0;           // -vl 2
   const bool fdscheme         = false;
   const int adapt_eval        = 0;
   const bool exactaction      = false;
   //const char *devopt          = "cpu:fast";  // -d cpu:fast
   const bool pa               = true;        // -pa
   const int n_hr_iter         = 5;
   const int n_h_iter          = 1;
   const bool benchmark        = true;        // -bm
   const int  benchmarkid      = 2;           // -bm_id 2
   const double ls_scale       = 1e-7;        // -scale 1.e-7

   Vector b;
   const int p, c, n, nx, ny, nz, dim = 3;
   const bool check_x, check_y, check_z, checked;
   std::function<ParMesh*()> GetParMesh = [&]()
   {
      Mesh smesh = Mesh::MakeCartesian3D(nx,ny,nz, Element::HEXAHEDRON);
      ParMesh *par_mesh = new ParMesh(MPI_COMM_WORLD, smesh);
      return par_mesh;
   };
   ParMesh *pmesh = nullptr;
   ParNonlinearForm *a = nullptr;
   FiniteElementCollection *fec = nullptr;
   ParFiniteElementSpace *pfespace = nullptr;
   int ndofs;
   double mdof;
   ConstantCoefficient *lim_coeff = nullptr;
   ConstantCoefficient *coef_zeta = nullptr;
   ConstantCoefficient *coef_ls = nullptr;

   TMOP_QualityMetric *metric = nullptr;
   ConstantCoefficient *coeff1 = nullptr;
   TMOP_QualityMetric *metric2 = nullptr;
   TargetConstructor *target_c2 = nullptr;
   TMOP_Integrator *he_nlf_integ = nullptr;
   Solver *S = nullptr, *S_prec = nullptr;
   AdaptivityEvaluator *adapt_evaluator = nullptr;
   AdaptivityEvaluator *adapt_surface = nullptr;
   TargetConstructor *target_c = nullptr;
   HessianCoefficient *adapt_coeff = nullptr;
   HRHessianCoefficient *hr_adapt_coeff = nullptr;
   TMOP_QualityMetric *h_metric = nullptr;

   StopWatch TimeSolver;
   ParGridFunction *x = nullptr;
   TMOPNewtonSolver *solver = nullptr;

   double init_energy, init_metric_energy;

   TMOP_PMESH_OPTIMIZER(int mesh_poly_deg, int side):
      // 0. Initialize MPI.
      mpi_done((MPI_Comm_size(MPI_COMM_WORLD, &num_procs),
                MPI_Comm_rank(MPI_COMM_WORLD, &myid), true)),
      mesh_poly_deg(mesh_poly_deg),
      quad_order(mesh_poly_deg * 2), // Used for all tests
      b(0), // Set up an empty right-hand side vector b, which is equivalent to b=0.
      p(mesh_poly_deg),
      c(side),
      n((assert(c>=p),c/p)),
      nx(n + (p*(n+1)*p*n*p*n < c*c*c ?1:0)),
      ny(n + (p*(n+1)*p*(n+1)*p*n < c*c*c ?1:0)),
      nz(n),
      check_x(p*nx * p*ny * p*nz <= c*c*c),
      check_y(p*(nx+1) * p*(ny+1) * p*nz > c*c*c),
      check_z(p*(nx+1) * p*(ny+1) * p*(nz+1) > c*c*c),
      checked((assert(check_x && check_y && check_z), true)),
      pmesh(GetParMesh()),
      mdof(0.0)
   {
      if (h_metric_id < 0) { h_metric_id = metric_id; }

      assert(!hradaptivity);
      //assert(num_procs == 1);

      // 4. Define a finite element space on the mesh. Here we use vector finite
      //    elements which are tensor products of quadratic finite elements. The
      //    number of components in the vector finite element space is specified by
      //    the last parameter of the FiniteElementSpace constructor.
      assert(mesh_poly_deg > 0);
      fec = new H1_FECollection(mesh_poly_deg, dim);
      pfespace = new ParFiniteElementSpace(pmesh, fec, dim);
      ndofs = pfespace->GlobalTrueVSize();

      // 5. Make the mesh curved based on the above finite element space. This
      //    means that we define the mesh elements through a fespace-based
      //    transformation of the reference element.
      pmesh->SetNodalFESpace(pfespace);

      // 7. Get the mesh nodes (vertices and other degrees of freedom in the finite
      //    element space) as a finite element grid function in fespace. Note that
      //    changing x automatically changes the shapes of the mesh elements.
      x = new ParGridFunction(pfespace);
      pmesh->SetNodalGridFunction(x);

      // 8. Define a vector representing the minimal local mesh size in the mesh
      //    nodes. We index the nodes using the scalar version of the degrees of
      //    freedom in pfespace. Note: this is partition-dependent.
      //
      //    In addition, compute average mesh size and total volume.
      Vector h0(pfespace->GetNDofs());
      h0 = infinity();
      double vol_loc = 0.0;
      Array<int> edofs;
      for (int i = 0; i < pmesh->GetNE(); i++)
      {
         // Get the local scalar element degrees of freedom in dofs.
         pfespace->GetElementDofs(i, edofs);
         // Adjust the value of h0 in dofs based on the local mesh size.
         const double hi = pmesh->GetElementSize(i);
         for (int j = 0; j < edofs.Size(); j++)
         {
            h0(edofs[j]) = min(h0(edofs[j]), hi);
         }
         vol_loc += pmesh->GetElementVolume(i);
      }
      double global_volume;
      MPI_Allreduce(&vol_loc, &global_volume, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      const double small_phys_size = pow(global_volume, 1.0 / dim) / 100.0;

      // 9. Add a random perturbation to the nodes in the interior of the domain.
      //    We define a random grid function of fespace and make sure that it is
      //    zero on the boundary and its values are locally of the order of h0.
      //    The latter is based on the DofToVDof() method which maps the scalar to
      //    the vector degrees of freedom in pfespace.
      ParGridFunction rdm(pfespace);
      rdm.Randomize();
      rdm -= 0.25; // Shift to random values in [-0.5,0.5].
      rdm *= jitter;
      rdm.HostReadWrite();
      // Scale the random values to be of order of the local mesh size.
      for (int i = 0; i < pfespace->GetNDofs(); i++)
      {
         for (int d = 0; d < dim; d++)
         {
            rdm(pfespace->DofToVDof(i,d)) *= h0(i);
         }
      }
      Array<int> be_vdofs;
      for (int i = 0; i < pfespace->GetNBE(); i++)
      {
         // Get the vector degrees of freedom in the boundary element.
         pfespace->GetBdrElementVDofs(i, be_vdofs);
         // Set the boundary values to zero.
         for (int j = 0; j < be_vdofs.Size(); j++) { rdm(be_vdofs[j]) = 0.0; }
      }
      *x -= rdm;
      // Set the perturbation of all nodes from the true nodes.
      x->SetTrueVector();
      x->SetFromTrueVector();

      x->HostReadWrite();
      // Add benchmark transformation
      if (benchmark)
      {
         for (int i = 0; i < pfespace->GetNDofs(); i++)
         {
            Array<double> xc(dim), xn(dim);
            for (int d = 0; d < dim; d++)
            {
               xc[d] = x->operator()(pfespace->DofToVDof(i,d));
            }
            double epsy = 0.3,
                   epsz = 0.3;

            if (benchmarkid == 1)
            {
               kershaw(epsy, epsz, xc[0], xc[1], xc[dim-1], xn[0], xn[1], xn[dim-1]);
            }
            else if (benchmarkid == 2)
            {
               if (dim == 2)
               {
                  stretching2D(xc[0], xc[1], xn[0], xn[1]);
               }
               else if (dim == 3)
               {
                  stretching3D(xc[0], xc[1], xc[dim-1], xn[0], xn[1], xn[dim-1]);
               }
            }
            else if (benchmarkid == 3)
            {
               rotation2D(xc[0], xc[1], xn[0], xn[1]);
               if (dim == 3) { xn[2] = xc[2]; }
            }
            else if (benchmarkid == 4)
            {
               kershaw8(epsy, epsz, xc[0], xc[1], xc[dim-1], xn[0], xn[1], xn[dim-1]);
            }

            for (int d = 0; d < dim; d++)
            {
               x->operator()(pfespace->DofToVDof(i,d)) = xn[d];
            }
         }
         x->SetTrueVector();
         x->SetFromTrueVector();
      }

      x->HostReadWrite();
      // Change boundary attribute for boundary element if tangential relaxation is allowed
      if (move_bnd && benchmark)
      {
         for (int e = 0; e < pmesh->GetNBE(); e++)
         {
            Array<int> be_dofs;
            pfespace->GetBdrElementDofs(e, be_dofs);
            Array<double> x_c(dim);
            Array<int> nnodes(dim);
            nnodes = 0;
            double tolerance = 1.e-6;
            for (int j = 0; j < be_dofs.Size(); j++)
            {
               if (j == 0)
               {
                  for (int d = 0; d < dim; d++)
                  {
                     x_c[d] = x->operator()(pfespace->DofToVDof(be_dofs[j], d));
                     nnodes[d]++;
                  }
               }
               else
               {
                  for (int d = 0; d < dim; d++)
                  {
                     if (abs(x_c[d] - x->operator()(pfespace->DofToVDof(be_dofs[j],d))) < tolerance)
                     {
                        nnodes[d]++;
                     }
                  }
               }
            }
            Element *be = pmesh->GetBdrElement(e);
            be->SetAttribute(4);
            for (int d = 0; d < dim; d++)
            {
               if (nnodes[d] == be_dofs.Size())
               {
                  be->SetAttribute(d+1);
               }
            }
         }
      }
      pmesh->SetAttributes();

      // 11. Store the starting (prior to the optimization) positions.
      ParGridFunction x0(pfespace);
      x0 = *x;

      // 12. Form the integrator that uses the chosen metric and target.
      double tauval = -0.1;
      switch (metric_id)
      {
         // T-metrics
         case 1: metric = new TMOP_Metric_001; break;
         case 2: metric = new TMOP_Metric_002; break;
         case 7: metric = new TMOP_Metric_007; break;
         case 9: metric = new TMOP_Metric_009; break;
         case 14: metric = new TMOP_Metric_014; break;
         case 22: metric = new TMOP_Metric_022(tauval); break;
         case 50: metric = new TMOP_Metric_050; break;
         case 55: metric = new TMOP_Metric_055; break;
         case 56: metric = new TMOP_Metric_056; break;
         case 58: metric = new TMOP_Metric_058; break;
         case 77: metric = new TMOP_Metric_077; break;
         case 80: metric = new TMOP_Metric_080(0.5); break;
         case 85: metric = new TMOP_Metric_085; break;
         case 98: metric = new TMOP_Metric_098; break;
         // case 211: metric = new TMOP_Metric_211; break;
         // case 252: metric = new TMOP_Metric_252(tauval); break;
         case 301: metric = new TMOP_Metric_301; break;
         case 302: metric = new TMOP_Metric_302; break;
         case 303: metric = new TMOP_Metric_303; break;
         // case 311: metric = new TMOP_Metric_311; break;
         case 313: metric = new TMOP_Metric_313(tauval); break;
         case 315: metric = new TMOP_Metric_315; break;
         case 316: metric = new TMOP_Metric_316; break;
         case 321: metric = new TMOP_Metric_321; break;
         // case 352: metric = new TMOP_Metric_352(tauval); break;
         // A-metrics
         case 11: metric = new TMOP_AMetric_011; break;
         case 36: metric = new TMOP_AMetric_036; break;
         case 107: metric = new TMOP_AMetric_107a; break;
         case 126: metric = new TMOP_AMetric_126(0.9); break;
         default:
            if (myid == 0) { cout << "Unknown metric_id: " << metric_id << endl; }
            assert(false);
            return;
      }

      if (hradaptivity)
      {
         switch (h_metric_id)
         {
            case 1: h_metric = new TMOP_Metric_001; break;
            case 2: h_metric = new TMOP_Metric_002; break;
            case 7: h_metric = new TMOP_Metric_007; break;
            case 9: h_metric = new TMOP_Metric_009; break;
            case 55: h_metric = new TMOP_Metric_055; break;
            case 56: h_metric = new TMOP_Metric_056; break;
            case 58: h_metric = new TMOP_Metric_058; break;
            case 77: h_metric = new TMOP_Metric_077; break;
            case 315: h_metric = new TMOP_Metric_315; break;
            case 316: h_metric = new TMOP_Metric_316; break;
            case 321: h_metric = new TMOP_Metric_321; break;
            default: std::cout << "Metric_id not supported for h-adaptivity: "
                                  << h_metric_id
                                  << std::endl;
               assert(false);
               return;
         }
      }

      if (metric_id < 300 || h_metric_id < 300)
      {
         MFEM_VERIFY(dim == 2, "Incompatible metric for 3D meshes");
      }
      if (metric_id >= 300 || h_metric_id >= 300)
      {
         MFEM_VERIFY(dim == 3, "Incompatible metric for 2D meshes");
      }

      TargetConstructor::TargetType target_t;
      H1_FECollection ind_fec(mesh_poly_deg, dim);
      ParFiniteElementSpace ind_fes(pmesh, &ind_fec);
      ParFiniteElementSpace ind_fesv(pmesh, &ind_fec, dim);
      ParGridFunction size(&ind_fes), aspr(&ind_fes), disc(&ind_fes), ori(&ind_fes);
      ParGridFunction aspr3d(&ind_fesv);

      const AssemblyLevel al =
         pa ? AssemblyLevel::PARTIAL : AssemblyLevel::LEGACY;

      switch (target_id)
      {
         case 1: target_t = TargetConstructor::IDEAL_SHAPE_UNIT_SIZE; break;
         case 2: target_t = TargetConstructor::IDEAL_SHAPE_EQUAL_SIZE; break;
         case 3: target_t = TargetConstructor::IDEAL_SHAPE_GIVEN_SIZE; break;
         case 4:
         {
            target_t = TargetConstructor::GIVEN_FULL;
            AnalyticAdaptTC *tc = new AnalyticAdaptTC(target_t);
            adapt_coeff = new HessianCoefficient(dim, metric_id);
            tc->SetAnalyticTargetSpec(NULL, NULL, adapt_coeff);
            target_c = tc;
            break;
         }
         case 5: // Discrete size 2D or 3D
         {
            target_t = TargetConstructor::IDEAL_SHAPE_GIVEN_SIZE;
            DiscreteAdaptTC *tc = new DiscreteAdaptTC(target_t);
            if (adapt_eval == 0)
            {
               tc->SetAdaptivityEvaluator(new AdvectorCG(al));
            }
            else
            {
#ifdef MFEM_USE_GSLIB
               tc->SetAdaptivityEvaluator(new InterpolatorFP);
#else
               MFEM_ABORT("MFEM is not built with GSLIB.");
#endif
            }
            if (dim == 2)
            {
               //FunctionCoefficient ind_coeff(discrete_size_2d);
               DiscreteSize2D ind_coeff(rs_levels);
               size.ProjectCoefficient(ind_coeff);
            }
            else if (dim == 3)
            {
               //FunctionCoefficient ind_coeff(discrete_size_3d);
               DiscreteSize3D ind_coeff(rs_levels);
               size.ProjectCoefficient(ind_coeff);
            }
            tc->SetParDiscreteTargetSize(size);
            target_c = tc;
            break;
         }
         case 6: // material indicator 2D
         {
            ParGridFunction d_x(&ind_fes), d_y(&ind_fes);

            target_t = TargetConstructor::GIVEN_SHAPE_AND_SIZE;
            DiscreteAdaptTC *tc = new DiscreteAdaptTC(target_t);
            FunctionCoefficient ind_coeff(material_indicator_2d);
            disc.ProjectCoefficient(ind_coeff);
            if (adapt_eval == 0)
            {
               tc->SetAdaptivityEvaluator(new AdvectorCG(al));
            }
            else
            {
#ifdef MFEM_USE_GSLIB
               tc->SetAdaptivityEvaluator(new InterpolatorFP);
#else
               MFEM_ABORT("MFEM is not built with GSLIB.");
#endif
            }
            // Diffuse the interface
            DiffuseField(disc,2);

            // Get  partials with respect to x and y of the grid function
            disc.GetDerivative(1,0,d_x);
            disc.GetDerivative(1,1,d_y);

            // Compute the squared magnitude of the gradient
            for (int i = 0; i < size.Size(); i++)
            {
               size(i) = std::pow(d_x(i),2)+std::pow(d_y(i),2);
            }
            const double max = size.Max();
            double max_all;
            MPI_Allreduce(&max, &max_all, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

            for (int i = 0; i < d_x.Size(); i++)
            {
               d_x(i) = std::abs(d_x(i));
               d_y(i) = std::abs(d_y(i));
            }
            const double eps = 0.01;
            const double aspr_ratio = 20.0;
            const double size_ratio = 40.0;

            for (int i = 0; i < size.Size(); i++)
            {
               size(i) = (size(i)/max_all);
               aspr(i) = (d_x(i)+eps)/(d_y(i)+eps);
               aspr(i) = 0.1 + 0.9*(1-size(i))*(1-size(i));
               if (aspr(i) > aspr_ratio) {aspr(i) = aspr_ratio;}
               if (aspr(i) < 1.0/aspr_ratio) {aspr(i) = 1.0/aspr_ratio;}
            }
            Vector vals;
            const int NE = pmesh->GetNE();
            double volume = 0.0, volume_ind = 0.0;

            for (int i = 0; i < NE; i++)
            {
               ElementTransformation *Tr = pmesh->GetElementTransformation(i);
               const IntegrationRule &ir =
                  IntRules.Get(pmesh->GetElementBaseGeometry(i), Tr->OrderJ());
               size.GetValues(i, ir, vals);
               for (int j = 0; j < ir.GetNPoints(); j++)
               {
                  const IntegrationPoint &ip = ir.IntPoint(j);
                  Tr->SetIntPoint(&ip);
                  volume     += ip.weight * Tr->Weight();
                  volume_ind += vals(j) * ip.weight * Tr->Weight();
               }
            }
            double volume_all, volume_ind_all;
            MPI_Allreduce(&volume, &volume_all, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(&volume_ind, &volume_ind_all, 1, MPI_DOUBLE, MPI_SUM,
                          MPI_COMM_WORLD);
            const int NE_ALL = pmesh->GetGlobalNE();

            const double avg_zone_size = volume_all / NE_ALL;

            const double small_avg_ratio =
               (volume_ind_all + (volume_all - volume_ind_all) / size_ratio)
               / volume_all;

            const double small_zone_size = small_avg_ratio * avg_zone_size;
            const double big_zone_size   = size_ratio * small_zone_size;

            for (int i = 0; i < size.Size(); i++)
            {
               const double val = size(i);
               const double alpha = (big_zone_size - small_zone_size) / small_zone_size;
               size(i) = big_zone_size / (1.0+alpha*val);
            }

            DiffuseField(size, 2);
            DiffuseField(aspr, 2);

            tc->SetParDiscreteTargetSize(size);
            tc->SetParDiscreteTargetAspectRatio(aspr);
            target_c = tc;
            break;
         }
         case 7: // Discrete aspect ratio 3D
         {
            target_t = TargetConstructor::GIVEN_SHAPE_AND_SIZE;
            DiscreteAdaptTC *tc = new DiscreteAdaptTC(target_t);
            if (adapt_eval == 0)
            {
               tc->SetAdaptivityEvaluator(new AdvectorCG(al));
            }
            else
            {
#ifdef MFEM_USE_GSLIB
               tc->SetAdaptivityEvaluator(new InterpolatorFP);
#else
               MFEM_ABORT("MFEM is not built with GSLIB.");
#endif
            }
            VectorFunctionCoefficient fd_aspr3d(dim, discrete_aspr_3d);
            aspr3d.ProjectCoefficient(fd_aspr3d);
            tc->SetParDiscreteTargetAspectRatio(aspr3d);
            target_c = tc;
            break;
         }
         case 8: // shape/size + orientation 2D
         {
            target_t = TargetConstructor::GIVEN_SHAPE_AND_SIZE;
            DiscreteAdaptTC *tc = new DiscreteAdaptTC(target_t);
            if (adapt_eval == 0)
            {
               tc->SetAdaptivityEvaluator(new AdvectorCG(al));
            }
            else
            {
#ifdef MFEM_USE_GSLIB
               tc->SetAdaptivityEvaluator(new InterpolatorFP);
#else
               MFEM_ABORT("MFEM is not built with GSLIB.");
#endif
            }

            if (metric_id == 14 || metric_id == 36)
            {
               ConstantCoefficient ind_coeff(0.1*0.1);
               size.ProjectCoefficient(ind_coeff);
               tc->SetParDiscreteTargetSize(size);
            }

            if (metric_id == 85)
            {
               FunctionCoefficient aspr_coeff(discrete_aspr_2d);
               aspr.ProjectCoefficient(aspr_coeff);
               DiffuseField(aspr,2);
               tc->SetParDiscreteTargetAspectRatio(aspr);
            }

            FunctionCoefficient ori_coeff(discrete_ori_2d);
            ori.ProjectCoefficient(ori_coeff);
            tc->SetParDiscreteTargetOrientation(ori);
            target_c = tc;
            break;
         }
         // Targets used for hr-adaptivity tests.
         case 9:  // size target in an annular region.
         case 10: // size+aspect-ratio in an annular region.
         case 11: // size+aspect-ratio target for a rotate sine wave
         {
            target_t = TargetConstructor::GIVEN_FULL;
            AnalyticAdaptTC *tc = new AnalyticAdaptTC(target_t);
            hr_adapt_coeff = new HRHessianCoefficient(dim, target_id - 9);
            tc->SetAnalyticTargetSpec(NULL, NULL, hr_adapt_coeff);
            target_c = tc;
            break;
         }
         default:
            if (myid == 0) { cout << "Unknown target_id: " << target_id << endl; }
            assert(false);
            return;
      }

      if (target_c == NULL)
      {
         target_c = new TargetConstructor(target_t, MPI_COMM_WORLD);
      }
      target_c->SetNodes(x0);
      he_nlf_integ = new TMOP_Integrator(metric, target_c,
                                         h_metric);

      // Finite differences for computations of derivatives.
      if (fdscheme)
      {
         MFEM_VERIFY(pa == false, "PA for finite differences is not implemented.");
         he_nlf_integ->EnableFiniteDifferences(*x);
      }
      he_nlf_integ->SetExactActionFlag(exactaction);

      // Setup the quadrature rules for the TMOP integrator.
      IntegrationRules *irules = NULL;
      switch (quad_type)
      {
         case 1: irules = &IntRulesLo; break;
         case 2: irules = &IntRules; break;
         case 3: irules = &IntRulesCU; break;
         default:
            if (myid == 0) { cout << "Unknown quad_type: " << quad_type << endl; }
            assert(false);
            return;
      }
      he_nlf_integ->SetIntegrationRules(*irules, quad_order);
      if (myid == 0 && dim == 2)
      {
         cout << "Triangle quadrature points: "
              << irules->Get(Geometry::TRIANGLE, quad_order).GetNPoints()
              << "\nQuadrilateral quadrature points: "
              << irules->Get(Geometry::SQUARE, quad_order).GetNPoints() << endl;
      }
      hexahedron_quad_points =
         irules->Get(Geometry::CUBE, quad_order).GetNPoints();
      /*if (myid == 0 && dim == 3)
      {
         cout << "Hexahedron quadrature points: "
              << hexahedron_quad_points << endl;
      }*/

      // Limit the node movement.
      // The limiting distances can be given by a general function of space.
      ParFiniteElementSpace dist_pfespace(pmesh, fec); // scalar space
      ParGridFunction dist(&dist_pfespace);
      dist = 1.0;
      // The small_phys_size is relevant only with proper normalization.
      if (normalization) { dist = small_phys_size; }
      lim_coeff = new ConstantCoefficient(lim_const);
      if (lim_const != 0.0) { he_nlf_integ->EnableLimiting(x0, dist, *lim_coeff); }

      // Adaptive limiting.
      ParGridFunction zeta_0(&ind_fes);
      coef_zeta = new ConstantCoefficient(adapt_lim_const);
      if (adapt_lim_const > 0.0)
      {
         MFEM_VERIFY(pa == false, "PA is not implemented for adaptive limiting");

         FunctionCoefficient alim_coeff(adapt_lim_fun);
         zeta_0.ProjectCoefficient(alim_coeff);

         if (adapt_eval == 0) { adapt_evaluator = new AdvectorCG(al); }
         else if (adapt_eval == 1)
         {
#ifdef MFEM_USE_GSLIB
            adapt_evaluator = new InterpolatorFP;
#else
            MFEM_ABORT("MFEM is not built with GSLIB support!");
#endif
         }
         else { MFEM_ABORT("Bad interpolation option."); }

         he_nlf_integ->EnableAdaptiveLimiting(zeta_0, *coef_zeta, *adapt_evaluator);
      }

      // Surface fitting.
      L2_FECollection mat_coll(0, dim);
      H1_FECollection sigma_fec(mesh_poly_deg, dim);
      ParFiniteElementSpace sigma_fes(pmesh, &sigma_fec);
      ParFiniteElementSpace mat_fes(pmesh, &mat_coll);
      ParGridFunction mat(&mat_fes);
      ParGridFunction marker_gf(&sigma_fes);
      ParGridFunction ls_0(&sigma_fes);
      Array<bool> marker(ls_0.Size());
      coef_ls = new ConstantCoefficient(surface_fit_const);
      if (surface_fit_const > 0.0)
      {
         MFEM_VERIFY(hradaptivity == false,
                     "Surface fitting with HR is not implemented yet.");
         MFEM_VERIFY(pa == false,
                     "Surface fitting with PA is not implemented yet.");

         FunctionCoefficient ls_coeff(surface_level_set);
         ls_0.ProjectCoefficient(ls_coeff);

         for (int i = 0; i < pmesh->GetNE(); i++)
         {
            mat(i) = material_id(i, ls_0);
            pmesh->SetAttribute(i, mat(i) + 1);
         }

         GridFunctionCoefficient coeff_mat(&mat);
         marker_gf.ProjectDiscCoefficient(coeff_mat, GridFunction::ARITHMETIC);
         for (int j = 0; j < marker.Size(); j++)
         {
            if (marker_gf(j) > 0.1 && marker_gf(j) < 0.9)
            {
               marker[j] = true;
               marker_gf(j) = 1.0;
            }
            else
            {
               marker[j] = false;
               marker_gf(j) = 0.0;
            }
         }

         if (adapt_eval == 0) { adapt_surface = new AdvectorCG; }
         else if (adapt_eval == 1)
         {
#ifdef MFEM_USE_GSLIB
            adapt_surface = new InterpolatorFP;
#else
            MFEM_ABORT("MFEM is not built with GSLIB support!");
#endif
         }
         else { MFEM_ABORT("Bad interpolation option."); }

         he_nlf_integ->EnableSurfaceFitting(ls_0, marker, *coef_ls, *adapt_surface);
      }

      // Has to be after the enabling of the limiting / alignment, as it computes
      // normalization factors for these terms as well.
      if (normalization) { he_nlf_integ->ParEnableNormalization(x0); }

      // 13. Setup the final NonlinearForm (which defines the integral of interest,
      //     its first and second derivatives). Here we can use a combination of
      //     metrics, i.e., optimize the sum of two integrals, where both are
      //     scaled by used-defined space-dependent weights.  Note that there are
      //     no command-line options for the weights and the type of the second
      //     metric; one should update those in the code.
      a = new ParNonlinearForm(pfespace);
      if (pa) { a->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
      FunctionCoefficient coeff2(weight_fun);

      // Explicit combination of metrics.
      if (combomet > 0)
      {
         // First metric.
         coeff1 = new ConstantCoefficient(1.0);
         he_nlf_integ->SetCoefficient(*coeff1);

         // Second metric.
         if (dim == 2) { metric2 = new TMOP_Metric_077; }
         else          { metric2 = new TMOP_Metric_315; }
         TMOP_Integrator *he_nlf_integ2 = NULL;
         if (combomet == 1)
         {
            target_c2 = new TargetConstructor(
               TargetConstructor::IDEAL_SHAPE_EQUAL_SIZE, MPI_COMM_WORLD);
            target_c2->SetVolumeScale(0.01);
            target_c2->SetNodes(x0);
            he_nlf_integ2 = new TMOP_Integrator(metric2, target_c2, h_metric);
            he_nlf_integ2->SetCoefficient(coeff2);
         }
         else { he_nlf_integ2 = new TMOP_Integrator(metric2, target_c, h_metric); }
         he_nlf_integ2->SetIntegrationRules(*irules, quad_order);
         if (fdscheme) { he_nlf_integ2->EnableFiniteDifferences(*x); }
         he_nlf_integ2->SetExactActionFlag(exactaction);

         TMOPComboIntegrator *combo = new TMOPComboIntegrator;
         combo->AddTMOPIntegrator(he_nlf_integ);
         combo->AddTMOPIntegrator(he_nlf_integ2);
         if (normalization) { combo->ParEnableNormalization(x0); }
         if (lim_const != 0.0) { combo->EnableLimiting(x0, dist, *lim_coeff); }

         a->AddDomainIntegrator(combo);
      }
      else
      {
         a->AddDomainIntegrator(he_nlf_integ);
      }

      if (pa) { a->Setup(); }

      // Perform the nonlinear optimization.
      const IntegrationRule &ir =
         irules->Get(pfespace->GetFE(0)->GetGeomType(), quad_order);
      solver = new TMOPNewtonSolver(pfespace->GetComm(), ir, solver_type);
      solver->SetInitialScale(ls_scale);

      // Compute the minimum det(J) of the starting mesh.
      tauval = infinity();
#if 0
      {
         const int NE = pmesh->GetNE();
         for (int i = 0; i < NE; i++)
         {
            const IntegrationRule &ir =
               irules->Get(pfespace->GetFE(i)->GetGeomType(), quad_order);
            ElementTransformation *transf = pmesh->GetElementTransformation(i);
            for (int j = 0; j < ir.GetNPoints(); j++)
            {
               transf->SetIntPoint(&ir.IntPoint(j));
               tauval = min(tauval, transf->Jacobian().Det());
            }
         }
      }
#else
      Vector x_out_loc(pfespace->GetVSize());
      pfespace->GetProlongationMatrix()->Mult(x->GetTrueVector(), x_out_loc);
      tauval = dim == 2 ? solver->MinDetJpr_2D(pfespace,x_out_loc):
               dim == 3 ? solver->MinDetJpr_3D(pfespace,x_out_loc): -1.0;
#endif

      double minJ0;
      MPI_Allreduce(&tauval, &minJ0, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
      tauval = minJ0;

      /*if (myid == 0)
      { cout << "Minimum det(J) of the original mesh is " << tauval << endl; }*/

      if (tauval < 0.0 && metric_id != 22 && metric_id != 211 && metric_id != 252
          && metric_id != 311 && metric_id != 313 && metric_id != 352)
      {
         MFEM_ABORT("The input mesh is inverted! Try an untangling metric.");
      }
      if (tauval < 0.0)
      {
         MFEM_VERIFY(target_t == TargetConstructor::IDEAL_SHAPE_UNIT_SIZE,
                     "Untangling is supported only for ideal targets.");

         const DenseMatrix &Wideal =
            Geometries.GetGeomToPerfGeomJac(pfespace->GetFE(0)->GetGeomType());
         tauval /= Wideal.Det();

         double h0min = h0.Min(), h0min_all;
         MPI_Allreduce(&h0min, &h0min_all, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
         // Slightly below minJ0 to avoid div by 0.
         tauval -= 0.01 * h0min_all;
      }

      // For HR tests, the energy is normalized by the number of elements.
      init_energy = a->GetParGridFunctionEnergy(*x) /
                    (hradaptivity ? pmesh->GetGlobalNE() : 1);
      init_metric_energy = init_energy;
      if (lim_const > 0.0 || adapt_lim_const > 0.0 || surface_fit_const > 0.0)
      {
         lim_coeff->constant = 0.0;
         coef_zeta->constant = 0.0;
         coef_ls->constant   = 0.0;
         init_metric_energy = a->GetParGridFunctionEnergy(*x) /
                              (hradaptivity ? pmesh->GetGlobalNE() : 1);
         lim_coeff->constant = lim_const;
         coef_zeta->constant = adapt_lim_const;
         coef_ls->constant   = surface_fit_const;
      }

      // 14. Fix all boundary nodes, or fix only a given component depending on the
      //     boundary attributes of the given mesh.  Attributes 1/2/3 correspond to
      //     fixed x/y/z components of the node.  Attribute 4 corresponds to an
      //     entirely fixed node.  Other boundary attributes do not affect the node
      //     movement boundary conditions.
      if (move_bnd == false)
      {
         Array<int> ess_bdr(pmesh->bdr_attributes.Max());
         ess_bdr = 1;
         a->SetEssentialBC(ess_bdr);
      }
      else
      {
         int ess_vdofs_size = 0;
         for (int i = 0; i < pmesh->GetNBE(); i++)
         {
            const int nd = pfespace->GetBE(i)->GetDof();
            const int attr = pmesh->GetBdrElement(i)->GetAttribute();
            MFEM_VERIFY(!(dim == 2 && attr == 3),
                        "Boundary attribute 3 must be used only for 3D meshes. "
                        "Adjust the attributes (1/2/3/4 for fixed x/y/z/all "
                        "components, rest for free nodes), or use -fix-bnd.");
            if (attr == 1 || attr == 2 || attr == 3) { ess_vdofs_size += nd; }
            if (attr == 4) { ess_vdofs_size += nd * dim; }
         }
         Array<int> ess_vdofs(ess_vdofs_size), vdofs;
         ess_vdofs_size = 0;
         for (int i = 0; i < pmesh->GetNBE(); i++)
         {
            const int nd = pfespace->GetBE(i)->GetDof();
            const int attr = pmesh->GetBdrElement(i)->GetAttribute();
            pfespace->GetBdrElementVDofs(i, vdofs);
            if (attr == 1) // Fix x components.
            {
               for (int j = 0; j < nd; j++)
               { ess_vdofs[ess_vdofs_size++] = vdofs[j]; }
            }
            else if (attr == 2) // Fix y components.
            {
               for (int j = 0; j < nd; j++)
               { ess_vdofs[ess_vdofs_size++] = vdofs[j+nd]; }
            }
            else if (attr == 3) // Fix z components.
            {
               for (int j = 0; j < nd; j++)
               { ess_vdofs[ess_vdofs_size++] = vdofs[j+2*nd]; }
            }
            else if (attr == 4) // Fix all components.
            {
               for (int j = 0; j < vdofs.Size(); j++)
               { ess_vdofs[ess_vdofs_size++] = vdofs[j]; }
            }
         }
         a->SetEssentialVDofs(ess_vdofs);
      }

      // 15. As we use the Newton method to solve the resulting nonlinear system,
      //     here we setup the linear solver for the system's Jacobian.
      const double linsol_rtol = 1e-12;
      if (lin_solver == 0)
      {
         S = new DSmoother(1, 1.0, max_lin_iter);
      }
      else if (lin_solver == 1)
      {
         CGSolver *cg = new CGSolver(MPI_COMM_WORLD);
         cg->SetMaxIter(max_lin_iter);
         cg->SetRelTol(linsol_rtol);
         cg->SetAbsTol(0.0);
         cg->SetPrintLevel(verbosity_level >= 2 ? 3 : -1);
         S = cg;
      }
      else
      {
         MINRESSolver *minres = new MINRESSolver(MPI_COMM_WORLD);
         minres->SetMaxIter(max_lin_iter);
         minres->SetRelTol(linsol_rtol);
         minres->SetAbsTol(0.0);
         if (verbosity_level > 2) { minres->SetPrintLevel(1); }
         else { minres->SetPrintLevel(verbosity_level == 2 ? 3 : -1); }
         if (lin_solver == 3 || lin_solver == 4)
         {
            if (pa)
            {
               MFEM_VERIFY(lin_solver != 4, "PA l1-Jacobi is not implemented");
               auto js = new OperatorJacobiSmoother;
               js->SetPositiveDiagonal(true);
               S_prec = js;
            }
            else
            {
               auto hs = new HypreSmoother;
               hs->SetType((lin_solver == 3) ? HypreSmoother::Jacobi
                           /* */             : HypreSmoother::l1Jacobi, 1);
               hs->SetPositiveDiagonal(true);
               S_prec = hs;
            }
            minres->SetPreconditioner(*S_prec);
         }
         S = minres;
      }

      // Provide all integration rules in case of a mixed mesh.
      solver->SetIntegrationRules(*irules, quad_order);
      if (solver_type == 0)
      {
         // Specify linear solver when we use a Newton-based solver.
         solver->SetPreconditioner(*S);
      }
      // For untangling, the solver will update the min det(T) values.
      if (tauval < 0.0) { solver->SetMinDetPtr(&tauval); }
      solver->SetMaxIter(solver_iter);
      solver->SetRelTol(solver_rtol);
      solver->SetAbsTol(0.0);
      if (solver_art_type > 0)
      {
         solver->SetAdaptiveLinRtol(solver_art_type, 0.5, 0.9);
      }
      solver->SetPrintLevel(verbosity_level >= 1 ? 1 : -1);

      // hr-adaptivity solver.
      // If hr-adaptivity is disabled, r-adaptivity is done once using the
      // TMOPNewtonSolver.
      // Otherwise, "hr_iter" iterations of r-adaptivity are done followed by
      // "h_per_r_iter" iterations of h-adaptivity after each r-adaptivity.
      // The solver terminates if an h-adaptivity iteration does not modify
      // any element in the mesh.
      /*TMOPHRSolver hr_solver(*pmesh, a, solver,
                             x, move_bnd, hradaptivity,
                             mesh_poly_deg, h_metric_id,
                             n_hr_iter, n_h_iter);
      hr_solver.AddGridFunctionForUpdate(&x0);
      if (adapt_lim_const > 0.)
      {
         hr_solver.AddGridFunctionForUpdate(&zeta_0);
         hr_solver.AddFESpaceForUpdate(&ind_fes);
      }
      hr_solver.Mult();*/

      solver->SetOperator(*a);
   }

   /////////////////////////////////////////////////////////////////////////////
   ~TMOP_PMESH_OPTIMIZER()
   {
      delete S;
      delete S_prec;
      delete target_c2;
      delete metric2;
      delete coeff1;
      delete adapt_evaluator;
      delete adapt_surface;
      delete target_c;
      delete hr_adapt_coeff;
      delete adapt_coeff;
      delete h_metric;
      delete metric;
      delete pfespace;
      delete fec;
      delete pmesh;

      // he_nlf_integ will be deleted with a
      delete a;
      delete lim_coeff;
      delete coef_zeta;
      delete coef_ls;
      delete x;
      delete solver;
   }

   /////////////////////////////////////////////////////////////////////////////
   void Mult()
   {
      TimeSolver.Start();
      solver->Mult(b, x->GetTrueVector());
      MFEM_DEVICE_SYNC;
      TimeSolver.Stop();
      mdof += 1e-6 * ndofs;
   }

   /////////////////////////////////////////////////////////////////////////////
   double Mdof() const { return mdof; }

   /////////////////////////////////////////////////////////////////////////////
   void Postfix()
   {
      x->SetFromTrueVector();
      const double solvertime = TimeSolver.RealTime(),
                   vectortime = solver->GetAssembleElementVectorTime(),
                   gradtime   = solver->GetAssembleElementGradTime(),
                   prectime   = solver->GetPrecMultTime(),
                   processnewstatetime = solver->GetProcessNewStateTime(),
                   scalefactortime = solver->GetComputeScalingTime();

      if (myid == 0 && solver->GetConverged() == false)
      {
         cout << "Nonlinear solver: rtol = " << solver_rtol << " not achieved.\n";
      }

      MPI_Barrier(MPI_COMM_WORLD);
      int NDofs = x->ParFESpace()->GlobalTrueVSize()/pmesh->Dimension(),
          NEGlob = pmesh->GetGlobalNE();
      int device_tag  = 0; //gpu
      const double fin_energy = a->GetParGridFunctionEnergy(*x) /
                                (hradaptivity ? pmesh->GetGlobalNE() : 1);
      if (myid == 0)
      {
         std::cout << "Monitoring info      :" << endl
                   << "Number of elements   :" << NEGlob << endl
                   << "Number of procs      :" << num_procs << endl
                   << "Polynomial degree    :" << mesh_poly_deg << endl
                   << "Total TDofs          :" << NDofs << endl
                   << std::setprecision(4)
                   << "Total Iterations     :" << solver->GetNumIterations() << endl
                   << "Total Prec Iterations:" << solver->GetTotalPrecIterations() << endl
                   << "Total Solver Time (%):" << solvertime << " "
                   << (solvertime*100/solvertime) << endl
                   << "Assemble Vector Time :" << vectortime << " "
                   << (vectortime*100/solvertime) << endl
                   << "Assemble Grad Time   :" << gradtime << " "
                   << gradtime*100/solvertime <<  endl
                   << "Prec Solve Time      :" << prectime << " "
                   << prectime*100/solvertime <<  endl
                   << "ProcessNewState Time :" << processnewstatetime << " "
                   << (processnewstatetime*100/solvertime) <<  endl
                   << "ComputeScale Time    :" << scalefactortime << " "
                   << (scalefactortime*100/solvertime) <<  "  " << endl
                   << "Device Tag (0 for gpu, 1 otherwise):" << device_tag << endl
                   << " Final energy: " << fin_energy << endl;

         std::cout << "run_info: " << std::setprecision(4) << " "
                   << rs_levels << " "
                   << mesh_poly_deg << " " << quad_order << " "
                   << solver_type << " " <<  solver_art_type << " "
                   << lin_solver << " " << max_lin_iter << " "
                   << pa << " " << metric_id << " " << num_procs
                   << std::setprecision(10) << " "
                   << NEGlob << " " << NDofs << " "
                   << solver->GetNumIterations() << " "
                   << solver->GetTotalPrecIterations() << " "
                   << solvertime << " "
                   << (vectortime*100/solvertime) << " "
                   << (gradtime*100/solvertime) << " "
                   << (prectime*100/solvertime) << " "
                   << (processnewstatetime*100/solvertime) << " "
                   << (scalefactortime*100/solvertime) << " "
                   << fin_energy << endl;
      }

      // Compute the final energy of the functional.
      double fin_metric_energy = fin_energy;
      if (lim_const > 0.0 || adapt_lim_const > 0.0 || surface_fit_const > 0.0)
      {
         lim_coeff->constant = 0.0;
         coef_zeta->constant = 0.0;
         coef_ls->constant   = 0.0;
         fin_metric_energy  = a->GetParGridFunctionEnergy(*x) /
                              (hradaptivity ? pmesh->GetGlobalNE() : 1);
         lim_coeff->constant = lim_const;
         coef_zeta->constant = adapt_lim_const;
         coef_ls->constant   = surface_fit_const;
      }

      if (myid == 0)
      {
         std::cout << std::scientific << std::setprecision(4);
         cout << "Initial strain energy: " << init_energy
              << " = metrics: " << init_metric_energy
              << " + extra terms: " << init_energy - init_metric_energy << endl;
         cout << "  Final strain energy: " << fin_energy
              << " = metrics: " << fin_metric_energy
              << " + extra terms: " << fin_energy - fin_metric_energy << endl;
         cout << "The strain energy decreased by: "
              << (init_energy - fin_energy) * 100.0 / init_energy << " %." << endl;
      }

      if (surface_fit_const > 0.0)
      {
         double err_avg, err_max;
         he_nlf_integ->GetSurfaceFittingErrors(err_avg, err_max);
         if (myid == 0)
         {
            std::cout << "Avg fitting error: " << err_avg << std::endl
                      << "Max fitting error: " << err_max << std::endl;
         }
      }
   }
};

//////////////////////////////////////////////////////////////////////////////
#define MAX_NDOFS 6*1024*1024

#if 0
static void OrderSideArgs(bmi::Benchmark *b)
{
   const auto ndofs_estimate = [](int p, int nx, int ny, int nz)
   {
      return (nx*p)*(ny*p)*(nz*p);
   };
   for (int p = 4; p > 0; p--)
   {
      for (int nx = 12; nx <= 60; nx += 6)
      {
         for (int ny = nx; ny < nx+6; ny += 2)
         {
            for (int nz = ny; nz < nx+6; nz += 2)
            {
               if (ndofs_estimate(p,nx,ny,nz) > MAX_NDOFS) { continue; }
               b->Args({p, nx, ny, nz});
            }
         }
      }
   }
}
#else
static void OrderSideArgs(bmi::Benchmark *b)
{
   const auto est = [](int c) { return 3*(c+1)*(c+1)*(c+1); };
   for (int p = 4; p > 0; p--)
   {
      for (int c = 12; est(c) <= MAX_NDOFS; c += 6)
      {
         if ((c==12) && (p<3)) { continue; } // skip these inverted meshes
         b->Args({p, c});
      }
   }
}
#endif

static void TMOP_FULL(bm::State &state)
{
   const int p = state.range(0);
   const int side = state.range(1);
   TMOP_PMESH_OPTIMIZER ker(p, side);
   const int ndofs = ker.ndofs;
   while (state.KeepRunning()) { ker.Mult(); }
   const double solvertime = ker.TimeSolver.RealTime();
   state.counters["P"] = bm::Counter(p);
   state.counters["T"] = bm::Counter(solvertime);
   state.counters["Q"] = bm::Counter(ker.hexahedron_quad_points);
   state.counters["Dofs"] = bm::Counter(ndofs);
   state.counters["MDofs"] = bm::Counter(ker.Mdof(), bm::Counter::kIsRate);
   //tmop_pmesh_optimizer.Postfix();
}
BENCHMARK(TMOP_FULL)\
-> Apply(OrderSideArgs) -> Unit(bm::kMillisecond) -> Iterations(10);

int main(int argc, char *argv[])
{
#ifdef MFEM_USE_MPI
   mfem::MPI_Session main_mpi(argc, argv);
   mpi = &main_mpi;
#endif

   bm::ConsoleReporter CR;
   bm::Initialize(&argc, argv);

   // Device setup, cpu by default
   std::string config_device = "cpu:fast";

   if (bmi::global_context != nullptr)
   {
      bmi::FindInContext("dev", config_device); // dev=cuda
      bmi::FindInContext("ndev", config_dev_size); // ndev=4
   }

   const int mpi_rank = mpi->WorldRank();
   const int mpi_size = mpi->WorldSize();
   const int dev = config_dev_size > 0 ? mpi_rank % config_dev_size : 0;

   Device device(config_device.c_str(), dev);
   if (mpi->Root()) { device.Print(); }

   if (bm::ReportUnrecognizedArguments(argc, argv)) { return 1; }

#ifndef MFEM_USE_MPI
   bm::RunSpecifiedBenchmarks(&CR);
#else
   if (mpi->Root()) { bm::RunSpecifiedBenchmarks(&CR); }
   else
   {
      // No display_reporter and file_reporter
      // bm::RunSpecifiedBenchmarks(NoReporter());
      bm::BenchmarkReporter *file_reporter = new NoReporter();
      bm::BenchmarkReporter *display_reporter = new NoReporter();
      bm::RunSpecifiedBenchmarks(display_reporter, file_reporter);
   }
#endif // MFEM_USE_MPI
   return 0;
}

#endif // MFEM_USE_BENCHMARK
