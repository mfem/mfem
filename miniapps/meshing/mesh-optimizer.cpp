#include "../../mfem.hpp"
#include <fstream>
#include <iostream>

using namespace mfem;
using namespace std;

#include "mesh-optimizer.hpp"

void TMOPupdate(NonlinearForm &a, Mesh &mesh, FiniteElementSpace &fespace,
                bool move_bnd)
{
   int dim = fespace.GetFE(0)->GetDim();
   if (move_bnd == false)
   {
      Array<int> ess_bdr(mesh.bdr_attributes.Max());
      ess_bdr = 1;
      a.SetEssentialBC(ess_bdr);
   }
   else
   {
      const int nd  = fespace.GetBE(0)->GetDof();
      int n = 0;
      for (int i = 0; i < mesh.GetNBE(); i++)
      {
         const int attr = mesh.GetBdrElement(i)->GetAttribute();
         MFEM_VERIFY(!(dim == 2 && attr == 3),
                     "Boundary attribute 3 must be used only for 3D meshes. "
                     "Adjust the attributes (1/2/3/4 for fixed x/y/z/all "
                     "components, rest for free nodes), or use -fix-bnd.");
         if (attr == 1 || attr == 2 || attr == 3) { n += nd; }
         if (attr == 4) { n += nd * dim; }
      }
      Array<int> ess_vdofs(n), vdofs;
      n = 0;
      for (int i = 0; i < mesh.GetNBE(); i++)
      {
         const int attr = mesh.GetBdrElement(i)->GetAttribute();
         fespace.GetBdrElementVDofs(i, vdofs);
         if (attr == 1) // Fix x components.
         {
            for (int j = 0; j < nd; j++)
            { ess_vdofs[n++] = vdofs[j]; }
         }
         else if (attr == 2) // Fix y components.
         {
            for (int j = 0; j < nd; j++)
            { ess_vdofs[n++] = vdofs[j+nd]; }
         }
         else if (attr == 3) // Fix z components.
         {
            for (int j = 0; j < nd; j++)
            { ess_vdofs[n++] = vdofs[j+2*nd]; }
         }
         else if (attr == 4) // Fix all components.
         {
            for (int j = 0; j < vdofs.Size(); j++)
            { ess_vdofs[n++] = vdofs[j]; }
         }
      }
      a.SetEssentialVDofs(ess_vdofs);
   }
};

// Additional IntegrationRules that can be used with the --quad-type option.
IntegrationRules IntRulesLo(0, Quadrature1D::GaussLobatto);
IntegrationRules IntRulesCU(0, Quadrature1D::ClosedUniform);


int main(int argc, char *argv[])
{
   // 0. Set the method's default parameters.
   const char *mesh_file = "icf.mesh";
   int mesh_poly_deg     = 1;
   int rs_levels         = 0;
   double jitter         = 0.0;
   int metric_id         = 1;
   int target_id         = 1;
   double lim_const      = 0.0;
   int quad_type         = 1;
   int quad_order        = 8;
   int newton_iter       = 10;
   double newton_rtol    = 1e-10;
   int lin_solver        = 2;
   int max_lin_iter      = 100;
   bool move_bnd         = true;
   bool combomet         = 0;
   int amr_flag          = 1;
   int amrmetric_id         = -1;
   bool normalization    = false;
   bool visualization    = true;
   int verbosity_level   = 0;
   int hessiantype       = 1;
   int fdscheme          = 0;
   int adapt_eval        = 1;

   // 1. Parse command-line options.
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&mesh_poly_deg, "-o", "--order",
                  "Polynomial degree of mesh finite element space.");
   args.AddOption(&rs_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&jitter, "-ji", "--jitter",
                  "Random perturbation scaling factor.");
   args.AddOption(&metric_id, "-mid", "--metric-id",
                  "Mesh optimization metric:\n\t"
                  "1  : |T|^2                          -- 2D shape\n\t"
                  "2  : 0.5|T|^2/tau-1                 -- 2D shape (condition number)\n\t"
                  "7  : |T-T^-t|^2                     -- 2D shape+size\n\t"
                  "9  : tau*|T-T^-t|^2                 -- 2D shape+size\n\t"
                  "14: 0.5*(1-cos(theta_A - theta_W)   -- 2D Sh+Sz+Alignment\n\t"
                  "22 : 0.5(|T|^2-2*tau)/(tau-tau_0)   -- 2D untangling\n\t"
                  "50 : 0.5|T^tT|^2/tau^2-1            -- 2D shape\n\t"
                  "55 : (tau-1)^2                      -- 2D size\n\t"
                  "56 : 0.5(sqrt(tau)-1/sqrt(tau))^2   -- 2D size\n\t"
                  "58 : |T^tT|^2/(tau^2)-2*|T|^2/tau+2 -- 2D shape\n\t"
                  "77 : 0.5(tau-1/tau)^2               -- 2D size\n\t"
                  "211: (tau-1)^2-tau+sqrt(tau^2)      -- 2D untangling\n\t"
                  "252: 0.5(tau-1)^2/(tau-tau_0)       -- 2D untangling\n\t"
                  "301: (|T||T^-1|)/3-1              -- 3D shape\n\t"
                  "302: (|T|^2|T^-1|^2)/9-1          -- 3D shape\n\t"
                  "303: (|T|^2)/3*tau^(2/3)-1        -- 3D shape\n\t"
                  "315: (tau-1)^2                    -- 3D size\n\t"
                  "316: 0.5(sqrt(tau)-1/sqrt(tau))^2 -- 3D size\n\t"
                  "321: |T-T^-t|^2                   -- 3D shape+size\n\t"
                  "352: 0.5(tau-1)^2/(tau-tau_0)     -- 3D untangling");
   args.AddOption(&target_id, "-tid", "--target-id",
                  "Target (ideal element) type:\n\t"
                  "1: Ideal shape, unit size\n\t"
                  "2: Ideal shape, equal size\n\t"
                  "3: Ideal shape, initial size\n\t"
                  "4: Given full analytic Jacobian (in physical space)\n\t"
                  "5: Ideal shape, given size (in physical space)");
   args.AddOption(&lim_const, "-lc", "--limit-const", "Limiting constant.");
   args.AddOption(&quad_type, "-qt", "--quad-type",
                  "Quadrature rule type:\n\t"
                  "1: Gauss-Lobatto\n\t"
                  "2: Gauss-Legendre\n\t"
                  "3: Closed uniform points");
   args.AddOption(&quad_order, "-qo", "--quad_order",
                  "Order of the quadrature rule.");
   args.AddOption(&newton_iter, "-ni", "--newton-iters",
                  "Maximum number of Newton iterations.");
   args.AddOption(&newton_rtol, "-rtol", "--newton-rel-tolerance",
                  "Relative tolerance for the Newton solver.");
   args.AddOption(&lin_solver, "-ls", "--lin-solver",
                  "Linear solver: 0 - l1-Jacobi, 1 - CG, 2 - MINRES.");
   args.AddOption(&max_lin_iter, "-li", "--lin-iter",
                  "Maximum number of iterations in the linear solve.");
   args.AddOption(&move_bnd, "-bnd", "--move-boundary", "-fix-bnd",
                  "--fix-boundary",
                  "Enable motion along horizontal and vertical boundaries.");
   args.AddOption(&combomet, "-cmb", "--combo-met", "-no-cmb", "--no-combo-met",
                  "Combination of metrics.");
   args.AddOption(&amr_flag, "-amr", "--amr-flag",
                  "1 - AMR after TMOP");
   args.AddOption(&amrmetric_id, "-amrm", "--amr-metric",
                  "0 - Size, 1 - AspectRatio, 2 - Size + AspectRatio");
   args.AddOption(&hessiantype, "-ht", "--Hessian Target type",
                  "1-6");
   args.AddOption(&normalization, "-nor", "--normalization", "-no-nor",
                  "--no-normalization",
                  "Make all terms in the optimization functional unitless.");
   args.AddOption(&fdscheme, "-fd", "--fd_approximation",
                  "Enable finite difference based derivative computations.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&verbosity_level, "-vl", "--verbosity-level",
                  "Set the verbosity level - 0, 1, or 2.");
   args.AddOption(&adapt_eval, "-ae", "--adaptivity evaluatior",
                  "0 - Advection based (DEFAULT), 1 - GSLIB.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   if (amrmetric_id < 0) { amrmetric_id = metric_id; }
   // 2. Initialize and refine the starting mesh->
   Mesh *mesh = new Mesh(mesh_file, 1, 1, false);
   for (int lev = 0; lev < rs_levels; lev++) { mesh->UniformRefinement(); }
   const int dim = mesh->Dimension();
   cout << "Mesh curvature: ";
   if (mesh->GetNodes()) { cout << mesh->GetNodes()->OwnFEC()->Name(); }
   else { cout << "(NONE)"; }
   cout << endl;

   // 3. Define a finite element space on the mesh-> Here we use vector finite
   //    elements which are tensor products of quadratic finite elements. The
   //    number of components in the vector finite element space is specified by
   //    the last parameter of the FiniteElementSpace constructor.
   H1_FECollection fec(mesh_poly_deg, dim);
   FiniteElementSpace fespace(mesh, &fec, dim);

   // 4. Make the mesh curved based on the above finite element space. This
   //    means that we define the mesh elements through a fespace-based
   //    transformation of the reference element.
   mesh->SetNodalFESpace(&fespace);

   // 5. Set up an empty right-hand side vector b, which is equivalent to b=0.
   Vector b(0);

   // 6. Get the mesh nodes (vertices and other degrees of freedom in the finite
   //    element space) as a finite element grid function in fespace. Note that
   //    changing x automatically changes the shapes of the mesh elements.
   GridFunction x(&fespace);
   GridFunction xnew(&fespace);
   GridFunction x0new(&fespace);
   mesh->SetNodalGridFunction(&x);

   // 7. Define a vector representing the minimal local mesh size in the mesh
   //    nodes. We index the nodes using the scalar version of the degrees of
   //    freedom in fespace. Note: this is partition-dependent.
   //
   //    In addition, compute average mesh size and total volume.
   Vector h0(fespace.GetNDofs());
   h0 = infinity();
   double volume = 0.0;
   Array<int> dofs;
   for (int i = 0; i < mesh->GetNE(); i++)
   {
      // Get the local scalar element degrees of freedom in dofs.
      fespace.GetElementDofs(i, dofs);
      // Adjust the value of h0 in dofs based on the local mesh size.
      const double hi = mesh->GetElementSize(i);
      for (int j = 0; j < dofs.Size(); j++)
      {
         h0(dofs[j]) = min(h0(dofs[j]), hi);
      }
      volume += mesh->GetElementVolume(i);
   }
   const double small_phys_size = pow(volume, 1.0 / dim) / 100.0;

   // 8. Add a random perturbation to the nodes in the interior of the domain.
   //    We define a random grid function of fespace and make sure that it is
   //    zero on the boundary and its values are locally of the order of h0.
   //    The latter is based on the DofToVDof() method which maps the scalar to
   //    the vector degrees of freedom in fespace.
   GridFunction rdm(&fespace);
   rdm.Randomize();
   rdm -= 0.25; // Shift to random values in [-0.5,0.5].
   rdm *= jitter;
   // Scale the random values to be of order of the local mesh size.
   for (int i = 0; i < fespace.GetNDofs(); i++)
   {
      for (int d = 0; d < dim; d++)
      {
         rdm(fespace.DofToVDof(i,d)) *= h0(i);
      }
   }
   Array<int> vdofs;
   for (int i = 0; i < fespace.GetNBE(); i++)
   {
      // Get the vector degrees of freedom in the boundary element.
      fespace.GetBdrElementVDofs(i, vdofs);
      // Set the boundary values to zero.
      for (int j = 0; j < vdofs.Size(); j++) { rdm(vdofs[j]) = 0.0; }
   }
   x -= rdm;
   x.SetTrueVector();
   x.SetFromTrueVector();

   // 9. Save the starting (prior to the optimization) mesh to a file. This
   //    output can be viewed later using GLVis: "glvis -m perturbed.mesh".
   {
      ofstream mesh_ofs("perturbed.mesh");
      mesh->Print(mesh_ofs);
   }

   // 10. Store the starting (prior to the optimization) positions.
   GridFunction x0(&fespace);
   x0 = x;

   // 11. Form the integrator that uses the chosen metric and target.
   double tauval = -0.1;
   TMOP_QualityMetric *metric = NULL;
   switch (metric_id)
   {
      case 1: metric = new TMOP_Metric_001; break;
      case 2: metric = new TMOP_Metric_002; break;
      case 7: metric = new TMOP_Metric_007; break;
      case 9: metric = new TMOP_Metric_009; break;
      case 14: metric = new TMOP_Metric_SSA2D; break;
      case 22: metric = new TMOP_Metric_022(tauval); break;
      case 50: metric = new TMOP_Metric_050; break;
      case 55: metric = new TMOP_Metric_055; break;
      case 56: metric = new TMOP_Metric_056; break;
      case 58: metric = new TMOP_Metric_058; break;
      case 77: metric = new TMOP_Metric_077; break;
      case 87: metric = new TMOP_Metric_SS2D; break;
      case 211: metric = new TMOP_Metric_211; break;
      case 252: metric = new TMOP_Metric_252(tauval); break;
      case 301: metric = new TMOP_Metric_301; break;
      case 302: metric = new TMOP_Metric_302; break;
      case 303: metric = new TMOP_Metric_303; break;
      case 315: metric = new TMOP_Metric_315; break;
      case 316: metric = new TMOP_Metric_316; break;
      case 321: metric = new TMOP_Metric_321; break;
      case 352: metric = new TMOP_Metric_352(tauval); break;
      default: cout << "Unknown metric_id: " << metric_id << endl; return 3;
   }
   TMOP_QualityMetric *amrmetric = NULL;
   switch (amrmetric_id)
   {
      case 1: amrmetric = new TMOP_Metric_001; break;
      case 2: amrmetric = new TMOP_Metric_002; break;
      case 7: amrmetric = new TMOP_Metric_007; break;
      case 9: amrmetric = new TMOP_Metric_009; break;
      case 55: amrmetric = new TMOP_Metric_055; break;
      case 56: amrmetric = new TMOP_Metric_056; break;
      case 58: amrmetric = new TMOP_Metric_058; break;
      case 77: amrmetric = new TMOP_Metric_077; break;
      case 302: amrmetric = new TMOP_Metric_302; break;
      default: cout << "Unknown metric_id: " << amrmetric_id << endl; return 3;
   }
   TargetConstructor::TargetType target_t;
   TargetConstructor *target_c = NULL;
   HessianCoefficient *adapt_coeff = NULL;
   H1_FECollection ind_fec(mesh_poly_deg, dim);
   DiscreteAdaptTC *tcd = NULL;
   AnalyticAdaptTC *tca = NULL;
   FiniteElementSpace ind_fes(mesh, &ind_fec);
   FiniteElementSpace ind_fesv(mesh, &ind_fec, dim);
   GridFunction size(&ind_fes), aspr(&ind_fes), disc(&ind_fes), ori(&ind_fes);
   GridFunction aspr3d(&ind_fesv), size3d(&ind_fesv);
   switch (target_id)
   {
      case 1: target_t = TargetConstructor::IDEAL_SHAPE_UNIT_SIZE; break;
      case 2: target_t = TargetConstructor::IDEAL_SHAPE_EQUAL_SIZE; break;
      case 3: target_t = TargetConstructor::IDEAL_SHAPE_GIVEN_SIZE; break;
      case 4: // Analytic
      {
         target_t = TargetConstructor::GIVEN_FULL;
         tca = new AnalyticAdaptTC(target_t);
         adapt_coeff = new HessianCoefficient(dim, hessiantype);
         tca->SetAnalyticTargetSpec(NULL, NULL, adapt_coeff);
         target_c = tca;
         break;
      }
      case 5: // Discrete size 2D
      {
         target_t = TargetConstructor::IDEAL_SHAPE_GIVEN_SIZE;
         tcd = new DiscreteAdaptTC(target_t);
         if (adapt_eval == 0)
         {
            tcd->SetAdaptivityEvaluator(new AdvectorCG);
         }
         else
         {
#ifdef MFEM_USE_GSLIB
            tcd->SetAdaptivityEvaluator(new InterpolatorFP);
#endif
         }
         FunctionCoefficient ind_coeff(discrete_size_2d);
         size.ProjectCoefficient(ind_coeff);
         tcd->SetSerialDiscreteTargetSize(size);
         tcd->FinalizeSerialDiscreteTargetSpec();
         target_c = tcd;
         break;
      }
      case 6: // Discrete size + aspect ratio - 2D
      {
         GridFunction d_x(&ind_fes), d_y(&ind_fes);

         target_t = TargetConstructor::GIVEN_SHAPE_AND_SIZE;
         tcd = new DiscreteAdaptTC(target_t);
         FunctionCoefficient ind_coeff(material_indicator_2d);
         disc.ProjectCoefficient(ind_coeff);
         if (adapt_eval == 0)
         {
            tcd->SetAdaptivityEvaluator(new AdvectorCG);
         }
         else
         {
#ifdef MFEM_USE_GSLIB
            tcd->SetAdaptivityEvaluator(new InterpolatorFP);
#endif
         }

         //Diffuse the interface
         DiffuseField(disc,2);

         //Get  partials with respect to x and y of the grid function
         disc.GetDerivative(1,0,d_x);
         disc.GetDerivative(1,1,d_y);

         //Compute the squared magnitude of the gradient
         for (int i = 0; i < size.Size(); i++)
         {
            size(i) = std::pow(d_x(i),2)+std::pow(d_y(i),2);
         }
         const double max = size.Max();

         for (int i = 0; i < d_x.Size(); i++)
         {
            d_x(i) = std::abs(d_x(i));
            d_y(i) = std::abs(d_y(i));
         }
         const double eps = 0.01;
         const double ratio = 20.0;
         const double big_small_ratio = 40.0;

         for (int i = 0; i < size.Size(); i++)
         {
            size(i) = (size(i)/max);
            aspr(i) = (d_x(i)+eps)/(d_y(i)+eps);
            aspr(i) = 0.1 + 0.9*(1-size(i))*(1-size(i));
            if (aspr(i) > ratio) {aspr(i) = ratio;}
            if (aspr(i) < 1.0/ratio) {aspr(i) = 1.0/ratio;}
         }
         Vector vals;
         const int NE = mesh->GetNE();
         double volume = 0.0, volume_ind = 0.0;

         for (int i = 0; i < NE; i++)
         {
            ElementTransformation *Tr = mesh->GetElementTransformation(i);
            const IntegrationRule &ir =
               IntRules.Get(mesh->GetElementBaseGeometry(i), Tr->OrderJ());
            size.GetValues(i, ir, vals);
            for (int j = 0; j < ir.GetNPoints(); j++)
            {
               const IntegrationPoint &ip = ir.IntPoint(j);
               Tr->SetIntPoint(&ip);
               volume     += ip.weight * Tr->Weight();
               volume_ind += vals(j) * ip.weight * Tr->Weight();
            }
         }

         const double avg_zone_size = volume / NE;

         const double small_avg_ratio = (volume_ind + (volume - volume_ind) /
                                         big_small_ratio) /
                                        volume;

         const double small_zone_size = small_avg_ratio * avg_zone_size;
         const double big_zone_size   = big_small_ratio * small_zone_size;

         for (int i = 0; i < size.Size(); i++)
         {
            const double val = size(i);
            const double a = (big_zone_size - small_zone_size) / small_zone_size;
            size(i) = big_zone_size / (1.0+a*val);
         }


         DiffuseField(size, 2);
         DiffuseField(aspr, 2);

         tcd->SetSerialDiscreteTargetSize(size);
         tcd->SetSerialDiscreteTargetAspectRatio(aspr);
         tcd->FinalizeSerialDiscreteTargetSpec();
         target_c = tcd;
         break;
      }
      case 7: // Discrete aspect ratio 3D
      {
         target_t = TargetConstructor::GIVEN_SHAPE_AND_SIZE;
         tcd = new DiscreteAdaptTC(target_t);
         if (adapt_eval == 0)
         {
            tcd->SetAdaptivityEvaluator(new AdvectorCG);
         }
         else
         {
#ifdef MFEM_USE_GSLIB
            tcd->SetAdaptivityEvaluator(new InterpolatorFP);
#endif
         }
         VectorFunctionCoefficient fd_aspr3d(dim, discrete_aspr_3d);
         aspr3d.ProjectCoefficient(fd_aspr3d);

         tcd->SetSerialDiscreteTargetAspectRatio(aspr3d);
         tcd->FinalizeSerialDiscreteTargetSpec();
         target_c = tcd;
         break;
      }
      case 8: // shape/size + orientation 2D
      {
         target_t = TargetConstructor::GIVEN_SHAPE_AND_SIZE;
         tcd = new DiscreteAdaptTC(target_t);
         if (adapt_eval == 0)
         {
            tcd->SetAdaptivityEvaluator(new AdvectorCG);
         }
         else
         {
#ifdef MFEM_USE_GSLIB
            tcd->SetAdaptivityEvaluator(new InterpolatorFP);
#endif
         }

         if (metric_id == 14)
         {
            ConstantCoefficient ind_coeff(0.1*0.1);
            size.ProjectCoefficient(ind_coeff);
            tcd->SetSerialDiscreteTargetSize(size);
         }

         if (metric_id == 87)
         {
            FunctionCoefficient aspr_coeff(discrete_aspr_2d);
            aspr.ProjectCoefficient(aspr_coeff);
            DiffuseField(aspr,2);
            tcd->SetSerialDiscreteTargetAspectRatio(aspr);
         }

         FunctionCoefficient ori_coeff(discrete_ori_2d);
         ori.ProjectCoefficient(ori_coeff);
         tcd->SetSerialDiscreteTargetOrientation(ori);
         tcd->FinalizeSerialDiscreteTargetSpec();
         target_c = tcd;
         break;
      }
      default: cout << "Unknown target_id: " << target_id << endl; return 3;
   }
   if (target_c == NULL)
   {
      target_c = new TargetConstructor(target_t);
   }
   target_c->SetNodes(x0);
   TMOP_Integrator *he_nlf_integ =
      new TMOP_Integrator(metric, target_c, amrmetric);
   if (fdscheme) { he_nlf_integ->EnableFiniteDifferences(x); }

   // 12. Setup the quadrature rule for the non-linear form integrator.
   const IntegrationRule *ir = NULL;
   const int geom_type = fespace.GetFE(0)->GetGeomType();
   switch (quad_type)
   {
      case 1: ir = &IntRulesLo.Get(geom_type, quad_order); break;
      case 2: ir = &IntRules.Get(geom_type, quad_order); break;
      case 3: ir = &IntRulesCU.Get(geom_type, quad_order); break;
      default: cout << "Unknown quad_type: " << quad_type << endl;
         delete he_nlf_integ; return 3;
   }
   cout << "Quadrature points per cell: " << ir->GetNPoints() << endl;
   he_nlf_integ->SetIntegrationRule(*ir);

   if (normalization) { he_nlf_integ->EnableNormalization(x0); }

   // 13. Limit the node movement.
   // The limiting distances can be given by a general function of space.
   GridFunction dist(&fespace);
   dist = 1.0;
   // The small_phys_size is relevant only with proper normalization.
   if (normalization) { dist = small_phys_size; }
   ConstantCoefficient lim_coeff(lim_const);
   if (lim_const != 0.0) { he_nlf_integ->EnableLimiting(x0, dist, lim_coeff); }

   // 14. Setup the final NonlinearForm (which defines the integral of interest,
   //     its first and second derivatives). Here we can use a combination of
   //     metrics, i.e., optimize the sum of two integrals, where both are
   //     scaled by used-defined space-dependent weights. Note that there are no
   //     command-line options for the weights and the type of the second
   //     metric; one should update those in the code.
   NonlinearForm a(&fespace);
   ConstantCoefficient *coeff1 = NULL;
   a.AddDomainIntegrator(he_nlf_integ);

   const double init_energy = a.GetGridFunctionEnergy(x);

   // 15. Visualize the starting mesh and metric values.
   if (visualization)
   {
      char title[] = "Initial metric values";
      vis_tmop_metric_s(mesh_poly_deg, *metric, *target_c, *mesh, title, 0);
   }

   // 16. Fix all boundary nodes, or fix only a given component depending on the
   //     boundary attributes of the given mesh-> Attributes 1/2/3 correspond to
   //     fixed x/y/z components of the node. Attribute 4 corresponds to an
   //     entirely fixed node. Other boundary attributes do not affect the node
   //     movement boundary conditions.
   if (move_bnd == false)
   {
      Array<int> ess_bdr(mesh->bdr_attributes.Max());
      ess_bdr = 1;
      a.SetEssentialBC(ess_bdr);
   }
   else
   {
      const int nd  = fespace.GetBE(0)->GetDof();
      int n = 0;
      for (int i = 0; i < mesh->GetNBE(); i++)
      {
         const int attr = mesh->GetBdrElement(i)->GetAttribute();
         MFEM_VERIFY(!(dim == 2 && attr == 3),
                     "Boundary attribute 3 must be used only for 3D meshes. "
                     "Adjust the attributes (1/2/3/4 for fixed x/y/z/all "
                     "components, rest for free nodes), or use -fix-bnd.");
         if (attr == 1 || attr == 2 || attr == 3) { n += nd; }
         if (attr == 4) { n += nd * dim; }
      }
      Array<int> ess_vdofs(n), vdofs;
      n = 0;
      for (int i = 0; i < mesh->GetNBE(); i++)
      {
         const int attr = mesh->GetBdrElement(i)->GetAttribute();
         fespace.GetBdrElementVDofs(i, vdofs);
         if (attr == 1) // Fix x components.
         {
            for (int j = 0; j < nd; j++)
            { ess_vdofs[n++] = vdofs[j]; }
         }
         else if (attr == 2) // Fix y components.
         {
            for (int j = 0; j < nd; j++)
            { ess_vdofs[n++] = vdofs[j+nd]; }
         }
         else if (attr == 3) // Fix z components.
         {
            for (int j = 0; j < nd; j++)
            { ess_vdofs[n++] = vdofs[j+2*nd]; }
         }
         else if (attr == 4) // Fix all components.
         {
            for (int j = 0; j < vdofs.Size(); j++)
            { ess_vdofs[n++] = vdofs[j]; }
         }
      }
      a.SetEssentialVDofs(ess_vdofs);
   }

   // 17. As we use the Newton method to solve the resulting nonlinear system,
   //     here we setup the linear solver for the system's Jacobian.
   Solver *S = NULL;
   const double linsol_rtol = 1e-12;
   if (lin_solver == 0)
   {
      S = new DSmoother(1, 1.0, max_lin_iter);
   }
   else if (lin_solver == 1)
   {
      CGSolver *cg = new CGSolver;
      cg->SetMaxIter(max_lin_iter);
      cg->SetRelTol(linsol_rtol);
      cg->SetAbsTol(0.0);
      cg->SetPrintLevel(verbosity_level >= 2 ? 3 : -1);
      S = cg;
   }
   else
   {
      MINRESSolver *minres = new MINRESSolver;
      minres->SetMaxIter(max_lin_iter);
      minres->SetRelTol(linsol_rtol);
      minres->SetAbsTol(0.0);
      minres->SetPrintLevel(verbosity_level >= 2 ? 3 : -1);
      S = minres;
   }

   // 18. Compute the minimum det(J) of the starting mesh->
   tauval = infinity();
   const int NE = mesh->GetNE();
   for (int i = 0; i < NE; i++)
   {
      ElementTransformation *transf = mesh->GetElementTransformation(i);
      for (int j = 0; j < ir->GetNPoints(); j++)
      {
         transf->SetIntPoint(&ir->IntPoint(j));
         tauval = min(tauval, transf->Jacobian().Det());
      }
   }
   cout << "Minimum det(J) of the original mesh is " << tauval << endl;

   // 19. Finally, perform the nonlinear optimization.
   NewtonSolver *newton = NULL;
   if (tauval > 0.0)
   {
      tauval = 0.0;
      TMOPNewtonSolver *tns = new TMOPNewtonSolver(*ir);
      newton = tns;
      cout << "TMOPNewtonSolver is used (as all det(J) > 0).\n";
   }
   else
   {
      if ( (dim == 2 && metric_id != 22 && metric_id != 252) ||
           (dim == 3 && metric_id != 352) )
      {
         cout << "The mesh is inverted. Use an untangling metric." << endl;
         return 3;
      }
      tauval -= 0.01 * h0.Min(); // Slightly below minJ0 to avoid div by 0.
      newton = new TMOPDescentNewtonSolver(*ir);
      cout << "The TMOPDescentNewtonSolver is used (as some det(J) < 0).\n";
   }
   newton->SetPreconditioner(*S);
   newton->SetMaxIter(newton_iter);
   newton->SetRelTol(newton_rtol);
   newton->SetAbsTol(0.0);
   newton->SetPrintLevel(verbosity_level >= 1 ? 1 : -1);;

   // 20. AMR based size refinemenet if a size metric is used
   TMOPEstimator       tmope(fespace, *he_nlf_integ, x, amrmetric_id);
   TMOPRefiner         tmopr(tmope, dim);
   //TMOPDeRefiner       tmopdr(tmope, dim);
   TMOPTypeRefiner     tmoptr(tmope, dim);
   TMOPTypeDeRefiner   tmoptdr(tmope, dim);
   TMOPAMRUpdate tmopamrupdate;
   tmopamrupdate.AddFESpace(&ind_fes);
   tmopamrupdate.AddFESpace(&fespace);
   tmopamrupdate.AddFESpace(&ind_fesv);
   tmopamrupdate.AddGF(&size);
   tmopamrupdate.AddGF(&aspr);
   tmopamrupdate.AddGF(&aspr3d);
   tmopamrupdate.AddMeshNodeAr(&x);
   tmopamrupdate.AddMeshNodeAr(&x0);

   int metrictype = he_nlf_integ->GetAMRQualityMetric().GetMetricType();
   int num_ref_types = 3 + 4*(dim-2);
   Vector amr_ref_check(num_ref_types);
   amr_ref_check = 0.;
   if (dim == 2)
   {
       if ( (metrictype & 1) && (metrictype & 2)) //Shape
       {
          amr_ref_check(0) = 1; //x
          amr_ref_check(1) = 1; //y
       }
       if (metrictype & 8)  // Size
       {
          amr_ref_check(2) = 1; //xy
       }
   }
   else
   {
       if ( (metrictype & 1) && (metrictype & 2)) //Shape
       {
          amr_ref_check(0) = 1; //x
          amr_ref_check(1) = 1; //y
          amr_ref_check(3) = 1; //z
          amr_ref_check(2) = 1; //xy
          amr_ref_check(4) = 1; //xz
          amr_ref_check(5) = 1; //yz
       }
       if (metrictype & 8)  // Size
       {
          amr_ref_check(6) = 1; //xy
       }
   }


   Array<Vector *> amrevecararay;

   int newtonstop = 0;

   if (amr_flag==1)
   {
      int ni_limit = 3; //Newton + AMR
      int nic_limit = std::max(ni_limit, 4); //Number of iterations with AMR
      int amrstop = 0;
      int nc_limit = 1; //AMR per iteration - FIXED FOR NOW

      tmopr.PreferNonconformingRefinement();
      tmopr.SetNCLimit(nc_limit);

      for (int it = 0; it<ni_limit; it++)
      {

         std::cout << it << " Begin NEWTON+AMR Iteration\n";

         newton->SetOperator(a);
         newton->Mult(b, x.GetTrueVector());
         x.SetFromTrueVector();
         if (newton->GetConverged() == false)
         {
            cout << "NewtonIteration: rtol = " << newton_rtol << " not achieved."
                 << endl;
         }
         else
         {
            cout << " NewtonSolver converged" << endl;
         }
         if (amrstop==1)
         {
            newtonstop = 1;
            cout << it << " Newton and AMR have converged" << endl;
            break;
         }
         char title1[10];
         sprintf(title1, "%s %d","Newton", it);

         for (int amrit=0; amrit<nc_limit; amrit++)
         {
            // need to remap discrete functions from old mesh to new mesh here
            if (target_id == 5) { tcd->GetSerialDiscreteTargetSize(size); }
            else if (target_id ==7) { tcd->GetSerialDiscreteTargetAspectRatio(aspr3d); }

            {
               // do all different refinements and measure energy

                int NEorig = fespace.GetNE();
                amrevecararay.DeleteAll();
                for (int i = 0; i < num_ref_types; i++)
                {
                   amrevecararay.Append(new Vector(NEorig));
                }
                Vector baseenergyvec(NEorig);
                tmope.GetAMRTypeEnergy(baseenergyvec);

               std::cout << 0 << " " <<  mesh->GetNE() << " " << a.GetGridFunctionEnergy(x)
                         << " k10basenergy\n";
               for (int i = 1;i < num_ref_types+1 && amrstop==0; i++)
               {
                   if (amr_ref_check(i-1)==1) {
                       tmoptr.Reset();
                       tmoptr.SetRefType(i);
                       if (nc_limit!=0 && amrstop==0) {tmoptr.Apply(*mesh);}
                       tmopamrupdate.hrupdateFEandGF();
                       if (target_id >= 5)
                       {
                          tmopamrupdate.hrupdatediscfunction(*tcd, size, aspr, aspr3d, target_id);
                          target_c = tcd;
                          he_nlf_integ->UpdateTargetConstructor(target_c);
                       }
                       tmopamrupdate.hrupdate(a, *mesh, fespace, move_bnd);

                       Vector amrevecin(fespace.GetNE());
                       tmope.GetAMRTypeEnergy(amrevecin);

                       tmoptdr.Reset();
                       tmoptdr.SetRefType(i);
                       tmoptdr.DetermineAMRTypeEnergy(*mesh, amrevecin, *amrevecararay[i-1]);
                       *amrevecararay[i-1] -= baseenergyvec;
                       *amrevecararay[i-1] *= -1.;

                       if (nc_limit!=0 && amrstop==0) { tmoptdr.Apply(*mesh); }
                       tmopamrupdate.hrupdateFEandGF();
                       if (target_id >= 5)
                       {
                          tmopamrupdate.hrupdatediscfunction(*tcd, size, aspr, aspr3d, target_id);
                          target_c = tcd;
                          he_nlf_integ->UpdateTargetConstructor(target_c);
                       }
                       tmopamrupdate.hrupdate(a, *mesh, fespace, move_bnd);
                       std::cout << i << " " <<  mesh->GetNE() << " " << NEorig
                                 << " k10derefinement\n";
                   }
                   else {
                       *amrevecararay[i-1] = -1.*std::numeric_limits<float>::max();
                   }

               }
            }

            tmope.Setevecptr(amrevecararay);

            {
               // Refiner
               tmopr.Reset();
               if (nc_limit!=0 && amrstop==0) {tmopr.Apply(*mesh);}
               //Update stuff
               tmopamrupdate.hrupdateFEandGF();
               if (target_id >= 5)
               {
                  tmopamrupdate.hrupdatediscfunction(*tcd, size, aspr, aspr3d, target_id);
                  target_c = tcd;
                  he_nlf_integ->UpdateTargetConstructor(target_c);
               }
               tmopamrupdate.hrupdate(a, *mesh, fespace, move_bnd);
            }

            if (amrstop==0)
            {
               if (tmopr.Stop())
               {
                  newtonstop = 1;
                  amrstop = 1;
                  cout << it << " " << amrit <<
                       " AMR stopping criterion satisfied. Stop." << endl;
               }
               else
               {std::cout << mesh->GetNE() << " Number of elements after AMR\n";}
            }
         }
         if (it==nic_limit-1) { amrstop=1; }

         sprintf(title1, "%s %d","AMR", it);
      } //ni_limit
   } //amr_flag==1
   if (newtonstop == 0)
   {
      newton->SetOperator(a);
      newton->Mult(b, x.GetTrueVector());
      x.SetFromTrueVector();
   }
   // 21. Save the optimized mesh to a file. This output can be viewed later
   //     using GLVis: "glvis -m optimized.mesh".
   {
      ofstream mesh_ofs("optimized.mesh");
      mesh_ofs.precision(14);
      mesh->Print(mesh_ofs);
   }
   {
      char numstr[1]; // enough to hold all numbers up to 64-bits
      sprintf(numstr, "%s%d%s", "optimized_ht_", hessiantype, ".mesh");
      ofstream mesh_ofs(numstr);
      mesh_ofs.precision(14);
      mesh->Print(mesh_ofs);
   }

   // 22. Compute the amount of energy decrease.
   const double fin_energy = a.GetGridFunctionEnergy(x);
   double metric_part = fin_energy;
   if (lim_const != 0.0)
   {
      lim_coeff.constant = 0.0;
      metric_part = a.GetGridFunctionEnergy(x);
      lim_coeff.constant = lim_const;
   }
   cout << "Initial strain energy: " << init_energy
        << " = metrics: " << init_energy
        << " + limiting term: " << 0.0 << endl;
   cout << "  Final strain energy: " << fin_energy
        << " = metrics: " << metric_part
        << " + limiting term: " << fin_energy - metric_part << endl;
   cout << "The strain energy decreased by: " << setprecision(12)
        << (init_energy - fin_energy) * 100.0 / init_energy << " %." << endl;

   // 22. Visualize the final mesh and metric values.
   if (visualization)
   {
      char title[] = "Final metric values";
      vis_tmop_metric_s(mesh_poly_deg, *metric, *target_c, *mesh, title, 600);
   }

   // 23. Visualize the mesh displacement.
   if (visualization)
   {
      osockstream sock(19916, "localhost");
      sock << "solution\n";
      mesh->Print(sock);
      x0 -= x;
      x0.Save(sock);
      sock.send();
      sock << "window_title 'Displacements'\n"
           << "window_geometry "
           << 1200 << " " << 0 << " " << 600 << " " << 600 << "\n"
           << "keys jRmclA" << endl;
   }

   // 24. Free the used memory.

   delete newton;
   delete S;
   delete coeff1;
   delete target_c;
   delete adapt_coeff;
   delete metric;

   return 0;
}
