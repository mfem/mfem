// Implements gradient-based MMPDE method for solving the nonlinear optimization
// problem in TMOP adaptivity

#include "mfem.hpp"
//#include "fem.hpp"
#include <fstream>
#include <iostream>
#include <ctime>
#include <typeinfo>

using namespace mfem;
using namespace std;


double weight_fun(const Vector &x);

// Metric values are visualized by creating an L2 finite element functions and
// computing the metric values at the nodes.
void vis_metric(int order, TMOP_QualityMetric &qm, const TargetConstructor &tc,
                Mesh &mesh, char *title, int position)
{
   L2_FECollection fec(order, mesh.Dimension(), BasisType::GaussLobatto);
   FiniteElementSpace fes(&mesh, &fec, 1);
   GridFunction metric(&fes);
   InterpolateTMOP_QualityMetric(qm, tc, mesh, metric);
   osockstream sock(19916, "localhost");
   sock << "solution\n";
   mesh.Print(sock);
   metric.Save(sock);
   sock.send();
   sock << "window_title '"<< title << "'\n"
        << "window_geometry "
        << position << " " << 0 << " " << 600 << " " << 600 << "\n"
        << "keys jRmclA" << endl;
}

// Define the TimeDependentOperator class needed for the ODESolver
class MMPDEOperator : public TimeDependentOperator
{
protected:
   FiniteElementSpace &fespace;
   ConstantCoefficient visc;
   NonlinearForm F;

   int ode_mode;
   bool move_bnd;
   mutable Vector zh, zs; // auxiliary vectors

public:
   MMPDEOperator(FiniteElementSpace &f, TMOP_Integrator *he_nlf_integ,
                        Array<int> &ess_bdr,
                        double viscosity, int _ode_mode = 0, bool move_bnd = true);
   //inputs include TMOP_Integrator as defined in mesh-optimizer

   virtual void Mult(const Vector &vx, Vector &dvx_dt) const;

   double GetEnergy(Vector &x) const;
   //virtual void SetEssentialBC(const Array<int> &bdr_attr_is_ess,
     //                              Vector *rhs);
   //virtual void SetEssentialVDofs(const Array<int> &ess_vdofs_list);

};

class MMPDEOperatorCombo : public TimeDependentOperator
{
protected:
   FiniteElementSpace &fespace;
   ConstantCoefficient visc;
   NonlinearForm F;

   int ode_mode;
   bool move_bnd;
   mutable Vector zh, zs; // auxiliary vectors

public:
   MMPDEOperatorCombo(FiniteElementSpace &f, TMOP_Integrator *he_nlf_integ, TMOP_Integrator *he_nlf_integ2,
                        Array<int> &ess_bdr,
                        double viscosity, int _ode_mode = 0, bool move_bnd = true);
   //inputs include TMOP_Integrator as defined in mesh-optimizer

   virtual void Mult(const Vector &vx, Vector &dvx_dt) const;

   double GetEnergy(Vector &x) const;
   //virtual void SetEssentialBC(const Array<int> &bdr_attr_is_ess,
     //                              Vector *rhs);
   //virtual void SetEssentialVDofs(const Array<int> &ess_vdofs_list);

};

IntegrationRules IntRulesLo(0, Quadrature1D::GaussLobatto);
IntegrationRules IntRulesCU(0, Quadrature1D::ClosedUniform);



int main (int argc, char *argv[])
{
   // Set the method's default parameters, same as mesh-optimizer 
      // "../../data/escher.mesh"
   //const char *mesh_file = "icf.mesh";
   const char *mesh_file = "../../data/star.mesh";
   int mesh_poly_deg     = 2;
   int rs_levels         = 2;
   double jitter         = 0.1;
   int metric_id         = 7;
   int target_id         = 1;


   double lim_const      = 0.0;
   int quad_type         = 1;
   int quad_order        = 8;
   bool move_bnd         = false;
   bool combomet         = false;
   bool normalization    = false;
   bool visualization    = true;
   int verbosity_level   = 0;   

   double rel_tol        = 1e-7;
   double dt             = 1e-6;
   int max_it            = 10000;
   int ode_mode          = 1;

   // Parse command-line options.
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&mesh_poly_deg, "-o", "--order",
                  "Polynomial degree of mesh finite element space.");
   args.AddOption(&rs_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&jitter, "-ji", "--jitter",
                  "Random perturbation scaling factor.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step size.");
   args.AddOption(&rel_tol, "-rt", "--relative-tolerance",
                  "Relative tolerance.");
   args.AddOption(&metric_id, "-mid", "--metric-id",
                  "Mesh optimization metric:\n\t"
                  "1  : |T|^2                          -- 2D shape\n\t"
                  "2  : 0.5|T|^2/tau-1                 -- 2D shape (condition number)\n\t"
                  "7  : |T-T^-t|^2                     -- 2D shape+size\n\t"
                  "9  : tau*|T-T^-t|^2                 -- 2D shape+size\n\t"
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
                  "3: Ideal shape, initial size");
   args.AddOption(&lim_const, "-lc", "--limit-const", "Limiting constant.");
   args.AddOption(&quad_type, "-qt", "--quad-type",
                  "Quadrature rule type:\n\t"
                  "1: Gauss-Lobatto\n\t"
                  "2: Gauss-Legendre\n\t"
                  "3: Closed uniform points");
   args.AddOption(&quad_order, "-qo", "--quad_order",
                  "Order of the quadrature rule.");
   args.AddOption(&move_bnd, "-bnd", "--move-boundary", "-fix-bnd",
                  "--fix-boundary",
                  "Enable motion along horizontal and vertical boundaries.");
   args.AddOption(&combomet, "-cmb", "--combo-met", "-no-cmb", "--no-combo-met",
                  "Combination of metrics.");
   args.AddOption(&normalization, "-nor", "--normalization", "-no-nor",
                  "--no-normalization",
                  "Make all terms in the optimization functional unitless.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&verbosity_level, "-vl", "--verbosity-level",
                  "Set the verbosity level - 0, 1, or 2.");
   args.AddOption(&ode_mode, "-om", "--ode-mode",
                  "Set the ODE mode - 0, or 1.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   std::clock_t c_start = std::clock();

   // Initialize mesh and finite element space

   Mesh *mesh = new Mesh(mesh_file, 1, 1, false);
   for (int lev = 0; lev < rs_levels; lev++) { mesh->UniformRefinement(); }

   int dim = mesh->Dimension();
   mesh->PrintCharacteristics();

   GridFunction *init_nodes = mesh->GetNodes();
   FiniteElementCollection *poly_fec;

   if(mesh_poly_deg == 1)
   {
      FiniteElementCollection *linear_fec = new LinearFECollection;
      FiniteElementSpace *linear_fespace =
         new FiniteElementSpace(mesh, linear_fec, dim);
      mesh->SetNodalFESpace(linear_fespace);
      init_nodes = mesh->GetNodes();
      init_nodes->MakeOwner(linear_fec);
   }
   else
   {
      poly_fec = new H1_FECollection(mesh_poly_deg, dim); 
      FiniteElementSpace *fespace = new FiniteElementSpace(mesh, poly_fec, dim);
      mesh->SetNodalFESpace(fespace);
      init_nodes = mesh->GetNodes();
      init_nodes->MakeOwner(poly_fec);
   }
      
   FiniteElementSpace &fespace = *init_nodes->FESpace();


   // v is velocity, x is nodal position, written as one vector vx
      // same as hyperelast-mesh-optimization (Veselin)
   int fes_size = fespace.GetVSize();
   Vector vx(2*fes_size);
   GridFunction v, x;
   v.MakeRef(&fespace, vx, 0);
   x.MakeRef(&fespace, vx, fes_size);

   // Initial conditions for v and x
   v = 0.0;
   x = *init_nodes;

   //set up random perturbation
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

   // Random perturbation of the nodes.
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

   // Set the perturbation of all nodes from the true nodes.
   x.SetTrueVector();
   x.SetFromTrueVector();

   GridFunction *nodes = &x;
   int mesh_owns_nodes = 0;
   mesh->SwapNodes(nodes, mesh_owns_nodes);

   // Boundary conditions based on the boundary attributes
   Array<int> ess_bdr(fespace.GetMesh()->bdr_attributes.Max());
   ess_bdr = 1; // all boundary attributes are fixed

   GridFunction x0(&fespace);
   x0 = x;

   // Form the integrator that uses the chosen metric and target.
   double tauval = -0.1;
   TMOP_QualityMetric *metric = NULL;
   switch (metric_id)
   {
      case 1: metric = new TMOP_Metric_001; break;
      case 2: metric = new TMOP_Metric_002; break;
      case 7: metric = new TMOP_Metric_007; break;
      case 9: metric = new TMOP_Metric_009; break;
      case 22: metric = new TMOP_Metric_022(tauval); break;
      case 50: metric = new TMOP_Metric_050; break;
      case 55: metric = new TMOP_Metric_055; break;
      case 56: metric = new TMOP_Metric_056; break;
      case 58: metric = new TMOP_Metric_058; break;
      case 77: metric = new TMOP_Metric_077; break;
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
   TargetConstructor::TargetType target_t;
   switch (target_id)
   {
      case 1: target_t = TargetConstructor::IDEAL_SHAPE_UNIT_SIZE; break;
      case 2: target_t = TargetConstructor::IDEAL_SHAPE_EQUAL_SIZE; break;
      case 3: target_t = TargetConstructor::IDEAL_SHAPE_GIVEN_SIZE; break;
      default: cout << "Unknown target_id: " << target_id << endl;
         delete metric; return 3;
   }
   TargetConstructor *target_c = new TargetConstructor(target_t);
   target_c->SetNodes(x0);
   TMOP_Integrator *he_nlf_integ = new TMOP_Integrator(metric, target_c); 
   //TMOP Integrator used as input into class 


   // Setup the quadrature rule for the non-linear form integrator.
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

   // Gradient-based ode_mode = 1
   // Velocity-based ode_mode = 0 (not yet implemented)
   
   //cout << "enter ODE mode [0/1] --> " << flush;
   //cin >> ode_mode;
   double visc = 0.;
   if (ode_mode == 0)
   {
      cout << "enter viscosity --> " << flush;
      cin >> visc;
   }


      // Weight of the original metric.
 /*  ConstantCoefficient *coeff1 = new ConstantCoefficient(1.0);
   he_nlf_integ->SetCoefficient(*coeff1);

   TMOP_QualityMetric *metric2 = new TMOP_Metric_077;
   TargetConstructor *target_c2 = new TargetConstructor(
      TargetConstructor::IDEAL_SHAPE_EQUAL_SIZE);
   target_c2->SetVolumeScale(0.01);
   target_c2->SetNodes(x0);
   TMOP_Integrator *he_nlf_integ2 = new TMOP_Integrator(metric2, target_c2);
   he_nlf_integ2->SetIntegrationRule(*ir);

      // Weight of metric2.
   FunctionCoefficient coeff2(weight_fun);
   he_nlf_integ2->SetCoefficient(coeff2);
   MMPDEOperatorCombo oper(fespace, he_nlf_integ, he_nlf_integ2, ess_bdr, visc, ode_mode, move_bnd);
*/
   MMPDEOperator oper(fespace, he_nlf_integ, ess_bdr, visc, ode_mode, move_bnd);


      if (visualization)
   {
      char title[] = "Initial metric values";
      vis_metric(mesh_poly_deg, *metric, *target_c, *mesh, title, 0);
   }

   ODESolver *ode_solver;
   int ode_solver_type = 4; //Using RK4

 /*  cout <<
      "choose an ode solver:\n"
      "1) forward Euler\n"
      "2) RK2 (midpoint method)\n"
      "3) RK3 (SSP method)\n"
      "4) RK4\n"
      "5) RK6\n"
      "6) RK8\n"
      " --> " << flush;
   cin >> ode_solver_type;*/
   switch (ode_solver_type)
   {
   case 1:
      ode_solver = new ForwardEulerSolver;
      break;
   case 2:
      ode_solver = new RK2Solver(0.5); // midpoint method
      break;
   case 3:
      ode_solver = new RK3SSPSolver;
      break;
   case 4:
      ode_solver = new RK4Solver;
      break;
   case 5:
      ode_solver = new RK6Solver;
      break;
   case 6:
   default:
      ode_solver = new RK8Solver;
      break;
   }

   ode_solver->Init(oper);

   double t = 0.0;

   // Compute the minimum det(J) of the starting mesh.
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
   double tauval_initial = tauval;
   //cout << "Minimum det(J) of the original mesh is " << tauval_initial << endl;

   //visualization
   bool vis_velocity = false;

   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream vis(vishost, visport);
   bool take_screenshots = false;
   const char ss_prefix[] = "mmpde-pt-";
   if (vis.good())
   {
      vis.precision(8);
      vis << "solution\n";
      mesh->Print(vis);
      v.Save(vis);
      vis << "window_title 'Initial Mesh'\n";
      if (dim == 2)
      {
         vis << "view 0 0\n";
         vis << "keys jl\n";
      }
      vis << "keys cm\n";
      vis << "autoscale value\n";
      
      vis << "autopause on\n";
      vis << flush;
      if (take_screenshots)
         vis << "screenshot " << ss_prefix << setfill('0') << setw(6)
             << 0 << ".png" << endl;
   }

   double old_t = t, old_delta_te = 0.0;
   Vector old_vx(vx);
   double old_diff_norm = 10000.0;
   double old_norm = x.Norml2();

   double F_initial_value = oper.GetEnergy(x);
   double F_value_old = 100.0;

   for (int i = 1; i < max_it; i++)
   {
      ode_solver->Step(vx, t, dt);

      double new_norm = x.Norml2();
      double F_value = oper.GetEnergy(x);
      double diff_norm = fabs(old_norm - new_norm);

      cout.precision(10);
      
      if ( diff_norm < rel_tol )
      {
         std::cout << "Converged!" << std::endl;
         std::cout << "Final difference norm: " << diff_norm << std::endl;
         std::clock_t c_end = std::clock();
         double time_elapsed_ms = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC;
         std::cout << "CPU time used: " << time_elapsed_ms / 1000.0 << " s\n";
         std::cout << "Initial F: " << F_initial_value << std::endl;
         std::cout << "Final F: " << F_value << std::endl;
         break;
      }
      else
      {
         std::cout << i << ": " << diff_norm << std::endl;

         old_norm = new_norm;
         old_vx = vx;
         old_t = t;
         old_diff_norm = diff_norm;
         F_value_old = F_value;
         continue;
      }

   }

   // Save the optimized mesh to a file. This output can be viewed later
   //    using GLVis: "glvis -m optimized.mesh".
   {
      ofstream mesh_ofs("optimized.mesh");
      mesh_ofs.precision(14);
      mesh->Print(mesh_ofs);
   }

   // Visualize the final mesh and metric values.
   if (visualization)
   {
      char title[] = "Final metric values MMPDE";
      vis_metric(mesh_poly_deg, *metric, *target_c, *mesh, title, 600);
   }

    cout << "Minimum det(J) of the initial mesh is " << tauval_initial << endl;


      tauval = infinity();
      const int NE2 = mesh->GetNE();
      for (int i = 0; i < NE2; i++)
      {
         ElementTransformation *transf = mesh->GetElementTransformation(i);
         for (int j = 0; j < ir->GetNPoints(); j++)
         {
            transf->SetIntPoint(&ir->IntPoint(j));
            tauval = min(tauval, transf->Jacobian().Det());
         }
      }
      cout << "Minimum det(J) of the final mesh is " << tauval << endl;



   // Visualize the mesh displacement.
   if (visualization)
   {
      x0 -= x;
 
      osockstream sock(19916, "localhost");
      sock << "solution\n";
      mesh->Print(sock);
      x0.Save(sock);
      sock.send();
      sock << "window_title 'Displacements MMPDE'\n"
           << "window_geometry "
           << 1200 << " " << 0 << " " << 600 << " " << 600 << "\n"
           << "keys jRmclA" << endl;
   }

   // 24. Free the used memory.
   delete target_c;
   delete metric;
   delete mesh;

   return 0;
}


// Defined with respect to the icf mesh.
double weight_fun(const Vector &x)
{
   const double r = sqrt(x(0)*x(0) + x(1)*x(1) + 1e-12);
   const double den = 0.002;
   double l2 = 0.2 + 0.5*std::tanh((r-0.16)/den) - 0.5*std::tanh((r-0.17)/den)
               + 0.5*std::tanh((r-0.23)/den) - 0.5*std::tanh((r-0.24)/den);
   return l2;
}

// MMPDEOperator Details
MMPDEOperator::MMPDEOperator(FiniteElementSpace &f,
                                          TMOP_Integrator *he_nlf_integ,
                                           Array<int> &ess_bdr,
                                           double viscosity, int _ode_mode, bool move_bnd)
   : TimeDependentOperator(2*f.GetVSize(), 0.0), fespace(f), F(&fespace), zh(height/2),
     zs(height/2)

{
   Mesh *mesh = fespace.GetMesh(); //Get the mesh
      // Computes reference det(J)
   double ref_dJ; 
   double sum_avg_dJ = 0.0;
   for (int i = 0; i < mesh->GetNE(); i++)
   {
      double vol_i = mesh->GetElementVolume(i);
      int geom = mesh->GetElementBaseGeometry(i);
      DenseMatrix *J = Geometries.GetPerfGeomToGeomJac(geom);
      if (J)
         vol_i *= J->Weight();
      sum_avg_dJ += vol_i/Geometry::Volume[geom];
   }
   ref_dJ = sum_avg_dJ/mesh->GetNE();
   cout << "Reference |J| = " << ref_dJ << endl;

   F.AddDomainIntegrator(he_nlf_integ);
   
   int dim = fespace.GetFE(0)->GetDim();

   if (move_bnd == false)
   {
      Array<int> ess_bdr(mesh->bdr_attributes.Max());
      ess_bdr = 1;
      F.SetEssentialBC(ess_bdr);
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
      F.SetEssentialVDofs(ess_vdofs);
   }

   //F.SetEssentialBC(ess_bdr);                   

   ode_mode = _ode_mode;
}

MMPDEOperatorCombo::MMPDEOperatorCombo(FiniteElementSpace &f,
                                          TMOP_Integrator *he_nlf_integ, TMOP_Integrator *he_nlf_integ2,
                                           Array<int> &ess_bdr,
                                           double viscosity, int _ode_mode, bool move_bnd)
   : TimeDependentOperator(2*f.GetVSize(), 0.0), fespace(f), F(&fespace), zh(height/2),
     zs(height/2)

{
   Mesh *mesh = fespace.GetMesh(); //Get the mesh
      // Computes reference det(J)
   double ref_dJ; 
   double sum_avg_dJ = 0.0;
   for (int i = 0; i < mesh->GetNE(); i++)
   {
      double vol_i = mesh->GetElementVolume(i);
      int geom = mesh->GetElementBaseGeometry(i);
      DenseMatrix *J = Geometries.GetPerfGeomToGeomJac(geom);
      if (J)
         vol_i *= J->Weight();
      sum_avg_dJ += vol_i/Geometry::Volume[geom];
   }
   ref_dJ = sum_avg_dJ/mesh->GetNE();
   cout << "Reference |J| = " << ref_dJ << endl;

   F.AddDomainIntegrator(he_nlf_integ);

   F.AddDomainIntegrator(he_nlf_integ2);

   int dim = fespace.GetFE(0)->GetDim();

   if (move_bnd == false)
   {
      Array<int> ess_bdr(mesh->bdr_attributes.Max());
      ess_bdr = 1;
      F.SetEssentialBC(ess_bdr);
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
      F.SetEssentialVDofs(ess_vdofs);
   }

   ode_mode = _ode_mode;
}


// MMPDEOperator Mult
void MMPDEOperator::Mult(const Vector &vx, Vector &dvx_dt) const
{
   int sc = height/2;
   Vector v(vx.GetData() +  0, sc);
   Vector x(vx.GetData() + sc, sc);
   Vector dv_dt(dvx_dt.GetData() +  0, sc);
   Vector dx_dt(dvx_dt.GetData() + sc, sc);

   if (ode_mode == 0) //Velocity-based MMPDE
   {
      /*F.Mult(x,zh);
      S.Mult(v,sz);
      add(-1,zh,zs,dv_dt); //dv/dt = -(Fx + Sv)
*/
      dx_dt = v; // dx/dt = v
   }
   else //Gradient-based MMPDE
   {
      F.Mult(x,dx_dt); // dF/dx is computed by NonLinearForm::Mult
      dx_dt.Neg(); // dx/dt = -dF/dx

      //dx_dt *= 0.001;

      dv_dt = 0.0;
   }
}


// MMPDEOperator Mult
void MMPDEOperatorCombo::Mult(const Vector &vx, Vector &dvx_dt) const
{
   int sc = height/2;
   Vector v(vx.GetData() +  0, sc);
   Vector x(vx.GetData() + sc, sc);
   Vector dv_dt(dvx_dt.GetData() +  0, sc);
   Vector dx_dt(dvx_dt.GetData() + sc, sc);

   if (ode_mode == 0) //Velocity-based MMPDE
   {
      /*F.Mult(x,zh);
      S.Mult(v,sz);
      add(-1,zh,zs,dv_dt); //dv/dt = -(Fx + Sv)
*/
      dx_dt = v; // dx/dt = v
   }
   else //Gradient-based MMPDE
   {
      F.Mult(x,dx_dt); // dF/dx is computed by NonLinearForm::Mult
      dx_dt.Neg(); // dx/dt = -dF/dx

      //dx_dt *= 0.001;

      dv_dt = 0.0;
   }
}

double MMPDEOperator::GetEnergy(Vector &x) const
{
   return F.GetEnergy(x);
}

double MMPDEOperatorCombo::GetEnergy(Vector &x) const
{
   return F.GetEnergy(x);
}