// Mirror Descent for Compliant Mechanism


#include "mfem.hpp"
#include <iostream>
#include <fstream>
#include "compMech.hpp"

using namespace std;
using namespace mfem;


enum LineSearchMethod
{
   ArmijoBackTracking,
   BregmanBBBackTracking,
   Linear
};

enum Problem
{
   Inverter
};
enum BdrType
{
   Fixed,
   XRoller,
   YRoller,
   Input,
   Output,
   Free,
   NumBdr
};

int main(int argc, char *argv[])
{

   // 1. Parse command-line options.
   int ref_levels = 1;
   int order = 1;
   double alpha = 1.0;
   double epsilon = 1e-2;
   double vol_fraction = 0.3;
   int max_it = 1e3;
   double itol = 1e-09;
   double ntol = 1e-6;
   double rho_min = 1e-6;
   double exponent = 3.0;
   // double lambda = 1.0;
   // double mu = 1.0;

   double E = 1;
   double nu = 1/3;
   double lambda = E*nu/((1+nu)*(1-nu));
   double mu = 3*E/(4*(1+nu));

   double c1 = 1e-04;
   bool glvis_visualization = true;
   bool save = true;

   ostringstream solfile, solfile2, meshfile;

   int lineSearchMethod = LineSearchMethod::BregmanBBBackTracking;
   int problem = Problem::Inverter;

   OptionsParser args(argc, argv);
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&problem, "-p", "--problem",
                  "Problem number: 0) Inverter");
   args.AddOption(&lineSearchMethod, "-lm", "--line-method",
                  "Line Search Method: 0) BackTracking, 1) BregmanBB, 2) BregmanBB + BackTracking.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&alpha, "-alpha", "--alpha-step-length",
                  "Step length for gradient descent.");
   args.AddOption(&epsilon, "-epsilon", "--epsilon-thickness",
                  "Length scale for ρ.");
   args.AddOption(&max_it, "-mi", "--max-it",
                  "Maximum number of gradient descent iterations.");
   args.AddOption(&ntol, "-ntol", "--rel-tol",
                  "Normalized exit tolerance.");
   args.AddOption(&itol, "-itol", "--abs-tol",
                  "Increment exit tolerance.");
   args.AddOption(&vol_fraction, "-vf", "--volume-fraction",
                  "Volume fraction for the material density.");
   args.AddOption(&lambda, "-lambda", "--lambda",
                  "Lamé constant λ.");
   args.AddOption(&mu, "-mu", "--mu",
                  "Lamé constant μ.");
   args.AddOption(&rho_min, "-rmin", "--psi-min",
                  "Minimum of density coefficient.");
   args.AddOption(&glvis_visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(mfem::out);
      return 1;
   }
   args.PrintOptions(mfem::out);

   Mesh mesh;
   Array2D<int> ess_bdr;
   Array<int> input_bdr(BdrType::NumBdr), output_bdr(BdrType::NumBdr);
   Array<int> ess_bdr_filter(BdrType::NumBdr);
   ess_bdr_filter = 0; input_bdr = 0; output_bdr = 0;
   double input_spring, output_spring;
   Vector input_direction, output_direction;
   string mesh_file;
   switch (problem)
   {
      case Problem::Inverter:
      {
         //                      o o o o o o o o o o o o roller (4)
         //                      -----------------------
         // input (0.1cm) (2) -> |                     | <- output (0.1cm) (3)
         //                      |                     |
         //    fixed (0.1cm) (1) |                     |
         //                      -----------------------
         mesh = mesh.MakeCartesian2D(160, 80, mfem::Element::Type::QUADRILATERAL, true,
                                     2.0, 1.0);
         double len = 0.025;
         input_spring = 1;
         output_spring = 1;
         input_direction.SetSize(2); input_direction = 0.0;
         output_direction.SetSize(2); output_direction = 0.0;
         input_direction[0] = 1.0;
         output_direction[0] = -1.0;
         ess_bdr.SetSize(3, BdrType::NumBdr); ess_bdr = 0;
         ess_bdr(1, BdrType::XRoller) = 1; // roller - y directional fixed
         ess_bdr(2, BdrType::Fixed) = 1; // fixed
         input_bdr[BdrType::Input] = 1; output_bdr[BdrType::Output] = 1;
         for (int i = 0; i<mesh.GetNBE(); i++)
         {
            int atr = mesh.GetBdrAttribute(i);
            if (atr == 1)
            {
               mesh.SetBdrAttribute(i, BdrType::Free + 1);
               continue;
            }
            else if (atr == 3)
            {
               mesh.SetBdrAttribute(i, BdrType::XRoller + 1);
               continue;
            }

            Element * be = mesh.GetBdrElement(i);
            Array<int> vertices;
            be->GetVertices(vertices);
            double * coords1 = mesh.GetVertex(vertices[0]);
            double * coords2 = mesh.GetVertex(vertices[1]);

            double y = 0.5*(coords1[1] + coords2[1]);
            if (atr == 4)
            {
               if (y < len)
               {
                  mesh.SetBdrAttribute(i, BdrType::Fixed + 1);
                  continue;
               }
               else if (y > 1 - len)
               {
                  mesh.SetBdrAttribute(i, BdrType::Input + 1);
                  continue;
               }
            }
            else if (atr == 2)
            {
               if (y > 1 - len)
               {
                  mesh.SetBdrAttribute(i, BdrType::Output + 1);
                  continue;
               }
            }
            mesh.SetBdrAttribute(i, BdrType::Free + 1);
         }
         solfile << "Inverter-";
         solfile2 << "Inverter-";
         meshfile << "Inverter";
         break;
      }
      default:
      {
         mfem_error("Undefined problem.");
      }
   }
   mesh.SetAttributes();

   int dim = mesh.Dimension();
   double h = std::pow(mesh.GetElementVolume(0), 1.0 / dim);

   // 3. Refine the mesh.
   for (int lev = 0; lev < ref_levels; lev++)
   {
      mesh.UniformRefinement();
      h *= 0.5;
   }
   epsilon = 2.5 * h / 2 / std::sqrt(3);

   // 4. Define the necessary finite element spaces on the mesh.
   H1_FECollection state_fec(order, dim); // space for u
   H1_FECollection filter_fec(order, dim); // space for ρ̃
   L2_FECollection control_fec(order-1, dim,
                               BasisType::GaussLobatto); // space for ψ
   FiniteElementSpace state_fes(&mesh, &state_fec,dim);
   FiniteElementSpace filter_fes(&mesh, &filter_fec);
   FiniteElementSpace control_fes(&mesh, &control_fec);

   int state_size = state_fes.GetTrueVSize();
   int control_size = control_fes.GetTrueVSize();
   int filter_size = filter_fes.GetTrueVSize();
   mfem::out << "Number of state unknowns: " << state_size << std::endl;
   mfem::out << "Number of filter unknowns: " << filter_size << std::endl;
   mfem::out << "Number of control unknowns: " << control_size << std::endl;

   // 5. Set the initial guess for ρ.
   GridFunction psi(&control_fes);
   GridFunction psi_old(&control_fes);
   psi = inv_sigmoid(vol_fraction);
   psi_old = inv_sigmoid(vol_fraction);

   // ρ = sigmoid(ψ)
   MappedGridFunctionCoefficient rho(&psi, sigmoid);
   // Interpolation of ρ = sigmoid(ψ) in control fes (for ParaView output)
   GridFunction rho_gf(&control_fes);
   // ρ - ρ_old = sigmoid(ψ) - sigmoid(ψ_old)
   DiffMappedGridFunctionCoefficient succ_diff_rho(&psi, &psi_old, sigmoid);
   LinearForm succ_diff_rho_form(&control_fes);
   succ_diff_rho_form.AddDomainIntegrator(new DomainLFIntegrator(succ_diff_rho));

   // 9. Define some tools for later
   ConstantCoefficient one(1.0);
   GridFunction zero_gf(&control_fes);
   zero_gf = 0.0;
   LinearForm vol_form(&control_fes);
   vol_form.AddDomainIntegrator(new DomainLFIntegrator(one));
   vol_form.Assemble();
   double domain_volume = vol_form.Sum();
   const double target_volume = domain_volume * vol_fraction;
   ConstantCoefficient lambda_cf(lambda), mu_cf(mu);
   ConstantCoefficient input_spring_cf(input_spring),
                       output_spring_cf(output_spring);
   VectorConstantCoefficient input_direction_cf(input_direction),
                             output_direction_cf(output_direction);
   CompliantMechanism obj(&lambda_cf, &mu_cf, epsilon,
                          &rho, target_volume,
                          ess_bdr, input_bdr, output_bdr,
                          input_spring_cf, output_spring_cf,
                          input_direction_cf, output_direction_cf,
                          &state_fes,
                          &filter_fes, exponent, rho_min);
   obj.SetGridFunction(&psi);
   LineSearchAlgorithm *lineSearch;
   switch (lineSearchMethod)
   {
      case LineSearchMethod::ArmijoBackTracking:
         lineSearch = new BackTracking(obj, alpha, 2.0, c1, 10, infinity());
         solfile << "EXP-";
         solfile2 << "EXP-";
         break;
      case LineSearchMethod::BregmanBBBackTracking:
         lineSearch = new BackTrackingLipschitzBregmanMirror(
            obj, succ_diff_rho_form, *(obj.Gradient()), psi, c1, 1.0, 1e-10, infinity());
         solfile << "BB-";
         solfile2 << "BB-";
         break;
      default:
         mfem_error("Undefined linesearch method.");
   }
   solfile << "0.gf";
   solfile2 << "f.gf";
   meshfile << ".mesh";

   MappedGridFunctionCoefficient &designDensity = obj.GetDesignDensity();
   GridFunction designDensity_gf(&filter_fes);
   designDensity_gf = pow(vol_fraction, exponent);
   // designDensity_gf.ProjectCoefficient(designDensity);

   GridFunction &u = *obj.GetDisplacement();

   // 10. Connect to GLVis. Prepare for VisIt output.
   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sout_SIMP, sout_r;
   if (glvis_visualization)
   {
      sout_SIMP.open(vishost, visport);
      sout_SIMP.precision(8);
      sout_SIMP << "solution\n" << mesh << designDensity_gf
                << "window_title 'Design density r(ρ̃) - MD "
                << problem << " " << lineSearchMethod << "'\n"
                << "keys Rjl***************\n"
                << flush;
      sout_r.open(vishost, visport);
      sout_r.precision(8);
      sout_r << "solution\n" << mesh << rho_gf
             << "window_title 'Raw density ρ - MD "
             << problem << " " << lineSearchMethod << "'\n"
             << "keys Rjl***************\n"
             << flush;
   }

   mfem::ParaViewDataCollection paraview_dc("ex37", &mesh);
   ofstream mesh_ofs(meshfile.str().c_str());
   mesh_ofs.precision(8);
   mesh.Print(mesh_ofs);
   // 11. Iterate
   obj.Eval();
   GridFunction d(&control_fes);
   int k;
   for (k = 1; k <= max_it; k++)
   {
      // mfem::out << "\nStep = " << k << std::endl;

      d = *obj.Gradient();
      d.Neg();
      double output_displacement = lineSearch->Step(psi, d);
      double norm_increment = zero_gf.ComputeLpError(1, succ_diff_rho);
      psi_old = psi;

      // mfem::out << "volume fraction = " <<  obj.GetVolume() / domain_volume <<
      //           std::endl;
      mfem::out <<  ", " << output_displacement << ", ";
      mfem::out << norm_increment << std::endl;

      if (glvis_visualization)
      {
         designDensity_gf.ProjectCoefficient(designDensity);
         sout_SIMP << "solution\n" << mesh << designDensity_gf
                   << flush;
         rho_gf.ProjectCoefficient(rho);
         sout_r << "solution\n" << mesh << rho_gf
                << flush;
      }
      if (norm_increment < itol)
      {
         break;
      }
   }
   if (save)
   {
      ofstream sol_ofs(solfile.str().c_str());
      sol_ofs.precision(8);
      sol_ofs << psi;

      ofstream sol_ofs2(solfile2.str().c_str());
      sol_ofs2.precision(8);
      sol_ofs2 << *obj.GetFilteredDensity();
   }
   out << "Total number of iteration = " << k << std::endl;
   delete lineSearch;

   return 0;
}
