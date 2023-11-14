//                              MFEM Example 37
//
//
// Compile with: make ex40
//
// Sample runs:
//     ex40 -alpha 10
//     ex40 -alpha 10 -pv
//     ex40 -lambda 0.1 -mu 0.1
//     ex40 -o 2 -alpha 5.0 -mi 50 -vf 0.4 -ntol 1e-5
//     ex40 -r 6 -o 1 -alpha 25.0 -epsilon 0.02 -mi 50 -ntol 1e-5
//
//
// Description: This example code demonstrates the use of MFEM to solve a
//              density-filtered [3] topology optimization problem. The
//              objective is to minimize the output_displacement
//
//                  minimize ∫_Ω f⋅u dx over u ∈ [H¹(Ω)]² and ρ ∈ L¹(Ω)
//
//                  subject to
//
//                    -Div(r(ρ̃)Cε(u)) = f       in Ω + BCs
//                    -ϵ²Δρ̃ + ρ̃ = ρ             in Ω + Neumann BCs
//                    0 ≤ ρ ≤ 1                 in Ω
//                    ∫_Ω ρ dx = θ vol(Ω)
//
//              Here, r(ρ̃) = ρ₀ + ρ̃³ (1-ρ₀) is the solid isotropic material
//              penalization (SIMP) law, C is the elasticity tensor for an
//              isotropic linearly elastic material, ϵ > 0 is the design
//              length scale, and 0 < θ < 1 is the volume fraction.
//
//              The problem is discretized and gradients are computing using
//              finite elements [1]. The design is optimized using an entropic
//              mirror descent algorithm introduced by Keith and Surowiec [2]
//              that is tailored to the bound constraint 0 ≤ ρ ≤ 1.
//
//              This example highlights the ability of MFEM to deliver high-
//              order solutions to inverse design problems and showcases how
//              to set up and solve PDE-constrained optimization problems
//              using the so-called reduced space approach.
//
//
// [1] Andreassen, E., Clausen, A., Schevenels, M., Lazarov, B. S., & Sigmund, O.
//    (2011). Efficient topology optimization in MATLAB using 88 lines of
//    code. Structural and Multidisciplinary Optimization, 43(1), 1-16.
// [2] Keith, B. and Surowiec, T. (2023) Proximal Galerkin: A structure-
//     preserving finite element method for pointwise bound constraints.
//     arXiv:2307.12444 [math.NA]
// [3] Lazarov, B. S., & Sigmund, O. (2011). Filters in topology optimization
//     based on Helmholtz‐type differential equations. International Journal
//     for Numerical Methods in Engineering, 86(6), 765-781.

#include "mfem.hpp"
#include <iostream>
#include <fstream>
#include "ex37.hpp"

using namespace std;
using namespace mfem;

/**
 * ---------------------------------------------------------------
 *                      ALGORITHM PREAMBLE
 * ---------------------------------------------------------------
 *
 *  The Lagrangian for this problem is
 *
 *          L(u,ρ,ρ̃,w,w̃) = (f,u) - (r(ρ̃) C ε(u),ε(w)) + (f,w)
 *                       - (ϵ² ∇ρ̃,∇w̃) - (ρ̃,w̃) + (ρ,w̃)
 *
 *  where
 *
 *    r(ρ̃) = ρ₀ + ρ̃³ (1 - ρ₀)       (SIMP rule)
 *
 *    ε(u) = (∇u + ∇uᵀ)/2           (symmetric gradient)
 *
 *    C e = λtr(e)I + 2μe           (isotropic material)
 *
 *  NOTE: The Lame parameters can be computed from Young's modulus E
 *        and Poisson's ratio ν as follows:
 *
 *             λ = E ν/((1+ν)(1-2ν)),      μ = E/(2(1+ν))
 *
 * ---------------------------------------------------------------
 *
 *  Discretization choices:
 *
 *     u ∈ V ⊂ (H¹)ᵈ (order p)
 *     ψ ∈ L² (order p - 1), ρ = sigmoid(ψ)
 *     ρ̃ ∈ H¹ (order p)
 *     w ∈ V  (order p)
 *     w̃ ∈ H¹ (order p)
 *
 * ---------------------------------------------------------------
 *                          ALGORITHM
 * ---------------------------------------------------------------
 *
 *  Update ρ with projected mirror descent via the following algorithm.
 *
 *  1. Initialize ψ = inv_sigmoid(vol_fraction) so that ∫ sigmoid(ψ) = θ vol(Ω)
 *
 *  While not converged:
 *
 *     2. Solve filter equation ∂_w̃ L = 0; i.e.,
 *
 *           (ϵ² ∇ ρ̃, ∇ v ) + (ρ̃,v) = (ρ,v)   ∀ v ∈ H¹.
 *
 *     3. Solve primal problem ∂_w L = 0; i.e.,
 *
 *      (λ r(ρ̃) ∇⋅u, ∇⋅v) + (2 μ r(ρ̃) ε(u), ε(v)) = (f,v)   ∀ v ∈ V.
 *
 *     NB. The dual problem ∂_u L = 0 is the negative of the primal problem due to symmetry.
 *
 *     4. Solve for filtered gradient ∂_ρ̃ L = 0; i.e.,
 *
 *      (ϵ² ∇ w̃ , ∇ v ) + (w̃ ,v) = (-r'(ρ̃) ( λ |∇⋅u|² + 2 μ |ε(u)|²),v)   ∀ v ∈ H¹.
 *
 *     5. Project the gradient onto the discrete latent space; i.e., solve
 *
 *                         (G,v) = (w̃,v)   ∀ v ∈ L².
 *
 *     6. Bregman proximal gradient update; i.e.,
 *
 *                            ψ ← ψ - αG + c,
 *
 *     where α > 0 is a step size parameter and c ∈ R is a constant ensuring
 *
 *                     ∫_Ω sigmoid(ψ - αG + c) dx = θ vol(Ω).
 *
 *  end
 *
 */


enum Problem
{
   Inverter
};
enum BdrType
{
   Fixed,
   XRoller,
   YRoller,
   ZRoller,
   Input,
   Output,
   Free,
   NumBdr
};

int main(int argc, char *argv[])
{

   // 1. Parse command-line options.
   int ref_levels = 0;
   int order = 1;
   double alpha = 1.0;
   double epsilon = 1e-2;
   double vol_fraction = 0.3;
   int max_it = 1e3;
   double itol = 1e-3;
   double ntol = 1e-6;
   double rho_min = 1e-6;
   double exponent = 3.0;
   double lambda = 1.0;
   double mu = 1.0;
   double c1 = 1e-04;
   double mv = 0.2;
   bool glvis_visualization = true;

   ostringstream solfile, solfile2, meshfile;

   int problem = Problem::Inverter;

   OptionsParser args(argc, argv);
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&problem, "-p", "--refine",
                  "Number of times to refine the mesh uniformly.");
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
   Array<int> input_bdr, output_bdr;
   Array<int> ess_bdr_filter;
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
         const int numel = 80;
         const int numelx(2*numel), numely(numel);
         mesh = mesh.MakeCartesian2D(numelx, numely, mfem::Element::Type::QUADRILATERAL,
                                     true,
                                     (double)numelx, (double)numely);
         input_spring = 1;
         output_spring = 0.0001;
         input_direction.SetSize(2);
         output_direction.SetSize(2);
         input_direction = 0.0; output_direction = 0.0;
         input_direction[0] = 1.0;
         output_direction[0] = 1.0;
         ess_bdr.SetSize(3, BdrType::NumBdr); ess_bdr_filter.SetSize(BdrType::NumBdr);
         input_bdr.SetSize(BdrType::NumBdr); output_bdr.SetSize(BdrType::NumBdr);
         ess_bdr = 0; ess_bdr_filter = 0;
         input_bdr = 0; output_bdr = 0;
         ess_bdr(0, BdrType::YRoller) = 1; // y-roller - x direction fixed
         ess_bdr(1, BdrType::XRoller) = 1; // x-roller - y direction fixed
         ess_bdr(2, BdrType::Fixed) = 1; // all direction fixed
         input_bdr[BdrType::Input] = 1; output_bdr[BdrType::Output] = 1;
         for (int i = 0; i<mesh.GetNBE(); i++)
         {
            Element * be = mesh.GetBdrElement(i);
            Array<int> vertices;
            be->GetVertices(vertices);

            double * coords1 = mesh.GetVertex(vertices[0]);
            double * coords2 = mesh.GetVertex(vertices[1]);

            Vector fc(2);
            fc(0) = 0.5*(coords1[0] + coords2[0]);
            fc(1) = 0.5*(coords1[1] + coords2[1]);
            const double len = 1;
            const double bdrtol = 1e-08;

            switch (be->GetAttribute())
            {
               case 1: // bottom
                  be->SetAttribute(BdrType::Free + 1);
                  break;
               case 2: // right
                  if (fc(1) > numely - len)
                  {
                     be->SetAttribute(BdrType::Output + 1);
                     break;
                  }
                  be->SetAttribute(BdrType::Free + 1);
                  break;
               case 3: // top
                  be->SetAttribute(BdrType::XRoller + 1);
                  break;
               case 4: // left
                  if (fc(1) > numely - len)
                  {
                     be->SetAttribute(BdrType::Input + 1);
                     break;
                  }
                  else if (fc(1) < len)
                  {
                     be->SetAttribute(BdrType::Fixed + 1);
                     break;
                  }
                  be->SetAttribute(BdrType::Free + 1);
                  break;
               default:
                  mfem_error("Something went wrong");
            }
         }
         solfile << "Inverter-";
         solfile2 << "Inverter-";
         meshfile << "Inverter";
         break;
      }
      default:
         mfem_error("Undefined problem.");
   }

   int dim = mesh.Dimension();
   double h = std::pow(mesh.GetElementVolume(0), 1.0 / dim);

   // 3. Refine the mesh.
   for (int lev = 0; lev < ref_levels; lev++)
   {
      mesh.UniformRefinement();
      h *= 0.5;
   }
   epsilon = 4 * h;
   meshfile << ".mesh";
   solfile << "OC-0.gf";
   solfile2 << "OC-f.gf";

   // 4. Define the necessary finite element spaces on the mesh.
   H1_FECollection state_fec(order, dim); // space for u
   H1_FECollection filter_fec(order, dim); // space for ρ̃
   L2_FECollection control_fec(order-1, dim,
                               BasisType::GaussLobatto); // space for ψ
   FiniteElementSpace state_fes(&mesh, &state_fec,dim,mfem::Ordering::byNODES);
   FiniteElementSpace filter_fes(&mesh, &filter_fec);
   FiniteElementSpace control_fes(&mesh, &control_fec);

   int state_size = state_fes.GetTrueVSize();
   int control_size = control_fes.GetTrueVSize();
   int filter_size = filter_fes.GetTrueVSize();
   mfem::out << "Number of state unknowns: " << state_size << std::endl;
   mfem::out << "Number of filter unknowns: " << filter_size << std::endl;
   mfem::out << "Number of control unknowns: " << control_size << std::endl;

   // 5. Set the initial guess for ρ.
   GridFunction rho(&control_fes);
   rho = vol_fraction;
   GridFunction rho_old(rho);
   GridFunctionCoefficient rho_cf(&rho);

   GridFunction frho(&filter_fes);
   frho = vol_fraction;
   MappedGridFunctionCoefficient SIMP_cf(&frho, [rho_min,
   exponent](double x) {return simp(x, rho_min, exponent, 1.0); });

   GridFunction u(&state_fes), adju(&state_fes);
   u = 0.0; adju = 0.0;

   ConstantCoefficient one(1.0), zero(0.0), eps2_cf(epsilon*epsilon),
                       lambda_cf(lambda),
                       mu_cf(mu);
   StrainEnergyDensityCoefficient strainEnergyDensity_cf(&lambda_cf, &mu_cf, &u,
                                                         &adju,
                                                         &frho, rho_min, exponent);

   ProductCoefficient SIMP_lam(lambda, SIMP_cf), SIMP_mu(mu, SIMP_cf);
   GridFunction zero_gf(&control_fes);
   zero_gf = 0.0;
   GridFunction dv(&control_fes);
   zero_gf.ComputeElementL1Errors(one, dv);
   dv.Reciprocal();

   // 10. Connect to GLVis. Prepare for VisIt output.
   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sout_SIMP, sout_r;
   if (glvis_visualization)
   {
      GridFunction designDensity_gf(&filter_fes);
      designDensity_gf.ProjectCoefficient(SIMP_cf);
      sout_SIMP.open(vishost, visport);
      sout_SIMP.precision(8);
      sout_SIMP << "solution\n" << mesh << designDensity_gf
                << "window_title 'Design density r(ρ̃) - OC'\n"
                << "keys Rjl***************\n"
                << flush;
      sout_r.open(vishost, visport);
      sout_r.precision(8);
      sout_r << "solution\n" << mesh << u
             << "window_title 'Raw density ρ - OC'\n"
             << "keys Rjl***************\n"
             << flush;
   }


   double domain_volume = zero_gf.ComputeL1Error(one);
   double target_volume = domain_volume * vol_fraction;

   GridFunction gradH1(&filter_fes), gradL2(&control_fes);
   gradH1 = 0.0; gradL2 = 0.0;
   GridFunctionCoefficient gradH1_cf(&gradH1);
   BilinearForm invMass(&control_fes);
   invMass.AddDomainIntegrator(new InverseIntegrator(new MassIntegrator()));
   invMass.Assemble();
   LinearForm gradH1Form(&control_fes);
   gradH1Form.AddDomainIntegrator(new DomainLFIntegrator(gradH1_cf));

   int k;
   for (k = 1; k <= max_it; k++)
   {
      mfem::out << "\nStep = " << k << std::endl;

      VectorConstantCoefficient input_d_cf(input_direction),
                                output_d_cf(output_direction);
      BilinearForm elasticityForm(&state_fes);
      elasticityForm.AddDomainIntegrator(new ElasticityIntegrator(SIMP_lam, SIMP_mu));
      elasticityForm.AddBdrFaceIntegrator(new VectorBoundaryDirectionalMassIntegrator(
                                             input_spring, input_d_cf), input_bdr);
      elasticityForm.AddBdrFaceIntegrator(new VectorBoundaryDirectionalMassIntegrator(
                                             output_spring, output_d_cf), output_bdr);

      LinearForm elasticityRHS(&state_fes);
      elasticityRHS.AddBdrFaceIntegrator(new VectorBoundaryDirectionalLFIntegrator(
                                            input_d_cf, input_d_cf), input_bdr);

      BilinearForm adjElasticityForm(&state_fes);
      adjElasticityForm.AddDomainIntegrator(new ElasticityIntegrator(SIMP_lam,
                                                                     SIMP_mu));
      adjElasticityForm.AddBdrFaceIntegrator(new
                                             VectorBoundaryDirectionalMassIntegrator(input_spring, input_d_cf), input_bdr);
      adjElasticityForm.AddBdrFaceIntegrator(new
                                             VectorBoundaryDirectionalMassIntegrator(output_spring, output_d_cf),
                                             output_bdr);
      LinearForm adjElasticityRHS(&state_fes);
      adjElasticityRHS.AddBdrFaceIntegrator(new VectorBoundaryDirectionalLFIntegrator(
                                               output_d_cf, output_d_cf), output_bdr);

      out << "Solving Elasticity" << std::endl;
      EllipticSolver elasticitySolver(&elasticityForm, &elasticityRHS, ess_bdr);
      elasticitySolver.Solve(&u);

      out << "Solving Adjoint Elasticity" << std::endl;
      EllipticSolver adjElasticitySolver(&adjElasticityForm, &adjElasticityRHS,
                                         ess_bdr);
      adjElasticitySolver.Solve(&adju);

      double output_displacement = -adjElasticityRHS(u);


      BilinearForm filterForm(&filter_fes);
      filterForm.AddDomainIntegrator(new DiffusionIntegrator(eps2_cf));
      filterForm.AddDomainIntegrator(new MassIntegrator());

      LinearForm filterRHS(&filter_fes);
      filterRHS.AddDomainIntegrator(new DomainLFIntegrator(rho_cf));


      BilinearForm dualFilterForm(&filter_fes);
      dualFilterForm.AddDomainIntegrator(new DiffusionIntegrator(eps2_cf));
      dualFilterForm.AddDomainIntegrator(new MassIntegrator());

      LinearForm dualFilterRHS(&filter_fes);
      dualFilterRHS.AddDomainIntegrator(new DomainLFIntegrator(
                                           strainEnergyDensity_cf));
      GridFunction strainEnergyDensity_gf(&control_fes);
      strainEnergyDensity_gf.ProjectCoefficient(strainEnergyDensity_cf);
      // disp(strainEnergyDensity_gf);
      EllipticSolver dualFilterSolver(&dualFilterForm, &dualFilterRHS,
                                      ess_bdr_filter);
      out << "Solving Dual Filter" << std::endl;
      dualFilterSolver.Solve(&gradH1);

      invMass.Mult(gradH1Form, gradL2);

      Vector B(gradL2);
      B *= dv;
      Vector lower(rho), upper(rho);
      lower -= mv; lower.Clip(0.001, infinity());
      upper += mv; upper.Clip(-infinity(), 1);
      double l1(0.0), l2(1e09);
      rho_old = rho;
      while ((l2 - l1)/(l2 + l1) > 1e-4 & l2 > 1e-40)
      {
         double lmid = (l1 + l2) / 2.0;
         rho = rho_old;
         Vector dc(B);
         dc *= 1.0 / lmid;
         dc.Clip(1e-10, infinity());
         for (double &val : dc) { val = std::pow(val, 0.3); }
         rho *= dc;
         rho.Clip(lower, upper);
         filterRHS.Assemble();
         if (filterRHS.Sum() > target_volume)
         {
            l1 = lmid;
         }
         else
         {
            l2 = lmid;
         }
      }

      EllipticSolver filterSolver(&filterForm, &filterRHS, ess_bdr_filter);
      out << "Solving filter" << std::endl;
      filterSolver.Solve(&frho);

      // mfem::out << "volume fraction = " <<  filterRHS.Sum() / domain_volume <<
      //  std::endl;
      mfem::out <<  output_displacement << ", ";

      // Compute ||ρ - ρ_old|| in control fes.
      // double norm_increment = zerogf.ComputeL1Error(succ_diff_rho);

      rho_old -= rho;
      double norm_increment = rho_old.ComputeL1Error(zero);

      mfem::out << norm_increment << endl;

      if (glvis_visualization)
      {
         GridFunction designDensity_gf(&filter_fes);
         designDensity_gf.ProjectCoefficient(SIMP_cf);
         sout_SIMP << "solution\n" << mesh << designDensity_gf
                   << flush;
         sout_r << "solution\n" << mesh << u
                << flush;

         ofstream sol_ofs(solfile.str().c_str());
         sol_ofs.precision(8);
         sol_ofs << rho;

         ofstream sol_ofs2(solfile2.str().c_str());
         sol_ofs2.precision(8);
         sol_ofs2 << frho;
      }

      if (norm_increment < itol)
      {
         break;
      }
   }
   out << "Total number of iteration = " << k << std::endl;

   return 0;
}
