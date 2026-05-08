//                       MFEM Example 37 - Parallel Version
//
// Compile with: make ex37p
//
// Sample runs:
//    mpirun -np 4 ex37p -alpha 10 -pv
//    mpirun -np 4 ex37p -lambda 0.1 -mu 0.1
//    mpirun -np 4 ex37p -o 2 -alpha 5.0 -mi 50 -vf 0.4 -ntol 1e-5
//    mpirun -np 4 ex37p -r 6 -o 2 -alpha 10.0 -epsilon 0.02 -mi 50 -ntol 1e-5
//
// Description: This example code demonstrates the use of MFEM to solve a
//              density-filtered [3] topology optimization problem. The
//              objective is to minimize the compliance
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
// [1] Andreassen, E., Clausen, A., Schevenels, M., Lazarov, B. S., & Sigmund, O.
//     (2011). Efficient topology optimization in MATLAB using 88 lines of
//     code. Structural and Multidisciplinary Optimization, 43(1), 1-16.
// [2] Keith, B. and Surowiec, T. (2023) Proximal Galerkin: A structure-
//     preserving finite element method for pointwise bound constraints.
//     arXiv:2307.12444 [math.NA]
// [3] Lazarov, B. S., & Sigmund, O. (2011). Filters in topology optimization
//     based on Helmholtz‐type differential equations. International Journal
//     for Numerical Methods in Engineering, 86(6), 765-781.
//#include "mfem.hpp"
// #include <iostream>
// #include <fstream>
#include "remap_TO.hpp"

using namespace std;
using namespace mfem;

/**
 * @brief Bregman projection of ρ = sigmoid(ψ) onto the subspace
 *        ∫_Ω ρ dx = θ vol(Ω) as follows:
 *
 *        1. Compute the root of the R → R function
 *            f(c) = ∫_Ω sigmoid(ψ + c) dx - θ vol(Ω)
 *        2. Set ψ ← ψ + c.
 *
 * @param psi a GridFunction to be updated
 * @param target_volume θ vol(Ω)
 * @param tol Newton iteration tolerance
 * @param max_its Newton maximum iteration number
 * @return real_t Final volume, ∫_Ω sigmoid(ψ)
 */
real_t proj(ParGridFunction &psi, real_t target_volume, real_t tol=1e-12,
            int max_its=10)
{
   MappedGridFunctionCoefficient sigmoid_psi(&psi, sigmoid);
   MappedGridFunctionCoefficient der_sigmoid_psi(&psi, der_sigmoid);

   ParLinearForm int_sigmoid_psi(psi.ParFESpace());
   int_sigmoid_psi.AddDomainIntegrator(new DomainLFIntegrator(sigmoid_psi));
   ParLinearForm int_der_sigmoid_psi(psi.ParFESpace());
   int_der_sigmoid_psi.AddDomainIntegrator(new DomainLFIntegrator(
                                              der_sigmoid_psi));
   bool done = false;
   for (int k=0; k<max_its; k++) // Newton iteration
   {
      int_sigmoid_psi.Assemble(); // Recompute f(c) with updated ψ
      real_t f = int_sigmoid_psi.Sum();
      MPI_Allreduce(MPI_IN_PLACE, &f, 1, MPITypeMap<real_t>::mpi_type,
                    MPI_SUM, MPI_COMM_WORLD);
      f -= target_volume;

      int_der_sigmoid_psi.Assemble(); // Recompute df(c) with updated ψ
      real_t df = int_der_sigmoid_psi.Sum();
      MPI_Allreduce(MPI_IN_PLACE, &df, 1, MPITypeMap<real_t>::mpi_type,
                    MPI_SUM, MPI_COMM_WORLD);

      const real_t dc = -f/df;
      psi += dc;
      if (abs(dc) < tol) { done = true; break; }
   }
   if (!done)
   {
      mfem_warning("Projection reached maximum iteration without converging. "
                   "Result may not be accurate.");
   }
   int_sigmoid_psi.Assemble();
   real_t material_volume = int_sigmoid_psi.Sum();
   MPI_Allreduce(MPI_IN_PLACE, &material_volume, 1,
                 MPITypeMap<real_t>::mpi_type, MPI_SUM, MPI_COMM_WORLD);
   return material_volume;
}

void GetDOFPositions(const ParMesh & pmesh, const ParFiniteElementSpace &pfes, Vector &pos_dofs)
{
   const int NE  = pfes.GetNE(), dim = pmesh.Dimension();
   const int nsp = pfes.GetFE(0)->GetNodes().GetNPoints();

   pos_dofs.SetSize(nsp * NE * dim);
   for (int e = 0; e < NE; e++)
   {
      const IntegrationRule &ir = pfes.GetFE(e)->GetNodes();

      // Transformation of the element with the pos_mesh coordinates.
      IsoparametricTransformation Tr;
      pmesh.GetElementTransformation(e, &Tr);

      // Node positions of pfes for pos_mesh.
      Vector rowx(pos_dofs.GetData() + e*nsp, nsp),
             rowy(pos_dofs.GetData() + e*nsp + NE*nsp, nsp), rowz;
      if (dim == 3)
      {
         rowz.SetDataAndSize(pos_dofs.GetData() + e*nsp + 2*NE*nsp, nsp);
      }

      DenseMatrix pos_nodes;
      Tr.Transform(ir, pos_nodes);
      pos_nodes.GetRow(0, rowx);
      pos_nodes.GetRow(1, rowy);
      if (dim == 3) { pos_nodes.GetRow(2, rowz); }
   }
}

/*
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
 */

int main(int argc, char *argv[])
{
   // 0. Initialize MPI and HYPRE.
   Mpi::Init();
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // 1. Parse command-line options.
   int ref_levels = 3;
   int mesh_order=1;
   int order = 2;
   real_t alpha = 1.0;
   real_t epsilon = 0.01;
   real_t vol_fraction = 0.5;
   int max_it = 1e3;
   real_t itol = 1e-1;
   real_t ntol = 1e-4;
   real_t rho_min = 1e-6;
   real_t lambda = 1.0;
   real_t mu = 1.0;
   bool glvis_visualization = false;
   bool paraview_output = true;

   OptionsParser args(argc, argv);
   args.AddOption(&ref_levels, "-r", "--refine",
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
   args.AddOption(&paraview_output, "-pv", "--paraview", "-no-pv",
                  "--no-paraview",
                  "Enable or disable ParaView output.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      MPI_Finalize();
      return 1;
   }
   if (myid == 0)
   {
      mfem::out << num_procs << " number of process created.\n";
      args.PrintOptions(cout);
   }

   // Mesh mesh_A = Mesh::MakeCartesian2D(9, 3, mfem::Element::Type::QUADRILATERAL,
   //                                   true, 3.0, 1.0);
   // mfem::Mesh mesh_A("3x1_rectangle.g");
   Mesh mesh_A = Mesh::MakeCartesian2DWith5QuadsPerQuad(9, 3,   3.0, 1.0);
   //MakeCartesian2DWith4TrisPerQuad  MakeCartesian2DWith5QuadsPerQuad
  if(true)
  {
     int tNumVertices  = mesh_A.GetNV();
     for (int i = 0; i < tNumVertices; ++i) {
        double * Coords = mesh_A.GetVertex(i);
        if (Coords[ 0 ] != 0.0 && Coords[ 0 ] != 3.0 && Coords[ 1 ] != 0.0 && Coords[ 1 ] != 1.0) {
           Coords[ 0 ] = Coords[ 0 ] + ((rand() / double(RAND_MAX)* 2.0 - 1.0)* 2e-2);
           Coords[ 1 ] = Coords[ 1 ] + ((rand() / double(RAND_MAX)* 2.0 - 1.0)* 2e-2);
        }
     }
  }

   Mesh mesh_B = Mesh::MakeCartesian2D(9, 3, mfem::Element::Type::QUADRILATERAL,
                                     true, 3.0, 1.0);
   int dim = mesh_B.Dimension();

   // 2. Set BCs.
   for (int i = 0; i<mesh_B.GetNBE(); i++)
   {
      Element * be = mesh_B.GetBdrElement(i);
      Array<int> vertices;
      be->GetVertices(vertices);

      real_t * coords1 = mesh_B.GetVertex(vertices[0]);
      real_t * coords2 = mesh_B.GetVertex(vertices[1]);

      Vector center(2);
      center(0) = 0.5*(coords1[0] + coords2[0]);
      center(1) = 0.5*(coords1[1] + coords2[1]);

      if (abs(center(0) - 0.0) < 1e-10)
      {
         // the left edge
         be->SetAttribute(1);
      }
      else
      {
         // all other boundaries
         be->SetAttribute(2);
      }
   }
   mesh_B.SetAttributes();

   // 3. Refine the mesh.
   for (int lev = 0; lev < ref_levels; lev++)
   {
      mesh_B.UniformRefinement();
      //mesh_A.UniformRefinement();
   }
     for (int lev = 0; lev < ref_levels-1; lev++)
   {
      //mesh_B.UniformRefinement();
      mesh_A.UniformRefinement();
   }

   ParMesh pmesh_A(MPI_COMM_WORLD, mesh_A);
   ParMesh pmesh_B(MPI_COMM_WORLD, mesh_B);
   mesh_A.Clear();
   mesh_B.Clear();

   pmesh_A.SetCurvature(mesh_order, false);
   pmesh_B.SetCurvature(mesh_order, false);
   //    ParGridFunction x(&mesh_pfes);
   // pmesh.SetNodalGridFunction(&x);

   // 4. Define the necessary finite element spaces on the mesh.
   H1_FECollection state_fec(order, dim); // space for u
   H1_FECollection filter_fec(order, dim); // space for ρ̃
   H1_FECollection mesh_fec_A(mesh_order, dim); 
   H1_FECollection mesh_fec_B(mesh_order, dim); 
   L2_FECollection control_fec(order-1, dim,
                               BasisType::GaussLobatto); // space for ψ
   ParFiniteElementSpace state_fes(&pmesh_B, &state_fec,dim);
   ParFiniteElementSpace filter_fes(&pmesh_B, &filter_fec);
   ParFiniteElementSpace control_fes_B(&pmesh_B, &control_fec);
   ParFiniteElementSpace control_fes_A(&pmesh_A, &control_fec);
   ParFiniteElementSpace mesh_fes_A(&pmesh_A, &mesh_fec_A, dim);
   ParFiniteElementSpace mesh_fes_B(&pmesh_B, &mesh_fec_B, dim);

   HYPRE_BigInt state_size = state_fes.GlobalTrueVSize();
   HYPRE_BigInt control_size = control_fes_B.GlobalTrueVSize();
   HYPRE_BigInt control_size_A = control_fes_A.GlobalTrueVSize();
   HYPRE_BigInt filter_size = filter_fes.GlobalTrueVSize();
   if (myid==0)
   {
      cout << "Number of state unknowns: " << state_size << endl;
      cout << "Number of filter unknowns: " << filter_size << endl;
      cout << "Number of control B unknowns: " << control_size << endl;
      cout << "Number of control A unknowns: " << control_size_A << endl;
   }

   // 5. Set the initial guess for ρ.
   ParGridFunction u(&state_fes);
   ParGridFunction psi_A(&control_fes_A);
   ParGridFunction psi_B(&control_fes_B);
   ParGridFunction psi_old(&control_fes_A);
   ParGridFunction rho_filter(&filter_fes);
   ParGridFunction mesh_gf_A(&mesh_fes_A);
   ParGridFunction mesh_gf_B(&mesh_fes_B);
   u = 0.0;
   rho_filter = vol_fraction;
   psi_A = inv_sigmoid(vol_fraction);
   psi_old = inv_sigmoid(vol_fraction);

   pmesh_A.SetNodalGridFunction(&mesh_gf_A);
   pmesh_B.SetNodalGridFunction(&mesh_gf_B);

   // ρ = sigmoid(ψ)
   MappedGridFunctionCoefficient rho(&psi_A, sigmoid);
   // Interpolation of ρ = sigmoid(ψ) in control fes (for ParaView output)
   ParGridFunction rho_gf(&control_fes_A);
   ParGridFunction rho_gf_B(&control_fes_B);
   // ρ - ρ_old = sigmoid(ψ) - sigmoid(ψ_old)
   DiffMappedGridFunctionCoefficient succ_diff_rho(&psi_A, &psi_old, sigmoid);

   // 6. Set-up the physics solver.
   int maxat = pmesh_B.bdr_attributes.Max();
   Array<int> ess_bdr(maxat);
   ess_bdr = 0;
   ess_bdr[0] = 1;
   ConstantCoefficient one(1.0);
   ConstantCoefficient lambda_cf(lambda);
   ConstantCoefficient mu_cf(mu);
   LinearElasticitySolver * ElasticitySolver = new LinearElasticitySolver();
   ElasticitySolver->SetMesh(&pmesh_B);
   ElasticitySolver->SetOrder(state_fec.GetOrder());
   ElasticitySolver->SetupFEM();
   Vector center(2); center(0) = 2.9; center(1) = 0.5;
   Vector force(2); force(0) = 0.0; force(1) = -1.0;
   real_t r = 0.05;
   VolumeForceCoefficient vforce_cf(r,center,force);
   ElasticitySolver->SetRHSCoefficient(&vforce_cf);
   ElasticitySolver->SetEssentialBoundary(ess_bdr);

   // 7. Set-up the filter solver.
   ConstantCoefficient eps2_cf(epsilon*epsilon);
   DiffusionSolver * FilterSolver = new DiffusionSolver();
   FilterSolver->SetMesh(&pmesh_B);
   FilterSolver->SetOrder(filter_fec.GetOrder());
   FilterSolver->SetDiffusionCoefficient(&eps2_cf);
   FilterSolver->SetMassCoefficient(&one);
   Array<int> ess_bdr_filter;
   if (pmesh_B.bdr_attributes.Size())
   {
      ess_bdr_filter.SetSize(pmesh_B.bdr_attributes.Max());
      ess_bdr_filter = 0;
   }
   FilterSolver->SetEssentialBoundary(ess_bdr_filter);
   FilterSolver->SetupFEM();

   ParBilinearForm mass(&control_fes_B);
   mass.AddDomainIntegrator(new InverseIntegrator(new MassIntegrator(one)));
   mass.Assemble();
   HypreParMatrix M;
   Array<int> empty;
   mass.FormSystemMatrix(empty,M);

   // 8. Define the Lagrange multiplier and gradient functions.
   ParGridFunction grad_A(&control_fes_A);
   ParGridFunction grad_B(&control_fes_B);
   ParGridFunction w_filter(&filter_fes);

   // 9. Define some tools for later.
   ConstantCoefficient zero(0.0);
   ParGridFunction onegf(&control_fes_A);
   onegf = 1.0;
   ParGridFunction zerogf(&control_fes_A);
   zerogf = 0.0;
   ParLinearForm vol_form(&control_fes_A);
   vol_form.AddDomainIntegrator(new DomainLFIntegrator(one));
   vol_form.Assemble();
   real_t domain_volume = vol_form(onegf);
   const real_t target_volume = domain_volume * vol_fraction;

   // 10. Connect to GLVis. Prepare for VisIt output.
   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sout_r;
   if (glvis_visualization)
   {
      sout_r.open(vishost, visport);
      sout_r.precision(8);
   }

   mfem::ParaViewDataCollection paraview_dc_A("remap_TO_A", &pmesh_A);
   mfem::ParaViewDataCollection paraview_dc_B("remap_TO_B", &pmesh_B);
   if (paraview_output)
   {
      rho_gf.ProjectCoefficient(rho);
      paraview_dc_A.SetPrefixPath("ParaView");
      paraview_dc_A.SetLevelsOfDetail(order);
      paraview_dc_A.SetDataFormat(VTKFormat::BINARY);
      paraview_dc_A.SetHighOrderOutput(true);
      paraview_dc_A.SetCycle(0);
      paraview_dc_A.SetTime(0.0);
      paraview_dc_A.RegisterField("density",&rho_gf);
      paraview_dc_A.Save();

      rho_gf_B = 0.5;
      paraview_dc_B.SetPrefixPath("ParaView");
      paraview_dc_B.SetLevelsOfDetail(order);
      paraview_dc_B.SetDataFormat(VTKFormat::BINARY);
      paraview_dc_B.SetHighOrderOutput(true);
      paraview_dc_B.SetCycle(0);
      paraview_dc_B.SetTime(0.0);
      paraview_dc_B.RegisterField("displacement",&u);
      paraview_dc_B.RegisterField("density",&rho_gf_B);
      paraview_dc_B.RegisterField("filtered_density",&rho_filter);
      paraview_dc_B.Save();
   }

   // const GridFunction * nodes_A = pmesh_A.GetNodes();
   // const GridFunction * nodes_B = pmesh_B.GetNodes();

   // 11. Iterate:
   for (int k = 1; k <= max_it; k++)
   {
      if (k > 1) { alpha *= ((real_t) k) / ((real_t) k-1); }

      if (myid == 0)
      {
         cout << "\nStep = " << k << endl;
      }
      Vector pos_dof_B;
      GetDOFPositions(pmesh_B, control_fes_B, pos_dof_B);
   
      FindPointsGSLIB finder_B(MPI_COMM_WORLD);
      finder_B.SetL2AvgType(FindPointsGSLIB::NONE);
      finder_B.Setup(pmesh_A);
      finder_B.Interpolate(pos_dof_B, rho_gf, rho_gf_B);

      // Step 1 - Filter solve
      // Solve (ϵ^2 ∇ ρ̃, ∇ v ) + (ρ̃,v) = (ρ,v)
      GridFunctionCoefficient roh_B(&rho_gf_B);
      FilterSolver->SetRHSCoefficient(&roh_B);
      FilterSolver->Solve();
      rho_filter = *FilterSolver->GetFEMSolution();

      // Step 2 - State solve
      // Solve (λ r(ρ̃) ∇⋅u, ∇⋅v) + (2 μ r(ρ̃) ε(u), ε(v)) = (f,v)
      SIMPInterpolationCoefficient SIMP_cf(&rho_filter,rho_min, 1.0);
      ProductCoefficient lambda_SIMP_cf(lambda_cf,SIMP_cf);
      ProductCoefficient mu_SIMP_cf(mu_cf,SIMP_cf);
      ElasticitySolver->SetLameCoefficients(&lambda_SIMP_cf,&mu_SIMP_cf);
      ElasticitySolver->Solve();
      u = *ElasticitySolver->GetFEMSolution();

      // Step 3 - Adjoint filter solve
      // Solve (ϵ² ∇ w̃, ∇ v) + (w̃ ,v) = (-r'(ρ̃) ( λ |∇⋅u|² + 2 μ |ε(u)|²),v)
      StrainEnergyDensityCoefficient rhs_cf(&lambda_cf,&mu_cf,&u, &rho_filter,
                                            rho_min);
      FilterSolver->SetRHSCoefficient(&rhs_cf);
      FilterSolver->Solve();
      w_filter = *FilterSolver->GetFEMSolution();

      // Step 4 - Compute gradient
      // Solve G = M⁻¹w̃
      GridFunctionCoefficient w_cf(&w_filter);
      ParLinearForm w_rhs(&control_fes_B);
      w_rhs.AddDomainIntegrator(new DomainLFIntegrator(w_cf));
      w_rhs.Assemble();
      M.Mult(w_rhs,grad_B);

      Vector pos_dof_A;
      GetDOFPositions(pmesh_A, control_fes_A, pos_dof_A);

      FindPointsGSLIB finder_A(MPI_COMM_WORLD);
      finder_A.SetL2AvgType(FindPointsGSLIB::NONE);
      finder_A.Setup(pmesh_B);
      finder_A.Interpolate(pos_dof_A, grad_B, grad_A);

      //mfem_error("aaaaaaaaabbbbbbbbbb");

      // Step 5 - Update design variable ψ ← proj(ψ - αG)

      psi_A.Add(-alpha, grad_A);

      const real_t material_volume = proj(psi_A, target_volume);


      // Compute ||ρ - ρ_old|| in control fes.
      real_t norm_increment = zerogf.ComputeL1Error(succ_diff_rho);
      real_t norm_reduced_gradient = norm_increment/alpha;
      psi_old = psi_A;

      real_t compliance = (*(ElasticitySolver->GetLinearForm()))(u);
      MPI_Allreduce(MPI_IN_PLACE, &compliance, 1, MPITypeMap<real_t>::mpi_type,
                    MPI_SUM, MPI_COMM_WORLD);
      if (myid == 0)
      {
         mfem::out << "norm of the reduced gradient = " << norm_reduced_gradient << endl;
         mfem::out << "norm of the increment = " << norm_increment << endl;
         mfem::out << "compliance = " << compliance << endl;
         mfem::out << "volume fraction = " << material_volume / domain_volume << endl;
      }

      if (glvis_visualization)
      {
         ParGridFunction r_gf(&filter_fes);
         r_gf.ProjectCoefficient(SIMP_cf);
         sout_r << "parallel " << num_procs << " " << myid << "\n";
         sout_r << "solution\n" << pmesh_B << r_gf
                << "window_title 'Design density r(ρ̃)'" << flush;
      }

      if (paraview_output)
      {
         rho_gf.ProjectCoefficient(rho);
         paraview_dc_B.SetCycle(k);
         paraview_dc_B.SetTime((real_t)k);
         paraview_dc_B.Save();

         paraview_dc_A.SetCycle(k);
         paraview_dc_A.SetTime((real_t)k);
         paraview_dc_A.Save();
      }

      if (norm_reduced_gradient < ntol && norm_increment < itol)
      {
         break;
      }
   }

   delete ElasticitySolver;
   delete FilterSolver;

   return 0;
}
