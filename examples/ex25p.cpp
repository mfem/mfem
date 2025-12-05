//                       MFEM Example 25 - Parallel Version
//
// Compile with: make ex25p
//
// Sample runs:  mpirun -np 4 ex25p -o 2 -f 1.0 -rs 1 -rp 1 -prob 0
//               mpirun -np 4 ex25p -o 3 -f 10.0 -rs 1 -rp 1 -prob 1
//               mpirun -np 4 ex25p -o 3 -f 5.0 -rs 3 -rp 1 -prob 2
//               mpirun -np 4 ex25p -o 2 -f 1.0 -rs 1 -rp 1 -prob 3
//               mpirun -np 4 ex25p -o 2 -f 1.0 -rs 2 -rp 2 -prob 0 -m ../data/beam-quad.mesh
//               mpirun -np 4 ex25p -o 2 -f 8.0 -rs 2 -rp 2 -prob 4 -m ../data/inline-quad.mesh
//               mpirun -np 4 ex25p -o 2 -f 2.0 -rs 1 -rp 1 -prob 4 -m ../data/inline-hex.mesh
//
// Device sample runs:
//               mpirun -np 4 ex25p -o 1 -f 3.0 -rs 3 -rp 1 -prob 2 -pa -d cuda
//               mpirun -np 4 ex25p -o 2 -f 1.0 -rs 1 -rp 1 -prob 3 -pa -d cuda
//
// Description:  This example code solves a simple electromagnetic wave
//               propagation problem corresponding to the second order
//               indefinite Maxwell equation
//
//                  (1/mu) * curl curl E - \omega^2 * epsilon E = f
//
//               with a Perfectly Matched Layer (PML).
//
//               The example demonstrates discretization with Nedelec finite
//               elements in 2D or 3D, as well as the use of complex-valued
//               bilinear and linear forms. Several test problems are included,
//               with prob = 0-3 having known exact solutions, see "On perfectly
//               matched layers for discontinuous Petrov-Galerkin methods" by
//               Vaziri Astaneh, Keith, Demkowicz, Comput Mech 63, 2019.
//
//               We recommend viewing Example 22 before viewing this example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

#ifdef _WIN32
#define jn(n, x) _jn(n, x)
#define yn(n, x) _yn(n, x)
#endif

using namespace std;
using namespace mfem;

// Class for setting up a simple Cartesian PML region
class PML
{
private:
   Mesh *mesh;

   int dim;

   // Length of the PML Region in each direction
   Array2D<real_t> length;

   // Computational Domain Boundary
   Array2D<real_t> comp_dom_bdr;

   // Domain Boundary
   Array2D<real_t> dom_bdr;

   // Integer Array identifying elements in the PML
   // 0: in the PML, 1: not in the PML
   Array<int> elems;

   // Compute Domain and Computational Domain Boundaries
   void SetBoundaries();

public:
   // Constructor
   PML(Mesh *mesh_,Array2D<real_t> length_);

   // Return Computational Domain Boundary
   Array2D<real_t> GetCompDomainBdr() {return comp_dom_bdr;}

   // Return Domain Boundary
   Array2D<real_t> GetDomainBdr() {return dom_bdr;}

   // Return Markers list for elements
   Array<int> * GetMarkedPMLElements() {return &elems;}

   // Mark elements in the PML region
   void SetAttributes(ParMesh *pmesh);

   // PML complex stretching function
   void StretchFunction(const Vector &x, vector<complex<real_t>> &dxs);
};

// Class for returning the PML coefficients of the bilinear form
class PMLDiagMatrixCoefficient : public VectorCoefficient
{
private:
   PML * pml = nullptr;
   void (*Function)(const Vector &, PML *, Vector &);
public:
   PMLDiagMatrixCoefficient(int dim, void(*F)(const Vector &, PML *,
                                              Vector &),
                            PML * pml_)
      : VectorCoefficient(dim), pml(pml_), Function(F)
   {}

   using VectorCoefficient::Eval;

   void Eval(Vector &K, ElementTransformation &T,
             const IntegrationPoint &ip) override
   {
      real_t x[3];
      Vector transip(x, 3);
      T.Transform(ip, transip);
      K.SetSize(vdim);
      (*Function)(transip, pml, K);
   }
};

void maxwell_solution(const Vector &x, vector<complex<real_t>> &Eval);

void E_bdr_data_Re(const Vector &x, Vector &E);
void E_bdr_data_Im(const Vector &x, Vector &E);

void E_exact_Re(const Vector &x, Vector &E);
void E_exact_Im(const Vector &x, Vector &E);

void source(const Vector &x, Vector & f);

// Functions for computing the necessary coefficients after PML stretching.
// J is the Jacobian matrix of the stretching function
void detJ_JT_J_inv_Re(const Vector &x, PML * pml, Vector & D);
void detJ_JT_J_inv_Im(const Vector &x, PML * pml, Vector & D);
void detJ_JT_J_inv_abs(const Vector &x, PML * pml, Vector & D);

void detJ_inv_JT_J_Re(const Vector &x, PML * pml, Vector & D);
void detJ_inv_JT_J_Im(const Vector &x, PML * pml, Vector & D);
void detJ_inv_JT_J_abs(const Vector &x, PML * pml, Vector & D);

Array2D<real_t> comp_domain_bdr;
Array2D<real_t> domain_bdr;

real_t mu = 1.0;
real_t epsilon = 1.0;
real_t omega;
int dim;
bool exact_known = false;

template <typename T> T pow2(const T &x) { return x*x; }

enum prob_type
{
   beam,     // Wave propagating in a beam-like domain
   disc,     // Point source propagating in the square-disc domain
   lshape,   // Point source propagating in the L-shape domain
   fichera,  // Point source propagating in the fichera domain
   load_src  // Approximated point source with PML all around
};
prob_type prob;

int main(int argc, char *argv[])
{
   // 1. Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // 2. Parse command-line options.
   const char *mesh_file = nullptr;
   int order = 1;
   int ref_levels = 1;
   int par_ref_levels = 2;
   int iprob = 4;
   real_t freq = 5.0;
   bool herm_conv = true;
   bool slu_solver  = false;
   bool mumps_solver = false;
   bool strumpack_solver = false;
   bool visualization = 1;
   bool pa = false;
   const char *device_config = "cpu";

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&iprob, "-prob", "--problem", "Problem case"
                  " 0: beam, 1: disc, 2: lshape, 3: fichera, 4: General");
   args.AddOption(&ref_levels, "-rs", "--refinements-serial",
                  "Number of serial refinements");
   args.AddOption(&par_ref_levels, "-rp", "--refinements-parallel",
                  "Number of parallel refinements");
   args.AddOption(&mu, "-mu", "--permeability",
                  "Permeability of free space (or 1/(spring constant)).");
   args.AddOption(&epsilon, "-eps", "--permittivity",
                  "Permittivity of free space (or mass constant).");
   args.AddOption(&freq, "-f", "--frequency",
                  "Frequency (in Hz).");
   args.AddOption(&herm_conv, "-herm", "--hermitian", "-no-herm",
                  "--no-hermitian", "Use convention for Hermitian operators.");
#ifdef MFEM_USE_SUPERLU
   args.AddOption(&slu_solver, "-slu", "--superlu", "-no-slu",
                  "--no-superlu", "Use the SuperLU Solver.");
#endif
#ifdef MFEM_USE_MUMPS
   args.AddOption(&mumps_solver, "-mumps", "--mumps-solver", "-no-mumps",
                  "--no-mumps-solver", "Use the MUMPS Solver.");
#endif
#ifdef MFEM_USE_STRUMPACK
   args.AddOption(&strumpack_solver, "-strumpack", "--strumpack-solver",
                  "-no-strumpack", "--no-strumpack-solver",
                  "Use the STRUMPACK Solver.");
#endif
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.Parse();
   if (slu_solver + mumps_solver + strumpack_solver > 1)
   {
      if (myid == 0)
         cout << "WARNING: More than one of SuperLU, MUMPS, and STRUMPACK have"
              << " been selected, please choose only one." << endl
              << "         Defaulting to SuperLU." << endl;
      mumps_solver = false;
      strumpack_solver = false;
   }

   if (iprob > 4) { iprob = 4; }
   prob = (prob_type)iprob;

   // 3. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   if (myid == 0) { device.Print(); }

   // 4. Setup the (serial) mesh on all processors.
   if (!mesh_file)
   {
      exact_known = true;
      switch (prob)
      {
         case beam:
            mesh_file = "../data/beam-hex.mesh";
            break;
         case disc:
            mesh_file = "../data/square-disc.mesh";
            break;
         case lshape:
            mesh_file = "../data/l-shape.mesh";
            break;
         case fichera:
            mesh_file = "../data/fichera.mesh";
            break;
         default:
            exact_known = false;
            mesh_file = "../data/inline-quad.mesh";
            break;
      }
   }

   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   Mesh * mesh = new Mesh(mesh_file, 1, 1);
   dim = mesh->Dimension();

   // Angular frequency
   omega = real_t(2.0 * M_PI) * freq;

   // Setup PML length
   Array2D<real_t> length(dim, 2); length = 0.0;

   // 5. Setup the Cartesian PML region.
   switch (prob)
   {
      case disc:
         length = 0.2;
         break;
      case lshape:
         length(0, 0) = 0.1;
         length(1, 0) = 0.1;
         break;
      case fichera:
         length(0, 1) = 0.5;
         length(1, 1) = 0.5;
         length(2, 1) = 0.5;
         break;
      case beam:
         length(0, 1) = 2.0;
         break;
      default:
         length = 0.25;
         break;
   }
   PML * pml = new PML(mesh,length);
   comp_domain_bdr = pml->GetCompDomainBdr();
   domain_bdr = pml->GetDomainBdr();

   // 6. Refine the serial mesh on all processors to increase the resolution.
   for (int l = 0; l < ref_levels; l++)
   {
      mesh->UniformRefinement();
   }

   // 7. Define a parallel mesh by a partitioning of the serial mesh.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   {
      for (int l = 0; l < par_ref_levels; l++)
      {
         pmesh->UniformRefinement();
      }
   }

   // 8. Set element attributes in order to distinguish elements in the PML
   pml->SetAttributes(pmesh);

   // 9. Define a parallel finite element space on the parallel mesh. Here we
   //    use the Nedelec finite elements of the specified order.
   FiniteElementCollection *fec = new ND_FECollection(order, dim);
   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);
   HYPRE_BigInt size = fespace->GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of finite element unknowns: " << size << endl;
   }

   // 10. Determine the list of true (i.e. parallel conforming) essential
   //     boundary dofs. In this example, the boundary conditions are defined
   //     based on the specific mesh and the problem type.
   Array<int> ess_tdof_list;
   Array<int> ess_bdr;
   if (pmesh->bdr_attributes.Size())
   {
      ess_bdr.SetSize(pmesh->bdr_attributes.Max());
      ess_bdr = 1;
      if (prob == lshape || prob == fichera)
      {
         ess_bdr = 0;
         for (int j = 0; j < pmesh->GetNBE(); j++)
         {
            Vector center(dim);
            int bdrgeom = pmesh->GetBdrElementGeometry(j);
            ElementTransformation * tr = pmesh->GetBdrElementTransformation(j);
            tr->Transform(Geometries.GetCenter(bdrgeom),center);
            int k = pmesh->GetBdrAttribute(j);
            switch (prob)
            {
               case lshape:
                  if (center[0] == 1_r || center[0] == 0.5_r ||
                      center[1] == 0.5_r)
                  {
                     ess_bdr[k - 1] = 1;
                  }
                  break;
               case fichera:
                  if (center[0] == -1_r || center[0] == 0_r ||
                      center[1] ==  0_r || center[2] == 0_r)
                  {
                     ess_bdr[k - 1] = 1;
                  }
                  break;
               default:
                  break;
            }
         }
      }
   }
   fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   // 11. Setup Complex Operator convention
   ComplexOperator::Convention conv =
      herm_conv ? ComplexOperator::HERMITIAN : ComplexOperator::BLOCK_SYMMETRIC;

   // 12. Set up the parallel linear form b(.) which corresponds to the
   //     right-hand side of the FEM linear system.
   VectorFunctionCoefficient f(dim, source);
   ParComplexLinearForm b(fespace, conv);
   if (prob == load_src)
   {
      b.AddDomainIntegrator(NULL, new VectorFEDomainLFIntegrator(f));
   }
   b = 0.0;
   b.Assemble();

   // 13. Define the solution vector x as a parallel complex finite element grid
   //     function corresponding to fespace.
   ParComplexGridFunction x(fespace);
   x = 0.0;
   VectorFunctionCoefficient E_Re(dim, E_bdr_data_Re);
   VectorFunctionCoefficient E_Im(dim, E_bdr_data_Im);
   x.ProjectBdrCoefficientTangent(E_Re, E_Im, ess_bdr);

   // 14. Set up the parallel sesquilinear form a(.,.)
   //
   //     In Comp
   //     Domain:   1/mu (Curl E, Curl F) - omega^2 * epsilon (E,F)
   //
   //     In PML:   1/mu (1/det(J) J^T J Curl E, Curl F)
   //               - omega^2 * epsilon (det(J) * (J^T J)^-1 * E, F)
   //
   //     where J denotes the Jacobian Matrix of the PML Stretching function
   Array<int> attr;
   Array<int> attrPML;
   if (pmesh->attributes.Size())
   {
      attr.SetSize(pmesh->attributes.Max());
      attrPML.SetSize(pmesh->attributes.Max());
      attr = 0;   attr[0] = 1;
      attrPML = 0;
      if (pmesh->attributes.Max() > 1)
      {
         attrPML[1] = 1;
      }
   }

   ConstantCoefficient muinv(1_r / mu);
   ConstantCoefficient omeg(-pow2(omega) * epsilon);
   RestrictedCoefficient restr_muinv(muinv,attr);
   RestrictedCoefficient restr_omeg(omeg,attr);

   // Integrators inside the computational domain (excluding the PML region)
   ParSesquilinearForm a(fespace, conv);
   a.AddDomainIntegrator(new CurlCurlIntegrator(restr_muinv),NULL);
   a.AddDomainIntegrator(new VectorFEMassIntegrator(restr_omeg),NULL);

   int cdim = (dim == 2) ? 1 : dim;
   PMLDiagMatrixCoefficient pml_c1_Re(cdim,detJ_inv_JT_J_Re, pml);
   PMLDiagMatrixCoefficient pml_c1_Im(cdim,detJ_inv_JT_J_Im, pml);
   ScalarVectorProductCoefficient c1_Re(muinv,pml_c1_Re);
   ScalarVectorProductCoefficient c1_Im(muinv,pml_c1_Im);
   VectorRestrictedCoefficient restr_c1_Re(c1_Re,attrPML);
   VectorRestrictedCoefficient restr_c1_Im(c1_Im,attrPML);

   PMLDiagMatrixCoefficient pml_c2_Re(dim, detJ_JT_J_inv_Re,pml);
   PMLDiagMatrixCoefficient pml_c2_Im(dim, detJ_JT_J_inv_Im,pml);
   ScalarVectorProductCoefficient c2_Re(omeg,pml_c2_Re);
   ScalarVectorProductCoefficient c2_Im(omeg,pml_c2_Im);
   VectorRestrictedCoefficient restr_c2_Re(c2_Re,attrPML);
   VectorRestrictedCoefficient restr_c2_Im(c2_Im,attrPML);

   // Integrators inside the PML region
   a.AddDomainIntegrator(new CurlCurlIntegrator(restr_c1_Re),
                         new CurlCurlIntegrator(restr_c1_Im));
   a.AddDomainIntegrator(new VectorFEMassIntegrator(restr_c2_Re),
                         new VectorFEMassIntegrator(restr_c2_Im));

   // 15. Assemble the parallel bilinear form and the corresponding linear
   //     system, applying any necessary transformations such as: parallel
   //     assembly, eliminating boundary conditions, applying conforming
   //     constraints for non-conforming AMR, etc.
   if (pa) { a.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   a.Assemble();

   OperatorPtr Ah;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, Ah, X, B);

   // 16. Solve using a direct or an iterative solver
#ifdef MFEM_USE_SUPERLU
   if (!pa && slu_solver)
   {
      // Transform to monolithic HypreParMatrix
      HypreParMatrix *A = Ah.As<ComplexHypreParMatrix>()->GetSystemMatrix();
      SuperLURowLocMatrix SA(*A);
      SuperLUSolver superlu(MPI_COMM_WORLD);
      superlu.SetPrintStatistics(false);
      superlu.SetSymmetricPattern(false);
      superlu.SetColumnPermutation(superlu::PARMETIS);
      superlu.SetOperator(SA);
      superlu.Mult(B, X);
      delete A;
   }
#endif
#ifdef MFEM_USE_STRUMPACK
   if (!pa && strumpack_solver)
   {
      HypreParMatrix *A = Ah.As<ComplexHypreParMatrix>()->GetSystemMatrix();
      STRUMPACKRowLocMatrix SA(*A);
      STRUMPACKSolver strumpack(MPI_COMM_WORLD, argc, argv);
      strumpack.SetPrintFactorStatistics(false);
      strumpack.SetPrintSolveStatistics(false);
      strumpack.SetKrylovSolver(strumpack::KrylovSolver::DIRECT);
      strumpack.SetReorderingStrategy(strumpack::ReorderingStrategy::METIS);
      strumpack.SetMatching(strumpack::MatchingJob::NONE);
      strumpack.SetCompression(strumpack::CompressionType::NONE);
      strumpack.SetFromCommandLine();
      strumpack.SetOperator(SA);
      strumpack.Mult(B, X);
      delete A;
   }
#endif
#ifdef MFEM_USE_MUMPS
   if (!pa && mumps_solver)
   {
      HypreParMatrix *A = Ah.As<ComplexHypreParMatrix>()->GetSystemMatrix();
      MUMPSSolver mumps(A->GetComm());
      mumps.SetPrintLevel(0);
      mumps.SetMatrixSymType(MUMPSSolver::MatType::UNSYMMETRIC);
      mumps.SetOperator(*A);
      mumps.Mult(B, X);
      delete A;
   }
#endif
   // 16a. Set up the parallel Bilinear form a(.,.) for the preconditioner
   //
   //    In Comp
   //    Domain:   1/mu (Curl E, Curl F) + omega^2 * epsilon (E,F)
   //
   //    In PML:   1/mu (abs(1/det(J) J^T J) Curl E, Curl F)
   //              + omega^2 * epsilon (abs(det(J) * (J^T J)^-1) * E, F)
   if (pa || (!slu_solver && !mumps_solver && !strumpack_solver))
   {
      ConstantCoefficient absomeg(pow2(omega) * epsilon);
      RestrictedCoefficient restr_absomeg(absomeg,attr);

      ParBilinearForm prec(fespace);
      prec.AddDomainIntegrator(new CurlCurlIntegrator(restr_muinv));
      prec.AddDomainIntegrator(new VectorFEMassIntegrator(restr_absomeg));

      PMLDiagMatrixCoefficient pml_c1_abs(cdim,detJ_inv_JT_J_abs, pml);
      ScalarVectorProductCoefficient c1_abs(muinv,pml_c1_abs);
      VectorRestrictedCoefficient restr_c1_abs(c1_abs,attrPML);

      PMLDiagMatrixCoefficient pml_c2_abs(dim, detJ_JT_J_inv_abs,pml);
      ScalarVectorProductCoefficient c2_abs(absomeg,pml_c2_abs);
      VectorRestrictedCoefficient restr_c2_abs(c2_abs,attrPML);

      prec.AddDomainIntegrator(new CurlCurlIntegrator(restr_c1_abs));
      prec.AddDomainIntegrator(new VectorFEMassIntegrator(restr_c2_abs));

      if (pa) { prec.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
      prec.Assemble();

      // 16b. Define and apply a parallel GMRES solver for AU=B with a block
      //      diagonal preconditioner based on hypre's AMS preconditioner.
      Array<int> offsets(3);
      offsets[0] = 0;
      offsets[1] = fespace->GetTrueVSize();
      offsets[2] = fespace->GetTrueVSize();
      offsets.PartialSum();

      std::unique_ptr<Operator> pc_r;
      std::unique_ptr<Operator> pc_i;
      int s = (conv == ComplexOperator::HERMITIAN) ? -1 : 1;
      if (pa)
      {
         // Jacobi Smoother
         pc_r.reset(new OperatorJacobiSmoother(prec, ess_tdof_list));
         pc_i.reset(new ScaledOperator(pc_r.get(), s));
      }
      else
      {
         OperatorPtr PCOpAh;
         prec.FormSystemMatrix(ess_tdof_list, PCOpAh);

         // Hypre AMS
         pc_r.reset(new HypreAMS(*PCOpAh.As<HypreParMatrix>(), fespace));
         pc_i.reset(new ScaledOperator(pc_r.get(), s));
      }

      BlockDiagonalPreconditioner BlockDP(offsets);
      BlockDP.SetDiagonalBlock(0, pc_r.get());
      BlockDP.SetDiagonalBlock(1, pc_i.get());

      GMRESSolver gmres(MPI_COMM_WORLD);
      gmres.SetPrintLevel(1);
      gmres.SetKDim(200);
      gmres.SetMaxIter(pa ? 5000 : 2000);
      gmres.SetRelTol(1e-5);
      gmres.SetAbsTol(0.0);
      gmres.SetOperator(*Ah);
      gmres.SetPreconditioner(BlockDP);
      gmres.Mult(B, X);
   }

   // 17. Recover the parallel grid function corresponding to X. This is the
   //     local finite element solution on each processor.
   a.RecoverFEMSolution(X, b, x);

   // If exact is known compute the error
   if (exact_known)
   {
      VectorFunctionCoefficient E_ex_Re(dim, E_exact_Re);
      VectorFunctionCoefficient E_ex_Im(dim, E_exact_Im);
      int order_quad = max(2, 2 * order + 1);
      const IntegrationRule *irs[Geometry::NumGeom];
      for (int i = 0; i < Geometry::NumGeom; ++i)
      {
         irs[i] = &(IntRules.Get(i, order_quad));
      }

      real_t L2Error_Re = x.real().ComputeL2Error(E_ex_Re, irs,
                                                  pml->GetMarkedPMLElements());
      real_t L2Error_Im = x.imag().ComputeL2Error(E_ex_Im, irs,
                                                  pml->GetMarkedPMLElements());

      ParComplexGridFunction x_gf0(fespace);
      x_gf0 = 0.0;
      real_t norm_E_Re, norm_E_Im;
      norm_E_Re = x_gf0.real().ComputeL2Error(E_ex_Re, irs,
                                              pml->GetMarkedPMLElements());
      norm_E_Im = x_gf0.imag().ComputeL2Error(E_ex_Im, irs,
                                              pml->GetMarkedPMLElements());

      if (myid == 0)
      {
         cout << "\n Relative Error (Re part): || E_h - E || / ||E|| = "
              << L2Error_Re / norm_E_Re
              << "\n Relative Error (Im part): || E_h - E || / ||E|| = "
              << L2Error_Im / norm_E_Im
              << "\n Total Error: "
              << sqrt(L2Error_Re*L2Error_Re + L2Error_Im*L2Error_Im) << "\n\n";
      }
   }

   // 18. Save the refined mesh and the solution in parallel. This output can be
   //     viewed later using GLVis: "glvis -np <np> -m mesh -g sol".
   {
      ostringstream mesh_name, sol_r_name, sol_i_name;
      mesh_name << "mesh." << setfill('0') << setw(6) << myid;
      sol_r_name << "ex25p-sol_r." << setfill('0') << setw(6) << myid;
      sol_i_name << "ex25p-sol_i." << setfill('0') << setw(6) << myid;

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh->Print(mesh_ofs);

      ofstream sol_r_ofs(sol_r_name.str().c_str());
      ofstream sol_i_ofs(sol_i_name.str().c_str());
      sol_r_ofs.precision(8);
      sol_i_ofs.precision(8);
      x.real().Save(sol_r_ofs);
      x.imag().Save(sol_i_ofs);
   }

   // 19. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      // Define visualization keys for GLVis (see GLVis documentation)
      string keys;
      keys = (dim == 3) ? "keys macF\n" : keys = "keys amrRljcUUuu\n";
      if (prob == beam && dim == 3) {keys = "keys macFFiYYYYYYYYYYYYYYYYYY\n";}
      if (prob == beam && dim == 2) {keys = "keys amrRljcUUuuu\n"; }

      char vishost[] = "localhost";
      int visport = 19916;

      {
         socketstream sol_sock_re(vishost, visport);
         sol_sock_re.precision(8);
         sol_sock_re << "parallel " << num_procs << " " << myid << "\n"
                     << "solution\n" << *pmesh << x.real() << keys
                     << "window_title 'Solution real part'" << flush;
         MPI_Barrier(MPI_COMM_WORLD); // try to prevent streams from mixing
      }

      {
         socketstream sol_sock_im(vishost, visport);
         sol_sock_im.precision(8);
         sol_sock_im << "parallel " << num_procs << " " << myid << "\n"
                     << "solution\n" << *pmesh << x.imag() << keys
                     << "window_title 'Solution imag part'" << flush;
         MPI_Barrier(MPI_COMM_WORLD); // try to prevent streams from mixing
      }

      {
         ParGridFunction x_t(fespace);
         x_t = x.real();

         socketstream sol_sock(vishost, visport);
         sol_sock.precision(8);
         sol_sock << "parallel " << num_procs << " " << myid << "\n"
                  << "solution\n" << *pmesh << x_t << keys << "autoscale off\n"
                  << "window_title 'Harmonic Solution (t = 0.0 T)'"
                  << "pause\n" << flush;

         if (myid == 0)
         {
            cout << "GLVis visualization paused."
                 << " Press space (in the GLVis window) to resume it.\n";
         }

         int num_frames = 32;
         int i = 0;
         while (sol_sock)
         {
            real_t t = (real_t)(i % num_frames) / num_frames;
            ostringstream oss;
            oss << "Harmonic Solution (t = " << t << " T)";

            add(cos(real_t(2.0*M_PI)*t), x.real(),
                sin(real_t(2.0*M_PI)*t), x.imag(), x_t);
            sol_sock << "parallel " << num_procs << " " << myid << "\n";
            sol_sock << "solution\n" << *pmesh << x_t
                     << "window_title '" << oss.str() << "'" << flush;
            i++;
         }
      }
   }

   // 20. Free the used memory.
   delete pml;
   delete fespace;
   delete fec;
   delete pmesh;
   return 0;
}

void source(const Vector &x, Vector &f)
{
   Vector center(dim);
   real_t r = 0.0;
   for (int i = 0; i < dim; ++i)
   {
      center(i) = real_t(0.5) * (comp_domain_bdr(i, 0) + comp_domain_bdr(i, 1));
      r += pow2(x[i] - center[i]);
   }
   real_t n = real_t(5) * omega * sqrt(epsilon * mu) / real_t(M_PI);
   real_t coeff = pow2(n) / real_t(M_PI);
   real_t alpha = -pow2(n) * r;
   f = 0.0;
   f[0] = coeff * exp(alpha);
}

void maxwell_solution(const Vector &x, vector<complex<real_t>> &E)
{
   // Initialize
   for (int i = 0; i < dim; ++i)
   {
      E[i] = 0.0;
   }

   constexpr complex<real_t> zi = complex<real_t>(0., 1.);
   real_t k = omega * sqrt(epsilon * mu);
   switch (prob)
   {
      case disc:
      case lshape:
      case fichera:
      {
         Vector shift(dim);
         shift = 0.0;
         if (prob == fichera) { shift =  1.0; }
         if (prob == disc)    { shift = -0.5; }
         if (prob == lshape)  { shift = -1.0; }

         if (dim == 2)
         {
            real_t x0 = x(0) + shift(0);
            real_t x1 = x(1) + shift(1);
            real_t r = sqrt(x0 * x0 + x1 * x1);
            real_t beta = k * r;

            // Bessel functions
            complex<real_t> Ho, Ho_r, Ho_rr;
            Ho = real_t(jn(0, beta)) + zi * real_t(yn(0, beta));
            Ho_r = -k * (real_t(jn(1, beta)) + zi * real_t(yn(1, beta)));
            Ho_rr = -k * k * (1_r / beta *
                              (real_t(jn(1, beta)) + zi * real_t(yn(1, beta))) -
                              (real_t(jn(2, beta)) + zi * real_t(yn(2, beta))));

            // First derivatives
            real_t r_x = x0 / r;
            real_t r_y = x1 / r;
            real_t r_xy = -(r_x / r) * r_y;
            real_t r_xx = (1_r / r) * (1_r - r_x * r_x);

            complex<real_t> val, val_xx, val_xy;
            val = real_t(0.25) * zi * Ho;
            val_xx = real_t(0.25) * zi * (r_xx * Ho_r + r_x * r_x * Ho_rr);
            val_xy = real_t(0.25) * zi * (r_xy * Ho_r + r_x * r_y * Ho_rr);
            E[0] = zi / k * (k * k * val + val_xx);
            E[1] = zi / k * val_xy;
         }
         else if (dim == 3)
         {
            real_t x0 = x(0) + shift(0);
            real_t x1 = x(1) + shift(1);
            real_t x2 = x(2) + shift(2);
            real_t r = sqrt(x0 * x0 + x1 * x1 + x2 * x2);

            real_t r_x = x0 / r;
            real_t r_y = x1 / r;
            real_t r_z = x2 / r;
            real_t r_xx = (1_r / r) * (1_r - r_x * r_x);
            real_t r_yx = -(r_y / r) * r_x;
            real_t r_zx = -(r_z / r) * r_x;

            complex<real_t> val, val_r, val_rr;
            val = exp(zi * k * r) / r;
            val_r = val / r * (zi * k * r - 1_r);
            val_rr = val / (r * r) * (-k * k * r * r
                                      - real_t(2) * zi * k * r + real_t(2));

            complex<real_t> val_xx, val_yx, val_zx;
            val_xx = val_rr * r_x * r_x + val_r * r_xx;
            val_yx = val_rr * r_x * r_y + val_r * r_yx;
            val_zx = val_rr * r_x * r_z + val_r * r_zx;

            complex<real_t> alpha = zi * k / real_t(4) / (real_t) M_PI / k / k;
            E[0] = alpha * (k * k * val + val_xx);
            E[1] = alpha * val_yx;
            E[2] = alpha * val_zx;
         }
         break;
      }
      case beam:
      {
         // T_10 mode
         if (dim == 3)
         {
            real_t k10 = sqrt(k * k - real_t(M_PI * M_PI));
            E[1] = -zi * k / (real_t) M_PI *
                   sin((real_t) M_PI*x(2))*exp(zi * k10 * x(0));
         }
         else if (dim == 2)
         {
            E[1] = -zi * k / (real_t) M_PI * exp(zi * k * x(0));
         }
         break;
      }
      default:
         break;
   }
}

void E_exact_Re(const Vector &x, Vector &E)
{
   vector<complex<real_t>> Eval(E.Size());
   maxwell_solution(x, Eval);
   for (int i = 0; i < dim; ++i)
   {
      E[i] = Eval[i].real();
   }
}

void E_exact_Im(const Vector &x, Vector &E)
{
   vector<complex<real_t>> Eval(E.Size());
   maxwell_solution(x, Eval);
   for (int i = 0; i < dim; ++i)
   {
      E[i] = Eval[i].imag();
   }
}

void E_bdr_data_Re(const Vector &x, Vector &E)
{
   E = 0.0;
   bool in_pml = false;

   for (int i = 0; i < dim; ++i)
   {
      // check if in PML
      if (x(i) - comp_domain_bdr(i, 0) < 0_r ||
          x(i) - comp_domain_bdr(i, 1) > 0_r)
      {
         in_pml = true;
         break;
      }
   }
   if (!in_pml)
   {
      vector<complex<real_t>> Eval(E.Size());
      maxwell_solution(x, Eval);
      for (int i = 0; i < dim; ++i)
      {
         E[i] = Eval[i].real();
      }
   }
}

// Define bdr_data solution
void E_bdr_data_Im(const Vector &x, Vector &E)
{
   E = 0.0;
   bool in_pml = false;

   for (int i = 0; i < dim; ++i)
   {
      // check if in PML
      if (x(i) - comp_domain_bdr(i, 0) < 0_r ||
          x(i) - comp_domain_bdr(i, 1) > 0_r)
      {
         in_pml = true;
         break;
      }
   }
   if (!in_pml)
   {
      vector<complex<real_t>> Eval(E.Size());
      maxwell_solution(x, Eval);
      for (int i = 0; i < dim; ++i)
      {
         E[i] = Eval[i].imag();
      }
   }
}

void detJ_JT_J_inv_Re(const Vector &x, PML * pml, Vector & D)
{
   vector<complex<real_t>> dxs(dim);
   complex<real_t> det(1.0, 0.0);
   pml->StretchFunction(x, dxs);

   for (int i = 0; i < dim; ++i)
   {
      det *= dxs[i];
   }

   for (int i = 0; i < dim; ++i)
   {
      D(i) = (det / pow2(dxs[i])).real();
   }
}

void detJ_JT_J_inv_Im(const Vector &x, PML * pml, Vector & D)
{
   vector<complex<real_t>> dxs(dim);
   complex<real_t> det = 1.0;
   pml->StretchFunction(x, dxs);

   for (int i = 0; i < dim; ++i)
   {
      det *= dxs[i];
   }

   for (int i = 0; i < dim; ++i)
   {
      D(i) = (det / pow2(dxs[i])).imag();
   }
}

void detJ_JT_J_inv_abs(const Vector &x, PML * pml, Vector & D)
{
   vector<complex<real_t>> dxs(dim);
   complex<real_t> det = 1.0;
   pml->StretchFunction(x, dxs);

   for (int i = 0; i < dim; ++i)
   {
      det *= dxs[i];
   }

   for (int i = 0; i < dim; ++i)
   {
      D(i) = abs(det / pow2(dxs[i]));
   }
}

void detJ_inv_JT_J_Re(const Vector &x, PML * pml, Vector & D)
{
   vector<complex<real_t>> dxs(dim);
   complex<real_t> det(1.0, 0.0);
   pml->StretchFunction(x, dxs);

   for (int i = 0; i < dim; ++i)
   {
      det *= dxs[i];
   }

   // in the 2D case the coefficient is scalar 1/det(J)
   if (dim == 2)
   {
      D = (1_r / det).real();
   }
   else
   {
      for (int i = 0; i < dim; ++i)
      {
         D(i) = (pow2(dxs[i]) / det).real();
      }
   }
}

void detJ_inv_JT_J_Im(const Vector &x, PML * pml, Vector & D)
{
   vector<complex<real_t>> dxs(dim);
   complex<real_t> det = 1.0;
   pml->StretchFunction(x, dxs);

   for (int i = 0; i < dim; ++i)
   {
      det *= dxs[i];
   }

   if (dim == 2)
   {
      D = (1_r / det).imag();
   }
   else
   {
      for (int i = 0; i < dim; ++i)
      {
         D(i) = (pow2(dxs[i]) / det).imag();
      }
   }
}

void detJ_inv_JT_J_abs(const Vector &x, PML * pml, Vector & D)
{
   vector<complex<real_t>> dxs(dim);
   complex<real_t> det = 1.0;
   pml->StretchFunction(x, dxs);

   for (int i = 0; i < dim; ++i)
   {
      det *= dxs[i];
   }

   if (dim == 2)
   {
      D = abs(1_r / det);
   }
   else
   {
      for (int i = 0; i < dim; ++i)
      {
         D(i) = abs(pow2(dxs[i]) / det);
      }
   }
}

PML::PML(Mesh *mesh_, Array2D<real_t> length_)
   : mesh(mesh_), length(length_)
{
   dim = mesh->Dimension();
   SetBoundaries();
}

void PML::SetBoundaries()
{
   comp_dom_bdr.SetSize(dim, 2);
   dom_bdr.SetSize(dim, 2);
   Vector pmin, pmax;
   mesh->GetBoundingBox(pmin, pmax);
   for (int i = 0; i < dim; i++)
   {
      dom_bdr(i, 0) = pmin(i);
      dom_bdr(i, 1) = pmax(i);
      comp_dom_bdr(i, 0) = dom_bdr(i, 0) + length(i, 0);
      comp_dom_bdr(i, 1) = dom_bdr(i, 1) - length(i, 1);
   }
}

void PML::SetAttributes(ParMesh *pmesh)
{
   // Initialize bdr attributes
   for (int i = 0; i < pmesh->GetNBE(); ++i)
   {
      pmesh->GetBdrElement(i)->SetAttribute(i+1);
   }

   int nrelem = pmesh->GetNE();

   // Initialize list with 1
   elems.SetSize(nrelem);

   // Loop through the elements and identify which of them are in the PML
   for (int i = 0; i < nrelem; ++i)
   {
      elems[i] = 1;
      bool in_pml = false;
      Element *el = pmesh->GetElement(i);
      Array<int> vertices;

      // Initialize attribute
      el->SetAttribute(1);
      el->GetVertices(vertices);
      int nrvert = vertices.Size();

      // Check if any vertex is in the PML
      for (int iv = 0; iv < nrvert; ++iv)
      {
         int vert_idx = vertices[iv];
         real_t *coords = pmesh->GetVertex(vert_idx);
         for (int comp = 0; comp < dim; ++comp)
         {
            if (coords[comp] > comp_dom_bdr(comp, 1) ||
                coords[comp] < comp_dom_bdr(comp, 0))
            {
               in_pml = true;
               break;
            }
         }
      }
      if (in_pml)
      {
         elems[i] = 0;
         el->SetAttribute(2);
      }
   }
   pmesh->SetAttributes();
}

void PML::StretchFunction(const Vector &x,
                          vector<complex<real_t>> &dxs)
{
   constexpr complex<real_t> zi = complex<real_t>(0., 1.);

   real_t n = 2.0;
   real_t c = 5.0;
   real_t coeff;
   real_t k = omega * sqrt(epsilon * mu);

   // Stretch in each direction independently
   for (int i = 0; i < dim; ++i)
   {
      dxs[i] = 1.0;
      if (x(i) >= comp_domain_bdr(i, 1))
      {
         coeff = n * c / k / pow(length(i, 1), n);
         dxs[i] = 1_r + zi * coeff *
                  abs(pow(x(i) - comp_domain_bdr(i, 1), n - 1_r));
      }
      if (x(i) <= comp_domain_bdr(i, 0))
      {
         coeff = n * c / k / pow(length(i, 0), n);
         dxs[i] = 1_r + zi * coeff *
                  abs(pow(x(i) - comp_domain_bdr(i, 0), n - 1_r));
      }
   }
}
