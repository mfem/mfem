#include "mfem.hpp"
#include "../common/particles_extras.hpp"
#include "../../general/text.hpp"

#include "../electromagnetics/electromagnetics.hpp"
#include <fstream>
#include <iostream>

// add timer
#include <chrono>
#define EPSILON 1 // ε_0

using namespace std;
using namespace mfem;
using namespace mfem::common;
using namespace mfem::electromagnetics;

struct LorentzContext
{
   // mesh related parameters
   int order = 1;
   int nx = 100;
   int ny = 100;
   real_t L_x = 1.0;
   real_t L_y = 1.0;

   // particle related parameters
   int ordering = 1;
   real_t q = 1.0;
   real_t m = 1.0;

   bool visualization = true;
   int visport = 19916;

   int N_max = 60;
} ctx;

class GreenFunctionCoefficient : public Coefficient
{
public:
   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      // G(x, y) = \sum_{(m,n)\neq(0,0)} \frac{(-1)^{m+n}}{4\pi^2(m^2+n^2)} \cos\big(2\pi(mx+ny)\big).
      // get r, z coordinates
      Vector x;
      T.Transform(ip, x);
      real_t val = 0.0;

      for (int m = -ctx.N_max; m <= ctx.N_max; ++m)
      {
         for (int n = -ctx.N_max; n <= ctx.N_max; ++n)
         {
            if (m == 0 && n == 0)
               continue;

            int parity = (m + n) & 1; // even/odd, works for negatives
            real_t numerator = (parity == 0) ? 1.0 : -1.0;

            real_t denominator = 4.0 * M_PI * M_PI * (m * m + n * n);
            real_t angle = 2.0 * M_PI * (m * x[0] + n * x[1]);

            val += numerator * cos(angle) / denominator;
         }
      }

      return val;
   }
};

class Boris
{
public:
   enum Fields
   {
      MASS,   // vdim = 1
      CHARGE, // vdim = 1
      MOM,    // vdim = dim
      EFIELD, // vdim = dim
      BFIELD, // vdim = dim
      SIZE
   };
};

class GridFunctionUpdates
{
private:
   real_t domain_volume;
   real_t neutralizing_const;
   ParLinearForm *precomputed_neutralizing_lf = nullptr;
   bool neutralizing_const_computed = false;
   bool use_precomputed_neutralizing_const = false;
   // Diffusion matrix
   HypreParMatrix *DiffusionMatrix;

public:
   // Update the phi_gf grid function from the particles.
   // Solve periodic Poisson: DiffusionMatrix * phi = (rho - <rho>)
   // with zero-mean enforcement via OrthoSolver.
   void UpdatePhiGridFunction(ParticleSet &particles,
                              ParGridFunction &phi_gf,
                              ParGridFunction &E_gf)
   {
      { // FE space / mesh
         ParFiniteElementSpace *pfes = phi_gf.ParFESpace();
         ParMesh *pmesh = pfes->GetParMesh();
         const int dim = pmesh->Dimension();

         // Particle data
         MultiVector &X = particles.Coords();             // coordinates (vdim x npt)
         MultiVector &Q = particles.Field(Boris::CHARGE); // charges (1 x npt)
         Ordering::Type ordering_type = X.GetOrdering();

         const int npt = X.GetNumVectors();
         MFEM_VERIFY(X.GetVDim() == dim, "Unexpected particle coordinate layout.");
         MFEM_VERIFY(Q.GetVDim() == 1, "Charge field must be scalar per particle.");

         // ------------------------------------------------------------------------
         // 1) Build positions in byVDIM ordering: (XYZ,XYZ,...)
         // ------------------------------------------------------------------------
         Vector point_pos(X.GetData(), dim * npt); // alias underlying storage

         // ------------------------------------------------------------------------
         // 2) Locate particles with FindPointsGSLIB
         // ------------------------------------------------------------------------
         FindPointsGSLIB finder(pmesh->GetComm());
         finder.Setup(*pmesh);
         finder.FindPoints(point_pos, ordering_type);

         const Array<unsigned int> &code = finder.GetCode(); // 0: inside, 1: boundary, 2: not found
         const Array<unsigned int> &proc = finder.GetProc(); // owning MPI rank
         const Array<unsigned int> &elem = finder.GetElem(); // local element id
         const Vector &rref = finder.GetReferencePosition(); // (r,s,t) byVDIM

         // ------------------------------------------------------------------------
         // 3) Make RHS and pre-subtract averaged charge density => enforce zero-mean RHS
         // ------------------------------------------------------------------------

         MPI_Comm comm = pfes->GetComm();

         if (!use_precomputed_neutralizing_const || !neutralizing_const_computed)
         {
            // compute neutralizing constant
            real_t local_sum = 0.0;
            for (int p = 0; p < npt; ++p)
            {
               // Skip particles not successfully found
               if (code[p] == 2) // not found
               {
                  continue;
               }

               local_sum += Q(p);
            }

            real_t global_sum = 0.0;
            MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, comm);

            neutralizing_const = -global_sum / domain_volume;
            neutralizing_const_computed = true;
            if (Mpi::Root())
            {
               cout << "Total charge: " << global_sum << ", Domain volume: " << domain_volume << ", Neutralizing constant: " << neutralizing_const << endl;
               if (use_precomputed_neutralizing_const)
               {
                  cout << "Further updates will use this precomputed neutralizing constant." << endl;
               }
            }
            neutralizing_const_computed = true;
            delete precomputed_neutralizing_lf;
            precomputed_neutralizing_lf = new ParLinearForm(pfes);
            *precomputed_neutralizing_lf = 0.0;
            ConstantCoefficient neutralizing_coeff(neutralizing_const);
            precomputed_neutralizing_lf->AddDomainIntegrator(new DomainLFIntegrator(neutralizing_coeff));
            precomputed_neutralizing_lf->Assemble();
         }
         ParLinearForm b(pfes);
         b = *precomputed_neutralizing_lf; // start with precomputed neutralizing contribution

         // ------------------------------------------------------------------------
         // 4) Deposit q_p * phi_i(x_p) into a ParLinearForm (RHS b)
         //      b_i = sum_p q_p * φ_i(x_p)
         // ------------------------------------------------------------------------
         int myid;
         MPI_Comm_rank(pmesh->GetComm(), &myid);

         Array<int> dofs;

         for (int p = 0; p < npt; ++p)
         {
            // Skip particles not successfully found
            if (code[p] == 2) // not found
            {
               continue;
            }

            // Raise error if particle is not on the current rank
            if ((int)proc[p] != myid)
            {
               // raise error
               MFEM_ABORT("Particle " << p << " found in element owned by rank "
                                      << proc[p] << " but current rank is " << myid << "." << endl
                                      << "You must call redistribute everytime before updating the density grid function.");
               continue;
            }
            const int e = elem[p];

            // Reference coordinates for this particle (r,s[,t]) with byVDIM layout
            IntegrationPoint ip;
            if (dim == 1)
            {
               ip.x = rref(p);
            }
            else if (dim == 2)
            {
               ip.Set2(rref[2 * p + 0], rref[2 * p + 1]);
            }
            else // dim == 3
            {
               ip.Set3(rref[3 * p + 0], rref[3 * p + 1], rref[3 * p + 2]);
            }

            const FiniteElement &fe = *pfes->GetFE(e);
            const int ldofs = fe.GetDof();

            Vector shape(ldofs);
            fe.CalcShape(ip, shape); // φ_i(x_p) in this element

            pfes->GetElementDofs(e, dofs); // local dof indices

            const real_t q_p = Q(p);

            // Add q_p * φ_i(x_p) to b_i
            b.AddElementVector(dofs, q_p, shape);
         }

         // Assemble to a global true-dof RHS vector compatible with MassMatrix
         HypreParVector *B = b.ParallelAssemble(); // owns new vector on heap

         // ------------------------------------------------------------------
         // 5) Solve A * phi = B with zero-mean enforcement via OrthoSolver
         // ------------------------------------------------------------------
         MFEM_VERIFY(DiffusionMatrix != nullptr, "DiffusionMatrix must be precomputed.");

         phi_gf = 0.0;
         HypreParVector Phi_true(pfes);
         Phi_true = 0.0;

         HyprePCG solver(DiffusionMatrix->GetComm());
         solver.SetOperator(*DiffusionMatrix);
         solver.SetTol(1e-24);
         solver.SetMaxIter(200);
         solver.SetPrintLevel(0);

         HypreBoomerAMG prec(*DiffusionMatrix);
         prec.SetPrintLevel(0);
         solver.SetPreconditioner(prec);

         OrthoSolver ortho(comm);
         ortho.SetSolver(solver);
         ortho.Mult(*B, Phi_true);

         // Map true-dof solution back to the ParGridFunction
         phi_gf.Distribute(Phi_true);
         delete B;
      }

      {
         // 1.a make the RHS bilinear form
         ParMixedBilinearForm b_bi(phi_gf.ParFESpace(), E_gf.ParFESpace());
         ConstantCoefficient neg_one_coef(-1.0);
         b_bi.AddDomainIntegrator(new MixedVectorGradientIntegrator(neg_one_coef));
         b_bi.Assemble();
         b_bi.Finalize();
         // 1.b form linear form from bilinear form
         ParLinearForm b(E_gf.ParFESpace());
         b = 0.0;
         b_bi.Mult(phi_gf, b);
         // Convert to true-dof (parallel) vector
         HypreParVector *B = b.ParallelAssemble();

         // 2. make the bilinear form
         ParBilinearForm a(E_gf.ParFESpace());
         ConstantCoefficient one_coef(1.0);
         a.AddDomainIntegrator(new VectorFEMassIntegrator(one_coef));
         a.Assemble();
         a.Finalize();
         // Parallel operator (HypreParMatrix)
         HypreParMatrix *A = a.ParallelAssemble();

         // 3. solve for E_gf
         CGSolver M_solver(E_gf.ParFESpace()->GetComm());
         M_solver.iterative_mode = false;
         M_solver.SetRelTol(1e-24);
         M_solver.SetAbsTol(0.0);
         M_solver.SetMaxIter(1e5);
         M_solver.SetPrintLevel(0);
         M_solver.SetOperator(*A);

         HypreParVector X(E_gf.ParFESpace()->GetComm(), E_gf.ParFESpace()->GlobalTrueVSize(), E_gf.ParFESpace()->GetTrueDofOffsets());
         X = 0.0;
         M_solver.Mult(*B, X);
         E_gf.SetFromTrueDofs(X);
         delete A;
         delete B;
      }

      {
         static socketstream sol_sock;
         static bool init = false;
         static ParMesh *pmesh = E_gf.ParFESpace()->GetParMesh();

         int num_procs = Mpi::WorldSize();
         int myid_vis = Mpi::WorldRank();
         char vishost[] = "localhost";
         int visport = ctx.visport;

         if (!init)
         {
            sol_sock.open(vishost, visport);
            if (sol_sock)
            {
               init = true;
            }
         }
         if (init)
         {
            sol_sock << "parallel " << num_procs << " " << myid_vis << "\n";
            sol_sock.precision(8);
            sol_sock << "solution\n"
                     << *pmesh << E_gf << std::flush;
         }
      }
      {
         static socketstream sol_sock;
         static bool init = false;
         static ParMesh *pmesh = phi_gf.ParFESpace()->GetParMesh();

         int num_procs = Mpi::WorldSize();
         int myid_vis = Mpi::WorldRank();
         char vishost[] = "localhost";
         int visport = ctx.visport;

         if (!init)
         {
            sol_sock.open(vishost, visport);
            if (sol_sock)
            {
               init = true;
            }
         }
         if (init)
         {
            sol_sock << "parallel " << num_procs << " " << myid_vis << "\n";
            sol_sock.precision(8);
            sol_sock << "solution\n"
                     << *pmesh << phi_gf << std::flush;
         }
      }
   }

   void PhiValidation(ParGridFunction &phi_gf,
                      ParGridFunction &E_gf)
   {
      // FE space / mesh
      ParFiniteElementSpace *phi_pfes = phi_gf.ParFESpace();
      ParGridFunction var_phi_gf(phi_pfes);
      GreenFunctionCoefficient green_coeff;
      var_phi_gf.ProjectCoefficient(green_coeff);
      ParMesh *pmesh = phi_pfes->GetParMesh();

      ParFiniteElementSpace *E_pfes = E_gf.ParFESpace();
      ParGridFunction var_E_gf(E_pfes);

      {
         // 1.a make the RHS bilinear form
         ParMixedBilinearForm b_bi(phi_gf.ParFESpace(), var_E_gf.ParFESpace());
         ConstantCoefficient neg_one_coef(-1.0);
         b_bi.AddDomainIntegrator(new MixedVectorGradientIntegrator(neg_one_coef));
         b_bi.Assemble();
         b_bi.Finalize();
         // 1.b form linear form from bilinear form
         ParLinearForm b(var_E_gf.ParFESpace());
         b = 0.0;
         b_bi.Mult(var_phi_gf, b);
         // Convert to true-dof (parallel) vector
         HypreParVector *B = b.ParallelAssemble();

         // 2. make the bilinear form
         ParBilinearForm a(var_E_gf.ParFESpace());
         ConstantCoefficient one_coef(1.0);
         a.AddDomainIntegrator(new VectorFEMassIntegrator(one_coef));
         a.Assemble();
         a.Finalize();
         // Parallel operator (HypreParMatrix)
         HypreParMatrix *A = a.ParallelAssemble();

         // 3. solve for var_E_gf
         CGSolver M_solver(var_E_gf.ParFESpace()->GetComm());
         M_solver.iterative_mode = false;
         M_solver.SetRelTol(1e-24);
         M_solver.SetAbsTol(0.0);
         M_solver.SetMaxIter(1e5);
         M_solver.SetPrintLevel(0);
         M_solver.SetOperator(*A);

         HypreParVector X(var_E_gf.ParFESpace()->GetComm(), var_E_gf.ParFESpace()->GlobalTrueVSize(), var_E_gf.ParFESpace()->GetTrueDofOffsets());
         X = 0.0;
         M_solver.Mult(*B, X);
         var_E_gf.SetFromTrueDofs(X);
         delete A;
         delete B;
      }
      {
         static socketstream sol_sock;
         static bool init = false;
         static ParMesh *pmesh = var_E_gf.ParFESpace()->GetParMesh();

         int num_procs = Mpi::WorldSize();
         int myid_vis = Mpi::WorldRank();
         char vishost[] = "localhost";
         int visport = ctx.visport;

         if (!init)
         {
            sol_sock.open(vishost, visport);
            if (sol_sock)
            {
               init = true;
            }
         }
         if (init)
         {
            sol_sock << "parallel " << num_procs << " " << myid_vis << "\n";
            sol_sock.precision(8);
            sol_sock << "solution\n"
                     << *pmesh << var_E_gf << std::flush;
         }
      }
      {
         static socketstream sol_sock;
         static bool init = false;
         static ParMesh *pmesh = var_phi_gf.ParFESpace()->GetParMesh();

         int num_procs = Mpi::WorldSize();
         int myid_vis = Mpi::WorldRank();
         char vishost[] = "localhost";
         int visport = ctx.visport;

         if (!init)
         {
            sol_sock.open(vishost, visport);
            if (sol_sock)
            {
               init = true;
            }
         }
         if (init)
         {
            sol_sock << "parallel " << num_procs << " " << myid_vis << "\n";
            sol_sock.precision(8);
            sol_sock << "solution\n"
                     << *pmesh << var_phi_gf << std::flush;
         }
      }
      var_phi_gf -= phi_gf;
      var_E_gf -= E_gf;
      {
         static socketstream sol_sock;
         static bool init = false;
         static ParMesh *pmesh = var_E_gf.ParFESpace()->GetParMesh();

         int num_procs = Mpi::WorldSize();
         int myid_vis = Mpi::WorldRank();
         char vishost[] = "localhost";
         int visport = ctx.visport;

         if (!init)
         {
            sol_sock.open(vishost, visport);
            if (sol_sock)
            {
               init = true;
            }
         }
         if (init)
         {
            sol_sock << "parallel " << num_procs << " " << myid_vis << "\n";
            sol_sock.precision(8);
            sol_sock << "solution\n"
                     << *pmesh << var_E_gf << std::flush;
         }
      }
      {
         static socketstream sol_sock;
         static bool init = false;
         static ParMesh *pmesh = var_phi_gf.ParFESpace()->GetParMesh();

         int num_procs = Mpi::WorldSize();
         int myid_vis = Mpi::WorldRank();
         char vishost[] = "localhost";
         int visport = ctx.visport;

         if (!init)
         {
            sol_sock.open(vishost, visport);
            if (sol_sock)
            {
               init = true;
            }
         }
         if (init)
         {
            sol_sock << "parallel " << num_procs << " " << myid_vis << "\n";
            sol_sock.precision(8);
            sol_sock << "solution\n"
                     << *pmesh << var_phi_gf << std::flush;
         }
      }
   }

   GridFunctionUpdates(ParGridFunction &phi_gf, bool use_precomputed_neutralizing_const_ = false)
       : use_precomputed_neutralizing_const(use_precomputed_neutralizing_const_)
   {
      // compute domain volume
      ParMesh *pmesh = phi_gf.ParFESpace()->GetParMesh();
      real_t local_domain_volume = 0.0;
      for (int i = 0; i < pmesh->GetNE(); i++)
         local_domain_volume += pmesh->GetElementVolume(i);
      MPI_Allreduce(&local_domain_volume, &domain_volume, 1, MPI_DOUBLE, MPI_SUM,
                    phi_gf.ParFESpace()->GetParMesh()->GetComm());

      ParFiniteElementSpace *pfes = phi_gf.ParFESpace();

      { // Par bilinear form for the gradgrad matrix
         ParBilinearForm dm(pfes);
         ConstantCoefficient epsilon(EPSILON);                     // ε_0
         dm.AddDomainIntegrator(new DiffusionIntegrator(epsilon)); // ∫ ∇φ_i · ∇φ_j

         dm.Assemble();
         dm.Finalize();

         DiffusionMatrix = dm.ParallelAssemble(); // global gradgrad matrix
      }
   }
   ~GridFunctionUpdates()
   {
      delete DiffusionMatrix;
      delete precomputed_neutralizing_lf;
   }
};

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   int size = Mpi::WorldSize();
   int rank = Mpi::WorldRank();
   Hypre::Init();
   MPI_Comm comm = MPI_COMM_WORLD;

   {
      OptionsParser args(argc, argv);
      args.AddOption(&ctx.order, "-O", "--order",
                     "Finite element polynomial degree");

      args.AddOption(&ctx.nx, "-nx", "--nx",
                     "Number of elements in the x-direction");

      args.AddOption(&ctx.ny, "-ny", "--ny",
                     "Number of elements in the y-direction");

      args.AddOption(&ctx.L_x, "-Lx", "--Lx",
                     "Domain length in the x-direction");

      args.AddOption(&ctx.L_y, "-Ly", "--Ly",
                     "Domain length in the y-direction");

      args.AddOption(&ctx.ordering, "-ord", "--ordering",
                     "Particle ordering (e.g., lexicographic, space-filling curve)");

      args.AddOption(&ctx.q, "-q", "--charge",
                     "Particle charge");

      args.AddOption(&ctx.m, "-m", "--mass",
                     "Particle mass");

      args.AddOption(&ctx.visualization, "-vis", "--visualization",
                     "-no-vis", "--no-visualization",
                     "Enable or disable visualization");

      args.AddOption(&ctx.visport, "-vp", "--visport",
                     "Visualization socket port");
      args.AddOption(&ctx.N_max, "-Nmax", "--Nmax",
                     "Maximum number of terms in Green's function summation");
      args.Parse();
      if (!args.Good())
      {
         if (Mpi::Root())
         {
            args.PrintUsage(cout);
         }
         return 1;
      }

      if (Mpi::Root())
      {
         args.PrintOptions(cout);
      }
   }

   int dim = 2;
   Ordering::Type ordering_type = ctx.ordering == 0 ? Ordering::byNODES : Ordering::byVDIM;

   // 1. make a 2D Cartesian Mesh
   Mesh serial_mesh(Mesh::MakeCartesian2D(ctx.nx, ctx.ny, Element::QUADRILATERAL, false, ctx.L_x, ctx.L_x));
   std::vector<Vector> translations = {Vector({ctx.L_x, 0.0}), Vector({0.0, ctx.L_x})};
   Mesh periodic_mesh(Mesh::MakePeriodic(serial_mesh, serial_mesh.CreatePeriodicVertexMapping(translations)));
   // 2. parallelize the mesh
   ParMesh mesh(MPI_COMM_WORLD, periodic_mesh);
   serial_mesh.Clear();   // the serial mesh is no longer needed
   periodic_mesh.Clear(); // the periodic mesh is no longer needed
   // 3. Define a finite element space on the parallel mesh
   H1_FECollection sca_fec(ctx.order, dim);
   ParFiniteElementSpace sca_fespace(&mesh, &sca_fec);
   ND_FECollection vec_fec(ctx.order, dim);
   ParFiniteElementSpace vec_fespace(&mesh, &vec_fec);

   // 4. Prepare an empty phi_gf and E_gf for later use
   ParGridFunction phi_gf(&sca_fespace);
   unique_ptr<ParGridFunction> E_gf;
   E_gf = std::make_unique<ParGridFunction>(&vec_fespace);
   phi_gf = 0.0; // Initialize phi_gf to zero
   *E_gf = 0.0;  // Initialize E_gf to zero

   GridFunctionUpdates gf_updates(phi_gf, true);

   unique_ptr<ParticleSet> charged_particles;
   Array<int> field_vdims({1, 1, dim, dim, dim});
   charged_particles = std::make_unique<ParticleSet>(comm, 1, dim,
                                                     field_vdims, 1, ordering_type);
   { // set q, m for all particles
      MultiVector &X = charged_particles->Coords();
      MultiVector &P = charged_particles->Field(Boris::MOM);
      MultiVector &M = charged_particles->Field(Boris::MASS);
      MultiVector &Q = charged_particles->Field(Boris::CHARGE);
      for (int i = 0; i < charged_particles->GetNP(); i++)
      {
         for (int d = 0; d < dim; d++)
         {
            P(i, d) = 0.0;
            X(i, d) = 0.5 * ctx.L_x; // place all particles at the center
         }
         M(i) = ctx.m;
         Q(i) = ctx.q;
      }
   }
   gf_updates.UpdatePhiGridFunction(*charged_particles, phi_gf, *E_gf);
   gf_updates.PhiValidation(phi_gf, *E_gf);

   // Finalize MPI.
   MPI_Finalize();
   return 0;
}