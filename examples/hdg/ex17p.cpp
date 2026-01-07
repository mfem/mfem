//                       MFEM Example 17 - Parallel Version
//
// Compile with: make ex17p
//
// Sample runs:
//
//       mpirun -np 4 ex17p -m ../data/beam-tri.mesh
//       mpirun -np 4 ex17p -m ../data/beam-quad.mesh
//       mpirun -np 4 ex17p -m ../data/beam-tet.mesh
//       mpirun -np 4 ex17p -m ../data/beam-hex.mesh
//       mpirun -np 4 ex17p -m ../data/beam-wedge.mesh
//       mpirun -np 4 ex17p -m ../data/beam-quad.mesh -rs 2 -rp 2 -o 3 -elast
//       mpirun -np 4 ex17p -m ../data/beam-quad.mesh -rs 2 -rp 3 -o 2 -a 1 -k 1
//       mpirun -np 4 ex17p -m ../data/beam-hex.mesh -rs 2 -rp 1 -o 2
//
// Description:  This example code solves a simple linear elasticity problem
//               describing a multi-material cantilever beam using symmetric or
//               non-symmetric discontinuous Galerkin (DG) formulation.
//
//               Specifically, we approximate the weak form of -div(sigma(u))=0
//               where sigma(u)=lambda*div(u)*I+mu*(grad*u+u*grad) is the stress
//               tensor corresponding to displacement field u, and lambda and mu
//               are the material Lame constants. The boundary conditions are
//               Dirichlet, u=u_D on the fixed part of the boundary, namely
//               boundary attributes 1 and 2; on the rest of the boundary we use
//               sigma(u).n=0 b.c. The geometry of the domain is assumed to be
//               as follows:
//
//                                 +----------+----------+
//                    boundary --->| material | material |<--- boundary
//                    attribute 1  |    1     |    2     |     attribute 2
//                    (fixed)      +----------+----------+     (fixed, nonzero)
//
//               The example demonstrates the use of high-order DG vector finite
//               element spaces with the linear DG elasticity bilinear form,
//               meshes with curved elements, and the definition of piece-wise
//               constant and function vector-coefficient objects. The use of
//               non-homogeneous Dirichlet b.c. imposed weakly, is also
//               illustrated.
//
//               We recommend viewing examples 2p and 14p before viewing this
//               example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// Initial displacement, used for Dirichlet boundary conditions on boundary
// attributes 1 and 2.
void InitDisplacement(const Vector &x, Vector &u);

// Simple GLVis visualization manager.
class VisMan : public iostream
{
protected:
   const char *host;
   int port;
   Array<socketstream *> sock;
   int sid; // active socket, index inside 'sock'.

   int win_x, win_y, win_w, win_h;
   int win_stride_x, win_stride_y, win_nx;

public:
   VisMan(const char *vishost, const int visport);
   void NewWindow();
   void CloseConnection();
   void PositionWindow();
   ~VisMan() override;
};

// Manipulators for the GLVis visualization manager.
void new_window      (VisMan &v) { v.NewWindow(); }
void position_window (VisMan &v) { v.PositionWindow(); }
void close_connection(VisMan &v) { v.CloseConnection(); }
ostream &operator<<(ostream &v, void (*f)(VisMan&));

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   // 1. Define and parse command-line options.
   const char *mesh_file = "../../data/beam-tri.mesh";
   int ser_ref_levels = -1;
   int par_ref_levels = 1;
   int order = 1;
   real_t td = 0.5;
   bool reduction = false;
   bool hybridization = false;
   bool trace_h1 = true;
   bool trace_ess_bc = true;
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly before parallel"
                  " partitioning, -1 for auto.");
   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly after parallel"
                  " partitioning.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&td, "-td", "--stab_diff",
                  "Diffusion stabilization factor (1/2=default)");
   args.AddOption(&reduction, "-rd", "--reduction", "-no-rd",
                  "--no-reduction", "Enable reduction of DG flux.");
   args.AddOption(&hybridization, "-hb", "--hybridization", "-no-hb",
                  "--no-hybridization", "Enable hybridization.");
   args.AddOption(&trace_h1, "-trh1", "--trace-H1", "-trdg",
                  "--trace-DG", "Switch between H1 and DG trace spaces (default H1).");
   args.AddOption(&trace_ess_bc, "-trbc", "--trace-ess-bc", "-no-trbc",
                  "--no-trace-ess-bc", "Switch between essential and weak trace BC.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      if (Mpi::Root()) { args.PrintUsage(cout); }
      return 1;
   }
   if (Mpi::Root()) { args.PrintOptions(cout); }

   // 2. Read the mesh from the given mesh file.
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   if (mesh.attributes.Max() < 2 || mesh.bdr_attributes.Max() < 2)
   {
      if (Mpi::Root())
      {
         cerr << "\nInput mesh should have at least two materials and "
              << "two boundary attributes! (See schematic in ex17p.cpp)\n"
              << endl;
      }
      return 3;
   }

   // 3. Refine the mesh to increase the resolution.
   if (ser_ref_levels < 0)
   {
      ser_ref_levels = (int)floor(log(5000./mesh.GetNE())/log(2.)/dim);
   }
   for (int l = 0; l < ser_ref_levels; l++)
   {
      mesh.UniformRefinement();
   }
   // Since NURBS meshes do not support DG integrators, we convert them to
   // regular polynomial mesh of the specified (solution) order.
   if (mesh.NURBSext) { mesh.SetCurvature(order); }

   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   for (int l = 0; l < par_ref_levels; l++)
   {
      pmesh.UniformRefinement();
   }

   // 4. Define a DG vector finite element space on the mesh. Here, we use
   //    Gauss-Lobatto nodal basis because it gives rise to a sparser matrix
   //    compared to the default Gauss-Legendre nodal basis.
   L2_FECollection R_coll(order, dim, BasisType::GaussLobatto);
   L2_FECollection W_coll(order, dim, BasisType::GaussLobatto);

   const int dim_lame = 1 + dim * (dim+1) / 2;
   ParFiniteElementSpace R_space(&pmesh, &R_coll, dim_lame);
   ParFiniteElementSpace W_space(&pmesh, &W_coll, dim);

   HYPRE_BigInt s_size = R_space.GlobalTrueVSize();
   HYPRE_BigInt u_size = W_space.GlobalTrueVSize();
   if (Mpi::Root())
   {
      cout << "Number of stress unknowns: " << s_size << endl;
      cout << "Number of displacement unknowns: " << u_size << endl;
      cout << "Assembling: " << flush;
   }

   ParDarcyForm darcy(&R_space, &W_space);
   const Array<int> &block_offsets = darcy.GetOffsets();
   const Array<int> &block_trueOffsets = darcy.GetTrueOffsets();

   // 5. In this example, the Dirichlet boundary conditions are defined by
   //    marking boundary attributes 1 and 2 in the marker Array 'dir_bdr'.
   //    These b.c. are imposed weakly, by adding the appropriate boundary
   //    integrators over the marked 'dir_bdr' to the bilinear and linear forms.
   //    With this DG formulation, there are no essential boundary conditions.
   Array<int> ess_flux_tdofs_list; // no essential b.c. (empty list)
   Array<int> dir_bdr(pmesh.bdr_attributes.Max());
   dir_bdr = 0;
   dir_bdr[0] = 1; // boundary attribute 1 is Dirichlet
   dir_bdr[1] = 1; // boundary attribute 2 is Dirichlet

   Array<int> neu_bdr(dir_bdr.Size());
   for (int i = 0; i < neu_bdr.Size(); i++)
   {
      neu_bdr[i] = dir_bdr[i] ? 0 : 1;
   }

   // 6. Define the DG solution vector 'x' as a finite element grid function
   //    corresponding to fespace. Initialize 'x' using the 'InitDisplacement'
   //    function.
   BlockVector x(block_offsets);
   x = 0.0;

   ParGridFunction u(&W_space, x.GetBlock(1), 0);
   VectorFunctionCoefficient init_u(dim, InitDisplacement);
   u.ProjectCoefficient(init_u);

   // 7. Set up the Lame constants for the two materials. They are defined as
   //    piece-wise (with respect to the element attributes) constant
   //    coefficients, i.e. type PWConstCoefficient.
   Vector lambda(pmesh.attributes.Max());
   lambda = 1.0;      // Set lambda = 1 for all element attributes.
   lambda(0) = 50.0;  // Set lambda = 50 for element attribute 1.
   PWConstCoefficient lambda_c(lambda);
   Vector mu(pmesh.attributes.Max());
   mu = 1.0;      // Set mu = 1 for all element attributes.
   mu(0) = 50.0;  // Set mu = 50 for element attribute 1.
   PWConstCoefficient mu_c(mu);

   // diffusion coefficient lambda+2*mu
   SumCoefficient sumlame_c(lambda_c, mu_c, 1., 2.);

   // 1/lambda coefficient
   Vector ilambda(lambda.Size());
   for (int i = 0; i < lambda.Size(); i++)
   {
      ilambda(i) = 1./lambda(i);
   }
   PWConstCoefficient ilambda_c(ilambda);

   // 1/2*mu coefficient
   Vector i2mu(mu.Size());
   for (int i = 0; i < mu.Size(); i++)
   {
      i2mu(i) = 1./(2.*mu(i));
   }
   PWConstCoefficient i2mu_c(i2mu);

   // inverse Lame coefficients for the decomposed stress tensor
   VectorArrayCoefficient ilame_c(dim_lame);
   ilame_c.Set(0, &ilambda_c, false);
   for (int i = 1; i < dim_lame; i++)
   {
      ilame_c.Set(i, &i2mu_c, false);
   }

   // 8. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system. In this example, the linear form b(.) consists
   //    only of the terms responsible for imposing weakly the Dirichlet
   //    boundary conditions, over the attributes marked in 'dir_bdr'. The
   //    values for the Dirichlet boundary condition are taken from the
   //    VectorFunctionCoefficient 'x_init' which in turn is based on the
   //    function 'InitDisplacement'.
   ParLinearForm *f = darcy.GetParFluxRHS();
   if (Mpi::Root()) { cout << "r.h.s. ... " << flush; }
   if (!hybridization || !trace_ess_bc)
   {
      f->AddBdrFaceIntegrator(new DGBdrDisplacementLFIntegrator(init_u), dir_bdr);
   }

   // 9. Set up the bilinear form a(.,.) on the DG finite element space
   //    corresponding to the linear elasticity integrator with coefficients
   //    lambda and mu as defined above. The additional interior face integrator
   //    ensures the weak continuity of the displacement field. The additional
   //    boundary face integrator works together with the boundary integrator
   //    added to the linear form b(.) to impose weakly the Dirichlet boundary
   //    conditions.
   ParBilinearForm *Ms = darcy.GetParFluxMassForm();
   ParMixedBilinearForm *Bs = darcy.GetParFluxDivForm();
   ParBilinearForm *Mu = darcy.GetParPotentialMassForm();

   Ms->AddDomainIntegrator(new VectorMassIntegrator(ilame_c));
   Bs->AddDomainIntegrator(new StressDivergenceIntegrator(-1.));
   Bs->AddInteriorFaceIntegrator(new TransposeIntegrator(
                                    new DGNormalStressIntegrator(+1.)));
   Bs->AddBdrFaceIntegrator(new TransposeIntegrator(
                               new DGNormalStressIntegrator(+2.)), neu_bdr);
   Mu->AddInteriorFaceIntegrator(new VectorBlockDiagonalIntegrator(
                                    dim, new HDGDiffusionIntegrator(sumlame_c, td)));
   if (hybridization && trace_ess_bc)
   {
      Bs->AddBdrFaceIntegrator(new TransposeIntegrator(
                                  new DGNormalStressIntegrator(+2.)), dir_bdr);
      Mu->AddBdrFaceIntegrator(new VectorBlockDiagonalIntegrator(
                                  dim, new HDGDiffusionIntegrator(sumlame_c, td)), dir_bdr);
   }

   //set hybridization / assembly level

   FiniteElementCollection *trace_coll = NULL;
   ParFiniteElementSpace *trace_space = NULL;
   Vector X;

   if (hybridization)
   {
      if (trace_h1)
      {
         trace_coll = new H1_Trace_FECollection(max(order, 1), dim);
      }
      else
      {
         trace_coll = new DG_Interface_FECollection(order, dim);
      }
      trace_space = new ParFiniteElementSpace(&pmesh, trace_coll, dim);
      darcy.EnableHybridization(trace_space,
                                new NormalStressJumpIntegrator(-1.),
                                ess_flux_tdofs_list);
      // set essential BC
      if (trace_ess_bc)
      {
         X.SetSize(trace_space->GetTrueVSize());
         // project essential BC
         ParGridFunction uhat;
         uhat.MakeTRef(trace_space, X, 0);
         uhat = 0.;
         uhat.ProjectBdrCoefficient(init_u, dir_bdr);
         uhat.SetTrueVector();

         darcy.GetHybridization()->SetEssentialBC(dir_bdr);
      }
   }
   else if (reduction)
   {
      darcy.EnableFluxReduction();
   }

   // 10. Assemble the bilinear form and the corresponding linear system.
   if (Mpi::Root()) { cout << "matrix ... " << flush; }
   darcy.Assemble();

   OperatorPtr A;
   Vector B;
   darcy.FormLinearSystem(ess_flux_tdofs_list, x, A, X, B, true);
   if (Mpi::Root()) { cout << "done." << endl; }

   constexpr int maxIter(500);
   constexpr real_t rtol(1.0e-6);
   constexpr real_t atol(0.0);

   if (hybridization || reduction)
   {
      // 10. Construct the preconditioner
      HypreBoomerAMG amg(*A.As<HypreParMatrix>());
      amg.SetSystemsOptions(dim, true);

      // 11. Solve the linear system with GMRES.
      //     Check the norm of the unpreconditioned residual.
      GMRESSolver solver(pmesh.GetComm());
      solver.SetAbsTol(atol);
      solver.SetRelTol(rtol);
      solver.SetMaxIter(maxIter);
      solver.SetKDim(50);
      solver.SetOperator(*A);
      solver.SetPreconditioner(amg);
      solver.SetPrintLevel(1);

      solver.Mult(B, X);
   }
   else
   {
      // 10. Construct the operators for preconditioner
      //
      //                 P = [ diag(M)         0         ]
      //                     [  0       B diag(M)^-1 B^T ]
      //
      //     Here we use Symmetric Gauss-Seidel to approximate the inverse of the
      //     pressure Schur Complement
      HypreParMatrix *MinvBt = NULL;
      HypreParVector *Md = NULL;
      HypreParMatrix *S = NULL;
      Solver *invM;
      HypreBoomerAMG *invS;

      HypreParMatrix &M = *Ms->ParallelAssembleInternalMatrix();
      Md = new HypreParVector(pmesh.GetComm(), M.GetGlobalNumRows(),
                              M.GetRowStarts());
      M.GetDiag(*Md);

      HypreParMatrix &Bm = *Bs->ParallelAssembleInternalMatrix();
      MinvBt = Bm.Transpose();
      MinvBt->InvScaleRows(*Md);
      S = ParMult(&Bm, MinvBt);

      if (Mu)
      {
         HypreParMatrix &Mum = *Mu->ParallelAssembleInternalMatrix();
         HypreParMatrix *Snew = ParAdd(&Mum, S);
         delete S;
         S = Snew;
      }

      invM = new HypreDiagScale(M);
      invS = new HypreBoomerAMG(*S);
      invS->SetSystemsOptions(dim, true);

      invM->iterative_mode = false;
      invS->iterative_mode = false;

      BlockDiagonalPreconditioner darcyPrec(block_trueOffsets);
      darcyPrec.SetDiagonalBlock(0, invM);
      darcyPrec.SetDiagonalBlock(1, invS);

      // 13. Solve the linear system with MINRES.
      //     Check the norm of the unpreconditioned residual.

      MINRESSolver solver(pmesh.GetComm());
      solver.SetAbsTol(atol);
      solver.SetRelTol(rtol);
      solver.SetMaxIter(maxIter);
      solver.SetOperator(*A);
      solver.SetPreconditioner(darcyPrec);
      solver.SetPrintLevel(1);

      solver.Mult(B, X);

      delete invM;
      delete invS;
      delete S;
      delete Md;
      delete MinvBt;
   }

   // 12. Recover the solution as a finite element grid function 'x'.
   darcy.RecoverFEMSolution(X, x);

   // 13. Use the DG solution space as the mesh nodal space. This allows us to
   //     save the displaced mesh as a curved DG mesh.
   pmesh.SetNodalFESpace(&W_space);

   Vector reference_nodes;
   if (visualization) { reference_nodes = *pmesh.GetNodes(); }

   // 14. Save the displaced mesh and minus the solution (which gives the
   //     backward displacements to the reference mesh). This output can be
   //     viewed later using GLVis: "glvis -m displaced.mesh -g sol.gf".
   {
      *pmesh.GetNodes() += u;
      u.Neg(); // x = -x

      ostringstream mesh_name, sol_name;
      mesh_name << "mesh." << setfill('0') << setw(6) << Mpi::WorldRank();
      sol_name << "sol." << setfill('0') << setw(6) << Mpi::WorldRank();

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      mesh_ofs << pmesh;

      ofstream sol_ofs(sol_name.str().c_str());
      sol_ofs.precision(8);
      sol_ofs << u;
   }

   // 15. Visualization: send data by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      VisMan vis(vishost, visport);
      const char *glvis_keys = (dim < 3) ? "Rjlc" : "c";

      // Visualize the deformed configuration.
      vis << new_window << setprecision(8)
          << "parallel " << pmesh.GetNRanks() << ' ' << pmesh.GetMyRank()
          << '\n'
          << "solution\n" << pmesh << u << flush
          << "keys " << glvis_keys << endl
          << "window_title 'Deformed configuration'" << endl
          << "plot_caption 'Backward displacement'" << endl
          << position_window << close_connection;

      // Visualize the stress components.
      const char *c = "xyz";
      ParFiniteElementSpace R_space_scalar(&pmesh, &R_coll);
      ParGridFunction stress_scal(&R_space_scalar, x.GetBlock(0), 0);
      ParGridFunction stress_idx, stress_diag(&R_space_scalar);
      *pmesh.GetNodes() = reference_nodes;

      int idx = 1;
      for (int si = 0; si < dim; si++)
      {
         for (int sj = si; sj < dim; sj++)
         {
            stress_idx.MakeRef(&R_space_scalar, x.GetBlock(0),
                               R_space_scalar.GetVSize() * idx);
            if (sj == si)
            {
               add(stress_scal, stress_idx, stress_diag);
            }

            MPI_Barrier(MPI_COMM_WORLD);
            vis << new_window << setprecision(8)
                << "parallel " << pmesh.GetNRanks() << ' ' << pmesh.GetMyRank()
                << '\n'
                << "solution\n" << pmesh << ((sj==si)?(stress_diag):(stress_idx)) << flush
                << "keys " << glvis_keys << endl
                << "window_title |Stress " << c[si] << c[sj] << '|' << endl
                << position_window << close_connection;
            idx++;
         }
      }
   }

   // clean-up
   delete trace_space;
   delete trace_coll;

   return 0;
}


void InitDisplacement(const Vector &x, Vector &u)
{
   u = 0.0;
   u(u.Size()-1) = -0.2*x(0);
}

VisMan::VisMan(const char *vishost, const int visport)
   : iostream(0),
     host(vishost), port(visport), sid(0)
{
   win_x = 0;
   win_y = 0;
   win_w = 400; // window width
   win_h = 350; // window height
   win_stride_x = win_w;
   win_stride_y = win_h + 20;
   win_nx = 4; // number of windows in a row
}

void VisMan::NewWindow()
{
   sock.Append(new socketstream(host, port));
   sid = sock.Size()-1;
   iostream::rdbuf(sock[sid]->rdbuf());
}

void VisMan::CloseConnection()
{
   if (sid < sock.Size())
   {
      delete sock[sid];
      sock[sid] = NULL;
      iostream::rdbuf(0);
   }
}

void VisMan::PositionWindow()
{
   *this << "window_geometry "
         << win_x + win_stride_x*(sid%win_nx) << ' '
         << win_y + win_stride_y*(sid/win_nx) << ' '
         << win_w << ' ' << win_h << endl;
}

VisMan::~VisMan()
{
   for (int i = sock.Size()-1; i >= 0; i--)
   {
      delete sock[i];
   }
}

ostream &operator<<(ostream &v, void (*f)(VisMan&))
{
   VisMan *vp = dynamic_cast<VisMan*>(&v);
   if (vp) { (*f)(*vp); }
   return v;
}
