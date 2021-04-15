
#include "mfem.hpp"
#include <fstream>
#include <iostream>
using namespace std;
using namespace mfem;

// #define DEFINITE

double p_exact(const Vector &x);
void u_exact(const Vector &x, Vector & u);
double rhs_func(const Vector &x);
void gradp_exact(const Vector &x, Vector &gradu);
double divu_exact(const Vector &x);
double d2_exact(const Vector &x);

int dim;
double omega;
int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 2. Parse command-line options.
   const char *mesh_file = "../../data/inline-quad.mesh";
   int order = 1;
   bool visualization = 1;
   int sr = 1;
   int pr = 1;
   double rnum;
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree)");
   args.AddOption(&sr, "-sr", "--serial_ref",
                  "Number of serial refinements.");
   args.AddOption(&pr, "-pr", "--parallel_ref",
                  "Number of parallel refinements.");
   args.AddOption(&rnum, "-rnum", "--number_of_wavelenths",
                  "Number of wavelengths");                  
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
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
      args.PrintOptions(cout);
   }

   omega = 2.0 * M_PI * rnum;

   // 3. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   dim = mesh->Dimension();

   // 4. Refine the serial mesh on all processors to increase the resolution.
   for (int i = 0; i < sr; i++ )
   {
      mesh->UniformRefinement();
   }

   // 5. Define a parallel mesh by a partitioning of the serial mesh. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   // 6. Define a parallel finite element space on the parallel mesh.
   FiniteElementCollection *H1fec = new H1_FECollection(order,dim); 
   ParFiniteElementSpace *H1fespace = new ParFiniteElementSpace(pmesh, H1fec);

   FiniteElementCollection *RTfec = new RT_FECollection(order,dim); 
   ParFiniteElementSpace *RTfespace = new ParFiniteElementSpace(pmesh, RTfec);


   // -------------------------------------------------------------------
   // |   |            p             |           u           |   RHS    | 
   // -------------------------------------------------------------------
   // | q | (gradp,gradq) + (p,q)    | (divu,q)-w^2(u, gradq) |  (f,q)   |
   // |   |                          |                        |          |
   // | v | (p,divv) - w^2(gradp,v)   | (divu,divv) + w^4(u,v)| (f,divv) |


   // omega(f,q) 
   ParLinearForm b_q(H1fespace);
   ConstantCoefficient omeg(omega);
   ConstantCoefficient omeg2(omega*omega);
   FunctionCoefficient f_rhs(rhs_func);
   ProductCoefficient omega2_f(omeg2,f_rhs);
   b_q.AddDomainIntegrator(new DomainLFIntegrator(f_rhs));
   // (f, div v)
   ParLinearForm b_v(RTfespace);
#ifdef DEFINITE
   ConstantCoefficient negone(-1.0);
   ProductCoefficient neg_f(negone,f_rhs);
   b_v.AddDomainIntegrator(new VectorFEDomainLFDivIntegrator(neg_f));
#else    
   b_v.AddDomainIntegrator(new VectorFEDomainLFDivIntegrator(f_rhs));
#endif

   ParBilinearForm a_qp(H1fespace);
   ConstantCoefficient one(1.0);
   ConstantCoefficient negone(-1.0);
   ConstantCoefficient negomeg(-omega);
   ConstantCoefficient negomeg2(-omega*omega);
   ConstantCoefficient omeg4(omega*omega*omega*omega);
   // (grad p, grad q) + (p,q)
   a_qp.AddDomainIntegrator(new DiffusionIntegrator(one));
   a_qp.AddDomainIntegrator(new MassIntegrator(one));

   ParMixedBilinearForm a_qu(RTfespace, H1fespace);
#ifdef DEFINITE
   // -w(divu,q)
   a_qu.AddDomainIntegrator(new MixedScalarDivergenceIntegrator(negomeg));
#else   
   // (divu,q)
   a_qu.AddDomainIntegrator(new MixedScalarDivergenceIntegrator(one));
#endif
   // -w^2(u, gradq)
   a_qu.AddDomainIntegrator(new MixedVectorWeakDivergenceIntegrator(omeg2));
   // w^2(p,divv) - (gradp,v)
   ParMixedBilinearForm a_vp(H1fespace, RTfespace);
#ifdef DEFINITE
   // -w(p,divv)
   a_vp.AddDomainIntegrator(new MixedScalarWeakGradientIntegrator(omeg));
#else
   // (p,divv)
   a_vp.AddDomainIntegrator(new MixedScalarWeakGradientIntegrator(negone));
#endif
   // - w^2(gradp,v)
   a_vp.AddDomainIntegrator(new MixedVectorGradientIntegrator(negomeg2));

   ParBilinearForm a_vu(RTfespace);
   a_vu.AddDomainIntegrator(new DivDivIntegrator(one));
   a_vu.AddDomainIntegrator(new VectorFEMassIntegrator(omeg4));


   ConvergenceStudy ratesH1;
   ConvergenceStudy ratesRT;
   FunctionCoefficient p_ex(p_exact);
   VectorFunctionCoefficient gradp_ex(dim,gradp_exact);
   VectorFunctionCoefficient u_ex(dim,u_exact);
   FunctionCoefficient divu_ex(divu_exact);
   ParGridFunction p_gf, u_gf;

   for (int l = 0; l <= pr; l++)
   {
      Array<int> ess_tdof_list;
      Array<int> ess_bdr;
      if (pmesh->bdr_attributes.Size())
      {
         ess_bdr.SetSize(pmesh->bdr_attributes.Max());
         ess_bdr = 1;
         H1fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      }

       Array<int> block_offsets(3);
      block_offsets[0] = 0;
      block_offsets[1] = H1fespace->GetVSize();
      block_offsets[2] = RTfespace->GetVSize();
      block_offsets.PartialSum();

      Array<int> block_trueOffsets(3);
      block_trueOffsets[0] = 0;
      block_trueOffsets[1] = H1fespace->TrueVSize();
      block_trueOffsets[2] = RTfespace->TrueVSize();
      block_trueOffsets.PartialSum();

      BlockVector x(block_offsets), rhs(block_offsets);
      BlockVector trueX(block_trueOffsets), trueRhs(block_trueOffsets);
      x = 0.0;  rhs = 0.0;
      trueX = 0.0;  trueRhs = 0.0;

      p_gf.MakeRef(H1fespace,x.GetBlock(0));
      p_gf.ProjectBdrCoefficient(p_ex,ess_bdr);

      u_gf.MakeRef(RTfespace,x.GetBlock(1));
      u_gf = 0.0;

      b_q.Update(H1fespace,rhs.GetBlock(0),0);
      b_q.Assemble();

      b_v.Update(RTfespace,rhs.GetBlock(1),0);
      b_v.Assemble();


      a_qp.Assemble();
      a_qp.EliminateEssentialBC(ess_bdr,x.GetBlock(0),rhs.GetBlock(0));
      a_qp.Finalize();
      HypreParMatrix * A_qp = a_qp.ParallelAssemble();

      a_qu.Assemble();
      a_qu.EliminateTestDofs(ess_bdr);
      a_qu.Finalize();
      HypreParMatrix * A_qu = a_qu.ParallelAssemble();


      a_vp.Assemble();
      a_vp.EliminateTrialDofs(ess_bdr,x.GetBlock(0),rhs.GetBlock(1));
      a_vp.Finalize();
      HypreParMatrix * A_vp = a_vp.ParallelAssemble();

      a_vu.Assemble();
      a_vu.Finalize();
      HypreParMatrix * A_vu = a_vu.ParallelAssemble();



      H1fespace->GetRestrictionMatrix()->Mult(x.GetBlock(0), trueX.GetBlock(0));
      H1fespace->GetProlongationMatrix()->MultTranspose(rhs.GetBlock(0),trueRhs.GetBlock(0));

      RTfespace->GetRestrictionMatrix()->Mult(x.GetBlock(1), trueX.GetBlock(1));
      RTfespace->GetProlongationMatrix()->MultTranspose(rhs.GetBlock(1),trueRhs.GetBlock(1));


      Array2D<HypreParMatrix *> Ah(2,2);
      Ah[0][0] = A_qp; 
      Ah[0][1] = A_qu;
      Ah[1][0] = A_vp;
      Ah[1][1] = A_vu;
      HypreParMatrix * A = HypreParMatrixFromBlocks(Ah);

      HypreBoomerAMG amg_p(*A_qp);
      amg_p.SetPrintLevel(0);

      Solver *prec = nullptr;
      if (dim == 2) 
      {
         prec = new HypreAMS(*A_vu,RTfespace);
         dynamic_cast<HypreAMS *>(prec)->SetPrintLevel(0);
      }
      else
      {
         prec = new HypreADS(*A_vu,RTfespace);
         dynamic_cast<HypreADS *>(prec)->SetPrintLevel(0);
      }

      BlockDiagonalPreconditioner M(block_trueOffsets);
      // BlockDiagonalMultiplicativePreconditioner M(block_trueOffsets);
      M.SetOperator(*A);
      M.SetDiagonalBlock(0,&amg_p);
      M.SetDiagonalBlock(1,prec);

      StopWatch chrono;
      chrono.Clear();
      chrono.Start();
      // GMRESSolver cg(MPI_COMM_WORLD);
      CGSolver cg(MPI_COMM_WORLD);
      cg.SetRelTol(1e-6);
      // cg.SetAbsTol(1e-6);
      cg.SetMaxIter(2000);
      cg.SetPrintLevel(1);
      cg.SetPreconditioner(M);
      cg.SetOperator(*A);
      cg.Mult(trueRhs, trueX);
      delete prec;
      chrono.Stop();
      cout << "PCG time " << chrono.RealTime() << endl;

      chrono.Clear();
      chrono.Start();
      MUMPSSolver mumps;
      mumps.SetPrintLevel(0);
      mumps.SetMatrixSymType(MUMPSSolver::MatType::UNSYMMETRIC);
      mumps.SetOperator(*A);
      mumps.Mult(trueRhs,trueX);
      chrono.Stop();
      cout << "MUMPS time " << chrono.RealTime() << endl;




      delete A;
      delete A_vu;
      delete A_qp;
      delete A_vp;
      delete A_qu;

      p_gf = 0.0;
      u_gf = 0.0;
      p_gf.Distribute(&(trueX.GetBlock(0)));
      u_gf.Distribute(&(trueX.GetBlock(1)));

      ratesH1.AddH1GridFunction(&p_gf,&p_ex,&gradp_ex);
      ratesRT.AddHdivGridFunction(&u_gf,&u_ex,&divu_ex);

      if (l==pr) break;

      pmesh->UniformRefinement();
      H1fespace->Update();
      RTfespace->Update();
      a_qp.Update();
      a_qu.Update();
      a_vp.Update();
      a_vu.Update();
      b_q.Update();
      b_v.Update();
      p_gf.Update();
      u_gf.Update();
   }
   ratesH1.Print(true);
   ratesRT.Print(true);

   // 10. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << *pmesh << p_gf <<
               "window_title 'Numerical Pressure (real part)' "
               << flush;
   }

   // // 11. Free the used memory.
   delete H1fespace;
   delete RTfespace;
   delete H1fec;
   delete RTfec;
   delete pmesh;
   MPI_Finalize();
   return 0;
}

double rhs_func(const Vector &x)
{
   double p = p_exact(x);
   double divu = divu_exact(x);
#ifdef DEFINITE   
   return -divu + omega * p;
#else
   return divu +  p;
#endif   
}

double p_exact(const Vector &x)
{
   return sin(omega*x.Sum());
}

void gradp_exact(const Vector &x, Vector &grad)
{
   grad.SetSize(x.Size());
   grad = omega * cos(omega * x.Sum());
}

void u_exact(const Vector &x, Vector & u)
{
   gradp_exact(x,u);
   u /= (omega*omega);
}

double divu_exact(const Vector &x)
{
   return d2_exact(x)/(omega*omega);
}

double d2_exact(const Vector &x)
{
   return -dim * omega * omega * sin(omega*x.Sum());
}