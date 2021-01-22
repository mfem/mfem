
#include "mfem.hpp"
#include <fstream>
#include <iostream>
using namespace std;
using namespace mfem;

double p_exact(const Vector &x);
double rhs_func(const Vector &x);
void gradp_exact(const Vector &x, Vector &gradu);
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
   // | q | (gradp,gradq) + w^2(p,q) | w(divu,q)-w(u, gradq) |  w(f,q)  |
   // |   |                          |                       |          |
   // | v | w(p,divv) - w(gradp,v)   | (divu,divv) + w^2(u,v)| (f,divv) |


   // omega(f,q) 
   ParLinearForm b_q(H1fespace);
   ConstantCoefficient omeg(omega);
   FunctionCoefficient f_rhs(rhs_func);
   ProductCoefficient omega_f(omeg,f_rhs);
   b_q.AddDomainIntegrator(new DomainLFIntegrator(omega_f));

   // (f, div v)
   ParLinearForm b_v(RTfespace);
   b_v.AddDomainIntegrator(new VectorFEDomainLFDivIntegrator(f_rhs));


   ParBilinearForm a_qp(H1fespace);
   ConstantCoefficient one(1.0);
   ConstantCoefficient negomeg(-omega);
   ConstantCoefficient omeg2(omega*omega);
   // (grad p, grad q) + \omega^2 (p,q)
   a_qp.AddDomainIntegrator(new DiffusionIntegrator(one));
   a_qp.AddDomainIntegrator(new MassIntegrator(omeg2));

   // w(divu,q)-w(u, gradq)
   ParMixedBilinearForm a_qu(RTfespace, H1fespace);
   a_qu.AddDomainIntegrator(new MixedScalarDivergenceIntegrator(omeg));
   a_qu.AddDomainIntegrator(new MixedVectorWeakDivergenceIntegrator(omeg));

   // w(p,divv) - w(gradp,v)
   ParMixedBilinearForm a_vp(H1fespace, RTfespace);
   a_vp.AddDomainIntegrator(new MixedScalarWeakGradientIntegrator(negomeg));
   a_vp.AddDomainIntegrator(new MixedVectorGradientIntegrator(omeg));



   ParBilinearForm a_vu(RTfespace);
   a_vu.AddDomainIntegrator(new DivDivIntegrator(one));
   a_vu.AddDomainIntegrator(new VectorFEMassIntegrator(omeg2));

   b_q.Assemble();
   b_v.Assemble();
   a_qp.Assemble(); a_qp.Finalize();
   a_qu.Assemble(); a_qu.Finalize();
   a_vp.Assemble(); a_vp.Finalize();
   a_vu.Assemble(); a_vu.Finalize();
   


   // ParGridFunction x(fespace);
   // x = 0.0;
   // FunctionCoefficient p_ex(p_exact);
   // VectorFunctionCoefficient gradp_ex(dim,gradp_exact);

   // // 9. Perform successive parallel refinements, compute the L2 error and the
   // //    corresponding rate of convergence.
   // ConvergenceStudy rates;
   // for (int l = 0; l <= pr; l++)
   // {

   //    Array<int> ess_tdof_list;
   //    Array<int> ess_bdr;
   //    if (pmesh->bdr_attributes.Size())
   //    {
   //       ess_bdr.SetSize(pmesh->bdr_attributes.Max());
   //       ess_bdr = 1;
   //       fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   //    }

   //    b.Assemble();
   //    a.Assemble();
   //    x.ProjectBdrCoefficient(p_ex,ess_bdr);      

   //    OperatorPtr A;
   //    Vector B, X;
   //    a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

   //    MUMPSSolver mumps;
   //    mumps.SetPrintLevel(0);
   //    mumps.SetMatrixSymType(MUMPSSolver::MatType::UNSYMMETRIC);
   //    mumps.SetOperator(*A);
   //    mumps.Mult(B,X);

   //    a.RecoverFEMSolution(X, b, x);

   //    rates.AddH1GridFunction(&x,&p_ex,&gradp_ex);

   //    if (l==pr) break;

   //    pmesh->UniformRefinement();
   //    fespace->Update();
   //    a.Update();
   //    b.Update();
   //    x.Update();
   // }
   // rates.Print();

   // // 10. Send the solution by socket to a GLVis server.
   // if (visualization)
   // {
   //    char vishost[] = "localhost";
   //    int  visport   = 19916;
   //    socketstream sol_sock(vishost, visport);
   //    sol_sock << "parallel " << num_procs << " " << myid << "\n";
   //    sol_sock.precision(8);
   //    sol_sock << "solution\n" << *pmesh << x <<
   //             "window_title 'Numerical Pressure (real part)' "
   //             << flush;
   // }

   // // 11. Free the used memory.
   // delete fespace;
   // delete fec;
   delete pmesh;
   MPI_Finalize();
   return 0;
}

double rhs_func(const Vector &x)
{
   double p = p_exact(x);
   double d2p = d2_exact(x);
   return -d2p - omega * omega * p;
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

double d2_exact(const Vector &x)
{
   return -dim * omega * omega * sin(omega*x.Sum());
}