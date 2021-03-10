

//                                MFEM Example multigrid-grid Cycle

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

void maxwell_solution(const Vector &x, std::vector<complex<double>> &sol);

void E_exact_re(const Vector &x, Vector &E);
void H_exact_re(const Vector &x, Vector &H);
void E_exact_im(const Vector &x, Vector &E);
void H_exact_im(const Vector &x, Vector &H);

void f_exact_re(const Vector &x, Vector &f);
void g_exact_re(const Vector &x, Vector &g);
void f_exact_im(const Vector &x, Vector &f);
void g_exact_im(const Vector &x, Vector &g);
void plotfield(socketstream &,ParMesh * pmesh,const ParGridFunction & , string &);

int dim;
double omega;

   // ----------------------------------------------------------------------
   // |   |            E             |             H          |     RHS    | 
   // ----------------------------------------------------------------------
   // | F | (curlE,curlF)+w^2(E,F)   | iw(curlH,F)+iw(H,curF) | -iw(J,F)   |
   // |   |                          |                        |            |
   // | G |-iw(E,curlG)-iw(curlE,G)  | (curlH,curlG)+w^2(H,G) | -(J,curlG) |

int main(int argc, char *argv[])
{
    // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 2. Parse command-line options.
   const char *mesh_file = "../../data/inline-hex.mesh";
   int order = 1;
   bool visualization = 1;
   int sr = 1;
   int pr = 1;
   double rnum=1.0;
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

   // omega = 2.0 * M_PI * rnum;
   omega = rnum;
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   dim = mesh->Dimension();

   MFEM_VERIFY(dim == 3, "only 3D problems supported by this formulation");

   for (int i = 0; i < sr; i++ )
   {
      mesh->UniformRefinement();
   }

   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   for (int i = 0; i < pr; i++ )
   {
      pmesh->UniformRefinement();
   }

   FiniteElementCollection *fec = new ND_FECollection(order,dim); 
   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);

   Array<int> ess_tdof_list;
   Array<int> ess_bdr;
   if (pmesh->bdr_attributes.Size())
   {
      ess_bdr.SetSize(pmesh->bdr_attributes.Max());
      ess_bdr = 1;
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }


   VectorFunctionCoefficient E_ex_re(dim,E_exact_re);
   VectorFunctionCoefficient E_ex_im(dim,E_exact_im);
   VectorFunctionCoefficient H_ex_re(dim,H_exact_re);
   VectorFunctionCoefficient H_ex_im(dim,H_exact_im);

   VectorFunctionCoefficient f_ex_re(dim,f_exact_re);
   VectorFunctionCoefficient f_ex_im(dim,f_exact_im);
   VectorFunctionCoefficient g_ex_re(dim,g_exact_re);
   VectorFunctionCoefficient g_ex_im(dim,g_exact_im);

   int n = fespace->GetVSize();
   int N = fespace->GetTrueVSize();
   Array<int> block_offsets(5);
   block_offsets[0] = 0;
   block_offsets[1] = n;
   block_offsets[2] = n;
   block_offsets[3] = n;
   block_offsets[4] = n;
   block_offsets.PartialSum();

   Array<int> block_trueOffsets(5);
   block_trueOffsets[0] = 0;
   block_trueOffsets[1] = N;
   block_trueOffsets[2] = N;
   block_trueOffsets[3] = N;
   block_trueOffsets[4] = N;
   block_trueOffsets.PartialSum();

   BlockVector x(block_offsets), rhs(block_offsets);
   BlockVector X(block_trueOffsets), Rhs(block_trueOffsets);
   x = 0.0;  rhs = 0.0; X = 0.0;  Rhs = 0.0;

   ParGridFunction E_gf_re, E_gf_im, H_gf_re, H_gf_im;

   E_gf_re.MakeRef(fespace,x.GetBlock(0)); E_gf_re = 0.0;
   E_gf_im.MakeRef(fespace,x.GetBlock(1)); E_gf_im = 0.0;
   H_gf_re.MakeRef(fespace,x.GetBlock(2)); H_gf_re = 0.0;
   H_gf_im.MakeRef(fespace,x.GetBlock(3)); H_gf_im = 0.0;

   // E_gf_re.ProjectBdrCoefficientTangent(E_ex_re,ess_bdr);
   // E_gf_im.ProjectBdrCoefficientTangent(E_ex_im,ess_bdr);
   E_gf_re.ProjectCoefficient(E_ex_re);
   E_gf_im.ProjectCoefficient(E_ex_im);

   // ----------------------------------------------------------------------
   // |   |            E             |             H          |     RHS    | 
   // ----------------------------------------------------------------------
   // | F | (curlE,curlF)+w^2(E,F)   | iw(curlH,F)+iw(H,curF) | -iw(J,F)   |
   // |   |                          |                        |            |
   // | G |-iw(E,curlG)-iw(curlE,G)  | (curlH,curlG)+w^2(H,G) | -(J,curlG) |
   // ----------------------------------------------------------------------

   // for convinience we convert the above 2 x 2 blocks to 4 x 4 in order
   // to accomodate complex valued operators

   // A = (curlE,curlF)+w^2(E,F)
   // B = w(curlH,F)+w(H,curF)
   // C = -w(E,curlG)-w(curlE,G)
   // D = (curlH,curlG)+w^2(H,G) 
   // b0 = w(J_im,F)
   // b1 = -w(J_re,F)
   // b2 = -(J_re,curlG)
   // b3 = -(J_im,curlG)

   // | A   0   0  -B |  | E_re |     | b0 |
   // | 0   A   B   0 |  | E_im |  =  | b1 |
   // | 0  -C   D   0 |  | H_re |     | b2 |
   // | C   0   0   D |  | H_im |     | b3 |

   ConstantCoefficient one(1.0);
   ConstantCoefficient negone(-1.0);
   ConstantCoefficient negomeg(-omega);
   ConstantCoefficient omeg(omega);
   ConstantCoefficient omeg2(omega * omega);



   // All these work with wrong sign
   ScalarVectorProductCoefficient prod0(omeg,g_ex_im);
   // This should be -omega * g_ex_re
   ScalarVectorProductCoefficient prod1(omeg,g_ex_re);
   // This should be - g_ex_re
   ScalarVectorProductCoefficient prod2(one,g_ex_re);
   // This should be - g_ex_im
   ScalarVectorProductCoefficient prod3(one,g_ex_im);


   ParLinearForm b0(fespace);
   ParLinearForm b1(fespace);
   ParLinearForm b2(fespace);
   ParLinearForm b3(fespace);
   b0.Update(fespace,rhs.GetBlock(0),0);
   b1.Update(fespace,rhs.GetBlock(1),0);
   b2.Update(fespace,rhs.GetBlock(2),0);
   b3.Update(fespace,rhs.GetBlock(3),0);
   
   b0.AddDomainIntegrator(new VectorFEDomainLFIntegrator(prod0));
   b1.AddDomainIntegrator(new VectorFEDomainLFIntegrator(prod1));
   b2.AddDomainIntegrator(new VectorFEDomainLFCurlIntegrator(prod2));
   b3.AddDomainIntegrator(new VectorFEDomainLFCurlIntegrator(prod3));
   
   b0.Assemble();
   b1.Assemble();
   b2.Assemble();
   b3.Assemble();

   Array2D<HypreParMatrix *> Ah(4,4); 
   for (int i = 0; i<4; i++)
   {
      for (int j = 0; j<4; j++)
      {
         Ah[i][j] = nullptr;
      }
   }


   ParBilinearForm a00(fespace);
   a00.AddDomainIntegrator(new CurlCurlIntegrator(one));
   a00.AddDomainIntegrator(new VectorFEMassIntegrator(omeg2));
   a00.Assemble();
   a00.EliminateEssentialBC(ess_bdr,x.GetBlock(0),rhs.GetBlock(0),mfem::Operator::DIAG_ONE);
   a00.Finalize();
   Ah[0][0] = a00.ParallelAssemble();

   ParMixedBilinearForm a03(fespace,fespace);
   a03.AddDomainIntegrator(new MixedVectorCurlIntegrator(negomeg));
   a03.AddDomainIntegrator(new MixedVectorWeakCurlIntegrator(negomeg));
   a03.Assemble();
   a03.EliminateTestDofs(ess_bdr);
   a03.Finalize();
   Ah[0][3] = a03.ParallelAssemble();

   ParBilinearForm a11(fespace);
   a11.AddDomainIntegrator(new CurlCurlIntegrator(one));
   a11.AddDomainIntegrator(new VectorFEMassIntegrator(omeg2));
   a11.Assemble();
   a11.EliminateEssentialBC(ess_bdr,x.GetBlock(1),rhs.GetBlock(1),mfem::Operator::DIAG_ONE);
   a11.Finalize();
   Ah[1][1] = a11.ParallelAssemble();

   ParMixedBilinearForm a12(fespace,fespace);
   a12.AddDomainIntegrator(new MixedVectorCurlIntegrator(omeg));
   a12.AddDomainIntegrator(new MixedVectorWeakCurlIntegrator(omeg));
   a12.Assemble();
   a12.EliminateTestDofs(ess_bdr);
   a12.Finalize();
   Ah[1][2] = a12.ParallelAssemble();

   ParMixedBilinearForm a21(fespace,fespace);
   a21.AddDomainIntegrator(new MixedVectorCurlIntegrator(omeg));
   a21.AddDomainIntegrator(new MixedVectorWeakCurlIntegrator(omeg));
   a21.Assemble();
   a21.EliminateTrialDofs(ess_bdr,x.GetBlock(1),rhs.GetBlock(2));
   a21.Finalize();
   Ah[2][1] = a21.ParallelAssemble();
   // Ah[2][1] = Ah[1][2]->Transpose();
   // (*Ah[2][1]) *=-1.0;

   ParBilinearForm a22(fespace);
   a22.AddDomainIntegrator(new CurlCurlIntegrator(one));
   a22.AddDomainIntegrator(new VectorFEMassIntegrator(omeg2));
   a22.Assemble();
   a22.Finalize();
   Ah[2][2] = a22.ParallelAssemble();

   ParMixedBilinearForm a30(fespace,fespace);
   a30.AddDomainIntegrator(new MixedVectorCurlIntegrator(omeg));
   a30.AddDomainIntegrator(new MixedVectorWeakCurlIntegrator(omeg));
   a30.Assemble();
   a30.EliminateTrialDofs(ess_bdr,x.GetBlock(0),rhs.GetBlock(3));
   a30.Finalize();
   Ah[3][0] = a30.ParallelAssemble();
   // Ah[3][0] = Ah[0][3]->Transpose();
   // (*Ah[3][0])*=-1.0;

   ParBilinearForm a33(fespace);
   a33.AddDomainIntegrator(new CurlCurlIntegrator(one));
   a33.AddDomainIntegrator(new VectorFEMassIntegrator(omeg2));
   a33.Assemble();
   a33.Finalize();
   Ah[3][3] = a33.ParallelAssemble();
   // Ah[3][3] = Ah[2][2];

   // HypreParMatrix * diff = new HypreParMatrix(*Ah[0][3]);
   // *diff += *Ah[3][0];


   for (int i = 0; i<4; i++)
   {
      fespace->GetRestrictionMatrix()->Mult(x.GetBlock(i), X.GetBlock(i));
      fespace->GetProlongationMatrix()->MultTranspose(rhs.GetBlock(i),Rhs.GetBlock(i));
   }

   HypreParMatrix * A = HypreParMatrixFromBlocks(Ah);
   {
      MUMPSSolver mumps;
      mumps.SetPrintLevel(0);
      mumps.SetOperator(*A);
      mumps.Mult(Rhs,X);
   }
   E_gf_re = 0.0;
   E_gf_im = 0.0;
   H_gf_re = 0.0;
   H_gf_im = 0.0;

   E_gf_re.Distribute(&(X.GetBlock(0)));
   E_gf_im.Distribute(&(X.GetBlock(1)));
   H_gf_re.Distribute(&(X.GetBlock(2)));
   H_gf_im.Distribute(&(X.GetBlock(3)));


   cout << "E_re L2 Error = " << E_gf_re.ComputeL2Error(E_ex_re) << endl;
   cout << "E_im L2 Error = " << E_gf_im.ComputeL2Error(E_ex_im) << endl;
   cout << "H_re L2 Error = " << H_gf_re.ComputeL2Error(H_ex_re) << endl;
   cout << "H_im L2 Error = " << H_gf_im.ComputeL2Error(H_ex_im) << endl;

   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock0(vishost, visport);
      socketstream sol_sock1(vishost, visport);
      socketstream sol_sock2(vishost, visport);
      socketstream sol_sock3(vishost, visport);
      string str0 = "E_re";
      plotfield(sol_sock0,pmesh, E_gf_re,str0);
      string str1 = "E_im";
      plotfield(sol_sock1,pmesh,E_gf_im,str1);
      string str2 = "H_re";
      plotfield(sol_sock2,pmesh,H_gf_re,str2);
      string str3 = "H_im";
      plotfield(sol_sock3,pmesh,H_gf_im,str3);

      ParGridFunction E_exact_re(fespace);
      ParGridFunction E_exact_im(fespace);
      ParGridFunction H_exact_re(fespace);
      ParGridFunction H_exact_im(fespace);
      E_exact_re.ProjectCoefficient(E_ex_re);
      E_exact_im.ProjectCoefficient(E_ex_im);
      H_exact_re.ProjectCoefficient(H_ex_re);
      H_exact_im.ProjectCoefficient(H_ex_im);

      socketstream sol_sock_ex0(vishost, visport);
      socketstream sol_sock_ex1(vishost, visport);
      socketstream sol_sock_ex2(vishost, visport);
      socketstream sol_sock_ex3(vishost, visport);
      str0 = "E_exact_re";
      plotfield(sol_sock_ex0,pmesh,E_exact_re,str0);
      str1 = "E_exact_im";
      plotfield(sol_sock_ex1,pmesh,E_exact_im,str1);
      str2 = "H_exact_re";
      plotfield(sol_sock_ex2,pmesh,H_exact_re,str2);
      str3 = "H_exact_im";
      plotfield(sol_sock_ex3,pmesh,H_exact_im,str3);
   }

   MPI_Finalize();
   return 0;
}


void maxwell_solution(const Vector &x, std::vector<complex<double>> &sol)
{

}

void E_exact_re(const Vector &x, Vector &E)
{
   // try constant 
   E = 1.0;
   // E[0] = 1.0;
}
void H_exact_re(const Vector &x, Vector &H)
{
   H = 0.0;
}
void E_exact_im(const Vector &x, Vector &E)
{
   E = 2.0;
}
void H_exact_im(const Vector &x, Vector &H)
{
   H = 0.0;
}

void f_exact_re(const Vector &x, Vector &f)
{
   f = 0.0;
}
void g_exact_re(const Vector &x, Vector &g)
{
   // J = i omega E - curl H 
   g = 0.0;
   g = 2*omega;
}
void f_exact_im(const Vector &x, Vector &f)
{
   f = 0.0;
}
void g_exact_im(const Vector &x, Vector &g)
{
   // J = i omega E - curl H 
   g = 0.0;
   // g(0) = omega;
   g = omega;
}


void plotfield(socketstream & socket, ParMesh * pmesh, const ParGridFunction & pgf, string & title )
{
   int num_procs, myid;
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   ostringstream oss;
   oss << title;
   socket << "parallel " << num_procs << " " << myid << "\n";
   socket.precision(8);
   socket << "solution\n" << *pmesh << pgf 
          << "window_title '" << oss.str() << "'" << flush;
}
