
#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "FOSLS.hpp"
using namespace std;
using namespace mfem;

// #define DEFINITE

const Array<int> &GetDofMap(FiniteElementSpace &fes, int i)
{
   const FiniteElement *fe = fes.GetFE(i);
   auto tfe = dynamic_cast<const TensorBasisElement*>(fe);
   MFEM_ASSERT(tfe != NULL, "");
   return tfe->GetDofMap();
}

Array<int> ComputeVectorFE_LORPermutation(
   FiniteElementSpace &fes_ho,
   FiniteElementSpace &fes_lor,
   FiniteElement::MapType type)
{
   // Given an index `i` of a LOR dof, `perm[i]` is the index of the
   // corresponding HO dof.
   Array<int> perm(fes_lor.GetVSize());
   Array<int> vdof_ho, vdof_lor;

   Mesh &mesh_lor = *fes_lor.GetMesh();
   int dim = mesh_lor.Dimension();
   const CoarseFineTransformations &cf_tr = mesh_lor.GetRefinementTransforms();
   for (int ilor=0; ilor<mesh_lor.GetNE(); ++ilor)
   {
      int iho = cf_tr.embeddings[ilor].parent;
      int lor_index = cf_tr.embeddings[ilor].matrix;

      int p = fes_ho.GetOrder(iho);
      int p1 = p+1;
      int ndof_per_dim = (dim == 2) ? p*p1 :
                         type == FiniteElement::H_CURL ? p*p1*p1 : p*p*p1;

      fes_ho.GetElementVDofs(iho, vdof_ho);
      fes_lor.GetElementVDofs(ilor, vdof_lor);

      const Array<int> &dofmap_ho = GetDofMap(fes_ho, iho);
      const Array<int> &dofmap_lor = GetDofMap(fes_lor, ilor);

      int off_x = lor_index % p;
      int off_y = (lor_index / p) % p;
      int off_z = (lor_index / p) / p;

      auto absdof = [](int i) { return i < 0 ? -1-i : i; };

      auto set_perm = [&](int off_lor, int off_ho, int n1, int n2)
      {
         for (int i1=0; i1<2; ++i1)
         {
            int m = (dim == 2 || type == FiniteElement::H_DIV) ? 1 : 2;
            for (int i2=0; i2<m; ++i2)
            {
               int i;
               i = dofmap_lor[off_lor + i1 + i2*2];
               int s1 = i < 0 ? -1 : 1;
               int idof_lor = vdof_lor[absdof(i)];
               i = dofmap_ho[off_ho + i1*n1 + i2*n2];
               int s2 = i < 0 ? -1 : 1;
               int idof_ho = vdof_ho[absdof(i)];
               int s3 = idof_lor < 0 ? -1 : 1;
               int s4 = idof_ho < 0 ? -1 : 1;
               int s = s1*s2*s3*s4;
               i = absdof(idof_ho);
               perm[absdof(idof_lor)] = s < 0 ? -1-absdof(i) : absdof(i);
            }
         }
      };

      int offset;

      if (type == FiniteElement::H_CURL)
      {
         // x
         offset = off_x + off_y*p + off_z*p*p1;
         set_perm(0, offset, p, p*p1);
         // y
         offset = ndof_per_dim + off_x + off_y*(p1) + off_z*p1*p;
         set_perm(dim == 2 ? 2 : 4, offset, 1, p*p1);
         // z
         if (dim == 3)
         {
            offset = 2*ndof_per_dim + off_x + off_y*p1 + off_z*p1*p1;
            set_perm(8, offset, 1, p+1);
         }
      }
      else
      {
         // x
         offset = off_x + off_y*p1 + off_z*p*p1;
         set_perm(0, offset, 1, 0);
         // y
         offset = ndof_per_dim + off_x + off_y*p + off_z*p1*p;
         set_perm(2, offset, p, 0);
         // z
         if (dim == 3)
         {
            offset = 2*ndof_per_dim + off_x + off_y*p + off_z*p*p;
            set_perm(4, offset, p*p, 0);
         }
      }
   }

   return perm;
}

class LORH1HdivDirectSolver : public Solver
{
public:
   LORH1HdivDirectSolver(HypreParMatrix & A, const Array<int> p_, bool exact = true, Solver * prec = nullptr)
   : Solver(A.Height()), p(p_)
{
   if (exact)
   {
      solv = new MUMPSSolver;
      dynamic_cast<MUMPSSolver*>(solv)->SetOperator(A);
   }
   else
   {
      solv = prec;
   }
   
   int n = A.Height();
   n2 = p.Size();
   n1 = n - n2;
   perm.SetSize(n);

   for (int i = 0; i<n1; i++) { perm[i] = i; }
   for (int i = 0; i<n2; i++) { perm[i+n1] = p[i]; }

}
void SetOperator(const Operator&) { }

void Mult(const Vector &b, Vector &x) const
{
   Vector bp(b.Size());
   Vector xp(x.Size());
   
   for (int i=0; i<n1; ++i) 
   { 
      bp[i] = b[i];
   }
   
   for (int i=n1; i<n1+n2; ++i) 
   { 
      int m = perm[i] < 0 ? n1-1-perm[i] : n1+perm[i];
      bp[i] = perm[i] < 0 ? -b[m] : b[m]; 
   }

   solv->Mult(bp, xp);

   for (int i=0; i<n1; ++i)
   {
      x[i] = xp[i];
   }
   for (int i=n1; i<x.Size(); ++i)
   {
      int pi = perm[i];
      int s = pi < 0 ? -1 : 1;
      int n = pi < 0 ? n1-1-pi : n1 + pi;
      x[n] = s*xp[i];
   }
}

private:
   int n1;
   int n2;
   Array<int> perm;
   Array<int> p;
   Solver *solv=nullptr;
};


double p_exact(const Vector &x);
void u_exact(const Vector &x, Vector & u);
double rhs_func(const Vector &x);
void gradp_exact(const Vector &x, Vector &gradu);
double divu_exact(const Vector &x);
double d2_exact(const Vector &x);

#ifdef DEFINITE   
   bool definite = true;
#else
   bool definite = false;
#endif
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

   omega = 2.0 * M_PI * rnum;

   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   dim = mesh->Dimension();

   // 4. Refine the serial mesh on all processors to increase the resolution.
   for (int i = 0; i < sr; i++ )
   {
      mesh->UniformRefinement();
   }

   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   int btype = BasisType::GaussLobatto;
   ParMesh pmesh_lor(pmesh, order, btype);

   unique_ptr<FiniteElementCollection> H1fec_ho, H1fec_lor;
   unique_ptr<FiniteElementCollection> RTfec_ho, RTfec_lor;

   H1fec_ho.reset(new H1_FECollection(order, dim));
   H1fec_lor.reset(new H1_FECollection(1, dim));
   RTfec_ho.reset(new RT_FECollection(order-1, dim, BasisType::GaussLobatto, BasisType::Integrated));
   RTfec_lor.reset(new RT_FECollection(0, dim, BasisType::GaussLobatto, BasisType::Integrated));
     
   ParFiniteElementSpace H1fes_ho(pmesh, H1fec_ho.get());
   ParFiniteElementSpace H1fes_lor(&pmesh_lor, H1fec_lor.get());
   ParFiniteElementSpace RTfes_ho(pmesh, RTfec_ho.get());
   ParFiniteElementSpace RTfes_lor(&pmesh_lor, RTfec_lor.get());

   Array<int> block_trueOffsets(3);
   block_trueOffsets[0] = 0;
   block_trueOffsets[1] = H1fes_ho.TrueVSize();
   block_trueOffsets[2] = RTfes_ho.TrueVSize();
   block_trueOffsets.PartialSum();

   BlockVector trueX(block_trueOffsets), trueRhs(block_trueOffsets);
   trueX = 0.0;  trueRhs = 0.0;

   FunctionCoefficient p_ex(p_exact);
   FunctionCoefficient f_rhs(rhs_func);
   VectorFunctionCoefficient gradp_ex(dim,gradp_exact);
   VectorFunctionCoefficient u_ex(dim,u_exact);
   FunctionCoefficient divu_ex(divu_exact);

   Vector trueY(trueX);
   Vector trueZ(trueX);

   Array<ParFiniteElementSpace *> fes_ho(2);
   fes_ho[0] = &H1fes_ho;
   fes_ho[1] = &RTfes_ho;

   HelmholtzFOSLS ho_system(fes_ho,definite);
   ho_system.SetOmega(omega);
   ho_system.SetLoadData(&f_rhs);
   ho_system.SetEssentialData(&p_ex);
   Array<ParFiniteElementSpace *> fes_lor(2);
   fes_lor[0] = &H1fes_lor;
   fes_lor[1] = &RTfes_lor;
   HelmholtzFOSLS lor_system(fes_lor,definite);
   lor_system.SetOmega(omega);

   Array2D<HypreParMatrix *> Ah_ho(2,2);
   ho_system.GetFOSLSLinearSystem(Ah_ho,trueX,trueRhs);
   Array2D<HypreParMatrix *> Ah_lor(2,2);
   lor_system.GetFOSLSMatrix(Ah_lor);
   HypreParMatrix * A_ho = HypreParMatrixFromBlocks(Ah_ho);
   HypreParMatrix * A_lor = HypreParMatrixFromBlocks(Ah_lor);


   HypreBoomerAMG amg_p(*Ah_ho[0][0]);
   amg_p.SetPrintLevel(0);
   HypreBoomerAMG amg_lor_p(*Ah_lor[0][0]);
   amg_lor_p.SetPrintLevel(0);

   Solver *prec = nullptr;
   Solver *prec_lor = nullptr;
   if (dim == 2) 
   {
      prec = new HypreAMS(*Ah_ho[1][1],&RTfes_ho);
      dynamic_cast<HypreAMS *>(prec)->SetPrintLevel(0);
      prec_lor = new HypreAMS(*Ah_lor[1][1],&RTfes_lor);
      dynamic_cast<HypreAMS *>(prec_lor)->SetPrintLevel(0);
   }
   else
   {
      prec = new HypreADS(*Ah_ho[1][1],&RTfes_ho);
      dynamic_cast<HypreADS *>(prec)->SetPrintLevel(0);
      prec_lor = new HypreADS(*Ah_lor[1][1],&RTfes_lor);
      dynamic_cast<HypreADS *>(prec_lor)->SetPrintLevel(0);
   }

   BlockDiagonalPreconditioner M(block_trueOffsets);
   BlockDiagonalPreconditioner M_lor2(block_trueOffsets);

   FiniteElement::MapType t = FiniteElement::H_DIV;
   Array<int> perm = ComputeVectorFE_LORPermutation(RTfes_ho, RTfes_lor, t);
   

   LORH1HdivDirectSolver M_lor(*A_lor, perm);
   M.SetDiagonalBlock(0,&amg_p);
   ScaledOperator S(prec,1.0);
   M.SetDiagonalBlock(1,&S);

   M_lor2.SetDiagonalBlock(0,&amg_lor_p);
   ScaledOperator S_lor(prec_lor,1.0);
   M_lor2.SetDiagonalBlock(1,&S_lor);

   LORH1HdivDirectSolver M_lor_inexact(*A_lor, perm, false, &M_lor2);

   StopWatch chrono;
   chrono.Clear();
   chrono.Start();
   // GMRESSolver cg(MPI_COMM_WORLD);
   CGSolver cg(MPI_COMM_WORLD);
   cg.SetRelTol(1e-6);
   // cg.SetAbsTol(1e-6);
   cg.SetMaxIter(2000);
   cg.SetPrintLevel(3);
   cg.SetOperator(*A_ho);
   // cg.SetPreconditioner(M);
   cg.SetPreconditioner(M_lor);
   cg.Mult(trueRhs, trueX);
   chrono.Stop();
   cout << "LOR exact - PCG time " << chrono.RealTime() << endl;

   chrono.Clear();
   chrono.Start();
   cg.SetPreconditioner(M_lor_inexact);
   cg.Mult(trueRhs, trueY);

   chrono.Stop();
   cout << "LOR inexact PCG time " << chrono.RealTime() << endl;

   chrono.Clear();
   chrono.Start();
   cg.SetPreconditioner(M);
   cg.Mult(trueRhs, trueZ);
   delete prec;

   chrono.Stop();
   cout << "AMG/AMS PCG time " << chrono.RealTime() << endl;

   for (int i = 0; i<2; i++)
   {
      for (int j = 0; j<2; j++)
      {
         delete Ah_ho[i][j];
         delete Ah_lor[i][j];
      }
   }

   ParGridFunction p_gf(&H1fes_ho);
   ParGridFunction u_gf(&RTfes_ho);
   ParGridFunction p_zero(&H1fes_ho);
   ParGridFunction u_zero(&RTfes_ho);
   p_gf = 0.0; p_zero = 0.0;
   u_gf = 0.0; u_zero = 0.0;
   p_gf.Distribute(&(trueX.GetBlock(0)));
   u_gf.Distribute(&(trueX.GetBlock(1)));

   double H1_error = p_gf.ComputeH1Error(&p_ex,&gradp_ex);
   double H1_norm  = p_zero.ComputeH1Error(&p_ex,&gradp_ex);
   double Hdiv_error = u_gf.ComputeHDivError(&u_ex,&divu_ex);
   double Hdiv_norm = u_zero.ComputeHDivError(&u_ex,&divu_ex);


   if (myid == 0)
   {
      cout << "H1 rel error     = " << H1_error/H1_norm << endl;
      cout << "H(div) rel error = " << Hdiv_error/Hdiv_norm << endl;
   }

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
   return divu + omega * p;
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
   u *= 1./omega;
}

double divu_exact(const Vector &x)
{
   return d2_exact(x)/omega;
}

double d2_exact(const Vector &x)
{
   return -dim * omega * omega * sin(omega*x.Sum());
}