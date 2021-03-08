
#include "mfem.hpp"
#include <fstream>
#include <iostream>
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

class LORH1HdivDirectSolver : Solver
{
public:
   LORH1HdivDirectSolver(HypreParMatrix & A, const Array<int> p_)
   : Solver(A.Height()), p(p_), bp(p.Size()), xp(p.Size()) 
{
   solv = new MUMPSSolver;
   solv->SetOperator(A);

   n2 = p.Size();
   n1 = A.Height() - n2;
   perm.SetSize(A.Height());
   for (int i = 0; i<n1; i++) { perm[i] = i; }
   for (int i = 0; i<n2; i++) { perm[i+n1] = n1 + p[i]; }


}
void SetOperator(const Operator&) { }

void Mult(const Vector &b, Vector &x) const
{
   for (int i=0; i<b.Size(); ++i) { bp[i] = perm[i] < 0 ? -b[-1-perm[i]] : b[perm[i]]; }
   solv->Mult(bp, xp);
   for (int i=0; i<x.Size(); ++i)
   {
      int pi = perm[i];
      int s = pi < 0 ? -1 : 1;
      x[pi < 0 ? -1-pi : pi] = s*xp[i];
   }
}

private:
   int n1;
   int n2;
   Array<int> perm;
   Array<int> p;
   mutable Vector bp, xp;
   MUMPSSolver *solv=nullptr;
};



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

   int btype = BasisType::GaussLobatto;
   ParMesh pmesh_lor(pmesh, order, btype);

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
#ifdef DEFINITE
   ConstantCoefficient negone(-1.0);
   ProductCoefficient neg_f(negone,f_rhs);
   b_v.AddDomainIntegrator(new VectorFEDomainLFDivIntegrator(neg_f));
#else    
   b_v.AddDomainIntegrator(new VectorFEDomainLFDivIntegrator(f_rhs));
#endif

   ParBilinearForm a_qp(H1fespace);
   ConstantCoefficient one(1.0);
   ConstantCoefficient negomeg(-omega);
   ConstantCoefficient omeg2(omega*omega);
   // (grad p, grad q) + \omega^2 (p,q)
   a_qp.AddDomainIntegrator(new DiffusionIntegrator(one));
   a_qp.AddDomainIntegrator(new MassIntegrator(omeg2));

   ParMixedBilinearForm a_qu(RTfespace, H1fespace);
#ifdef DEFINITE
   // -w(divu,q)
   a_qu.AddDomainIntegrator(new MixedScalarDivergenceIntegrator(negomeg));
#else   
   // w(divu,q)
   a_qu.AddDomainIntegrator(new MixedScalarDivergenceIntegrator(omeg));
#endif
   // -w(u, gradq)
   a_qu.AddDomainIntegrator(new MixedVectorWeakDivergenceIntegrator(omeg));
   // w(p,divv) - w(gradp,v)
   ParMixedBilinearForm a_vp(H1fespace, RTfespace);
#ifdef DEFINITE
   // -w(p,divv)
   a_vp.AddDomainIntegrator(new MixedScalarWeakGradientIntegrator(omeg));
#else
   // w(p,divv)
   a_vp.AddDomainIntegrator(new MixedScalarWeakGradientIntegrator(negomeg));
#endif
   // - w(gradp,v)
   a_vp.AddDomainIntegrator(new MixedVectorGradientIntegrator(negomeg));

   ParBilinearForm a_vu(RTfespace);
   a_vu.AddDomainIntegrator(new DivDivIntegrator(one));
   a_vu.AddDomainIntegrator(new VectorFEMassIntegrator(omeg2));


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
      // M.SetOperator(*A);
      M.SetDiagonalBlock(0,&amg_p);
      ScaledOperator S(prec,1.0);
      M.SetDiagonalBlock(1,&S);

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
      Vector trueY(trueX.Size());
      mumps.Mult(trueRhs,trueY);
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