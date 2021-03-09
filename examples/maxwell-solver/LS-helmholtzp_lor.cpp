
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
   // FiniteElementCollection *H1fec = new H1_FECollection(order,dim); 
   // ParFiniteElementSpace *H1fespace = new ParFiniteElementSpace(pmesh, H1fec);

   // FiniteElementCollection *RTfec = new RT_FECollection(order,dim); 
   // ParFiniteElementSpace *RTfespace = new ParFiniteElementSpace(pmesh, RTfec);

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

   // -------------------------------------------------------------------
   // |   |            p             |           u           |   RHS    | 
   // -------------------------------------------------------------------
   // | q | (gradp,gradq) + w^2(p,q) | w(divu,q)-w(u, gradq) |  w(f,q)  |
   // |   |                          |                       |          |
   // | v | w(p,divv) - w(gradp,v)   | (divu,divv) + w^2(u,v)| (f,divv) |


   // omega(f,q) 
   ParLinearForm b_q_ho(&H1fes_ho);
   ConstantCoefficient omeg(omega);
   FunctionCoefficient f_rhs(rhs_func);
   ProductCoefficient omega_f(omeg,f_rhs);
   b_q_ho.AddDomainIntegrator(new DomainLFIntegrator(omega_f));
   // (f, div v)
   ParLinearForm b_v_ho(&RTfes_ho);
   ParLinearForm b_v_lor(&RTfes_lor);
#ifdef DEFINITE
   ConstantCoefficient negone(-1.0);
   ProductCoefficient neg_f(negone,f_rhs);
   b_v_ho.AddDomainIntegrator(new VectorFEDomainLFDivIntegrator(neg_f));
#else    
   b_v_ho.AddDomainIntegrator(new VectorFEDomainLFDivIntegrator(f_rhs));
#endif

   ParBilinearForm a_qp_ho(&H1fes_ho);
   ParBilinearForm a_qp_lor(&H1fes_lor);
   ConstantCoefficient one(1.0);
   ConstantCoefficient negomeg(-omega);
   ConstantCoefficient omeg2(omega*omega);
   // (grad p, grad q) + \omega^2 (p,q)
   a_qp_ho.AddDomainIntegrator(new DiffusionIntegrator(one));
   a_qp_ho.AddDomainIntegrator(new MassIntegrator(omeg2));

   a_qp_lor.AddDomainIntegrator(new DiffusionIntegrator(one));
   a_qp_lor.AddDomainIntegrator(new MassIntegrator(omeg2));

   ParMixedBilinearForm a_qu_ho(&RTfes_ho, &H1fes_ho);
   ParMixedBilinearForm a_qu_lor(&RTfes_lor, &H1fes_lor);
#ifdef DEFINITE
   // -w(divu,q)
   a_qu_ho.AddDomainIntegrator(new MixedScalarDivergenceIntegrator(negomeg));
   a_qu_lor.AddDomainIntegrator(new MixedScalarDivergenceIntegrator(negomeg));
#else   
   // w(divu,q)
   a_qu_ho.AddDomainIntegrator(new MixedScalarDivergenceIntegrator(omeg));
   a_qu_lor.AddDomainIntegrator(new MixedScalarDivergenceIntegrator(omeg));
#endif
   // -w(u, gradq)
   a_qu_ho.AddDomainIntegrator(new MixedVectorWeakDivergenceIntegrator(omeg));
   a_qu_lor.AddDomainIntegrator(new MixedVectorWeakDivergenceIntegrator(omeg));
   // w(p,divv) - w(gradp,v)
   ParMixedBilinearForm a_vp_ho(&H1fes_ho, &RTfes_ho);
   ParMixedBilinearForm a_vp_lor(&H1fes_lor, &RTfes_lor);
#ifdef DEFINITE
   // -w(p,divv)
   a_vp_ho.AddDomainIntegrator(new MixedScalarWeakGradientIntegrator(omeg));
   a_vp_lor.AddDomainIntegrator(new MixedScalarWeakGradientIntegrator(omeg));
#else
   // w(p,divv)
   a_vp_ho.AddDomainIntegrator(new MixedScalarWeakGradientIntegrator(negomeg));
   a_vp_lor.AddDomainIntegrator(new MixedScalarWeakGradientIntegrator(negomeg));
#endif
   // - w(gradp,v)
   a_vp_ho.AddDomainIntegrator(new MixedVectorGradientIntegrator(negomeg));
   a_vp_lor.AddDomainIntegrator(new MixedVectorGradientIntegrator(negomeg));

   ParBilinearForm a_vu_ho(&RTfes_ho);
   ParBilinearForm a_vu_lor(&RTfes_lor);
   a_vu_ho.AddDomainIntegrator(new DivDivIntegrator(one));
   a_vu_ho.AddDomainIntegrator(new VectorFEMassIntegrator(omeg2));

   a_vu_lor.AddDomainIntegrator(new DivDivIntegrator(one));
   a_vu_lor.AddDomainIntegrator(new VectorFEMassIntegrator(omeg2));


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
         H1fes_ho.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      }

       Array<int> block_offsets(3);
      block_offsets[0] = 0;
      block_offsets[1] = H1fes_ho.GetVSize();
      block_offsets[2] = RTfes_ho.GetVSize();
      block_offsets.PartialSum();

      Array<int> block_trueOffsets(3);
      block_trueOffsets[0] = 0;
      block_trueOffsets[1] = H1fes_ho.TrueVSize();
      block_trueOffsets[2] = RTfes_ho.TrueVSize();
      block_trueOffsets.PartialSum();

      BlockVector x(block_offsets), rhs(block_offsets);
      BlockVector trueX(block_trueOffsets), trueRhs(block_trueOffsets);
      x = 0.0;  rhs = 0.0;
      trueX = 0.0;  trueRhs = 0.0;

      p_gf.MakeRef(&H1fes_ho,x.GetBlock(0));
      p_gf.ProjectBdrCoefficient(p_ex,ess_bdr);
      u_gf.MakeRef(&RTfes_ho,x.GetBlock(1));
      u_gf = 0.0;

      b_q_ho.Update(&H1fes_ho,rhs.GetBlock(0),0);
      b_q_ho.Assemble();
      b_v_ho.Update(&RTfes_ho,rhs.GetBlock(1),0);
      b_v_ho.Assemble();

      a_qp_ho.Assemble();
      a_qp_ho.EliminateEssentialBC(ess_bdr,x.GetBlock(0),rhs.GetBlock(0));
      a_qp_ho.Finalize();
      HypreParMatrix * A_qp_ho = a_qp_ho.ParallelAssemble();

      a_qp_lor.Assemble();
      HypreParMatrix A_qp_lor; 
      a_qp_lor.FormSystemMatrix(ess_tdof_list,A_qp_lor);

      a_qu_ho.Assemble();
      a_qu_ho.EliminateTestDofs(ess_bdr);
      a_qu_ho.Finalize();
      HypreParMatrix * A_qu_ho = a_qu_ho.ParallelAssemble();

      a_qu_lor.Assemble();
      a_qu_lor.EliminateTestDofs(ess_bdr);
      a_qu_lor.Finalize();
      HypreParMatrix * A_qu_lor = a_qu_lor.ParallelAssemble();

      a_vp_ho.Assemble();
      a_vp_ho.EliminateTrialDofs(ess_bdr,x.GetBlock(0),rhs.GetBlock(1));
      a_vp_ho.Finalize();
      HypreParMatrix * A_vp_ho = a_vp_ho.ParallelAssemble();

      // a_vp_lor.Assemble();
      // Array<int> temp_list;
      // // a_vp_lor.FormRectangularSystemMatrix()
      // // a_vp_lor.EliminateTrialDofs(ess_bdr,x_lor.GetBlock(0),rhs_lor.GetBlock(1));
      // a_vp_lor.Finalize();
      HypreParMatrix * A_vp_lor = A_qu_lor->Transpose();

      a_vu_ho.Assemble();
      a_vu_ho.Finalize();
      HypreParMatrix * A_vu_ho = a_vu_ho.ParallelAssemble();

      a_vu_lor.Assemble();
      a_vu_lor.Finalize();
      HypreParMatrix * A_vu_lor = a_vu_lor.ParallelAssemble();


      H1fes_ho.GetRestrictionMatrix()->Mult(x.GetBlock(0), trueX.GetBlock(0));
      H1fes_ho.GetProlongationMatrix()->MultTranspose(rhs.GetBlock(0),trueRhs.GetBlock(0));

      RTfes_ho.GetRestrictionMatrix()->Mult(x.GetBlock(1), trueX.GetBlock(1));
      RTfes_ho.GetProlongationMatrix()->MultTranspose(rhs.GetBlock(1),trueRhs.GetBlock(1));


      Vector trueY(trueX);
      Vector trueZ(trueX);


      Array2D<HypreParMatrix *> Ah_ho(2,2);
      Ah_ho[0][0] = A_qp_ho; 
      Ah_ho[0][1] = A_qu_ho;
      Ah_ho[1][0] = A_vp_ho;
      Ah_ho[1][1] = A_vu_ho;
      HypreParMatrix * A_ho = HypreParMatrixFromBlocks(Ah_ho);

      Array2D<HypreParMatrix *> Ah_lor(2,2);
      Ah_lor[0][0] = &A_qp_lor; 
      Ah_lor[0][1] = A_qu_lor;
      Ah_lor[1][0] = A_vp_lor;
      Ah_lor[1][1] = A_vu_lor;
      HypreParMatrix * A_lor = HypreParMatrixFromBlocks(Ah_lor);


      HypreBoomerAMG amg_p(*A_qp_ho);
      amg_p.SetPrintLevel(0);
      HypreBoomerAMG amg_lor_p(A_qp_lor);
      amg_lor_p.SetPrintLevel(0);

      Solver *prec = nullptr;
      Solver *prec_lor = nullptr;
      if (dim == 2) 
      {
         prec = new HypreAMS(*A_vu_ho,&RTfes_ho);
         dynamic_cast<HypreAMS *>(prec)->SetPrintLevel(0);
         prec_lor = new HypreAMS(*A_vu_lor,&RTfes_lor);
         dynamic_cast<HypreAMS *>(prec_lor)->SetPrintLevel(0);
      }
      else
      {
         prec = new HypreADS(*A_vu_ho,&RTfes_ho);
         dynamic_cast<HypreADS *>(prec)->SetPrintLevel(0);
         prec_lor = new HypreADS(*A_vu_lor,&RTfes_lor);
         dynamic_cast<HypreADS *>(prec_lor)->SetPrintLevel(0);
      }

      BlockDiagonalPreconditioner M(block_trueOffsets);
      BlockDiagonalPreconditioner M_lor2(block_trueOffsets);

      FiniteElement::MapType t = FiniteElement::H_DIV;
      Array<int> perm = ComputeVectorFE_LORPermutation(RTfes_ho, RTfes_lor, t);
      

      LORH1HdivDirectSolver M_lor(*A_lor, perm);
      // for (int i = 0; i<perm.Size(); i++) perm[i] = i;
      // LORH1HdivDirectSolver M_lor(*A_ho, perm);

      // BlockDiagonalMultiplicativePreconditioner M(block_trueOffsets);
      // M.SetOperator(*A);
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

      // chrono.Clear();
      // chrono.Start();
      // MUMPSSolver mumps;
      // mumps.SetPrintLevel(0);
      // mumps.SetMatrixSymType(MUMPSSolver::MatType::UNSYMMETRIC);
      // mumps.SetOperator(*A_ho);
      // Vector trueY(trueX.Size());
      // mumps.Mult(trueRhs,trueY);
      // chrono.Stop();
      // cout << "MUMPS time " << chrono.RealTime() << endl;


      delete A_ho;
      delete A_vu_ho;
      delete A_qp_ho;
      delete A_vp_ho;
      delete A_qu_ho;

      p_gf = 0.0;
      u_gf = 0.0;
      p_gf.Distribute(&(trueX.GetBlock(0)));
      u_gf.Distribute(&(trueX.GetBlock(1)));

      ratesH1.AddH1GridFunction(&p_gf,&p_ex,&gradp_ex);
      ratesRT.AddHdivGridFunction(&u_gf,&u_ex,&divu_ex);

      if (l==pr) break;

      pmesh->UniformRefinement();
      pmesh_lor.UniformRefinement();
      H1fes_ho.Update();
      RTfes_ho.Update();
      a_qp_ho.Update();
      a_qu_ho.Update();
      a_vp_ho.Update();
      a_vu_ho.Update();
      b_q_ho.Update();
      b_v_ho.Update();
      H1fes_lor.Update();
      RTfes_lor.Update();
      a_qp_lor.Update();
      a_qu_lor.Update();
      a_vp_lor.Update();
      a_vu_lor.Update();
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