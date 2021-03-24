#include "FOSLS.hpp"


ComplexMaxwellFOSLS::ComplexMaxwellFOSLS(ParFiniteElementSpace * fes_) : fes(fes_)
{ };

void ComplexMaxwellFOSLS::SetLoadData(Array<VectorFunctionCoefficient *> & loads_)
{
   loads = loads_;
}
void ComplexMaxwellFOSLS::SetEssentialData(Array<VectorFunctionCoefficient *> &  ess_data_)
{
   ess_data = ess_data_;
}

void ComplexMaxwellFOSLS::GetFOSLSLinearSystem(Array2D<HypreParMatrix *> & A_, 
                                               BlockVector & X_,
                                               BlockVector & Rhs_)
{
   if (A.NumCols() == 0)
   {
      FormSystem(true);
   }
   A_ = A;
   X_ = X;
   Rhs_ = Rhs;
}

void ComplexMaxwellFOSLS::GetFOSLSMatrix(Array2D<HypreParMatrix *> & A_)
{
   if (A.NumCols() == 0)
   {
      FormSystem(false);
   }
   A_ = A;
}

void ComplexMaxwellFOSLS::FormSystem(bool system)
{
   HYPRE_Int size = fes->GlobalTrueVSize();

   Array<int> ess_tdof_list;
   Array<int> ess_bdr;
   pmesh = fes->GetParMesh();
   if (pmesh->bdr_attributes.Size())
   {
      ess_bdr.SetSize(pmesh->bdr_attributes.Max());
      ess_bdr = 1;
      fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   VectorFunctionCoefficient * E_ex_re = ess_data[0];
   VectorFunctionCoefficient * H_ex_re = ess_data[1];
   VectorFunctionCoefficient * E_ex_im = ess_data[2];
   VectorFunctionCoefficient * H_ex_im = ess_data[3];

   VectorFunctionCoefficient * f_ex_re = loads[0];
   VectorFunctionCoefficient * g_ex_re = loads[1];
   VectorFunctionCoefficient * f_ex_im = loads[2];
   VectorFunctionCoefficient * g_ex_im = loads[3];

   int n = fes->GetVSize();
   int N = fes->GetTrueVSize();
   block_offsets.SetSize(5);
   block_offsets[0] = 0;
   block_offsets[1] = n;
   block_offsets[2] = n;
   block_offsets[3] = n;
   block_offsets[4] = n;
   block_offsets.PartialSum();

   block_trueOffsets.SetSize(5);
   block_trueOffsets[0] = 0;
   block_trueOffsets[1] = N;
   block_trueOffsets[2] = N;
   block_trueOffsets[3] = N;
   block_trueOffsets[4] = N;
   block_trueOffsets.PartialSum();

   ParGridFunction E_gf_re, E_gf_im, H_gf_re, H_gf_im;

   if(system)
   {
      x.Update(block_offsets);
      rhs.Update(block_offsets);
      X.Update(block_trueOffsets);
      Rhs.Update(block_trueOffsets);
      x = 0.0;  rhs = 0.0; X = 0.0;  Rhs = 0.0;

      E_gf_re.MakeRef(fes,x.GetBlock(0)); E_gf_re = 0.0;
      H_gf_re.MakeRef(fes,x.GetBlock(1)); H_gf_re = 0.0;
      E_gf_im.MakeRef(fes,x.GetBlock(2)); E_gf_im = 0.0;
      H_gf_im.MakeRef(fes,x.GetBlock(3)); H_gf_im = 0.0;

      E_gf_re.ProjectCoefficient(*E_ex_re);
      E_gf_im.ProjectCoefficient(*E_ex_im);
   }

   ConstantCoefficient one(1.0);
   ConstantCoefficient negone(-1.0);
   ConstantCoefficient negomeg(-omega);
   ConstantCoefficient omeg(omega);
   ConstantCoefficient omeg2(omega * omega);
   ScalarVectorProductCoefficient wJi(omeg,*g_ex_im);
   ScalarVectorProductCoefficient negJr(negone,*g_ex_re);
   ScalarVectorProductCoefficient negwJr(negomeg,*g_ex_re);
   ScalarVectorProductCoefficient negJi(negone,*g_ex_im);

   ParLinearForm b0, b1, b2, b3;

   if(system)
   {
      b0.Update(fes,rhs.GetBlock(0),0);
      b1.Update(fes,rhs.GetBlock(1),0);
      b2.Update(fes,rhs.GetBlock(2),0);
      b3.Update(fes,rhs.GetBlock(3),0);
      b0.AddDomainIntegrator(new VectorFEDomainLFIntegrator(wJi));
      b1.AddDomainIntegrator(new VectorFEDomainLFCurlIntegrator(negJr));
      b2.AddDomainIntegrator(new VectorFEDomainLFIntegrator(negwJr));
      b3.AddDomainIntegrator(new VectorFEDomainLFCurlIntegrator(negJi));
      b0.Assemble();
      b1.Assemble();
      b2.Assemble();
      b3.Assemble();
   }
   A.SetSize(4,4); 
   for (int i = 0; i<4; i++)
   {
      for (int j = 0; j<4; j++)
      {
         A[i][j] = nullptr;
      }
   }

   ParBilinearForm a00(fes);
   a00.AddDomainIntegrator(new CurlCurlIntegrator(one));
   a00.AddDomainIntegrator(new VectorFEMassIntegrator(omeg2));
   a00.Assemble();
   if (system)
   {
      a00.EliminateEssentialBC(ess_bdr,x.GetBlock(0),rhs.GetBlock(0),mfem::Operator::DIAG_ONE);
   }
   else
   {
      a00.EliminateEssentialBC(ess_bdr);
   }
   a00.Finalize();
   A[0][0] = a00.ParallelAssemble();

   ParMixedBilinearForm a03(fes,fes);
   a03.AddDomainIntegrator(new MixedVectorCurlIntegrator(negomeg));
   a03.AddDomainIntegrator(new MixedVectorWeakCurlIntegrator(negomeg));
   a03.Assemble();
   a03.EliminateTestDofs(ess_bdr);
   a03.Finalize();
   A[0][3] = a03.ParallelAssemble();

   ParBilinearForm a11(fes);
   a11.AddDomainIntegrator(new CurlCurlIntegrator(one));
   a11.AddDomainIntegrator(new VectorFEMassIntegrator(omeg2));
   a11.Assemble();
   a11.Finalize();
   A[1][1] = a11.ParallelAssemble();

   ParMixedBilinearForm a21(fes,fes);
   a21.AddDomainIntegrator(new MixedVectorCurlIntegrator(omeg));
   a21.AddDomainIntegrator(new MixedVectorWeakCurlIntegrator(omeg));
   a21.Assemble();
   a21.EliminateTestDofs(ess_bdr);
   a21.Finalize();
   A[2][1] = a21.ParallelAssemble();

   if (system)
   {
      ParMixedBilinearForm a12(fes,fes);
      a12.AddDomainIntegrator(new MixedVectorCurlIntegrator(omeg));
      a12.AddDomainIntegrator(new MixedVectorWeakCurlIntegrator(omeg));
      a12.Assemble();
      a12.EliminateTrialDofs(ess_bdr,x.GetBlock(2),rhs.GetBlock(1));
      a12.Finalize();
      A[1][2] = a12.ParallelAssemble();
   }
   else
   {
      A[1][2] = A[2][1]->Transpose();
   }


   ParBilinearForm a22(fes);
   a22.AddDomainIntegrator(new CurlCurlIntegrator(one));
   a22.AddDomainIntegrator(new VectorFEMassIntegrator(omeg2));
   a22.Assemble();
   if (system)
   {
      a22.EliminateEssentialBC(ess_bdr,x.GetBlock(2),rhs.GetBlock(2),mfem::Operator::DIAG_ONE);
   }
   else
   {
      a22.EliminateEssentialBC(ess_bdr);
   }
   a22.Finalize();
   A[2][2] = a22.ParallelAssemble();

   if (system)
   {
      ParMixedBilinearForm a30(fes,fes);
      a30.AddDomainIntegrator(new MixedVectorCurlIntegrator(negomeg));
      a30.AddDomainIntegrator(new MixedVectorWeakCurlIntegrator(negomeg));
      a30.Assemble();
      a30.EliminateTrialDofs(ess_bdr,x.GetBlock(0),rhs.GetBlock(3));
      a30.Finalize();
      A[3][0] = a30.ParallelAssemble();
   }
   else
   {
      A[3][0] = A[0][3]->Transpose();
   }

   ParBilinearForm a33(fes);
   a33.AddDomainIntegrator(new CurlCurlIntegrator(one));
   a33.AddDomainIntegrator(new VectorFEMassIntegrator(omeg2));
   a33.Assemble();
   a33.Finalize();
   A[3][3] = a33.ParallelAssemble();

   if (system)
   {
      for (int i = 0; i<4; i++)
      {
         fes->GetRestrictionMatrix()->Mult(x.GetBlock(i), X.GetBlock(i));
         fes->GetProlongationMatrix()->MultTranspose(rhs.GetBlock(i),Rhs.GetBlock(i));
      }
   }
};

HelmholtzFOSLS::HelmholtzFOSLS(Array<ParFiniteElementSpace * > & fes_, 
bool definite_, bool complex_) : fes(fes_), definite(definite_), complex(complex_)
{ 
   n = complex ? 2 : 1;
   Init();
};

void HelmholtzFOSLS::Init()
{
   f.SetSize(n);
   Q.SetSize(n);
   p_ex_coeff.SetSize(n);
   u_ex_coeff.SetSize(n);
   for (int i = 0; i<n; i++)
   {
      f[i] = nullptr;
      Q[i] = nullptr;
      p_ex_coeff[i] = nullptr;
      u_ex_coeff[i] = nullptr;
   }
}

void HelmholtzFOSLS::SetLoadData(Array<FunctionCoefficient * > & f_)
{
   f = f_;
}
void HelmholtzFOSLS::SetLoadData(Array<VectorFunctionCoefficient * > & Q_)
{
   Q = Q_;
}

void HelmholtzFOSLS::SetEssentialData(Array<FunctionCoefficient * > & p_ex_coeff_)
{
   p_ex_coeff = p_ex_coeff_;
}
void HelmholtzFOSLS::SetEssentialData(Array<VectorFunctionCoefficient * > & u_ex_coeff_)
{
   u_ex_coeff = u_ex_coeff_;
}

void HelmholtzFOSLS::GetFOSLSLinearSystem(Array2D<HypreParMatrix *> & A_, 
                                          BlockVector & X_,
                                          BlockVector & Rhs_)
{
   if (A.NumCols() == 0)
   {
      FormSystem(true);
   }
   A_ = A;
   X_ = X;
   Rhs_ = Rhs;
}

void HelmholtzFOSLS::GetFOSLSMatrix(Array2D<HypreParMatrix *> & A_)
{
   if (A.NumCols() == 0)
   {
      FormSystem(false);
   }
   A_ = A;
}

void HelmholtzFOSLS::FormSystem(bool system)
{
   HYPRE_Int size = fes[0]->GlobalTrueVSize() + fes[1]->GlobalTrueVSize();

   Array<int> ess_tdof_list;
   Array<int> ess_bdr;
   pmesh = fes[0]->GetParMesh();
   if (pmesh->bdr_attributes.Size())
   {
      ess_bdr.SetSize(pmesh->bdr_attributes.Max());
      ess_bdr = 1;
   }
   int blksize = complex ? 5 : 3;
   block_offsets.SetSize(blksize);
   block_offsets[0] = 0;
   block_offsets[1] = fes[0]->GetVSize();
   block_offsets[2] = fes[1]->GetVSize();
   if (complex)
   {
      block_offsets[3] = fes[0]->GetVSize();
      block_offsets[4] = fes[1]->GetVSize();
   }
   block_offsets.PartialSum();

   block_trueOffsets.SetSize(blksize);
   block_trueOffsets[0] = 0;
   block_trueOffsets[1] = fes[0]->GetTrueVSize();
   block_trueOffsets[2] = fes[1]->GetTrueVSize();
   if (complex)
   {
      block_trueOffsets[3] = fes[0]->GetTrueVSize();
      block_trueOffsets[4] = fes[1]->GetTrueVSize();
   }
   block_trueOffsets.PartialSum();

   ParGridFunction p_gf_re, u_gf_re, p_gf_im, u_gf_im;

   if(system)
   {
      x.Update(block_offsets);
      rhs.Update(block_offsets);
      X.Update(block_trueOffsets);
      Rhs.Update(block_trueOffsets);
      x = 0.0;  rhs = 0.0; X = 0.0;  Rhs = 0.0;

      p_gf_re.MakeRef(fes[0],x.GetBlock(0)); p_gf_re = 0.0;
      u_gf_re.MakeRef(fes[1],x.GetBlock(1)); u_gf_re = 0.0;

      if (complex)
      {
         p_gf_im.MakeRef(fes[0],x.GetBlock(2)); p_gf_im = 0.0;
         u_gf_im.MakeRef(fes[1],x.GetBlock(3)); u_gf_im = 0.0;
      }

      if (p_ex_coeff[0])
      {
         p_gf_re.ProjectCoefficient(*p_ex_coeff[0]);
      }
      if (u_ex_coeff[0])
      {
         u_gf_re.ProjectCoefficient(*u_ex_coeff[0]);
      }
      if (complex)
      {
         if (p_ex_coeff[1])
         {
            p_gf_im.ProjectCoefficient(*p_ex_coeff[1]);
         }
         if (u_ex_coeff[1])
         {
            u_gf_im.ProjectCoefficient(*u_ex_coeff[1]);
         }
      }
   }

   ConstantCoefficient negone(-1.0);
   ConstantCoefficient one(1.0);
   ConstantCoefficient negomeg(-omega);
   ConstantCoefficient omeg(omega);
   ConstantCoefficient omeg2(omega * omega);
   ProductCoefficient omega_f(omeg,*f[0]);
   ProductCoefficient neg_f(negone,*f[0]);

   ParLinearForm b0, b1;

   if(system)
   {
      b0.Update(fes[0],rhs.GetBlock(0),0);
      b1.Update(fes[1],rhs.GetBlock(1),0);
      b0.AddDomainIntegrator(new DomainLFIntegrator(omega_f));
      if (definite)
      {
         b1.AddDomainIntegrator(new VectorFEDomainLFDivIntegrator(neg_f));
      }
      else
      {
         b1.AddDomainIntegrator(new VectorFEDomainLFDivIntegrator(*f[0]));
      }
      b0.Assemble();
      b1.Assemble();
   }
   A.SetSize(2,2); 
   for (int i = 0; i<2; i++)
   {
      for (int j = 0; j<2; j++)
      {
         A[i][j] = nullptr;
      }
   }

   ParBilinearForm a00(fes[0]);
   a00.AddDomainIntegrator(new DiffusionIntegrator(one));
   a00.AddDomainIntegrator(new MassIntegrator(omeg2));
   a00.Assemble();
   if (system)
   {
      a00.EliminateEssentialBC(ess_bdr,x.GetBlock(0),rhs.GetBlock(0),mfem::Operator::DIAG_ONE);
   }
   else
   {
      a00.EliminateEssentialBC(ess_bdr,mfem::Operator::DIAG_ONE);
   }
   a00.Finalize();
   A[0][0] = a00.ParallelAssemble();

   ParMixedBilinearForm a01(fes[1],fes[0]);
   if (definite)
   {
      a01.AddDomainIntegrator(new MixedScalarDivergenceIntegrator(negomeg));
   }
   else
   {
      a01.AddDomainIntegrator(new MixedScalarDivergenceIntegrator(omeg));
   }
   a01.AddDomainIntegrator(new MixedVectorWeakDivergenceIntegrator(omeg));
   a01.Assemble();
   a01.EliminateTestDofs(ess_bdr);
   a01.Finalize();
   A[0][1] = a01.ParallelAssemble();

   if (system)
   {
      ParMixedBilinearForm a10(fes[0],fes[1]);
      if (definite)
      {
         a10.AddDomainIntegrator(new MixedScalarWeakGradientIntegrator(omeg));
      }
      else
      {
         a10.AddDomainIntegrator(new MixedScalarWeakGradientIntegrator(negomeg));
      }
      a10.AddDomainIntegrator(new MixedVectorGradientIntegrator(negomeg));
      a10.Assemble();
      a10.EliminateTrialDofs(ess_bdr,x.GetBlock(0),rhs.GetBlock(1));
      a10.Finalize();
      A[1][0] = a10.ParallelAssemble();
   }
   else
   {
      A[1][0] = A[0][1]->Transpose();
   }

   ParBilinearForm a11(fes[1]);
   a11.AddDomainIntegrator(new DivDivIntegrator(one));
   a11.AddDomainIntegrator(new VectorFEMassIntegrator(omeg2));
   a11.Assemble();
   a11.Finalize();
   A[1][1] = a11.ParallelAssemble();


   if (system)
   {
      fes[0]->GetRestrictionMatrix()->Mult(x.GetBlock(0), X.GetBlock(0));
      fes[1]->GetRestrictionMatrix()->Mult(x.GetBlock(1), X.GetBlock(1));
      fes[0]->GetProlongationMatrix()->MultTranspose(rhs.GetBlock(0),Rhs.GetBlock(0));
      fes[1]->GetProlongationMatrix()->MultTranspose(rhs.GetBlock(1),Rhs.GetBlock(1));
   }
}