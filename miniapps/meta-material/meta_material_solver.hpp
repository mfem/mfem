// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#ifndef MFEM_META_MATERIAL_SOLVER
#define MFEM_META_MATERIAL_SOLVER

#include "../../config/config.hpp"

#ifdef MFEM_USE_MPI

#include "../common/pfem_extras.hpp"
#include "../common/bravais.hpp"

extern "C"
{
   void dsygv_(int *ITYPE, char *JOBZ, char *UPLO, int *N,
               double *A, int * LDA, double *B, int *LDB, double *W,
               double *WORK, int *LWORK, int *INFO);
}

namespace mfem
{

using common::H1_ParFESpace;
using common::ND_ParFESpace;
using common::RT_ParFESpace;
using common::L2_ParFESpace;
using common::ParDiscreteGradOperator;
using common::ParDiscreteCurlOperator;
using common::ParDiscreteInterpolationOperator;
using common::VisData;
using bravais::BravaisLattice;

namespace meta_material
{

class LinearCoefficient : public GridFunctionCoefficient
{
public:
   LinearCoefficient(GridFunction * gf, double a0, double a1);

   virtual double Eval(ElementTransformation &T,
                       const IntegrationPoint &ip);

   double GetSensitivity(ElementTransformation &T,
                         const IntegrationPoint &ip)
   { return (c1_ - c0_); }

private:
   double c0_;
   double c1_;
};
/*
class PenaltyCoefficient : public GridFunctionCoefficient
{
public:
   PenaltyCoefficient(GridFunction * gf, int penalty, double a0, double a1);

   virtual double Eval(ElementTransformation &T,
                       const IntegrationPoint &ip);

   double GetSensitivity(ElementTransformation &T,
                         const IntegrationPoint &ip);

private:
   int penalty_;
   double c0_;
   double c1_;
};
*/
class Homogenization
{
public:
   Homogenization(MPI_Comm comm);
   virtual ~Homogenization() {}

   void SetVolumeFraction(ParGridFunction & vf) { vf_ = &vf; newVF_ = true; }

   virtual void GetHomogenizedProperties(std::vector<double> & p) = 0;
   virtual void GetPropertySensitivities(std::vector<ParGridFunction> & dp)
      = 0;

   void InitializeGLVis(VisData & vd) {}

   void DisplayToGLVis() {}

   void WriteVisItFields(const std::string & prefix,
                         const std::string & label) {}

protected:
   MPI_Comm comm_;
   int      myid_;
   int      numProcs_;
   bool     newVF_;

   ParGridFunction * vf_;
};

class Density : public Homogenization
{
public:
   // Density(ParMesh & pmesh, double vol,
   //       double rho0, double rho1);
   Density(ParMesh & pmesh, double refDensity, double vol,
           Coefficient &rhoCoef, double tol = 0.05);
   ~Density();

   // void SetVolumeFraction(ParGridFunction & vf);

   void GetHomogenizedProperties(std::vector<double> & p);
   void GetPropertySensitivities(std::vector<ParGridFunction> & dp) {}

   void InitializeGLVis(VisData & vd);

   void DisplayToGLVis();

   void WriteVisItFields(const std::string & prefix,
                         const std::string & label);

private:
   void updateRho();

   ParMesh         * pmesh_;
   L2_ParFESpace   * L2FESpace_;
   ParGridFunction * rho_;
   ParGridFunction * divGradRho_;
   ParLinearForm   * cellVol_;

   // LinearCoefficient   rhoCoef_;
   Coefficient * rhoCoef_;
   ConstantCoefficient one_;

   double refDensity_;
   double vol_;
   double tol_;

   VisData      * vd_;
   socketstream * sock_;
   socketstream * sock2_;
};

class StiffnessTensor : public Homogenization
{
public:
   /*
   StiffnessTensor(ParMesh & pmesh, double vol,
                    double lambda0, double mu0,
                    double lambda1, double mu1);
   */
   StiffnessTensor(ParMesh & pmesh, double vol,
                   Coefficient &lambdaCoef, Coefficient &muCoef,
                   double tol = 0.05);
   ~StiffnessTensor();

   // void SetVolumeFraction(ParGridFunction & vf);

   void GetHomogenizedProperties(std::vector<double> & p);
   void GetPropertySensitivities(std::vector<ParGridFunction> & dp) {}

   void InitializeGLVis(VisData & vd);

   void DisplayToGLVis();

   void WriteVisItFields(const std::string & prefix,
                         const std::string & label);

private:

   void TensorGradient(const Vector & x, Vector & y);
   void TensorGradientTranspose(const Vector & x, Vector & y);
   void TensorMassMatrix(const Vector & x, Vector & y);
   void RestrictedTensorMassMatrix(int r, const Vector & x, Vector & y);
   void RestrictedVectorAdd(int r, const Vector & x, Vector & y);

   // Produces a 2-tensor coefficient for the diagonal portion of an
   // elasticity operator.  The coefficient is defined by:
   //     / lambda + 2 * mu, i == j == axis
   // K = |              mu, i == j != axis
   //     \               0, i != j
   class DiagElasticityCoef : public MatrixCoefficient
   {
   public:
      DiagElasticityCoef(Coefficient & lambda,
                         Coefficient & mu,
                         int axis);

      virtual void Eval(DenseMatrix &K, ElementTransformation &T,
                        const IntegrationPoint &ip);

   private:
      int axis_;
      Coefficient * lambda_;
      Coefficient * mu_;
   };

   // Produces a 2-tensor coefficient for the off-diagonal portion of an
   // elasticity operator.  The coefficient is defined by:
   //     / lambda, i == axis0, j == axis1
   // K = |     mu, i == axis1, j == axis0
   //     \      0, otherwise
   class OffDiagElasticityCoef : public MatrixCoefficient
   {
   public:
      OffDiagElasticityCoef(Coefficient & lambda,
                            Coefficient & mu,
                            int axis0, int axis1);

      virtual void Eval(DenseMatrix &K, ElementTransformation &T,
                        const IntegrationPoint &ip);

   private:
      int axis0_;
      int axis1_;
      Coefficient * lambda_;
      Coefficient * mu_;
   };

   void solve(const Vector & E, Vector & Chi);

   int dim_;
   int irOrder_;
   int geom_;
   bool amg_elast_;

   ParMesh * pmesh_;

   L2_ParFESpace * L2FESpace_;
   L2_ParFESpace * L2VFESpace_;
   H1_ParFESpace * H1FESpace_;
   H1_ParFESpace * H1VFESpace_;
   ND_ParFESpace * HCurlFESpace_;
   ND_ParFESpace * HCurlVFESpace_;
   RT_ParFESpace * HDivFESpace_;

   // LinearCoefficient lambdaCoef_;
   // LinearCoefficient muCoef_;
   Coefficient * lambdaCoef_;
   Coefficient * muCoef_;

   DiagElasticityCoef    xxCoef_;
   DiagElasticityCoef    yyCoef_;
   DiagElasticityCoef    zzCoef_;
   OffDiagElasticityCoef yzCoef_;
   OffDiagElasticityCoef xzCoef_;
   OffDiagElasticityCoef xyCoef_;

   ParGridFunction * lambda_;
   ParGridFunction * mu_;

   ParBilinearForm * a_;
   ParBilinearForm * m_[6];
   ParGridFunction * Chi_[6];
   ParGridFunction * E_[3];
   ParGridFunction * F_[6];
   ParLinearForm   * MF_[6];

   ParDiscreteGradOperator * grad_;

   BilinearFormIntegrator * diffInteg_[3];
   ParGridFunction * errSol_;
   ErrorEstimator  * errEst_[3];
   ParGridFunction * errors_[4];

   ParGridFunction * b_;
   // HypreParVector  * tmp1_;

   double vol_;
   double tol_;

   VisData      * vd_;
   socketstream   socks_[8];
   socketstream   err_socks_[4];

   int seqVF_;
};

class ParDiscreteVectorProductOperator
   : public ParDiscreteInterpolationOperator
{
public:
   ParDiscreteVectorProductOperator(ParFiniteElementSpace *dfes,
                                    ParFiniteElementSpace *rfes,
                                    const Vector & v);

private:
   VectorConstantCoefficient vCoef_;
};

class ParDiscreteVectorCrossProductOperator
   : public ParDiscreteInterpolationOperator
{
public:
   ParDiscreteVectorCrossProductOperator(ParFiniteElementSpace *dfes,
                                         ParFiniteElementSpace *rfes,
                                         const Vector & v);
private:
   VectorConstantCoefficient vCoef_;
};

//class MaxwellBlochWaveEquation;
//class MaxwellBlochWaveEquation::MaxwellBlochWaveProjector;

class MaxwellBlochWaveEquation
{
public:
   MaxwellBlochWaveEquation(ParMesh & pmesh,
                            int order);
   ~MaxwellBlochWaveEquation();

   // Where kappa is the phase shift vector
   void SetKappa(const Vector & kappa);

   // Where beta*zeta = kappa
   void SetBeta(double beta);
   void SetZeta(const Vector & zeta);

   // void SetAzimuth(double alpha_a);
   // void SetInclination(double alpha_i);

   // void SetOmega(double omega);
   void SetAbsoluteTolerance(double atol);
   void SetNumEigs(int nev);
   void SetMassCoef(Coefficient & m);
   void SetStiffnessCoef(Coefficient & k);

   void Setup();

   void SetInitialVectors(int num_vecs, HypreParVector ** vecs);

   // void SetBravaisLattice(BravaisLattice & bravais) { bravais_ = &bravais; }

   void Update();

   /// Solve the eigenproblem
   void Solve();

   /// Collect the converged eigenvalues
   void GetEigenvalues(std::vector<double> & eigenvalues);

   /// A convenience method which combines six methods into one
   void GetEigenvalues(int nev, const Vector & kappa,
                       std::vector<HypreParVector*> & init_vecs,
                       std::vector<double> & eigenvalues);

   /// Extract a single eigenvector
   HypreParVector * ReturnEigenvector(unsigned int i);

   void CopyEigenvector(unsigned int i,
                        HypreParVector & V);

   void GetEigenvector(unsigned int i,
                       HypreParVector & Er,
                       HypreParVector & Ei,
                       HypreParVector & Br,
                       HypreParVector & Bi);
   void GetEigenvectorE(unsigned int i,
                        HypreParVector & Er,
                        HypreParVector & Ei);
   void GetEigenvectorB(unsigned int i,
                        HypreParVector & Br,
                        HypreParVector & Bi);

   BlockOperator * GetAOperator() { return A_; }
   BlockOperator * GetMOperator() { return M_; }

   Solver   * GetPreconditioner() { return Precond_; }
   Operator * GetSubSpaceProjector() { return SubSpaceProj_; }

   ParFiniteElementSpace * GetH1FESpace() { return H1FESpace_; }
   ParFiniteElementSpace * GetHCurlFESpace() { return HCurlFESpace_; }
   ParFiniteElementSpace * GetHDivFESpace()  { return HDivFESpace_; }

   // void TestVector(const HypreParVector & v);

   ParGridFunction * GetEigenvectorEnergy(unsigned int i) { return energy_[i]; }

   void GetFourierCoefficients(HypreParVector & Vr,
                               HypreParVector & Vi,
                               Array2D<double> &f);

   void IdentifyDegeneracies(double zero_tol, double rel_tol,
                             std::vector<std::set<int> > & degen);

   void GetFieldAverages(unsigned int i,
                         Vector & Er, Vector & Ei,
                         Vector & Br, Vector & Bi,
                         Vector & Dr, Vector & Di,
                         Vector & Hr, Vector & Hi);

   void ComputeHomogenizedCoefs();

   void DetermineBasis(const Vector & v1, std::vector<Vector> & e);

   void WriteVisitFields(const std::string & prefix,
                         const std::string & label);

   void GetSolverStats(double &meanTime, double &stdDevTime,
                       double &meanIter, double &stdDevIter,
                       int &nSolves);

   void TestProjector() const;

private:

   MPI_Comm comm_;
   int myid_;
   int hcurl_loc_size_;
   int hdiv_loc_size_;
   int nev_;

   bool newBeta_;
   bool newZeta_;
   bool newOmega_;
   bool newMCoef_;
   bool newKCoef_;

   ParMesh        * pmesh_;
   H1_ParFESpace  * H1FESpace_;
   ND_ParFESpace  * HCurlFESpace_;
   RT_ParFESpace  * HDivFESpace_;
   // L2_ParFESpace  * L2FESpace_;

   // BravaisLattice     * bravais_;
   // HCurlFourierSeries * fourierHCurl_;

   double           atol_;
   double           beta_;

   Vector           zeta_;
   Vector           kappa_;

   Coefficient    * mCoef_;
   Coefficient    * kCoef_;

   Array<int>       block_offsets_;
   Array<int>       block_trueOffsets_;
   Array<int>       block_trueOffsets2_;
   Array<HYPRE_Int> tdof_offsets_;

   BlockOperator  * A_;
   BlockOperator  * M_;
   BlockOperator  * C_;

   BlockVector    * blkHCurl_;
   BlockVector    * blkHDiv_;

   HypreParMatrix * M1_;
   HypreParMatrix * M2_;
   HypreParMatrix * S1_;
   HypreParMatrix * T1_;
   HypreParMatrix * T12_;
   HypreParMatrix * Z12_;

   HypreParMatrix * DKZ_;
   // HypreParMatrix * DKZT_;

   HypreAMS       * T1Inv_;

   ParDiscreteCurlOperator * Curl_;
   ParDiscreteVectorCrossProductOperator * Zeta_;

   BlockDiagonalPreconditioner * BDP_;

   Solver   * Precond_;
   //MaxwellBlochWaveProjector * SubSpaceProj_;
   Operator * SubSpaceProj_;

   HypreParVector ** vecs_;
   HypreParVector * vec0_;

   HypreLOBPCG * lobpcg_;
   HypreAME    * ame_;

   ParGridFunction ** energy_;
   /*
    HypreParVector * AvgHCurl_coskx_[3];
    HypreParVector * AvgHCurl_sinkx_[3];
    HypreParVector * AvgHDiv_coskx_[3];
    HypreParVector * AvgHDiv_sinkx_[3];

    HypreParVector * AvgHCurl_eps_coskx_[3];
    HypreParVector * AvgHCurl_eps_sinkx_[3];
    HypreParVector * AvgHDiv_muInv_coskx_[3];
    HypreParVector * AvgHDiv_muInv_sinkx_[3];
   */
   std::vector<double> solve_times_;
   std::vector<int>    solve_iters_;

   class MaxwellBlochWavePrecond : public Solver
   {
   public:
      MaxwellBlochWavePrecond(ParFiniteElementSpace & HCurlFESpace,
                              BlockDiagonalPreconditioner & BDP,
                              Operator & subSpaceProj,
                              double w);

      ~MaxwellBlochWavePrecond();

      void Mult(const Vector & x, Vector & y) const;

      void SetOperator(const Operator & A);

   private:
      int myid_;

      // ParFiniteElementSpace * HCurlFESpace_;
      BlockDiagonalPreconditioner * BDP_;
      const Operator * A_;
      Operator * subSpaceProj_;
      // mutable HypreParVector *r_, *u_, *v_;
      mutable HypreParVector *u_;
      // double w_;
   };

   class MaxwellBlochWaveProjector : public Operator
   {
   public:
      MaxwellBlochWaveProjector(ParFiniteElementSpace & HCurlFESpace,
                                ParFiniteElementSpace & H1FESpace,
                                BlockOperator & M,
                                double beta, const Vector & zeta);
      ~MaxwellBlochWaveProjector();

      void SetBeta(double beta);
      void SetZeta(const Vector & zeta);

      void Setup();
      virtual void Mult(const Vector &x, Vector &y) const;

   private:
      int myid_;
      int locSize_;

      bool newBeta_;
      bool newZeta_;

      ParFiniteElementSpace * HCurlFESpace_;
      ParFiniteElementSpace * H1FESpace_;

      double beta_;
      Vector zeta_;

      HypreParMatrix * T01_;
      HypreParMatrix * Z01_;
      HypreParMatrix * A0_;
      HypreParMatrix * DKZ_;

      MINRESSolver   * minres_;

      Array<int>       block_offsets0_;
      Array<int>       block_offsets1_;
      Array<int>       block_trueOffsets0_;
      Array<int>       block_trueOffsets1_;

      BlockOperator * S0_;
      BlockOperator * M_;
      BlockOperator * G_;

      mutable HypreParVector * urDummy_;
      mutable HypreParVector * uiDummy_;
      mutable HypreParVector * vrDummy_;
      mutable HypreParVector * viDummy_;

      mutable BlockVector * u0_;
      mutable BlockVector * v0_;
      mutable BlockVector * u1_;
      mutable BlockVector * v1_;
   };
};

class InverseCoefficient : public TransformedCoefficient
{
public:
   InverseCoefficient(Coefficient * q) : TransformedCoefficient(q, inv_) {}
private:
   static double inv_(double v) { return 1.0/v; }
};

class MaxwellBlochWaveSolver
{
public:
   MaxwellBlochWaveSolver(ParMesh & pmesh, BravaisLattice & bravais,
                          Coefficient & epsCoef, Coefficient & muCoef,
                          int max_ref = 2, int nev = 24, double tol = 0.05);
   ~MaxwellBlochWaveSolver();

   // Where kappa is the phase shift vector
   void SetKappa(const Vector & kappa);

   // Where beta*zeta = kappa and |zeta| = 1
   void SetBeta(double beta);
   void SetZeta(const Vector & zeta);

   void GetEigenfrequencies(std::vector<double> & omega);

   MaxwellBlochWaveEquation * GetFineSolver()
   { return mbwe_[mbwe_.size()-1]; }

   HypreParVector * ReturnFineEigenvector(int i);

   void InitializeGLVis(VisData & vd);

   void DisplayToGLVis();

   void WriteVisItFields(const std::string & prefix,
                         const std::string & label);

private:

   void createPartitioning(ParFiniteElementSpace & pfes, HYPRE_Int *& part);

   int max_lvl_;
   int nev_;
   double tol_;

   std::vector<ParMesh*> pmesh_;
   std::vector<MaxwellBlochWaveEquation*> mbwe_;
   std::vector<const Operator*> refineOp_;
   std::vector<std::pair<HypreParVector*, HypreParVector*> > EField_;
   //std::vector<std::vector<std::pair<ParGridFunction,
   //                ParGridFunction> > > efield_;
   std::vector<std::pair<ParGridFunction*, ParGridFunction*> > efield_;
   std::vector<std::vector<HypreParVector*> > initialVecs_;
   std::vector<int> locSize_;
   std::vector<HYPRE_Int*> part_;

   Vector kappa_;
   Coefficient * epsCoef_;
   InverseCoefficient muInvCoef_;

   // Coefficient * muCoef_;

};

class MaxwellDispersion
{
public:
   MaxwellDispersion(ParMesh & pmesh, BravaisLattice & bravais,
                     int sample_power,
                     Coefficient & epsCoef, Coefficient & muCoef,
                     bool midPts = false, int max_ref = 2,
                     int nev = 24, double tol = 0.05);
   ~MaxwellDispersion();

   const std::vector<std::vector<std::map<int,std::vector<double> > > > &
   GetDispersionData();

   void PrintDispersionPlot(std::ostream & os);

   void InitializeGLVis(VisData & vd);

   void DisplayToGLVis();

   void WriteVisItFields(const std::string & prefix,
                         const std::string & label);

private:

   void buildRawBasis();

   void approxEigenfrequencies(std::vector<double> & omega);

   void traverseBrillouinZone();

   std::string modLabel(const std::string & label) const;

   void findAndReplace(const std::string & f, const std::string & r,
                       std::string & str) const;

   BravaisLattice         * bravais_;
   MaxwellBlochWaveSolver * mbws_;

   HypreParVector * Ax_;
   HypreParVector * Mx_;

   DenseMatrix A_;
   DenseMatrix M_;

   std::vector<HypreParVector*> rawBasis_;
   std::vector<HypreParVector*> projBasis_;

   std::map<std::string,std::vector<double> > sp_eigs_;
   std::vector<std::vector<std::map<int,std::vector<double> > > > seg_eigs_;

   int n_pow_;
   int n_div_;
   int samp_pow_;
   int nev_;

   bool midPts_;
};

class MaxwellBandGap : public Homogenization
{
public:
   MaxwellBandGap(ParMesh & pmesh, BravaisLattice & bravais,
                  int samp_pow,
                  Coefficient & epsCoef, Coefficient & muCoef,
                  bool midPts = false, int max_ref = 2, double tol = 0.05);
   ~MaxwellBandGap();

   void GetHomogenizedProperties(std::vector<double> & p);
   void GetPropertySensitivities(std::vector<ParGridFunction> & dp) {}

   void PrintDispersionPlot(std::ostream & os);

   void InitializeGLVis(VisData & vd);

   void DisplayToGLVis();

   void WriteVisItFields(const std::string & prefix,
                         const std::string & label);
private:

   MaxwellDispersion * disp_;
};

} // namespace meta_material
} // namespace mfem

#endif // MFEM_USE_MPI

#endif // MFEM_META_MATERIAL_SOLVER
