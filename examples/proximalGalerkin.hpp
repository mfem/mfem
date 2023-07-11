#ifndef PROXIMAL_GALERKIN_HPP
#define PROXIMAL_GALERKIN_HPP
#include "mfem.hpp"


double sigmoid(const double x)
{
   if (x < 0)
   {
      const double exp_x = std::exp(x);
      return exp_x / (1.0 + exp_x);
   }
   return 1.0 / (1.0 + std::exp(-x));
}

double dsigmoiddx(const double x)
{
   const double tmp = sigmoid(x);
   return tmp - std::pow(tmp, 2);
}

double logit(const double x)
{
   return std::log(x / (1.0 - x));
}

double simpRule(const double rho, const int exponent, const double rho0)
{
   return rho0 + (1.0 - rho0)*std::pow(rho, exponent);
}

double dsimpRuledx(const double rho, const int exponent, const double rho0)
{
   return exponent*(1.0 - rho0)*std::pow(rho, exponent - 1);
}

double d2simpRuledx2(const double rho, const int exponent, const double rho0)
{
   return exponent*(exponent - 1)*(1.0 - rho0)*std::pow(rho, exponent - 2);
}


namespace mfem
{
class MultiProductCoefficient : public Coefficient
{
private:
   double aConst;
   Array<Coefficient *> a;

public:
   /// Constructor with one coefficient.  Result is A * B.
   MultiProductCoefficient(double A, Coefficient &B)
      : aConst(A), a(1) { a[0] = &B;}

   /// Constructor with two coefficients.  Result is A * B.
   MultiProductCoefficient(Coefficient &A, Coefficient &B)
      : aConst(1.0), a(2) { a[0] = &A; a[1] = &B;}

   /// Constructor with two coefficients.  Result is A * B.
   MultiProductCoefficient(double A, Coefficient &B, Coefficient &C)
      : aConst(A), a(2) { a[0] = &B; a[1] = &C;}

   /// Constructor with two coefficients.  Result is A * B.
   MultiProductCoefficient(Coefficient &A, Coefficient &B, Coefficient &C)
      : aConst(1.0), a(3) { a[0] = &A; a[1] = &B; a[2] = &C; }

   /// Constructor with two coefficients.  Result is A * B.
   MultiProductCoefficient(double A, Coefficient &B, Coefficient &C,
                           Coefficient &D)
      : aConst(A), a(3) { a[0] = &B; a[1] = &C; a[2] = &D;}

   /// Constructor with two coefficients.  Result is A * B.
   MultiProductCoefficient(Array<Coefficient *> &A)
      : aConst(1.0), a(A) {}
   /// Constructor with two coefficients.  Result is A * B.
   MultiProductCoefficient(double A, Array<Coefficient *> &B)
      : aConst(A), a(B) {}

   /// Set the time for internally stored coefficients
   void SetTime(double t)
   {
      for (auto &c : a) { c->SetTime(t); }
   }

   void Mult(double A) { aConst *= A; }
   void Mult(Coefficient &A) { a.Append(&A); }

   /// Evaluate the coefficient at @a ip.
   virtual double Eval(ElementTransformation &T,
                       const IntegrationPoint &ip)
   {
      double val = aConst;
      for (auto &c : a) { val *= c->Eval(T, ip); }
      return val;
   }
};
class MultiProductVectorCoefficient : public VectorCoefficient
{
private:
   double aConst;
   Array<Coefficient *> a;
   VectorCoefficient * v;

public:
   /// Constructor with one coefficient.  Result is A * B.
   MultiProductVectorCoefficient(double A, VectorCoefficient &V)
      : aConst(A), a(0), v(&V), VectorCoefficient(V.GetVDim()) {}

   MultiProductVectorCoefficient(Coefficient &A, VectorCoefficient &V)
      : aConst(1.0), a(0), v(&V), VectorCoefficient(V.GetVDim()) { a.Append(&A); }
   /// Constructor with one coefficient.  Result is A * V.
   MultiProductVectorCoefficient(double A, Coefficient &B, VectorCoefficient &V)
      : aConst(A), a(0), v(&V), VectorCoefficient(V.GetVDim()) { a.Append(&B); }

   MultiProductVectorCoefficient(Coefficient &A, Coefficient &B,
                                 VectorCoefficient &V)
      : aConst(1.0), a(0), v(&V), VectorCoefficient(V.GetVDim()) { a.Append(&A); a.Append(&B); }

   /// Constructor with one coefficient.  Result is A * V.
   MultiProductVectorCoefficient(double A, Coefficient &B, Coefficient &C,
                                 VectorCoefficient &V)
      : aConst(A), a(0), v(&V), VectorCoefficient(V.GetVDim()) { a.Append(&B); a.Append(&C); }

   MultiProductVectorCoefficient(Coefficient &A, Coefficient &B, Coefficient &C,
                                 VectorCoefficient &V)
      : aConst(1.0), a(0), v(&V), VectorCoefficient(V.GetVDim()) { a.Append(&A); a.Append(&B); a.Append(&C); }

   MultiProductVectorCoefficient(Array<Coefficient*> &A, VectorCoefficient &V)
      : aConst(1.0), a(A), v(&V), VectorCoefficient(V.GetVDim()) {}

   MultiProductVectorCoefficient(double A, Array<Coefficient*> &B,
                                 VectorCoefficient &V)
      : aConst(A), a(B), v(&V), VectorCoefficient(V.GetVDim()) {}

   /// Set the time for internally stored coefficients
   void SetTime(double t)
   {
      for (auto &c : a) { c->SetTime(t); }
      v->SetTime(t);
   };

   void Mult(double A) { aConst *= A; }
   void Mult(Coefficient &A) { a.Append(&A); }
   virtual void Eval(Vector &V, ElementTransformation &T,
                     const IntegrationPoint &ip)
   {
      double val = aConst;
      for (auto &c : a) { val *= c->Eval(T, ip); }
      v->Eval(V, T, ip);
      V *= val;
   };
   virtual void Eval(DenseMatrix &M, ElementTransformation &T,
                     const IntegrationRule &ir)
   {
      const int N = ir.GetNPoints();
      M.SetSize(v->GetVDim(), N);
      Vector col;
      for (int i=0; i<N; i++)
      {
         M.GetColumnReference(i, col);
         const IntegrationPoint &ip = ir.IntPoint(i);
         T.SetIntPoint(&ip);
         v->Eval(col, T, ip);
         double val = aConst;
         for (auto &c : a) { val *= c->Eval(T, ip); }
         col *= val;
      }
   }
};


class MappedGridFunctionCoefficient :public GridFunctionCoefficient
{
   typedef std::__1::function<double (const double)> Mapping;
public:
   MappedGridFunctionCoefficient(GridFunction *gf, Mapping f_x, int comp=1)
      :GridFunctionCoefficient(gf, comp), map(f_x) { }

   /// Evaluate the coefficient at @a ip.
   virtual double Eval(ElementTransformation &T,
                       const IntegrationPoint &ip)
   {
      return map(GridFunctionCoefficient::Eval(T, ip));
   }

   inline void SetMapping(Mapping &f_x) { map = f_x; }
private:
   Mapping map;
};

MappedGridFunctionCoefficient SIMPCoefficient(GridFunction *gf,
                                              const int exponent, const double rho0)
{
   auto map = [exponent, rho0](const double x) {return simpRule(x, exponent, rho0); };
   return MappedGridFunctionCoefficient(gf, map);
}

MappedGridFunctionCoefficient DerSIMPCoefficient(GridFunction *gf,
                                                 const int exponent, const double rho0)
{
   auto map = [exponent, rho0](const double x) {return dsimpRuledx(x, exponent, rho0); };
   return MappedGridFunctionCoefficient(gf, map);
}

MappedGridFunctionCoefficient Der2SIMPCoefficient(GridFunction *gf,
                                                  const int exponent, const double rho0)
{
   auto map = [exponent, rho0](const double x) {return d2simpRuledx2(x, exponent, rho0); };
   return MappedGridFunctionCoefficient(gf, map);
}

MappedGridFunctionCoefficient SigmoidCoefficient(GridFunction *gf)
{
   return MappedGridFunctionCoefficient(gf, sigmoid);
}

MappedGridFunctionCoefficient DerSigmoidCoefficient(GridFunction *gf)
{
   return MappedGridFunctionCoefficient(gf, dsigmoiddx);
}

double VolumeProjection(GridFunction &psi, const double target_volume,
                        const double tol=1e-12,
                        const int max_its=10)
{
   auto sigmoid_psi = SigmoidCoefficient(&psi);
   auto der_sigmoid_psi = DerSigmoidCoefficient(&psi);

   LinearForm int_sigmoid_psi(psi.FESpace());
   int_sigmoid_psi.AddDomainIntegrator(new DomainLFIntegrator(sigmoid_psi));
   LinearForm int_der_sigmoid_psi(psi.FESpace());
   int_der_sigmoid_psi.AddDomainIntegrator(new DomainLFIntegrator(
                                              der_sigmoid_psi));
   bool done = false;
   for (int k=0; k<max_its; k++) // Newton iteration
   {
      int_sigmoid_psi.Assemble(); // Recompute f(c) with updated ψ
      const double f = int_sigmoid_psi.Sum() - target_volume;

      int_der_sigmoid_psi.Assemble(); // Recompute df(c) with updated ψ
      const double df = int_der_sigmoid_psi.Sum();

      const double dc = -f/df;
      psi += dc;
      if (abs(dc) < tol) { done = true; break; }
   }
   if (!done)
   {
      mfem_warning("Projection reached maximum iteration without converging. Result may not be accurate.");
   }
   int_sigmoid_psi.Assemble();
   return int_sigmoid_psi.Sum();
}

#ifdef MFEM_USE_MPI
double ParVolumeProjection(ParGridFunction &psi, const double target_volume,
                           const double tol=1e-12, const int max_its=10)
{
   auto sigmoid_psi = SigmoidCoefficient(&psi);
   auto der_sigmoid_psi = DerSigmoidCoefficient(&psi);
   auto comm = psi.ParFESpace()->GetComm();

   ParLinearForm int_sigmoid_psi(psi.ParFESpace());
   int_sigmoid_psi.AddDomainIntegrator(new DomainLFIntegrator(sigmoid_psi));
   ParLinearForm int_der_sigmoid_psi(psi.ParFESpace());
   int_der_sigmoid_psi.AddDomainIntegrator(new DomainLFIntegrator(
                                              der_sigmoid_psi));
   bool done = false;
   for (int k=0; k<max_its; k++) // Newton iteration
   {
      int_sigmoid_psi.Assemble(); // Recompute f(c) with updated ψ
      const double myf = int_sigmoid_psi.Sum();
      double f;
      MPI_Allreduce(&myf, &f, 1, MPI_DOUBLE, MPI_SUM, comm);
      f -= target_volume;

      int_der_sigmoid_psi.Assemble(); // Recompute df(c) with updated ψ
      const double mydf = int_der_sigmoid_psi.Sum();
      double df;
      MPI_Allreduce(&mydf, &df, 1, MPI_DOUBLE, MPI_SUM, comm);

      const double dc = -f/df;
      psi += dc;
      if (abs(dc) < tol) { done = true; break; }
   }
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   if ((rank == 0) & (!done))
   {
      mfem_warning("Projection reached maximum iteration without converging. Result may not be accurate.");
   }
   const double myf = int_sigmoid_psi.Sum();
   double f;
   MPI_Allreduce(&myf, &f, 1, MPI_DOUBLE, MPI_SUM, comm);
   return f;
}
#endif // end of MFEM_USE_MPI for ParProjit

Array<int> getOffsets(Array<FiniteElementSpace*> &spaces)
{
   Array<int> offsets(spaces.Size() + 1);
   offsets[0] = 0;
   for (int i=0; i<spaces.Size(); i++)
   {
      offsets[i + 1] = spaces[i]->GetVSize();
   }
   offsets.PartialSum();
   return offsets;
}

class BlockLinearSystem
{
public:
   bool own_blocks = false;
   BlockLinearSystem(Array<int> &offsets,
                     Array<FiniteElementSpace*> &array_of_spaces,
                     Array2D<int> &array_of_ess_bdr)
      : spaces(array_of_spaces), ess_bdr(array_of_ess_bdr),
        numSpaces(array_of_spaces.Size()),
        A(offsets), b(offsets), prec(offsets)
   {
      A_forms.SetSize(numSpaces, numSpaces);
      A_forms = nullptr;

      b_forms.SetSize(numSpaces);
      for (int i=0; i<numSpaces; i++) { b_forms[i] = new LinearForm(spaces[i], b.GetBlock(i).GetData()); }

      prec.owns_blocks = true;
   }

   void SetBlockMatrix(int i, int j, Matrix *mat)
   {

      if (i == j)
      {
         auto bilf = static_cast<BilinearForm*>(mat);
         if (!bilf) { mfem_error("Cannot convert provided Matrix to BilinearForm"); }
         SetDiagBlockMatrix(i, bilf);
         return;
      }
      auto bilf = static_cast<MixedBilinearForm*>(mat);
      if (!bilf) { mfem_error("Cannot convert provided Matrix to MixedBilinearForm"); }
      if (bilf->TestFESpace() != spaces[i])
      {
         mfem_error("The provided BilinearForm's test space does not match with the provided array of spaces. Check the initialization and block index");
      }
      if (bilf->TrialFESpace() != spaces[j])
      {
         mfem_error("The provided BilinearForm's trial space does not match with the provided array of spaces. Check the initialization and block index");
      }
      A_forms(i, j) = bilf;
   }

   void SetDiagBlockMatrix(int i, BilinearForm *bilf)
   {
      if (bilf->FESpace() != spaces[i]) {mfem_error("The provided BilinearForm's space does not match with the provided array of spaces. Check the initialization and block index"); }
      A_forms(i, i) = bilf;
   }

   inline MixedBilinearForm *GetBlock(int i, int j)
   {
      if (i == j) { mfem_error("For diagonal block, use GetDiagBlock(i)."); }
      return static_cast<MixedBilinearForm*>(A_forms(i, j));
   }
   inline BilinearForm *GetDiagBlock(int i)
   {
      return static_cast<BilinearForm*>(A_forms(i, i));
   }
   inline LinearForm *GetLinearForm(int i)
   {
      return b_forms[i];
   }

   /**
    * @brief Assemble block matrices. X should contain essential boundary condition.
    *
    * @param x Solution vector that contains the essential boundary condition.
    */
   void Assemble(BlockVector &x);

   void SolveDiag(BlockVector &x, const Array<int> &ordering,
                  const bool isSPD=false);

   void GMRES(BlockVector &x);
   void PCG(BlockVector &x);

   ~BlockLinearSystem() = default;

private:
   Array<FiniteElementSpace*> &spaces;
   Array2D<int> &ess_bdr;
   int numSpaces;
   BlockOperator A;
   BlockVector b;
   Array2D<Matrix*> A_forms;
   Array<LinearForm*> b_forms;
   BlockDiagonalPreconditioner prec;
};

class VectorGradientGridFunctionCoefficient : public MatrixCoefficient
{
public:
   VectorGradientGridFunctionCoefficient(GridFunction *gf)
      : GridFunc(gf), vdim(gf->VectorDim()),
        sdim(gf->FESpace()->GetMesh()->Dimension()),
        MatrixCoefficient(
           gf->VectorDim(), gf->FESpace()->GetMesh()->Dimension(), false) {}
   virtual void Eval(DenseMatrix &K, ElementTransformation &T,
                     const IntegrationPoint &ip)
   {
      Mesh *gf_mesh = GridFunc->FESpace()->GetMesh();
      if (T.mesh->GetNE() == gf_mesh->GetNE())
      {
         Vector V;
         GridFunc->GetVectorGradient(T, K);
      }
      else
      {
         mfem_error("Inconsistent mesh.");
      }
   }

private:
   const int vdim, sdim;
   GridFunction *GridFunc;
};
class FrobeniusNormCoefficient : public Coefficient
{
private:
   MatrixCoefficient * a;

   mutable DenseMatrix mat_a;
public:
   /// Construct with the two vector coefficients.  Result is \f$ A \cdot B \f$.
   FrobeniusNormCoefficient(MatrixCoefficient &A): a(&A), mat_a(A.GetHeight(),
                                                                   A.GetWidth()) {};

   /// Set the time for internally stored coefficients
   void SetTime(double t) { Coefficient::SetTime(t); a->SetTime(t); }

   /// Reset the first vector in the inner product
   void SetACoef(MatrixCoefficient &A) { a = &A; }
   /// Return the first vector coefficient in the inner product
   MatrixCoefficient * GetACoef() const { return a; }

   /// Evaluate the coefficient at @a ip.
   virtual double Eval(ElementTransformation &T,
                       const IntegrationPoint &ip)
   {
      a->Eval(mat_a, T, ip);
      Vector v(mat_a.GetData(), mat_a.Height()*mat_a.Width());
      return v.Norml2();
   }
};


void BlockLinearSystem::Assemble(BlockVector &x)
{
   Array<int> trial_ess_bdr;
   Array<int> test_ess_bdr;
   b = 0.0;
   for (int row = 0; row < numSpaces; row++)
   {
      b_forms[row]->Assemble();
      test_ess_bdr.MakeRef(ess_bdr.GetRow(row), ess_bdr.NumCols());
      if (A_forms(row, row))
      {
         BilinearForm* bilf = this->GetDiagBlock(row);
         delete bilf->LoseMat();
         bilf->SetDiagonalPolicy(mfem::Operator::DIAG_ONE);
         bilf->Assemble();
         bilf->EliminateEssentialBC(test_ess_bdr, x.GetBlock(row), b.GetBlock(row));
         bilf->Finalize();
         A.SetBlock(row, row, &(bilf->SpMat()));
         prec.SetDiagonalBlock(row, new GSSmoother(bilf->SpMat()));
      }
      for (int col = 0; col < numSpaces; col++)
      {
         if (col == row) // if diagonal, already handled using bilinear form
         {
            continue;
         }
         trial_ess_bdr.MakeRef(ess_bdr.GetRow(col), ess_bdr.NumCols());
         if (A_forms(row, col))
         {
            MixedBilinearForm* bilf = this->GetBlock(row, col);
            delete bilf->LoseMat();
            bilf->Assemble();
            bilf->EliminateTrialDofs(trial_ess_bdr, x.GetBlock(col), b.GetBlock(col));
            bilf->EliminateTestDofs(test_ess_bdr);
            bilf->Finalize();
            A.SetBlock(row, col, &(bilf->SpMat()));
         }
      }
   }
}


void BlockLinearSystem::SolveDiag(BlockVector &x, const Array<int> &ordering,
                                  const bool isSPD)
{
   Array<int> curr_ess_bdr;
   for (int i:ordering)
   {
      curr_ess_bdr.MakeRef(ess_bdr.GetRow(i), ess_bdr.NumCols());
      b.GetBlock(i) = 0.0;
      b_forms[i]->Assemble();
      BilinearForm* bilf = this->GetDiagBlock(i);
      delete bilf->LoseMat();
      bilf->SetDiagonalPolicy(mfem::Operator::DIAG_ONE);
      bilf->Assemble();
      bilf->EliminateEssentialBC(curr_ess_bdr, x.GetBlock(i), b.GetBlock(i));
      bilf->Finalize();
      SparseMatrix &mat = this->GetDiagBlock(i)->SpMat();
      GSSmoother curr_prec(mat);
      if (isSPD)
      {
         mfem::PCG(mat, curr_prec, b.GetBlock(i), x.GetBlock(i), 0, 2000, 1e-12, 0.0);
      }
      else
      {
         mfem::GMRES(mat, curr_prec, b.GetBlock(i), x.GetBlock(i), 0, 2000, 50, 1e-12,
                     0.0);
      }
   }
}

void BlockLinearSystem::GMRES(BlockVector &x)
{
   mfem::GMRES(A, prec, b, x, 0, 2000, 50, 1e-12, 0.0);
}

void BlockLinearSystem::PCG(BlockVector &x)
{
   mfem::PCG(A, prec, b, x, 0, 2000, 1e-12, 0.0);
}

// class AndersonAccelerator
// {
//    typedef std::__1::function<void(Vector &)> FixedPointMap;
//    public:
//    AndersonAccelerator(FixedPointMap f, Vector &x, const int m_=3, const double tolerance=1e-6, const int max_iteration=100): op(f), sol(x), X(m_), Gk(m_), m(m_), tol(tolerance), max_it(max_iteration) {
//       Gk = nullptr;
//       k = 0;
//       converged = false;
//    }
//    void Solve()
//    {
//       X[0] = new Vector(sol);
//       Gk[0] =
//       for(k=)
//       {
//          const int m_k = std::min(m, k);
//          FixedPointIteration();
//          if (m < k)
//          {
//             X[k] = new Vector()
//          }
//       }
//    }
//    void FixedPointIteration()
//    {
//    }

//    private:
//    FixedPointMap op;
//    Vector &sol;
//    Array<Vector*> X;
//    Array<Vector*> Gk;
//    const int m;
//    const double tol;
//    const int max_it;
//    int k;
//    bool converged;

// };
} // end of namespace mfem

#endif // end of proximalGalerkin.hpp