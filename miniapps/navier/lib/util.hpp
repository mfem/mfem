#pragma once
#include <mfem.hpp>

namespace mfem {
class BlockOrthoSolver : public Solver
{
private:
#ifdef MFEM_USE_MPI
   MPI_Comm mycomm;
   mutable HYPRE_BigInt global_size;
   const bool parallel;
#else
   mutable int global_size;
#endif

public:
   BlockOrthoSolver();
#ifdef MFEM_USE_MPI
   BlockOrthoSolver(MPI_Comm mycomm_, Array<int>& offsets)
   : Solver(0, false), mycomm(mycomm_), global_size(-1), parallel(true), offsets(offsets)
   {
      b_ortho.Update(offsets);
   };
#endif

   void SetSolver(Solver &s)
   {
      solver = &s;
      height = s.Height();
      width = s.Width();
      MFEM_VERIFY(height == width, "Solver must be a square Operator!");
      global_size = -1; // lazy evaluated
   }

   virtual void SetOperator(const Operator &op)
   {
      MFEM_VERIFY(solver, "Solver hasn't been set, call SetSolver() first.");
      solver->SetOperator(op);
      height = solver->Height();
      width = solver->Width();
      MFEM_VERIFY(height == width, "Solver must be a square Operator!");
      global_size = -1; // lazy evaluated
   }

   void Mult(const Vector &b, Vector &x) const
   {
      MFEM_VERIFY(solver, "Solver hasn't been set, call SetSolver() first.");
      MFEM_VERIFY(height == solver->Height(),
                  "solver was modified externally! call SetSolver() again!");
      MFEM_VERIFY(height == b.Size(), "incompatible input Vector size!");
      MFEM_VERIFY(height == x.Size(), "incompatible output Vector size!");

      const BlockVector bb(b.GetData(), offsets);
      BlockVector xb(x.GetData(), offsets);

      // Orthogonalize input
      b_ortho.GetBlock(0) = bb.GetBlock(0);
      Orthogonalize(bb.GetBlock(1), b_ortho.GetBlock(1));

      // Propagate iterative_mode to the solver:
      solver->iterative_mode = iterative_mode;

      // Apply the Solver
      solver->Mult(b_ortho, x);

      // Orthogonalize output
      Orthogonalize(xb.GetBlock(1), xb.GetBlock(1));
   }

private:
   Solver *solver = nullptr;

   mutable BlockVector b_ortho;

   Array<int>& offsets;

   void Orthogonalize(const Vector &v, Vector &v_ortho) const
   {
      if (global_size == -1)
      {
         global_size = height;
#ifdef MFEM_USE_MPI
         if (parallel)
         {
            MPI_Allreduce(MPI_IN_PLACE, &global_size, 1, HYPRE_MPI_BIG_INT,
                        MPI_SUM, mycomm);
         }
#endif
      }

      // TODO: GPU/device implementation

      double global_sum = v.Sum();

#ifdef MFEM_USE_MPI
      if (parallel)
      {
         MPI_Allreduce(MPI_IN_PLACE, &global_sum, 1, MPI_DOUBLE, MPI_SUM, mycomm);
      }
#endif

      double ratio = global_sum / static_cast<double>(global_size);
      v_ortho.SetSize(v.Size());
      v.HostRead();
      v_ortho.HostWrite();
      for (int i = 0; i < v_ortho.Size(); ++i)
      {
         v_ortho(i) = v(i) - ratio;
      }
   }
};

class MyFGMRESSolver : public FGMRESSolver
{
public:
   MyFGMRESSolver(MPI_Comm mpi_comm) : FGMRESSolver(mpi_comm) { m = 50; }

   void SetOperator(const Operator &op)
   {
      oper = &op;
      height = op.Height();
      width = op.Width();
      // if (prec)
      // {
      //    prec->SetOperator(*oper);
      // }
}

};

static inline
void Orthogonalize(Vector &v)
{
   double loc_sum = v.Sum();
   double global_sum = 0.0;
   int loc_size = v.Size();
   int global_size = 0;

   MPI_Allreduce(&loc_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
   MPI_Allreduce(&loc_size, &global_size, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

   v -= global_sum / static_cast<double>(global_size);
}

}

static inline
void PrintJulia(mfem::Operator &A, std::string filename)
{
   std::ofstream of(filename);
   A.PrintMatlab(of);
   of.close();
}

namespace mfem
{

class FDJacobian : public Operator
{
public:
   FDJacobian(const Operator &op, const Vector &x) :
      Operator(op.Height()),
      op(op),
      x(x)
   {
      f.SetSize(Height());
      xpev.SetSize(Height());
      op.Mult(x, f);
      xnorm = x.Norml2();
   }

   void Mult(const Vector &v, Vector &y) const
   {
      // See [1] for choice of eps.
      //
      // [1] Woodward, C.S., Gardner, D.J. and Evans, K.J., 2015. On the use of
      // finite difference matrix-vector products in Newton-Krylov solvers for
      // implicit climate dynamics with spectral elements. Procedia Computer
      // Science, 51, pp.2036-2045.
      double eps = lambda * (lambda + xnorm / v.Norml2());

      for (int i = 0; i < x.Size(); i++)
      {
         xpev(i) = x(i) + eps * v(i);
      }

      // y = f(x + eps * v)
      op.Mult(xpev, y);

      // y = (f(x + eps * v) - f(x)) / eps
      for (int i = 0; i < x.Size(); i++)
      {
         y(i) = (y(i) - f(i)) / eps;
      }
   }

private:
   const Operator &op;
   Vector x, f;
   mutable Vector xpev;
   double lambda = 1.0e-6;
   double xnorm;
};

}