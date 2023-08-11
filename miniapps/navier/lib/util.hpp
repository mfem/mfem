#pragma once
#include <mfem.hpp>

static inline
void OrthoRHS(mfem::Vector &y)
{
   double local_sum = y.Sum();
   double global_sum = 0.0;
   MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM,
                 MPI_COMM_WORLD);

   int local_size = y.Size();
   int global_size = 0;
   MPI_Allreduce(&local_size, &global_size, 1, MPI_INT, MPI_SUM,
                 MPI_COMM_WORLD);
   y -= global_sum/static_cast<double>(global_size);
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