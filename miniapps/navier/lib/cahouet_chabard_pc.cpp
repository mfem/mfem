#include "cahouet_chabard_pc.hpp"

using namespace mfem;

CahouetChabardPC::CahouetChabardPC(Solver &Mp_inv, Solver &Lp_inv,
                                   const double dt,
                                   Array<int> &pres_ess_tdofs) :
   Solver(Mp_inv.Height()),
   Mp_inv(Mp_inv),
   Lp_inv(Lp_inv),
   z(Mp_inv.Height()),
   dt(dt),
   pres_ess_tdofs(pres_ess_tdofs) { }

void CahouetChabardPC::Mult(const Vector &x, Vector &y) const
{
   z.SetSize(y.Size());

   Lp_inv.Mult(x, y);
   y /= dt;
   Mp_inv.Mult(x, z);
   y += z;

   for (int i = 0; i < pres_ess_tdofs.Size(); i++)
   {
      y[pres_ess_tdofs[i]] = x[pres_ess_tdofs[i]];
   }
}
