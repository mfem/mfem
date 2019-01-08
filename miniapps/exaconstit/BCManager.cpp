

#include "mfem.hpp"
#include "BCManager.hpp"
#include <fstream>

namespace mfem
{

BCManager::BCManager() 
{
   // TODO constructor stub
}

BCManager::~BCManager()
{
   // TODO destructor stub
}

static void mark_dofs(const Array<int> &dofs, Array<int> &mark_array)
{
   for (int i = 0; i < dofs.Size(); i++)
   {
      int k = dofs[i];
      if (k < 0) { k = -1 - k; }
      mark_array[k] = -1;
   }
}

// set partial dof component list for all essential BCs based on my 
// custom BC manager and input, srw.
void NonlinearForm::SetEssentialBCPartial(const Array<int> &bdr_attr_is_ess,
                                          Vector *rhs)
{
   Array2D<int> component;
   Array<int> cmp_row;
   int id = -1;
   //The size here is set explicitly
   component.SetSize(bdr_attr_is_ess.Size(), 3);
   cmp_row.SetSize(3);
   
   component = 0;
   cmp_row = 0;

   for (int i=0; i<bdr_attr_is_ess.Size(); ++i) {
      if (bdr_attr_is_ess[i])
      {
         BCManager & bcManager = BCManager::getInstance();
         BCData & bc = bcManager.GetBCInstance(i+1); // this requires contiguous attribute ids
         BCData::getComponents(bc.compID, cmp_row);
         
         component(i, 0) = cmp_row[0];
         component(i, 1) = cmp_row[1];
         component(i, 2) = cmp_row[2];
      }
   }

   fes->GetEssentialTrueDofs(bdr_attr_is_ess, ess_tdof_list, component);

   if (rhs)
   {
      for (int i = 0; i < ess_tdof_list.Size(); i++)
      {
         (*rhs)(ess_tdof_list[i]) = 0.0;
      }
   }

}

void GridFunction::ProjectBdrCoefficient(VectorFunctionRestrictedCoefficient &vfcoeff)
                                         
{
   int i, j, fdof, d, ind, vdim;
   Vector val;
   const FiniteElement *fe;
   ElementTransformation *transf;
   Array<int> vdofs;
   int* active_attr = vfcoeff.GetActiveAttr();

   vdim = fes->GetVDim();
   // loop over boundary elements
   for (i = 0; i < fes->GetNBE(); i++)
   {
      // if boundary attribute is 1 (Dirichlet)
      if (active_attr[fes->GetBdrAttribute(i) - 1])
      {
         // instantiate a BC object
         BCManager & bcManager = BCManager::getInstance();
         BCData & bc = bcManager.GetBCInstance(fes->GetBdrAttribute(i));

         fe = fes->GetBE(i);
         fdof = fe->GetDof();
         transf = fes->GetBdrElementTransformation(i);
         const IntegrationRule &ir = fe->GetNodes();
         fes->GetBdrElementVDofs(i, vdofs);

         // loop over dofs
         for (j = 0; j < fdof; j++)
         {
            const IntegrationPoint &ip = ir.IntPoint(j);
            transf->SetIntPoint(&ip);
            
            vfcoeff.Eval(val, *transf, ip);

            // loop over vector dimensions
            for (d = 0; d < vdim; d++)
            {
               // check if the vector component (i.e. dof) is not constrained by a
               // partial essential BC
               if (bc.scale[d] > 0.0) 
               {
//                  printf("ProjectBdr, active_attr, component: %d %d \n", fes->GetBdrAttribute(i), d);
                  ind = vdofs[fdof*d+j];
                  if ( (ind = vdofs[fdof*d+j]) < 0 )
                  {
                     val(d) = -val(d), ind = -1-ind;
                  }
                  (*this)(ind) = val(d); // placing computed value in grid function
               }
            }
         }
      }
   }

}

}
