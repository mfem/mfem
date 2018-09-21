

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
         id = fes->GetBdrAttribute(i) - 1;
         BCManager & bcManager = BCManager::getInstance();
         BCData & bc = bcManager.GetBCInstance(i+1); // this requires contiguous attribute ids
         BCData::getComponents(bc.compID, cmp_row);
         
         component(id, 0) = cmp_row[0];
         component(id, 1) = cmp_row[1];
         component(id, 2) = cmp_row[2];

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

//void ParFiniteElementSpace::GetEssentialTrueDofs(const Array<int> &bdr_attr_is_ess,
//                                                 Array<int> &ess_tdof_list,
//                                                 Array2D<int> componentID)
//{
//   Array<int> ess_dofs, true_ess_dofs;
//
//   GetEssentialVDofs(bdr_attr_is_ess, ess_dofs, componentID);
//   GetRestrictionMatrix()->BooleanMult(ess_dofs, true_ess_dofs);
//   MarkerToList(true_ess_dofs, ess_tdof_list);
//}

//void FiniteElementSpace::GetEssentialTrueDofs(const Array<int> &bdr_attr_is_ess,
//                                              Array<int> &ess_tdof_list,
//                                              Array2D<int> componentID)
//{
//   Array<int> ess_vdofs, ess_tdofs;
//   GetEssentialVDofs(bdr_attr_is_ess, ess_vdofs, componentID);
//   const SparseMatrix *R = GetConformingRestriction();
//   if (!R)
//   {
//      ess_tdofs.MakeRef(ess_vdofs);
//   }
//   else
//   {
//      R->BooleanMult(ess_vdofs, ess_tdofs);
//   }
//   MarkerToList(ess_tdofs, ess_tdof_list);
//}

//void ParFiniteElementSpace::GetEssentialVDofs(const Array<int> &bdr_attr_is_ess,
//                                              Array<int> &ess_dofs,
//                                              Array2D<int> componentID) const
//{
//   FiniteElementSpace::GetEssentialVDofs(bdr_attr_is_ess, ess_dofs, componentID);
//
//   if (Conforming())
//   {
//      Synchronize(ess_dofs);
//   }
//}

//void FiniteElementSpace::GetEssentialVDofs(const Array<int> &bdr_attr_is_ess,
//                                           Array<int> &ess_vdofs,
//                                           Array2D<int> componentID) const
//{
//   // Note this doesn't treat mesh->ncmesh like the GetEssentialVDofs in
//   // fem/fespace.cpp
//   Array<int> vdofs, dofs, ess_comp;
//
//   bool cmpID_rows = FALSE;
//
//   ess_vdofs.SetSize(GetVSize());
////   printf("ess_vdofs.size %d \n", GetVSize());
//   ess_vdofs = 0;
//
//   ess_comp.SetSize(3);
//   ess_comp = 0;
//
//   for (int i = 0; i < GetNBE(); i++)
//   {
//      int id = GetBdrAttribute(i)-1;
//      if (bdr_attr_is_ess[id])
//      {
//        cmpID_rows = TRUE;
//        //Checking to see if all the values in the row are greater than -1 if so then
//        //all components on a boundary are said to be marked.
//        for(int j = 0; j < componentID.NumCols(); ++j)
//        {
//            cmpID_rows = cmpID_rows && ( componentID(id, j) > -1);
//        }
////         printf("GetEssentialVDofs componentID %d \n", componentID[id]);
//         if (cmpID_rows) // same as srw component id system
//         {
//            // Mark all components.
//            GetBdrElementVDofs(i, vdofs);
//            mark_dofs(vdofs, ess_vdofs);
//         }
//         else // changed based on srw component id system
//         {
//            GetBdrElementDofs(i, dofs);
////            BCData::getComponents(componentID[id], ess_comp);
//            for (int d = 0; d < dofs.Size(); d++)
//            {
//               // loop over actively constrained components
//               for (int k = 0; k < componentID.NumCols(); ++k)
//               {
//                  if (componentID(id, k) != -1) { // -1 means inactive component
//                                           // valid components are x = 0, y = 1, z = 2
//                     dofs[d] = DofToVDof(dofs[d], componentID(id, k));
////                     printf("GetEssentialVDofs: %d %d %d %d \n", (id+1), k, ess_comp[k], dofs[d]);
//                  }
//               }
//            }
////            printf("GetEssentialVDofs, size of dofs, ess_vdofs: %d %d \n", dofs.Size(), ess_vdofs.Size());
//            mark_dofs(dofs, ess_vdofs); // do this only once?
//         }
//      }
//   }
//}

}
