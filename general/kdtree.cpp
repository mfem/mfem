// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "kdtree.hpp"

namespace mfem
{

int KDTreeM::FindClosestPoint(mfem::Vector& pt)
{
   PointS best_candidate;
   best_candidate.sp.xx=pt.GetData();
   //initialize the best candidate
   best_candidate.pos =0;
   best_candidate.dist=Dist(data[0], best_candidate.sp);
   best_candidate.level=0;
   PSearch(data.begin(), data.end(), 0, best_candidate);
   int ind=(best_candidate.sp.xx-gf->GetData())/ndim;
   return ind;
}


void KDTreeM::FindClosestPoint(Vector& pt, int &ind, double& dist)
{
   PointS best_candidate;
   best_candidate.sp.xx=pt.GetData();
   //initialize the best candidate
   best_candidate.pos =0;
   best_candidate.dist=Dist(data[0], best_candidate.sp);
   best_candidate.level=0;
   PSearch(data.begin(), data.end(), 0, best_candidate);

   dist=best_candidate.dist;
   ind=data[best_candidate.pos].ind;
}

void KDTreeM::_FindClosestPoint(Vector& pt,int& ind, double& dist)
{
   PointS best_candidate;
   best_candidate.sp.xx=pt.GetData();


   //initialize the best candidate
   best_candidate.pos =0;
   best_candidate.dist=Dist(data[0], best_candidate.sp);
   double dd;
   for (size_t i=1; i<data.size(); i++)
   {
      dd=Dist(data[i],  best_candidate.sp);
      if (dd<best_candidate.dist)
      {
         best_candidate.pos=i;
         best_candidate.dist=dd;
      }
   }
   dist=best_candidate.dist;
   ind=data[best_candidate.pos].ind;
}

void KDTreeM::PSearch(typename std::vector<NodeND>::iterator itb,
                      typename std::vector<NodeND>::iterator ite,
                      size_t level, PointS& bc)
{
   std::uint8_t dim=(std::uint8_t) (level%ndim);
   size_t siz=ite-itb;
   typename std::vector<NodeND>::iterator mtb=itb+siz/2;

   if (siz>2)
   {
      //median is at itb+siz/2
      level=level+1;
      if ((bc.sp.xx[dim]-bc.dist)>mtb->xx[dim]) //look on the right only
      {
         PSearch(itb+siz/2+1, ite, level, bc);
      }
      else if ((bc.sp.xx[dim]+bc.dist)<mtb->xx[dim]) //look on the left only
      {
         PSearch(itb,itb+siz/2, level, bc);
      }
      else  //check all
      {
         if (bc.sp.xx[dim]<mtb->xx[dim])
         {
            //start with the left portion
            PSearch(itb,itb+siz/2, level, bc);
            //and continue to the right
            if (!((bc.sp.xx[dim]+bc.dist)<mtb->xx[dim]))
            {
               PSearch(itb+siz/2+1, ite, level, bc);
               {
                  //check central one
                  double dd=Dist(*mtb, bc.sp);
                  if (dd<bc.dist) { bc.dist=dd; bc.pos=mtb-data.begin(); bc.level=level; }
               }//end central point check
            }
         }
         else
         {
            //start with the right portion
            PSearch(itb+siz/2+1, ite, level, bc);
            //and continue with left
            if (!((bc.sp.xx[dim]-bc.dist)>mtb->xx[dim]))
            {
               PSearch(itb, itb+siz/2, level, bc);
               {
                  //check central one
                  double dd=Dist(*mtb, bc.sp);
                  if (dd<bc.dist) { bc.dist=dd; bc.pos=mtb-data.begin(); bc.level=level; }
               }//end central point check
            }
         }
      }
   }
   else
   {
      //check the nodes
      double dd;
      for (auto it=itb; it!=ite; it++)
      {
         dd=Dist(*it, bc.sp);
         if (dd<bc.dist) //update bc
         {
            bc.pos=it-data.begin();
            bc.dist=dd;
            bc.level=level;
         }
      }
   }
}

}
