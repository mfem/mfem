// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details

#ifndef MFEM_KDTREE_HPP
#define MFEM_KDTREE_HPP

#include "../config/config.hpp"

#include <vector>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <cmath>
#include <cstdint>
#include <tuple>

namespace mfem
{

namespace KDTreeNorms
{

/// Evaluates l1 norm of a vector.
template <typename Tfloat, int ndim>
struct Norm_l1
{
   Tfloat operator()(const Tfloat* xx) const
   {
      Tfloat tm=abs(xx[0]);
      for (int i=1; i<ndim; i++)
      {
         tm=tm+abs(xx[i]);
      }
      return tm;
   }
};

/// Evaluates l2 norm of a vector.
template<typename Tfloat,int ndim>
struct Norm_l2
{
   Tfloat operator()(const Tfloat* xx) const
   {
      Tfloat tm;
      tm=xx[0]*xx[0];
      for (int i=1; i<ndim; i++)
      {
         tm=tm+xx[i]*xx[i];
      }
      return sqrt(tm);
   }
};

/// Finds the max absolute value of a vector.
template<typename Tfloat,int ndim>
struct Norm_li
{
   Tfloat operator()(const Tfloat* xx) const
   {
      Tfloat tm;
      if (xx[0]<Tfloat(0.0)) { tm=-xx[0];}
      else { tm=xx[0];}
      for (int i=1; i<ndim; i++)
      {
         if (xx[i]<Tfloat(0.0))
         {
            if (tm<(-xx[i])) {tm=-xx[i];}
         }
         else
         {
            if (tm<xx[i]) {tm=xx[i];}
         }
      }
      return tm;
   }
};

}

/// @brief Abstract base class for KDTree. Can be used when the dimension of the
/// space is known dynamically.
template <typename Tindex, typename Tfloat>
class KDTreeBase
{
public:
   /// Adds a point to the tree. See KDTree::AddPoint().
   virtual void AddPoint(const Tfloat *xx, Tindex ii) = 0;
   /// @brief Sorts the tree. Should be performed after adding points and before
   /// performing queries. See KDTree::Sort().
   virtual void Sort() = 0;
   /// Returns the index of the closest point to @a xx.
   virtual Tindex FindClosestPoint(const Tfloat *xx) const = 0;
   /// Virtual destructor.
   virtual ~KDTreeBase() { }
};

/// Template class for build KDTree with template parameters Tindex
/// specifying the type utilized for indexing the points, Tfloat
/// specifying a float type for representing the coordinates of the
/// points, integer parameter ndim specifying the dimensionality of the
/// space and template function Tnorm for evaluating the distance
/// between two points. The KDTree class implements the standard k-d
/// tree data structure that can be used to transfer a ParGridFunction
/// defined on one MPI communicator to a ParGridFunction/GridFunction
/// defined on different MPI communicator. This can be useful when
/// comparing a solution computed on m ranks against a solution
/// computed with n or 1 rank(s).
template <typename Tindex, typename Tfloat, size_t ndim=3,
          typename Tnorm=KDTreeNorms::Norm_l2<Tfloat,ndim> >
class KDTree : public KDTreeBase<Tindex, Tfloat>
{
public:

   /// Structure defining a geometric point in the ndim-dimensional space. The
   /// coordinate type (Tfloat) can be any floating or integer type. It can be
   /// even a character if necessary. For such types users should redefine the
   /// norms.
   struct PointND
   {
      /// Default constructor: fill with zeros
      PointND() { std::fill(xx,xx+ndim,Tfloat(0.0)); }
      /// Copy coordinates from pointer/array @a xx_
      PointND(const Tfloat *xx_) { std::copy(xx_,xx_+ndim,xx); }
      /// Coordinates of the point
      Tfloat xx[ndim];
   };

   /// Structure defining a node in the KDTree.
   struct NodeND
   {
      /// Defines a point in the ndim-dimensional space
      PointND pt;
      /// Defines the attached index
      Tindex  ind = 0;
      /// Default constructor: fill with zeros
      NodeND() = default;
      /// Create from given point and index
      NodeND(PointND pt_, Tindex ind_ = 0) : pt(pt_), ind(ind_) { }
   };

   /// Default constructor
   KDTree() = default;

   /// Returns the spatial dimension of the points
   int SpaceDimension() const
   {
      return ndim;
   }

   /// Data iterator
   typedef typename std::vector<NodeND>::iterator iterator;

   /// Returns iterator to beginning of the point cloud
   iterator begin()
   {
      return data.begin();
   }

   /// Returns iterator to the end of the point cloud
   iterator end()
   {
      return data.end();
   }

   /// Returns the size of the point cloud
   size_t size() const
   {
      return data.size();
   }

   /// Clears the point cloud
   void clear()
   {
      data.clear();
   }

   /// Builds the KDTree. If the point cloud is modified the tree
   /// needs to be rebuild by a new call to Sort().
   void Sort() override
   {
      SortInPlace(data.begin(),data.end(),0);
   }

   /// Adds a new node to the point cloud
   void AddPoint(const PointND &pt, Tindex ii)
   {
      data.emplace_back(pt, ii);
   }

   /// Adds a new node by coordinates and an associated index
   void AddPoint(const Tfloat *xx,Tindex ii) override
   {
      data.emplace_back(xx, ii);
   }

   /// Finds the nearest neighbour index
   Tindex FindClosestPoint(const PointND &pt) const
   {
      PointS best_candidate;
      best_candidate.sp=pt;
      //initialize the best candidate
      best_candidate.pos =0;
      best_candidate.dist=Dist(data[0].pt, best_candidate.sp);
      best_candidate.level=0;
      PSearch(data.begin(), data.end(), 0, best_candidate);
      return data[best_candidate.pos].ind;
   }

   Tindex FindClosestPoint(const Tfloat *xx) const override
   {
      return FindClosestPoint(PointND(xx));
   }

   /// Finds the nearest neighbour index and return the clossest point in clp
   Tindex FindClosestPoint(const PointND &pt, const PointND &clp) const
   {
      PointS best_candidate;
      best_candidate.sp=pt;
      //initialize the best candidate
      best_candidate.pos =0;
      best_candidate.dist=Dist(data[0].pt, best_candidate.sp);
      best_candidate.level=0;
      PSearch(data.begin(), data.end(), 0, best_candidate);

      clp=data[best_candidate.pos].pt;
      return data[best_candidate.pos].ind;
   }

   /// Returns the closest point and the distance to the input point pt.
   void FindClosestPoint(const PointND &pt, Tindex &ind, Tfloat &dist) const
   {
      PointND clp;
      FindClosestPoint(pt,ind,dist,clp);
   }

   /// Returns the closest point and the distance to the input point pt.
   void FindClosestPoint(const PointND &pt, Tindex &ind, Tfloat &dist,
                         PointND &clp) const
   {
      PointS best_candidate;
      best_candidate.sp=pt;
      //initialize the best candidate
      best_candidate.pos =0;
      best_candidate.dist=Dist(data[0].pt, best_candidate.sp);
      best_candidate.level=0;
      PSearch(data.begin(), data.end(), 0, best_candidate);

      ind=data[best_candidate.pos].ind;
      dist=best_candidate.dist;
      clp=data[best_candidate.pos].pt;
   }


   /// Brute force search - please, use it only for debuging purposes
   void FindClosestPointSlow(const PointND &pt, Tindex &ind, Tfloat &dist) const
   {
      PointS best_candidate;
      best_candidate.sp=pt;
      //initialize the best candidate
      best_candidate.pos =0;
      best_candidate.dist=Dist(data[0].pt, best_candidate.sp);
      Tfloat dd;
      for (auto iti=data.begin()+1; iti!=data.end(); iti++)
      {
         dd=Dist(iti->pt,  best_candidate.sp);
         if (dd<best_candidate.dist)
         {
            best_candidate.pos=iti-data.begin();
            best_candidate.dist=dd;
         }
      }

      ind=data[best_candidate.pos].ind;
      dist=best_candidate.dist;
   }

   /// Finds all points within a distance R from point pt. The indices are
   /// returned in the vector res and the correponding distances in vector dist.
   void FindNeighborPoints(const PointND &pt,Tfloat R, std::vector<Tindex> & res,
                           std::vector<Tfloat> & dist)
   {
      FindNeighborPoints(pt,R,data.begin(),data.end(),0,res,dist);
   }

   /// Finds all points within a distance R from point pt. The indices are
   /// returned in the vector res and the correponding distances in vector dist.
   void FindNeighborPoints(const PointND &pt,Tfloat R, std::vector<Tindex> & res)
   {
      FindNeighborPoints(pt,R,data.begin(),data.end(),0,res);
   }

   /// Brute force search - please, use it only for debuging purposes
   void FindNeighborPointsSlow(const PointND &pt,Tfloat R,
                               std::vector<Tindex> &res,
                               std::vector<Tfloat> &dist)
   {
      Tfloat dd;
      for (auto iti=data.begin(); iti!=data.end(); iti++)
      {
         dd=Dist(iti->pt,  pt);
         if (dd<R)
         {
            res.push_back(iti->ind);
            dist.push_back(dd);
         }
      }
   }

   /// Brute force search - please, use it only for debuging purposes
   void FindNeighborPointsSlow(const PointND &pt,Tfloat R,
                               std::vector<Tindex> &res)
   {
      Tfloat dd;
      for (auto iti=data.begin(); iti!=data.end(); iti++)
      {
         dd=Dist(iti->pt,  pt);
         if (dd<R)
         {
            res.push_back(iti->ind);
         }
      }
   }

private:

   /// Functor utilized in the coordinate comparison
   /// for building the KDTree
   struct CompN
   {
      /// Current coordinate index
      std::uint8_t dim;

      /// Constructor for the comparison
      CompN(std::uint8_t dd):dim(dd) {}

      /// Compares two points p1 and p2
      bool operator() (const PointND& p1, const PointND& p2)
      {
         return p1.xx[dim]<p2.xx[dim];
      }

      /// Compares two nodes n1 and n2
      bool operator() (const NodeND& n1, const NodeND& n2)
      {
         return  n1.pt.xx[dim]<n2.pt.xx[dim];
      }
   };

   mutable PointND tp; ///< Point for storing tmp data
   Tnorm fnorm;

   /// Computes the distance between two nodes
   Tfloat Dist(const PointND &pt1, const PointND &pt2) const
   {
      for (size_t i=0; i<ndim; i++)
      {
         tp.xx[i]=pt1.xx[i]-pt2.xx[i];
      }
      return fnorm(tp.xx);
   }

   /// The point cloud is stored in a vector.
   std::vector<NodeND> data;

   /// Finds the median for a sequence of nodes starting with itb
   /// and ending with ite. The current coordinate index is set by cdim.
   Tfloat FindMedian(typename std::vector<NodeND>::iterator itb,
                     typename std::vector<NodeND>::iterator ite,
                     std::uint8_t cdim)
   {
      size_t siz=ite-itb;
      std::nth_element(itb, itb+siz/2, ite, CompN(cdim));
      return itb->pt.xx[cdim];
   }

   /// Sorts the point cloud
   void SortInPlace(typename std::vector<NodeND>::iterator itb,
                    typename std::vector<NodeND>::iterator ite,
                    size_t level)
   {
      std::uint8_t cdim=(std::uint8_t)(level%ndim);
      size_t siz=ite-itb;
      if (siz>2)
      {
         std::nth_element(itb, itb+siz/2, ite, CompN(cdim));
         level=level+1;
         SortInPlace(itb, itb+siz/2, level);
         SortInPlace(itb+siz/2+1,ite, level);
      }
   }

   /// Structure utilized for nearest neighbor search (NNS)
   struct PointS
   {
      Tfloat dist;
      size_t pos;
      size_t level;
      PointND sp;
   };

   /// Finds the closest point to bc.sp in the point cloud
   /// bounded between [itb,ite).
   void PSearch(typename std::vector<NodeND>::const_iterator itb,
                typename std::vector<NodeND>::const_iterator ite,
                size_t level, PointS& bc) const
   {
      std::uint8_t dim=(std::uint8_t) (level%ndim);
      size_t siz=ite-itb;
      typename std::vector<NodeND>::const_iterator mtb=itb+siz/2;
      if (siz>2)
      {
         // median is at itb+siz/2
         level=level+1;
         if ((bc.sp.xx[dim]-bc.dist)>mtb->pt.xx[dim]) // look on the right only
         {
            PSearch(itb+siz/2+1, ite, level, bc);
         }
         else if ((bc.sp.xx[dim]+bc.dist)<mtb->pt.xx[dim]) // look on the left only
         {
            PSearch(itb,itb+siz/2, level, bc);
         }
         else  // check all
         {
            if (bc.sp.xx[dim]<mtb->pt.xx[dim])
            {
               // start with the left portion
               PSearch(itb,itb+siz/2, level, bc);
               // and continue to the right
               if (!((bc.sp.xx[dim]+bc.dist)<mtb->pt.xx[dim]))
               {
                  PSearch(itb+siz/2+1, ite, level, bc);
                  {
                     // check central one
                     Tfloat dd=Dist(mtb->pt, bc.sp);
                     if (dd<bc.dist)
                     {
                        bc.dist=dd; bc.pos=mtb-data.begin(); bc.level=level;
                     }
                  } // end central point check
               }
            }
            else
            {
               // start with the right portion
               PSearch(itb+siz/2+1, ite, level, bc);
               // and continue with left
               if (!((bc.sp.xx[dim]-bc.dist)>mtb->pt.xx[dim]))
               {
                  PSearch(itb, itb+siz/2, level, bc);
                  {
                     // check central one
                     Tfloat dd=Dist(mtb->pt, bc.sp);
                     if (dd<bc.dist)
                     {
                        bc.dist=dd; bc.pos=mtb-data.begin(); bc.level=level;
                     }
                  } // end central point check
               }
            }
         }
      }
      else
      {
         // check the nodes
         Tfloat dd;
         for (auto it=itb; it!=ite; it++)
         {
            dd=Dist(it->pt, bc.sp);
            if (dd<bc.dist) // update bc
            {
               bc.pos=it-data.begin();
               bc.dist=dd;
               bc.level=level;
            }
         }
      }
   }

   /// Returns distances and indices of the n closest points to a point pt.
   void NNS(PointND& pt,const int& npoints,
            typename std::vector<NodeND>::iterator itb,
            typename std::vector<NodeND>::iterator ite,
            size_t level,
            std::vector< std::tuple<Tfloat,Tindex> > & res) const
   {
      std::uint8_t dim=(std::uint8_t) (level%ndim);
      size_t siz=ite-itb;
      typename std::vector<NodeND>::iterator mtb=itb+siz/2;
      if (siz>2)
      {
         // median is at itb+siz/2
         level=level+1;
         Tfloat R=std::get<0>(res[npoints-1]);
         // check central one
         Tfloat dd=Dist(mtb->pt, pt);
         if (dd<R)
         {
            res[npoints-1]=std::make_tuple(dd,mtb->ind);
            std::nth_element(res.begin(), res.end()-1, res.end());
            R=std::get<0>(res[npoints-1]);
         }
         if ((pt.xx[dim]-R)>mtb->pt.xx[dim]) // look to the right only
         {
            NNS(pt, npoints, itb+siz/2+1, ite, level, res);
         }
         else if ((pt.xx[dim]+R)<mtb->pt.xx[dim]) // look to the left only
         {
            NNS(pt, npoints, itb, itb+siz/2, level,   res);
         }
         else  // check all
         {
            NNS(pt,npoints, itb+siz/2+1, ite, level, res); // right
            NNS(pt,npoints, itb, itb+siz/2, level,   res); // left
         }
      }
      else
      {
         Tfloat dd;
         for (auto it=itb; it!=ite; it++)
         {
            dd=Dist(it->pt, pt);
            if (dd< std::get<0>(res[npoints-1])) // update the list
            {
               res[npoints-1]=std::make_tuple(dd,it->ind);
               std::nth_element(res.begin(), res.end()-1, res.end());
            }
         }
      }
   }

   /// Finds the set of indices of points within a distance R of a point pt.
   void FindNeighborPoints(PointND& pt, Tfloat R,
                           typename std::vector<NodeND>::iterator itb,
                           typename std::vector<NodeND>::iterator ite,
                           size_t level,
                           std::vector<Tindex> & res) const
   {
      std::uint8_t dim=(std::uint8_t) (level%ndim);
      size_t siz=ite-itb;
      typename std::vector<NodeND>::iterator mtb=itb+siz/2;
      if (siz>2)
      {
         // median is at itb+siz/2
         level=level+1;
         if ((pt.xx[dim]-R)>mtb->pt.xx[dim]) // look to the right only
         {
            FindNeighborPoints(pt, R, itb+siz/2+1, ite, level, res);
         }
         else if ((pt.xx[dim]+R)<mtb->pt.xx[dim]) // look to the left only
         {
            FindNeighborPoints(pt,R, itb, itb+siz/2, level,   res);
         }
         else  //check all
         {
            FindNeighborPoints(pt,R, itb+siz/2+1, ite, level, res); // right
            FindNeighborPoints(pt,R, itb, itb+siz/2, level,   res); // left

            // check central one
            Tfloat dd=Dist(mtb->pt, pt);
            if (dd<R)
            {
               res.push_back(mtb->ind);
            }
         }
      }
      else
      {
         Tfloat dd;
         for (auto it=itb; it!=ite; it++)
         {
            dd=Dist(it->pt, pt);
            if (dd<R) // update bc
            {
               res.push_back(it->ind);
            }
         }
      }
   }

   /// Finds the set of indices of points within a distance R of a point pt.
   void FindNeighborPoints(PointND& pt, Tfloat R,
                           typename std::vector<NodeND>::iterator itb,
                           typename std::vector<NodeND>::iterator ite,
                           size_t level,
                           std::vector<Tindex> & res, std::vector<Tfloat> & dist) const
   {
      std::uint8_t dim=(std::uint8_t) (level%ndim);
      size_t siz=ite-itb;
      typename std::vector<NodeND>::iterator mtb=itb+siz/2;
      if (siz>2)
      {
         // median is at itb+siz/2
         level=level+1;
         if ((pt.xx[dim]-R)>mtb->pt.xx[dim]) // look to the right only
         {
            FindNeighborPoints(pt, R, itb+siz/2+1, ite, level, res, dist);
         }
         else if ((pt.xx[dim]+R)<mtb->pt.xx[dim]) // look to the left only
         {
            FindNeighborPoints(pt,R, itb, itb+siz/2, level,   res, dist);
         }
         else  // check all
         {
            FindNeighborPoints(pt,R, itb+siz/2+1, ite, level, res, dist); // right
            FindNeighborPoints(pt,R, itb, itb+siz/2, level,   res, dist); // left

            // check central one
            Tfloat dd=Dist(mtb->pt, pt);
            if (dd<R)
            {
               res.push_back(mtb->ind);
               dist.push_back(dd);
            }
         }
      }
      else
      {
         Tfloat dd;
         for (auto it=itb; it!=ite; it++)
         {
            dd=Dist(it->pt, pt);
            if (dd<R) // update bc
            {
               res.push_back(it->ind);
               dist.push_back(dd);
            }
         }
      }
   }
};

/// Defines KDTree in 3D
typedef KDTree<int,real_t,3> KDTree3D;

/// Defines KDTree in 2D
typedef KDTree<int,real_t,2> KDTree2D;

/// Defines KDTree in 1D
typedef KDTree<int,real_t,1> KDTree1D;

} // namespace mfem

#endif // MFEM_KDTREE_HPP
