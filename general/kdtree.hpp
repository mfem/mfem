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

#ifndef MFEM_KDTREE_HPP
#define MFEM_KDTREE_HPP

#include <vector>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <cmath>

#include "../linalg/vector.hpp"

namespace mfem
{

namespace KDTreeNorms
{

/// Evaluates l1 norm of a vector.
template <typename Tfloat, int ndim>
struct Norm_l1
{
   Tfloat tm;
   Tfloat operator() (const Tfloat* xx)
   {
      if (xx[0]<Tfloat(0.0)) { tm=-xx[0];}
      else { tm=xx[0];}
      for (int i=1; i<ndim; i++)
      {
         if (xx[i]<Tfloat(0.0)) { tm=tm-xx[i]; }
         else { tm=tm+xx[i];}
      }
      return tm;
   }
};

/// Evaluates l2 norm of a vector.
template<typename Tfloat,int ndim>
struct Norm_l2
{
   Tfloat tm;
   Tfloat operator() (const Tfloat* xx)
   {
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
   Tfloat tm;
   Tfloat operator() (const Tfloat* xx)
   {
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

/// Template class for build KDTree with template parameters Tindex
/// specifying the type utilized for indexing the points, Tfloat
/// specifying a float type for representing the coordinates of the
/// points, integer parameter ndim specifying the dimensionality of the
/// space and template function Tnorm for evaluating the distance
/// between two points.
template <typename Tindex, typename Tfloat, size_t ndim=3, typename Tnorm=KDTreeNorms::Norm_l2<Tfloat,ndim> >
class KDTree
{
public:

   /// Structure defining a geometric point in the ndim-dimensional
   /// space.
   struct PointND
   {

      /// Geometric point constructor
      PointND() {for (size_t i=0; i<ndim; i++) {xx[i]=Tfloat(0.0);}}

      /// Coordinates of the point
      Tfloat xx[ndim];
   };

   /// Structure defining a node in the KDTree.
   struct NodeND
   {

      /// Defines a point in the ndim-dimensional space
      PointND pt;

      /// Defines the attached index
      Tindex  ind;
   };

   /// Default constructor
   KDTree() {}

   /// Returns the spatial dimension of the points
   int SpatialDimension() {return ndim;}

   /// Data iterator
   typedef typename std::vector<NodeND>::iterator iterator;

   /// Returns iterator to beginning of the point cloud
   iterator begin() {return data.begin();}

   /// Returns iterator to the end of the point cloud
   iterator end() {return data.end();}

   /// Returns the size of the point cloud
   size_t size() {return data.size();}

   /// Clears the point cloud
   void clear() { data.clear();}

   /// Builds the KDTree. If the point cloud is modified the tree
   /// needs to be rebuild by a new call to Sort().
   void Sort() { SortInPlace(data.begin(),data.end(),0);}

   /// Adds a new node to the point cloud
   void AddPoint(PointND& pt, Tindex ii)
   {
      NodeND nd;
      nd.pt=pt;
      nd.ind=ii;
      data.push_back(nd);
   }

   /// Adds a new node by 3 coordinates and an associated index
   void AddPoint(Tfloat x, Tfloat y, Tfloat z, Tindex ii)
   {
      MFEM_ASSERT(ndim==3,"The spatial dimension for the KDTree should be 3!")
      NodeND nd;
      nd.pt.xx[0]=x;  nd.pt.xx[1]=y; nd.pt.xx[2]=z; nd.ind=ii;
      data.push_back(nd);
   }

   /// Adds a new node by 2 coordinates and an associated index
   void AddPoint(Tfloat x, Tfloat y, Tindex ii)
   {
      MFEM_ASSERT(ndim==2,"The spatial dimension for the KDTree should be 2!")
      NodeND nd;
      nd.pt.xx[0]=x;  nd.pt.xx[1]=y; nd.ind=ii;
      data.push_back(nd);
   }

   /// Adds a new node by 1 coordinate and an associated index
   void AddPoint(Tfloat x, Tindex ii)
   {
      MFEM_ASSERT(ndim==1,"The spatial dimension for the KDTree should be 1!")
      NodeND nd;
      nd.pt.xx[0]=x; nd.ind=ii;
      data.push_back(nd);
   }

   /// Finds the nearest neighbour index
   Tindex FindClosestPoint(PointND& pt)
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

   /// Finds the nearest neighbour index and return the clossest poitn in clp
   Tindex FindClosestPoint(PointND& pt, PointND& clp)
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
   void FindClosestPoint(PointND& pt, Tindex& ind, Tfloat& dist)
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
   }

   /// Returns the closest point and the distance to the input point pt.
   void FindClosestPoint(PointND& pt, Tindex& ind, Tfloat& dist,  PointND& clp)
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
   void rFindClosestPoint(PointND& pt, Tindex& ind, Tfloat& dist)
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
   /// returned in the vestor res and the correponding distances in vector dist.
   void FindNeighborPoints(PointND& pt,Tfloat R, std::vector<Tindex> & res,
                           std::vector<Tfloat> & dist)
   {
      FindNeighborPoints(pt,R,data.begin(),data.end(),0,res,dist);
   }

   /// Finds all points within a distance R from point pt. The indices are
   /// returned in the vestor res and the correponding distances in vector dist.
   void FindNeighborPoints(PointND& pt,Tfloat R, std::vector<Tindex> & res)
   {
      FindNeighborPoints(pt,R,data.begin(),data.end(),0,res);
   }

   /// Brute force search - please, use it only for debuging purposes
   void _FindNeighborPoints(PointND& pt,Tfloat R, std::vector<Tindex> & res,
                            std::vector<Tfloat> & dist)
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
   void _FindNeighborPoints(PointND& pt,Tfloat R, std::vector<Tindex> & res)
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

   /// Point for storing tmp data
   PointND tp;
   Tnorm fnorm;

   /// Computes the distance between two nodes
   Tfloat Dist(const PointND& pt1,const  PointND& pt2)
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
   void PSearch(typename std::vector<NodeND>::iterator itb,
                typename std::vector<NodeND>::iterator ite,
                size_t level, PointS& bc)
   {
      std::uint8_t dim=(std::uint8_t) (level%ndim);
      size_t siz=ite-itb;
      typename std::vector<NodeND>::iterator mtb=itb+siz/2;
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
                  }// end central point check
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
            std::vector< std::tuple<Tfloat,Tindex> > & res)
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
                           std::vector<Tindex> & res)
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
            RNS(pt, R, itb+siz/2+1, ite, level, res);
         }
         else if ((pt.xx[dim]+R)<mtb->pt.xx[dim]) // look to the left only
         {
            RNS(pt,R, itb, itb+siz/2, level,   res);
         }
         else  //check all
         {
            RNS(pt,R, itb+siz/2+1, ite, level, res); // right
            RNS(pt,R, itb, itb+siz/2, level,   res); // left

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
            if (dd<R) //update bc
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
                           std::vector<Tindex> & res, std::vector<Tfloat> & dist)
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
            RNS(pt, R, itb+siz/2+1, ite, level, res, dist);
         }
         else if ((pt.xx[dim]+R)<mtb->pt.xx[dim]) // look to the left only
         {
            RNS(pt,R, itb, itb+siz/2, level,   res, dist);
         }
         else  // check all
         {
            RNS(pt,R, itb+siz/2+1, ite, level, res, dist); // right
            RNS(pt,R, itb, itb+siz/2, level,   res, dist); // left

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
typedef KDTree<int,double,3> KDTree3D;

/// Defines Contructs KDTree in 2D
typedef KDTree<int,double,2> KDTree2D;

/// Defines KDTree in 1D
typedef KDTree<int,double,1> KDTree1D;


/// Defines KDTree with points stored outside the class.
/// The external storage slows down the search and the construction
/// methods.
class KDTreeM
{
public:
   /// Construtor
   KDTreeM(mfem::Vector& coords_,int dim_)
   {
      gf=&coords_;
      ndim=dim_;
      tp.SetSize(ndim);
      int np=(gf->Size())/ndim;
      data.resize(np);
      for (int i=0; i<np; i++)
      {
         data[i].xx=coords_.GetData()+i*ndim;
         data[i].ind=i;
      }
      Sort();
   }

   /// Internal structure used as a node in the KDTree
   struct NodeND
   {
      size_t ind;
      double *xx;//coordinates of the node
   };


   /// Returns the spatial dimension of the points
   int SpatialDimension() {return ndim;}

   /// Data iterator
   typedef typename std::vector<NodeND>::iterator iterator;

   /// Returns iterator to beginning of the point cloud
   iterator begin() {return data.begin();}

   /// Returns iterator to the end of the point cloud
   iterator end() {return data.end();}

   /// Returns the size of the point cloud
   size_t size() {return data.size();}

   /// Clears the point cloud.
   void clear() { data.clear();}

   /// Builds the KDTree.
   void Sort() { SortInPlace(data.begin(),data.end(),0);}

   /// Returns the index of the closest point to the input point pt.
   int FindClosestPoint(mfem::Vector& pt);

   ///  Returns the closest point and the distance to the input point pt.
   void FindClosestPoint(Vector& pt, int &ind, double& dist);

   /// Brute force search - please, use it only for debuging purposes.
   void _FindClosestPoint(Vector& pt,int& ind, double& dist);

private:
   mfem::Vector* gf;
   int ndim;
   Vector tp;

   /// The actual data in the KDTree.
   std::vector<NodeND> data;

   /// Structure utilized for nearest neighbor search (NNS)
   struct PointS
   {
      double dist;
      size_t pos;
      size_t level;
      NodeND sp; //index of the point
   };

   /// Functor utilized in the coordinate comparison
   /// for building the KDTree
   struct CompN
   {
      /// Current coordinate index
      std::uint8_t dim;

      /// Constructor for the comparison
      CompN(std::uint8_t dd):dim(dd) {}

      /// Compares two points p1 and p2
      bool operator() (const NodeND& p1, const NodeND& p2)
      {
         return p1.xx[dim]<p2.xx[dim];
      }
   };

   /// Computes the distance between two points
   double Dist(const NodeND& p1, const NodeND& p2)
   {
      for (int i=0; i<ndim; i++)
      {
         tp(i)=p1.xx[i]-p2.xx[i];
      }
      return tp.Norml2();
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

   /// Finds the closest point to bc.sp in the point cloud
   /// bounded between [itb,ite).
   void PSearch(typename std::vector<NodeND>::iterator itb,
                typename std::vector<NodeND>::iterator ite,
                size_t level, PointS& bc);

};

}
#endif
