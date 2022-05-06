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

#include "mfem.hpp"

namespace mfem {


template <typename Tindex, typename Tfloat, size_t ndim=3>
class KDTree{
public:

    /// Structure defining a geometric point in the ndim-dimensional
    /// space.
    struct PointND{

        /// Geometric point constructor
        PointND(){for(size_t i=0;i<ndim;i++){xx[i]=Tfloat(0.0);}}

        /// Coordinates of the point
        Tfloat xx[ndim];
    };

    /// Structure defining a node in the KDTree.
    struct NodeND{

        /// Defines a point in the ndim-dimensional space
        PointND pt;

        /// Defines the attached index
        Tindex  ind;
    };

    /// Default constructor
    KDTree(){}

    /// Returns the spatial dimension of the points
    int SpatialDimension(){return ndim;}

    /// Data iterator
    typedef typename std::vector<NodeND>::iterator iterator;

    /// Returns iterator to beginning of the point cloud
    iterator begin(){return data.begin();}

    /// Returns iterator to the end of the point cloud
    iterator end(){return data.end();}

    /// Returns the size of the point cloud
    size_t size(){return data.size();}

    /// Clears the point cloud
    void clear(){ data.clear();}

    /// Builds the KDTree. If the point cloud is modified the tree
    /// needs to be rebuild by a new call to Sort().
    void Sort(){ SortInPlace(data.begin(),data.end(),0);}

    /// Adds a new node to the point cloud
    void AddNode(PointND& pt, Tindex ii){
        NodeND nd;
        nd.pt=pt;
        nd.ind=ii;
        data.push_back(nd);
    }

    /// Adds a new node by 3 coordinates and an associated index
    void AddNode(Tfloat x, Tfloat y, Tfloat z, Tindex ii)
    {
        NodeND nd;
        nd.pt.xx[0]=x;  nd.pt.xx[1]=y; nd.pt.xx[2]=z; nd.ind=ii;
        data.push_back(nd);
    }

    /// Adds a new node by 2 coordinates and an associated index
    void AddNode(Tfloat x, Tfloat y, Tindex ii)
    {
        NodeND nd;
        nd.pt.xx[0]=x;  nd.pt.xx[1]=y; nd.ind=ii;
        data.push_back(nd);
    }

    /// Adds a new node by 1 coordinate and an associated index
    void AddNode(Tfloat x, Tindex ii)
    {
        NodeND nd;
        nd.pt.xx[0]=x; nd.ind=ii;
        data.push_back(nd);
    }

    /// Finds the nearest neighbour index
    Tindex NNS(PointND& pt)
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

    /// Returns the closest point and the distance to the input point pt.
    void NNS(PointND& pt, Tindex& ind, Tfloat& dist)
    {
        PointS best_candidate;
        best_candidate.sp=pt;
        //initialize the best candidate
        best_candidate.pos =0;
        best_candidate.dist=Dist(data[0].pt, best_candidate.sp);
        best_candidate.level=0;
        PSearch(data.begin(), data.end(), 0, best_candidate);

        ind=data[best_candidate.pos].ind;
        dist=Dist(data[best_candidate.pos].pt, best_candidate.sp);
    }

    /// Brute force search - please, use it only for debuging purposes
    void rNNS(PointND& pt, Tindex& ind, Tfloat& dist)
    {
        PointS best_candidate;
        best_candidate.sp=pt;
        //initialize the best candidate
        best_candidate.pos =0;
        best_candidate.dist=Dist(data[0].pt, best_candidate.sp);
        Tfloat dd;
        for(auto iti=data.begin()+1; iti!=data.end();iti++){
            dd=Dist(iti->pt,  best_candidate.sp);
            if(dd<best_candidate.dist){
                best_candidate.pos=iti-data.begin();
                best_candidate.dist=dd;
            }
        }

        ind=data[best_candidate.pos].ind;
        dist=best_candidate.dist;
    }

private:

    /// Functor utilized in the coordinate comparison
    /// for building the KDTree
    struct CompN{
        /// Current coordinate index
        std::uint8_t dim;

        /// Constructor for the comparison
        CompN(std::uint8_t dd):dim(dd){}

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

    /// Computes the distance between two nodes
    Tfloat Dist(const PointND& pt1,const  PointND& pt2){
        Tfloat d=0.0;
        for(size_t i=0;i<ndim;i++){
            d=d+(pt1.xx[i]-pt2.xx[i])*(pt1.xx[i]-pt2.xx[i]);
        }
        return std::sqrt(d);
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
                     size_t level){
        std::uint8_t cdim=(std::uint8_t)(level%ndim);
        size_t siz=ite-itb;
        if(siz>2){
            std::nth_element(itb, itb+siz/2, ite, CompN(cdim));
            level=level+1;
            SortInPlace(itb, itb+siz/2, level);
            SortInPlace(itb+siz/2+1,ite, level);
        }
    }

    /// Structure utilized for nearest neighbor search (NNS)
    struct PointS{
        double dist;
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
        if(siz>2){
            //median is at itb+siz/2
            level=level+1;
            if((bc.sp.xx[dim]-bc.dist)>mtb->pt.xx[dim]){//look on the right only
                PSearch(itb+siz/2+1, ite, level, bc);
            }else
            if((bc.sp.xx[dim]+bc.dist)<mtb->pt.xx[dim]){ //look on the left only
                PSearch(itb,itb+siz/2, level, bc);
            }else{//check all
                if(bc.sp.xx[dim]<mtb->pt.xx[dim]){
                    //start with the left portion
                    PSearch(itb,itb+siz/2, level, bc);
                    //and continue to the right
                    if(!((bc.sp.xx[dim]+bc.dist)<mtb->pt.xx[dim])){
                        PSearch(itb+siz/2+1, ite, level, bc);
                        {//check central one
                             Tfloat dd=Dist(mtb->pt, bc.sp);
                             if(dd<bc.dist){ bc.dist=dd; bc.pos=mtb-data.begin(); bc.level=level; }
                        }//end central point check
                    }
                }else{
                    //start with the right portion
                    PSearch(itb+siz/2+1, ite, level, bc);
                    //and continue with left
                    if(!((bc.sp.xx[dim]-bc.dist)>mtb->pt.xx[dim])){
                        PSearch(itb, itb+siz/2, level, bc);
                        {//check central one
                            double dd=Dist(mtb->pt, bc.sp);
                            if(dd<bc.dist){ bc.dist=dd; bc.pos=mtb-data.begin(); bc.level=level; }
                        }//end central point check
                    }
                }
            }
        }else{
            //check the nodes
            Tfloat dd;
            for(auto it=itb; it!=ite; it++){
                dd=Dist(it->pt, bc.sp);
                if(dd<bc.dist){//update bc
                    bc.pos=it-data.begin();
                    bc.dist=dd;
                    bc.level=level;
                }
            }
        }
    }

};

typedef KDTree<int,double,3> KDTree3D;
typedef KDTree<int,double,2> KDTree2D;
typedef KDTree<int,double,1> KDTree1D;

/// Constructs KDTree from the nodes of the mesh.
KDTree3D* BuildKDTree3D(Mesh* mesh);
KDTree2D* BuildKDTree2D(Mesh* mesh);
KDTree1D* BuildKDTree1D(Mesh* mesh);


};
#endif
