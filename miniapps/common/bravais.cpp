// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#include "../../config/config.hpp"

#include "bravais.hpp"

#include <fstream>

using namespace std;

namespace mfem
{

namespace bravais
{

BravaisLattice::BravaisLattice(unsigned int dim)
   : T_(dim),
     dim_(dim),
     vol_(0.0),
     bz_vol_(0.0)
{
   label_ = "";
   type_  = INVALID_TYPE;

   lat_vecs_.resize(dim_);
   rec_vecs_.resize(dim_);
   for (unsigned int i=0; i<dim_; i++)
   {
      lat_vecs_[i].SetSize(dim_);
      rec_vecs_[i].SetSize(dim_);
   }
}

void
BravaisLattice::SetVectorSizes()
{
   sp_.resize(this->GetNumberSymmetryPoints());
   sl_.resize(this->GetNumberSymmetryPoints());
   for (unsigned int i=0; i<sp_.size(); i++)
   {
      sp_[i].SetSize(dim_);
   }

   path_.resize(this->GetNumberPaths());
   for (unsigned int i=0; i<path_.size(); i++)
   {
      path_[i].resize(this->GetNumberPathSegments(i)+1);
   }
}

void
BravaisLattice::SetIntermediatePoints()
{
   ip_.resize(path_.size());
   il_.resize(path_.size());
   for (unsigned int p=0; p<path_.size(); p++)
   {
      ip_[p].resize(path_[p].size()-1);
      il_[p].resize(path_[p].size()-1);
      for (unsigned int s=0; s<path_[p].size()-1; s++)
      {
         ip_[p][s].SetSize(dim_);
         add(0.5,sp_[path_[p][s]],0.5,sp_[path_[p][s+1]],ip_[p][s]);
      }
   }
}

void
BravaisLattice::SetCellVolumes()
{
   vol_    = this->ComputeCellVolume(lat_vecs_);
   bz_vol_ = this->ComputeCellVolume(rec_vecs_);
}

double
BravaisLattice::ComputeCellVolume(const vector<Vector> & vecs)
{
   double vol = 0.0;
   switch (dim_)
   {
      case 1:
         // Compute the vector norm in 1D
         vol = vecs[0][0]; // assuming displacement vector is positive
         break;
      case 2:
         // Compute the scalar cross product in 2D
         vol = vecs[0][0]*vecs[1][1] - vecs[0][1]*vecs[1][0];
         break;
      case 3:
         // Compute the vector triple product
         vol =
            vecs[0][0] * (vecs[1][1]*vecs[2][2] - vecs[1][2]*vecs[2][1]) +
            vecs[0][1] * (vecs[1][2]*vecs[2][0] - vecs[1][0]*vecs[2][2]) +
            vecs[0][2] * (vecs[1][0]*vecs[2][1] - vecs[1][1]*vecs[2][0]);
         break;
      default:
         vol = 0.0;
   }
   return vol;
}

void
BravaisLattice::GetLatticeVectors(vector<Vector> & a) const
{
   a.resize(lat_vecs_.size());
   for (unsigned int i=0; i<lat_vecs_.size(); i++)
   {
      a[i] = lat_vecs_[i];
   }
}

void
BravaisLattice::GetReciprocalLatticeVectors(vector<Vector> & b) const
{
   b.resize(rec_vecs_.size());
   for (unsigned int i=0; i<rec_vecs_.size(); i++)
   {
      b[i] = rec_vecs_[i];
   }
}

void
BravaisLattice::GetTranslationVectors(vector<Vector> & t) const
{
   t.resize(trn_vecs_.size());
   for (unsigned int i=0; i<trn_vecs_.size(); i++)
   {
      t[i] = trn_vecs_[i];
   }
}

void
BravaisLattice::GetFaceRadii(vector<double> & r) const
{
   r.resize(face_radii_.size());
   for (unsigned int i=0; i<face_radii_.size(); i++)
   {
      r[i] = face_radii_[i];
   }
}

/** The Primitive Cell is the unique unit cell which is centered on
    the lattice point which is at the origin.  We can compute this
    mapping by minimizing ||pt - A.n|| over n.  Where n is a vector of
    integers and A is the matrix of lattice vectors.  This minimum may
    not be unique.  The inverse of A is the transposed matrix of
    reciprocal vectors B.  So, we can estimate n using B^T.pt as a
    starting point.  Then try all integer vectors in the immediate
    neighborhood of the real vector B^T.pt.
 */
bool
BravaisLattice::MapToPrimitiveCell(const Vector & pt, Vector & ipt) const
{
   bool map = false;
   ipt = pt;

   Vector v(dim_);
   Array<int> npt(2*dim_);

   for (unsigned int i=0; i<dim_; i++)
   {
      // Compute B^T.pt
      v[i] = rec_vecs_[i] * pt;

      // Grab the integer values in the neighborhood of B^T.pt
      npt[2*i+0] = (int)floor(v[i]);
      npt[2*i+1] =  (int)ceil(v[i]);
   }

   double pmin = pt.Norml2();

   // Search for the minimum ||pt - A.n||
   for (int j=0; j<(1<<dim_); j++)
   {
      bool m = true;
      v = pt;
      for (unsigned int i=0; i<dim_; i++)
      {
         v.Add((double)npt[2*i+((j>>i)%2)], lat_vecs_[i]);
         m = m && (npt[2*i+((j>>i)%2)] == 0);
      }
      double nrm = v.Norml2();
      if ( nrm < pmin )
      {
         ipt  = v;
         pmin = nrm;
         map  = m;
      }
   }
   return map;
}

void
BravaisLattice::GetSymmetryPoint(int i, Vector & pt)
{
   pt.SetSize(sp_[i].Size());
   pt.Set(2.0 * M_PI, sp_[i]);
}

string &
BravaisLattice::GetSymmetryPointLabel(int i)
{
   return sl_[i];
}

int
BravaisLattice::GetSymmetryPointIndex(const string & label)
{
   map<string,int>::const_iterator mit = si_.find(label);
   if ( mit!=si_.end() )
   {
      return mit->second;
   }
   else
   {
      return -1;
   }
}

void
BravaisLattice::GetIntermediatePoint(int p, int s, Vector & pt)
{
   pt.SetSize(ip_[p][s].Size());
   pt.Set(2.0 * M_PI, ip_[p][s]);
}

string &
BravaisLattice::GetIntermediatePointLabel(int p, int s)
{
   return il_[p][s];
}

void
BravaisLattice::GetPathSegmentEndPointIndices(int p, int s,
                                              int & e0, int & e1)
{
   e0 = path_[p][s];
   e1 = path_[p][s+1];
}

mfem::Mesh * BravaisLattice::GetWignerSeitzMesh(bool) const
{
   unsigned int ntrans = this->GetNumberTransformations();

   Mesh * fd_mesh = this->GetFundamentalDomainMesh();

   int nverts = fd_mesh->GetNV();
   int nelems = fd_mesh->GetNE();

   Array<int> vi0;
   Array<int> vi1;
   double vdata[3];
   Vector v1(vdata, dim_);

   Mesh * mesh = new Mesh(dim_, nverts * ntrans, nelems * ntrans);

   for (unsigned int i=0; i<ntrans; i++)
   {
      const DenseMatrix & T = this->GetTransformation(i);

      double detT = T.Det();

      for (int j=0; j<nverts; j++)
      {
         Vector v0(fd_mesh->GetVertex(j), dim_);
         T.Mult(v0, v1);

         mesh->AddVertex(v1);
      }

      for (int j=0; j<nelems; j++)
      {
         const Element * el = fd_mesh->GetElement(j);
         int geom = el->GetGeometryType();

         el->GetVertices(vi0);

         vi1.SetSize(vi0.Size());
         for (int k=0; k<vi0.Size(); k++)
         {
            vi1[k] = vi0[k] + i * nverts;
         }

         if ( detT < 0.0 )
         {
            switch (geom)
            {
               case Geometry::TRIANGLE:
                  std::swap(vi1[1], vi1[2]);
                  break;
               case Geometry::SQUARE:
                  std::swap(vi1[1], vi1[3]);
                  break;
               case Geometry::TETRAHEDRON:
                  std::swap(vi1[1], vi1[2]);
                  break;
               case Geometry::PYRAMID:
                  std::swap(vi1[1], vi1[3]);
                  break;
               case Geometry::PRISM:
                  std::swap(vi1[1], vi1[2]);
                  std::swap(vi1[4], vi1[5]);
                  break;
               case Geometry::CUBE:
                  std::swap(vi1[1], vi1[3]);
                  std::swap(vi1[5], vi1[7]);
                  break;
            }
         }

         Element * new_elem = mesh->NewElement(geom);
         new_elem->SetVertices(vi1);
         new_elem->SetAttribute(i+1);
         mesh->AddElement(new_elem);
      }
   }
   mesh->GenerateBoundaryElements();
   mesh->FinalizeMesh(true);

   MergeMeshNodes(mesh);

   delete fd_mesh;

   return mesh;
}

Mesh *
BravaisLattice::GetPeriodicWignerSeitzMesh(bool triMesh) const
{
   Mesh * mesh = this->GetWignerSeitzMesh(triMesh);

   mesh->UniformRefinement();

   Mesh * per_mesh = MakePeriodicMesh(mesh, trn_vecs_);

   delete mesh;

   return per_mesh;
}

BravaisLattice1D::BravaisLattice1D(double a)
   : BravaisLattice(1)
   , a_(a)
{}

void
BravaisLattice1D::GetAxialLength(double &a)
{ a = a_; }

BravaisLattice2D::BravaisLattice2D(double a, double b, double gamma)
   : BravaisLattice(2)
   , a_(a)
   , b_(b)
   , gamma_(gamma)
{}

void
BravaisLattice2D::GetAxialLengths(double &a, double &b)
{ a = a_; b = b_; }

void
BravaisLattice2D::GetInteraxialAngle(double &gamma)
{ gamma = gamma_; }

BravaisLattice3D::BravaisLattice3D(double a, double b, double c,
                                   double alpha, double beta, double gamma)
   : BravaisLattice(3)
   , a_(a)
   , b_(b)
   , c_(c)
   , alpha_(alpha)
   , beta_(beta)
   , gamma_(gamma)
{}

void
BravaisLattice3D::GetAxialLengths(double &a, double &b, double &c)
{ a = a_; b = b_; c = c_; }

void
BravaisLattice3D::GetInteraxialAngles(double &alpha,
                                      double &beta,
                                      double &gamma)
{ alpha = alpha_; beta = beta_; gamma = gamma_; }


LinearLattice::LinearLattice(double a)
   : BravaisLattice1D(a)
{
   label_ = "LIN";
   type_  = PRIMITIVE_SEGMENT;

   // Set Lattice Vectors
   lat_vecs_[0][0] = 1.0;

   // Set Reciprocal Lattice Vectors
   rec_vecs_[0][0] = 1.0;

   // Set Translation Vectors
   trn_vecs_.resize(1);
   for (unsigned int i=0; i<trn_vecs_.size(); i++)
   {
      trn_vecs_[i].SetSize(1);
   }
   trn_vecs_[0][0] = 1.0;

   for (unsigned int i=0; i<lat_vecs_.size(); i++)
   {
      lat_vecs_[i] *= a_;
      rec_vecs_[i] *= 1.0/a_;
      trn_vecs_[i] *= a_;
   }

   // Set the face radii
   face_radii_.resize(1);
   for (int i=0; i<1; i++) { face_radii_[i] = 0.0; }

   this->SetCellVolumes();

   // Allocate sl_, sp_, and path_ vectors
   this->SetVectorSizes();

   // Set Symmetry Points and Labels
   sl_[0] = "Gamma";
   sp_[0] = 0.0;

   sl_[1] = "X";
   sp_[1].Set(0.5,rec_vecs_[0]);

   for (unsigned int i=0; i<sl_.size(); i++)
   {
      si_[sl_[i]] = i;
   }

   // Define Paths
   path_[0][0] = 0;
   path_[0][1] = 1;

   // Set Intermediate Symmetry Points
   this->SetIntermediatePoints();

   // Set Intermediate Symmetry Point Labels
   il_[0][0] = "Delta";  // Gamma -> X

   // Set Mesh data
   fd_vert_[0] =  0.0;      fd_vert_[ 1] = 0.0; fd_vert_[ 2] = 0.0;
   fd_vert_[3] =  0.5 * a_; fd_vert_[ 4] = 0.0; fd_vert_[ 5] = 0.0;

   fd_e2v_[0] = 0; fd_e2v_[1] = 1;
   fd_elem_att_[0] = 1;

   fd_be2v_[0] = 0; fd_be2v_[1] = 1;

   fd_belem_att_[0] = 10; fd_belem_att_[1] = 1;
   /*
   ws_vert_[0] = -0.5 * a_; ws_vert_[ 1] = 0.0; ws_vert_[ 2] = 0.0;
   ws_vert_[3] =  0.5 * a_; ws_vert_[ 4] = 0.0; ws_vert_[ 5] = 0.0;

   ws_e2v_[0] = 0; ws_e2v_[1] = 1;
   ws_elem_att_[0] = 1;

   ws_be2v_[0] = 0; ws_be2v_[1] = 1;

   ws_belem_att_[0] = 1; ws_belem_att_[1] = 1;
   */
}

bool
LinearLattice::MapToFundamentalDomain(const Vector & pt, Vector & ipt) const
{
   bool map = false;
   ipt = pt;
   if ( ipt[0] < 0.0 )
   {
      ipt[0] *= -1.0;
      map = true;
   }
   return map;
}

const DenseMatrix & LinearLattice::GetTransformation(int ti) const
{
   T_(0,0) = (ti == 0) ? 1.0 : -1.0;
   return T_;
}

Mesh *
LinearLattice::GetFundamentalDomainMesh() const
{
   Mesh * mesh = new Mesh((double*)fd_vert_, 2,
                          (int*)fd_e2v_, Geometry::SEGMENT,
                          (int*)fd_elem_att_, 1,
                          (int*)fd_be2v_, Geometry::POINT,
                          (int*)fd_belem_att_, 2,
                          1, 1);
   mesh->Finalize();

   return mesh;
}
/*
Mesh *
LinearLattice::GetWignerSeitzMesh(bool triMesh) const
{
   Mesh * mesh = new Mesh((double*)ws_vert_, 2,
                          (int*)ws_e2v_, Geometry::SEGMENT,
                          (int*)ws_elem_att_, 1,
                          (int*)ws_be2v_, Geometry::POINT,
                          (int*)ws_belem_att_, 2,
                          1, 1);
   mesh->Finalize();

   return mesh;
}

Mesh *
LinearLattice::GetPeriodicWignerSeitzMesh(bool triMesh) const
{
   Mesh * mesh = this->GetWignerSeitzMesh(triMesh);

   mesh->UniformRefinement();
   mesh->UniformRefinement();

   Mesh * per_mesh = MakePeriodicMesh(mesh, trn_vecs_);

   delete mesh;

   return per_mesh;
}
*/
SquareLattice::SquareLattice(double a)
   : BravaisLattice2D(a, a, 0.5 * M_PI)
{
   label_ = "SQR";
   type_  = PRIMITIVE_SQUARE;

   // Set Lattice Vectors
   lat_vecs_[0][0] = 1.0; lat_vecs_[0][1] = 0.0;
   lat_vecs_[1][0] = 0.0; lat_vecs_[1][1] = 1.0;

   // Set Reciprocal Lattice Vectors
   rec_vecs_[0][0] = 1.0; rec_vecs_[0][1] = 0.0;
   rec_vecs_[1][0] = 0.0; rec_vecs_[1][1] = 1.0;

   // Set Translation Vectors
   trn_vecs_.resize(2);
   for (unsigned int i=0; i<trn_vecs_.size(); i++)
   {
      trn_vecs_[i].SetSize(2);
   }
   trn_vecs_[0][0] = 1.0; trn_vecs_[0][1] = 0.0;
   trn_vecs_[1][0] = 0.0; trn_vecs_[1][1] = 1.0;

   for (unsigned int i=0; i<lat_vecs_.size(); i++)
   {
      lat_vecs_[i] *= a_;
      rec_vecs_[i] *= 1.0/a_;
      trn_vecs_[i] *= a_;
   }

   // Set the face radii
   face_radii_.resize(2);
   for (int i=0; i<2; i++) { face_radii_[i] = 0.5 * a_; }

   this->SetCellVolumes();

   // Allocate sl_, sp_, and path_ vectors
   this->SetVectorSizes();

   // Set Symmetry Points and Labels
   sl_[0] = "Gamma";
   sp_[0] = 0.0;

   sl_[1] = "X";
   sp_[1].Set(0.5,rec_vecs_[0]);

   sl_[2] = "M";
   add(0.5,rec_vecs_[0],0.5,rec_vecs_[1],sp_[2]);

   for (unsigned int i=0; i<sl_.size(); i++)
   {
      si_[sl_[i]] = i;
   }

   // Define Paths
   path_[0][0] = 0;
   path_[0][1] = 1;
   path_[0][2] = 2;
   path_[0][3] = 0;

   // Set Intermediate Symmetry Points
   this->SetIntermediatePoints();

   // Set Intermediate Symmetry Point Labels
   il_[0][0] = "Delta";  // Gamma -> X
   il_[0][1] = "Z";      // X     -> M
   il_[0][2] = "Sigma";  // M     -> Gamma

   // Set Mesh data
   fd_vert_[0] = 0.0;      fd_vert_[ 1] = 0.0;      fd_vert_[ 2] = 0.0;
   fd_vert_[3] = 0.5 * a_; fd_vert_[ 4] = 0.0;      fd_vert_[ 5] = 0.0;
   fd_vert_[6] = 0.5 * a_; fd_vert_[ 7] = 0.5 * a_; fd_vert_[ 8] = 0.0;

   fd_e2v_[0] = 0; fd_e2v_[1] = 1; fd_e2v_[2] = 2;
   fd_elem_att_[0] = 1;

   fd_be2v_[0] = 0; fd_be2v_[1] = 1;
   fd_be2v_[2] = 1; fd_be2v_[3] = 2;
   fd_be2v_[4] = 2; fd_be2v_[5] = 0;

   fd_belem_att_[0] = 10; fd_belem_att_[1] = 1;
   fd_belem_att_[2] = 10;
   /*
   ws_vert_[0] = -0.5 * a_; ws_vert_[ 1] = -0.5 * a_; ws_vert_[ 2] = 0.0;
   ws_vert_[3] =  0.5 * a_; ws_vert_[ 4] = -0.5 * a_; ws_vert_[ 5] = 0.0;
   ws_vert_[6] =  0.5 * a_; ws_vert_[ 7] =  0.5 * a_; ws_vert_[ 8] = 0.0;
   ws_vert_[9] = -0.5 * a_; ws_vert_[10] =  0.5 * a_; ws_vert_[11] = 0.0;

   ws_e2v_[0] = 0; ws_e2v_[1] = 1; ws_e2v_[2] = 2; ws_e2v_[3] = 3;
   ws_elem_att_[0] = 1;

   ws_be2v_[0] = 0; ws_be2v_[1] = 1;
   ws_be2v_[2] = 1; ws_be2v_[3] = 2;
   ws_be2v_[4] = 2; ws_be2v_[5] = 3;
   ws_be2v_[6] = 3; ws_be2v_[7] = 0;

   ws_belem_att_[0] = 1; ws_belem_att_[1] = 1;
   ws_belem_att_[2] = 1; ws_belem_att_[3] = 1;
   */
}

bool
SquareLattice::MapToFundamentalDomain(const Vector & pt, Vector & ipt) const
{
   bool map = false;
   ipt = pt;
   if ( ipt[1] < 0.0 )
   {
      ipt[1] *= -1.0;
      map = true;
   }
   if ( ipt[0] < 0.0 )
   {
      ipt[0] *= -1.0;
      map = true;
   }
   if ( ipt[0] > ipt[1] )
   {
      double d = ipt[0];
      ipt[0] = ipt[1];
      ipt[1] = d;
      map = true;
   }
   return map;
}

const DenseMatrix & SquareLattice::GetTransformation(int ti) const
{
   int ir = ti % 2;
   int iq = ti / 2;

   T_ = 0.0;

   if ( ir % 2 == 0 )
   {
      T_(0, 0) = 1.0;
      T_(1, 1) = 1.0;
   }
   else
   {
      T_(0, 1) = 1.0;
      T_(1, 0) = 1.0;
   }

   for (int i=0; i<2; i++)
   {
      if ( iq & (int)pow(2, i) )
      {
         T_(i,0) *= -1.0;
         T_(i,1) *= -1.0;
      }
   }

   return T_;
}

Mesh *
SquareLattice::GetFundamentalDomainMesh() const
{
   Mesh * mesh = new Mesh((double*)fd_vert_, 3,
                          (int*)fd_e2v_, Geometry::TRIANGLE,
                          (int*)fd_elem_att_, 1,
                          (int*)fd_be2v_, Geometry::SEGMENT,
                          (int*)fd_belem_att_, 3,
                          2, 2);
   mesh->Finalize();

   return mesh;
}
/*
Mesh *
SquareLattice::GetWignerSeitzMesh(bool triMesh) const
{
 Mesh * mesh = new Mesh((double*)ws_vert_, 4,
                        (int*)ws_e2v_, Geometry::SQUARE,
                        (int*)ws_elem_att_, 1,
                        (int*)ws_be2v_, Geometry::SEGMENT,
                        (int*)ws_belem_att_, 4,
                        2, 2);
 mesh->Finalize();

 return mesh;
}

Mesh *
SquareLattice::GetPeriodicWignerSeitzMesh(bool triMesh) const
{
 Mesh * mesh = this->GetWignerSeitzMesh(triMesh);

 mesh->UniformRefinement();
 mesh->UniformRefinement();

 Mesh * per_mesh = MakePeriodicMesh(mesh, trn_vecs_);

 delete mesh;

 return per_mesh;
}
*/
HexagonalLattice::HexagonalLattice(double a)
   : BravaisLattice2D(a, a, 2.0 * M_PI / 3.0)//, R60_(2), RX_(2), TTmp_(2)
{
   label_ = "HEX2D";
   type_  = PRIMITIVE_HEXAGONAL;

   // Set Lattice Vectors
   lat_vecs_[0][0] =  0.5 * a_;
   lat_vecs_[0][1] = -sqrt(0.75) * a_;

   lat_vecs_[1][0] =  0.5 * a_;
   lat_vecs_[1][1] =  sqrt(0.75) * a_;

   // Set Reciprocal Lattice Vectors
   rec_vecs_[0][0] =  1.0 / a_;
   rec_vecs_[0][1] = -1.0 / ( sqrt(3.0) * a_ );

   rec_vecs_[1][0] =  1.0 / a_;
   rec_vecs_[1][1] =  1.0 / ( sqrt(3.0) * a_ );

   // Set the Translation Vectors
   trn_vecs_.resize(3);
   for (unsigned int i=0; i<trn_vecs_.size(); i++)
   {
      trn_vecs_[i].SetSize(2);
   }
   trn_vecs_[0][0] =  1.0 * a; trn_vecs_[0][1] =  0.0;
   trn_vecs_[1][0] =  0.5 * a; trn_vecs_[1][1] =  sqrt(0.75) * a;
   trn_vecs_[2][0] =  0.5 * a; trn_vecs_[2][1] = -sqrt(0.75) * a;

   // Set the face radii
   face_radii_.resize(3);
   for (int i=0; i<3; i++) { face_radii_[i] = 0.5 * a_ / sqrt(3.0); }

   this->SetCellVolumes();

   // Allocate sl_, sp_, and path_ vectors
   this->SetVectorSizes();

   // Set Symmetry Points and Labels
   sl_[0] = "Gamma";
   sp_[0] = 0.0;

   sl_[1] = "K";
   add(1.0/3.0, rec_vecs_[0], 1.0/3.0, rec_vecs_[1], sp_[1]);

   sl_[2] = "M";
   sp_[2].Set(0.5, rec_vecs_[0]);

   // Define Paths
   path_[0][0] = 0;
   path_[0][1] = 2;
   path_[0][2] = 1;
   path_[0][3] = 0;

   // Set Intermediate Symmetry Points
   this->SetIntermediatePoints();

   // Set Intermediate Symmetry Point Labels
   il_[0][0] = "Sigma";  // Gamma -> M
   il_[0][1] = "MK";     // M     -> K
   il_[0][2] = "GammaK"; // K     -> Gamma

   // Create transformation generators
   // R60_(0,0) =  0.5; R60_(0,1) = -sqrt(0.75);
   // R60_(1,0) =  sqrt(0.75); R60_(1,1) =  0.5;

   // RX_(0,0)  =  1.0; RX_(0,1) =  0.0;
   // RX_(1,0)  =  0.0; RX_(1,1) = -1.0;

   // Set Mesh data
   fd_vert_[0] = 0.0;      fd_vert_[ 1] = 0.0;      fd_vert_[ 2] = 0.0;
   fd_vert_[3] = 0.5 * a_; fd_vert_[ 4] = 0.0;      fd_vert_[ 5] = 0.0;
   fd_vert_[6] = 0.5 * a_; fd_vert_[ 7] = 0.5 * sqrt(1.0/3.0) * a_;
   fd_vert_[ 8] = 0.0;

   fd_e2v_[0] = 0; fd_e2v_[1] = 1; fd_e2v_[2] = 2;
   fd_elem_att_[0] = 1;

   fd_be2v_[0] = 0; fd_be2v_[1] = 1;
   fd_be2v_[2] = 1; fd_be2v_[3] = 2;
   fd_be2v_[4] = 2; fd_be2v_[5] = 0;

   fd_belem_att_[0] = 10; fd_belem_att_[1] = 1;
   fd_belem_att_[2] = 10;
   /*
   for (int i=0; i<21; i++) { ws_vert_[i] = 0.0; }
   ws_vert_[ 0] =  0.0;      ws_vert_[ 1] =  0.0;
   ws_vert_[ 3] =  0.0;      ws_vert_[ 4] = -sqrt(1.0/3.0) * a_;
   ws_vert_[ 6] =  0.5 * a_; ws_vert_[ 7] = -0.5 * sqrt(1.0/3.0) * a_;
   ws_vert_[ 9] =  0.5 * a_; ws_vert_[10] =  0.5 * sqrt(1.0/3.0) * a_;
   ws_vert_[12] =  0.0;      ws_vert_[13] =  sqrt(1.0/3.0) * a_;
   ws_vert_[15] = -0.5 * a_; ws_vert_[16] =  0.5 * sqrt(1.0/3.0) * a_;
   ws_vert_[18] = -0.5 * a_; ws_vert_[19] = -0.5 * sqrt(1.0/3.0) * a_;

   ws_e2v_[0] = 0; ws_e2v_[1] = 1; ws_e2v_[ 2] = 2; ws_e2v_[ 3] = 3;
   ws_e2v_[4] = 0; ws_e2v_[5] = 3; ws_e2v_[ 6] = 4; ws_e2v_[ 7] = 5;
   ws_e2v_[8] = 0; ws_e2v_[9] = 5; ws_e2v_[10] = 6; ws_e2v_[11] = 1;

   ws_elem_att_[0] = 1;
   ws_elem_att_[1] = 1;
   ws_elem_att_[2] = 1;

   ws_be2v_[ 0] = 1; ws_be2v_[ 1] = 2;
   ws_be2v_[ 2] = 2; ws_be2v_[ 3] = 3;
   ws_be2v_[ 4] = 3; ws_be2v_[ 5] = 4;
   ws_be2v_[ 6] = 4; ws_be2v_[ 7] = 5;
   ws_be2v_[ 8] = 5; ws_be2v_[ 9] = 6;
   ws_be2v_[10] = 6; ws_be2v_[11] = 1;

   for (int i=0; i<6; i++) { ws_belem_att_[i] = 1; }
   */
}

bool
HexagonalLattice::MapToFundamentalDomain(const Vector & pt,
                                         Vector & ipt) const
{
   bool map = false;
   ipt = pt;
   if ( ipt[1] > 0.0 )
   {
      ipt[1] *= -1.0;
      map = true;
   }
   if ( ipt[0] < 0.0 )
   {
      ipt[0] *= -1.0;
      map = true;
   }
   if ( ipt[1] < -sqrt(3.0) * ipt[0] )
   {
      double d0 = ipt[0], d1 = ipt[1];
      ipt[0] = -0.5 * (d0 + sqrt(3.0) * d1);
      ipt[1] =  0.5 * (d1 - sqrt(3.0) * d0);
      map = true;
   }
   if ( ipt[0] > -sqrt(3.0) * ipt[1] )
   {
      double d0 = ipt[0], d1 = ipt[1];
      ipt[0] =  0.5 * (d0 - sqrt(3.0) * d1);
      ipt[1] = -0.5 * (d1 + sqrt(3.0) * d0);
      map = true;
   }
   return map;
}

const DenseMatrix & HexagonalLattice::GetTransformation(int ti) const
{
   int ir = ti % 3;
   int iq = ti / 3;

   T_ = 0.0;

   switch (ir)
   {
      case 0:
         T_(0, 0) =  1.0;
         T_(1, 1) =  1.0;
         break;
      case 1:
         T_(0, 0) =  0.5;
         T_(0, 1) =  sqrt(0.75);
         T_(1, 0) =  sqrt(0.75);
         T_(1, 1) = -0.5;
         break;
      case 2:
         T_(0, 0) =  0.5;
         T_(0, 1) = -sqrt(0.75);
         T_(1, 0) =  sqrt(0.75);
         T_(1, 1) =  0.5;
         break;
   }

   for (int i=0; i<2; i++)
   {
      if ( iq & (int)pow(2, i) )
      {
         T_(i,0) *= -1.0;
         T_(i,1) *= -1.0;
      }
   }

   return T_;
}

Mesh *
HexagonalLattice::GetFundamentalDomainMesh() const
{
   Mesh * mesh = new Mesh((double*)fd_vert_, 3,
                          (int*)fd_e2v_, Geometry::TRIANGLE,
                          (int*)fd_elem_att_, 1,
                          (int*)fd_be2v_, Geometry::SEGMENT,
                          (int*)fd_belem_att_, 3,
                          2, 2);

   mesh->Finalize();

   return mesh;
}
/*
Mesh *
HexagonalLattice::GetWignerSeitzMesh(bool triMesh) const
{
 Mesh * mesh = new Mesh((double*)ws_vert_, 7,
                        (int*)ws_e2v_, Geometry::SQUARE,
                        (int*)ws_elem_att_, 3,
                        (int*)ws_be2v_, Geometry::SEGMENT,
                        (int*)ws_belem_att_, 6,
                        2, 2);
 mesh->Finalize();

 return mesh;
}

Mesh *
HexagonalLattice::GetPeriodicWignerSeitzMesh(bool triMesh) const
{
 Mesh * mesh = this->GetWignerSeitzMesh(triMesh);

 mesh->UniformRefinement();

 Mesh * per_mesh = MakePeriodicMesh(mesh, trn_vecs_);

 delete mesh;

 return per_mesh;
}
*/
RectangularLattice::RectangularLattice(double a, double b)
   : BravaisLattice2D(a, b, 0.5 * M_PI)
{
   label_ = "RECT";
   type_  = PRIMITIVE_RECTANGULAR;

   // Set Lattice Vectors
   lat_vecs_[0][0] =  a_; lat_vecs_[0][1] = 0.0;
   lat_vecs_[1][0] = 0.0; lat_vecs_[1][1] =  b_;

   // Set Reciprocal Lattice Vectors
   rec_vecs_[0][0] = 1.0 / a_; rec_vecs_[0][1] = 0.0;
   rec_vecs_[1][0] = 0.0;      rec_vecs_[1][1] = 1.0 / b_;

   // Set Translation Vectors
   trn_vecs_.resize(2);
   for (unsigned int i=0; i<trn_vecs_.size(); i++)
   {
      trn_vecs_[i].SetSize(2);
   }
   trn_vecs_[0][0] =  a_; trn_vecs_[0][1] = 0.0;
   trn_vecs_[1][0] = 0.0; trn_vecs_[1][1] =  b_;

   // Set the face radii
   face_radii_.resize(2);
   face_radii_[0] = 0.5 * b_;
   face_radii_[1] = 0.5 * a_;

   this->SetCellVolumes();

   // Allocate sl_, sp_, and path_ vectors
   this->SetVectorSizes();

   // Set Symmetry Points and Labels
   sl_[0] = "Gamma";
   sp_[0] = 0.0;

   sl_[1] = "X";
   sp_[1].Set(0.5,rec_vecs_[0]);

   sl_[2] = "S";
   add(0.5,rec_vecs_[0],0.5,rec_vecs_[1],sp_[2]);

   sl_[3] = "Y";
   sp_[3].Set(0.5,rec_vecs_[1]);

   for (unsigned int i=0; i<sl_.size(); i++)
   {
      si_[sl_[i]] = i;
   }

   // Define Paths
   path_[0][0] = 0;
   path_[0][1] = 1;
   path_[0][2] = 2;
   path_[0][3] = 3;
   path_[0][4] = 0;

   // Set Intermediate Symmetry Points
   this->SetIntermediatePoints();

   // Set Intermediate Symmetry Point Labels
   il_[0][0] = "GammaX"; // Gamma -> X
   il_[0][1] = "XS";     // X     -> S
   il_[0][2] = "SY";     // S     -> Y
   il_[0][3] = "YGamma"; // Y     -> Gamma

   // Set Mesh data
   fd_vert_[0] = 0.0;      fd_vert_[ 1] = 0.0;      fd_vert_[ 2] = 0.0;
   fd_vert_[3] = 0.5 * a_; fd_vert_[ 4] = 0.0;      fd_vert_[ 5] = 0.0;
   fd_vert_[6] = 0.5 * a_; fd_vert_[ 7] = 0.5 * b_; fd_vert_[ 8] = 0.0;
   fd_vert_[9] = 0.0;      fd_vert_[10] = 0.5 * b_; fd_vert_[11] = 0.0;

   fd_e2v_[0] = 0; fd_e2v_[1] = 1; fd_e2v_[2] = 2; fd_e2v_[3] = 3;
   fd_elem_att_[0] = 1;

   fd_be2v_[0] = 0; fd_be2v_[1] = 1;
   fd_be2v_[2] = 1; fd_be2v_[3] = 2;
   fd_be2v_[4] = 2; fd_be2v_[5] = 3;
   fd_be2v_[6] = 3; fd_be2v_[7] = 0;

   fd_belem_att_[0] = 10; fd_belem_att_[1] = 1;
   fd_belem_att_[2] =  1; fd_belem_att_[3] = 10;
   /*
   ws_vert_[0] = -0.5 * a_; ws_vert_[ 1] = -0.5 * b_; ws_vert_[ 2] = 0.0;
   ws_vert_[3] =  0.5 * a_; ws_vert_[ 4] = -0.5 * b_; ws_vert_[ 5] = 0.0;
   ws_vert_[6] =  0.5 * a_; ws_vert_[ 7] =  0.5 * b_; ws_vert_[ 8] = 0.0;
   ws_vert_[9] = -0.5 * a_; ws_vert_[10] =  0.5 * b_; ws_vert_[11] = 0.0;

   ws_e2v_[0] = 0; ws_e2v_[1] = 1; ws_e2v_[2] = 2; ws_e2v_[3] = 3;
   ws_elem_att_[0] = 1;

   ws_be2v_[0] = 0; ws_be2v_[1] = 1;
   ws_be2v_[2] = 1; ws_be2v_[3] = 2;
   ws_be2v_[4] = 2; ws_be2v_[5] = 3;
   ws_be2v_[6] = 3; ws_be2v_[7] = 0;

   ws_belem_att_[0] = 1; ws_belem_att_[1] = 1;
   ws_belem_att_[2] = 1; ws_belem_att_[3] = 1;
   */
}

bool
RectangularLattice::MapToFundamentalDomain(const Vector & pt,
                                           Vector & ipt) const
{
   bool map = false;
   ipt = pt;
   if ( ipt[1] < 0.0 )
   {
      ipt[1] *= -1.0;
      map = true;
   }
   if ( ipt[0] < 0.0 )
   {
      ipt[0] *= -1.0;
      map = true;
   }
   return map;
}

const DenseMatrix & RectangularLattice::GetTransformation(int ti) const
{
   switch (ti)
   {
      case 0:
         T_(0,0) =  1.0; T_(0,1) =  0.0;
         T_(1,0) =  0.0; T_(1,1) =  1.0;
         break;
      case 1:
         T_(0,0) = -1.0; T_(0,1) =  0.0;
         T_(1,0) =  0.0; T_(1,1) =  1.0;
         break;
      case 2:
         T_(0,0) = -1.0; T_(0,1) =  0.0;
         T_(1,0) =  0.0; T_(1,1) = -1.0;
         break;
      case 3:
         T_(0,0) =  1.0; T_(0,1) =  0.0;
         T_(1,0) =  0.0; T_(1,1) = -1.0;
         break;
   }
   return T_;
}

Mesh *
RectangularLattice::GetFundamentalDomainMesh() const
{
   Mesh * mesh = new Mesh((double*)fd_vert_, 4,
                          (int*)fd_e2v_, Geometry::SQUARE,
                          (int*)fd_elem_att_, 1,
                          (int*)fd_be2v_, Geometry::SEGMENT,
                          (int*)fd_belem_att_, 4,
                          2, 2);
   mesh->Finalize();

   return mesh;
}
/*
Mesh *
RectangularLattice::GetWignerSeitzMesh(bool triMesh) const
{
   Mesh * mesh = new Mesh((double*)ws_vert_, 4,
                          (int*)ws_e2v_, Geometry::SQUARE,
                          (int*)ws_elem_att_, 1,
                          (int*)ws_be2v_, Geometry::SEGMENT,
                          (int*)ws_belem_att_, 4,
                          2, 2);
   mesh->Finalize();

   return mesh;
}

Mesh *
RectangularLattice::GetPeriodicWignerSeitzMesh(bool triMesh) const
{
   Mesh * mesh = this->GetWignerSeitzMesh(triMesh);

   mesh->UniformRefinement();
   mesh->UniformRefinement();

   Mesh * per_mesh = MakePeriodicMesh(mesh, trn_vecs_);

   delete mesh;

   return per_mesh;
}
*/
CenteredRectangularLattice::CenteredRectangularLattice(double a, double b)
   : BravaisLattice2D(a, b, 0.5 * M_PI)
{
   label_ = ( b_ < a_ )?"CR1":"CR2";
   type_  = CENTERED_RECTANGULAR;

   // Set Lattice Vectors
   lat_vecs_[0][0] = 0.5 * a_; lat_vecs_[0][1] = -0.5 * b_;
   lat_vecs_[1][0] = 0.5 * a_; lat_vecs_[1][1] =  0.5 * b_;

   // Set Reciprocal Lattice Vectors
   rec_vecs_[0][0] = 1.0 / a_; rec_vecs_[0][1] = -1.0 / b_;
   rec_vecs_[1][0] = 1.0 / a_; rec_vecs_[1][1] =  1.0 / b_;

   // Set Translation Vectors
   trn_vecs_.resize(3);
   for (unsigned int i=0; i<trn_vecs_.size(); i++)
   {
      trn_vecs_[i].SetSize(2);
   }
   if ( b_ < a_ )
   {
      trn_vecs_[0][0] = 0.5 * a_; trn_vecs_[0][1] = -0.5 * b_;
      trn_vecs_[1][0] = 0.5 * a_; trn_vecs_[1][1] =  0.5 * b_;
      trn_vecs_[2][0] = 0.0;      trn_vecs_[2][1] = b_;
   }
   else
   {
      trn_vecs_[0][0] =  a_;       trn_vecs_[0][1] =  0.0;
      trn_vecs_[1][0] =  0.5 * a_; trn_vecs_[1][1] =  0.5 * b_;
      trn_vecs_[2][0] = -0.5 * a_; trn_vecs_[2][1] =  0.5 * b_;
   }

   // Set the face radii
   face_radii_.resize(3);
   if ( b_ < a_ )
   {
      face_radii_[0] = 0.25 * b_ * sqrt(1.0 + b_ * b_  / (a_ * a_));
      face_radii_[1] = 0.25 * b_ * sqrt(1.0 + b_ * b_  / (a_ * a_));
      face_radii_[2] = 0.25 * (a_ - b_ * b_ / a_);
   }
   else
   {
      face_radii_[0] = 0.25 * (b_ - a_ * a_ / b_);
      face_radii_[1] = 0.25 * a_ * sqrt(1.0 + a_ * a_  / (b_ * b_));
      face_radii_[2] = 0.25 * a_ * sqrt(1.0 + a_ * a_  / (b_ * b_));
   }

   this->SetCellVolumes();

   // Allocate sl_, sp_, and path_ vectors
   this->SetVectorSizes();

   // Set Symmetry Points and Labels
   sl_[0] = "Gamma";
   sp_[0] = 0.0;

   sl_[1] = "X";
   sp_[1].Set(0.5,rec_vecs_[0]);

   sl_[2] = "S";
   add(0.5,rec_vecs_[0],0.5,rec_vecs_[1],sp_[2]);

   sl_[3] = "Y";
   sp_[3].Set(0.5,rec_vecs_[1]);

   for (unsigned int i=0; i<sl_.size(); i++)
   {
      si_[sl_[i]] = i;
   }

   // Define Paths
   path_[0][0] = 0;
   path_[0][1] = 1;
   path_[0][2] = 2;
   path_[0][3] = 3;
   path_[0][4] = 0;

   // Set Intermediate Symmetry Points
   this->SetIntermediatePoints();

   // Set Intermediate Symmetry Point Labels
   il_[0][0] = "GammaX"; // Gamma -> X
   il_[0][1] = "XS";     // X     -> S
   il_[0][2] = "SY";     // S     -> Y
   il_[0][3] = "YGamma"; // Y     -> Gamma

   // Set Mesh data
   fd_vert_[ 0] = 0.0;
   fd_vert_[ 1] = 0.0;
   fd_vert_[ 2] = 0.0;

   fd_vert_[ 3] = min(0.5 * a_, 0.25 * (a_ * a_ + b_ * b_) / a_);
   fd_vert_[ 4] = 0.0;
   fd_vert_[ 5] = 0.0;

   fd_vert_[ 6] = (a_ < b_) ? (0.5 * a_) : (0.25 * (a_ * a_ - b_ * b_) / a_);
   fd_vert_[ 7] = (a_ < b_) ? (0.25 * (b_ * b_ - a_ * a_) / b_) : (0.5 * b_);
   fd_vert_[ 8] = 0.0;

   fd_vert_[ 9] = 0.0;
   fd_vert_[10] = min(0.5 * b_, 0.25 * (a_ * a_ + b_ * b_) / b_);
   fd_vert_[11] = 0.0;

   fd_e2v_[0] = 0; fd_e2v_[1] = 1; fd_e2v_[2] = 2; fd_e2v_[3] = 3;
   fd_elem_att_[0] = 1;

   fd_be2v_[0] = 0; fd_be2v_[1] = 1;
   fd_be2v_[2] = 1; fd_be2v_[3] = 2;
   fd_be2v_[4] = 2; fd_be2v_[5] = 3;
   fd_be2v_[6] = 3; fd_be2v_[7] = 0;

   fd_belem_att_[0] = 10; fd_belem_att_[1] = 1;
   fd_belem_att_[2] =  1; fd_belem_att_[3] = 10;
   /*
   ws_vert_[0] = -0.5 * a_; ws_vert_[ 1] = -0.5 * b_; ws_vert_[ 2] = 0.0;
   ws_vert_[3] =  0.5 * a_; ws_vert_[ 4] = -0.5 * b_; ws_vert_[ 5] = 0.0;
   ws_vert_[6] =  0.5 * a_; ws_vert_[ 7] =  0.5 * b_; ws_vert_[ 8] = 0.0;
   ws_vert_[9] = -0.5 * a_; ws_vert_[10] =  0.5 * b_; ws_vert_[11] = 0.0;

   ws_e2v_[0] = 0; ws_e2v_[1] = 1; ws_e2v_[2] = 2; ws_e2v_[3] = 3;
   ws_elem_att_[0] = 1;

   ws_be2v_[0] = 0; ws_be2v_[1] = 1;
   ws_be2v_[2] = 1; ws_be2v_[3] = 2;
   ws_be2v_[4] = 2; ws_be2v_[5] = 3;
   ws_be2v_[6] = 3; ws_be2v_[7] = 0;

   ws_belem_att_[0] = 1; ws_belem_att_[1] = 1;
   ws_belem_att_[2] = 1; ws_belem_att_[3] = 1;
   */
}

bool
CenteredRectangularLattice::MapToFundamentalDomain(const Vector & pt,
                                                   Vector & ipt) const
{
   bool map = false;
   ipt = pt;
   if ( ipt[1] < 0.0 )
   {
      ipt[1] *= -1.0;
      map = true;
   }
   if ( ipt[0] < 0.0 )
   {
      ipt[0] *= -1.0;
      map = true;
   }
   return map;
}

const DenseMatrix & CenteredRectangularLattice::GetTransformation(int ti) const
{
   T_(0,0) = (ti > 0 && ti < 3) ? 1.0 : -1.0;
   T_(0,1) = 0.0;
   T_(1,0) = 0.0;
   T_(1,1) = (ti < 2) ? 1.0 : -1.0;
   return T_;
}

Mesh *
CenteredRectangularLattice::GetFundamentalDomainMesh() const
{
   Mesh * mesh = new Mesh((double*)fd_vert_, 4,
                          (int*)fd_e2v_, Geometry::SQUARE,
                          (int*)fd_elem_att_, 1,
                          (int*)fd_be2v_, Geometry::SEGMENT,
                          (int*)fd_belem_att_, 4,
                          2, 2);
   mesh->Finalize();

   return mesh;
}
/*
Mesh *
CenteredRectangularLattice::GetWignerSeitzMesh(bool triMesh) const
{
    Mesh * mesh = new Mesh((double*)ws_vert_, 4,
                           (int*)ws_e2v_, Geometry::SQUARE,
                           (int*)ws_elem_att_, 1,
                           (int*)ws_be2v_, Geometry::SEGMENT,
                           (int*)ws_belem_att_, 4,
                           2, 2);
    mesh->Finalize();

    return mesh;
}

Mesh *
CenteredRectangularLattice::GetPeriodicWignerSeitzMesh(bool triMesh) const
{
   Mesh * mesh = this->GetWignerSeitzMesh(triMesh);

   mesh->UniformRefinement();
   mesh->UniformRefinement();

   Mesh * per_mesh = MakePeriodicMesh(mesh, trn_vecs_);

   delete mesh;

   return per_mesh;
}
*/
ObliqueLattice::ObliqueLattice(double a, double b, double gamma)
   : BravaisLattice2D(a, b, gamma)
{
   MFEM_ASSERT( a_ <= b_,
                "Monoclinic unit cells must have a <= b!");
   MFEM_ASSERT( b_ * cos(gamma_) < a_ && gamma_ < 0.5 * M_PI,
                "Oblique unit cells must have arccos(a/b) < gamma < pi/2!");

   label_ = "OBL";
   type_  = PRIMITIVE_OBLIQUE;

   // Set Lattice Vectors
   lat_vecs_[0][0] = a_; lat_vecs_[0][1] = 0.0;
   lat_vecs_[1][0] = b_ * cos(gamma_); lat_vecs_[1][1] = b_ * sin(gamma_);

   // Set Reciprocal Lattice Vectors
   rec_vecs_[0][0] = 1.0 / a_; rec_vecs_[0][1] = -1.0 / (a_ * tan(gamma_));
   rec_vecs_[1][0] = 0.0; rec_vecs_[1][1] = 1.0 / (b_ * sin(gamma_));

   // Set Translation Vectors
   trn_vecs_.resize(3);
   for (unsigned int i=0; i<trn_vecs_.size(); i++)
   {
      trn_vecs_[i].SetSize(2);
   }
   trn_vecs_[0] =  lat_vecs_[0];
   trn_vecs_[1] =  lat_vecs_[1];
   trn_vecs_[2] =  lat_vecs_[0]; trn_vecs_[2] -= lat_vecs_[1];

   // Set the face radii
   double cosg = cos(gamma_);
   double sing = sin(gamma_);
   double cscg = 1.0 / sing;
   face_radii_.resize(3);
   face_radii_[0] = 0.5 * (b_ - a_ * cosg) * cscg;
   face_radii_[1] = 0.5 * (a_ - b_ * cosg) * cscg;
   face_radii_[2] = 0.5 * sqrt(a_ * a_ + b_ * b_ - 2.0 * a_ * b_ * cosg)
                    * cosg * cscg;

   this->SetCellVolumes();

   // Allocate sl_, sp_, and path_ vectors
   this->SetVectorSizes();

   // Set Symmetry Points and Labels
   double sing2 = pow(sing, 2.0);
   double eta = 0.5 * (1.0 - a_ * cosg / b_) / sing2;
   double nu = 0.5 - eta * b_ * cosg / a_;

   sl_[0] = "Gamma";
   sp_[0] = 0.0;

   sl_[1] = "Y";
   sp_[1].Set(0.5, rec_vecs_[1]);

   sl_[2] = "H";
   add(eta, rec_vecs_[0], 1.0-nu, rec_vecs_[1], sp_[2]);

   sl_[3] = "C";
   add(0.5, rec_vecs_[0], 0.5, rec_vecs_[1], sp_[3]);

   sl_[4] = "H1";
   add(1.0-eta, rec_vecs_[0], nu, rec_vecs_[1], sp_[4]);

   sl_[5] = "X";
   sp_[5].Set(0.5, rec_vecs_[0]);

   sl_[6] = "H2";
   add(eta, rec_vecs_[0], -nu, rec_vecs_[1], sp_[6]);

   sl_[7] = "Y1";
   sp_[7].Set(-0.5, rec_vecs_[1]);

   for (unsigned int i=0; i<sl_.size(); i++)
   {
      si_[sl_[i]] = i;
   }

   // Define Paths
   path_[0][0] = 0;
   path_[0][1] = 1;
   path_[0][2] = 2;
   path_[0][3] = 3;
   path_[0][4] = 4;
   path_[0][5] = 5;
   path_[0][6] = 6;
   path_[0][7] = 7;
   path_[0][8] = 0;

   // Set Intermediate Symmetry Points
   this->SetIntermediatePoints();

   // Set Intermediate Symmetry Point Labels
   il_[0][0] = "GammaY";  // Gamma -> Y
   il_[0][1] = "YH";      // Y     -> H
   il_[0][2] = "HC";      // H     -> C
   il_[0][3] = "CH1";     // C     -> H1
   il_[0][4] = "H1X";     // H1    -> X
   il_[0][5] = "XH2";     // X     -> H2
   il_[0][6] = "H2Y1";    // H2    -> Y1
   il_[0][7] = "Y1Gamma"; // Y1     -> Gamma

   // Set Mesh data
   double cos2g = cos(2.0 * gamma_);

   fd_vert_[ 0] =  0.0;
   fd_vert_[ 1] =  0.0;
   fd_vert_[ 2] =  0.0;

   fd_vert_[ 3] = -0.5 * b_ * cosg;
   fd_vert_[ 4] = -0.5 * b_ * sing;
   fd_vert_[ 5] =  0.0;

   fd_vert_[ 6] =  0.5 * a_ - b_ * cosg;
   fd_vert_[ 7] = -0.5 * (a_ * cosg - b_ * cos2g) * cscg;
   fd_vert_[ 8] =  0.0;

   fd_vert_[ 9] =  0.5 * (a_ - b_ * cosg);
   fd_vert_[10] = -0.5 * b_ * sing;
   fd_vert_[11] =  0.0;

   fd_vert_[12] =  0.5 * a_;
   fd_vert_[13] =  0.5 * (a_ * cosg - b_) * cscg;
   fd_vert_[14] =  0.0;

   fd_vert_[15] =  0.5 * a_;
   fd_vert_[16] =  0.0;
   fd_vert_[17] =  0.0;

   fd_vert_[18] =  0.5 * a_;
   fd_vert_[19] =  0.5 * (b_ - a_ * cosg) * cscg;
   fd_vert_[20] =  0.0;

   fd_vert_[21] =  0.5 * b_ * cosg;
   fd_vert_[22] =  0.5 * b_ * sing;
   fd_vert_[23] =  0.0;
   /*
   fd_e2v_[ 0] = 0; fd_e2v_[ 1] = 1; fd_e2v_[ 2] = 2;
   fd_e2v_[ 3] = 0; fd_e2v_[ 4] = 2; fd_e2v_[ 5] = 3;
   fd_e2v_[ 6] = 0; fd_e2v_[ 7] = 3; fd_e2v_[ 8] = 4;
   fd_e2v_[ 9] = 0; fd_e2v_[10] = 4; fd_e2v_[11] = 5;
   fd_e2v_[12] = 0; fd_e2v_[13] = 5; fd_e2v_[14] = 6;
   fd_e2v_[15] = 0; fd_e2v_[16] = 6; fd_e2v_[17] = 7;
   fd_elem_att_[0] = 1; fd_elem_att_[1] = 1; fd_elem_att_[2] = 1;
   fd_elem_att_[3] = 1; fd_elem_att_[4] = 1; fd_elem_att_[5] = 1;
   */
   fd_e2v_[ 0] = 0; fd_e2v_[ 1] = 1; fd_e2v_[ 2] = 2; fd_e2v_[ 3] = 3;
   fd_e2v_[ 4] = 0; fd_e2v_[ 5] = 3; fd_e2v_[ 6] = 4; fd_e2v_[ 7] = 5;
   fd_e2v_[ 8] = 0; fd_e2v_[ 9] = 5; fd_e2v_[10] = 6; fd_e2v_[11] = 7;
   fd_elem_att_[0] = 1; fd_elem_att_[1] = 1; fd_elem_att_[2] = 1;

   fd_be2v_[ 0] = 0; fd_be2v_[ 1] = 1;
   fd_be2v_[ 2] = 1; fd_be2v_[ 3] = 2;
   fd_be2v_[ 4] = 2; fd_be2v_[ 5] = 3;
   fd_be2v_[ 6] = 3; fd_be2v_[ 7] = 4;
   fd_be2v_[ 8] = 4; fd_be2v_[ 9] = 5;
   fd_be2v_[10] = 5; fd_be2v_[11] = 6;
   fd_be2v_[12] = 6; fd_be2v_[13] = 7;
   fd_be2v_[14] = 7; fd_be2v_[15] = 0;

   fd_belem_att_[0] = 10; fd_belem_att_[1] = 1;
   fd_belem_att_[2] =  1; fd_belem_att_[3] = 1;
   fd_belem_att_[4] =  1; fd_belem_att_[5] = 1;
   fd_belem_att_[6] =  1; fd_belem_att_[7] = 10;
   /*
   ws_vert_[0] = -0.5 * a_; ws_vert_[ 1] = -0.5 * b_; ws_vert_[ 2] = 0.0;
   ws_vert_[3] =  0.5 * a_; ws_vert_[ 4] = -0.5 * b_; ws_vert_[ 5] = 0.0;
   ws_vert_[6] =  0.5 * a_; ws_vert_[ 7] =  0.5 * b_; ws_vert_[ 8] = 0.0;
   ws_vert_[9] = -0.5 * a_; ws_vert_[10] =  0.5 * b_; ws_vert_[11] = 0.0;

   ws_e2v_[0] = 0; ws_e2v_[1] = 1; ws_e2v_[2] = 2; ws_e2v_[3] = 3;
   ws_elem_att_[0] = 1;

   ws_be2v_[0] = 0; ws_be2v_[1] = 1;
   ws_be2v_[2] = 1; ws_be2v_[3] = 2;
   ws_be2v_[4] = 2; ws_be2v_[5] = 3;
   ws_be2v_[6] = 3; ws_be2v_[7] = 0;

   ws_belem_att_[0] = 1; ws_belem_att_[1] = 1;
   ws_belem_att_[2] = 1; ws_belem_att_[3] = 1;
   */
}

bool
ObliqueLattice::MapToFundamentalDomain(const Vector & pt,
                                       Vector & ipt) const
{
   bool map = false;
   ipt = pt;
   if ( tan(gamma_) * ipt[0] < ipt[1] )
   {
      ipt[0] *= -1.0;
      ipt[1] *= -1.0;
      map = true;
   }
   return map;
}

const DenseMatrix & ObliqueLattice::GetTransformation(int ti) const
{
   T_(0,0) = (ti == 0) ? 1.0 : -1.0;
   T_(0,1) = 0.0;
   T_(1,0) = 0.0;
   T_(1,1) = (ti == 0) ? 1.0 : -1.0;
   return T_;
}

Mesh *
ObliqueLattice::GetFundamentalDomainMesh() const
{
   /*
   Mesh * mesh = new Mesh((double*)fd_vert_, 8,
                          (int*)fd_e2v_, Geometry::TRIANGLE,
                          (int*)fd_elem_att_, 6,
                          (int*)fd_be2v_, Geometry::SEGMENT,
                          (int*)fd_belem_att_, 8,
                          2, 2);
   */
   Mesh * mesh = new Mesh((double*)fd_vert_, 8,
                          (int*)fd_e2v_, Geometry::SQUARE,
                          (int*)fd_elem_att_, 3,
                          (int*)fd_be2v_, Geometry::SEGMENT,
                          (int*)fd_belem_att_, 8,
                          2, 2);

   mesh->Finalize();

   return mesh;
}
/*
Mesh *
ObliqueLattice::GetWignerSeitzMesh(bool triMesh) const
{
    Mesh * mesh = new Mesh((double*)ws_vert_, 4,
                           (int*)ws_e2v_, Geometry::SQUARE,
                           (int*)ws_elem_att_, 1,
                           (int*)ws_be2v_, Geometry::SEGMENT,
                           (int*)ws_belem_att_, 4,
                           2, 2);
    mesh->Finalize();

    return mesh;
}

Mesh *
ObliqueLattice::GetPeriodicWignerSeitzMesh(bool triMesh) const
{
   Mesh * mesh = this->GetWignerSeitzMesh(triMesh);

   mesh->UniformRefinement();
   mesh->UniformRefinement();

   Mesh * per_mesh = MakePeriodicMesh(mesh, trn_vecs_);

   delete mesh;

   return per_mesh;
}
*/
CubicLattice::CubicLattice(double a)
   : BravaisLattice3D(a, a, a, 0.5 * M_PI, 0.5 * M_PI, 0.5 * M_PI)
{
   label_ = "CUB";
   type_  = PRIMITIVE_CUBIC;

   // Set Lattice Vectors
   lat_vecs_[0][0] = 1.0; lat_vecs_[0][1] = 0.0; lat_vecs_[0][2] = 0.0;
   lat_vecs_[1][0] = 0.0; lat_vecs_[1][1] = 1.0; lat_vecs_[1][2] = 0.0;
   lat_vecs_[2][0] = 0.0; lat_vecs_[2][1] = 0.0; lat_vecs_[2][2] = 1.0;

   // Set Reciprocal Lattice Vectors
   rec_vecs_[0][0] = 1.0; rec_vecs_[0][1] = 0.0; rec_vecs_[0][2] = 0.0;
   rec_vecs_[1][0] = 0.0; rec_vecs_[1][1] = 1.0; rec_vecs_[1][2] = 0.0;
   rec_vecs_[2][0] = 0.0; rec_vecs_[2][1] = 0.0; rec_vecs_[2][2] = 1.0;

   // Set Translation Vectors
   trn_vecs_.resize(3);
   for (unsigned int i=0; i<trn_vecs_.size(); i++)
   {
      trn_vecs_[i].SetSize(3);
   }
   trn_vecs_[0][0] = 1.0; trn_vecs_[0][1] = 0.0; trn_vecs_[0][2] = 0.0;
   trn_vecs_[1][0] = 0.0; trn_vecs_[1][1] = 1.0; trn_vecs_[1][2] = 0.0;
   trn_vecs_[2][0] = 0.0; trn_vecs_[2][1] = 0.0; trn_vecs_[2][2] = 1.0;

   // Set the face radii
   face_radii_.resize(3);
   for (int i=0; i<3; i++) { face_radii_[i] = 0.5 * a_; }

   for (unsigned int i=0; i<lat_vecs_.size(); i++)
   {
      lat_vecs_[i] *= a_;
      rec_vecs_[i] *= 1.0/a_;
      trn_vecs_[i] *= a_;
   }

   this->SetCellVolumes();

   // Allocate sl_, sp_, and path_ vectors
   this->SetVectorSizes();

   // Set Symmetry Points and Labels
   sl_[0] = "Gamma";
   sp_[0] = 0.0;

   sl_[1] = "X";
   sp_[1].Set(0.5,rec_vecs_[1]);

   sl_[2] = "M";
   add(0.5,rec_vecs_[0],0.5,rec_vecs_[1],sp_[2]);

   sl_[3] = "R";
   add(0.5,rec_vecs_[0],0.5,rec_vecs_[1],sp_[3]);
   sp_[3].Add(0.5,rec_vecs_[2]);

   for (unsigned int i=0; i<sl_.size(); i++)
   {
      si_[sl_[i]] = i;
   }

   // Define Paths
   path_[0][0] = 0;
   path_[0][1] = 1;
   path_[0][2] = 2;
   path_[0][3] = 0;
   path_[0][4] = 3;
   path_[0][5] = 1;

   path_[1][0] = 2;
   path_[1][1] = 3;

   // Set Intermediate Symmetry Points
   this->SetIntermediatePoints();

   // Set Intermediate Symmetry Point Labels
   il_[0][0] = "Delta";  // Gamma -> X
   il_[0][1] = "Z";      // X     -> M
   il_[0][2] = "Sigma";  // M     -> Gamma
   il_[0][3] = "Lambda"; // Gamma -> R
   il_[0][4] = "S";      // R     -> X
   il_[1][0] = "T";      // M     -> R

   // Set Mesh data
   fd_vert_[0] = 0.0;      fd_vert_[ 1] = 0.0;      fd_vert_[ 2] = 0.0;
   fd_vert_[3] = 0.5 * a_; fd_vert_[ 4] = 0.0;      fd_vert_[ 5] = 0.0;
   fd_vert_[6] = 0.5 * a_; fd_vert_[ 7] = 0.5 * a_; fd_vert_[ 8] = 0.0;
   fd_vert_[9] = 0.5 * a_; fd_vert_[10] = 0.5 * a_; fd_vert_[11] = 0.5 * a_;

   fd_e2v_[0] = 0; fd_e2v_[1] = 1; fd_e2v_[2] = 2; fd_e2v_[3] = 3;
   fd_elem_att_[0] = 1;

   fd_be2v_[0] = 0; fd_be2v_[ 1] = 2; fd_be2v_[ 2] = 1;
   fd_be2v_[3] = 0; fd_be2v_[ 4] = 1; fd_be2v_[ 5] = 3;
   fd_be2v_[6] = 1; fd_be2v_[ 7] = 2; fd_be2v_[ 8] = 3;
   fd_be2v_[9] = 2; fd_be2v_[10] = 0; fd_be2v_[11] = 3;

   fd_belem_att_[0] = 10; fd_belem_att_[1] = 10;
   fd_belem_att_[2] =  1; fd_belem_att_[3] = 10;
   /*
   ws_vert_[ 0] = -0.5 * a_; ws_vert_[ 1] = -0.5 * a_; ws_vert_[ 2] = -0.5 * a_;
   ws_vert_[ 3] =  0.5 * a_; ws_vert_[ 4] = -0.5 * a_; ws_vert_[ 5] = -0.5 * a_;
   ws_vert_[ 6] =  0.5 * a_; ws_vert_[ 7] =  0.5 * a_; ws_vert_[ 8] = -0.5 * a_;
   ws_vert_[ 9] = -0.5 * a_; ws_vert_[10] =  0.5 * a_; ws_vert_[11] = -0.5 * a_;
   ws_vert_[12] = -0.5 * a_; ws_vert_[13] = -0.5 * a_; ws_vert_[14] =  0.5 * a_;
   ws_vert_[15] =  0.5 * a_; ws_vert_[16] = -0.5 * a_; ws_vert_[17] =  0.5 * a_;
   ws_vert_[18] =  0.5 * a_; ws_vert_[19] =  0.5 * a_; ws_vert_[20] =  0.5 * a_;
   ws_vert_[21] = -0.5 * a_; ws_vert_[22] =  0.5 * a_; ws_vert_[23] =  0.5 * a_;

   for (int i=0; i<8; i++) { ws_e2v_[i] = i; }
   ws_elem_att_[0] = 1;

   ws_be2v_[ 0] = 0; ws_be2v_[ 1] = 3; ws_be2v_[ 2] = 2; ws_be2v_[ 3] = 1;
   ws_be2v_[ 4] = 4; ws_be2v_[ 5] = 5; ws_be2v_[ 6] = 6; ws_be2v_[ 7] = 7;

   ws_be2v_[ 8] = 0; ws_be2v_[ 9] = 1; ws_be2v_[10] = 5; ws_be2v_[11] = 4;
   ws_be2v_[12] = 1; ws_be2v_[13] = 2; ws_be2v_[14] = 6; ws_be2v_[15] = 5;
   ws_be2v_[16] = 2; ws_be2v_[17] = 3; ws_be2v_[18] = 7; ws_be2v_[19] = 6;
   ws_be2v_[20] = 3; ws_be2v_[21] = 0; ws_be2v_[22] = 4; ws_be2v_[23] = 7;

   for (int i=0; i<6; i++) { ws_belem_att_[i] = 1; }
   */
   // Tetrahedral Mesh Data
   /*
   for (int k=0; k<3; k++)
   {
      for (int j=0; j<3; j++)
      {
    for (int i=0; i<3; i++)
    {
       ws_vert_tet_[27 * k + 9 * j + 3 * i + 0] = 0.5 * a_ * (i - 1);
       ws_vert_tet_[27 * k + 9 * j + 3 * i + 1] = 0.5 * a_ * (j - 1);
       ws_vert_tet_[27 * k + 9 * j + 3 * i + 2] = 0.5 * a_ * (k - 1);
    }
      }
   }
   */
   /*
   ws_tet_vert_[ 0] =  0.0;
   ws_tet_vert_[ 1] =  0.0;
   ws_tet_vert_[ 2] =  0.0;

   ws_tet_vert_[ 3] = -0.5 * a_;
   ws_tet_vert_[ 4] =  0.0;
   ws_tet_vert_[ 5] =  0.0;

   ws_tet_vert_[78] =  0.5 * a_;
   ws_tet_vert_[79] =  0.0;
   ws_tet_vert_[80] =  0.0;

   int vtx = 2;

   for (int k=0; k<3; k++)
   {
      double xc =  0.5 * a_ * (k - 1);

      for (int j=0; j<4; j++)
      {
         double yc = 0.5 * a_ * ((j % 2 - 1) + (j / 2) * (4 - j));
         double zc = 0.5 * a_ * ( j / 3 - 1);

         ws_tet_vert_[3 * vtx + 0] = xc;
         ws_tet_vert_[3 * vtx + 1] = yc;
         ws_tet_vert_[3 * vtx + 2] = zc;

         ws_tet_vert_[3 * vtx + 12] =  xc;
         ws_tet_vert_[3 * vtx + 13] = -yc;
         ws_tet_vert_[3 * vtx + 14] = -zc;

         vtx++;
      }
      vtx += 4;
   }
   */
   /*
   for (int k=0; k<4; k++)
   {
     ws_vert_tet_[3 * k + 6] = -0.5 * a_;
     ws_vert_tet_[3 * k + 7] =  0.5 * a_ * (2 * (((k + 1) / 2) % 2) - 1);
     ws_vert_tet_[3 * k + 8] =  0.5 * a_ * (2 * (k / 2) - 1);
   }
   for (int k=0; k<4; k++)
   {
     ws_vert_tet_[3 * k + 18] = -0.5 * a_;
     ws_vert_tet_[3 * k + 19] =  0.5 * a_ * (k % 2) * (2 - k);
     ws_vert_tet_[3 * k + 20] =  0.5 * a_ * (1 - k % 2) * (k - 1);
   }
   */
   /*
   for (int k=0; k<6; k++)
   {
     for (int j=0; j<4; j++)
     {
       for (int i=0; i<2; i++)
       {
    ws_e2v_tet_[4 * (8 * k + 2 * j + i) + 0] = 0;
    ws_e2v_tet_[4 * (8 * k + 2 * j + i) + 1] = 0;
    ws_e2v_tet_[4 * (8 * k + 2 * j + i) + 2] = 0;
    ws_e2v_tet_[4 * (8 * k + 2 * j + i) + 3] = 0;
       }
     }
   }
   */
   /*
   int elm = 0;
   // Elements touching plane at x = -a/2
   for (int i=0; i<8; i++)
   {
      ws_tet_e2v_[4*elm+0] = 0;
      ws_tet_e2v_[4*elm+1] = 1;
      ws_tet_e2v_[4*elm+2] = i + 2;
      ws_tet_e2v_[4*elm+3] = (i + 7) % 8 + 2;

      ws_tet_be2v_[3*elm+0] = 1;
      ws_tet_be2v_[3*elm+1] = i + 2;
      ws_tet_be2v_[3*elm+2] = (i + 7) % 8 + 2;

      elm++;
   }
   // Elements touching plane at x = a/2
   for (int i=0; i<8; i++)
   {
      ws_tet_e2v_[4*elm+0] = 0;
      ws_tet_e2v_[4*elm+1] = 26;
      ws_tet_e2v_[4*elm+2] = (i + 7) % 8 + 18;
      ws_tet_e2v_[4*elm+3] = i + 18;

      ws_tet_be2v_[3*elm+0] = 26;
      ws_tet_be2v_[3*elm+1] = (i + 7) % 8 + 18;
      ws_tet_be2v_[3*elm+2] = i + 18;

      elm++;
   }
   // Elements touching plane at y = -a/2
   int f2[8] = {2,10,18,25,24,16,8,9};
   for (int i=0; i<8; i++)
   {
      ws_tet_e2v_[4*elm+0] = 0;
      ws_tet_e2v_[4*elm+1] = 17;
      ws_tet_e2v_[4*elm+2] = f2[i];
      ws_tet_e2v_[4*elm+3] = f2[(i + 1) % 8];

      ws_tet_be2v_[3*elm+0] = 17;
      ws_tet_be2v_[3*elm+1] = f2[i];
      ws_tet_be2v_[3*elm+2] = f2[(i + 1) % 8];

      elm++;
   }
   // Elements touching plane at y = a/2
   int f3[8] = {4,5,6,14,22,21,20,12};
   for (int i=0; i<8; i++)
   {
      ws_tet_e2v_[4*elm+0] = 0;
      ws_tet_e2v_[4*elm+1] = 13;
      ws_tet_e2v_[4*elm+2] = f3[i];
      ws_tet_e2v_[4*elm+3] = f3[(i + 1) % 8];

      ws_tet_be2v_[3*elm+0] = 13;
      ws_tet_be2v_[3*elm+1] = f3[i];
      ws_tet_be2v_[3*elm+2] = f3[(i + 1) % 8];

      elm++;
   }
   // Elements touching plane at z = -a/2
   int f4[8] = {2,3,4,12,20,19,18,10};
   for (int i=0; i<8; i++)
   {
      ws_tet_e2v_[4*elm+0] = 0;
      ws_tet_e2v_[4*elm+1] = 11;
      ws_tet_e2v_[4*elm+2] = f4[i];
      ws_tet_e2v_[4*elm+3] = f4[(i + 1) % 8];

      ws_tet_be2v_[3*elm+0] = 11;
      ws_tet_be2v_[3*elm+1] = f4[i];
      ws_tet_be2v_[3*elm+2] = f4[(i + 1) % 8];

      elm++;
   }
   // Elements touching plane at z = a/2
   int f5[8] = {6,7,8,16,24,23,22,14};
   for (int i=0; i<8; i++)
   {
      ws_tet_e2v_[4*elm+0] = 0;
      ws_tet_e2v_[4*elm+1] = 15;
      ws_tet_e2v_[4*elm+2] = f5[i];
      ws_tet_e2v_[4*elm+3] = f5[(i + 1) % 8];

      ws_tet_be2v_[3*elm+0] = 15;
      ws_tet_be2v_[3*elm+1] = f5[i];
      ws_tet_be2v_[3*elm+2] = f5[(i + 1) % 8];

      elm++;
   }

   for (int i=0; i<48; i++) { ws_tet_elem_att_[i]  = 1; }
   for (int i=0; i<48; i++) { ws_tet_belem_att_[i] = 1; }
   */
}

bool
CubicLattice::MapToFundamentalDomain(const Vector & pt, Vector & ipt) const
{
   bool map = false;
   ipt = pt;
   if ( ipt[2] < 0.0 )
   {
      ipt[2] *= -1.0;
      map = true;
   }
   if ( ipt[1] < 0.0 )
   {
      ipt[1] *= -1.0;
      map = true;
   }
   if ( ipt[0] < 0.0 )
   {
      ipt[0] *= -1.0;
      map = true;
   }
   if ( ipt[0] > ipt[1] )
   {
      double d = ipt[0];
      ipt[0] = ipt[1];
      ipt[1] = d;
      map = true;
   }
   if ( ipt[2] > ipt[1] )
   {
      double d = ipt[2];
      ipt[2] = ipt[1];
      ipt[1] = d;
      map = true;
   }
   if ( ipt[0] < ipt[2] )
   {
      double d = ipt[0];
      ipt[0] = ipt[2];
      ipt[2] = d;
      map = true;
   }
   return map;
}

const DenseMatrix & CubicLattice::GetTransformation(int ti) const
{
   int ir = ti % 6;
   int iq = ti / 6;

   T_ = 0.0;

   if ( ir % 2 == 0 )
   {
      for (int i=0; i<3; i++)
      {
         T_(i, (i + (ir / 2)) % 3) = 1.0;
      }
   }
   else
   {
      for (int i=0; i<3; i++)
      {
         T_(i, (3 - i + (ir / 2)) % 3) = 1.0;
      }
   }

   for (int i=0; i<3; i++)
   {
      if ( iq & (int)pow(2, i) )
      {
         T_(i,0) *= -1.0;
         T_(i,1) *= -1.0;
         T_(i,2) *= -1.0;
      }
   }

   return T_;
}

Mesh *
CubicLattice::GetFundamentalDomainMesh() const
{
   Mesh * mesh = new Mesh((double*)fd_vert_, 4,
                          (int*)fd_e2v_, Geometry::TETRAHEDRON,
                          (int*)fd_elem_att_, 1,
                          (int*)fd_be2v_, Geometry::TRIANGLE,
                          (int*)fd_belem_att_, 4,
                          3, 3);
   mesh->Finalize();

   return mesh;
}
/*
Mesh *
CubicLattice::GetWignerSeitzMesh(bool tetMesh) const
{
   Mesh * mesh = NULL;

   if ( tetMesh )
   {
      mesh = new Mesh((double*)ws_tet_vert_, 27,
                      (int*)ws_tet_e2v_, Geometry::TETRAHEDRON,
                      (int*)ws_tet_elem_att_, 48,
                      (int*)ws_tet_be2v_, Geometry::TRIANGLE,
                      (int*)ws_tet_belem_att_, 48,
                      3, 3);
      mesh->FinalizeTetMesh(1,1,true);
      mesh->ReorientTetMesh();
   }
   else
   {
      mesh = new Mesh((double*)ws_vert_, 8,
                      (int*)ws_e2v_, Geometry::CUBE,
                      (int*)ws_elem_att_, 1,
                      (int*)ws_be2v_, Geometry::SQUARE,
                      (int*)ws_belem_att_, 6,
                      3, 3);
      mesh->Finalize();
   }

   return mesh;
}

Mesh *
CubicLattice::GetPeriodicWignerSeitzMesh(bool tetMesh) const
{
   Mesh * mesh = this->GetWignerSeitzMesh(tetMesh);
   cout << "gpwsm 1" << endl;
   mesh->UniformRefinement();
   cout << "gpwsm 2" << endl;
   mesh->UniformRefinement();
   cout << "gpwsm 3" << endl;
   Mesh * per_mesh = MakePeriodicMesh(mesh, trn_vecs_);
   cout << "gpwsm 4" << endl;
   delete mesh;

   return per_mesh;
}
*/
FaceCenteredCubicLattice::FaceCenteredCubicLattice(double a)
   : BravaisLattice3D(a, a, a, 0.5 * M_PI, 0.5 * M_PI, 0.5 * M_PI)
{
   label_ = "FCC";
   type_  = FACE_CENTERED_CUBIC;

   // Set Lattice Vectors
   lat_vecs_[0][0] =  0.0; lat_vecs_[0][1] =  0.5; lat_vecs_[0][2] =  0.5;
   lat_vecs_[1][0] =  0.5; lat_vecs_[1][1] =  0.0; lat_vecs_[1][2] =  0.5;
   lat_vecs_[2][0] =  0.5; lat_vecs_[2][1] =  0.5; lat_vecs_[2][2] =  0.0;

   // Set Reciprocal Lattice Vectors
   rec_vecs_[0][0] = -1.0; rec_vecs_[0][1] =  1.0; rec_vecs_[0][2] =  1.0;
   rec_vecs_[1][0] =  1.0; rec_vecs_[1][1] = -1.0; rec_vecs_[1][2] =  1.0;
   rec_vecs_[2][0] =  1.0; rec_vecs_[2][1] =  1.0; rec_vecs_[2][2] = -1.0;

   // Set Translation Vectors
   trn_vecs_.resize(6);
   for (unsigned int i=0; i<trn_vecs_.size(); i++)
   {
      trn_vecs_[i].SetSize(3);
   }

   trn_vecs_[0][0] =  0.5; trn_vecs_[0][1] =  0.5; trn_vecs_[0][2] =  0.0;
   trn_vecs_[1][0] =  0.5; trn_vecs_[1][1] = -0.5; trn_vecs_[1][2] =  0.0;
   trn_vecs_[2][0] =  0.5; trn_vecs_[2][1] =  0.0; trn_vecs_[2][2] =  0.5;
   trn_vecs_[3][0] =  0.0; trn_vecs_[3][1] =  0.5; trn_vecs_[3][2] =  0.5;
   trn_vecs_[4][0] = -0.5; trn_vecs_[4][1] =  0.0; trn_vecs_[4][2] =  0.5;
   trn_vecs_[5][0] =  0.0; trn_vecs_[5][1] = -0.5; trn_vecs_[5][2] =  0.5;
   /*
   trn_vecs_[0][0] =  0.5; trn_vecs_[0][1] =  0.0; trn_vecs_[0][2] =  0.5;
   trn_vecs_[1][0] =  0.0; trn_vecs_[1][1] =  0.5; trn_vecs_[1][2] =  0.5;
   trn_vecs_[2][0] =  0.0; trn_vecs_[2][1] = -0.5; trn_vecs_[2][2] =  0.5;
   trn_vecs_[3][0] = -0.5; trn_vecs_[3][1] =  0.0; trn_vecs_[3][2] =  0.5;
   trn_vecs_[4][0] =  0.5; trn_vecs_[4][1] = -0.5; trn_vecs_[4][2] =  0.0;
   trn_vecs_[5][0] =  0.5; trn_vecs_[5][1] =  0.5; trn_vecs_[5][2] =  0.0;
   */
   for (unsigned int i=0; i<lat_vecs_.size(); i++)
   {
      lat_vecs_[i] *= a_;
      rec_vecs_[i] *= 1.0/a_;
   }
   for (unsigned int i=0; i<trn_vecs_.size(); i++)
   {
      trn_vecs_[i] *= a_;
   }

   // Set the face radii
   face_radii_.resize(6);
   for (int i=0; i<6; i++) { face_radii_[i] = 0.5 * a_ / sqrt(6.0); }

   this->SetCellVolumes();

   // Allocate sl_, sp_, and path_ vectors
   this->SetVectorSizes();

   // Set Symmetry Points and Labels
   sl_[0] = "Gamma";
   sp_[0] = 0.0;

   sl_[1] = "X";
   add(0.5  ,rec_vecs_[0],0.5  ,rec_vecs_[2],sp_[1]);

   sl_[2] = "W";
   add(0.5  ,rec_vecs_[0],0.25 ,rec_vecs_[1],sp_[2]);
   sp_[2].Add(0.75 ,rec_vecs_[2]);

   sl_[3] = "K";
   add(0.375,rec_vecs_[0],0.375,rec_vecs_[1],sp_[3]);
   sp_[3].Add(0.75 ,rec_vecs_[2]);

   sl_[4] = "L";
   add(0.5  ,rec_vecs_[0],0.5  ,rec_vecs_[1],sp_[4]);
   sp_[4].Add(0.5  ,rec_vecs_[2]);

   sl_[5] = "U";
   add(0.625,rec_vecs_[0],0.25 ,rec_vecs_[1],sp_[5]);
   sp_[5].Add(0.625,rec_vecs_[2]);

   for (unsigned int i=0; i<sl_.size(); i++)
   {
      si_[sl_[i]] = i;
   }

   // Define Paths
   path_[0][0] = 0;
   path_[0][1] = 1;
   path_[0][2] = 2;
   path_[0][3] = 3;
   path_[0][4] = 0;
   path_[0][5] = 4;
   path_[0][6] = 5;
   path_[0][7] = 2;
   path_[0][8] = 4;
   path_[0][9] = 3;

   path_[1][0] = 5;
   path_[1][1] = 1;

   // Set Intermediate Symmetry Points
   this->SetIntermediatePoints();

   // Set Intermediate Symmetry Point Labels
   il_[0][0] = "Delta";  // Gamma -> X
   il_[0][1] = "Z";      // X     -> W
   il_[0][2] = "WK";     // W     -> K
   il_[0][3] = "Sigma";  // K     -> Gamma
   il_[0][4] = "Lambda"; // Gamma -> L
   il_[0][5] = "LU";     // L     -> U
   il_[0][6] = "UW";     // U     -> W
   il_[0][7] = "Q";      // W     -> L
   il_[0][8] = "LK";     // L     -> K
   il_[1][0] = "T";      // U     -> X

   // Set Mesh data
   fd_vert_[0] = 0.0;       fd_vert_[ 1] = 0.0;       fd_vert_[ 2] = 0.0;
   fd_vert_[3] = 0.5 * a_;  fd_vert_[ 4] = 0.0;       fd_vert_[ 5] = 0.0;
   fd_vert_[6] = 0.25 * a_; fd_vert_[ 7] = 0.25 * a_; fd_vert_[ 8] = 0.0;
   fd_vert_[9] = 0.25 * a_; fd_vert_[10] = 0.25 * a_; fd_vert_[11] = 0.25 * a_;

   fd_e2v_[0] = 0; fd_e2v_[1] = 1; fd_e2v_[2] = 2; fd_e2v_[3] = 3;
   fd_elem_att_[0] = 1;

   fd_be2v_[0] = 0; fd_be2v_[ 1] = 2; fd_be2v_[ 2] = 1;
   fd_be2v_[3] = 0; fd_be2v_[ 4] = 1; fd_be2v_[ 5] = 3;
   fd_be2v_[6] = 1; fd_be2v_[ 7] = 2; fd_be2v_[ 8] = 3;
   fd_be2v_[9] = 2; fd_be2v_[10] = 0; fd_be2v_[11] = 3;

   fd_belem_att_[0] = 10; fd_belem_att_[1] = 10;
   fd_belem_att_[2] =  1; fd_belem_att_[3] = 10;
   /*
   for (int i=0; i<45; i++) { ws_vert_[i] = 0.0; }
   for (int d=0; d<3; d++)
   {
      ws_vert_[6 * d + d + 3] = -0.5 * a_;
      ws_vert_[6 * d + d + 6] =  0.5 * a_;
   }
   for (int i=0; i<2; i++)
   {
      for (int j=0; j<2; j++)
      {
         for (int k=0; k<2; k++)
         {
            int v = 4 * i + 2 * j + k + 7;
            ws_vert_[3 * v + 0] = (0.5 * i - 0.25)* a_;
            ws_vert_[3 * v + 1] = (0.5 * j - 0.25)* a_;
            ws_vert_[3 * v + 2] = (0.5 * k - 0.25)* a_;
         }
      }
   }

   ws_e2v_[ 0] =  0; ws_e2v_[ 1] =  9; ws_e2v_[ 2] =  5; ws_e2v_[ 3] = 11;
   ws_e2v_[ 4] =  8; ws_e2v_[ 5] =  1; ws_e2v_[ 6] =  7; ws_e2v_[ 7] =  3;

   ws_e2v_[ 8] =  0; ws_e2v_[ 9] = 11; ws_e2v_[10] =  5; ws_e2v_[11] =  9;
   ws_e2v_[12] = 14; ws_e2v_[13] =  2; ws_e2v_[14] = 13; ws_e2v_[15] =  4;

   ws_e2v_[16] =  0; ws_e2v_[17] =  8; ws_e2v_[18] =  6; ws_e2v_[19] = 14;
   ws_e2v_[20] =  9; ws_e2v_[21] =  1; ws_e2v_[22] = 10; ws_e2v_[23] =  4;

   ws_e2v_[24] =  0; ws_e2v_[25] = 14; ws_e2v_[26] =  6; ws_e2v_[27] =  8;
   ws_e2v_[28] = 11; ws_e2v_[29] =  2; ws_e2v_[30] = 12; ws_e2v_[31] =  3;

   for (int i=0; i<4; i++) { ws_elem_att_[i] = 1; }

   ws_be2v_[ 0] =  8; ws_be2v_[ 1] =  1; ws_be2v_[ 2] =  7; ws_be2v_[ 3] =  3;
   ws_be2v_[ 4] = 14; ws_be2v_[ 5] =  2; ws_be2v_[ 6] = 13; ws_be2v_[ 7] =  4;
   ws_be2v_[ 8] =  9; ws_be2v_[ 9] =  1; ws_be2v_[10] = 10; ws_be2v_[11] =  4;
   ws_be2v_[12] = 11; ws_be2v_[13] =  2; ws_be2v_[14] = 12; ws_be2v_[15] =  3;

   ws_be2v_[16] =  5; ws_be2v_[17] =  7; ws_be2v_[18] =  1; ws_be2v_[19] =  9;
   ws_be2v_[20] =  5; ws_be2v_[21] = 13; ws_be2v_[22] =  2; ws_be2v_[23] = 11;
   ws_be2v_[24] =  5; ws_be2v_[25] = 11; ws_be2v_[26] =  3; ws_be2v_[27] =  7;
   ws_be2v_[28] =  5; ws_be2v_[29] =  9; ws_be2v_[30] =  4; ws_be2v_[31] = 13;

   ws_be2v_[32] =  6; ws_be2v_[33] = 10; ws_be2v_[34] =  1; ws_be2v_[35] =  8;
   ws_be2v_[36] =  6; ws_be2v_[37] = 12; ws_be2v_[38] =  2; ws_be2v_[39] = 14;
   ws_be2v_[40] =  6; ws_be2v_[41] =  8; ws_be2v_[42] =  3; ws_be2v_[43] = 12;
   ws_be2v_[44] =  6; ws_be2v_[45] = 14; ws_be2v_[46] =  4; ws_be2v_[47] = 10;

   for (int i=0; i<12; i++) { ws_belem_att_[i] = 1; }
   */
}

bool
FaceCenteredCubicLattice::MapToFundamentalDomain(const Vector & pt,
                                                 Vector & ipt) const
{
   bool map = false;
   ipt = pt;
   return map;
}

const DenseMatrix & FaceCenteredCubicLattice::GetTransformation(int ti) const
{
   int ir = ti % 6;
   int iq = ti / 6;

   T_ = 0.0;

   if ( ir % 2 == 0 )
   {
      for (int i=0; i<3; i++)
      {
         T_(i, (i + (ir / 2)) % 3) = 1.0;
      }
   }
   else
   {
      for (int i=0; i<3; i++)
      {
         T_(i, (3 - i + (ir / 2)) % 3) = 1.0;
      }
   }

   for (int i=0; i<3; i++)
   {
      if ( iq & (int)pow(2, i) )
      {
         T_(i,0) *= -1.0;
         T_(i,1) *= -1.0;
         T_(i,2) *= -1.0;
      }
   }

   return T_;
}

Mesh *
FaceCenteredCubicLattice::GetFundamentalDomainMesh() const
{
   Mesh * mesh = new Mesh((double*)fd_vert_, 4,
                          (int*)fd_e2v_, Geometry::TETRAHEDRON,
                          (int*)fd_elem_att_, 1,
                          (int*)fd_be2v_, Geometry::TRIANGLE,
                          (int*)fd_belem_att_, 4,
                          3, 3);
   mesh->Finalize();

   return mesh;
}
/*
Mesh *
FaceCenteredCubicLattice::GetWignerSeitzMesh(bool tetMesh) const
{
   Mesh * mesh = new Mesh((double*)ws_vert_, 15,
                          (int*)ws_e2v_, Geometry::CUBE,
                          (int*)ws_elem_att_, 4,
                          (int*)ws_be2v_, Geometry::SQUARE,
                          (int*)ws_belem_att_, 12,
                          3, 3);
   mesh->Finalize();

   std::ofstream ofs("fcc.mesh");
   mesh->Print(ofs);
   ofs.close();

   return mesh;
}

Mesh *
FaceCenteredCubicLattice::GetPeriodicWignerSeitzMesh(bool tetMesh) const
{
   Mesh * mesh = this->GetWignerSeitzMesh(tetMesh);

   mesh->UniformRefinement();

   Mesh * per_mesh = MakePeriodicMesh(mesh, trn_vecs_);

   delete mesh;

   std::ofstream ofs("fcc_per.mesh");
   per_mesh->Print(ofs);
   ofs.close();

   return per_mesh;
}
*/
BodyCenteredCubicLattice::BodyCenteredCubicLattice(double a)
   : BravaisLattice3D(a, a, a, 0.5 * M_PI, 0.5 * M_PI, 0.5 * M_PI)
{
   label_ = "BCC";
   type_  = BODY_CENTERED_CUBIC;

   // Set Lattice Vectors
   lat_vecs_[0][0] = -0.5; lat_vecs_[0][1] =  0.5; lat_vecs_[0][2] =  0.5;
   lat_vecs_[1][0] =  0.5; lat_vecs_[1][1] = -0.5; lat_vecs_[1][2] =  0.5;
   lat_vecs_[2][0] =  0.5; lat_vecs_[2][1] =  0.5; lat_vecs_[2][2] = -0.5;

   // Set Reciprocal Lattice Vectors
   rec_vecs_[0][0] =  0.0; rec_vecs_[0][1] =  1.0; rec_vecs_[0][2] =  1.0;
   rec_vecs_[1][0] =  1.0; rec_vecs_[1][1] =  0.0; rec_vecs_[1][2] =  1.0;
   rec_vecs_[2][0] =  1.0; rec_vecs_[2][1] =  1.0; rec_vecs_[2][2] =  0.0;

   // Set Translation Vectors
   trn_vecs_.resize(7);
   for (unsigned int i=0; i<trn_vecs_.size(); i++)
   {
      trn_vecs_[i].SetSize(3);
   }
   trn_vecs_[0][0] =  1.0; trn_vecs_[0][1] =  0.0; trn_vecs_[0][2] =  0.0;
   trn_vecs_[1][0] =  0.0; trn_vecs_[1][1] =  1.0; trn_vecs_[1][2] =  0.0;
   trn_vecs_[2][0] =  0.0; trn_vecs_[2][1] =  0.0; trn_vecs_[2][2] =  1.0;
   trn_vecs_[3][0] =  0.5; trn_vecs_[3][1] =  0.5; trn_vecs_[3][2] =  0.5;
   trn_vecs_[4][0] = -0.5; trn_vecs_[4][1] =  0.5; trn_vecs_[4][2] =  0.5;
   trn_vecs_[5][0] = -0.5; trn_vecs_[5][1] = -0.5; trn_vecs_[5][2] =  0.5;
   trn_vecs_[6][0] =  0.5; trn_vecs_[6][1] = -0.5; trn_vecs_[6][2] =  0.5;

   for (unsigned int i=0; i<lat_vecs_.size(); i++)
   {
      lat_vecs_[i] *= a_;
      rec_vecs_[i] *= 1.0/a_;
   }
   for (unsigned int i=0; i<trn_vecs_.size(); i++)
   {
      trn_vecs_[i] *= a_;
   }

   // Set the face radii
   face_radii_.resize(7);
   for (int i=0; i<3; i++) { face_radii_[i] = 0.25 * a_ / M_SQRT2; }
   for (int i=3; i<7; i++) { face_radii_[i] = 0.25 * a_ * sqrt(1.5); }

   this->SetCellVolumes();

   // Allocate sl_, sp_, and path_ vectors
   this->SetVectorSizes();

   // Set Symmetry Points and Labels
   sl_[0] = "Gamma";
   sp_[0] = 0.0;

   sl_[1] = "H";
   add(0.5  ,rec_vecs_[0],-0.5  ,rec_vecs_[1],sp_[1]);
   sp_[1].Add(0.5  ,rec_vecs_[2]);

   sl_[2] = "N";
   sp_[2].Set(0.5,rec_vecs_[2]);

   sl_[3] = "P";
   add(0.25 ,rec_vecs_[0], 0.25 ,rec_vecs_[1],sp_[3]);
   sp_[3].Add(0.25 ,rec_vecs_[2]);

   for (unsigned int i=0; i<sl_.size(); i++)
   {
      si_[sl_[i]] = i;
   }

   // Define Paths
   path_[0][0] = 0;
   path_[0][1] = 1;
   path_[0][2] = 2;
   path_[0][3] = 0;
   path_[0][4] = 3;
   path_[0][5] = 1;

   path_[1][0] = 3;
   path_[1][1] = 2;

   // Set Intermediate Symmetry Points
   this->SetIntermediatePoints();

   // Set Intermediate Symmetry Point Labels
   il_[0][0] = "Delta";  // Gamma -> H
   il_[0][1] = "G";      // H     -> N
   il_[0][2] = "Sigma";  // N     -> Gamma
   il_[0][3] = "Lambda"; // Gamma -> P
   il_[0][4] = "F";      // P     -> H
   il_[1][0] = "D";      // P     -> N

   // Set Mesh data
   fd_vert_[ 0] = 0.0;
   fd_vert_[ 1] = 0.0;
   fd_vert_[ 2] = 0.0;

   fd_vert_[ 3] = 0.375 * a_;
   fd_vert_[ 4] = 0.375 * a_;
   fd_vert_[ 5] = 0.0;

   fd_vert_[ 6] = 0.25 * a_;
   fd_vert_[ 7] = 0.25 * a_;
   fd_vert_[ 8] = 0.25 * a_;

   fd_vert_[ 9] = 0.5 * a_;
   fd_vert_[10] = 0.0;
   fd_vert_[11] = 0.0;

   fd_vert_[12] = 0.5 * a_;
   fd_vert_[13] = 0.25 * a_;
   fd_vert_[14] = 0.0;

   fd_vert_[15] = 0.5 * a_;
   fd_vert_[16] = 0.125 * a_;
   fd_vert_[17] = 0.125 * a_;
   /*
   fd_e2v_[0] = 0; fd_e2v_[1] = 1; fd_e2v_[2] = 2;
   fd_e2v_[3] = 3; fd_e2v_[4] = 4; fd_e2v_[5] = 5;
   fd_elem_att_[0] = 1;

   fd_be2v_[ 0] = 0; fd_be2v_[ 1] = 2; fd_be2v_[ 2] = 1;
   fd_be2v_[ 3] = 3; fd_be2v_[ 4] = 4; fd_be2v_[ 5] = 5;
   fd_be2v_[ 6] = 0; fd_be2v_[ 7] = 1; fd_be2v_[ 8] = 4; fd_be2v_[ 9] = 3;
   fd_be2v_[10] = 1; fd_be2v_[11] = 2; fd_be2v_[12] = 5; fd_be2v_[13] = 4;
   fd_be2v_[14] = 2; fd_be2v_[15] = 0; fd_be2v_[16] = 3; fd_be2v_[17] = 5;

   fd_belem_att_[0] = 10; fd_belem_att_[1] =  1; fd_belem_att_[2] = 10;
   fd_belem_att_[3] =  1; fd_belem_att_[4] = 10;
   */
   fd_e2v_[0] = 0; fd_e2v_[1] = 1; fd_e2v_[2] = 2; fd_e2v_[3] = 4;
   fd_e2v_[4] = 0; fd_e2v_[5] = 2; fd_e2v_[6] = 5; fd_e2v_[7] = 3;
   fd_e2v_[8] = 4;
   fd_elem_att_[0] = 1; fd_elem_att_[1] = 1;

   fd_be2v_[ 0] = 0; fd_be2v_[ 1] = 2; fd_be2v_[ 2] = 1;
   fd_be2v_[ 3] = 3; fd_be2v_[ 4] = 4; fd_be2v_[ 5] = 5;
   fd_be2v_[ 6] = 0; fd_be2v_[ 7] = 1; fd_be2v_[ 8] = 4;
   fd_be2v_[ 9] = 0; fd_be2v_[10] = 4; fd_be2v_[11] = 3;
   fd_be2v_[12] = 1; fd_be2v_[13] = 2; fd_be2v_[14] = 4;
   fd_be2v_[15] = 2; fd_be2v_[16] = 5; fd_be2v_[17] = 4;
   fd_be2v_[18] = 2; fd_be2v_[19] = 0; fd_be2v_[20] = 3; fd_be2v_[21] = 5;

   fd_belem_att_[0] = 10; fd_belem_att_[1] =  1; fd_belem_att_[2] = 10;
   fd_belem_att_[3] =  1; fd_belem_att_[4] =  1; fd_belem_att_[5] =  1;
   fd_belem_att_[6] = 10;
   /*
   ws_vert_[  0] = -0.50; ws_vert_[  1] =  0.25; ws_vert_[  2] =  0.00;
   ws_vert_[  3] = -0.50; ws_vert_[  4] =  0.00; ws_vert_[  5] =  0.25;
   ws_vert_[  6] = -0.50; ws_vert_[  7] = -0.25; ws_vert_[  8] =  0.00;
   ws_vert_[  9] = -0.50; ws_vert_[ 10] =  0.00; ws_vert_[ 11] = -0.25;
   ws_vert_[ 12] =  0.50; ws_vert_[ 13] =  0.25; ws_vert_[ 14] =  0.00;
   ws_vert_[ 15] =  0.50; ws_vert_[ 16] =  0.00; ws_vert_[ 17] =  0.25;
   ws_vert_[ 18] =  0.50; ws_vert_[ 19] = -0.25; ws_vert_[ 20] =  0.00;
   ws_vert_[ 21] =  0.50; ws_vert_[ 22] =  0.00; ws_vert_[ 23] = -0.25;
   ws_vert_[ 24] =  0.25; ws_vert_[ 25] = -0.50; ws_vert_[ 26] =  0.00;
   ws_vert_[ 27] =  0.00; ws_vert_[ 28] = -0.50; ws_vert_[ 29] =  0.25;
   ws_vert_[ 30] = -0.25; ws_vert_[ 31] = -0.50; ws_vert_[ 32] =  0.00;
   ws_vert_[ 33] =  0.00; ws_vert_[ 34] = -0.50; ws_vert_[ 35] = -0.25;
   ws_vert_[ 36] =  0.25; ws_vert_[ 37] =  0.50; ws_vert_[ 38] =  0.00;
   ws_vert_[ 39] =  0.00; ws_vert_[ 40] =  0.50; ws_vert_[ 41] =  0.25;
   ws_vert_[ 42] = -0.25; ws_vert_[ 43] =  0.50; ws_vert_[ 44] =  0.00;
   ws_vert_[ 45] =  0.00; ws_vert_[ 46] =  0.50; ws_vert_[ 47] = -0.25;
   ws_vert_[ 48] =  0.25; ws_vert_[ 49] =  0.00; ws_vert_[ 50] = -0.50;
   ws_vert_[ 51] =  0.00; ws_vert_[ 52] =  0.25; ws_vert_[ 53] = -0.50;
   ws_vert_[ 54] = -0.25; ws_vert_[ 55] =  0.00; ws_vert_[ 56] = -0.50;
   ws_vert_[ 57] =  0.00; ws_vert_[ 58] = -0.25; ws_vert_[ 59] = -0.50;
   ws_vert_[ 60] =  0.25; ws_vert_[ 61] =  0.00; ws_vert_[ 62] =  0.50;
   ws_vert_[ 63] =  0.00; ws_vert_[ 64] =  0.25; ws_vert_[ 65] =  0.50;
   ws_vert_[ 66] = -0.25; ws_vert_[ 67] =  0.00; ws_vert_[ 68] =  0.50;
   ws_vert_[ 69] =  0.00; ws_vert_[ 70] = -0.25; ws_vert_[ 71] =  0.50;
   ws_vert_[ 72] = -0.25; ws_vert_[ 73] = -0.25; ws_vert_[ 74] = -0.25;
   ws_vert_[ 75] = -0.25; ws_vert_[ 76] = -0.25; ws_vert_[ 77] =  0.25;
   ws_vert_[ 78] = -0.25; ws_vert_[ 79] =  0.25; ws_vert_[ 80] = -0.25;
   ws_vert_[ 81] = -0.25; ws_vert_[ 82] =  0.25; ws_vert_[ 83] =  0.25;
   ws_vert_[ 84] =  0.25; ws_vert_[ 85] = -0.25; ws_vert_[ 86] = -0.25;
   ws_vert_[ 87] =  0.25; ws_vert_[ 88] = -0.25; ws_vert_[ 89] =  0.25;
   ws_vert_[ 90] =  0.25; ws_vert_[ 91] =  0.25; ws_vert_[ 92] = -0.25;
   ws_vert_[ 93] =  0.25; ws_vert_[ 94] =  0.25; ws_vert_[ 95] =  0.25;
   ws_vert_[ 96] = -0.25; ws_vert_[ 97] =  0.00; ws_vert_[ 98] =  0.00;
   ws_vert_[ 99] =  0.25; ws_vert_[100] =  0.00; ws_vert_[101] =  0.00;
   ws_vert_[102] =  0.00; ws_vert_[103] = -0.25; ws_vert_[104] =  0.00;
   ws_vert_[105] =  0.00; ws_vert_[106] =  0.25; ws_vert_[107] =  0.00;
   ws_vert_[108] =  0.00; ws_vert_[109] =  0.00; ws_vert_[110] = -0.25;
   ws_vert_[111] =  0.00; ws_vert_[112] =  0.00; ws_vert_[113] =  0.25;
   if ( a_ != 1.0 )
      for (int i=0; i<114; i++) { ws_vert_[i] *= a_; }

   ws_e2v_[  0] =  0; ws_e2v_[  1] =  1; ws_e2v_[  2] =  2; ws_e2v_[  3] =  3;
   ws_e2v_[  4] = 26; ws_e2v_[  5] = 32; ws_e2v_[  6] = 24; ws_e2v_[  7] = 18;

   ws_e2v_[  8] = 26; ws_e2v_[  9] = 32; ws_e2v_[ 10] = 24; ws_e2v_[ 11] = 18;
   ws_e2v_[ 12] = 15; ws_e2v_[ 13] = 35; ws_e2v_[ 14] = 36; ws_e2v_[ 15] = 17;

   ws_e2v_[ 16] = 15; ws_e2v_[ 17] = 35; ws_e2v_[ 18] = 36; ws_e2v_[ 19] = 17;
   ws_e2v_[ 20] = 30; ws_e2v_[ 21] = 33; ws_e2v_[ 22] = 28; ws_e2v_[ 23] = 16;

   ws_e2v_[ 24] = 30; ws_e2v_[ 25] = 33; ws_e2v_[ 26] = 28; ws_e2v_[ 27] = 16;
   ws_e2v_[ 28] =  4; ws_e2v_[ 29] =  5; ws_e2v_[ 30] =  6; ws_e2v_[ 31] =  7;

   ws_e2v_[ 32] =  9; ws_e2v_[ 33] =  8; ws_e2v_[ 34] = 11; ws_e2v_[ 35] = 10;
   ws_e2v_[ 36] = 23; ws_e2v_[ 37] = 29; ws_e2v_[ 38] = 34; ws_e2v_[ 39] = 25;

   ws_e2v_[ 40] = 23; ws_e2v_[ 41] = 29; ws_e2v_[ 42] = 34; ws_e2v_[ 43] = 25;
   ws_e2v_[ 44] = 22; ws_e2v_[ 45] = 37; ws_e2v_[ 46] = 32; ws_e2v_[ 47] =  1;

   ws_e2v_[ 48] = 22; ws_e2v_[ 49] = 37; ws_e2v_[ 50] = 32; ws_e2v_[ 51] =  1;
   ws_e2v_[ 52] = 21; ws_e2v_[ 53] = 31; ws_e2v_[ 54] = 35; ws_e2v_[ 55] = 27;

   ws_e2v_[ 56] = 21; ws_e2v_[ 57] = 31; ws_e2v_[ 58] = 35; ws_e2v_[ 59] = 27;
   ws_e2v_[ 60] = 13; ws_e2v_[ 61] = 12; ws_e2v_[ 62] = 15; ws_e2v_[ 63] = 14;

   ws_e2v_[ 64] = 16; ws_e2v_[ 65] = 17; ws_e2v_[ 66] = 18; ws_e2v_[ 67] = 19;
   ws_e2v_[ 68] = 28; ws_e2v_[ 69] = 36; ws_e2v_[ 70] = 24; ws_e2v_[ 71] = 11;

   ws_e2v_[ 72] = 28; ws_e2v_[ 73] = 36; ws_e2v_[ 74] = 24; ws_e2v_[ 75] = 11;
   ws_e2v_[ 76] = 33; ws_e2v_[ 77] = 35; ws_e2v_[ 78] = 32; ws_e2v_[ 79] = 34;

   ws_e2v_[ 80] = 33; ws_e2v_[ 81] = 35; ws_e2v_[ 82] = 32; ws_e2v_[ 83] = 34;
   ws_e2v_[ 84] =  5; ws_e2v_[ 85] = 31; ws_e2v_[ 86] = 37; ws_e2v_[ 87] = 29;

   ws_e2v_[ 88] =  5; ws_e2v_[ 89] = 31; ws_e2v_[ 90] = 37; ws_e2v_[ 91] = 29;
   ws_e2v_[ 92] = 20; ws_e2v_[ 93] = 21; ws_e2v_[ 94] = 22; ws_e2v_[ 95] = 23;

   ws_e2v_[ 96] = 24; ws_e2v_[ 97] = 11; ws_e2v_[ 98] = 34; ws_e2v_[ 99] = 32;
   ws_e2v_[100] =  2; ws_e2v_[101] = 10; ws_e2v_[102] = 25; ws_e2v_[103] =  1;

   ws_e2v_[104] = 11; ws_e2v_[105] = 28; ws_e2v_[106] = 33; ws_e2v_[107] = 34;
   ws_e2v_[108] =  8; ws_e2v_[109] =  6; ws_e2v_[110] =  5; ws_e2v_[111] = 29;

   ws_e2v_[112] = 30; ws_e2v_[113] = 15; ws_e2v_[114] = 35; ws_e2v_[115] = 33;
   ws_e2v_[116] =  4; ws_e2v_[117] = 12; ws_e2v_[118] = 31; ws_e2v_[119] =  5;

   ws_e2v_[120] = 15; ws_e2v_[121] = 26; ws_e2v_[122] = 32; ws_e2v_[123] = 35;
   ws_e2v_[124] = 14; ws_e2v_[125] =  0; ws_e2v_[126] =  1; ws_e2v_[127] = 27;

   for (int i=0; i<16; i++) { ws_elem_att_[i] = 1; }

   ws_be2v_[ 0] =  3; ws_be2v_[ 1] =  2; ws_be2v_[ 2] =  1; ws_be2v_[ 3] =  0;
   ws_be2v_[ 4] =  2; ws_be2v_[ 5] =  3; ws_be2v_[ 6] = 18; ws_be2v_[ 7] = 24;
   ws_be2v_[ 8] =  3; ws_be2v_[ 9] =  0; ws_be2v_[10] = 26; ws_be2v_[11] = 18;
   ws_be2v_[12] =  4; ws_be2v_[13] =  5; ws_be2v_[14] =  6; ws_be2v_[15] =  7;
   ws_be2v_[16] = 16; ws_be2v_[17] = 30; ws_be2v_[18] =  4; ws_be2v_[19] =  7;
   ws_be2v_[20] = 16; ws_be2v_[21] =  7; ws_be2v_[22] =  6; ws_be2v_[23] = 28;
   ws_be2v_[24] = 17; ws_be2v_[25] = 15; ws_be2v_[26] = 30; ws_be2v_[27] = 16;
   ws_be2v_[28] = 15; ws_be2v_[29] = 17; ws_be2v_[30] = 18; ws_be2v_[31] = 26;
   ws_be2v_[32] =  9; ws_be2v_[33] = 10; ws_be2v_[34] = 11; ws_be2v_[35] =  8;
   ws_be2v_[36] =  9; ws_be2v_[37] =  8; ws_be2v_[38] = 29; ws_be2v_[39] = 23;
   ws_be2v_[40] =  9; ws_be2v_[41] = 23; ws_be2v_[42] = 25; ws_be2v_[43] = 10;
   ws_be2v_[44] =  1; ws_be2v_[45] = 25; ws_be2v_[46] = 23; ws_be2v_[47] = 22;
   ws_be2v_[48] =  1; ws_be2v_[49] = 22; ws_be2v_[50] = 21; ws_be2v_[51] = 27;
   ws_be2v_[52] = 13; ws_be2v_[53] = 12; ws_be2v_[54] = 15; ws_be2v_[55] = 14;
   ws_be2v_[56] = 21; ws_be2v_[57] = 31; ws_be2v_[58] = 12; ws_be2v_[59] = 13;
   ws_be2v_[60] = 21; ws_be2v_[61] = 13; ws_be2v_[62] = 14; ws_be2v_[63] = 27;
   ws_be2v_[64] = 16; ws_be2v_[65] = 19; ws_be2v_[66] = 18; ws_be2v_[67] = 17;
   ws_be2v_[68] = 11; ws_be2v_[69] = 24; ws_be2v_[70] = 18; ws_be2v_[71] = 19;
   ws_be2v_[72] = 11; ws_be2v_[73] = 19; ws_be2v_[74] = 16; ws_be2v_[75] = 28;
   ws_be2v_[76] = 20; ws_be2v_[77] = 21; ws_be2v_[78] = 22; ws_be2v_[79] = 23;
   ws_be2v_[80] =  5; ws_be2v_[81] = 31; ws_be2v_[82] = 21; ws_be2v_[83] = 20;
   ws_be2v_[84] =  5; ws_be2v_[85] = 20; ws_be2v_[86] = 23; ws_be2v_[87] = 29;
   ws_be2v_[88] =  2; ws_be2v_[89] = 10; ws_be2v_[90] = 25; ws_be2v_[91] =  1;
   ws_be2v_[92] = 24; ws_be2v_[93] = 11; ws_be2v_[94] = 10; ws_be2v_[95] =  2;
   ws_be2v_[96] =  8; ws_be2v_[97] =  6; ws_be2v_[98] =  5; ws_be2v_[99] = 29;
   ws_be2v_[100] = 11; ws_be2v_[101] = 28; ws_be2v_[102] =  6;
   ws_be2v_[103] =  8;
   ws_be2v_[104] =  4; ws_be2v_[105] = 12; ws_be2v_[106] = 31;
   ws_be2v_[107] =  5;
   ws_be2v_[108] = 30; ws_be2v_[109] = 15; ws_be2v_[110] = 12;
   ws_be2v_[111] =  4;
   ws_be2v_[112] = 15; ws_be2v_[113] = 26; ws_be2v_[114] =  0;
   ws_be2v_[115] = 14;
   ws_be2v_[116] = 14; ws_be2v_[117] =  0; ws_be2v_[118] =  1;
   ws_be2v_[119] = 27;

   for (int i=0; i<30; i++) { ws_belem_att_[i] = 1; }
   */
}

bool
BodyCenteredCubicLattice::MapToFundamentalDomain(const Vector & pt,
                                                 Vector & ipt) const
{
   bool map = false;
   ipt = pt;
   return map;
}

const DenseMatrix & BodyCenteredCubicLattice::GetTransformation(int ti) const
{
   int ir = ti % 6;
   int iq = ti / 6;

   T_ = 0.0;

   if ( ir % 2 == 0 )
   {
      for (int i=0; i<3; i++)
      {
         T_(i, (i + (ir / 2)) % 3) = 1.0;
      }
   }
   else
   {
      for (int i=0; i<3; i++)
      {
         T_(i, (3 - i + (ir / 2)) % 3) = 1.0;
      }
   }

   for (int i=0; i<3; i++)
   {
      if ( iq & (int)pow(2, i) )
      {
         T_(i,0) *= -1.0;
         T_(i,1) *= -1.0;
         T_(i,2) *= -1.0;
      }
   }

   return T_;
}

Mesh *
BodyCenteredCubicLattice::GetFundamentalDomainMesh() const
{
   /*
   Mesh * mesh = new Mesh((double*)fd_vert_, 4,
                          (int*)fd_e2v_, Geometry::TETRAHEDRON,
                          (int*)fd_elem_att_, 1,
                          (int*)fd_be2v_, Geometry::TRIANGLE,
                          (int*)fd_belem_att_, 4,
                          3, 3);
   */
   /*
   Mesh * mesh = new Mesh(3, 6, 1, 5, 3);
   mesh->AddWedge(fd_e2v_, 1);
   mesh->AddBdrTriangle(&fd_be2v_[0], 10);
   mesh->AddBdrTriangle(&fd_be2v_[3],  1);
   mesh->AddBdrQuad(&fd_be2v_[ 6], 10);
   mesh->AddBdrQuad(&fd_be2v_[10],  1);
   mesh->AddBdrQuad(&fd_be2v_[14], 10);
   */
   Mesh * mesh = new Mesh(3, 6, 2, 7, 3);
   for (int i=0; i<6; i++)
   {
      mesh->AddVertex((const double *)&fd_vert_[3*i]);
   }
   mesh->AddTet(&fd_e2v_[0], 1);
   mesh->AddPyramid(&fd_e2v_[4], 1);
   for (int i=0; i<6; i++)
   {
      mesh->AddBdrTriangle((const int*)&fd_be2v_[3*i], fd_belem_att_[i]);
   }
   mesh->AddBdrQuad((const int*)&fd_be2v_[18], fd_belem_att_[6]);

   mesh->FinalizeMesh(true);

   return mesh;
}
/*
Mesh *
BodyCenteredCubicLattice::GetWignerSeitzMesh(bool tetMesh) const
{
   Mesh * mesh = new Mesh((double*)ws_vert_, 38,
                          (int*)ws_e2v_, Geometry::CUBE,
                          (int*)ws_elem_att_, 16,
                          (int*)ws_be2v_, Geometry::SQUARE,
                          (int*)ws_belem_att_, 30,
                          3, 3);
   mesh->Finalize();

   return mesh;
}

Mesh *
BodyCenteredCubicLattice::GetPeriodicWignerSeitzMesh(bool tetMesh) const
{
   Mesh * mesh = this->GetWignerSeitzMesh(tetMesh);

   mesh->UniformRefinement();

   Mesh * per_mesh = MakePeriodicMesh(mesh, trn_vecs_);

   delete mesh;

   return per_mesh;
}
*/
TetragonalLattice::TetragonalLattice(double a, double c)
   : BravaisLattice3D(a, a, c, 0.5 * M_PI, 0.5 * M_PI, 0.5 * M_PI)
{
   label_ = "TET";
   type_  = PRIMITIVE_TETRAGONAL;

   // Set Lattice Vectors
   lat_vecs_[0][0] =  a_; lat_vecs_[0][1] = 0.0; lat_vecs_[0][2] = 0.0;
   lat_vecs_[1][0] = 0.0; lat_vecs_[1][1] =  a_; lat_vecs_[1][2] = 0.0;
   lat_vecs_[2][0] = 0.0; lat_vecs_[2][1] = 0.0; lat_vecs_[2][2] =  c_;

   // Set Reciprocal Lattice Vectors
   rec_vecs_[0][0] = 1.0 / a_; rec_vecs_[0][1] = 0.0; rec_vecs_[0][2] = 0.0;
   rec_vecs_[1][0] = 0.0; rec_vecs_[1][1] = 1.0 / a_; rec_vecs_[1][2] = 0.0;
   rec_vecs_[2][0] = 0.0; rec_vecs_[2][1] = 0.0; rec_vecs_[2][2] = 1.0 / c_;

   // Set Translation Vectors
   trn_vecs_.resize(3);
   for (unsigned int i=0; i<trn_vecs_.size(); i++)
   {
      trn_vecs_[i].SetSize(3);
   }
   trn_vecs_[0][0] =  a_; trn_vecs_[0][1] = 0.0; trn_vecs_[0][2] = 0.0;
   trn_vecs_[1][0] = 0.0; trn_vecs_[1][1] =  a_; trn_vecs_[1][2] = 0.0;
   trn_vecs_[2][0] = 0.0; trn_vecs_[2][1] = 0.0; trn_vecs_[2][2] =  c_;

   // Set the face radii
   face_radii_.resize(3);
   face_radii_[0] = 0.5 * min(a_,c_);
   face_radii_[1] = 0.5 * min(a_,c_);
   face_radii_[2] = 0.5 * a_;

   this->SetCellVolumes();

   // Allocate sl_, sp_, and path_ vectors
   this->SetVectorSizes();

   // Set Symmetry Points and Labels
   sl_[0] = "Gamma";
   sp_[0] = 0.0;

   sl_[1] = "X";
   sp_[1].Set(0.5,rec_vecs_[1]);

   sl_[2] = "M";
   add(0.5,rec_vecs_[0],0.5,rec_vecs_[1],sp_[2]);

   sl_[3] = "Z";
   sp_[3].Set(0.5,rec_vecs_[2]);

   sl_[4] = "R";
   add(0.5,rec_vecs_[1],0.5,rec_vecs_[2],sp_[4]);

   sl_[5] = "A";
   add(0.5,rec_vecs_[0],0.5,rec_vecs_[1],sp_[5]);
   sp_[5].Add(0.5,rec_vecs_[2]);


   for (unsigned int i=0; i<sl_.size(); i++)
   {
      si_[sl_[i]] = i;
   }

   // Define Paths
   path_[0][0] = 0;
   path_[0][1] = 1;
   path_[0][2] = 2;
   path_[0][3] = 0;
   path_[0][4] = 3;
   path_[0][5] = 4;
   path_[0][6] = 5;
   path_[0][7] = 3;

   path_[1][0] = 1;
   path_[1][1] = 4;

   path_[2][0] = 2;
   path_[2][1] = 5;

   // Set Intermediate Symmetry Points
   this->SetIntermediatePoints();

   // Set Intermediate Symmetry Point Labels
   il_[0][0] = "Delta";  // Gamma -> X
   il_[0][1] = "XM";     // X     -> M
   il_[0][2] = "MGamma"; // M     -> Gamma
   il_[0][3] = "GammaZ"; // Gamma -> Z
   il_[0][4] = "ZR";     // Z     -> R
   il_[0][5] = "RA";     // R     -> A
   il_[0][6] = "AZ";     // A     -> Z
   il_[1][0] = "XR";     // X     -> R
   il_[2][0] = "MA";     // M     -> A

   // Set Mesh data
   fd_vert_[ 0] = 0.0;      fd_vert_[ 1] = 0.0;      fd_vert_[ 2] = 0.0;
   fd_vert_[ 3] = 0.5 * a_; fd_vert_[ 4] = 0.0;      fd_vert_[ 5] = 0.0;
   fd_vert_[ 6] = 0.5 * a_; fd_vert_[ 7] = 0.5 * a_; fd_vert_[ 8] = 0.0;
   for (int i=0; i<3; i++)
   {
      fd_vert_[3 * i +  9] = fd_vert_[3 * i + 0];
      fd_vert_[3 * i + 10] = fd_vert_[3 * i + 1];
      fd_vert_[3 * i + 11] = 0.5 * c_;
   }

   fd_e2v_[0] = 0; fd_e2v_[1] = 1; fd_e2v_[2] = 2;
   fd_e2v_[3] = 3; fd_e2v_[4] = 4; fd_e2v_[5] = 5;
   fd_elem_att_[0] = 1;

   fd_be2v_[ 0] = 2; fd_be2v_[ 1] = 1; fd_be2v_[ 2] = 0;
   fd_be2v_[ 3] = 3; fd_be2v_[ 4] = 4; fd_be2v_[ 5] = 5;

   fd_be2v_[ 6] = 0; fd_be2v_[ 7] = 1; fd_be2v_[ 8] = 4; fd_be2v_[ 9] = 3;
   fd_be2v_[10] = 1; fd_be2v_[11] = 2; fd_be2v_[12] = 5; fd_be2v_[13] = 4;
   fd_be2v_[14] = 2; fd_be2v_[15] = 0; fd_be2v_[16] = 3; fd_be2v_[17] = 5;

   fd_belem_att_[0] = 10; fd_belem_att_[1] = 1;
   fd_belem_att_[2] = 10; fd_belem_att_[3] = 1; fd_belem_att_[4] = 10;
   /*
   ws_vert_[ 0] = -0.5 * a_; ws_vert_[ 1] = -0.5 * a_; ws_vert_[ 2] = -0.5 * c_;
   ws_vert_[ 3] =  0.5 * a_; ws_vert_[ 4] = -0.5 * a_; ws_vert_[ 5] = -0.5 * c_;
   ws_vert_[ 6] =  0.5 * a_; ws_vert_[ 7] =  0.5 * a_; ws_vert_[ 8] = -0.5 * c_;
   ws_vert_[ 9] = -0.5 * a_; ws_vert_[10] =  0.5 * a_; ws_vert_[11] = -0.5 * c_;
   ws_vert_[12] = -0.5 * a_; ws_vert_[13] = -0.5 * a_; ws_vert_[14] =  0.5 * c_;
   ws_vert_[15] =  0.5 * a_; ws_vert_[16] = -0.5 * a_; ws_vert_[17] =  0.5 * c_;
   ws_vert_[18] =  0.5 * a_; ws_vert_[19] =  0.5 * a_; ws_vert_[20] =  0.5 * c_;
   ws_vert_[21] = -0.5 * a_; ws_vert_[22] =  0.5 * a_; ws_vert_[23] =  0.5 * c_;

   for (int i=0; i<8; i++) { ws_e2v_[i] = i; }
   ws_elem_att_[0] = 1;

   ws_be2v_[ 0] = 0; ws_be2v_[ 1] = 3; ws_be2v_[ 2] = 2; ws_be2v_[ 3] = 1;
   ws_be2v_[ 4] = 4; ws_be2v_[ 5] = 5; ws_be2v_[ 6] = 6; ws_be2v_[ 7] = 7;

   ws_be2v_[ 8] = 0; ws_be2v_[ 9] = 1; ws_be2v_[10] = 5; ws_be2v_[11] = 4;
   ws_be2v_[12] = 1; ws_be2v_[13] = 2; ws_be2v_[14] = 6; ws_be2v_[15] = 5;
   ws_be2v_[16] = 2; ws_be2v_[17] = 3; ws_be2v_[18] = 7; ws_be2v_[19] = 6;
   ws_be2v_[20] = 3; ws_be2v_[21] = 0; ws_be2v_[22] = 4; ws_be2v_[23] = 7;

   for (int i=0; i<6; i++) { ws_belem_att_[i] = 1; }
   */
}

bool
TetragonalLattice::MapToFundamentalDomain(const Vector & pt, Vector & ipt) const
{
   bool map = false;
   ipt = pt;
   if ( ipt[2] < 0.0 )
   {
      ipt[2] *= -1.0;
      map = true;
   }
   if ( ipt[1] < 0.0 )
   {
      ipt[1] *= -1.0;
      map = true;
   }
   if ( ipt[0] < 0.0 )
   {
      ipt[0] *= -1.0;
      map = true;
   }
   if ( ipt[0] > ipt[1] )
   {
      double d = ipt[0];
      ipt[0] = ipt[1];
      ipt[1] = d;
      map = true;
   }
   if ( ipt[2] > ipt[1] )
   {
      double d = ipt[2];
      ipt[2] = ipt[1];
      ipt[1] = d;
      map = true;
   }
   if ( ipt[0] < ipt[2] )
   {
      double d = ipt[0];
      ipt[0] = ipt[2];
      ipt[2] = d;
      map = true;
   }
   return map;
}

const DenseMatrix & TetragonalLattice::GetTransformation(int ti) const
{
   int ir = ti % 2;
   int iq = ti / 2;

   T_ = 0.0;

   if ( ir == 0 )
   {
      T_(0, 0) = 1.0;
      T_(1, 1) = 1.0;
   }
   else
   {
      T_(0, 1) = 1.0;
      T_(1, 0) = 1.0;
   }
   T_(2, 2) = 1.0;

   for (int i=0; i<3; i++)
   {
      if ( iq & (int)pow(2, i) )
      {
         T_(i,0) *= -1.0;
         T_(i,1) *= -1.0;
         T_(i,2) *= -1.0;
      }
   }

   return T_;
}

Mesh *
TetragonalLattice::GetFundamentalDomainMesh() const
{
   /*
   Mesh * mesh = new Mesh((double*)fd_vert_, 4,
                          (int*)fd_e2v_, Geometry::TETRAHEDRON,
                          (int*)fd_elem_att_, 1,
                          (int*)fd_be2v_, Geometry::TRIANGLE,
                          (int*)fd_belem_att_, 4,
                          3, 3);
   */
   Mesh * mesh = new Mesh(3, 6, 1, 5, 3);
   for (int i=0; i<6; i++)
   {
      mesh->AddVertex((const double *)&fd_vert_[3*i]);
   }
   mesh->AddWedge(fd_e2v_, 1);
   for (int i=0; i<2; i++)
   {
      mesh->AddBdrTriangle((const int*)&fd_be2v_[3*i], fd_belem_att_[i]);
   }
   for (int i=0; i<3; i++)
   {
      mesh->AddBdrQuad((const int*)&fd_be2v_[4*i+6], fd_belem_att_[i]);
   }

   mesh->FinalizeMesh(true);

   return mesh;
}
/*
Mesh *
TetragonalLattice::GetWignerSeitzMesh(bool tetMesh) const
{
   Mesh * mesh = new Mesh((double*)ws_vert_, 8,
                          (int*)ws_e2v_, Geometry::CUBE,
                          (int*)ws_elem_att_, 1,
                          (int*)ws_be2v_, Geometry::SQUARE,
                          (int*)ws_belem_att_, 6,
                          3, 3);
   mesh->Finalize();

   return mesh;
}

Mesh *
TetragonalLattice::GetPeriodicWignerSeitzMesh(bool tetMesh) const
{
   Mesh * mesh = this->GetWignerSeitzMesh(tetMesh);

   mesh->UniformRefinement();
   mesh->UniformRefinement();

   Mesh * per_mesh = MakePeriodicMesh(mesh, trn_vecs_);

   delete mesh;

   return per_mesh;
}
*/
BodyCenteredTetragonalLattice::BodyCenteredTetragonalLattice(double a, double c)
   : BravaisLattice3D(a, a, c, 0.5 * M_PI, 0.5 * M_PI, 0.5 * M_PI)
{
   label_ = ( c_ < a_ )?"BCT1":((c_ < M_SQRT2 * a_)?"BCT12":"BCT2");

   type_  = BODY_CENTERED_TETRAGONAL;

   // Set Lattice Vectors
   lat_vecs_[0][0] = -0.5 * a_;
   lat_vecs_[0][1] =  0.5 * a_;
   lat_vecs_[0][2] =  0.5 * c_;

   lat_vecs_[1][0] =  0.5 * a_;
   lat_vecs_[1][1] = -0.5 * a_;
   lat_vecs_[1][2] =  0.5 * c_;

   lat_vecs_[2][0] =  0.5 * a_;
   lat_vecs_[2][1] =  0.5 * a_;
   lat_vecs_[2][2] = -0.5 * c_;

   // Set Reciprocal Lattice Vectors
   rec_vecs_[0][0] = 0.0;
   rec_vecs_[0][1] = 1.0 / a_;
   rec_vecs_[0][2] = 1.0 / c_;

   rec_vecs_[1][0] = 1.0 / a_;
   rec_vecs_[1][1] = 0.0;
   rec_vecs_[1][2] = 1.0 / c_;

   rec_vecs_[2][0] = 1.0 / a_;
   rec_vecs_[2][1] = 1.0 / a_;
   rec_vecs_[2][2] = 0.0;

   // Set Translation Vectors
   trn_vecs_.resize((c_>M_SQRT2*a_)?6:7);
   for (unsigned int i=0; i<trn_vecs_.size(); i++)
   {
      trn_vecs_[i].SetSize(3);
   }
   if ( c_ > M_SQRT2 * a_ )
   {
      // Wigner-Seitz cell is an elongated dodecahedron
      trn_vecs_[0][0] =  0.5 * a_;
      trn_vecs_[0][1] =  0.5 * a_;
      trn_vecs_[0][2] =  0.5 * c_;

      trn_vecs_[1][0] = -0.5 * a_;
      trn_vecs_[1][1] =  0.5 * a_;
      trn_vecs_[1][2] =  0.5 * c_;

      trn_vecs_[2][0] = -0.5 * a_;
      trn_vecs_[2][1] = -0.5 * a_;
      trn_vecs_[2][2] =  0.5 * c_;

      trn_vecs_[3][0] =  0.5 * a_;
      trn_vecs_[3][1] = -0.5 * a_;
      trn_vecs_[3][2] =  0.5 * c_;

      trn_vecs_[4][0] =   a_; trn_vecs_[4][1] =  0.0; trn_vecs_[4][2] =  0.0;
      trn_vecs_[5][0] =  0.0; trn_vecs_[5][1] =   a_; trn_vecs_[5][2] =  0.0;

      // Set the face radii
      face_radii_.resize(6);
      for (int i=0; i<4; i++) face_radii_[i] =
            0.25 * a_ * sqrt(c_ * c_ + 2.0 * a_ * a_) /
            sqrt(a_ * a_ + c_ * c_);
      for (int i=4; i<6; i++) face_radii_[i] =
            min(0.50 * a_,
                0.25 * c_ * c_ / sqrt(a_ * a_ + c_ * c_));
   }
   else
   {
      // Wigner-Seitz cell is a truncated octahedron
      trn_vecs_[0][0] =   a_; trn_vecs_[0][1] =  0.0; trn_vecs_[0][2] =  0.0;
      trn_vecs_[1][0] =  0.0; trn_vecs_[1][1] =   a_; trn_vecs_[1][2] =  0.0;
      trn_vecs_[2][0] =  0.0; trn_vecs_[2][1] =  0.0; trn_vecs_[2][2] =   c_;

      trn_vecs_[3][0] =  0.5 * a_;
      trn_vecs_[3][1] =  0.5 * a_;
      trn_vecs_[3][2] =  0.5 * c_;

      trn_vecs_[4][0] = -0.5 * a_;
      trn_vecs_[4][1] =  0.5 * a_;
      trn_vecs_[4][2] =  0.5 * c_;

      trn_vecs_[5][0] = -0.5 * a_;
      trn_vecs_[5][1] = -0.5 * a_;
      trn_vecs_[5][2] =  0.5 * c_;

      trn_vecs_[6][0] =  0.5 * a_;
      trn_vecs_[6][1] = -0.5 * a_;
      trn_vecs_[6][2] =  0.5 * c_;

      // Set the face radii
      face_radii_.resize(7);
      face_radii_[0] = 0.25 * c_ * c_ / sqrt(a_ * a_ + c_ * c_);
      face_radii_[1] = 0.25 * c_ * c_ / sqrt(a_ * a_ + c_ * c_);
      face_radii_[2] = 0.25 * (2.0 * a_ - c_* c_ / a_) / M_SQRT2;
      for (int i=3; i<7; i++)
         face_radii_[i] = min(0.25 * c_* sqrt(1.0 + 0.5 * c_ * c_ / (a_ * a_)),
                              0.125 * sqrt(8.0 * a_ * a_ - 3.0 * c_ * c_ +
                                           pow(c_ * c_ / a_, 2)));
   }

   this->SetCellVolumes();

   // Allocate sl_, sp_, and path_ vectors
   this->SetVectorSizes();

   // Set Symmetry Points and Labels
   if ( c_ < a_ )
   {
      double eta = 0.25 * (1.0 + c_ * c_ / (a_ * a_));

      sl_[0] = "Gamma";
      sp_[0] = 0.0;

      sl_[1] = "X";
      sp_[1].Set(0.5,rec_vecs_[2]);

      sl_[2] = "M";
      add(-0.5,rec_vecs_[0],0.5,rec_vecs_[1],sp_[2]);
      sp_[2].Add(0.5,rec_vecs_[2]);

      sl_[3] = "Z";
      add(eta,rec_vecs_[0],eta,rec_vecs_[1],sp_[3]);
      sp_[3].Add(-eta,rec_vecs_[2]);

      sl_[4] = "N";
      sp_[4].Set(0.5,rec_vecs_[1]);

      sl_[5] = "Z1";
      add(-eta,rec_vecs_[0],1.0-eta,rec_vecs_[1],sp_[5]);
      sp_[5].Add(eta,rec_vecs_[2]);

      sl_[6] = "P";
      add(0.25,rec_vecs_[0],0.25,rec_vecs_[1],sp_[6]);
      sp_[6].Add(0.25,rec_vecs_[2]);
   }
   else
   {
      double  eta = 0.25 * (1.0 + a_ * a_ / (c_ * c_));
      double zeta = 0.5 * a_ * a_ / (c_ * c_);

      sl_[0] = "Gamma";
      sp_[0] = 0.0;

      sl_[1] = "X";
      sp_[1].Set(0.5,rec_vecs_[2]);

      sl_[2] = "N";
      sp_[2].Set(0.5,rec_vecs_[1]);

      sl_[3] = "Y";
      add(-zeta,rec_vecs_[0],zeta,rec_vecs_[1],sp_[3]);
      sp_[3].Add(0.5,rec_vecs_[2]);

      sl_[4] = "P";
      add(0.25,rec_vecs_[0],0.25,rec_vecs_[1],sp_[4]);
      sp_[4].Add(0.25,rec_vecs_[2]);

      sl_[5] = "Y1";
      add(0.5,rec_vecs_[0],0.5,rec_vecs_[1],sp_[5]);
      sp_[5].Add(-zeta,rec_vecs_[2]);

      sl_[6] = "Sigma";
      add(-eta,rec_vecs_[0],eta,rec_vecs_[1],sp_[6]);
      sp_[6].Add(eta,rec_vecs_[2]);

      sl_[7] = "Z";
      add(0.5,rec_vecs_[0],0.5,rec_vecs_[1],sp_[7]);
      sp_[7].Add(-0.5,rec_vecs_[2]);

      sl_[8] = "Sigma1";
      add(eta,rec_vecs_[0],1.0-eta,rec_vecs_[1],sp_[8]);
      sp_[8].Add(-eta,rec_vecs_[2]);
   }

   for (unsigned int i=0; i<sl_.size(); i++)
   {
      si_[sl_[i]] = i;
   }

   // Define Paths
   if ( c_ < a_ )
   {
      path_[0][0] = 0;
      path_[0][1] = 1;
      path_[0][2] = 2;
      path_[0][3] = 0;
      path_[0][4] = 3;
      path_[0][5] = 6;
      path_[0][6] = 4;
      path_[0][7] = 5;
      path_[0][8] = 2;

      path_[1][0] = 1;
      path_[1][1] = 6;
   }
   else
   {
      path_[0][ 0] = 0;
      path_[0][ 1] = 1;
      path_[0][ 2] = 3;
      path_[0][ 3] = 6;
      path_[0][ 4] = 0;
      path_[0][ 5] = 7;
      path_[0][ 6] = 8;
      path_[0][ 7] = 2;
      path_[0][ 8] = 4;
      path_[0][ 9] = 5;
      path_[0][10] = 7;

      path_[1][ 0] = 1;
      path_[1][ 1] = 4;
   }

   // Set Intermediate Symmetry Points
   this->SetIntermediatePoints();

   // Set Intermediate Symmetry Point Labels
   if ( c_ < a_ )
   {
      il_[0][0] = "GammaX"; // Gamma -> X
      il_[0][1] = "XM";     // X     -> M
      il_[0][2] = "MGamma"; // M     -> Gamma
      il_[0][3] = "GammaZ"; // Gamma -> Z
      il_[0][4] = "ZP";     // Z     -> P
      il_[0][5] = "PN";     // P     -> N
      il_[0][6] = "NZ1";    // N     -> Z1
      il_[0][7] = "Z1M";    // Z1    -> M
      il_[1][0] = "XP";     // X     -> P
   }
   else
   {
      il_[0][0] = "GammaX";     // Gamma  -> X
      il_[0][1] = "XY";         // X      -> Y
      il_[0][2] = "YSigma";     // Y      -> Sigma
      il_[0][3] = "SigmaGamma"; // Sigma  -> Gamma
      il_[0][4] = "GammaZ";     // Gamma  -> Z
      il_[0][5] = "ZSigma1";    // Z      -> Sigma1
      il_[0][6] = "Sigma1N";    // Sigma1 -> N
      il_[0][7] = "NP";         // N      -> P
      il_[0][8] = "PY1";        // P      -> Y1
      il_[0][9] = "Y1Z";        // Y1     -> Z
      il_[1][0] = "XP";         // X      -> P
   }

   // Set Mesh data
   if ( c_ > M_SQRT2 * a_ )
   {
      this->createElongatedDodecahedron();
   }
   else
   {
      this->createTruncatedOctahedron();
   }
}

void
BodyCenteredTetragonalLattice::createElongatedDodecahedron()
{
   cout << "Setting up data for elongated dodecahedron" << endl;

   // Data for mesh of the fundamental domain
   double a2 = a_ * a_;
   double a2c = a2 / c_;
   fd_nvert_ = 7;
   for (int i=0; i<3*fd_nvert_; i++) { fd_vert_[i] = 0.0; } // Gamma, etc.
   fd_vert_[ 3] = 0.5 * a_;  fd_vert_[ 4] = fd_vert_[3]; // M
   fd_vert_[ 6] = 0.25 * a_; fd_vert_[ 7] = fd_vert_[6];
   fd_vert_[ 8] = 0.25 * c_; // N
   fd_vert_[ 9] = 0.5 * a_;  fd_vert_[11] = 0.25 * c_; // P
   fd_vert_[12] = 0.5 * a_; // X
   fd_vert_[17] = 0.25 * c_ + 0.5 * a2c; // Z
   fd_vert_[18] = 0.5 * a_;  fd_vert_[19] = fd_vert_[18];
   fd_vert_[20] = 0.25 * c_ - 0.5 * a2c; // Z_1

   fd_e2v_[ 0] = 0; fd_e2v_[ 1] = 3; fd_e2v_[ 2] = 2; fd_e2v_[ 3] = 5;
   fd_e2v_[ 4] = 0; fd_e2v_[ 5] = 3; fd_e2v_[ 6] = 6; fd_e2v_[ 7] = 2;
   fd_e2v_[ 8] = 4; fd_e2v_[ 9] = 3; fd_e2v_[10] = 6; fd_e2v_[11] = 1;
   fd_e2v_[12] = 0;

   fd_elem_att_[0] = 1; fd_elem_att_[1] = 1; fd_elem_att_[2] = 1;

   fd_nbt_ = 8; fd_nbq_ = 1;
   fd_be2v_[ 0] = 0; fd_be2v_[ 1] = 1; fd_be2v_[ 2] = 4;
   fd_be2v_[ 3] = 0; fd_be2v_[ 4] = 4; fd_be2v_[ 5] = 3;
   fd_be2v_[ 6] = 0; fd_be2v_[ 7] = 3; fd_be2v_[ 8] = 5;
   fd_be2v_[ 9] = 0; fd_be2v_[10] = 6; fd_be2v_[11] = 1;
   fd_be2v_[12] = 0; fd_be2v_[13] = 2; fd_be2v_[14] = 6;
   fd_be2v_[15] = 0; fd_be2v_[16] = 5; fd_be2v_[17] = 2;
   fd_be2v_[18] = 2; fd_be2v_[19] = 3; fd_be2v_[20] = 6;
   fd_be2v_[21] = 2; fd_be2v_[22] = 5; fd_be2v_[23] = 3;
   fd_be2v_[24] = 1; fd_be2v_[25] = 6; fd_be2v_[26] = 3; fd_be2v_[27] = 4;

   fd_belem_att_[0] = 10; fd_belem_att_[1] = 10; fd_belem_att_[2] = 10;
   fd_belem_att_[3] = 10; fd_belem_att_[4] = 10; fd_belem_att_[5] = 10;
   fd_belem_att_[6] =  1; fd_belem_att_[7] =  1; fd_belem_att_[8] =  1;
   /*
   // Set Mesh data
   for (int i=0; i<114; i++) { ws_vert_[i] = 0.0; }

   for (int i=0; i<3; i++)
   {
      for (int j=0; j<4; j++)
      {
         ws_vert_[3 * (4 * i + j)]      = 0.5 * a_ * (i - 1);
         ws_vert_[3 * (4 * i + j) + 37] = 0.5 * a_ * (i - 1);

         ws_vert_[3 * (4 * i + j) + 38] = 0.25 * c_ -
                                          0.5 * (a_ * a_ / c_) * (1.0 - fabs(j - 2) + fabs(i - 1));
         ws_vert_[3 * (4 * i + j) + 2]  = -1.0 * ws_vert_[3 * (4 * i + j) + 38];
      }
   }
   for (int i=0; i<3; i++)
   {
      for (int j=1; j<4; j++)
      {
         ws_vert_[3 * (4 * i + j) + 1]  = 0.5 * a_ * (2 - j);
         ws_vert_[3 * (4 * i + j) + 36] = 0.5 * a_ * (2 - j);
      }
   }

   for (int i=0; i<2; i++)
   {
      for (int j=0; j<8; j++)
      {
         ws_e2v_[8 * i + j]      = 4 * i + j;
         ws_e2v_[8 * i + j + 16] = 4 * i + j + 12;
      }
   }

   ws_e2v_[32] =  6; ws_e2v_[33] =  2; ws_e2v_[34] =  3; ws_e2v_[35] =  7;
   ws_e2v_[36] = 18; ws_e2v_[37] = 19; ws_e2v_[38] = 15; ws_e2v_[39] = 14;

   ws_e2v_[40] =  6; ws_e2v_[41] =  7; ws_e2v_[42] = 11; ws_e2v_[43] = 10;
   ws_e2v_[44] = 18; ws_e2v_[45] = 14; ws_e2v_[46] = 13; ws_e2v_[47] = 17;

   ws_e2v_[48] =  6; ws_e2v_[49] = 10; ws_e2v_[50] =  9; ws_e2v_[51] =  5;
   ws_e2v_[52] = 18; ws_e2v_[53] = 17; ws_e2v_[54] = 21; ws_e2v_[55] = 22;

   ws_e2v_[56] =  6; ws_e2v_[57] =  5; ws_e2v_[58] =  1; ws_e2v_[59] =  2;
   ws_e2v_[60] = 18; ws_e2v_[61] = 22; ws_e2v_[62] = 23; ws_e2v_[63] = 19;

   for (int i=64; i<128; i++) { ws_e2v_[i] = -1; }

   for (int i=0; i< 8; i++) { ws_elem_att_[i] =  1; }
   for (int i=8; i<16; i++) { ws_elem_att_[i] = -1; }

   ws_be2v_[ 0] =  4; ws_be2v_[ 1] =  7; ws_be2v_[ 2] =  3; ws_be2v_[ 3] =  0;
   ws_be2v_[ 4] =  4; ws_be2v_[ 5] =  8; ws_be2v_[ 6] = 11; ws_be2v_[ 7] =  7;
   ws_be2v_[ 8] =  4; ws_be2v_[ 9] =  5; ws_be2v_[10] =  9; ws_be2v_[11] =  8;
   ws_be2v_[12] =  4; ws_be2v_[13] =  0; ws_be2v_[14] =  1; ws_be2v_[15] =  5;

   ws_be2v_[16] = 16; ws_be2v_[17] = 19; ws_be2v_[18] = 15; ws_be2v_[19] = 12;
   ws_be2v_[20] = 16; ws_be2v_[21] = 12; ws_be2v_[22] = 13; ws_be2v_[23] = 17;
   ws_be2v_[24] = 16; ws_be2v_[25] = 17; ws_be2v_[26] = 21; ws_be2v_[27] = 20;
   ws_be2v_[28] = 16; ws_be2v_[29] = 20; ws_be2v_[30] = 23; ws_be2v_[31] = 19;

   ws_be2v_[32] =  0; ws_be2v_[33] =  3; ws_be2v_[34] =  2; ws_be2v_[35] =  1;
   ws_be2v_[36] =  8; ws_be2v_[37] =  9; ws_be2v_[38] = 10; ws_be2v_[39] = 11;
   ws_be2v_[40] = 12; ws_be2v_[41] = 15; ws_be2v_[42] = 14; ws_be2v_[43] = 13;
   ws_be2v_[44] = 20; ws_be2v_[45] = 21; ws_be2v_[46] = 22; ws_be2v_[47] = 23;

   ws_be2v_[48] =  2; ws_be2v_[49] =  3; ws_be2v_[50] = 15; ws_be2v_[51] = 19;
   ws_be2v_[52] =  3; ws_be2v_[53] =  7; ws_be2v_[54] = 14; ws_be2v_[55] = 15;
   ws_be2v_[56] =  7; ws_be2v_[57] = 11; ws_be2v_[58] = 13; ws_be2v_[59] = 14;
   ws_be2v_[60] = 11; ws_be2v_[61] = 10; ws_be2v_[62] = 17; ws_be2v_[63] = 13;

   ws_be2v_[64] = 10; ws_be2v_[65] =  9; ws_be2v_[66] = 21; ws_be2v_[67] = 17;
   ws_be2v_[68] =  9; ws_be2v_[69] =  5; ws_be2v_[70] = 22; ws_be2v_[71] = 21;
   ws_be2v_[72] =  5; ws_be2v_[73] =  1; ws_be2v_[74] = 23; ws_be2v_[75] = 22;
   ws_be2v_[76] =  1; ws_be2v_[77] =  2; ws_be2v_[78] = 19; ws_be2v_[79] = 23;

   for (int i=80; i<120; i++) { ws_be2v_[i] = -1; }

   for (int i= 0; i<20; i++) { ws_belem_att_[i] =  1; }
   for (int i=20; i<30; i++) { ws_belem_att_[i] = -1; }
   */
}

void
BodyCenteredTetragonalLattice::createTruncatedOctahedron()
{
   cout << "Setting up data for truncated octahedron" << endl;

   // Data for mesh of the fundamental domain
   double c2 = c_ * c_;
   double c2a = c2 / a_;
   fd_nvert_ = 9;
   for (int i=0; i<3*fd_nvert_; i++) { fd_vert_[i] = 0.0; } // Gamma, etc.
   fd_vert_[ 3] = 0.25 * a_; fd_vert_[4] = fd_vert_[3];
   fd_vert_[ 5] = 0.25 * c_; // N
   fd_vert_[ 6] = 0.5 * a_; fd_vert_[ 8] = 0.25 * c_; // P
   fd_vert_[ 9] = 0.25 * a_ + 0.125 * c2a; fd_vert_[10] = fd_vert_[9]; // Sigma
   fd_vert_[12] = 0.25 * a_ - 0.125 * c2a; fd_vert_[13] = fd_vert_[12];
   fd_vert_[14] = 0.5 * c_; // Sigma_1
   fd_vert_[15] = 0.5 * a_; // X
   fd_vert_[18] = 0.5 * a_; fd_vert_[19] = 0.25 * c2a; // Y
   fd_vert_[21] = 0.5 * a_ - 0.25 * c2a; fd_vert_[23] = 0.5 * c_; // Y_1
   fd_vert_[26] = 0.5 * c_; // Z

   fd_e2v_[ 0] = 0; fd_e2v_[ 1] = 5; fd_e2v_[ 2] = 6; fd_e2v_[ 3] = 2;
   fd_e2v_[ 4] = 3; fd_e2v_[ 5] = 6; fd_e2v_[ 6] = 2; fd_e2v_[ 7] = 1;
   fd_e2v_[ 8] = 0;
   fd_e2v_[ 9] = 0; fd_e2v_[10] = 2; fd_e2v_[11] = 1;
   fd_e2v_[12] = 8; fd_e2v_[13] = 7; fd_e2v_[14] = 4;

   fd_elem_att_[0] = 1; fd_elem_att_[1] = 1; fd_elem_att_[2] = 1;

   fd_nbt_ = 6; fd_nbq_ = 4;
   fd_be2v_[ 0] = 0; fd_be2v_[ 1] = 6; fd_be2v_[ 2] = 5;
   fd_be2v_[ 3] = 0; fd_be2v_[ 4] = 3; fd_be2v_[ 5] = 6;
   fd_be2v_[ 6] = 4; fd_be2v_[ 7] = 8; fd_be2v_[ 8] = 7;
   fd_be2v_[ 9] = 0; fd_be2v_[10] = 5; fd_be2v_[11] = 2;
   fd_be2v_[12] = 0; fd_be2v_[13] = 1; fd_be2v_[14] = 3;
   fd_be2v_[15] = 5; fd_be2v_[16] = 6; fd_be2v_[17] = 2;
   fd_be2v_[18] = 0; fd_be2v_[19] = 2; fd_be2v_[20] = 7; fd_be2v_[21] = 8;
   fd_be2v_[22] = 0; fd_be2v_[23] = 8; fd_be2v_[24] = 4; fd_be2v_[25] = 1;
   fd_be2v_[26] = 1; fd_be2v_[27] = 2; fd_be2v_[28] = 6; fd_be2v_[29] = 3;
   fd_be2v_[30] = 1; fd_be2v_[31] = 4; fd_be2v_[32] = 7; fd_be2v_[33] = 2;

   fd_belem_att_[0] = 10; fd_belem_att_[1] = 10; fd_belem_att_[2] = 1;
   fd_belem_att_[3] = 10; fd_belem_att_[4] = 10; fd_belem_att_[5] = 1;
   fd_belem_att_[6] = 10; fd_belem_att_[7] = 10;
   fd_belem_att_[8] =  1; fd_belem_att_[9] = 1;
   /*
   // Set Mesh data
   for (int i=0; i<114; i++) { ws_vert_[i] = 0.0; }

   ws_vert_[  0] = -0.50 * a_;           ws_vert_[  1] =  0.25 * c_ * c_ / a_;
   ws_vert_[  3] = -0.50 * a_;           ws_vert_[  5] =  0.25 * c_;
   ws_vert_[  6] = -0.50 * a_;           ws_vert_[  7] = -0.25 * c_ * c_ / a_;
   ws_vert_[  9] = -0.50 * a_;           ws_vert_[ 11] = -0.25 * c_;

   ws_vert_[ 12] =  0.50 * a_;           ws_vert_[ 13] =  0.25 * c_ * c_ / a_;
   ws_vert_[ 15] =  0.50 * a_;           ws_vert_[ 17] =  0.25 * c_;
   ws_vert_[ 18] =  0.50 * a_;           ws_vert_[ 19] = -0.25 * c_ * c_ / a_;
   ws_vert_[ 21] =  0.50 * a_;           ws_vert_[ 23] = -0.25 * c_;

   ws_vert_[ 24] =  0.25 * c_ * c_ / a_; ws_vert_[ 25] = -0.50 * a_;
   ws_vert_[ 28] = -0.50 * a_;           ws_vert_[ 29] =  0.25 * c_;
   ws_vert_[ 30] = -0.25 * c_ * c_ / a_; ws_vert_[ 31] = -0.50 * a_;
   ws_vert_[ 34] = -0.50 * a_;           ws_vert_[ 35] = -0.25 * c_;

   ws_vert_[ 36] =  0.25 * c_ * c_ / a_; ws_vert_[ 37] =  0.50 * a_;
   ws_vert_[ 40] =  0.50 * a_;           ws_vert_[ 41] =  0.25 * c_;
   ws_vert_[ 42] = -0.25 * c_ * c_ / a_; ws_vert_[ 43] =  0.50 * a_;
   ws_vert_[ 46] =  0.50 * a_;           ws_vert_[ 47] = -0.25 * c_;

   ws_vert_[ 48] =  0.5 * a_ - 0.25 * c_ * c_ / a_; ws_vert_[ 50] = -0.50 * c_;
   ws_vert_[ 52] =  0.5 * a_ - 0.25 * c_ * c_ / a_; ws_vert_[ 53] = -0.50 * c_;
   ws_vert_[ 54] = -0.5 * a_ + 0.25 * c_ * c_ / a_; ws_vert_[ 56] = -0.50 * c_;
   ws_vert_[ 58] = -0.5 * a_ + 0.25 * c_ * c_ / a_; ws_vert_[ 59] = -0.50 * c_;

   ws_vert_[ 60] =  0.5 * a_ - 0.25 * c_ * c_ / a_; ws_vert_[ 62] =  0.50 * c_;
   ws_vert_[ 64] =  0.5 * a_ - 0.25 * c_ * c_ / a_; ws_vert_[ 65] =  0.50 * c_;
   ws_vert_[ 66] = -0.5 * a_ + 0.25 * c_ * c_ / a_; ws_vert_[ 68] =  0.50 * c_;
   ws_vert_[ 70] = -0.5 * a_ + 0.25 * c_ * c_ / a_; ws_vert_[ 71] =  0.50 * c_;

   ws_vert_[ 72] = -0.5 * a_ + 0.25 * c_ * c_ / a_;
   ws_vert_[ 73] = -0.25 * c_ * c_ / a_;
   ws_vert_[ 74] = -0.25 * c_;

   ws_vert_[ 75] = -0.25 * c_ * c_ / a_;
   ws_vert_[ 76] = -0.5 * a_ + 0.25 * c_ * c_ / a_;
   ws_vert_[ 77] =  0.25 * c_;

   ws_vert_[ 78] = -0.5 * a_ + 0.25 * c_ * c_ / a_;
   ws_vert_[ 79] =  0.25 * c_ * c_ / a_;
   ws_vert_[ 80] = -0.25 * c_;

   ws_vert_[ 81] = -0.25 * c_ * c_ / a_;
   ws_vert_[ 82] =  0.5 * a_ - 0.25 * c_ * c_ / a_;
   ws_vert_[ 83] =  0.25 * c_;

   ws_vert_[ 84] =  0.5 * a_ - 0.25 * c_ * c_ / a_;
   ws_vert_[ 85] = -0.25 * c_ * c_ / a_;
   ws_vert_[ 86] = -0.25 * c_;

   ws_vert_[ 87] =  0.25 * c_ * c_ / a_;
   ws_vert_[ 88] = -0.5 * a_ + 0.25 * c_ * c_ / a_;
   ws_vert_[ 89] =  0.25 * c_;

   ws_vert_[ 90] =  0.5 * a_ - 0.25 * c_ * c_ / a_;
   ws_vert_[ 91] =  0.25 * c_ * c_ / a_;
   ws_vert_[ 92] = -0.25 * c_;

   ws_vert_[ 93] =  0.25 * c_ * c_ / a_;
   ws_vert_[ 94] =  0.5 * a_ - 0.25 * c_ * c_ / a_;
   ws_vert_[ 95] =  0.25 * c_;

   ws_vert_[ 96] = -0.5 * a_ + 0.25 * c_ * c_ / a_;
   ws_vert_[ 99] =  0.5 * a_ - 0.25 * c_ * c_ / a_;
   ws_vert_[103] = -0.5 * a_ + 0.25 * c_ * c_ / a_;
   ws_vert_[106] =  0.5 * a_ - 0.25 * c_ * c_ / a_;

   ws_vert_[109] =  0.5 * a_ - 0.5 * c_ * c_ / a_; ws_vert_[110] = -0.25 * c_;
   ws_vert_[111] = -0.5 * a_ + 0.5 * c_ * c_ / a_; ws_vert_[113] =  0.25 * c_;

   ws_e2v_[  0] =  0; ws_e2v_[  1] =  1; ws_e2v_[  2] =  2; ws_e2v_[  3] =  3;
   ws_e2v_[  4] = 26; ws_e2v_[  5] = 32; ws_e2v_[  6] = 24; ws_e2v_[  7] = 18;

   ws_e2v_[  8] = 26; ws_e2v_[  9] = 32; ws_e2v_[ 10] = 24; ws_e2v_[ 11] = 18;
   ws_e2v_[ 12] = 15; ws_e2v_[ 13] = 35; ws_e2v_[ 14] = 36; ws_e2v_[ 15] = 17;

   ws_e2v_[ 16] = 15; ws_e2v_[ 17] = 35; ws_e2v_[ 18] = 36; ws_e2v_[ 19] = 17;
   ws_e2v_[ 20] = 30; ws_e2v_[ 21] = 33; ws_e2v_[ 22] = 28; ws_e2v_[ 23] = 16;

   ws_e2v_[ 24] = 30; ws_e2v_[ 25] = 33; ws_e2v_[ 26] = 28; ws_e2v_[ 27] = 16;
   ws_e2v_[ 28] =  4; ws_e2v_[ 29] =  5; ws_e2v_[ 30] =  6; ws_e2v_[ 31] =  7;

   ws_e2v_[ 32] =  9; ws_e2v_[ 33] =  8; ws_e2v_[ 34] = 11; ws_e2v_[ 35] = 10;
   ws_e2v_[ 36] = 23; ws_e2v_[ 37] = 29; ws_e2v_[ 38] = 34; ws_e2v_[ 39] = 25;

   ws_e2v_[ 40] = 23; ws_e2v_[ 41] = 29; ws_e2v_[ 42] = 34; ws_e2v_[ 43] = 25;
   ws_e2v_[ 44] = 22; ws_e2v_[ 45] = 37; ws_e2v_[ 46] = 32; ws_e2v_[ 47] =  1;

   ws_e2v_[ 48] = 22; ws_e2v_[ 49] = 37; ws_e2v_[ 50] = 32; ws_e2v_[ 51] =  1;
   ws_e2v_[ 52] = 21; ws_e2v_[ 53] = 31; ws_e2v_[ 54] = 35; ws_e2v_[ 55] = 27;

   ws_e2v_[ 56] = 21; ws_e2v_[ 57] = 31; ws_e2v_[ 58] = 35; ws_e2v_[ 59] = 27;
   ws_e2v_[ 60] = 13; ws_e2v_[ 61] = 12; ws_e2v_[ 62] = 15; ws_e2v_[ 63] = 14;

   ws_e2v_[ 64] = 16; ws_e2v_[ 65] = 17; ws_e2v_[ 66] = 18; ws_e2v_[ 67] = 19;
   ws_e2v_[ 68] = 28; ws_e2v_[ 69] = 36; ws_e2v_[ 70] = 24; ws_e2v_[ 71] = 11;

   ws_e2v_[ 72] = 28; ws_e2v_[ 73] = 36; ws_e2v_[ 74] = 24; ws_e2v_[ 75] = 11;
   ws_e2v_[ 76] = 33; ws_e2v_[ 77] = 35; ws_e2v_[ 78] = 32; ws_e2v_[ 79] = 34;

   ws_e2v_[ 80] = 33; ws_e2v_[ 81] = 35; ws_e2v_[ 82] = 32; ws_e2v_[ 83] = 34;
   ws_e2v_[ 84] =  5; ws_e2v_[ 85] = 31; ws_e2v_[ 86] = 37; ws_e2v_[ 87] = 29;

   ws_e2v_[ 88] =  5; ws_e2v_[ 89] = 31; ws_e2v_[ 90] = 37; ws_e2v_[ 91] = 29;
   ws_e2v_[ 92] = 20; ws_e2v_[ 93] = 21; ws_e2v_[ 94] = 22; ws_e2v_[ 95] = 23;

   ws_e2v_[ 96] = 24; ws_e2v_[ 97] = 11; ws_e2v_[ 98] = 34; ws_e2v_[ 99] = 32;
   ws_e2v_[100] =  2; ws_e2v_[101] = 10; ws_e2v_[102] = 25; ws_e2v_[103] =  1;

   ws_e2v_[104] = 11; ws_e2v_[105] = 28; ws_e2v_[106] = 33; ws_e2v_[107] = 34;
   ws_e2v_[108] =  8; ws_e2v_[109] =  6; ws_e2v_[110] =  5; ws_e2v_[111] = 29;

   ws_e2v_[112] = 30; ws_e2v_[113] = 15; ws_e2v_[114] = 35; ws_e2v_[115] = 33;
   ws_e2v_[116] =  4; ws_e2v_[117] = 12; ws_e2v_[118] = 31; ws_e2v_[119] =  5;

   ws_e2v_[120] = 15; ws_e2v_[121] = 26; ws_e2v_[122] = 32; ws_e2v_[123] = 35;
   ws_e2v_[124] = 14; ws_e2v_[125] =  0; ws_e2v_[126] =  1; ws_e2v_[127] = 27;

   for (int i=0; i<16; i++) { ws_elem_att_[i] = 1; }

   ws_be2v_[ 0] =  3; ws_be2v_[ 1] =  2; ws_be2v_[ 2] =  1; ws_be2v_[ 3] =  0;
   ws_be2v_[ 4] =  2; ws_be2v_[ 5] =  3; ws_be2v_[ 6] = 18; ws_be2v_[ 7] = 24;
   ws_be2v_[ 8] =  3; ws_be2v_[ 9] =  0; ws_be2v_[10] = 26; ws_be2v_[11] = 18;
   ws_be2v_[12] =  4; ws_be2v_[13] =  5; ws_be2v_[14] =  6; ws_be2v_[15] =  7;
   ws_be2v_[16] = 16; ws_be2v_[17] = 30; ws_be2v_[18] =  4; ws_be2v_[19] =  7;
   ws_be2v_[20] = 16; ws_be2v_[21] =  7; ws_be2v_[22] =  6; ws_be2v_[23] = 28;
   ws_be2v_[24] = 17; ws_be2v_[25] = 15; ws_be2v_[26] = 30; ws_be2v_[27] = 16;
   ws_be2v_[28] = 15; ws_be2v_[29] = 17; ws_be2v_[30] = 18; ws_be2v_[31] = 26;
   ws_be2v_[32] =  9; ws_be2v_[33] = 10; ws_be2v_[34] = 11; ws_be2v_[35] =  8;
   ws_be2v_[36] =  9; ws_be2v_[37] =  8; ws_be2v_[38] = 29; ws_be2v_[39] = 23;
   ws_be2v_[40] =  9; ws_be2v_[41] = 23; ws_be2v_[42] = 25; ws_be2v_[43] = 10;
   ws_be2v_[44] =  1; ws_be2v_[45] = 25; ws_be2v_[46] = 23; ws_be2v_[47] = 22;
   ws_be2v_[48] =  1; ws_be2v_[49] = 22; ws_be2v_[50] = 21; ws_be2v_[51] = 27;
   ws_be2v_[52] = 13; ws_be2v_[53] = 12; ws_be2v_[54] = 15; ws_be2v_[55] = 14;
   ws_be2v_[56] = 21; ws_be2v_[57] = 31; ws_be2v_[58] = 12; ws_be2v_[59] = 13;
   ws_be2v_[60] = 21; ws_be2v_[61] = 13; ws_be2v_[62] = 14; ws_be2v_[63] = 27;
   ws_be2v_[64] = 16; ws_be2v_[65] = 19; ws_be2v_[66] = 18; ws_be2v_[67] = 17;
   ws_be2v_[68] = 11; ws_be2v_[69] = 24; ws_be2v_[70] = 18; ws_be2v_[71] = 19;
   ws_be2v_[72] = 11; ws_be2v_[73] = 19; ws_be2v_[74] = 16; ws_be2v_[75] = 28;
   ws_be2v_[76] = 20; ws_be2v_[77] = 21; ws_be2v_[78] = 22; ws_be2v_[79] = 23;
   ws_be2v_[80] =  5; ws_be2v_[81] = 31; ws_be2v_[82] = 21; ws_be2v_[83] = 20;
   ws_be2v_[84] =  5; ws_be2v_[85] = 20; ws_be2v_[86] = 23; ws_be2v_[87] = 29;
   ws_be2v_[88] =  2; ws_be2v_[89] = 10; ws_be2v_[90] = 25; ws_be2v_[91] =  1;
   ws_be2v_[92] = 24; ws_be2v_[93] = 11; ws_be2v_[94] = 10; ws_be2v_[95] =  2;
   ws_be2v_[96] =  8; ws_be2v_[97] =  6; ws_be2v_[98] =  5; ws_be2v_[99] = 29;
   ws_be2v_[100] = 11; ws_be2v_[101] = 28; ws_be2v_[102] =  6;
   ws_be2v_[103] =  8;
   ws_be2v_[104] =  4; ws_be2v_[105] = 12; ws_be2v_[106] = 31;
   ws_be2v_[107] =  5;
   ws_be2v_[108] = 30; ws_be2v_[109] = 15; ws_be2v_[110] = 12;
   ws_be2v_[111] =  4;
   ws_be2v_[112] = 15; ws_be2v_[113] = 26; ws_be2v_[114] =  0;
   ws_be2v_[115] = 14;
   ws_be2v_[116] = 14; ws_be2v_[117] =  0; ws_be2v_[118] =  1;
   ws_be2v_[119] = 27;

   for (int i=0; i<30; i++) { ws_belem_att_[i] = 1; }
   */
}

bool
BodyCenteredTetragonalLattice::MapToFundamentalDomain(const Vector & pt,
                                                      Vector & ipt) const
{
   bool map = false;
   /*
   ipt = pt;
   if ( ipt[2] < 0.0 )
   {
      ipt[2] *= -1.0;
      map = true;
   }
   if ( ipt[1] < 0.0 )
   {
      ipt[1] *= -1.0;
      map = true;
   }
   if ( ipt[0] < 0.0 )
   {
      ipt[0] *= -1.0;
      map = true;
   }
   if ( ipt[0] > ipt[1] )
   {
      double d = ipt[0];
      ipt[0] = ipt[1];
      ipt[1] = d;
      map = true;
   }
   if ( ipt[2] > ipt[1] )
   {
      double d = ipt[2];
      ipt[2] = ipt[1];
      ipt[1] = d;
      map = true;
   }
   if ( ipt[0] < ipt[2] )
   {
      double d = ipt[0];
      ipt[0] = ipt[2];
      ipt[2] = d;
      map = true;
   }
   */
   return map;
}

const DenseMatrix &
BodyCenteredTetragonalLattice::GetTransformation(int ti) const
{
   int ir = ti % 2;
   int iq = ti / 2;

   T_ = 0.0;

   if ( ir == 0 )
   {
      T_(0, 0) = 1.0;
      T_(1, 1) = 1.0;
   }
   else
   {
      T_(0, 1) = 1.0;
      T_(1, 0) = 1.0;
   }
   T_(2, 2) = 1.0;

   for (int i=0; i<3; i++)
   {
      if ( iq & (int)pow(2, i) )
      {
         T_(i,0) *= -1.0;
         T_(i,1) *= -1.0;
         T_(i,2) *= -1.0;
      }
   }

   return T_;
}

Mesh *
BodyCenteredTetragonalLattice::GetFundamentalDomainMesh() const
{
   Mesh * mesh = new Mesh(3, fd_nvert_, 3, fd_nbt_ + fd_nbq_, 3);
   // Mesh * mesh = new Mesh(3, fd_nvert_, 2, 0*fd_nbt_ + 0*fd_nbq_, 3);
   for (int i=0; i<fd_nvert_; i++)
   {
      mesh->AddVertex((const double *)&fd_vert_[3*i]);
   }
   if (fd_nvert_ == 7)
   {
      mesh->AddTet(&fd_e2v_[0], fd_elem_att_[0]);
      mesh->AddTet(&fd_e2v_[4], fd_elem_att_[1]);
      mesh->AddPyramid(&fd_e2v_[8], fd_elem_att_[2]);
   }
   else
   {
      mesh->AddTet(&fd_e2v_[0], fd_elem_att_[0]);
      mesh->AddPyramid(&fd_e2v_[4], fd_elem_att_[1]);
      mesh->AddWedge(&fd_e2v_[9], fd_elem_att_[2]);
   }
   for (int i=0; i<fd_nbt_; i++)
   {
      mesh->AddBdrTriangle((const int*)&fd_be2v_[3*i], fd_belem_att_[i]);
   }
   for (int i=0; i<fd_nbq_; i++)
   {
      mesh->AddBdrQuad((const int*)&fd_be2v_[4*i+3*fd_nbt_],
                       fd_belem_att_[i+fd_nbt_]);
   }

   mesh->FinalizeMesh(true);

   return mesh;
}
/*
Mesh *
BodyCenteredTetragonalLattice::GetWignerSeitzMesh(bool tetMesh) const
{
 Mesh * mesh = new Mesh((double*)ws_vert_, (c_>M_SQRT2*a_)?24:38,
                        (int*)ws_e2v_, Geometry::CUBE,
                        (int*)ws_elem_att_, (c_>M_SQRT2*a_)?8:16,
                        (int*)ws_be2v_, Geometry::SQUARE,
                        (int*)ws_belem_att_, (c_>M_SQRT2*a_)?20:30,
                        3, 3);

 mesh->Finalize();

 return mesh;
}

Mesh *
BodyCenteredTetragonalLattice::GetPeriodicWignerSeitzMesh(bool tetMesh) const
{
 Mesh * mesh = this->GetWignerSeitzMesh(tetMesh);

 mesh->UniformRefinement();

 Mesh * per_mesh = MakePeriodicMesh(mesh, trn_vecs_);

 delete mesh;

 return per_mesh;
}
*/
OrthorhombicLattice::OrthorhombicLattice(double a, double b, double c)
   : BravaisLattice3D(a, b, c, 0.5 * M_PI, 0.5 * M_PI, 0.5 * M_PI)
{
   label_ = "ORC";
   type_  = PRIMITIVE_ORTHORHOMBIC;

   // Set Lattice Vectors
   lat_vecs_[0][0] =  a_; lat_vecs_[0][1] = 0.0; lat_vecs_[0][2] = 0.0;
   lat_vecs_[1][0] = 0.0; lat_vecs_[1][1] =  b_; lat_vecs_[1][2] = 0.0;
   lat_vecs_[2][0] = 0.0; lat_vecs_[2][1] = 0.0; lat_vecs_[2][2] =  c_;

   // Set Reciprocal Lattice Vectors
   rec_vecs_[0][0] = 1.0 / a_; rec_vecs_[0][1] = 0.0; rec_vecs_[0][2] = 0.0;
   rec_vecs_[1][0] = 0.0; rec_vecs_[1][1] = 1.0 / b_; rec_vecs_[1][2] = 0.0;
   rec_vecs_[2][0] = 0.0; rec_vecs_[2][1] = 0.0; rec_vecs_[2][2] = 1.0 / c_;

   // Set Translation Vectors
   trn_vecs_.resize(3);
   for (unsigned int i=0; i<trn_vecs_.size(); i++)
   {
      trn_vecs_[i].SetSize(3);
   }
   trn_vecs_[0][0] =  a_; trn_vecs_[0][1] = 0.0; trn_vecs_[0][2] = 0.0;
   trn_vecs_[1][0] = 0.0; trn_vecs_[1][1] =  b_; trn_vecs_[1][2] = 0.0;
   trn_vecs_[2][0] = 0.0; trn_vecs_[2][1] = 0.0; trn_vecs_[2][2] =  c_;

   // Set the face radii
   face_radii_.resize(3);
   face_radii_[0] = 0.5 * min(b_,c_);
   face_radii_[1] = 0.5 * min(c_,a_);
   face_radii_[2] = 0.5 * min(a_,b_);

   this->SetCellVolumes();

   // Allocate sl_, sp_, and path_ vectors
   this->SetVectorSizes();

   // Set Symmetry Points and Labels
   sl_[0] = "Gamma";
   sp_[0] = 0.0;

   sl_[1] = "U";
   add(0.5,rec_vecs_[0],0.5,rec_vecs_[2],sp_[1]);

   sl_[2] = "R";
   add(0.5,rec_vecs_[0],0.5,rec_vecs_[1],sp_[2]);
   sp_[2].Add(0.5,rec_vecs_[2]);

   sl_[3] = "X";
   sp_[3].Set(0.5,rec_vecs_[0]);

   sl_[4] = "S";
   add(0.5,rec_vecs_[0],0.5,rec_vecs_[1],sp_[4]);

   sl_[5] = "Y";
   sp_[5].Set(0.5,rec_vecs_[1]);

   sl_[6] = "T";
   add(0.5,rec_vecs_[1],0.5,rec_vecs_[2],sp_[6]);

   sl_[7] = "Z";
   sp_[7].Set(0.5,rec_vecs_[2]);

   for (unsigned int i=0; i<sl_.size(); i++)
   {
      si_[sl_[i]] = i;
   }

   // Define Paths
   path_[0][0] = 0;
   path_[0][1] = 3;
   path_[0][2] = 4;
   path_[0][3] = 5;
   path_[0][4] = 0;
   path_[0][5] = 7;
   path_[0][6] = 1;
   path_[0][7] = 2;
   path_[0][8] = 6;
   path_[0][9] = 7;

   path_[1][0] = 5;
   path_[1][1] = 6;

   path_[2][0] = 1;
   path_[2][1] = 3;

   path_[3][0] = 4;
   path_[3][1] = 2;

   // Set Intermediate Symmetry Points
   this->SetIntermediatePoints();

   // Set Intermediate Symmetry Point Labels
   il_[0][0] = "GammaX"; // Gamma -> X
   il_[0][1] = "XS";     // X     -> S
   il_[0][2] = "SY";     // S     -> Y
   il_[0][3] = "YGamma"; // Y     -> Gamma
   il_[0][4] = "GammaZ"; // Gamma -> Z
   il_[0][0] = "ZU";     // Z     -> U
   il_[0][0] = "UR";     // U     -> R
   il_[0][1] = "RT";     // R     -> T
   il_[0][2] = "TZ";     // T     -> Z

   il_[1][0] = "YT";     // Y     -> T
   il_[2][0] = "UX";     // U     -> X
   il_[3][0] = "SR";     // S     -> R

   // Set Mesh data
   fd_vert_[ 0] = 0.0;      fd_vert_[ 1] = 0.0;      fd_vert_[ 2] = 0.0;
   fd_vert_[ 3] = 0.5 * a_; fd_vert_[ 4] = 0.0;      fd_vert_[ 5] = 0.0;
   fd_vert_[ 6] = 0.5 * a_; fd_vert_[ 7] = 0.5 * b_; fd_vert_[ 8] = 0.0;
   fd_vert_[ 9] = 0.0;      fd_vert_[10] = 0.5 * b_; fd_vert_[11] = 0.0;
   for (int i=0; i<4; i++)
   {
      fd_vert_[3 * i + 12] = fd_vert_[3 * i + 0];
      fd_vert_[3 * i + 13] = fd_vert_[3 * i + 1];
      fd_vert_[3 * i + 14] = 0.5 * c_;
   }

   fd_e2v_[0] = 0; fd_e2v_[1] = 1; fd_e2v_[2] = 2; fd_e2v_[3] = 3;
   fd_e2v_[4] = 4; fd_e2v_[5] = 5; fd_e2v_[6] = 6; fd_e2v_[7] = 7;
   fd_elem_att_[0] = 1;

   fd_be2v_[ 0] = 3; fd_be2v_[ 1] = 2; fd_be2v_[ 2] = 1; fd_be2v_[ 3] = 0;
   fd_be2v_[ 4] = 0; fd_be2v_[ 5] = 1; fd_be2v_[ 6] = 5; fd_be2v_[ 7] = 4;
   fd_be2v_[ 8] = 1; fd_be2v_[ 9] = 2; fd_be2v_[10] = 6; fd_be2v_[11] = 5;
   fd_be2v_[12] = 2; fd_be2v_[13] = 3; fd_be2v_[14] = 7; fd_be2v_[15] = 6;
   fd_be2v_[16] = 3; fd_be2v_[17] = 0; fd_be2v_[18] = 4; fd_be2v_[19] = 7;
   fd_be2v_[20] = 4; fd_be2v_[21] = 5; fd_be2v_[22] = 6; fd_be2v_[23] = 7;

   fd_belem_att_[0] = 10; fd_belem_att_[1] = 10; fd_belem_att_[2] = 1;
   fd_belem_att_[3] =  1; fd_belem_att_[4] = 10; fd_belem_att_[5] = 1;
   /*
   ws_vert_[ 0] = -0.5 * a_; ws_vert_[ 1] = -0.5 * b_; ws_vert_[ 2] = -0.5 * c_;
   ws_vert_[ 3] =  0.5 * a_; ws_vert_[ 4] = -0.5 * b_; ws_vert_[ 5] = -0.5 * c_;
   ws_vert_[ 6] =  0.5 * a_; ws_vert_[ 7] =  0.5 * b_; ws_vert_[ 8] = -0.5 * c_;
   ws_vert_[ 9] = -0.5 * a_; ws_vert_[10] =  0.5 * b_; ws_vert_[11] = -0.5 * c_;
   ws_vert_[12] = -0.5 * a_; ws_vert_[13] = -0.5 * b_; ws_vert_[14] =  0.5 * c_;
   ws_vert_[15] =  0.5 * a_; ws_vert_[16] = -0.5 * b_; ws_vert_[17] =  0.5 * c_;
   ws_vert_[18] =  0.5 * a_; ws_vert_[19] =  0.5 * b_; ws_vert_[20] =  0.5 * c_;
   ws_vert_[21] = -0.5 * a_; ws_vert_[22] =  0.5 * b_; ws_vert_[23] =  0.5 * c_;

   for (int i=0; i<8; i++) { ws_e2v_[i] = i; }
   ws_elem_att_[0] = 1;

   ws_be2v_[ 0] = 0; ws_be2v_[ 1] = 3; ws_be2v_[ 2] = 2; ws_be2v_[ 3] = 1;
   ws_be2v_[ 4] = 4; ws_be2v_[ 5] = 5; ws_be2v_[ 6] = 6; ws_be2v_[ 7] = 7;

   ws_be2v_[ 8] = 0; ws_be2v_[ 9] = 1; ws_be2v_[10] = 5; ws_be2v_[11] = 4;
   ws_be2v_[12] = 1; ws_be2v_[13] = 2; ws_be2v_[14] = 6; ws_be2v_[15] = 5;
   ws_be2v_[16] = 2; ws_be2v_[17] = 3; ws_be2v_[18] = 7; ws_be2v_[19] = 6;
   ws_be2v_[20] = 3; ws_be2v_[21] = 0; ws_be2v_[22] = 4; ws_be2v_[23] = 7;

   for (int i=0; i<6; i++) { ws_belem_att_[i] = 1; }
   */
}

bool
OrthorhombicLattice::MapToFundamentalDomain(const Vector & pt,
                                            Vector & ipt) const
{
   bool map = false;
   /*
   ipt = pt;
   if ( ipt[2] < 0.0 )
   {
      ipt[2] *= -1.0;
      map = true;
   }
   if ( ipt[1] < 0.0 )
   {
      ipt[1] *= -1.0;
      map = true;
   }
   if ( ipt[0] < 0.0 )
   {
      ipt[0] *= -1.0;
      map = true;
   }
   if ( ipt[0] > ipt[1] )
   {
      double d = ipt[0];
      ipt[0] = ipt[1];
      ipt[1] = d;
      map = true;
   }
   if ( ipt[2] > ipt[1] )
   {
      double d = ipt[2];
      ipt[2] = ipt[1];
      ipt[1] = d;
      map = true;
   }
   if ( ipt[0] < ipt[2] )
   {
      double d = ipt[0];
      ipt[0] = ipt[2];
      ipt[2] = d;
      map = true;
   }
   */
   return map;
}

const DenseMatrix &
OrthorhombicLattice::GetTransformation(int ti) const
{
   T_ = 0.0;

   T_(0, 0) = 1.0;
   T_(1, 1) = 1.0;
   T_(2, 2) = 1.0;

   for (int i=0; i<3; i++)
   {
      if ( ti & (int)pow(2, i) )
      {
         T_(i,0) *= -1.0;
         T_(i,1) *= -1.0;
         T_(i,2) *= -1.0;
      }
   }

   return T_;
}

Mesh *
OrthorhombicLattice::GetFundamentalDomainMesh() const
{
   Mesh * mesh = new Mesh((double*)fd_vert_, 8,
                          (int*)fd_e2v_, Geometry::CUBE,
                          (int*)fd_elem_att_, 1,
                          (int*)fd_be2v_, Geometry::SQUARE,
                          (int*)fd_belem_att_, 6,
                          3, 3);

   mesh->Finalize();

   return mesh;
}
/*
Mesh *
OrthorhombicLattice::GetWignerSeitzMesh(bool tetMesh) const
{
   Mesh * mesh = new Mesh((double*)ws_vert_, 8,
                          (int*)ws_e2v_, Geometry::CUBE,
                          (int*)ws_elem_att_, 1,
                          (int*)ws_be2v_, Geometry::SQUARE,
                          (int*)ws_belem_att_, 6,
                          3, 3);
   mesh->Finalize();

   return mesh;
}

Mesh *
OrthorhombicLattice::GetPeriodicWignerSeitzMesh(bool tetMesh) const
{
   Mesh * mesh = this->GetWignerSeitzMesh(tetMesh);

   mesh->UniformRefinement();
   mesh->UniformRefinement();

   Mesh * per_mesh = MakePeriodicMesh(mesh, trn_vecs_);

   delete mesh;

   return per_mesh;
}
*/
FaceCenteredOrthorhombicLattice::FaceCenteredOrthorhombicLattice(double a,
                                                                 double b,
                                                                 double c)
   : BravaisLattice3D(a, b, c, 0.5 * M_PI, 0.5 * M_PI, 0.5 * M_PI)
{
   type_  = FACE_CENTERED_ORTHORHOMBIC;

   // Determine lattice variety based on relationships between parameters.
   if (1.0/(a_*a_) > 1.0/(b_*b_) + 1.0/(c_*c_))
   {
      label_ = "ORCF1";
      variety_= 1;
   }
   else if (1.0/(a_*a_) < 1.0/(b_*b_) + 1.0/(c_*c_))
   {
      label_ = "ORCF2";
      variety_ = 2;
   }
   else
   {
      label_ = "ORCF3";
      variety_ = 3;
   }

   // Set Lattice Vectors
   lat_vecs_[0][0] =  0.0;
   lat_vecs_[0][1] =  0.5 * b_;
   lat_vecs_[0][2] =  0.5 * c_;

   lat_vecs_[1][0] =  0.5 * a_;
   lat_vecs_[1][1] =  0.0;
   lat_vecs_[1][2] =  0.5 * c_;

   lat_vecs_[2][0] =  0.5 * a_;
   lat_vecs_[2][1] =  0.5 * b_;
   lat_vecs_[2][2] =  0.0;

   // Set Reciprocal Lattice Vectors
   rec_vecs_[0][0] = -1.0 / a_;
   rec_vecs_[0][1] =  1.0 / b_;
   rec_vecs_[0][2] =  1.0 / c_;

   rec_vecs_[1][0] =  1.0 / a_;
   rec_vecs_[1][1] = -1.0 / b_;
   rec_vecs_[1][2] =  1.0 / c_;

   rec_vecs_[2][0] =  1.0 / a_;
   rec_vecs_[2][1] =  1.0 / b_;
   rec_vecs_[2][2] = -1.0 / c_;

   // Set Translation Vectors
   trn_vecs_.resize(7);
   for (unsigned int i=0; i<trn_vecs_.size(); i++)
   {
      trn_vecs_[i].SetSize(3);
   }
   trn_vecs_[0][0] =   a_;
   trn_vecs_[0][1] =  0.0;
   trn_vecs_[0][2] =  0.0;

   trn_vecs_[1][0] =  0.5 * a_;
   trn_vecs_[1][1] =  0.5 * b_;
   trn_vecs_[1][2] =  0.0;

   trn_vecs_[2][0] =  0.5 * a_;
   trn_vecs_[2][1] = -0.5 * b_;
   trn_vecs_[2][2] =  0.0;

   trn_vecs_[3][0] =  0.5 * a_;
   trn_vecs_[3][1] =  0.0;
   trn_vecs_[3][2] =  0.5 * c_;

   trn_vecs_[4][0] =  0.5 * a_;
   trn_vecs_[4][1] =  0.0;
   trn_vecs_[4][2] = -0.5 * c_;

   trn_vecs_[5][0] =  0.0;
   trn_vecs_[5][1] =  0.5 * b_;
   trn_vecs_[5][2] =  0.5 * c_;

   trn_vecs_[6][0] =  0.0;
   trn_vecs_[6][1] =  0.5 * b_;
   trn_vecs_[6][2] = -0.5 * c_;

   // Set the face radii
   face_radii_.resize(7);
   face_radii_[0] = 0.25 * min((b_ * b_ - a_ * a_) / b_,
                               (c_ * c_ - a_ * a_) / c_);
   face_radii_[1] =  0.25 * sqrt(a_ * a_ + b_ * b_) *
                     min(c_ * c_ / sqrt(b_ * b_ * c_ * c_ + a_ * a_ * (b_ * b_ + c_ * c_)),
                         a_ / b_);
   face_radii_[2] = face_radii_[1];
   face_radii_[3] = 0.25 * sqrt(a_ * a_ + c_ * c_) *
                    min(b_ * b_ / sqrt(b_ * b_ * c_ * c_ + a_ * a_ * (b_ * b_ + c_ * c_)),
                        a_ / c_);
   face_radii_[4] = face_radii_[3];
   face_radii_[5] = 0.25 * a_ * a_ * sqrt((b_ * b_ + c_ * c_)) /
                    sqrt(b_ * b_ * c_ * c_ + a_ * a_ * (b_ * b_ + c_ * c_));
   face_radii_[6] = face_radii_[5];

   this->SetCellVolumes();

   // Allocate sl_, sp_, and path_ vectors
   this->SetVectorSizes();

   // Set Symmetry Points and Labels
   double a2 = a_ * a_;
   double b2 = b_ * b_;
   double c2 = c_ * c_;

   if (variety_ == 1)
   {
      double zeta = 0.25 * (1.0 + a2 / b2 - a2 / c2);
      double  eta = 0.25 * (1.0 + a2 / b2 + a2 / c2);

      sl_[0] = "Gamma";
      sp_[0] = 0.0;

      sl_[1] = "X";
      add(eta,rec_vecs_[1],eta,rec_vecs_[2],sp_[1]);

      sl_[2] = "A";
      add(0.5,rec_vecs_[0],0.5+zeta,rec_vecs_[1],sp_[2]);
      sp_[2].Add(zeta,rec_vecs_[2]);

      sl_[3] = "X1";
      add(1.0,rec_vecs_[0],1.0-eta,rec_vecs_[1],sp_[3]);
      sp_[3].Add(1.0-eta,rec_vecs_[2]);

      sl_[4] = "A1";
      add(0.5,rec_vecs_[0],0.5-zeta,rec_vecs_[1],sp_[4]);
      sp_[4].Add(1.0-zeta,rec_vecs_[2]);

      sl_[5] = "Y";
      add(0.5,rec_vecs_[0],0.5,rec_vecs_[2],sp_[5]);

      sl_[6] = "L";
      add(0.5,rec_vecs_[0],0.5,rec_vecs_[1],sp_[6]);
      sp_[6].Add(0.5,rec_vecs_[2]);

      sl_[7] = "Z";
      add(0.5,rec_vecs_[0],0.5,rec_vecs_[1],sp_[7]);

      sl_[8] = "T";
      add(1.0,rec_vecs_[0],0.5,rec_vecs_[1],sp_[8]);
      sp_[8].Add(0.5,rec_vecs_[2]);
   }
   else if (variety_ == 2)
   {
      double   eta = 0.25 * (1.0 + a2 / b2 - a2 / c2);
      double delta = 0.25 * (1.0 + b2 / a2 - b2 / c2);
      double   phi = 0.25 * (1.0 + c2 / b2 - c2 / a2);

      sl_[0] = "Gamma";
      sp_[0] = 0.0;

      sl_[1] = "H";
      add(1.0-phi,rec_vecs_[0],0.5-phi,rec_vecs_[1],sp_[1]);
      sp_[1].Add(0.5,rec_vecs_[2]);

      sl_[2] = "C";
      add(0.5,rec_vecs_[0],0.5-eta,rec_vecs_[1],sp_[2]);
      sp_[2].Add(1.0-eta,rec_vecs_[2]);

      sl_[3] = "H1";
      add(phi,rec_vecs_[0],0.5+phi,rec_vecs_[1],sp_[3]);
      sp_[3].Add(0.5,rec_vecs_[2]);

      sl_[4] = "C1";
      add(0.5,rec_vecs_[0],0.5+eta,rec_vecs_[1],sp_[4]);
      sp_[4].Add(eta,rec_vecs_[2]);

      sl_[5] = "X";
      add(0.5,rec_vecs_[1],0.5,rec_vecs_[2],sp_[5]);

      sl_[6] = "D";
      add(0.5-delta,rec_vecs_[0],0.5,rec_vecs_[1],sp_[6]);
      sp_[6].Add(1.0-delta,rec_vecs_[2]);

      sl_[7] = "Y";
      add(0.5,rec_vecs_[0],0.5,rec_vecs_[2],sp_[7]);

      sl_[8] = "D1";
      add(0.5+delta,rec_vecs_[0],0.5,rec_vecs_[1],sp_[8]);
      sp_[8].Add(delta,rec_vecs_[2]);

      sl_[9] = "Z";
      add(0.5,rec_vecs_[0],0.5,rec_vecs_[1],sp_[9]);

      sl_[10] = "L";
      add(0.5,rec_vecs_[0],0.5,rec_vecs_[1],sp_[10]);
      sp_[10].Add(0.5,rec_vecs_[2]);
   }
   else
   {
      double zeta = 0.25 * (1.0 + a2 / b2 - a2 / c2);
      double  eta = 0.25 * (1.0 + a2 / b2 + a2 / c2);

      sl_[0] = "Gamma";
      sp_[0] = 0.0;

      sl_[1] = "X";
      add(eta,rec_vecs_[1],eta,rec_vecs_[2],sp_[1]);

      sl_[2] = "A";
      add(0.5,rec_vecs_[0],0.5+zeta,rec_vecs_[1],sp_[2]);
      sp_[2].Add(zeta,rec_vecs_[2]);

      sl_[3] = "A1";
      add(0.5,rec_vecs_[0],0.5-zeta,rec_vecs_[1],sp_[3]);
      sp_[3].Add(1.0-zeta,rec_vecs_[2]);

      sl_[4] = "Y";
      add(0.5,rec_vecs_[0],0.5,rec_vecs_[2],sp_[4]);

      sl_[5] = "L";
      add(0.5,rec_vecs_[0],0.5,rec_vecs_[1],sp_[5]);
      sp_[5].Add(0.5,rec_vecs_[2]);

      sl_[6] = "Z";
      add(0.5,rec_vecs_[0],0.5,rec_vecs_[1],sp_[6]);

      sl_[7] = "T";
      add(1.0,rec_vecs_[0],0.5,rec_vecs_[1],sp_[7]);
      sp_[7].Add(0.5,rec_vecs_[2]);
   }

   for (unsigned int i=0; i<sl_.size(); i++)
   {
      si_[sl_[i]] = i;
   }

   // Define Paths
   if (variety_ == 1)
   {
      path_[0][0] = 0;
      path_[0][1] = 5;
      path_[0][2] = 8;
      path_[0][3] = 7;
      path_[0][4] = 0;
      path_[0][5] = 1;
      path_[0][6] = 4;
      path_[0][7] = 5;

      path_[1][0] = 8;
      path_[1][1] = 3;

      path_[2][0] = 1;
      path_[2][1] = 2;
      path_[2][2] = 7;

      path_[3][0] = 6;
      path_[3][1] = 0;
   }
   else if (variety_ == 2)
   {
      path_[0][0] = 0;
      path_[0][1] = 7;
      path_[0][2] = 2;
      path_[0][3] = 6;
      path_[0][4] = 5;
      path_[0][5] = 0;
      path_[0][6] = 9;
      path_[0][7] = 8;
      path_[0][8] = 1;
      path_[0][9] = 2;

      path_[1][0] = 4;
      path_[1][1] = 9;

      path_[2][0] = 5;
      path_[2][1] = 3;

      path_[3][0] = 1;
      path_[3][1] = 7;

      path_[4][0] = 10;
      path_[4][1] = 0;
   }
   else
   {
      path_[0][0] = 0;
      path_[0][1] = 4;
      path_[0][2] = 7;
      path_[0][3] = 6;
      path_[0][4] = 0;
      path_[0][5] = 1;
      path_[0][6] = 3;
      path_[0][7] = 4;

      path_[1][0] = 1;
      path_[1][1] = 2;
      path_[1][2] = 6;

      path_[2][0] = 5;
      path_[2][1] = 0;
   }
   // Set Intermediate Symmetry Points
   this->SetIntermediatePoints();

   // Set Intermediate Symmetry Point Labels
   if ( variety_ == 1)
   {
      il_[0][0] = "GammaY"; // Gamma -> Y
      il_[0][1] = "YT";     // Y     -> T
      il_[0][2] = "TZ";     // T     -> Z
      il_[0][3] = "ZGamma"; // Z     -> Gamma
      il_[0][4] = "GammaX"; // Gamma -> X
      il_[0][5] = "XA1";    // X     -> A1
      il_[0][6] = "A1Y";    // A1    -> Y

      il_[1][0] = "TX1";    // T     -> X1

      il_[2][0] = "XA";     // X     -> A
      il_[2][1] = "AZ";     // A     -> Z

      il_[3][0] = "LGamma"; // L     -> Gamma
   }
   else if (variety_ == 2)
   {
      il_[0][0] = "GammaY"; // Gamma -> Y
      il_[0][1] = "YC";     // Y     -> C
      il_[0][2] = "CD";     // C     -> D
      il_[0][3] = "DX";     // D     -> X
      il_[0][4] = "XGamma"; // X     -> Gamma
      il_[0][5] = "GammaZ"; // Gamma -> Z
      il_[0][6] = "ZD1";    // Z     -> D1
      il_[0][7] = "D1H";    // D1    -> H
      il_[0][8] = "HC";     // H     -> C

      il_[1][0] = "C1Z";    // C1    -> Z

      il_[2][0] = "XH1";    // X     -> H1

      il_[3][0] = "HY";     // H     -> Y

      il_[4][0] = "LGamma"; // L     -> Gamma
   }
   else
   {
      il_[0][0] = "GammaY"; // Gamma -> Y
      il_[0][1] = "YT";     // Y     -> T
      il_[0][2] = "TZ";     // T     -> Z
      il_[0][3] = "ZGamma"; // Z     -> Gamma
      il_[0][4] = "GammaX"; // Gamma -> X
      il_[0][5] = "XA1";    // X     -> A1
      il_[0][6] = "A1Y";    // A1    -> Y

      il_[1][0] = "XA";     // X     -> A
      il_[1][1] = "AZ";     // A     -> Z

      il_[2][0] = "LGamma"; // L     -> Gamma
   }

   // Set Mesh data
   ws_vert_[  0] = -0.50 * a_;
   ws_vert_[  1] =  0.25 * b_ - 0.25 * a_ * a_ / b_;
   ws_vert_[  2] =  0.25 * c_ - 0.25 * a_ * a_ / c_;

   ws_vert_[  3] = -0.50 * a_;
   ws_vert_[  4] = -0.25 * b_ + 0.25 * a_ * a_ / b_;
   ws_vert_[  5] =  0.25 * c_ - 0.25 * a_ * a_ / c_;

   ws_vert_[  6] = -0.50 * a_;
   ws_vert_[  7] = -0.25 * b_ + 0.25 * a_ * a_ / b_;
   ws_vert_[  8] = -0.25 * c_ + 0.25 * a_ * a_ / c_;

   ws_vert_[  9] = -0.50 * a_;
   ws_vert_[ 10] =  0.25 * b_ - 0.25 * a_ * a_ / b_;
   ws_vert_[ 11] = -0.25 * c_ + 0.25 * a_ * a_ / c_;

   ws_vert_[ 12] =  0.50 * a_;
   ws_vert_[ 13] =  0.25 * b_ - 0.25 * a_ * a_ / b_;
   ws_vert_[ 14] =  0.25 * c_ - 0.25 * a_ * a_ / c_;

   ws_vert_[ 15] =  0.50 * a_;
   ws_vert_[ 16] = -0.25 * b_ + 0.25 * a_ * a_ / b_;
   ws_vert_[ 17] =  0.25 * c_ - 0.25 * a_ * a_ / c_;

   ws_vert_[ 18] =  0.50 * a_;
   ws_vert_[ 19] = -0.25 * b_ + 0.25 * a_ * a_ / b_;
   ws_vert_[ 20] = -0.25 * c_ + 0.25 * a_ * a_ / c_;

   ws_vert_[ 21] =  0.50 * a_;
   ws_vert_[ 22] =  0.25 * b_ - 0.25 * a_ * a_ / b_;
   ws_vert_[ 23] = -0.25 * c_ + 0.25 * a_ * a_ / c_;

   ws_vert_[ 24] =  0.25 * a_;
   ws_vert_[ 25] = -0.25 * b_;
   ws_vert_[ 26] = -0.25 * c_;

   ws_vert_[ 27] =  0.00;
   ws_vert_[ 28] = -0.25 * b_ - 0.25 * a_ * a_ / b_;
   ws_vert_[ 29] = -0.25 * c_ + 0.25 * a_ * a_ / c_;

   ws_vert_[ 30] = -0.25 * a_;
   ws_vert_[ 31] = -0.25 * b_;
   ws_vert_[ 32] = -0.25 * c_;

   ws_vert_[ 33] =  0.00;
   ws_vert_[ 34] = -0.25 * b_ + 0.25 * a_ * a_ / b_;
   ws_vert_[ 35] = -0.25 * c_ - 0.25 * a_ * a_ / c_;

   ws_vert_[ 36] =  0.25 * a_;
   ws_vert_[ 37] =  0.25 * b_;
   ws_vert_[ 38] =  0.25 * c_;

   ws_vert_[ 39] =  0.00;
   ws_vert_[ 40] =  0.25 * b_ - 0.25 * a_ * a_ / b_;
   ws_vert_[ 41] =  0.25 * c_ + 0.25 * a_ * a_ / c_;

   ws_vert_[ 42] = -0.25 * a_;
   ws_vert_[ 43] =  0.25 * b_;
   ws_vert_[ 44] =  0.25 * c_;

   ws_vert_[ 45] =  0.00;
   ws_vert_[ 46] =  0.25 * b_ + 0.25 * a_ * a_ / b_;
   ws_vert_[ 47] =  0.25 * c_ - 0.25 * a_ * a_ / c_;

   ws_vert_[ 48] =  0.25 * a_;
   ws_vert_[ 49] =  0.25 * b_;
   ws_vert_[ 50] = -0.25 * c_;

   ws_vert_[ 51] =  0.00;
   ws_vert_[ 52] =  0.25 * b_ + 0.25 * a_ * a_ / b_;
   ws_vert_[ 53] = -0.25 * c_ + 0.25 * a_ * a_ / c_;

   ws_vert_[ 54] = -0.25 * a_;
   ws_vert_[ 55] =  0.25 * b_;
   ws_vert_[ 56] = -0.25 * c_;

   ws_vert_[ 57] =  0.00;
   ws_vert_[ 58] =  0.25 * b_ - 0.25 * a_ * a_ / b_;
   ws_vert_[ 59] = -0.25 * c_ - 0.25 * a_ * a_ / c_;

   ws_vert_[ 60] =  0.25 * a_;
   ws_vert_[ 61] = -0.25 * b_;
   ws_vert_[ 62] =  0.25 * c_;

   ws_vert_[ 63] =  0.00;
   ws_vert_[ 64] = -0.25 * b_ + 0.25 * a_ * a_ / b_;
   ws_vert_[ 65] =  0.25 * c_ + 0.25 * a_ * a_ / c_;

   ws_vert_[ 66] = -0.25 * a_;
   ws_vert_[ 67] = -0.25 * b_;
   ws_vert_[ 68] =  0.25 * c_;

   ws_vert_[ 69] =  0.00;
   ws_vert_[ 70] = -0.25 * b_ - 0.25 * a_ * a_ / b_;
   ws_vert_[ 71] =  0.25 * c_ - 0.25 * a_ * a_ / c_;

   ws_vert_[ 72] = -0.25 * a_;
   ws_vert_[ 73] = -0.25 * b_ + 0.50 * a_ * a_ / b_;
   ws_vert_[ 74] = -0.25 * c_;

   ws_vert_[ 75] = -0.25 * a_;
   ws_vert_[ 76] = -0.25 * b_;
   ws_vert_[ 77] =  0.25 * c_ - 0.50 * a_ * a_ / c_;

   ws_vert_[ 78] = -0.25 * a_;
   ws_vert_[ 79] =  0.25 * b_;
   ws_vert_[ 80] =  0.25 * c_ - 0.50 * a_ * a_ / c_;

   ws_vert_[ 81] = -0.25 * a_;
   ws_vert_[ 82] = -0.25 * b_ + 0.50 * a_ * a_ / b_;
   ws_vert_[ 83] =  0.25 * c_;

   ws_vert_[ 84] =  0.25 * a_;
   ws_vert_[ 85] = -0.25 * b_ + 0.50 * a_ * a_ / b_;
   ws_vert_[ 86] = -0.25 * c_;

   ws_vert_[ 87] =  0.25 * a_;
   ws_vert_[ 88] = -0.25 * b_;
   ws_vert_[ 89] =  0.25 * c_ - 0.50 * a_ * a_ / c_;

   ws_vert_[ 90] =  0.25 * a_;
   ws_vert_[ 91] =  0.25 * b_;
   ws_vert_[ 92] =  0.25 * c_ - 0.50 * a_ * a_ / c_;

   ws_vert_[ 93] =  0.25 * a_;
   ws_vert_[ 94] = -0.25 * b_ + 0.50 * a_ * a_ / b_;
   ws_vert_[ 95] =  0.25 * c_;

   ws_vert_[ 96] = -0.25 * a_;
   ws_vert_[ 97] = -0.25 * b_ + 0.50 * a_ * a_ / b_;
   ws_vert_[ 98] =  0.25 * c_ - 0.50 * a_ * a_ / c_;

   ws_vert_[ 99] =  0.25 * a_;
   ws_vert_[100] = -0.25 * b_ + 0.50 * a_ * a_ / b_;
   ws_vert_[101] =  0.25 * c_ - 0.50 * a_ * a_ / c_;

   ws_vert_[102] =  0.00;
   ws_vert_[103] = -0.25 * b_ + 0.25 * a_ * a_ / b_;
   ws_vert_[104] =  0.25 * c_ - 0.75 * a_ * a_ / c_;

   ws_vert_[105] =  0.00;
   ws_vert_[106] = -0.25 * b_ + 0.75 * a_ * a_ / b_;
   ws_vert_[107] =  0.25 * c_ - 0.25 * a_ * a_ / c_;

   ws_vert_[108] =  0.00;
   ws_vert_[109] = -0.25 * b_ + 0.75 * a_ * a_ / b_;
   ws_vert_[110] = -0.25 * c_ + 0.25 * a_ * a_ / c_;

   ws_vert_[111] =  0.00;
   ws_vert_[112] = -0.25 * b_ + 0.25 * a_ * a_ / b_;
   ws_vert_[113] =  0.25 * c_ - 0.25 * a_ * a_ / c_;

   ws_e2v_[  0] =  0; ws_e2v_[  1] =  1; ws_e2v_[  2] =  2; ws_e2v_[  3] =  3;
   ws_e2v_[  4] = 26; ws_e2v_[  5] = 32; ws_e2v_[  6] = 24; ws_e2v_[  7] = 18;

   ws_e2v_[  8] = 26; ws_e2v_[  9] = 32; ws_e2v_[ 10] = 24; ws_e2v_[ 11] = 18;
   ws_e2v_[ 12] = 15; ws_e2v_[ 13] = 35; ws_e2v_[ 14] = 36; ws_e2v_[ 15] = 17;

   ws_e2v_[ 16] = 15; ws_e2v_[ 17] = 35; ws_e2v_[ 18] = 36; ws_e2v_[ 19] = 17;
   ws_e2v_[ 20] = 30; ws_e2v_[ 21] = 33; ws_e2v_[ 22] = 28; ws_e2v_[ 23] = 16;

   ws_e2v_[ 24] = 30; ws_e2v_[ 25] = 33; ws_e2v_[ 26] = 28; ws_e2v_[ 27] = 16;
   ws_e2v_[ 28] =  4; ws_e2v_[ 29] =  5; ws_e2v_[ 30] =  6; ws_e2v_[ 31] =  7;

   ws_e2v_[ 32] =  9; ws_e2v_[ 33] =  8; ws_e2v_[ 34] = 11; ws_e2v_[ 35] = 10;
   ws_e2v_[ 36] = 23; ws_e2v_[ 37] = 29; ws_e2v_[ 38] = 34; ws_e2v_[ 39] = 25;

   ws_e2v_[ 40] = 23; ws_e2v_[ 41] = 29; ws_e2v_[ 42] = 34; ws_e2v_[ 43] = 25;
   ws_e2v_[ 44] = 22; ws_e2v_[ 45] = 37; ws_e2v_[ 46] = 32; ws_e2v_[ 47] =  1;

   ws_e2v_[ 48] = 22; ws_e2v_[ 49] = 37; ws_e2v_[ 50] = 32; ws_e2v_[ 51] =  1;
   ws_e2v_[ 52] = 21; ws_e2v_[ 53] = 31; ws_e2v_[ 54] = 35; ws_e2v_[ 55] = 27;

   ws_e2v_[ 56] = 21; ws_e2v_[ 57] = 31; ws_e2v_[ 58] = 35; ws_e2v_[ 59] = 27;
   ws_e2v_[ 60] = 13; ws_e2v_[ 61] = 12; ws_e2v_[ 62] = 15; ws_e2v_[ 63] = 14;

   ws_e2v_[ 64] = 16; ws_e2v_[ 65] = 17; ws_e2v_[ 66] = 18; ws_e2v_[ 67] = 19;
   ws_e2v_[ 68] = 28; ws_e2v_[ 69] = 36; ws_e2v_[ 70] = 24; ws_e2v_[ 71] = 11;

   ws_e2v_[ 72] = 28; ws_e2v_[ 73] = 36; ws_e2v_[ 74] = 24; ws_e2v_[ 75] = 11;
   ws_e2v_[ 76] = 33; ws_e2v_[ 77] = 35; ws_e2v_[ 78] = 32; ws_e2v_[ 79] = 34;

   ws_e2v_[ 80] = 33; ws_e2v_[ 81] = 35; ws_e2v_[ 82] = 32; ws_e2v_[ 83] = 34;
   ws_e2v_[ 84] =  5; ws_e2v_[ 85] = 31; ws_e2v_[ 86] = 37; ws_e2v_[ 87] = 29;

   ws_e2v_[ 88] =  5; ws_e2v_[ 89] = 31; ws_e2v_[ 90] = 37; ws_e2v_[ 91] = 29;
   ws_e2v_[ 92] = 20; ws_e2v_[ 93] = 21; ws_e2v_[ 94] = 22; ws_e2v_[ 95] = 23;

   ws_e2v_[ 96] = 24; ws_e2v_[ 97] = 11; ws_e2v_[ 98] = 34; ws_e2v_[ 99] = 32;
   ws_e2v_[100] =  2; ws_e2v_[101] = 10; ws_e2v_[102] = 25; ws_e2v_[103] =  1;

   ws_e2v_[104] = 11; ws_e2v_[105] = 28; ws_e2v_[106] = 33; ws_e2v_[107] = 34;
   ws_e2v_[108] =  8; ws_e2v_[109] =  6; ws_e2v_[110] =  5; ws_e2v_[111] = 29;

   ws_e2v_[112] = 30; ws_e2v_[113] = 15; ws_e2v_[114] = 35; ws_e2v_[115] = 33;
   ws_e2v_[116] =  4; ws_e2v_[117] = 12; ws_e2v_[118] = 31; ws_e2v_[119] =  5;

   ws_e2v_[120] = 15; ws_e2v_[121] = 26; ws_e2v_[122] = 32; ws_e2v_[123] = 35;
   ws_e2v_[124] = 14; ws_e2v_[125] =  0; ws_e2v_[126] =  1; ws_e2v_[127] = 27;

   for (int i=0; i<16; i++) { ws_elem_att_[i] = 1; }

   ws_be2v_[ 0] =  3; ws_be2v_[ 1] =  2; ws_be2v_[ 2] =  1; ws_be2v_[ 3] =  0;
   ws_be2v_[ 4] =  2; ws_be2v_[ 5] =  3; ws_be2v_[ 6] = 18; ws_be2v_[ 7] = 24;
   ws_be2v_[ 8] =  3; ws_be2v_[ 9] =  0; ws_be2v_[10] = 26; ws_be2v_[11] = 18;
   ws_be2v_[12] =  4; ws_be2v_[13] =  5; ws_be2v_[14] =  6; ws_be2v_[15] =  7;
   ws_be2v_[16] = 16; ws_be2v_[17] = 30; ws_be2v_[18] =  4; ws_be2v_[19] =  7;
   ws_be2v_[20] = 16; ws_be2v_[21] =  7; ws_be2v_[22] =  6; ws_be2v_[23] = 28;
   ws_be2v_[24] = 17; ws_be2v_[25] = 15; ws_be2v_[26] = 30; ws_be2v_[27] = 16;
   ws_be2v_[28] = 15; ws_be2v_[29] = 17; ws_be2v_[30] = 18; ws_be2v_[31] = 26;
   ws_be2v_[32] =  9; ws_be2v_[33] = 10; ws_be2v_[34] = 11; ws_be2v_[35] =  8;
   ws_be2v_[36] =  9; ws_be2v_[37] =  8; ws_be2v_[38] = 29; ws_be2v_[39] = 23;
   ws_be2v_[40] =  9; ws_be2v_[41] = 23; ws_be2v_[42] = 25; ws_be2v_[43] = 10;
   ws_be2v_[44] =  1; ws_be2v_[45] = 25; ws_be2v_[46] = 23; ws_be2v_[47] = 22;
   ws_be2v_[48] =  1; ws_be2v_[49] = 22; ws_be2v_[50] = 21; ws_be2v_[51] = 27;
   ws_be2v_[52] = 13; ws_be2v_[53] = 12; ws_be2v_[54] = 15; ws_be2v_[55] = 14;
   ws_be2v_[56] = 21; ws_be2v_[57] = 31; ws_be2v_[58] = 12; ws_be2v_[59] = 13;
   ws_be2v_[60] = 21; ws_be2v_[61] = 13; ws_be2v_[62] = 14; ws_be2v_[63] = 27;
   ws_be2v_[64] = 16; ws_be2v_[65] = 19; ws_be2v_[66] = 18; ws_be2v_[67] = 17;
   ws_be2v_[68] = 11; ws_be2v_[69] = 24; ws_be2v_[70] = 18; ws_be2v_[71] = 19;
   ws_be2v_[72] = 11; ws_be2v_[73] = 19; ws_be2v_[74] = 16; ws_be2v_[75] = 28;
   ws_be2v_[76] = 20; ws_be2v_[77] = 21; ws_be2v_[78] = 22; ws_be2v_[79] = 23;
   ws_be2v_[80] =  5; ws_be2v_[81] = 31; ws_be2v_[82] = 21; ws_be2v_[83] = 20;
   ws_be2v_[84] =  5; ws_be2v_[85] = 20; ws_be2v_[86] = 23; ws_be2v_[87] = 29;
   ws_be2v_[88] =  2; ws_be2v_[89] = 10; ws_be2v_[90] = 25; ws_be2v_[91] =  1;
   ws_be2v_[92] = 24; ws_be2v_[93] = 11; ws_be2v_[94] = 10; ws_be2v_[95] =  2;
   ws_be2v_[96] =  8; ws_be2v_[97] =  6; ws_be2v_[98] =  5; ws_be2v_[99] = 29;
   ws_be2v_[100] = 11; ws_be2v_[101] = 28; ws_be2v_[102] =  6;
   ws_be2v_[103] =  8;
   ws_be2v_[104] =  4; ws_be2v_[105] = 12; ws_be2v_[106] = 31;
   ws_be2v_[107] =  5;
   ws_be2v_[108] = 30; ws_be2v_[109] = 15; ws_be2v_[110] = 12;
   ws_be2v_[111] =  4;
   ws_be2v_[112] = 15; ws_be2v_[113] = 26; ws_be2v_[114] =  0;
   ws_be2v_[115] = 14;
   ws_be2v_[116] = 14; ws_be2v_[117] =  0; ws_be2v_[118] =  1;
   ws_be2v_[119] = 27;

   for (int i=0; i<30; i++) { ws_belem_att_[i] = 1; }
}

bool
FaceCenteredOrthorhombicLattice::MapToFundamentalDomain(const Vector & pt,
                                                        Vector & ipt) const
{
   bool map = false;
   ipt = pt;
   return map;
}

const DenseMatrix &
FaceCenteredOrthorhombicLattice::GetTransformation(int ti) const
{
   T_ = 0.0;

   T_(0, 0) = 1.0;
   T_(1, 1) = 1.0;
   T_(2, 2) = 1.0;

   for (int i=0; i<3; i++)
   {
      if ( ti & (int)pow(2, i) )
      {
         T_(i,0) *= -1.0;
         T_(i,1) *= -1.0;
         T_(i,2) *= -1.0;
      }
   }

   return T_;
}

Mesh *
FaceCenteredOrthorhombicLattice::GetWignerSeitzMesh(bool tetMesh) const
{
   Mesh * mesh = new Mesh((double*)ws_vert_, 38,
                          (int*)ws_e2v_, Geometry::CUBE,
                          (int*)ws_elem_att_, 16,
                          (int*)ws_be2v_, Geometry::SQUARE,
                          (int*)ws_belem_att_, 30,
                          3, 3);
   mesh->Finalize();

   return mesh;
}

Mesh *
FaceCenteredOrthorhombicLattice::GetPeriodicWignerSeitzMesh(bool tetMesh) const
{
   Mesh * mesh = this->GetWignerSeitzMesh(tetMesh);

   {
      std::ofstream ofs("orcf.mesh");
      mesh->Print(ofs);
      ofs.close();
   }

   mesh->UniformRefinement();

   Mesh * per_mesh = MakePeriodicMesh(mesh, trn_vecs_);

   delete mesh;

   {
      std::ofstream ofs("per-orcf.mesh");
      per_mesh->Print(ofs);
      ofs.close();
   }

   return per_mesh;
}

BodyCenteredOrthorhombicLattice::BodyCenteredOrthorhombicLattice(double a,
                                                                 double b,
                                                                 double c)
   : BravaisLattice3D(a, b, c, 0.5 * M_PI, 0.5 * M_PI, 0.5 * M_PI)
{
   if ( c_ > sqrt( a_ * a_ + b_ * b_ ) )
   {
      label_ = "ORCI1";
   }
   else
   {
      label_ = "ORCI2";
   }
   type_  = BODY_CENTERED_ORTHORHOMBIC;

   // Set Lattice Vectors
   lat_vecs_[0][0] = -0.5 * a_;
   lat_vecs_[0][1] =  0.5 * b_;
   lat_vecs_[0][2] =  0.5 * c_;

   lat_vecs_[1][0] =  0.5 * a_;
   lat_vecs_[1][1] = -0.5 * b_;
   lat_vecs_[1][2] =  0.5 * c_;

   lat_vecs_[2][0] =  0.5 * a_;
   lat_vecs_[2][1] =  0.5 * b_;
   lat_vecs_[2][2] = -0.5 * c_;

   // Set Reciprocal Lattice Vectors
   rec_vecs_[0][0] =  0.0;
   rec_vecs_[0][1] =  1.0 / b_;
   rec_vecs_[0][2] =  1.0 / c_;

   rec_vecs_[1][0] =  1.0 / a_;
   rec_vecs_[1][1] =  0.0;
   rec_vecs_[1][2] =  1.0 / c_;

   rec_vecs_[2][0] =  1.0 / a_;
   rec_vecs_[2][1] =  1.0 / b_;
   rec_vecs_[2][2] =  0.0;

   // Set Translation Vectors
   trn_vecs_.resize((c_>sqrt(a_*a_+b_*b_))?6:7);
   for (unsigned int i=0; i<trn_vecs_.size(); i++)
   {
      trn_vecs_[i].SetSize(3);
   }
   if ( c_ > sqrt( a_ * a_ + b_ * b_ ) )
   {
      // Wigner-Seitz cell is an elongated dodecahedron
      trn_vecs_[0][0] =  0.5 * a_;
      trn_vecs_[0][1] =  0.5 * b_;
      trn_vecs_[0][2] =  0.5 * c_;

      trn_vecs_[1][0] = -0.5 * a_;
      trn_vecs_[1][1] =  0.5 * b_;
      trn_vecs_[1][2] =  0.5 * c_;

      trn_vecs_[2][0] = -0.5 * a_;
      trn_vecs_[2][1] = -0.5 * b_;
      trn_vecs_[2][2] =  0.5 * c_;

      trn_vecs_[3][0] =  0.5 * a_;
      trn_vecs_[3][1] = -0.5 * b_;
      trn_vecs_[3][2] =  0.5 * c_;

      trn_vecs_[4][0] =   a_; trn_vecs_[4][1] =  0.0; trn_vecs_[4][2] =  0.0;
      trn_vecs_[5][0] =  0.0; trn_vecs_[5][1] =   b_; trn_vecs_[5][2] =  0.0;

      // Set the face radii
      face_radii_.resize(6);
      for (int i=0; i<4; i++)
         face_radii_[i] = 0.25 * sqrt(a_ * a_ + b_ * b_ + c_ * c_) *
                          min(a_ / sqrt(b_ * b_ + c_ * c_), b_ / sqrt(a_ * a_ + c_ * c_));
      face_radii_[4] = min(0.50 * b_,
                           0.25 * (b_ * b_ - a_ * a_ + c_ * c_) /
                           sqrt(b_ * b_ + c_ * c_)
                          );
      face_radii_[5] = min(0.50 * a_,
                           0.25 * (a_ * a_ - b_ * b_ + c_ * c_) /
                           sqrt(a_ * a_ + c_ * c_)
                          );
   }
   else
   {
      // Wigner-Seitz cell is a truncated octahedron
      trn_vecs_[0][0] =   a_; trn_vecs_[0][1] =  0.0; trn_vecs_[0][2] =  0.0;
      trn_vecs_[1][0] =  0.0; trn_vecs_[1][1] =   b_; trn_vecs_[1][2] =  0.0;
      trn_vecs_[2][0] =  0.0; trn_vecs_[2][1] =  0.0; trn_vecs_[2][2] =   c_;

      trn_vecs_[3][0] =  0.5 * a_;
      trn_vecs_[3][1] =  0.5 * b_;
      trn_vecs_[3][2] =  0.5 * c_;

      trn_vecs_[4][0] = -0.5 * a_;
      trn_vecs_[4][1] =  0.5 * b_;
      trn_vecs_[4][2] =  0.5 * c_;

      trn_vecs_[5][0] = -0.5 * a_;
      trn_vecs_[5][1] = -0.5 * b_;
      trn_vecs_[5][2] =  0.5 * c_;

      trn_vecs_[6][0] =  0.5 * a_;
      trn_vecs_[6][1] = -0.5 * b_;
      trn_vecs_[6][2] =  0.5 * c_;

      // Set the face radii
      face_radii_.resize(7);
      face_radii_[0] = 0.25 * (b_ * b_ - a_ * a_ + c_ * c_) /
                       sqrt(b_ * b_ + c_ * c_);
      face_radii_[1] = 0.25 * (a_ * a_ - b_ * b_ + c_ * c_) /
                       sqrt(a_ * a_ + c_ * c_);
      face_radii_[2] = 0.25 * (a_ * a_ + b_ * b_ - c_ * c_) /
                       sqrt(a_ * a_ + b_ * b_);
      face_radii_[3] = 0.25 * sqrt(a_ * a_ + b_ * b_ + c_ * c_) *
                       min(a_ / sqrt(b_ * b_ + c_ * c_),
                           min(b_ / sqrt(a_ * a_ + c_ * c_),
                               c_ / sqrt(a_ * a_ + b_ * b_))
                          );
      for (int i=4; i<7; i++) { face_radii_[i] = face_radii_[3]; }
   }

   this->SetCellVolumes();

   // Allocate sl_, sp_, and path_ vectors
   this->SetVectorSizes();

   // Set Symmetry Points and Labels
   double a2 = a_ * a_;
   double b2 = b_ * b_;
   double c2 = c_ * c_;
   double  zeta = 0.25 * (1.0 + a2 / c2);
   double delta = 0.25 * (b2 - a2) / c2;
   double   eta = 0.25 * (1.0 + b2 / c2);
   double    mu = 0.25 * (a2 + b2) / c2;

   sl_[0] = "Gamma";
   sp_[0] = 0.0;

   sl_[1] = "W";
   add(0.25, rec_vecs_[0], 0.25, rec_vecs_[1], sp_[1]);
   sp_[1].Add(0.25, rec_vecs_[2]);

   sl_[2] = "L";
   add(-mu, rec_vecs_[0], mu, rec_vecs_[1], sp_[2]);
   sp_[2].Add(0.5-delta, rec_vecs_[2]);

   sl_[3] = "X";
   add(-zeta, rec_vecs_[0], zeta, rec_vecs_[1], sp_[3]);
   sp_[3].Add(zeta, rec_vecs_[2]);

   sl_[4] = "L1";
   add(mu, rec_vecs_[0], -mu, rec_vecs_[1], sp_[4]);
   sp_[4].Add(0.5+delta, rec_vecs_[2]);

   sl_[5] = "X1";
   add(zeta, rec_vecs_[0], 1.0-zeta, rec_vecs_[1], sp_[5]);
   sp_[5].Add(-zeta, rec_vecs_[2]);

   sl_[6] = "L2"; /// Note that this point does not fall on any path
   add(0.5-delta, rec_vecs_[0], 0.5+delta, rec_vecs_[1], sp_[6]);
   sp_[6].Add(-mu, rec_vecs_[2]);

   sl_[7] = "Y";
   add(eta, rec_vecs_[0], -eta, rec_vecs_[1], sp_[7]);
   sp_[7].Add(eta, rec_vecs_[2]);

   sl_[8] = "R";
   sp_[8].Set(0.5,rec_vecs_[1]);

   sl_[9] = "Y1";
   add(1.0-eta, rec_vecs_[0], eta, rec_vecs_[1], sp_[9]);
   sp_[9].Add(-eta, rec_vecs_[2]);

   sl_[10] = "S";
   sp_[10].Set(0.5,rec_vecs_[0]);

   sl_[11] = "Z";
   add(0.5, rec_vecs_[0], 0.5, rec_vecs_[1], sp_[11]);
   sp_[11].Add(-0.5, rec_vecs_[2]);

   sl_[12] = "T";
   sp_[12].Set(0.5,rec_vecs_[2]);

   for (unsigned int i=0; i<sl_.size(); i++)
   {
      si_[sl_[i]] = i;
   }

   // Define Paths
   path_[0][ 0] =  0;
   path_[0][ 1] =  3;
   path_[0][ 2] =  2;
   path_[0][ 3] = 12;
   path_[0][ 4] =  1;
   path_[0][ 5] =  8;
   path_[0][ 6] =  5;
   path_[0][ 7] = 11;
   path_[0][ 8] =  0;
   path_[0][ 9] =  7;
   path_[0][10] = 10;
   path_[0][11] =  1;

   path_[1][ 0] =  4;
   path_[1][ 1] =  7;

   path_[2][ 0] =  9;
   path_[2][ 1] = 11;

   // Set Intermediate Symmetry Points
   this->SetIntermediatePoints();

   // Set Intermediate Symmetry Point Labels
   il_[0][ 0] = "GammaX"; // Gamma -> X
   il_[0][ 1] = "XL";     // X     -> L
   il_[0][ 2] = "LT";     // L     -> T
   il_[0][ 3] = "TW";     // T     -> W
   il_[0][ 4] = "WR";     // W     -> R
   il_[0][ 5] = "RX1";    // R     -> X1
   il_[0][ 6] = "X1Z";    // X1    -> Z
   il_[0][ 7] = "ZGamma"; // Z     -> Gamma
   il_[0][ 8] = "GammaY"; // Gamma -> Y
   il_[0][ 9] = "YS";     // Y     -> S
   il_[0][10] = "SW";     // S     -> W
   il_[1][ 0] = "L1Y";    // L1    -> Y
   il_[2][ 0] = "Y1Z";    // Y1    -> Z

   // Set Mesh data
   if ( c_ > sqrt( a_ * a_ + b_ * b_ ) )
   {
      this->createElongatedDodecahedron();
   }
   else
   {
      this->createTruncatedOctahedron();
   }
}

void
BodyCenteredOrthorhombicLattice::createElongatedDodecahedron()
{
   cout << "Setting up data for elongated dodecahedron" << endl;

   // Set Mesh data
   for (int i=0; i<114; i++) { ws_vert_[i] = 0.0; }

   for (int i=0; i<3; i++)
   {
      for (int j=0; j<4; j++)
      {
         ws_vert_[3 * (4 * i + j)]      = 0.5 * a_ * (i - 1);
         ws_vert_[3 * (4 * i + j) + 37] = 0.5 * b_ * (i - 1);

         ws_vert_[3 * (4 * i + j) + 2]  =
            - 0.25 * ( a_ * a_ - 3.0 * b_ * b_ + c_ * c_) / c_
            + 0.50 * ( a_ * a_ * fabs(i - 1) - b_ * b_ * fabs(j - 2) ) / c_;
         ws_vert_[3 * (4 * i + j) + 38] =
            + 0.25 * ( b_ * b_ - 3.0 * a_ * a_ + c_ * c_ ) / c_
            - 0.50 * ( b_ * b_ * fabs(i - 1) - a_ * a_ * fabs(j - 2) ) / c_;
      }
   }
   for (int i=0; i<3; i++)
   {
      for (int j=1; j<4; j++)
      {
         ws_vert_[3 * (4 * i + j) + 1]  = 0.5 * b_ * (2 - j);
         ws_vert_[3 * (4 * i + j) + 36] = 0.5 * a_ * (2 - j);
      }
   }

   for (int i=0; i<2; i++)
   {
      for (int j=0; j<8; j++)
      {
         ws_e2v_[8 * i + j]      = 4 * i + j;
         ws_e2v_[8 * i + j + 16] = 4 * i + j + 12;
      }
   }

   ws_e2v_[32] =  6; ws_e2v_[33] =  2; ws_e2v_[34] =  3; ws_e2v_[35] =  7;
   ws_e2v_[36] = 18; ws_e2v_[37] = 19; ws_e2v_[38] = 15; ws_e2v_[39] = 14;

   ws_e2v_[40] =  6; ws_e2v_[41] =  7; ws_e2v_[42] = 11; ws_e2v_[43] = 10;
   ws_e2v_[44] = 18; ws_e2v_[45] = 14; ws_e2v_[46] = 13; ws_e2v_[47] = 17;

   ws_e2v_[48] =  6; ws_e2v_[49] = 10; ws_e2v_[50] =  9; ws_e2v_[51] =  5;
   ws_e2v_[52] = 18; ws_e2v_[53] = 17; ws_e2v_[54] = 21; ws_e2v_[55] = 22;

   ws_e2v_[56] =  6; ws_e2v_[57] =  5; ws_e2v_[58] =  1; ws_e2v_[59] =  2;
   ws_e2v_[60] = 18; ws_e2v_[61] = 22; ws_e2v_[62] = 23; ws_e2v_[63] = 19;

   for (int i=64; i<128; i++) { ws_e2v_[i] = -1; }

   for (int i=0; i< 8; i++) { ws_elem_att_[i] =  1; }
   for (int i=8; i<16; i++) { ws_elem_att_[i] = -1; }

   ws_be2v_[ 0] =  4; ws_be2v_[ 1] =  7; ws_be2v_[ 2] =  3; ws_be2v_[ 3] =  0;
   ws_be2v_[ 4] =  4; ws_be2v_[ 5] =  8; ws_be2v_[ 6] = 11; ws_be2v_[ 7] =  7;
   ws_be2v_[ 8] =  4; ws_be2v_[ 9] =  5; ws_be2v_[10] =  9; ws_be2v_[11] =  8;
   ws_be2v_[12] =  4; ws_be2v_[13] =  0; ws_be2v_[14] =  1; ws_be2v_[15] =  5;

   ws_be2v_[16] = 16; ws_be2v_[17] = 19; ws_be2v_[18] = 15; ws_be2v_[19] = 12;
   ws_be2v_[20] = 16; ws_be2v_[21] = 12; ws_be2v_[22] = 13; ws_be2v_[23] = 17;
   ws_be2v_[24] = 16; ws_be2v_[25] = 17; ws_be2v_[26] = 21; ws_be2v_[27] = 20;
   ws_be2v_[28] = 16; ws_be2v_[29] = 20; ws_be2v_[30] = 23; ws_be2v_[31] = 19;

   ws_be2v_[32] =  0; ws_be2v_[33] =  3; ws_be2v_[34] =  2; ws_be2v_[35] =  1;
   ws_be2v_[36] =  8; ws_be2v_[37] =  9; ws_be2v_[38] = 10; ws_be2v_[39] = 11;
   ws_be2v_[40] = 12; ws_be2v_[41] = 15; ws_be2v_[42] = 14; ws_be2v_[43] = 13;
   ws_be2v_[44] = 20; ws_be2v_[45] = 21; ws_be2v_[46] = 22; ws_be2v_[47] = 23;

   ws_be2v_[48] =  2; ws_be2v_[49] =  3; ws_be2v_[50] = 15; ws_be2v_[51] = 19;
   ws_be2v_[52] =  3; ws_be2v_[53] =  7; ws_be2v_[54] = 14; ws_be2v_[55] = 15;
   ws_be2v_[56] =  7; ws_be2v_[57] = 11; ws_be2v_[58] = 13; ws_be2v_[59] = 14;
   ws_be2v_[60] = 11; ws_be2v_[61] = 10; ws_be2v_[62] = 17; ws_be2v_[63] = 13;

   ws_be2v_[64] = 10; ws_be2v_[65] =  9; ws_be2v_[66] = 21; ws_be2v_[67] = 17;
   ws_be2v_[68] =  9; ws_be2v_[69] =  5; ws_be2v_[70] = 22; ws_be2v_[71] = 21;
   ws_be2v_[72] =  5; ws_be2v_[73] =  1; ws_be2v_[74] = 23; ws_be2v_[75] = 22;
   ws_be2v_[76] =  1; ws_be2v_[77] =  2; ws_be2v_[78] = 19; ws_be2v_[79] = 23;

   for (int i=80; i<120; i++) { ws_be2v_[i] = -1; }

   for (int i= 0; i<20; i++) { ws_belem_att_[i] =  1; }
   for (int i=20; i<30; i++) { ws_belem_att_[i] = -1; }
}

void
BodyCenteredOrthorhombicLattice::createTruncatedOctahedron()
{
   cout << "Setting up data for truncated octahedron" << endl;

   // Set Mesh data
   for (int i=0; i<114; i++) { ws_vert_[i] = 0.0; }

   ws_vert_[  0] = -0.50 * a_;
   ws_vert_[  1] =  0.25 * ( c_ * c_ - a_ * a_ + b_ * b_ ) / b_;

   ws_vert_[  3] = -0.50 * a_;
   ws_vert_[  5] =  0.25 * ( c_ * c_ - a_ * a_ + b_ * b_ ) / c_;

   ws_vert_[  6] = -0.50 * a_;
   ws_vert_[  7] = -0.25 * ( c_ * c_ - a_ * a_ + b_ * b_ ) / b_;

   ws_vert_[  9] = -0.50 * a_;
   ws_vert_[ 11] = -0.25 * ( c_ * c_ - a_ * a_ + b_ * b_ ) / c_;

   ws_vert_[ 12] =  0.50 * a_;
   ws_vert_[ 13] =  0.25 * ( c_ * c_ - a_ * a_ + b_ * b_ ) / b_;

   ws_vert_[ 15] =  0.50 * a_;
   ws_vert_[ 17] =  0.25 * ( c_ * c_ - a_ * a_ + b_ * b_ ) / c_;

   ws_vert_[ 18] =  0.50 * a_;
   ws_vert_[ 19] = -0.25 * ( c_ * c_ - a_ * a_ + b_ * b_ ) / b_;

   ws_vert_[ 21] =  0.50 * a_;
   ws_vert_[ 23] = -0.25 * ( c_ * c_ - a_ * a_ + b_ * b_ ) / c_;

   ws_vert_[ 24] =  0.25 * ( c_ * c_ + a_ * a_ - b_ * b_ ) / a_;
   ws_vert_[ 25] = -0.50 * b_;

   ws_vert_[ 28] = -0.50 * b_;
   ws_vert_[ 29] =  0.25 * ( c_ * c_ + a_ * a_ - b_ * b_ ) / c_;

   ws_vert_[ 30] = -0.25 * ( c_ * c_ + a_ * a_ - b_ * b_ ) / a_;
   ws_vert_[ 31] = -0.50 * b_;

   ws_vert_[ 34] = -0.50 * b_;
   ws_vert_[ 35] = -0.25 * ( c_ * c_ + a_ * a_ - b_ * b_ ) / c_;

   ws_vert_[ 36] =  0.25 * ( c_ * c_ + a_ * a_ - b_ * b_ ) / a_;
   ws_vert_[ 37] =  0.50 * b_;

   ws_vert_[ 40] =  0.50 * b_;
   ws_vert_[ 41] =  0.25 * ( c_ * c_ + a_ * a_ - b_ * b_ ) / c_;

   ws_vert_[ 42] = -0.25 * ( c_ * c_ + a_ * a_ - b_ * b_ ) / a_;
   ws_vert_[ 43] =  0.50 * b_;

   ws_vert_[ 46] =  0.50 * b_;
   ws_vert_[ 47] = -0.25 * ( c_ * c_ + a_ * a_ - b_ * b_ ) / c_;

   ws_vert_[ 48] = -0.25 * ( c_ * c_ - a_ * a_ - b_ * b_ ) / a_;
   ws_vert_[ 50] = -0.50 * c_;

   ws_vert_[ 52] = -0.25 * ( c_ * c_ - a_ * a_ - b_ * b_ ) / b_;
   ws_vert_[ 53] = -0.50 * c_;

   ws_vert_[ 54] =  0.25 * ( c_ * c_ - a_ * a_ - b_ * b_ ) / a_;
   ws_vert_[ 56] = -0.50 * c_;

   ws_vert_[ 58] =  0.25 * ( c_ * c_ - a_ * a_ - b_ * b_ ) / b_;
   ws_vert_[ 59] = -0.50 * c_;

   ws_vert_[ 60] = -0.25 * ( c_ * c_ - a_ * a_ - b_ * b_ ) / a_;
   ws_vert_[ 62] =  0.50 * c_;

   ws_vert_[ 64] = -0.25 * ( c_ * c_ - a_ * a_ - b_ * b_ ) / b_;
   ws_vert_[ 65] =  0.50 * c_;

   ws_vert_[ 66] =  0.25 * ( c_ * c_ - a_ * a_ - b_ * b_ ) / a_;
   ws_vert_[ 68] =  0.50 * c_;

   ws_vert_[ 70] =  0.25 * ( c_ * c_ - a_ * a_ - b_ * b_ ) / b_;
   ws_vert_[ 71] =  0.50 * c_;

   ws_vert_[ 72] =  0.25 * ( c_ * c_ - a_ * a_ - b_ * b_ ) / a_;
   ws_vert_[ 73] = -0.25 * ( c_ * c_ - a_ * a_ + b_ * b_ ) / b_;
   ws_vert_[ 74] = -0.25 * ( c_ * c_ + a_ * a_ - b_ * b_ ) / c_;

   ws_vert_[ 75] = -0.25 * ( c_ * c_ + a_ * a_ - b_ * b_ ) / a_;
   ws_vert_[ 76] =  0.25 * ( c_ * c_ - a_ * a_ - b_ * b_ ) / b_;
   ws_vert_[ 77] =  0.25 * ( c_ * c_ - a_ * a_ + b_ * b_ ) / c_;

   ws_vert_[ 78] =  0.25 * ( c_ * c_ - a_ * a_ - b_ * b_ ) / a_;
   ws_vert_[ 79] =  0.25 * ( c_ * c_ - a_ * a_ + b_ * b_ ) / b_;
   ws_vert_[ 80] = -0.25 * ( c_ * c_ + a_ * a_ - b_ * b_ ) / c_;

   ws_vert_[ 81] = -0.25 * ( c_ * c_ + a_ * a_ - b_ * b_ ) / a_;
   ws_vert_[ 82] = -0.25 * ( c_ * c_ - a_ * a_ - b_ * b_ ) / b_;
   ws_vert_[ 83] =  0.25 * ( c_ * c_ - a_ * a_ + b_ * b_ ) / c_;

   ws_vert_[ 84] = -0.25 * ( c_ * c_ - a_ * a_ - b_ * b_ ) / a_;
   ws_vert_[ 85] = -0.25 * ( c_ * c_ - a_ * a_ + b_ * b_ ) / b_;
   ws_vert_[ 86] = -0.25 * ( c_ * c_ + a_ * a_ - b_ * b_ ) / c_;

   ws_vert_[ 87] =  0.25 * ( c_ * c_ + a_ * a_ - b_ * b_ ) / a_;
   ws_vert_[ 88] =  0.25 * ( c_ * c_ - a_ * a_ - b_ * b_ ) / b_;
   ws_vert_[ 89] =  0.25 * ( c_ * c_ - a_ * a_ + b_ * b_ ) / c_;

   ws_vert_[ 90] = -0.25 * ( c_ * c_ - a_ * a_ - b_ * b_ ) / a_;
   ws_vert_[ 91] =  0.25 * ( c_ * c_ - a_ * a_ + b_ * b_ ) / b_;
   ws_vert_[ 92] = -0.25 * ( c_ * c_ + a_ * a_ - b_ * b_ ) / c_;

   ws_vert_[ 93] =  0.25 * ( c_ * c_ + a_ * a_ - b_ * b_ ) / a_;
   ws_vert_[ 94] = -0.25 * ( c_ * c_ - a_ * a_ - b_ * b_ ) / b_;
   ws_vert_[ 95] =  0.25 * ( c_ * c_ - a_ * a_ + b_ * b_ ) / c_;

   ws_vert_[ 96] =  0.25 * ( c_ * c_ - a_ * a_ - b_ * b_ ) / a_;
   ws_vert_[ 98] =  0.50 * ( b_ * b_ - a_ * a_ ) / c_;

   ws_vert_[ 99] = -0.25 * ( c_ * c_ - a_ * a_ - b_ * b_ ) / a_;
   ws_vert_[101] =  0.50 * ( b_ * b_ - a_ * a_ ) / c_;

   ws_vert_[103] =  0.25 * ( c_ * c_ - a_ * a_ - b_ * b_ ) / b_;
   ws_vert_[104] =  0.50 * ( b_ * b_ - a_ * a_ ) / c_;

   ws_vert_[106] = -0.25 * ( c_ * c_ - a_ * a_ - b_ * b_ ) / b_;
   ws_vert_[107] =  0.50 * ( b_ * b_ - a_ * a_ ) / c_;

   ws_vert_[109] = -0.50 * ( c_ * c_ - a_ * a_ ) / b_;
   ws_vert_[110] = -0.25 * ( c_ * c_ + a_ * a_ - b_ * b_ ) / c_;

   ws_vert_[111] =  0.50 * ( c_ * c_ - b_ * b_ ) / a_;
   ws_vert_[113] =  0.25 * ( c_ * c_ - a_ * a_ + b_ * b_ ) / c_;

   ws_e2v_[  0] =  0; ws_e2v_[  1] =  1; ws_e2v_[  2] =  2; ws_e2v_[  3] =  3;
   ws_e2v_[  4] = 26; ws_e2v_[  5] = 32; ws_e2v_[  6] = 24; ws_e2v_[  7] = 18;

   ws_e2v_[  8] = 26; ws_e2v_[  9] = 32; ws_e2v_[ 10] = 24; ws_e2v_[ 11] = 18;
   ws_e2v_[ 12] = 15; ws_e2v_[ 13] = 35; ws_e2v_[ 14] = 36; ws_e2v_[ 15] = 17;

   ws_e2v_[ 16] = 15; ws_e2v_[ 17] = 35; ws_e2v_[ 18] = 36; ws_e2v_[ 19] = 17;
   ws_e2v_[ 20] = 30; ws_e2v_[ 21] = 33; ws_e2v_[ 22] = 28; ws_e2v_[ 23] = 16;

   ws_e2v_[ 24] = 30; ws_e2v_[ 25] = 33; ws_e2v_[ 26] = 28; ws_e2v_[ 27] = 16;
   ws_e2v_[ 28] =  4; ws_e2v_[ 29] =  5; ws_e2v_[ 30] =  6; ws_e2v_[ 31] =  7;

   ws_e2v_[ 32] =  9; ws_e2v_[ 33] =  8; ws_e2v_[ 34] = 11; ws_e2v_[ 35] = 10;
   ws_e2v_[ 36] = 23; ws_e2v_[ 37] = 29; ws_e2v_[ 38] = 34; ws_e2v_[ 39] = 25;

   ws_e2v_[ 40] = 23; ws_e2v_[ 41] = 29; ws_e2v_[ 42] = 34; ws_e2v_[ 43] = 25;
   ws_e2v_[ 44] = 22; ws_e2v_[ 45] = 37; ws_e2v_[ 46] = 32; ws_e2v_[ 47] =  1;

   ws_e2v_[ 48] = 22; ws_e2v_[ 49] = 37; ws_e2v_[ 50] = 32; ws_e2v_[ 51] =  1;
   ws_e2v_[ 52] = 21; ws_e2v_[ 53] = 31; ws_e2v_[ 54] = 35; ws_e2v_[ 55] = 27;

   ws_e2v_[ 56] = 21; ws_e2v_[ 57] = 31; ws_e2v_[ 58] = 35; ws_e2v_[ 59] = 27;
   ws_e2v_[ 60] = 13; ws_e2v_[ 61] = 12; ws_e2v_[ 62] = 15; ws_e2v_[ 63] = 14;

   ws_e2v_[ 64] = 16; ws_e2v_[ 65] = 17; ws_e2v_[ 66] = 18; ws_e2v_[ 67] = 19;
   ws_e2v_[ 68] = 28; ws_e2v_[ 69] = 36; ws_e2v_[ 70] = 24; ws_e2v_[ 71] = 11;

   ws_e2v_[ 72] = 28; ws_e2v_[ 73] = 36; ws_e2v_[ 74] = 24; ws_e2v_[ 75] = 11;
   ws_e2v_[ 76] = 33; ws_e2v_[ 77] = 35; ws_e2v_[ 78] = 32; ws_e2v_[ 79] = 34;

   ws_e2v_[ 80] = 33; ws_e2v_[ 81] = 35; ws_e2v_[ 82] = 32; ws_e2v_[ 83] = 34;
   ws_e2v_[ 84] =  5; ws_e2v_[ 85] = 31; ws_e2v_[ 86] = 37; ws_e2v_[ 87] = 29;

   ws_e2v_[ 88] =  5; ws_e2v_[ 89] = 31; ws_e2v_[ 90] = 37; ws_e2v_[ 91] = 29;
   ws_e2v_[ 92] = 20; ws_e2v_[ 93] = 21; ws_e2v_[ 94] = 22; ws_e2v_[ 95] = 23;

   ws_e2v_[ 96] = 24; ws_e2v_[ 97] = 11; ws_e2v_[ 98] = 34; ws_e2v_[ 99] = 32;
   ws_e2v_[100] =  2; ws_e2v_[101] = 10; ws_e2v_[102] = 25; ws_e2v_[103] =  1;

   ws_e2v_[104] = 11; ws_e2v_[105] = 28; ws_e2v_[106] = 33; ws_e2v_[107] = 34;
   ws_e2v_[108] =  8; ws_e2v_[109] =  6; ws_e2v_[110] =  5; ws_e2v_[111] = 29;

   ws_e2v_[112] = 30; ws_e2v_[113] = 15; ws_e2v_[114] = 35; ws_e2v_[115] = 33;
   ws_e2v_[116] =  4; ws_e2v_[117] = 12; ws_e2v_[118] = 31; ws_e2v_[119] =  5;

   ws_e2v_[120] = 15; ws_e2v_[121] = 26; ws_e2v_[122] = 32; ws_e2v_[123] = 35;
   ws_e2v_[124] = 14; ws_e2v_[125] =  0; ws_e2v_[126] =  1; ws_e2v_[127] = 27;

   for (int i=0; i<16; i++) { ws_elem_att_[i] = 1; }

   ws_be2v_[ 0] =  3; ws_be2v_[ 1] =  2; ws_be2v_[ 2] =  1; ws_be2v_[ 3] =  0;
   ws_be2v_[ 4] =  2; ws_be2v_[ 5] =  3; ws_be2v_[ 6] = 18; ws_be2v_[ 7] = 24;
   ws_be2v_[ 8] =  3; ws_be2v_[ 9] =  0; ws_be2v_[10] = 26; ws_be2v_[11] = 18;
   ws_be2v_[12] =  4; ws_be2v_[13] =  5; ws_be2v_[14] =  6; ws_be2v_[15] =  7;
   ws_be2v_[16] = 16; ws_be2v_[17] = 30; ws_be2v_[18] =  4; ws_be2v_[19] =  7;
   ws_be2v_[20] = 16; ws_be2v_[21] =  7; ws_be2v_[22] =  6; ws_be2v_[23] = 28;
   ws_be2v_[24] = 17; ws_be2v_[25] = 15; ws_be2v_[26] = 30; ws_be2v_[27] = 16;
   ws_be2v_[28] = 15; ws_be2v_[29] = 17; ws_be2v_[30] = 18; ws_be2v_[31] = 26;
   ws_be2v_[32] =  9; ws_be2v_[33] = 10; ws_be2v_[34] = 11; ws_be2v_[35] =  8;
   ws_be2v_[36] =  9; ws_be2v_[37] =  8; ws_be2v_[38] = 29; ws_be2v_[39] = 23;
   ws_be2v_[40] =  9; ws_be2v_[41] = 23; ws_be2v_[42] = 25; ws_be2v_[43] = 10;
   ws_be2v_[44] =  1; ws_be2v_[45] = 25; ws_be2v_[46] = 23; ws_be2v_[47] = 22;
   ws_be2v_[48] =  1; ws_be2v_[49] = 22; ws_be2v_[50] = 21; ws_be2v_[51] = 27;
   ws_be2v_[52] = 13; ws_be2v_[53] = 12; ws_be2v_[54] = 15; ws_be2v_[55] = 14;
   ws_be2v_[56] = 21; ws_be2v_[57] = 31; ws_be2v_[58] = 12; ws_be2v_[59] = 13;
   ws_be2v_[60] = 21; ws_be2v_[61] = 13; ws_be2v_[62] = 14; ws_be2v_[63] = 27;
   ws_be2v_[64] = 16; ws_be2v_[65] = 19; ws_be2v_[66] = 18; ws_be2v_[67] = 17;
   ws_be2v_[68] = 11; ws_be2v_[69] = 24; ws_be2v_[70] = 18; ws_be2v_[71] = 19;
   ws_be2v_[72] = 11; ws_be2v_[73] = 19; ws_be2v_[74] = 16; ws_be2v_[75] = 28;
   ws_be2v_[76] = 20; ws_be2v_[77] = 21; ws_be2v_[78] = 22; ws_be2v_[79] = 23;
   ws_be2v_[80] =  5; ws_be2v_[81] = 31; ws_be2v_[82] = 21; ws_be2v_[83] = 20;
   ws_be2v_[84] =  5; ws_be2v_[85] = 20; ws_be2v_[86] = 23; ws_be2v_[87] = 29;
   ws_be2v_[88] =  2; ws_be2v_[89] = 10; ws_be2v_[90] = 25; ws_be2v_[91] =  1;
   ws_be2v_[92] = 24; ws_be2v_[93] = 11; ws_be2v_[94] = 10; ws_be2v_[95] =  2;
   ws_be2v_[96] =  8; ws_be2v_[97] =  6; ws_be2v_[98] =  5; ws_be2v_[99] = 29;
   ws_be2v_[100] = 11; ws_be2v_[101] = 28; ws_be2v_[102] =  6;
   ws_be2v_[103] =  8;
   ws_be2v_[104] =  4; ws_be2v_[105] = 12; ws_be2v_[106] = 31;
   ws_be2v_[107] =  5;
   ws_be2v_[108] = 30; ws_be2v_[109] = 15; ws_be2v_[110] = 12;
   ws_be2v_[111] =  4;
   ws_be2v_[112] = 15; ws_be2v_[113] = 26; ws_be2v_[114] =  0;
   ws_be2v_[115] = 14;
   ws_be2v_[116] = 14; ws_be2v_[117] =  0; ws_be2v_[118] =  1;
   ws_be2v_[119] = 27;

   for (int i=0; i<30; i++) { ws_belem_att_[i] = 1; }
}

bool
BodyCenteredOrthorhombicLattice::MapToFundamentalDomain(const Vector & pt,
                                                        Vector & ipt) const
{
   bool map = false;
   ipt = pt;
   return map;
}

const DenseMatrix &
BodyCenteredOrthorhombicLattice::GetTransformation(int ti) const
{
   T_ = 0.0;

   T_(0, 0) = 1.0;
   T_(1, 1) = 1.0;
   T_(2, 2) = 1.0;

   for (int i=0; i<3; i++)
   {
      if ( ti & (int)pow(2, i) )
      {
         T_(i,0) *= -1.0;
         T_(i,1) *= -1.0;
         T_(i,2) *= -1.0;
      }
   }

   return T_;
}

Mesh *
BodyCenteredOrthorhombicLattice::GetWignerSeitzMesh(bool tetMesh) const
{
   bool ed = c_ > sqrt( a_ * a_ + b_ * b_ );

   Mesh * mesh = new Mesh((double*)ws_vert_, ed?24:38,
                          (int*)ws_e2v_, Geometry::CUBE,
                          (int*)ws_elem_att_, ed?8:16,
                          (int*)ws_be2v_, Geometry::SQUARE,
                          (int*)ws_belem_att_, ed?20:30,
                          3, 3);

   mesh->Finalize();

   return mesh;
}

Mesh *
BodyCenteredOrthorhombicLattice::GetPeriodicWignerSeitzMesh(bool tetMesh) const
{
   Mesh * mesh = this->GetWignerSeitzMesh(tetMesh);

   {
      std::ofstream ofs("orci.mesh");
      mesh->Print(ofs);
      ofs.close();
   }

   mesh->UniformRefinement();
   {
      std::ofstream ofs("orci_r1.mesh");
      mesh->Print(ofs);
      ofs.close();
   }

   Mesh * per_mesh = MakePeriodicMesh(mesh, trn_vecs_);

   delete mesh;

   {
      std::ofstream ofs("per-orci.mesh");
      per_mesh->Print(ofs);
      ofs.close();
   }

   return per_mesh;
}

BaseCenteredOrthorhombicLattice::BaseCenteredOrthorhombicLattice(double a,
                                                                 double b,
                                                                 double c)
   : BravaisLattice3D(a, b, c, 0.5 * M_PI, 0.5 * M_PI, 0.5 * M_PI)
{
   MFEM_ASSERT( a_ < b_ , "Base-Centered Orthorhombic unit cells "
                "must have a < b!");

   label_ = "ORCC";
   type_  = BASE_CENTERED_ORTHORHOMBIC;

   // Set Lattice Vectors
   lat_vecs_[0][0] =  0.5 * a_;
   lat_vecs_[0][1] = -0.5 * b_;
   lat_vecs_[0][2] =  0.0;

   lat_vecs_[1][0] =  0.5 * a_;
   lat_vecs_[1][1] =  0.5 * b_;
   lat_vecs_[1][2] =  0.0;

   lat_vecs_[2][0] =  0.0;
   lat_vecs_[2][1] =  0.0;
   lat_vecs_[2][2] =   c_;

   // Set Reciprocal Lattice Vectors
   rec_vecs_[0][0] =  1.0 / a_;
   rec_vecs_[0][1] = -1.0 / b_;
   rec_vecs_[0][2] =  0.0;

   rec_vecs_[1][0] =  1.0 / a_;
   rec_vecs_[1][1] =  1.0 / b_;
   rec_vecs_[1][2] =  0.0;

   rec_vecs_[2][0] =  0.0;
   rec_vecs_[2][1] =  0.0;
   rec_vecs_[2][2] =  1.0 / c_;

   // Set the Translation Vectors
   trn_vecs_.resize(4);
   for (unsigned int i=0; i<trn_vecs_.size(); i++)
   {
      trn_vecs_[i].SetSize(3);
   }
   trn_vecs_[0][0] =  1.0 * a;
   trn_vecs_[0][1] =  0.0;
   trn_vecs_[0][2] =  0.0;

   trn_vecs_[1][0] =  0.5 * a;
   trn_vecs_[1][1] =  0.5 * b;
   trn_vecs_[1][2] =  0.0;

   trn_vecs_[2][0] =  0.5 * a;
   trn_vecs_[2][1] = -0.5 * b;
   trn_vecs_[2][2] =  0.0;

   trn_vecs_[3][0] =  0.0;
   trn_vecs_[3][1] =  0.0;
   trn_vecs_[3][2] =  c;

   // Set the face radii
   face_radii_.resize(4);
   face_radii_[0] = min(0.25 * (b_ - a_* a_ / b_), 0.5 * c_);
   face_radii_[1] = min(0.25 * a_ * sqrt(a_ * a_ + b_* b_) / b_, 0.5 * c_);
   face_radii_[2] = min(0.25 * a_ * sqrt(a_ * a_ + b_* b_) / b_, 0.5 * c_);
   face_radii_[3] = min(0.5 * a_,
                        0.25 * sqrt(a_ * a_ + b_ * b_));

   this->SetCellVolumes();

   // Allocate sl_, sp_, and path_ vectors
   this->SetVectorSizes();

   // Set Symmetry Points and Labels
   double a2 = a_ * a_;
   double b2 = b_ * b_;
   double zeta = 0.25 * (1.0 + a2 / b2);

   sl_[0] = "Gamma";
   sp_[0] = 0.0;

   sl_[1] = "T";
   add(-0.5, rec_vecs_[0], 0.5, rec_vecs_[1], sp_[1]);
   sp_[1].Add(0.5, rec_vecs_[2]);

   sl_[2] = "A";
   add(zeta, rec_vecs_[0], zeta, rec_vecs_[1], sp_[2]);
   sp_[2].Add(0.5, rec_vecs_[2]);

   sl_[3] = "X";
   add(zeta, rec_vecs_[0], zeta, rec_vecs_[1], sp_[3]);

   sl_[4] = "A1";
   add(-zeta, rec_vecs_[0], 1.0-zeta, rec_vecs_[1], sp_[4]);
   sp_[4].Add(0.5, rec_vecs_[2]);

   sl_[5] = "X1";
   add(-zeta, rec_vecs_[0], 1.0-zeta, rec_vecs_[1], sp_[5]);

   sl_[6] = "R";
   add(0.5, rec_vecs_[1], 0.5, rec_vecs_[2], sp_[6]);

   sl_[7] = "Y";
   add(-0.5, rec_vecs_[0], 0.5, rec_vecs_[1], sp_[7]);

   sl_[8] = "S";
   sp_[8].Set(0.5, rec_vecs_[1]);

   sl_[9] = "Z";
   sp_[9].Set(0.5, rec_vecs_[2]);

   // Define Paths
   path_[0][ 0] = 0;
   path_[0][ 1] = 3;
   path_[0][ 2] = 8;
   path_[0][ 3] = 6;
   path_[0][ 4] = 2;
   path_[0][ 5] = 9;
   path_[0][ 6] = 0;
   path_[0][ 7] = 7;
   path_[0][ 8] = 5;
   path_[0][ 9] = 4;
   path_[0][10] = 1;
   path_[0][11] = 7;

   path_[1][ 0] = 9;
   path_[1][ 1] = 1;

   // Set Intermediate Symmetry Points
   this->SetIntermediatePoints();

   // Set Intermediate Symmetry Point Labels
   il_[0][ 0] = "GammaX"; // Gamma -> X
   il_[0][ 1] = "XS";     // X     -> S
   il_[0][ 2] = "SR";     // S     -> R
   il_[0][ 3] = "RA";     // R     -> A
   il_[0][ 4] = "AZ";     // A     -> Z
   il_[0][ 5] = "ZGamma"; // Z     -> Gamma
   il_[0][ 6] = "GammaY"; // Gamma -> Y
   il_[0][ 7] = "YX1";    // Y     -> X1
   il_[0][ 8] = "X1A1";   // X1    -> A1
   il_[0][ 9] = "A1T";    // A1    -> T
   il_[0][10] = "TY";     // T     -> Y

   il_[1] [0] = "ZT";     // Z     -> T

   // Set Mesh data
   for (int i=0; i<63; i++) { ws_vert_[i] = 0.0; }
   for (int i=0; i<3; i++)
   {
      int o = 21*i;
      ws_vert_[ 0 + o] =  0.0;
      ws_vert_[ 1 + o] = -0.75 * a_ * a_ / b_ + 0.25 * b_;

      ws_vert_[ 3 + o] =  0.0;
      ws_vert_[ 4 + o] = -0.25 * (a_ * a_ + b_ * b_) / b_;

      ws_vert_[ 6 + o] =  0.5 * a_;
      ws_vert_[ 7 + o] = -0.25 * (b_ * b_ - a_ * a_) / b_;

      ws_vert_[ 9 + o] =  0.5 * a_;
      ws_vert_[10 + o] =  0.25 * (b_ * b_ - a_ * a_) / b_;

      ws_vert_[12 + o] =  0.0;
      ws_vert_[13 + o] =  0.25 * (a_ * a_ + b_ * b_) / b_;

      ws_vert_[15 + o] = -0.5 * a_;
      ws_vert_[16 + o] =  0.25 * (b_ * b_ - a_ * a_) / b_;

      ws_vert_[18 + o] = -0.5 * a_;
      ws_vert_[19 + o] = -0.25 * (b_ * b_ - a_ * a_) / b_;

      double z = 0.5 * c_ * (-1.0 + i);
      for (int j=0; j<7; j++) { ws_vert_[3 * j + 2 + o] = z; }
   }

   for (int i=0; i<2; i++)
   {
      int o = 24*i;
      int v =  7*i;

      for (int j=0; j<3; j++)
      {
         ws_e2v_[8*j+o+0] = 0 + v;
         ws_e2v_[8*j+o+1] = 2 * j + 1 + v;
         ws_e2v_[8*j+o+2] = 2 * j + 2 + v;
         ws_e2v_[8*j+o+3] = ((2 * j + 2) % 6) + 1 + v;

         for (int k=0; k<4; k++) { ws_e2v_[8*j+o+4+k] = ws_e2v_[8*j+o+k] + 7; }
      }
   }

   for (int i=0; i<6; i++) { ws_elem_att_[i] = 1; }

   ws_be2v_[0] = 0; ws_be2v_[1] = 3; ws_be2v_[ 2] = 2; ws_be2v_[ 3] = 1;
   ws_be2v_[4] = 0; ws_be2v_[5] = 5; ws_be2v_[ 6] = 4; ws_be2v_[ 7] = 3;
   ws_be2v_[8] = 0; ws_be2v_[9] = 1; ws_be2v_[10] = 6; ws_be2v_[11] = 5;

   for (int i=0; i<2; i++)
   {
      for (int j=0; j<6; j++)
      {
         ws_be2v_[24*i+4*j+0+12] = 7*i+j+1;
         ws_be2v_[24*i+4*j+1+12] = 7*i+((j+1)%6)+1;
         ws_be2v_[24*i+4*j+2+12] = 7*i+((j+1)%6)+1+7;
         ws_be2v_[24*i+4*j+3+12] = 7*i+j+1+7;
      }
   }

   ws_be2v_[60] = 14; ws_be2v_[61] = 15; ws_be2v_[62] = 16; ws_be2v_[63] = 17;
   ws_be2v_[64] = 14; ws_be2v_[65] = 17; ws_be2v_[66] = 18; ws_be2v_[67] = 19;
   ws_be2v_[68] = 14; ws_be2v_[69] = 19; ws_be2v_[70] = 20; ws_be2v_[71] = 15;

   for (int i=0; i<18; i++) { ws_belem_att_[i] = 1; }
}

bool
BaseCenteredOrthorhombicLattice::MapToFundamentalDomain(const Vector & pt,
                                                        Vector & ipt) const
{
   bool map = false;
   /*
   ipt = pt;
   if ( ipt[2] < 0.0 )
   {
      ipt[2] *= -1.0;
      map = true;
   }
   if ( ipt[1] > 0.0 )
   {
      ipt[1] *= -1.0;
      map = true;
   }
   if ( ipt[0] < 0.0 )
   {
      ipt[0] *= -1.0;
      map = true;
   }
   if ( ipt[1] < -sqrt(3.0) * ipt[0] )
   {
      double d0 = ipt[0], d1 = ipt[1];
      ipt[0] = -0.5 * (d0 + sqrt(3.0) * d1);
      ipt[1] =  0.5 * (d1 - sqrt(3.0) * d0);
      map = true;
   }
   if ( ipt[0] > -sqrt(3.0) * ipt[1] )
   {
      double d0 = ipt[0], d1 = ipt[1];
      ipt[0] =  0.5 * (d0 - sqrt(3.0) * d1);
      ipt[1] = -0.5 * (d1 + sqrt(3.0) * d0);
      map = true;
   }
   */
   return map;
}

const DenseMatrix &
BaseCenteredOrthorhombicLattice::GetTransformation(int ti) const
{
   T_ = 0.0;

   T_(0, 0) = 1.0;
   T_(1, 1) = 1.0;
   T_(2, 2) = 1.0;

   for (int i=0; i<3; i++)
   {
      if ( ti & (int)pow(2, i) )
      {
         T_(i,0) *= -1.0;
         T_(i,1) *= -1.0;
         T_(i,2) *= -1.0;
      }
   }

   return T_;
}

Mesh *
BaseCenteredOrthorhombicLattice::GetWignerSeitzMesh(bool tetMesh) const
{
   Mesh * mesh = new Mesh((double*)ws_vert_, 21,
                          (int*)ws_e2v_, Geometry::CUBE,
                          (int*)ws_elem_att_, 6,
                          (int*)ws_be2v_, Geometry::SQUARE,
                          (int*)ws_belem_att_, 18,
                          3, 3);
   mesh->Finalize();

   return mesh;
}

Mesh *
BaseCenteredOrthorhombicLattice::GetPeriodicWignerSeitzMesh(bool tetMesh) const
{
   Mesh * mesh = this->GetWignerSeitzMesh(tetMesh);

   {
      std::ofstream ofs("orcc.mesh");
      mesh->Print(ofs);
      ofs.close();
   }

   mesh->UniformRefinement();

   Mesh * per_mesh = MakePeriodicMesh(mesh, trn_vecs_);

   delete mesh;

   {
      std::ofstream ofs("per-orcc.mesh");
      per_mesh->Print(ofs);
      ofs.close();
   }

   return per_mesh;
}

HexagonalPrismLattice::HexagonalPrismLattice(double a, double c)
   : BravaisLattice3D(a, a, c, 0.5 * M_PI, 0.5 * M_PI, 2.0 * M_PI / 3.0)
{
   label_ = "HEX";
   type_  = PRIMITIVE_HEXAGONAL_PRISM;

   // Set Lattice Vectors
   lat_vecs_[0][0] =  0.5 * a_;
   lat_vecs_[0][1] = -sqrt(0.75) * a_;
   lat_vecs_[0][2] =  0.0;

   lat_vecs_[1][0] =  0.5 * a_;
   lat_vecs_[1][1] =  sqrt(0.75) * a_;
   lat_vecs_[1][2] =  0.0;

   lat_vecs_[2][0] =  0.0;
   lat_vecs_[2][1] =  0.0;
   lat_vecs_[2][2] =   c_;

   // Set Reciprocal Lattice Vectors
   rec_vecs_[0][0] =  1.0 / a_;
   rec_vecs_[0][1] = -1.0 / ( sqrt(3.0) * a_ );
   rec_vecs_[0][2] =  0.0;

   rec_vecs_[1][0] =  1.0 / a_;
   rec_vecs_[1][1] =  1.0 / ( sqrt(3.0) * a_ );
   rec_vecs_[1][2] =  0.0;

   rec_vecs_[2][0] =  0.0;
   rec_vecs_[2][1] =  0.0;
   rec_vecs_[2][2] =  1.0 / c_;

   // Set the Translation Vectors
   trn_vecs_.resize(4);
   for (unsigned int i=0; i<trn_vecs_.size(); i++)
   {
      trn_vecs_[i].SetSize(3);
   }
   trn_vecs_[0][0] =  1.0 * a;
   trn_vecs_[0][1] =  0.0;
   trn_vecs_[0][2] =  0.0;

   trn_vecs_[1][0] =  0.5 * a;
   trn_vecs_[1][1] =  sqrt(0.75) * a;
   trn_vecs_[1][2] =  0.0;

   trn_vecs_[2][0] =  0.5 * a;
   trn_vecs_[2][1] = -sqrt(0.75) * a;
   trn_vecs_[2][2] =  0.0;

   trn_vecs_[3][0] =  0.0;
   trn_vecs_[3][1] =  0.0;
   trn_vecs_[3][2] =  c;

   // Set the face radii
   face_radii_.resize(4);
   face_radii_[0] = min(0.5 * a_ / sqrt(3.0), 0.5 * c_);
   for (int i=0; i<3; i++) { face_radii_[i] = face_radii_[0]; }
   face_radii_[3] = 0.5 * a_;

   this->SetCellVolumes();

   // Allocate sl_, sp_, and path_ vectors
   this->SetVectorSizes();

   // Set Symmetry Points and Labels
   sl_[0] = "Gamma";
   sp_[0] = 0.0;

   sl_[1] = "A";
   sp_[1].Set(0.5, rec_vecs_[2]);

   sl_[2] = "H";
   add(1.0/3.0, rec_vecs_[0], 1.0/3.0, rec_vecs_[1], sp_[2]);
   sp_[2].Add(0.5, rec_vecs_[2]);

   sl_[3] = "K";
   add(1.0/3.0, rec_vecs_[0], 1.0/3.0, rec_vecs_[1], sp_[3]);

   sl_[4] = "L";
   add(0.5, rec_vecs_[0], 0.5, rec_vecs_[2], sp_[4]);

   sl_[5] = "M";
   sp_[5].Set(0.5, rec_vecs_[0]);

   // Define Paths
   path_[0][0] = 0;
   path_[0][1] = 5;
   path_[0][2] = 3;
   path_[0][3] = 0;
   path_[0][4] = 1;
   path_[0][5] = 4;
   path_[0][6] = 2;
   path_[0][7] = 1;

   path_[1][0] = 4;
   path_[1][1] = 5;

   path_[2][0] = 3;
   path_[2][1] = 2;

   // Set Intermediate Symmetry Points
   this->SetIntermediatePoints();

   // Set Intermediate Symmetry Point Labels
   il_[0][0] = "Sigma";  // Gamma -> M
   il_[0][1] = "MK";     // M     -> K
   il_[0][2] = "GammaK"; // K     -> Gamma
   il_[0][3] = "Delta";  // Gamma -> A
   il_[0][4] = "AL";     // A     -> L
   il_[0][5] = "LH";     // L     -> H
   il_[0][6] = "AH";     // H     -> A
   il_[1][0] = "LM";     // L     -> M
   il_[2][0] = "HK";     // K     -> H

   // Set Mesh data
   fd_vert_[ 0] = 0.0;      fd_vert_[ 1] = 0.0;      fd_vert_[ 2] = 0.0;
   fd_vert_[ 3] = 0.5 * a_; fd_vert_[ 4] = 0.0;      fd_vert_[ 5] = 0.0;
   fd_vert_[ 6] = 0.5 * a_; fd_vert_[ 7] = 0.5 * sqrt(1.0/3.0) * a_;
   fd_vert_[ 8] = 0.0;
   for (int i=0; i<3; i++)
   {
      fd_vert_[3 * i +  9] = fd_vert_[3 * i + 0];
      fd_vert_[3 * i + 10] = fd_vert_[3 * i + 1];
      fd_vert_[3 * i + 11] = 0.5 * c_;
   }

   fd_e2v_[0] = 0; fd_e2v_[1] = 1; fd_e2v_[2] = 2;
   fd_e2v_[3] = 3; fd_e2v_[4] = 4; fd_e2v_[5] = 5;
   fd_elem_att_[0] = 1;

   fd_be2v_[ 0] = 0; fd_be2v_[ 1] = 1; fd_be2v_[ 2] = 4; fd_be2v_[ 3] = 3;
   fd_be2v_[ 4] = 1; fd_be2v_[ 5] = 2; fd_be2v_[ 6] = 5; fd_be2v_[ 7] = 4;
   fd_be2v_[ 8] = 2; fd_be2v_[ 9] = 0; fd_be2v_[10] = 3; fd_be2v_[11] = 5;

   fd_be2v_[12] = 2; fd_be2v_[13] = 1; fd_be2v_[14] = 0; fd_be2v_[15] = -1;
   fd_be2v_[16] = 3; fd_be2v_[17] = 4; fd_be2v_[18] = 5; fd_be2v_[19] = -1;

   fd_belem_att_[0] = 10; fd_belem_att_[1] = 1;
   fd_belem_att_[2] = 10;
   fd_belem_att_[3] = 10; fd_belem_att_[4] = 1;
   /*
   for (int i=0; i<63; i++) { ws_vert_[i] = 0.0; }
   for (int i=0; i<3; i++)
   {
      int o = 21*i;
      ws_vert_[ 0 + o] =  0.0;      ws_vert_[ 1 + o] =  0.0;
      ws_vert_[ 3 + o] =  0.0;      ws_vert_[ 4 + o] = -sqrt(1.0/3.0) * a_;
      ws_vert_[ 6 + o] =  0.5 * a_; ws_vert_[ 7 + o] = -0.5 * sqrt(1.0/3.0) * a_;
      ws_vert_[ 9 + o] =  0.5 * a_; ws_vert_[10 + o] =  0.5 * sqrt(1.0/3.0) * a_;
      ws_vert_[12 + o] =  0.0;      ws_vert_[13 + o] =  sqrt(1.0/3.0) * a_;
      ws_vert_[15 + o] = -0.5 * a_; ws_vert_[16 + o] =  0.5 * sqrt(1.0/3.0) * a_;
      ws_vert_[18 + o] = -0.5 * a_; ws_vert_[19 + o] = -0.5 * sqrt(1.0/3.0) * a_;

      double z = 0.5 * c_ * (-1.0 + i);
      for (int i=0; i<7; i++) { ws_vert_[3 * i + 2 + o] = z; }
   }

   for (int i=0; i<2; i++)
   {
      int o = 24*i;
      int v =  7*i;

      for (int j=0; j<3; j++)
      {
         ws_e2v_[8*j+o+0] = 0 + v;
         ws_e2v_[8*j+o+1] = 2 * j + 1 + v;
         ws_e2v_[8*j+o+2] = 2 * j + 2 + v;
         ws_e2v_[8*j+o+3] = ((2 * j + 2) % 6) + 1 + v;

         for (int k=0; k<4; k++) { ws_e2v_[8*j+o+4+k] = ws_e2v_[8*j+o+k] + 7; }
      }
   }

   for (int i=0; i<6; i++) { ws_elem_att_[i] = 1; }

   ws_be2v_[0] = 0; ws_be2v_[1] = 3; ws_be2v_[ 2] = 2; ws_be2v_[ 3] = 1;
   ws_be2v_[4] = 0; ws_be2v_[5] = 5; ws_be2v_[ 6] = 4; ws_be2v_[ 7] = 3;
   ws_be2v_[8] = 0; ws_be2v_[9] = 1; ws_be2v_[10] = 6; ws_be2v_[11] = 5;

   for (int i=0; i<2; i++)
   {
      for (int j=0; j<6; j++)
      {
         ws_be2v_[24*i+4*j+0+12] = 7*i+j+1;
         ws_be2v_[24*i+4*j+1+12] = 7*i+((j+1)%6)+1;
         ws_be2v_[24*i+4*j+2+12] = 7*i+((j+1)%6)+1+7;
         ws_be2v_[24*i+4*j+3+12] = 7*i+j+1+7;
      }
   }

   ws_be2v_[60] = 14; ws_be2v_[61] = 15; ws_be2v_[62] = 16; ws_be2v_[63] = 17;
   ws_be2v_[64] = 14; ws_be2v_[65] = 17; ws_be2v_[66] = 18; ws_be2v_[67] = 19;
   ws_be2v_[68] = 14; ws_be2v_[69] = 19; ws_be2v_[70] = 20; ws_be2v_[71] = 15;

   for (int i=0; i<18; i++) { ws_belem_att_[i] = 1; }
   */
}

bool
HexagonalPrismLattice::MapToFundamentalDomain(const Vector & pt,
                                              Vector & ipt) const
{
   bool map = false;
   ipt = pt;
   if ( ipt[2] < 0.0 )
   {
      ipt[2] *= -1.0;
      map = true;
   }
   if ( ipt[1] > 0.0 )
   {
      ipt[1] *= -1.0;
      map = true;
   }
   if ( ipt[0] < 0.0 )
   {
      ipt[0] *= -1.0;
      map = true;
   }
   if ( ipt[1] < -sqrt(3.0) * ipt[0] )
   {
      double d0 = ipt[0], d1 = ipt[1];
      ipt[0] = -0.5 * (d0 + sqrt(3.0) * d1);
      ipt[1] =  0.5 * (d1 - sqrt(3.0) * d0);
      map = true;
   }
   if ( ipt[0] > -sqrt(3.0) * ipt[1] )
   {
      double d0 = ipt[0], d1 = ipt[1];
      ipt[0] =  0.5 * (d0 - sqrt(3.0) * d1);
      ipt[1] = -0.5 * (d1 + sqrt(3.0) * d0);
      map = true;
   }
   return map;
}

const DenseMatrix &
HexagonalPrismLattice::GetTransformation(int ti) const
{
   int ir = ti % 3;
   int iq = ti / 3;

   T_ = 0.0;

   T_(2, 2) =  1.0;

   switch (ir)
   {
      case 0:
         T_(0, 0) =  1.0;
         T_(1, 1) =  1.0;
         break;
      case 1:
         T_(0, 0) =  0.5;
         T_(0, 1) =  sqrt(0.75);
         T_(1, 0) =  sqrt(0.75);
         T_(1, 1) = -0.5;
         break;
      case 2:
         T_(0, 0) =  0.5;
         T_(0, 1) = -sqrt(0.75);
         T_(1, 0) =  sqrt(0.75);
         T_(1, 1) =  0.5;
         break;
   }

   for (int i=0; i<3; i++)
   {
      if ( iq & (int)pow(2, i) )
      {
         T_(i,0) *= -1.0;
         T_(i,1) *= -1.0;
         T_(i,2) *= -1.0;
      }
   }

   return T_;
}

Mesh *
HexagonalPrismLattice::GetFundamentalDomainMesh() const
{
   Mesh * mesh = new Mesh(3, 6, 1, 5, 3);
   mesh->AddWedge((const int*)fd_e2v_, fd_elem_att_[0]);
   for (int i=0; i<3; i++)
   {
      mesh->AddBdrQuad((const int*)&fd_be2v_[4*i], fd_belem_att_[i]);
   }
   for (int i=3; i<5; i++)
   {
      mesh->AddBdrTriangle((const int*)&fd_be2v_[4*i], fd_belem_att_[i]);
   }
   for (int i=0; i<6; i++)
   {
      mesh->AddVertex((const double *)&fd_vert_[3*i]);
   }

   mesh->FinalizeMesh(true);

   return mesh;
}
/*
Mesh *
HexagonalPrismLattice::GetWignerSeitzMesh(bool tetMesh) const
{
   Mesh * mesh = new Mesh((double*)ws_vert_, 21,
                          (int*)ws_e2v_, Geometry::CUBE,
                          (int*)ws_elem_att_, 6,
                          (int*)ws_be2v_, Geometry::SQUARE,
                          (int*)ws_belem_att_, 18,
                          3, 3);
   mesh->Finalize();

   return mesh;
}

Mesh *
HexagonalPrismLattice::GetPeriodicWignerSeitzMesh(bool tetMesh) const
{
   Mesh * mesh = this->GetWignerSeitzMesh(tetMesh);

   {
      std::ofstream ofs("hex.mesh");
      mesh->Print(ofs);
      ofs.close();
   }

   mesh->UniformRefinement();

   Mesh * per_mesh = MakePeriodicMesh(mesh, trn_vecs_);

   delete mesh;

   {
      std::ofstream ofs("per-hex.mesh");
      per_mesh->Print(ofs);
      ofs.close();
   }

   return per_mesh;
}
*/
RhombohedralLattice::RhombohedralLattice(double a, double alpha)
   : BravaisLattice3D(a, a, a, alpha, alpha, alpha)
{
   MFEM_ASSERT( 0.0 < alpha_ && alpha_ < 2.0 * M_PI / 3.0 ,
                "Rhombohedral unit cells must have 0 < alpha < 2 pi / 3 !");

   if ( alpha_ < 0.5 * M_PI )
   {
      label_ = "RHL1";
   }
   else
   {
      label_ = "RHL2";
   }
   type_  = PRIMITIVE_RHOMBOHEDRAL;

   double sinha = sin(0.5 * alpha_);
   double cosha = cos(0.5 * alpha_);
   double cosa  = cos(alpha_);

   // Set Lattice Vectors
   lat_vecs_[0][0] =  a_ * cosha;
   lat_vecs_[0][1] = -a_ * sinha;
   lat_vecs_[0][2] =  0.0;

   lat_vecs_[1][0] =  a_ * cosha;
   lat_vecs_[1][1] =  a_ * sinha;
   lat_vecs_[1][2] =  0.0;

   lat_vecs_[2][0] =  a_ * cosa / cosha;
   lat_vecs_[2][1] =  0.0;
   lat_vecs_[2][2] =  a_ * sqrt(1.0 - cosa * cosa / (cosha * cosha));

   // Set Reciprocal Lattice Vectors
   rec_vecs_[0][0] =  0.5 / (a_ * cosha);
   rec_vecs_[0][1] = -0.5 / (a_ * sinha);
   rec_vecs_[0][2] = -0.5 * cosa
                     / (a_ * cosha * sqrt(cosha * cosha - cosa * cosa));

   rec_vecs_[1][0] =  0.5 / (a_ * cosha);
   rec_vecs_[1][1] =  0.5 / (a_ * sinha);
   rec_vecs_[1][2] = -0.5 * cosa
                     / (a_ * cosha * sqrt(cosha * cosha - cosa * cosa));

   rec_vecs_[2][0] =  0.0;
   rec_vecs_[2][1] =  0.0;
   rec_vecs_[2][2] =  1.0 / (a_ * sqrt(1.0 - cosa * cosa / (cosha * cosha)));

   // Set Translation Vectors
   trn_vecs_.resize((alpha_<0.5*M_PI)?6:7);
   for (unsigned int i=0; i<trn_vecs_.size(); i++)
   {
      trn_vecs_[i].SetSize(3);
   }
   if ( alpha_ < 0.5 * M_PI )
   {
      // Wigner-Seitz cell is a rhombic dodecahedron
      double sin3ha = sin(1.5 * alpha_);
      // double sinh  = sin(0.5 * alpha_);
      // double cosh  = cos(0.5 * alpha_);
      double tanha = tan(0.5 * alpha_);
      double secha = 1.0 / cos(0.5 * alpha_);
      // double cosa  = cos(alpha_);
      double csca  = 1.0 / sin(alpha_);
      double seca  = 1.0 / cos(alpha_);

      trn_vecs_[0][0] =  a_ * cosha;
      trn_vecs_[0][1] =  a_ * sinha;
      trn_vecs_[0][2] =  0.0;

      trn_vecs_[1][0] =  a_ * cosha;
      trn_vecs_[1][1] = -a_ * sinha;
      trn_vecs_[1][2] =  0.0;

      trn_vecs_[2][0] =  2.0 * a_ * csca * sinha * sinha * sinha;
      trn_vecs_[2][1] =  a_ * sinha;
      trn_vecs_[2][2] = -a_ * tanha * sqrt(1.0 + 2.0 * cosa);

      trn_vecs_[3][0] =  2.0 * a_ * csca * sinha * sinha * sinha;
      trn_vecs_[3][1] = -a_ * sinha;
      trn_vecs_[3][2] = -a_ * tanha * sqrt(1.0 + 2.0 * cosa);

      trn_vecs_[4][0] =  0.0;
      trn_vecs_[4][1] =  2.0 * a_ * sinha;
      trn_vecs_[4][2] =  0.0;

      trn_vecs_[5][0] =  a_ * cosa * secha;
      trn_vecs_[5][1] =  0.0;
      trn_vecs_[5][2] =  a_ * secha * sin3ha / sqrt(1.0 + 2.0 * cosa);

      // Set the face radii
      face_radii_.resize(6);
      face_radii_[0] = 0.5 * a_ * tanha;
      face_radii_[1] = face_radii_[0];
      face_radii_[2] = 0.5 * a_ * (sin3ha - sinha) *
	min(csca, seca / sqrt(3.0));
      face_radii_[3] = face_radii_[2];
      face_radii_[4] = face_radii_[2];
      face_radii_[5] = face_radii_[0];
   }
   else
   {
      // Wigner-Seitz cell is a truncated octahedron
      // double sinha = sin(0.5 * alpha_);
      // double cosha = cos(0.5 * alpha_);
      double tanha = tan(0.5 * alpha_);
      double secha = 1.0 / cosha;
      double cscha = 1.0 / sinha;
      double cotha = cosha / sinha;
      double sina  = sin(alpha_);
      // double cosa  = cos(alpha_);

      trn_vecs_[0][0] =  a_ * cosha;
      trn_vecs_[0][1] =  a_ * sinha;
      trn_vecs_[0][2] =  0.0;

      trn_vecs_[1][0] =  a_ * cosha;
      trn_vecs_[1][1] = -a_ * sinha;
      trn_vecs_[1][2] =  0.0;

      trn_vecs_[2][0] =  2.0 * a_ * cosha;
      trn_vecs_[2][1] =  0.0;
      trn_vecs_[2][2] =  0.0;

      trn_vecs_[3][0] =  a_ * cosa * secha;
      trn_vecs_[3][1] =  0.0;
      trn_vecs_[3][2] =  a_ * tanha * sqrt(1.0 + 2.0 * cosa);

      trn_vecs_[4][0] =  a_ * (1.0 + 2.0 * cosa) * secha;
      trn_vecs_[4][1] =  0.0;
      trn_vecs_[4][2] =  a_ * tanha * sqrt(1.0 + 2.0 * cosa);

      trn_vecs_[5][0] =  0.5 * a_ * (1.0 + 3.0 * cosa) * secha;
      trn_vecs_[5][1] =  a_ * sinha;
      trn_vecs_[5][2] =  a_ * tanha * sqrt(1.0 + 2.0 * cosa);

      trn_vecs_[6][0] =  0.5 * a_ * (1.0 + 3.0 * cosa) * secha;
      trn_vecs_[6][1] = -a_ * sinha;
      trn_vecs_[6][2] =  a_ * tanha * sqrt(1.0 + 2.0 * cosa);

      // Set the face radii
      face_radii_.resize(7);
      face_radii_[0] = a_ * min(0.5 * cotha,
                                sina * sqrt((1.0 + 2.0 * cosa) /
                                            (2.0*cosa+2.0) ) / (1.0 - cosa));
      face_radii_[1] = face_radii_[0];
      face_radii_[2] = 0.5 * a_ * cscha * min(-cosa,
                                             cosha * sqrt(1.0 + 2.0 * cosa));
      face_radii_[3] = face_radii_[0];
      face_radii_[4] = -0.5 * sqrt(3.0) * a_ * cosa * cscha;
      face_radii_[5] = face_radii_[2];
      face_radii_[6] = face_radii_[2];
   }

   this->SetCellVolumes();

   // Allocate sl_, sp_, and path_ vectors
   this->SetVectorSizes();

   // Set Symmetry Points and Labels
   if ( alpha_ < 0.5 * M_PI )
   {
      double cosa = cos(alpha_);
      double  eta = (1.0 + 4.0 * cosa) / (2.0 + 4.0 * cosa);
      double   nu = 0.75 - 0.5 * eta;

      sl_[0] = "Gamma";
      sp_[0] = 0.0;

      sl_[1] = "P";
      add(eta, rec_vecs_[0], nu, rec_vecs_[1], sp_[1]);
      sp_[1].Add(nu, rec_vecs_[2]);

      sl_[2] = "B";
      add(eta, rec_vecs_[0], 0.5, rec_vecs_[1], sp_[2]);
      sp_[2].Add(1.0-eta, rec_vecs_[2]);

      sl_[3] = "P1";
      add(1.0-nu, rec_vecs_[0], 1.0-nu, rec_vecs_[1], sp_[3]);
      sp_[3].Add(1.0-eta, rec_vecs_[2]);

      sl_[4] = "B1";
      add(0.5, rec_vecs_[0], 1.0-eta, rec_vecs_[1], sp_[4]);
      sp_[4].Add(eta-1.0, rec_vecs_[2]);

      sl_[5] = "P2"; /// Note that this point does not fall on any path
      add(nu, rec_vecs_[0], nu, rec_vecs_[1], sp_[5]);
      sp_[5].Add(eta-1.0, rec_vecs_[2]);

      sl_[6] = "F";
      add(0.5, rec_vecs_[0], 0.5, rec_vecs_[1], sp_[6]);

      sl_[7] = "Q";
      add(1.0-nu, rec_vecs_[0], nu, rec_vecs_[1], sp_[7]);

      sl_[8] = "L";
      sp_[8].Set(0.5, rec_vecs_[0]);

      sl_[9] = "X";
      add(nu, rec_vecs_[0], -nu, rec_vecs_[2], sp_[9]);

      sl_[10] = "L1"; /// Note that this point does not fall on any path
      sp_[10].Set(0.5, rec_vecs_[0]);

      sl_[11] = "Z";
      add(0.5, rec_vecs_[0], 0.5, rec_vecs_[1], sp_[11]);
      sp_[11].Add(0.5, rec_vecs_[2]);
   }
   else
   {
      double tana2 = pow(tan(0.5 * alpha_), 2.0);
      double   eta = 0.5 / tana2;
      double    nu = 0.75 - 0.5 * eta;

      sl_[0] = "Gamma";
      sp_[0] = 0.0;

      sl_[1] = "P1";
      add(nu, rec_vecs_[0], nu-1.0, rec_vecs_[1], sp_[1]);
      sp_[1].Add(nu-1.0, rec_vecs_[2]);

      sl_[2] = "F";
      add(0.5, rec_vecs_[0], -0.5, rec_vecs_[1], sp_[2]);

      sl_[3] = "Q";
      add(eta, rec_vecs_[0], eta, rec_vecs_[1], sp_[3]);
      sp_[3].Add(eta, rec_vecs_[2]);

      sl_[4] = "L";
      sp_[4].Set(0.5, rec_vecs_[0]);

      sl_[5] = "Q1";
      add(1.0-eta, rec_vecs_[0], -eta, rec_vecs_[1], sp_[5]);
      sp_[5].Add(-eta, rec_vecs_[2]);

      sl_[6] = "P";
      add(1.0-nu, rec_vecs_[0], -nu, rec_vecs_[1], sp_[6]);
      sp_[6].Add(1.0-nu, rec_vecs_[2]);

      sl_[7] = "Z";
      add(0.5, rec_vecs_[0], -0.5, rec_vecs_[1], sp_[7]);
      sp_[7].Add(0.5, rec_vecs_[2]);
   }

   for (unsigned int i=0; i<sl_.size(); i++)
   {
      si_[sl_[i]] = i;
   }

   // Define Paths
   if ( alpha_ < 0.5 * M_PI )
   {
      path_[0][0] =  0;
      path_[0][1] =  8;
      path_[0][2] =  4;

      path_[1][0] =  2;
      path_[1][1] = 11;
      path_[1][2] =  0;
      path_[1][3] =  9;

      path_[2][0] =  7;
      path_[2][1] =  6;
      path_[2][2] =  3;
      path_[2][3] = 11;

      path_[3][0] =  8;
      path_[3][1] =  1;
   }
   else
   {
      path_[0][0] =  0;
      path_[0][1] =  6;
      path_[0][2] =  7;
      path_[0][3] =  3;
      path_[0][4] =  0;
      path_[0][5] =  2;
      path_[0][6] =  1;
      path_[0][7] =  5;
      path_[0][8] =  4;
      path_[0][9] =  7;
   }

   // Set Intermediate Symmetry Points
   this->SetIntermediatePoints();

   // Set Intermediate Symmetry Point Labels
   if ( alpha_ < 0.5 * M_PI )
   {
      il_[0][0] = "GammaL"; // Gamma -> L
      il_[0][1] = "LB1";    // P     -> B1

      il_[1][0] = "BZ";     // B     -> Z
      il_[1][1] = "ZGamma"; // Z     -> Gamma
      il_[1][2] = "GammaX"; // Gamma -> X

      il_[2][0] = "QF";     // Q     -> F
      il_[2][1] = "FP1";    // F     -> P1
      il_[2][2] = "P1Z";    // P1    -> Z

      il_[3][0] = "LP";     // L     -> P
   }
   else
   {
      il_[0][0] = "GammaP"; // Gamma -> P
      il_[0][1] = "PZ";     // P     -> Z
      il_[0][2] = "ZQ";     // Z     -> Q
      il_[0][3] = "QGamma"; // Q     -> Gamma
      il_[0][4] = "GammaF"; // Gamma -> F
      il_[0][5] = "FP1";    // F     -> P1
      il_[0][6] = "P1Q1";   // P1    -> Q1
      il_[0][7] = "Q1L";    // Q1    -> L
      il_[0][8] = "LZ";     // L     -> Z
   }

   // Set Mesh data
   if ( alpha_ < 0.5 * M_PI )
   {
      this->createRhombicDodecahedron();
   }
   else
   {
      this->createTruncatedOctahedron();
   }
}

void
RhombohedralLattice::createRhombicDodecahedron()
{
   cout << "Setting up data for rhombic dodecahedron" << endl;

   // Set Mesh data
   double cosa = cos(alpha_);
   double csca = 1.0 / sin(alpha_);
   double sinh = sin(0.5 * alpha_);
   double tanh = tan(0.5 * alpha_);
   double sech = 1.0 / cos(0.5 * alpha_);

   for (int i=0; i<114; i++) { ws_vert_[i] = 0.0; }

   ws_vert_[ 0] =  0.5 * a_ * (1.0 - 2.0 * cosa) * sech;
   ws_vert_[ 2] =  0.5 * a_ * (1.0 - 2.0 * cosa) * tanh
                   / sqrt(1.0 + 2.0 * cosa);

   ws_vert_[ 3] = -0.5 * a_ * cosa * sech;
   ws_vert_[ 4] = -a_ * sinh;
   ws_vert_[ 5] = -0.5 * a_ * sqrt(1.0 + 2.0 * cosa) * tanh;

   ws_vert_[ 6] =  0.5 * a_ * cosa * sech;
   ws_vert_[ 7] =  a_ * sinh;
   ws_vert_[ 8] =  0.5 * a_ * sqrt(1.0 + 2.0 * cosa) * tanh;

   ws_vert_[ 9] =  0.5 * a_ * cosa * sech;
   ws_vert_[10] = -a_ * sinh;
   ws_vert_[11] =  0.5 * a_ * sqrt(1.0 + 2.0 * cosa) * tanh;

   ws_vert_[12] = -0.5 * a_ * cosa * sech;
   ws_vert_[13] =  a_ * sinh;
   ws_vert_[14] = -0.5 * a_ * sqrt(1.0 + 2.0 * cosa) * tanh;

   ws_vert_[15] =  a_ * csca * sinh;
   ws_vert_[17] = -0.5 * a_ * sqrt(1.0 + 2.0 * cosa) * tanh;

   ws_vert_[18] = -0.5 * a_ * sech;
   ws_vert_[20] =  0.5 * a_ * sqrt(1.0 + 2.0 * cosa) * tanh;

   ws_vert_[21] =  0.5 * a_ * cosa * sech;
   ws_vert_[22] = -a_ * sinh;
   ws_vert_[23] = -0.5 * a_ * tanh / sqrt(1.0 + 2.0 * cosa);

   ws_vert_[24] = -0.5 * a_ * cosa * sech;
   ws_vert_[25] = -a_ * sinh;
   ws_vert_[26] =  0.5 * a_ * tanh / sqrt(1.0 + 2.0 * cosa);

   ws_vert_[27] =  0.5 * a_ * (1.0 - 2.0 * cosa) * sech;
   ws_vert_[29] = -0.5 * a_ * (1.0 + 4.0 * cosa) * tanh
                  / sqrt(1.0 + 2.0 * cosa);

   ws_vert_[30] = -0.5 * a_ * sech;
   ws_vert_[32] = -0.5 * a_ * tanh / sqrt(1.0 + 2.0 * cosa);

   ws_vert_[33] =  a_ * csca * sinh;
   ws_vert_[35] =  0.5 * a_ * tanh / sqrt(1.0 + 2.0 * cosa);

   ws_vert_[36] = -0.5 * a_ * (1.0 - 2.0 * cosa) * sech;
   ws_vert_[38] =  0.5 * a_ * (1.0 + 4.0 * cosa) * tanh
                   / sqrt(1.0 + 2.0 * cosa);

   ws_vert_[39] =  0.5 * a_ * cosa * sech;
   ws_vert_[40] =  a_ * sinh;
   ws_vert_[41] = -0.5 * a_ * tanh / sqrt(1.0 + 2.0 * cosa);

   ws_vert_[42] = -0.5 * a_ * cosa * sech;
   ws_vert_[43] =  a_ * sinh;
   ws_vert_[44] =  0.5 * a_ * tanh / sqrt(1.0 + 2.0 * cosa);

   ws_e2v_[ 0] =  0; ws_e2v_[ 1] =  9; ws_e2v_[ 2] =  5; ws_e2v_[ 3] = 11;
   ws_e2v_[ 4] =  8; ws_e2v_[ 5] =  1; ws_e2v_[ 6] =  7; ws_e2v_[ 7] =  3;

   ws_e2v_[ 8] =  0; ws_e2v_[ 9] = 11; ws_e2v_[10] =  5; ws_e2v_[11] =  9;
   ws_e2v_[12] = 14; ws_e2v_[13] =  2; ws_e2v_[14] = 13; ws_e2v_[15] =  4;

   ws_e2v_[16] =  0; ws_e2v_[17] =  8; ws_e2v_[18] =  6; ws_e2v_[19] = 14;
   ws_e2v_[20] =  9; ws_e2v_[21] =  1; ws_e2v_[22] = 10; ws_e2v_[23] =  4;

   ws_e2v_[24] =  0; ws_e2v_[25] = 14; ws_e2v_[26] =  6; ws_e2v_[27] =  8;
   ws_e2v_[28] = 11; ws_e2v_[29] =  2; ws_e2v_[30] = 12; ws_e2v_[31] =  3;

   for (int i=0; i<4; i++) { ws_elem_att_[i] = 1; }

   ws_be2v_[ 0] =  8; ws_be2v_[ 1] =  1; ws_be2v_[ 2] =  7; ws_be2v_[ 3] =  3;
   ws_be2v_[ 4] = 14; ws_be2v_[ 5] =  2; ws_be2v_[ 6] = 13; ws_be2v_[ 7] =  4;
   ws_be2v_[ 8] =  9; ws_be2v_[ 9] =  1; ws_be2v_[10] = 10; ws_be2v_[11] =  4;
   ws_be2v_[12] = 11; ws_be2v_[13] =  2; ws_be2v_[14] = 12; ws_be2v_[15] =  3;

   ws_be2v_[16] =  5; ws_be2v_[17] =  7; ws_be2v_[18] =  1; ws_be2v_[19] =  9;
   ws_be2v_[20] =  5; ws_be2v_[21] = 13; ws_be2v_[22] =  2; ws_be2v_[23] = 11;
   ws_be2v_[24] =  5; ws_be2v_[25] = 11; ws_be2v_[26] =  3; ws_be2v_[27] =  7;
   ws_be2v_[28] =  5; ws_be2v_[29] =  9; ws_be2v_[30] =  4; ws_be2v_[31] = 13;

   ws_be2v_[32] =  6; ws_be2v_[33] = 10; ws_be2v_[34] =  1; ws_be2v_[35] =  8;
   ws_be2v_[36] =  6; ws_be2v_[37] = 12; ws_be2v_[38] =  2; ws_be2v_[39] = 14;
   ws_be2v_[40] =  6; ws_be2v_[41] =  8; ws_be2v_[42] =  3; ws_be2v_[43] = 12;
   ws_be2v_[44] =  6; ws_be2v_[45] = 14; ws_be2v_[46] =  4; ws_be2v_[47] = 10;

   for (int i=48; i<120; i++) { ws_be2v_[i] = -1; }

   for (int i= 0; i<12; i++) { ws_belem_att_[i] =  1; }
   for (int i=12; i<30; i++) { ws_belem_att_[i] = -1; }
}

void
RhombohedralLattice::createTruncatedOctahedron()
{
   cout << "Setting up data for truncated octahedron" << endl;

   // Set Mesh data
   for (int i=0; i<114; i++) { ws_vert_[i] = 0.0; }

   double cosh  = cos(0.5 * alpha_);
   double tanh  = tan(0.5 * alpha_);
   double sech  = 1.0 / cos(0.5 * alpha_);
   double csch  = 1.0 / sin(0.5 * alpha_);
   double cosa  = cos(alpha_);

   ws_vert_[  0] = -a_ * cosh;
   ws_vert_[  1] = -0.50 * a_ * cosa * csch;
   ws_vert_[  2] =  0.50 * a_ * sqrt(1.0 + 2.0 * cosa) / tanh;

   ws_vert_[  3] = -a_ * cosh;
   ws_vert_[  4] =  0.50 * a_ * cosa * csch;
   ws_vert_[  5] =  0.50 * a_ * sqrt(1.0 + 2.0 * cosa) / tanh;

   ws_vert_[  6] = -a_ * cosh;
   ws_vert_[  7] =  0.50 * a_ * cosa * csch;
   ws_vert_[  8] = -0.50 * a_ * sqrt(1.0 + 2.0 * cosa) / tanh;

   ws_vert_[  9] = -a_ * cosh;
   ws_vert_[ 10] = -0.50 * a_ * cosa * csch;
   ws_vert_[ 11] = -0.50 * a_ * sqrt(1.0 + 2.0 * cosa) / tanh;

   ws_vert_[ 12] =  a_ * cosh;
   ws_vert_[ 13] = -0.50 * a_ * cosa * csch;
   ws_vert_[ 14] =  0.50 * a_ * sqrt(1.0 + 2.0 * cosa) / tanh;

   ws_vert_[ 15] =  a_ * cosh;
   ws_vert_[ 16] =  0.50 * a_ * cosa * csch;
   ws_vert_[ 17] =  0.50 * a_ * sqrt(1.0 + 2.0 * cosa) / tanh;

   ws_vert_[ 18] =  a_ * cosh;
   ws_vert_[ 19] =  0.50 * a_ * cosa * csch;
   ws_vert_[ 20] = -0.50 * a_ * sqrt(1.0 + 2.0 * cosa) / tanh;

   ws_vert_[ 21] =  a_ * cosh;
   ws_vert_[ 22] = -0.50 * a_ * cosa * csch;
   ws_vert_[ 23] = -0.50 * a_ * sqrt(1.0 + 2.0 * cosa) / tanh;

   ws_vert_[ 24] = -0.50 * a_ * cosa * sech;
   ws_vert_[ 25] = -a_ * cosh * cosh * csch;
   ws_vert_[ 26] = -0.50 * a_ * sqrt(1.0 + 2.0 * cosa) * tanh;

   ws_vert_[ 28] = -0.50 * a_ * csch;
   ws_vert_[ 29] = -0.50 * a_ * sqrt(1.0 + 2.0 * cosa) / tanh;

   ws_vert_[ 30] = -0.50 * a_ * (1.0 + 2.0 * cosa) * sech;
   ws_vert_[ 31] =  a_ * cosa * csch;
   ws_vert_[ 32] = -0.50 * a_ * sqrt(1.0 + 2.0 * cosa) * tanh;

   ws_vert_[ 33] = -0.50 * a_ * (1.0 + 3.0 * cosa) * sech;
   ws_vert_[ 34] =  0.50 * a_ * cosa * csch;
   ws_vert_[ 35] = -0.25 * a_ * (1.0 - 3.0 * cosa) * sqrt(1.0 + 2.0 * cosa)
                   * sech * csch;

   ws_vert_[ 36] =  0.50 * a_ * (1.0 + 2.0 * cosa) * sech;
   ws_vert_[ 37] = -a_ * cosa * csch;
   ws_vert_[ 38] =  0.50 * a_ * sqrt(1.0 + 2.0 * cosa) * tanh;

   ws_vert_[ 39] =  0.50 * a_ * (1.0 + 3.0 * cosa) * sech;
   ws_vert_[ 40] = -0.50 * a_ * cosa * csch;
   ws_vert_[ 41] =  0.25 * a_ * (1.0 - 3.0 * cosa) * sqrt(1.0 + 2.0 * cosa)
                    * sech * csch;

   ws_vert_[ 42] =  0.50 * a_ * cosa * sech;
   ws_vert_[ 43] =  a_ * cosh * cosh * csch;
   ws_vert_[ 44] =  0.50 * a_ * sqrt(1.0 + 2.0 * cosa) * tanh;

   ws_vert_[ 46] =  0.50 * a_ * csch;
   ws_vert_[ 47] =  0.50 * a_ * sqrt(1.0 + 2.0 * cosa) / tanh;

   ws_vert_[ 48] = -0.50 * a_ * cosa * sech;
   ws_vert_[ 49] =  a_ * cosh * cosh * csch;
   ws_vert_[ 50] = -0.50 * a_ * sqrt(1.0 + 2.0 * cosa) * tanh;

   ws_vert_[ 52] =  0.50 * a_ * csch;
   ws_vert_[ 53] = -0.50 * a_ * sqrt(1.0 + 2.0 * cosa) / tanh;

   ws_vert_[ 54] = -0.50 * a_ * (1.0 + 2.0 * cosa) * sech;
   ws_vert_[ 55] = -a_ * cosa * csch;
   ws_vert_[ 56] = -0.50 * a_ * sqrt(1.0 + 2.0 * cosa) * tanh;

   ws_vert_[ 57] = -0.50 * a_ * (1.0 + 3.0 * cosa) * sech;
   ws_vert_[ 58] = -0.50 * a_ * cosa * csch;
   ws_vert_[ 59] = -0.25 * a_ * (1.0 - 3.0 * cosa) * sqrt(1.0 + 2.0 * cosa)
                   * sech * csch;

   ws_vert_[ 60] =  0.50 * a_ * (1.0 + 2.0 * cosa) * sech;
   ws_vert_[ 61] =  a_ * cosa * csch;
   ws_vert_[ 62] =  0.50 * a_ * sqrt(1.0 + 2.0 * cosa) * tanh;

   ws_vert_[ 63] =  0.50 * a_ * (1.0 + 3.0 * cosa) * sech;
   ws_vert_[ 64] =  0.50 * a_ * cosa * csch;
   ws_vert_[ 65] =  0.25 * a_ * (1.0 - 3.0 * cosa) * sqrt(1.0 + 2.0 * cosa)
                    * sech * csch;

   ws_vert_[ 66] =  0.50 * a_ * cosa * sech;
   ws_vert_[ 67] = -a_ * cosh * cosh * csch;
   ws_vert_[ 68] =  0.50 * a_ * sqrt(1.0 + 2.0 * cosa) * tanh;

   ws_vert_[ 70] = -0.50 * a_ * csch;
   ws_vert_[ 71] =  0.50 * a_ * sqrt(1.0 + 2.0 * cosa) / tanh;

   ws_vert_[ 72] = -0.50 * a_ * (1.0 + 2.0 * cosa) * sech;
   ws_vert_[ 74] = -0.50 * a_ * sqrt(1.0 + 2.0 * cosa) * tanh;

   ws_vert_[ 75] = -0.50 * a_ * (1.0 + 2.0 * cosa) * sech;
   ws_vert_[ 76] =  a_ * cosa * csch;
   ws_vert_[ 77] =  0.25 * a_ * (1.0 + 3.0 * cosa) * sqrt(1.0 + 2.0 * cosa)
                    * sech * csch;

   ws_vert_[ 78] = -0.50 * a_ * (1.0 + 2.0 * cosa) * sech;
   ws_vert_[ 79] = -a_ * cosa * csch;
   ws_vert_[ 80] =  0.25 * a_ * (1.0 + 3.0 * cosa) * sqrt(1.0 + 2.0 * cosa)
                    * sech * csch;

   ws_vert_[ 81] =  0.50 * a_ * cosa * sech;
   ws_vert_[ 82] =  0.50 * a_ * (1.0 + 3.0 * cosa) * csch;
   ws_vert_[ 83] =  0.50 * a_ * sqrt(1.0 + 2.0 * cosa) * tanh;

   ws_vert_[ 84] = -0.50 * a_ * cosa * sech;
   ws_vert_[ 85] =  0.50 * a_ * (1.0 + 3.0 * cosa) * csch;
   ws_vert_[ 86] = -0.50 * a_ * sqrt(1.0 + 2.0 * cosa) * tanh;

   ws_vert_[ 87] = -0.50 * a_ * cosa * sech;
   ws_vert_[ 88] = -a_ * cosh * cosh * csch;
   ws_vert_[ 89] =  0.25 * a_ * (1.0 + 3.0 * cosa) * sqrt(1.0 + 2.0 * cosa)
                    * sech * csch;

   ws_vert_[ 90] = -0.50 * a_ * cosa * sech;
   ws_vert_[ 91] =  a_ * cosh * cosh * csch;
   ws_vert_[ 92] =  0.25 * a_ * (1.0 + 3.0 * cosa) * sqrt(1.0 + 2.0 * cosa)
                    * sech * csch;

   ws_vert_[ 93] =  0.50 * a_ * (1.0 + 2.0 * cosa) * sech;
   ws_vert_[ 95] =  0.50 * a_ * sqrt(1.0 + 2.0 * cosa) * tanh;

   ws_vert_[ 96] = -0.50 * a_ * (1.0 + 2.0 * cosa) * sech;
   ws_vert_[ 98] =  0.25 * a_ * (1.0 + 3.0 * cosa) * sqrt(1.0 + 2.0 * cosa)
                    * sech * csch;

   ws_vert_[ 99] = -0.50 * a_ * cosa * sech;
   ws_vert_[100] =  0.50 * a_ * (1.0 + 3.0 * cosa) * csch;
   ws_vert_[101] =  0.25 * a_ * (1.0 + 3.0 * cosa) * sqrt(1.0 + 2.0 * cosa)
                    * sech * csch;

   ws_vert_[102] = -0.50 * a_ * (1.0 + 3.0 * cosa) * sech;
   ws_vert_[103] =  0.50 * a_ * cosa * csch;
   ws_vert_[104] =  0.25 * a_ * (1.0 + 5.0 * cosa) * sqrt(1.0 + 2.0 * cosa)
                    * sech * csch;

   ws_vert_[106] =  0.50 * a_ * (1.0 + 2.0 * cosa) * csch;
   ws_vert_[107] =  0.50 * a_ * sqrt(1.0 + 2.0 * cosa) / tanh;

   ws_vert_[109] =  0.50 * a_ * (1.0 + 2.0 * cosa) * csch;
   ws_vert_[110] = -0.50 * a_ * sqrt(1.0 + 2.0 * cosa) / tanh;

   ws_vert_[112] = -0.50 * a_ * (1.0 + 2.0 * cosa) * csch;
   ws_vert_[113] =  0.50 * a_ * sqrt(1.0 + 2.0 * cosa) / tanh;

   ws_e2v_[  0] =  0; ws_e2v_[  1] =  1; ws_e2v_[  2] =  2; ws_e2v_[  3] =  3;
   ws_e2v_[  4] = 26; ws_e2v_[  5] = 32; ws_e2v_[  6] = 24; ws_e2v_[  7] = 18;

   ws_e2v_[  8] = 26; ws_e2v_[  9] = 32; ws_e2v_[ 10] = 24; ws_e2v_[ 11] = 18;
   ws_e2v_[ 12] = 15; ws_e2v_[ 13] = 35; ws_e2v_[ 14] = 36; ws_e2v_[ 15] = 17;

   ws_e2v_[ 16] = 15; ws_e2v_[ 17] = 35; ws_e2v_[ 18] = 36; ws_e2v_[ 19] = 17;
   ws_e2v_[ 20] = 30; ws_e2v_[ 21] = 33; ws_e2v_[ 22] = 28; ws_e2v_[ 23] = 16;

   ws_e2v_[ 24] = 30; ws_e2v_[ 25] = 33; ws_e2v_[ 26] = 28; ws_e2v_[ 27] = 16;
   ws_e2v_[ 28] =  4; ws_e2v_[ 29] =  5; ws_e2v_[ 30] =  6; ws_e2v_[ 31] =  7;

   ws_e2v_[ 32] =  9; ws_e2v_[ 33] =  8; ws_e2v_[ 34] = 11; ws_e2v_[ 35] = 10;
   ws_e2v_[ 36] = 23; ws_e2v_[ 37] = 29; ws_e2v_[ 38] = 34; ws_e2v_[ 39] = 25;

   ws_e2v_[ 40] = 23; ws_e2v_[ 41] = 29; ws_e2v_[ 42] = 34; ws_e2v_[ 43] = 25;
   ws_e2v_[ 44] = 22; ws_e2v_[ 45] = 37; ws_e2v_[ 46] = 32; ws_e2v_[ 47] =  1;

   ws_e2v_[ 48] = 22; ws_e2v_[ 49] = 37; ws_e2v_[ 50] = 32; ws_e2v_[ 51] =  1;
   ws_e2v_[ 52] = 21; ws_e2v_[ 53] = 31; ws_e2v_[ 54] = 35; ws_e2v_[ 55] = 27;

   ws_e2v_[ 56] = 21; ws_e2v_[ 57] = 31; ws_e2v_[ 58] = 35; ws_e2v_[ 59] = 27;
   ws_e2v_[ 60] = 13; ws_e2v_[ 61] = 12; ws_e2v_[ 62] = 15; ws_e2v_[ 63] = 14;

   ws_e2v_[ 64] = 16; ws_e2v_[ 65] = 17; ws_e2v_[ 66] = 18; ws_e2v_[ 67] = 19;
   ws_e2v_[ 68] = 28; ws_e2v_[ 69] = 36; ws_e2v_[ 70] = 24; ws_e2v_[ 71] = 11;

   ws_e2v_[ 72] = 28; ws_e2v_[ 73] = 36; ws_e2v_[ 74] = 24; ws_e2v_[ 75] = 11;
   ws_e2v_[ 76] = 33; ws_e2v_[ 77] = 35; ws_e2v_[ 78] = 32; ws_e2v_[ 79] = 34;

   ws_e2v_[ 80] = 33; ws_e2v_[ 81] = 35; ws_e2v_[ 82] = 32; ws_e2v_[ 83] = 34;
   ws_e2v_[ 84] =  5; ws_e2v_[ 85] = 31; ws_e2v_[ 86] = 37; ws_e2v_[ 87] = 29;

   ws_e2v_[ 88] =  5; ws_e2v_[ 89] = 31; ws_e2v_[ 90] = 37; ws_e2v_[ 91] = 29;
   ws_e2v_[ 92] = 20; ws_e2v_[ 93] = 21; ws_e2v_[ 94] = 22; ws_e2v_[ 95] = 23;

   ws_e2v_[ 96] = 24; ws_e2v_[ 97] = 11; ws_e2v_[ 98] = 34; ws_e2v_[ 99] = 32;
   ws_e2v_[100] =  2; ws_e2v_[101] = 10; ws_e2v_[102] = 25; ws_e2v_[103] =  1;

   ws_e2v_[104] = 11; ws_e2v_[105] = 28; ws_e2v_[106] = 33; ws_e2v_[107] = 34;
   ws_e2v_[108] =  8; ws_e2v_[109] =  6; ws_e2v_[110] =  5; ws_e2v_[111] = 29;

   ws_e2v_[112] = 30; ws_e2v_[113] = 15; ws_e2v_[114] = 35; ws_e2v_[115] = 33;
   ws_e2v_[116] =  4; ws_e2v_[117] = 12; ws_e2v_[118] = 31; ws_e2v_[119] =  5;

   ws_e2v_[120] = 15; ws_e2v_[121] = 26; ws_e2v_[122] = 32; ws_e2v_[123] = 35;
   ws_e2v_[124] = 14; ws_e2v_[125] =  0; ws_e2v_[126] =  1; ws_e2v_[127] = 27;

   for (int i=0; i<16; i++) { ws_elem_att_[i] = 1; }

   ws_be2v_[ 0] =  3; ws_be2v_[ 1] =  2; ws_be2v_[ 2] =  1; ws_be2v_[ 3] =  0;
   ws_be2v_[ 4] =  2; ws_be2v_[ 5] =  3; ws_be2v_[ 6] = 18; ws_be2v_[ 7] = 24;
   ws_be2v_[ 8] =  3; ws_be2v_[ 9] =  0; ws_be2v_[10] = 26; ws_be2v_[11] = 18;
   ws_be2v_[12] =  4; ws_be2v_[13] =  5; ws_be2v_[14] =  6; ws_be2v_[15] =  7;
   ws_be2v_[16] = 16; ws_be2v_[17] = 30; ws_be2v_[18] =  4; ws_be2v_[19] =  7;
   ws_be2v_[20] = 16; ws_be2v_[21] =  7; ws_be2v_[22] =  6; ws_be2v_[23] = 28;
   ws_be2v_[24] = 17; ws_be2v_[25] = 15; ws_be2v_[26] = 30; ws_be2v_[27] = 16;
   ws_be2v_[28] = 15; ws_be2v_[29] = 17; ws_be2v_[30] = 18; ws_be2v_[31] = 26;
   ws_be2v_[32] =  9; ws_be2v_[33] = 10; ws_be2v_[34] = 11; ws_be2v_[35] =  8;
   ws_be2v_[36] =  9; ws_be2v_[37] =  8; ws_be2v_[38] = 29; ws_be2v_[39] = 23;
   ws_be2v_[40] =  9; ws_be2v_[41] = 23; ws_be2v_[42] = 25; ws_be2v_[43] = 10;
   ws_be2v_[44] =  1; ws_be2v_[45] = 25; ws_be2v_[46] = 23; ws_be2v_[47] = 22;
   ws_be2v_[48] =  1; ws_be2v_[49] = 22; ws_be2v_[50] = 21; ws_be2v_[51] = 27;
   ws_be2v_[52] = 13; ws_be2v_[53] = 12; ws_be2v_[54] = 15; ws_be2v_[55] = 14;
   ws_be2v_[56] = 21; ws_be2v_[57] = 31; ws_be2v_[58] = 12; ws_be2v_[59] = 13;
   ws_be2v_[60] = 21; ws_be2v_[61] = 13; ws_be2v_[62] = 14; ws_be2v_[63] = 27;
   ws_be2v_[64] = 16; ws_be2v_[65] = 19; ws_be2v_[66] = 18; ws_be2v_[67] = 17;
   ws_be2v_[68] = 11; ws_be2v_[69] = 24; ws_be2v_[70] = 18; ws_be2v_[71] = 19;
   ws_be2v_[72] = 11; ws_be2v_[73] = 19; ws_be2v_[74] = 16; ws_be2v_[75] = 28;
   ws_be2v_[76] = 20; ws_be2v_[77] = 21; ws_be2v_[78] = 22; ws_be2v_[79] = 23;
   ws_be2v_[80] =  5; ws_be2v_[81] = 31; ws_be2v_[82] = 21; ws_be2v_[83] = 20;
   ws_be2v_[84] =  5; ws_be2v_[85] = 20; ws_be2v_[86] = 23; ws_be2v_[87] = 29;
   ws_be2v_[88] =  2; ws_be2v_[89] = 10; ws_be2v_[90] = 25; ws_be2v_[91] =  1;
   ws_be2v_[92] = 24; ws_be2v_[93] = 11; ws_be2v_[94] = 10; ws_be2v_[95] =  2;
   ws_be2v_[96] =  8; ws_be2v_[97] =  6; ws_be2v_[98] =  5; ws_be2v_[99] = 29;
   ws_be2v_[100] = 11; ws_be2v_[101] = 28; ws_be2v_[102] =  6;
   ws_be2v_[103] =  8;
   ws_be2v_[104] =  4; ws_be2v_[105] = 12; ws_be2v_[106] = 31;
   ws_be2v_[107] =  5;
   ws_be2v_[108] = 30; ws_be2v_[109] = 15; ws_be2v_[110] = 12;
   ws_be2v_[111] =  4;
   ws_be2v_[112] = 15; ws_be2v_[113] = 26; ws_be2v_[114] =  0;
   ws_be2v_[115] = 14;
   ws_be2v_[116] = 14; ws_be2v_[117] =  0; ws_be2v_[118] =  1;
   ws_be2v_[119] = 27;

   for (int i=0; i<30; i++) { ws_belem_att_[i] = 1; }
}

bool
RhombohedralLattice::MapToFundamentalDomain(const Vector & pt,
                                            Vector & ipt) const
{
   bool map = false;
   ipt = pt;
   return map;
}

Mesh *
RhombohedralLattice::GetWignerSeitzMesh(bool tetMesh) const
{
   bool rd = alpha_ < 0.5 * M_PI;

   Mesh * mesh = new Mesh((double*)ws_vert_, rd?15:38,
                          (int*)ws_e2v_, Geometry::CUBE,
                          (int*)ws_elem_att_, rd?4:16,
                          (int*)ws_be2v_, Geometry::SQUARE,
                          (int*)ws_belem_att_, rd?12:30,
                          3, 3);

   mesh->Finalize();

   std::ofstream ofs("rhl.mesh");
   mesh->Print(ofs);
   ofs.close();

   return mesh;
}

Mesh *
RhombohedralLattice::GetPeriodicWignerSeitzMesh(bool tetMesh) const
{
   Mesh * mesh = this->GetWignerSeitzMesh(tetMesh);

   // if ( rd )
   // mesh->UniformRefinement();
   mesh->UniformRefinement();

   Mesh * per_mesh = MakePeriodicMesh(mesh, trn_vecs_);

   delete mesh;

   {
      std::ofstream ofs("rhl_per.mesh");
      per_mesh->Print(ofs);
      ofs.close();
   }

   return per_mesh;
}

MonoclinicLattice::MonoclinicLattice(double a, double b, double c,
                                     double alpha)
   : BravaisLattice3D(a, b, c, alpha, 0.5 * M_PI, 0.5 * M_PI)
{
   MFEM_ASSERT( b_ <= c_,
                "Monoclinic unit cells must have b <= c!");
   MFEM_ASSERT( c_ * cos(alpha_) < b_ && alpha_ < 0.5 * M_PI,
                "Monoclinic unit cells must have arccos(b/c) < alpha < pi/2!");

   label_ = "MCL";
   type_  = PRIMITIVE_MONOCLINIC;

   // Set Lattice Vectors
   lat_vecs_[0][0] =   a_;
   lat_vecs_[0][1] =  0.0;
   lat_vecs_[0][2] =  0.0;

   lat_vecs_[1][0] =  0.0;
   lat_vecs_[1][1] =   b_;
   lat_vecs_[1][2] =  0.0;

   lat_vecs_[2][0] =  0.0;
   lat_vecs_[2][1] =  c_ * cos(alpha_);
   lat_vecs_[2][2] =  c_ * sin(alpha_);

   // Set Reciprocal Lattice Vectors
   rec_vecs_[0][0] =  1.0 / a_;
   rec_vecs_[0][1] =  0.0;
   rec_vecs_[0][2] =  0.0;

   rec_vecs_[1][0] =  0.0;
   rec_vecs_[1][1] =  1.0 / b_;
   rec_vecs_[1][2] = -1.0 / ( tan(alpha_) * b_ );

   rec_vecs_[2][0] =  0.0;
   rec_vecs_[2][1] =  0.0;
   rec_vecs_[2][2] =  1.0 / ( sin(alpha_) * c_ );

   // Set the Translation Vectors
   trn_vecs_.resize(4);
   for (unsigned int i=0; i<trn_vecs_.size(); i++)
   {
      trn_vecs_[i].SetSize(3);
   }
   trn_vecs_[0] =  lat_vecs_[0];
   trn_vecs_[1] =  lat_vecs_[1];
   trn_vecs_[2] =  lat_vecs_[2];
   trn_vecs_[3] =  lat_vecs_[2]; trn_vecs_[3] -= lat_vecs_[1];

   // Set the face radii
   double cosa = cos(alpha_);
   double sina = sin(alpha_);
   double csca = 1.0 / sina;
   face_radii_.resize(4);
   face_radii_[0] = 0.5 * min(sqrt(b_ * b_ + c_ * c_ - 2.0 * b_ * c_ * cosa),
                              min(b_, c_));
   face_radii_[1] = 0.5 * min(a_, (c_ - b_ * cosa) * csca);
   face_radii_[2] = 0.5 * min(a_, (b_ - c_ * cosa) * csca);
   face_radii_[3] = 0.5 * min(a_, sqrt(b_ * b_ + c_ * c_ - 2.0 * b_ * c_ * cosa)
                              * cosa * csca);

   this->SetCellVolumes();

   // Allocate sl_, sp_, and path_ vectors
   this->SetVectorSizes();

   // Set Symmetry Points and Labels
   double sina2 = pow(sina, 2.0);
   double eta = 0.5 * (1.0 - b_ * cosa / c_) / sina2;
   double nu = 0.5 - eta * c_ * cosa / b_;

   sl_[0] = "Gamma";
   sp_[0] = 0.0;

   sl_[1] = "A";
   add(0.5, rec_vecs_[0], 0.5, rec_vecs_[1], sp_[1]);

   sl_[2] = "C";
   add(0.5, rec_vecs_[1], 0.5, rec_vecs_[2], sp_[2]);

   sl_[3] = "D";
   add(0.5, rec_vecs_[0], 0.5, rec_vecs_[2], sp_[3]);

   sl_[4] = "D1"; /// Note that this point does not fall on any path
   add(0.5, rec_vecs_[0], -0.5, rec_vecs_[2], sp_[4]);

   sl_[5] = "E";
   add(0.5, rec_vecs_[0], 0.5, rec_vecs_[1], sp_[5]);
   sp_[5].Add(0.5, rec_vecs_[2]);

   sl_[6] = "H";
   add(eta, rec_vecs_[1], 1.0-nu, rec_vecs_[2], sp_[6]);

   sl_[7] = "H1";
   add(1.0-eta, rec_vecs_[1], nu, rec_vecs_[2], sp_[7]);

   sl_[8] = "H2"; /// Note that this point does not fall on any path
   add(eta, rec_vecs_[1], -nu, rec_vecs_[2], sp_[8]);

   sl_[9] = "M";
   add(0.5, rec_vecs_[0], eta, rec_vecs_[1], sp_[9]);
   sp_[9].Add(1.0-nu, rec_vecs_[2]);

   sl_[10] = "M1";
   add(0.5, rec_vecs_[0], 1.0-eta, rec_vecs_[1], sp_[10]);
   sp_[10].Add(nu, rec_vecs_[2]);

   sl_[11] = "M2"; /// Note that this point does not fall on any path
   add(0.5, rec_vecs_[0], eta, rec_vecs_[1], sp_[11]);
   sp_[11].Add(-nu, rec_vecs_[2]);

   sl_[12] = "X";
   sp_[12].Set(0.5, rec_vecs_[1]);

   sl_[13] = "Y";
   sp_[13].Set(0.5, rec_vecs_[2]);

   sl_[14] = "Y1"; /// Note that this point does not fall on any path
   sp_[14].Set(-0.5, rec_vecs_[2]);

   sl_[15] = "Z";
   sp_[15].Set(0.5, rec_vecs_[0]);

   // Define Paths
   path_[0][0] =  0;
   path_[0][1] = 13;
   path_[0][2] =  6;
   path_[0][3] =  2;
   path_[0][4] =  5;
   path_[0][5] = 10;
   path_[0][6] =  1;
   path_[0][7] = 12;
   path_[0][8] =  7;

   path_[1][0] =  9;
   path_[1][1] =  3;
   path_[1][2] = 15;

   path_[2][0] = 13;
   path_[2][1] =  3;

   // Set Intermediate Symmetry Points
   this->SetIntermediatePoints();

   // Set Intermediate Symmetry Point Labels
   il_[0][0] = "GammaY"; // Gamma -> Y
   il_[0][1] = "YH";     // Y     -> H
   il_[0][2] = "HC";     // H     -> C
   il_[0][3] = "CE";     // C     -> E
   il_[0][4] = "EM1";    // E     -> M1
   il_[0][5] = "M1A";    // M1    -> A
   il_[0][6] = "AX";     // A     -> X
   il_[0][7] = "XH1";    // X     -> H1

   il_[1][0] = "MD";     // M     -> D
   il_[1][1] = "DZ";     // D     -> Z

   il_[2][0] = "YD";     // Y     -> D

   // Set Mesh data
   double cos2a = cos(2.0 * alpha_);

   fd_vert_[ 0] =  0.0;
   fd_vert_[ 1] =  0.0;
   fd_vert_[ 2] =  0.0;

   fd_vert_[ 3] =  0.0;
   fd_vert_[ 4] = -0.5 * c_ * cosa;
   fd_vert_[ 5] = -0.5 * c_ * sina;

   fd_vert_[ 6] =  0.0;
   fd_vert_[ 7] =  0.5 * b_ - c_ * cosa;
   fd_vert_[ 8] = -0.5 * (b_ * cosa - c_ * cos2a) * csca;

   fd_vert_[ 9] =  0.0;
   fd_vert_[10] =  0.5 * (b_ - c_ * cosa);
   fd_vert_[11] = -0.5 * c_ * sina;

   fd_vert_[12] =  0.0;
   fd_vert_[13] =  0.5 * b_;
   fd_vert_[14] =  0.5 * (b_ * cosa - c_) * csca;

   fd_vert_[15] =  0.0;
   fd_vert_[16] =  0.5 * b_;
   fd_vert_[17] =  0.0;

   fd_vert_[18] =  0.0;
   fd_vert_[19] =  0.5 * b_;
   fd_vert_[20] =  0.5 * (c_ - b_ * cosa) * csca;

   fd_vert_[21] =  0.0;
   fd_vert_[22] =  0.5 * c_ * cosa;
   fd_vert_[23] =  0.5 * c_ * sina;

   for (int i=0; i<8; i++)
   {
      fd_vert_[3*i+24] = 0.5 * a_;
      fd_vert_[3*i+25] = fd_vert_[3*i+1];
      fd_vert_[3*i+26] = fd_vert_[3*i+2];
   }

   fd_e2v_[ 0] =  0; fd_e2v_[ 1] =  1; fd_e2v_[ 2] =  2; fd_e2v_[ 3] =  3;
   fd_e2v_[ 4] =  8; fd_e2v_[ 5] =  9; fd_e2v_[ 6] = 10; fd_e2v_[ 7] = 11;

   fd_e2v_[ 8] =  0; fd_e2v_[ 9] =  3; fd_e2v_[10] =  4; fd_e2v_[11] =  5;
   fd_e2v_[12] =  8; fd_e2v_[13] = 11; fd_e2v_[14] = 12; fd_e2v_[15] = 13;

   fd_e2v_[16] =  0; fd_e2v_[17] =  5; fd_e2v_[18] =  6; fd_e2v_[19] =  7;
   fd_e2v_[20] =  8; fd_e2v_[21] = 13; fd_e2v_[22] = 14; fd_e2v_[23] = 15;

   fd_elem_att_[0] = 1; fd_elem_att_[1] = 1; fd_elem_att_[2] = 1;
   fd_elem_att_[3] = 1; fd_elem_att_[4] = 1; fd_elem_att_[5] = 1;

   fd_be2v_[ 0] = 0; fd_be2v_[ 1] = 1; fd_be2v_[ 2] =  9; fd_be2v_[ 3] =  8;
   fd_be2v_[ 4] = 1; fd_be2v_[ 5] = 2; fd_be2v_[ 6] = 10; fd_be2v_[ 7] =  9;
   fd_be2v_[ 8] = 2; fd_be2v_[ 9] = 3; fd_be2v_[10] = 11; fd_be2v_[11] = 10;
   fd_be2v_[12] = 3; fd_be2v_[13] = 4; fd_be2v_[14] = 12; fd_be2v_[15] = 11;
   fd_be2v_[16] = 4; fd_be2v_[17] = 5; fd_be2v_[18] = 13; fd_be2v_[19] = 12;
   fd_be2v_[20] = 5; fd_be2v_[21] = 6; fd_be2v_[22] = 14; fd_be2v_[23] = 13;
   fd_be2v_[24] = 6; fd_be2v_[25] = 7; fd_be2v_[26] = 15; fd_be2v_[27] = 14;
   fd_be2v_[28] = 7; fd_be2v_[29] = 0; fd_be2v_[30] =  8; fd_be2v_[31] = 15;

   fd_be2v_[32] = 3; fd_be2v_[33] = 2; fd_be2v_[34] =  1; fd_be2v_[35] =  0;
   fd_be2v_[36] = 5; fd_be2v_[37] = 4; fd_be2v_[38] =  3; fd_be2v_[39] =  0;
   fd_be2v_[40] = 7; fd_be2v_[41] = 6; fd_be2v_[42] =  5; fd_be2v_[43] =  0;

   fd_be2v_[44] = 8; fd_be2v_[45] =  9; fd_be2v_[46] = 10; fd_be2v_[47] = 11;
   fd_be2v_[48] = 8; fd_be2v_[49] = 11; fd_be2v_[50] = 12; fd_be2v_[51] = 13;
   fd_be2v_[52] = 8; fd_be2v_[53] = 13; fd_be2v_[54] = 14; fd_be2v_[55] = 15;

   fd_belem_att_[ 0] = 10; fd_belem_att_[ 1] =  1;
   fd_belem_att_[ 2] =  1; fd_belem_att_[ 3] =  1;
   fd_belem_att_[ 4] =  1; fd_belem_att_[ 5] =  1;
   fd_belem_att_[ 6] =  1; fd_belem_att_[ 7] = 10;
   fd_belem_att_[ 8] = 10; fd_belem_att_[ 9] = 10; fd_belem_att_[10] = 10;
   fd_belem_att_[11] =  1; fd_belem_att_[12] =  1; fd_belem_att_[13] =  1;

   for (int i=0; i<63; i++) { ws_vert_[i] = 0.0; }
   for (int i=0; i<3; i++)
   {
      int o = 21*i;

      ws_vert_[ 1 + o] =  0.5 * ( b_ - 2.0 * c_ * cos(alpha_) );
      ws_vert_[ 2 + o] = -0.5 * ( 3.0 * b_ * cos(alpha_)
                                  - c_ * ( 2.0 + cos(2.0  * alpha_) )
                                ) / sin(alpha_);

      ws_vert_[ 4 + o] =  0.5 * ( b_ - 2.0 * c_ * cos(alpha_) );
      ws_vert_[ 5 + o] = -0.5 * ( b_ * cos(alpha_) - c_ * cos(2.0 * alpha_)
                                ) / sin(alpha_);

      ws_vert_[ 7 + o] =  0.5 * b_;
      ws_vert_[ 8 + o] = -0.5 * ( c_ - b_ * cos(alpha_) ) / sin(alpha_);

      ws_vert_[10 + o] =  0.5 * b_;
      ws_vert_[11 + o] =  0.5 * ( c_ - b_ * cos(alpha_) ) / sin(alpha_);

      ws_vert_[13 + o] = -0.5 * ( b_ - 2.0 * c_ * cos(alpha_) );
      ws_vert_[14 + o] =  0.5 * ( b_ * cos(alpha_) - c_ * cos(2.0 * alpha_)
                                ) / sin(alpha_);

      ws_vert_[16 + o] = -0.5 * b_;
      ws_vert_[17 + o] =  0.5 * ( c_ - b_ * cos(alpha_) ) / sin(alpha_);

      ws_vert_[19 + o] = -0.5 * b_;
      ws_vert_[20 + o] = -0.5 * ( c_ - b_ * cos(alpha_) ) / sin(alpha_);

      double x = 0.5 * a_ * (-1.0 + i);
      for (int i=0; i<7; i++) { ws_vert_[3 * i + 0 + o] = x; }
   }

   for (int i=0; i<2; i++)
   {
      int o = 24*i;
      int v =  7*i;

      for (int j=0; j<3; j++)
      {
         ws_e2v_[8*j+o+0] = 0 + v;
         ws_e2v_[8*j+o+1] = 2 * j + 1 + v;
         ws_e2v_[8*j+o+2] = 2 * j + 2 + v;
         ws_e2v_[8*j+o+3] = ((2 * j + 2) % 6) + 1 + v;

         for (int k=0; k<4; k++) { ws_e2v_[8*j+o+4+k] = ws_e2v_[8*j+o+k] + 7; }
      }
   }

   for (int i=0; i<6; i++) { ws_elem_att_[i] = 1; }

   ws_be2v_[0] = 0; ws_be2v_[1] = 3; ws_be2v_[ 2] = 2; ws_be2v_[ 3] = 1;
   ws_be2v_[4] = 0; ws_be2v_[5] = 5; ws_be2v_[ 6] = 4; ws_be2v_[ 7] = 3;
   ws_be2v_[8] = 0; ws_be2v_[9] = 1; ws_be2v_[10] = 6; ws_be2v_[11] = 5;

   for (int i=0; i<2; i++)
   {
      for (int j=0; j<6; j++)
      {
         ws_be2v_[24*i+4*j+0+12] = 7*i+j+1;
         ws_be2v_[24*i+4*j+1+12] = 7*i+((j+1)%6)+1;
         ws_be2v_[24*i+4*j+2+12] = 7*i+((j+1)%6)+1+7;
         ws_be2v_[24*i+4*j+3+12] = 7*i+j+1+7;
      }
   }

   ws_be2v_[60] = 14; ws_be2v_[61] = 15; ws_be2v_[62] = 16; ws_be2v_[63] = 17;
   ws_be2v_[64] = 14; ws_be2v_[65] = 17; ws_be2v_[66] = 18; ws_be2v_[67] = 19;
   ws_be2v_[68] = 14; ws_be2v_[69] = 19; ws_be2v_[70] = 20; ws_be2v_[71] = 15;

   for (int i=0; i<18; i++) { ws_belem_att_[i] = 1; }
}

bool
MonoclinicLattice::MapToFundamentalDomain(const Vector & pt,
                                          Vector & ipt) const
{
   bool map = false;
   /*
   ipt = pt;
   if ( ipt[2] < 0.0 )
   {
      ipt[2] *= -1.0;
      map = true;
   }
   if ( ipt[1] > 0.0 )
   {
      ipt[1] *= -1.0;
      map = true;
   }
   if ( ipt[0] < 0.0 )
   {
      ipt[0] *= -1.0;
      map = true;
   }
   if ( ipt[1] < -sqrt(3.0) * ipt[0] )
   {
      double d0 = ipt[0], d1 = ipt[1];
      ipt[0] = -0.5 * (d0 + sqrt(3.0) * d1);
      ipt[1] =  0.5 * (d1 - sqrt(3.0) * d0);
      map = true;
   }
   if ( ipt[0] > -sqrt(3.0) * ipt[1] )
   {
      double d0 = ipt[0], d1 = ipt[1];
      ipt[0] =  0.5 * (d0 - sqrt(3.0) * d1);
      ipt[1] = -0.5 * (d1 + sqrt(3.0) * d0);
      map = true;
   }
   */
   return map;
}

const DenseMatrix & MonoclinicLattice::GetTransformation(int ti) const
{
   T_ = 0.0;

   for (int i=0; i<3; i++)
   {
      T_(i,i) = 1.0;
   }

   if (ti == 1 || ti == 2)
   {
      T_(1,1) *= -1.0;
      T_(2,2) *= -1.0;
   }
   if ( ti >= 2 )
   {
      T_(0,0) *= -1.0;
   }

   return T_;
}

Mesh *
MonoclinicLattice::GetFundamentalDomainMesh() const
{
   /*
   Mesh * mesh = new Mesh((double*)fd_vert_, 8,
                          (int*)fd_e2v_, Geometry::TRIANGLE,
                          (int*)fd_elem_att_, 6,
                          (int*)fd_be2v_, Geometry::SEGMENT,
                          (int*)fd_belem_att_, 8,
                          2, 2);
   */
   Mesh * mesh = new Mesh((double*)fd_vert_, 16,
                          (int*)fd_e2v_, Geometry::CUBE,
                          (int*)fd_elem_att_, 3,
                          (int*)fd_be2v_, Geometry::SQUARE,
                          (int*)fd_belem_att_, 14,
                          3, 3);

   mesh->Finalize();

   return mesh;
}
/*
Mesh *
MonoclinicLattice::GetWignerSeitzMesh(bool tetMesh) const
{
 Mesh * mesh = new Mesh((double*)ws_vert_, 21,
                        (int*)ws_e2v_, Geometry::CUBE,
                        (int*)ws_elem_att_, 6,
                        (int*)ws_be2v_, Geometry::SQUARE,
                        (int*)ws_belem_att_, 18,
                        3, 3);
 mesh->Finalize();

 return mesh;
}

Mesh *
MonoclinicLattice::GetPeriodicWignerSeitzMesh(bool tetMesh) const
{
 Mesh * mesh = this->GetWignerSeitzMesh(tetMesh);

 {
    std::ofstream ofs("mcl.mesh");
    mesh->Print(ofs);
    ofs.close();
 }

 mesh->UniformRefinement();

 Mesh * per_mesh = MakePeriodicMesh(mesh, trn_vecs_, 1);

 delete mesh;

 {
    std::ofstream ofs("per-mcl.mesh");
    per_mesh->Print(ofs);
    ofs.close();
 }

 return per_mesh;
}
*/
BaseCenteredMonoclinicLattice::BaseCenteredMonoclinicLattice(double a,
                                                             double b,
                                                             double c,
                                                             double alpha)
   : BravaisLattice3D(a, b, c, alpha, 0.5 * M_PI, 0.5 * M_PI)
{
   MFEM_ASSERT( b_ <= c_,
                "Base-Centered Monoclinic unit cells must have b <= c!");
   MFEM_ASSERT( c_ * cos(alpha_) < b_ && alpha_ < 0.5 * M_PI,
                "Base-Centered Monoclinic unit cells must have "
                "arccos(b/c) < alpha < pi/2!");

   label_ = "MCLC";
   type_  = BASE_CENTERED_MONOCLINIC;

   // Set Lattice Vectors
   lat_vecs_[0][0] =  0.5 * a_;
   lat_vecs_[0][1] =  0.5 * b_;
   lat_vecs_[0][2] =  0.0;

   lat_vecs_[1][0] = -0.5 * a_;
   lat_vecs_[1][1] =  0.5 * b_;
   lat_vecs_[1][2] =  0.0;

   lat_vecs_[2][0] =  0.0;
   lat_vecs_[2][1] =  c_ * cos(alpha_);
   lat_vecs_[2][2] =  c_ * sin(alpha_);

   // Set Reciprocal Lattice Vectors
   rec_vecs_[0][0] =  1.0 / a_;
   rec_vecs_[0][1] =  1.0 / b_;
   rec_vecs_[0][2] = -cos(alpha_) / ( b_ * sin(alpha_) );

   rec_vecs_[1][0] = -1.0 / a_;
   rec_vecs_[1][1] =  1.0 / b_;
   rec_vecs_[1][2] = -cos(alpha_) / ( b_ * sin(alpha_) );

   rec_vecs_[2][0] =  0.0;
   rec_vecs_[2][1] =  0.0;
   rec_vecs_[2][2] =  1.0 / ( c_ * sin(alpha_) );

   // Set the Translation Vectors
   trn_vecs_.resize((2.0*b_*c_*cos(alpha_)+a_*a_-b_*b_<0.0)?6:7);
   for (unsigned int i=0; i<trn_vecs_.size(); i++)
   {
      trn_vecs_[i].SetSize(3);
   }
   if ( 2.0 * b_ * c_ * cos(alpha_) + a_ * a_ - b_ * b_ < 0.0 )
   {
   }
   else
   {
      if ( a_ < b_ )
      {
         trn_vecs_[0] =  lat_vecs_[0];
         trn_vecs_[1] =  lat_vecs_[1];
         trn_vecs_[2] =  lat_vecs_[2];
         trn_vecs_[3] =  lat_vecs_[2]; trn_vecs_[3] -= lat_vecs_[0];
         trn_vecs_[4] =  lat_vecs_[2]; trn_vecs_[4] -= lat_vecs_[1];
         trn_vecs_[5] =  lat_vecs_[0]; trn_vecs_[5] -= lat_vecs_[1];
      }
      else
      {
         trn_vecs_[0] =  lat_vecs_[0];
         trn_vecs_[1] =  lat_vecs_[1];
         trn_vecs_[2] =  lat_vecs_[2];
         trn_vecs_[3] =  lat_vecs_[2]; trn_vecs_[3] -= lat_vecs_[0];
         trn_vecs_[4] =  lat_vecs_[2]; trn_vecs_[4] -= lat_vecs_[1];
         trn_vecs_[5] =  lat_vecs_[0]; trn_vecs_[5] += lat_vecs_[1];
         trn_vecs_[6] =  lat_vecs_[2]; trn_vecs_[6] -= trn_vecs_[5];
      }
   }

   this->SetCellVolumes();

   // Allocate sl_, sp_, and path_ vectors
   this->SetVectorSizes();
   /*
   // Set Symmetry Points and Labels
   sl_[0] = "Gamma";
   sp_[0] = 0.0;

   sl_[1] = "A";
   sp_[1].Set(0.5, rec_vecs_[2]);

   sl_[2] = "H";
   add(1.0/3.0, rec_vecs_[0], 1.0/3.0, rec_vecs_[1], sp_[2]);
   sp_[2].Add(0.5, rec_vecs_[2]);

   sl_[3] = "K";
   add(1.0/3.0, rec_vecs_[0], 1.0/3.0, rec_vecs_[1], sp_[3]);

   sl_[4] = "L";
   add(0.5, rec_vecs_[0], 0.5, rec_vecs_[2], sp_[4]);

   sl_[5] = "M";
   sp_[5].Set(0.5, rec_vecs_[0]);

   // Define Paths
   path_[0][0] = 0;
   path_[0][1] = 5;
   path_[0][2] = 3;
   path_[0][3] = 0;
   path_[0][4] = 1;
   path_[0][5] = 4;
   path_[0][6] = 2;
   path_[0][7] = 1;

   path_[1][0] = 4;
   path_[1][1] = 5;

   path_[2][0] = 3;
   path_[2][1] = 2;

   // Set Intermediate Symmetry Points
   this->SetIntermediatePoints();

   // Set Intermediate Symmetry Point Labels
   il_[0][0] = "Sigma";  // Gamma -> M
   il_[0][1] = "MK";     // M     -> K
   il_[0][2] = "GammaK"; // K     -> Gamma
   il_[0][3] = "Delta";  // Gamma -> A
   il_[0][4] = "AL";     // A     -> L
   il_[0][5] = "LH";     // L     -> H
   il_[0][6] = "AH";     // H     -> A
   il_[1][0] = "LM";     // L     -> M
   il_[2][0] = "HK";     // K     -> H
   */

   if ( 2.0 * b_ * c_ * cos(alpha_) + a_ * a_ - b_ * b_ < 0.0 )
   {
      this->createElongatedDodecahedron();
   }
   else
   {
      this->createTruncatedOctahedron();
   }

   std::ofstream ofs("mclc.coords");
   ofs << "{";

   int nn = ( 2.0 * b_ * c_ * cos(alpha_) + a_ * a_ - b_ * b_ < 0.0 )?24:38;
   for (int i=0; i<nn; i++)
   {
      ofs << "{";
      ofs << ws_vert_[3*i+0] << ","
          << ws_vert_[3*i+1] << ","
          << ws_vert_[3*i+2];
      ofs << "}";
      if ( i < nn-1 ) { ofs << "," << endl; }
   }
   ofs << "}" << endl;
   ofs.close();

}

void
BaseCenteredMonoclinicLattice::createElongatedDodecahedron()
{
   cout << "Setting up data for elongated dodecahedron" << endl;

   // Set Mesh data
   double cosa = cos(alpha_);
   double csca = 1.0 / sin(alpha_);

   for (int i=0; i<114; i++) { ws_vert_[i] = 0.0; }

   ws_vert_[ 0] = -0.50 * a_;
   ws_vert_[ 1] =  0.25 * (b_ - a_ * a_ / b_);
   ws_vert_[ 2] = -0.50 * (c_ - 0.50 * (b_ + a_ * a_ / b_) * cosa) * csca;

   ws_vert_[ 3] = -0.50 * a_;
   ws_vert_[ 4] =  0.25 * (b_ - a_ * a_ / b_);
   ws_vert_[ 5] =  0.50 * (c_ - 0.50 * (b_ - a_ * a_ / b_) * cosa) * csca;

   ws_vert_[ 6] = -0.50 * a_;
   ws_vert_[ 7] =  0.25 * (b_ - a_ * a_ / b_) - c_ * cosa;
   ws_vert_[ 8] =  0.50 * (c_ * (1.0 + 2.0 * cosa * cosa)
                           - 0.50 * (3.0 * b_ - a_ * a_ / b_) * cosa ) * csca;

   ws_vert_[ 9] = -0.50 * a_;
   ws_vert_[10] =  0.25 * (b_ - a_ * a_ / b_) - c_ * cosa;
   ws_vert_[11] = -0.50 * (c_ * (1.0 - 2.0 * cosa * cosa)
                           + 0.5 * (b_ * b_ - a_ * a_) * cosa) * csca;

   ws_vert_[13] =  0.25 * (b_ + a_ * a_ / b_);
   ws_vert_[14] = -0.50 * (c_ - 0.50 * (b_ - a_ * a_ / b_) * cosa) * csca;

   ws_vert_[16] =  0.25 * (b_ + a_ * a_ / b_);
   ws_vert_[17] =  0.50 * (c_ - 0.50 * (b_ + a_ * a_ / b_) * cosa) * csca;

   ws_vert_[19] =  0.25 * (b_ + a_ * a_ / b_) - c_ * cosa;
   ws_vert_[20] =  0.50 * (c_ * (1.0 + 2.0 * cosa * cosa)
                           - 0.50 * (3.0 * b_ + a_ * a_ / b_) * cosa ) * csca;

   ws_vert_[22] =  0.25 * (b_ + a_ * a_ / b_) - c_ * cosa;
   ws_vert_[23] = -0.25 * (2.0 * c_ * (1.0 - 2.0 * cosa * cosa)
                           + (b_ + a_ * a_ / b_) * cosa) * csca;

   ws_vert_[24] =  0.50 * a_;
   ws_vert_[25] =  0.25 * (b_ - a_ * a_ / b_);
   ws_vert_[26] = -0.50 * (c_ - 0.50 * (b_ + a_ * a_ / b_) * cosa) * csca;

   ws_vert_[27] =  0.50 * a_;
   ws_vert_[28] =  0.25 * (b_ - a_ * a_ / b_);
   ws_vert_[29] =  0.50 * (c_ - 0.50 * (b_ - a_ * a_ / b_) * cosa) * csca;

   ws_vert_[30] =  0.50 * a_;
   ws_vert_[31] =  0.25 * (b_ - a_ * a_ / b_) - c_ * cosa;
   ws_vert_[32] =  0.50 * (c_ * (1.0 + 2.0 * cosa * cosa)
                           - 0.50 * (3.0 * b_ - a_ * a_ / b_) * cosa ) * csca;

   ws_vert_[33] =  0.50 * a_;
   ws_vert_[34] =  0.25 * (b_ - a_ * a_ / b_) - c_ * cosa;
   ws_vert_[35] = -0.25 * (2.0 * c_ * (1.0 - 2.0 * cosa * cosa)
                           + (b_ - a_ * a_ / b_) * cosa) * csca;

   ws_vert_[37] = -0.25 * (b_ + a_ * a_ / b_);
   ws_vert_[38] = -0.50 * (c_ - 0.50 * (b_ + a_ * a_ / b_) * cosa) * csca;

   ws_vert_[39] =  0.50 * a_;
   ws_vert_[40] = -0.25 * (b_ - a_ * a_ / b_);
   ws_vert_[41] = -0.50 * (c_ - 0.50 * (b_ - a_ * a_ / b_) * cosa) * csca;

   ws_vert_[43] = -0.25 * (b_ - 3.0 * a_ * a_ / b_);
   ws_vert_[44] = -0.50 * (c_ - 0.50 * (b_ - 3.0 * a_ * a_ / b_) * cosa) * csca;

   ws_vert_[45] = -0.50 * a_;
   ws_vert_[46] = -0.25 * (b_ - a_ * a_ / b_);
   ws_vert_[47] = -0.50 * (c_ - 0.50 * (b_ - a_ * a_ / b_) * cosa) * csca;

   ws_vert_[49] = -0.25 * (b_ + a_ * a_ / b_);
   ws_vert_[50] =  0.50 * (c_ - 0.50 * (b_ - a_ * a_ / b_) * cosa) * csca;

   ws_vert_[51] =  0.50 * a_;
   ws_vert_[52] = -0.25 * (b_ - a_ * a_ / b_);
   ws_vert_[53] =  0.50 * (c_ - 0.50 * (b_ + a_ * a_ / b_) * cosa) * csca;

   ws_vert_[55] = -0.25 * (b_ - 3.0 * a_ * a_ / b_);
   ws_vert_[56] =  0.50 * (c_ - 0.50 * (b_ + 3.0 * a_ * a_ / b_) * cosa) * csca;

   ws_vert_[57] = -0.50 * a_;
   ws_vert_[58] = -0.25 * (b_ - a_ * a_ / b_);
   ws_vert_[59] =  0.50 * (c_ - 0.50 * (b_ + a_ * a_ / b_) * cosa) * csca;

   ws_vert_[61] = -0.25 * (b_ + a_ * a_ / b_) + c_ * cosa;
   ws_vert_[62] =  0.25 * (2.0 * c_ * (1.0 - 2.0 * cosa * cosa)
                           + (b_ + a_ * a_ / b_) * cosa) * csca;

   ws_vert_[63] =  0.50 * a_;
   ws_vert_[64] = -0.25 * (b_ - a_ * a_ / b_) + c_ * cosa;
   ws_vert_[65] =  0.25 * (2.0 * c_ * (1.0 - 2.0 * cosa * cosa)
                           + (b_ - a_ * a_ / b_) * cosa) * csca;

   ws_vert_[67] = -0.25 * (b_ - 3.0 * a_ * a_ / b_) + c_ * cosa;
   ws_vert_[68] = 0.25 * (2.0 * c_ * (1.0 - 2.0 * cosa * cosa)
                          + (b_ - 3.0 * a_ * a_ / b_) * cosa) * csca;

   ws_vert_[69] = -0.50 * a_;
   ws_vert_[70] = -0.25 * (b_ - a_ * a_ / b_) + c_ * cosa;
   ws_vert_[71] =  0.25 * (2.0 * c_ * (1.0 - 2.0 * cosa * cosa)
                           + (b_ - a_ * a_ / b_) * cosa) * csca;

   for (int i=0; i<2; i++)
   {
      for (int j=0; j<8; j++)
      {
         ws_e2v_[8 * i + j]      = 4 * i + j;
         ws_e2v_[8 * i + j + 16] = 4 * i + j + 12;
      }
   }

   ws_e2v_[32] =  6; ws_e2v_[33] =  2; ws_e2v_[34] =  3; ws_e2v_[35] =  7;
   ws_e2v_[36] = 18; ws_e2v_[37] = 19; ws_e2v_[38] = 15; ws_e2v_[39] = 14;

   ws_e2v_[40] =  6; ws_e2v_[41] =  7; ws_e2v_[42] = 11; ws_e2v_[43] = 10;
   ws_e2v_[44] = 18; ws_e2v_[45] = 14; ws_e2v_[46] = 13; ws_e2v_[47] = 17;

   ws_e2v_[48] =  6; ws_e2v_[49] = 10; ws_e2v_[50] =  9; ws_e2v_[51] =  5;
   ws_e2v_[52] = 18; ws_e2v_[53] = 17; ws_e2v_[54] = 21; ws_e2v_[55] = 22;

   ws_e2v_[56] =  6; ws_e2v_[57] =  5; ws_e2v_[58] =  1; ws_e2v_[59] =  2;
   ws_e2v_[60] = 18; ws_e2v_[61] = 22; ws_e2v_[62] = 23; ws_e2v_[63] = 19;

   for (int i=64; i<128; i++) { ws_e2v_[i] = -1; }

   for (int i=0; i< 8; i++) { ws_elem_att_[i] =  1; }
   for (int i=8; i<16; i++) { ws_elem_att_[i] = -1; }

   ws_be2v_[ 0] =  4; ws_be2v_[ 1] =  7; ws_be2v_[ 2] =  3; ws_be2v_[ 3] =  0;
   ws_be2v_[ 4] =  4; ws_be2v_[ 5] =  8; ws_be2v_[ 6] = 11; ws_be2v_[ 7] =  7;
   ws_be2v_[ 8] =  4; ws_be2v_[ 9] =  5; ws_be2v_[10] =  9; ws_be2v_[11] =  8;
   ws_be2v_[12] =  4; ws_be2v_[13] =  0; ws_be2v_[14] =  1; ws_be2v_[15] =  5;

   ws_be2v_[16] = 16; ws_be2v_[17] = 19; ws_be2v_[18] = 15; ws_be2v_[19] = 12;
   ws_be2v_[20] = 16; ws_be2v_[21] = 12; ws_be2v_[22] = 13; ws_be2v_[23] = 17;
   ws_be2v_[24] = 16; ws_be2v_[25] = 17; ws_be2v_[26] = 21; ws_be2v_[27] = 20;
   ws_be2v_[28] = 16; ws_be2v_[29] = 20; ws_be2v_[30] = 23; ws_be2v_[31] = 19;

   ws_be2v_[32] =  0; ws_be2v_[33] =  3; ws_be2v_[34] =  2; ws_be2v_[35] =  1;
   ws_be2v_[36] =  8; ws_be2v_[37] =  9; ws_be2v_[38] = 10; ws_be2v_[39] = 11;
   ws_be2v_[40] = 12; ws_be2v_[41] = 15; ws_be2v_[42] = 14; ws_be2v_[43] = 13;
   ws_be2v_[44] = 20; ws_be2v_[45] = 21; ws_be2v_[46] = 22; ws_be2v_[47] = 23;

   ws_be2v_[48] =  2; ws_be2v_[49] =  3; ws_be2v_[50] = 15; ws_be2v_[51] = 19;
   ws_be2v_[52] =  3; ws_be2v_[53] =  7; ws_be2v_[54] = 14; ws_be2v_[55] = 15;
   ws_be2v_[56] =  7; ws_be2v_[57] = 11; ws_be2v_[58] = 13; ws_be2v_[59] = 14;
   ws_be2v_[60] = 11; ws_be2v_[61] = 10; ws_be2v_[62] = 17; ws_be2v_[63] = 13;

   ws_be2v_[64] = 10; ws_be2v_[65] =  9; ws_be2v_[66] = 21; ws_be2v_[67] = 17;
   ws_be2v_[68] =  9; ws_be2v_[69] =  5; ws_be2v_[70] = 22; ws_be2v_[71] = 21;
   ws_be2v_[72] =  5; ws_be2v_[73] =  1; ws_be2v_[74] = 23; ws_be2v_[75] = 22;
   ws_be2v_[76] =  1; ws_be2v_[77] =  2; ws_be2v_[78] = 19; ws_be2v_[79] = 23;

   for (int i=80; i<120; i++) { ws_be2v_[i] = -1; }

   for (int i= 0; i<20; i++) { ws_belem_att_[i] =  1; }
   for (int i=20; i<30; i++) { ws_belem_att_[i] = -1; }
}

void
BaseCenteredMonoclinicLattice::createTruncatedOctahedron()
{
   cout << "Setting up data for truncated octahedron" << endl;

   // Set Mesh data
   for (int i=0; i<114; i++) { ws_vert_[i] = 0.0; }

   double sina  = sin(alpha_);
   double cosa  = cos(alpha_);
   // double seca  = 1.0 / cos(alpha_);
   double csca  = 1.0 / sin(alpha_);

   if ( a_ < b_ )
   {
      ws_vert_[  0] = -0.50 * a_;
      ws_vert_[  1] =  0.25 * ( b_ - a_ * a_ / b_ );
      ws_vert_[  2] = -0.50 * (c_ - 0.50 * ( b_ + a_ * a_ / b_ ) * cosa) * csca;

      ws_vert_[  3] = -0.50 * a_;
      ws_vert_[  4] =  0.25 * ( b_ - a_ * a_ / b_ );
      ws_vert_[  5] =  0.25 * (2.0 * c_ + (b_ * b_ - a_ * a_) / c_ -
                               (3.0 * b_ - a_ * a_ / b_) * cosa) * csca;

      ws_vert_[  6] = -0.50 * a_;
      ws_vert_[  7] = -0.25 * ( b_ - a_ * a_ / b_ );
      ws_vert_[  8] =  0.50 * (c_ - 0.50 * ( b_ + a_ * a_ / b_ ) * cosa) * csca;

      ws_vert_[  9] = -0.50 * a_;
      ws_vert_[ 10] = -0.25 * ( b_ - a_ * a_ / b_ );
      ws_vert_[ 11] = -0.25 * (2.0 * c_ + (b_ * b_ - a_ * a_) / c_ -
                               (3.0 * b_ - a_ * a_ / b_) * cosa) * csca;

      ws_vert_[ 12] =  0.50 * a_;
      ws_vert_[ 13] =  0.25 * ( b_ - a_ * a_ / b_ );
      ws_vert_[ 14] = -0.50 * (c_ - 0.50 * ( b_ + a_ * a_ / b_ ) * cosa) * csca;

      ws_vert_[ 15] =  0.50 * a_;
      ws_vert_[ 16] =  0.25 * ( b_ - a_ * a_ / b_ );
      ws_vert_[ 17] =  0.25 * (2.0 * c_ + (b_ * b_ - a_ * a_) / c_ -
                               (3.0 * b_ - a_ * a_ / b_) * cosa) * csca;

      ws_vert_[ 18] =  0.50 * a_;
      ws_vert_[ 19] = -0.25 * ( b_ - a_ * a_ / b_ );
      ws_vert_[ 20] =  0.50 * (c_ - 0.50 * ( b_ + a_ * a_ / b_ ) * cosa) * csca;

      ws_vert_[ 21] =  0.50 * a_;
      ws_vert_[ 22] = -0.25 * ( b_ - a_ * a_ / b_ );
      ws_vert_[ 23] = -0.25 * (2.0 * c_ + (b_ * b_ - a_ * a_) / c_ -
                               (3.0 * b_ - a_ * a_ / b_) * cosa) * csca;

      ws_vert_[ 24] =  0.50 * b_ * c_ * cosa / a_ - 0.25 * (b_ * b_ / a_ - a_);
      ws_vert_[ 25] = -0.50 * (b_ - c_ * cosa);
      ws_vert_[ 26] =  0.50 * c_ * sina;

      ws_vert_[ 28] =  0.25 * a_ * a_ / b_ - 0.75 * b_ + c_ * cosa;
      ws_vert_[ 29] = -0.25 * (2.0 * c_ * (2.0 * cosa * cosa - 1.0)
                               + (b_ * b_ - a_ * a_) / c_ -
                               (3.0 * b_ - a_ * a_ / b_) * cosa) * csca;

      ws_vert_[ 30] = -0.50 * b_ * c_ * cosa / a_ + 0.25 * (b_ * b_ / a_ - a_);
      ws_vert_[ 31] = -0.50 * (b_ - c_ * cosa);
      ws_vert_[ 32] =  0.50 * c_ * sina;

      ws_vert_[ 34] = -0.25 * ( b_ + a_ * a_ / b_);
      ws_vert_[ 35] =  0.25 * (2.0 * c_ + (b_ * b_ - a_ * a_) / c_ -
                               (3.0 * b_ - a_ * a_ / b_) * cosa) * csca;

      ws_vert_[ 36] =  0.50 * b_ * c_ * cosa / a_ - 0.25 * (b_ * b_ / a_ - a_);
      ws_vert_[ 37] =  0.50 * (b_ - c_ * cosa);
      ws_vert_[ 38] = -0.50 * c_ * sina;

      ws_vert_[ 40] =  0.25 * ( b_ + a_ * a_ / b_);
      ws_vert_[ 41] = -0.25 * (2.0 * c_ + (b_ * b_ - a_ * a_) / c_ -
                               (3.0 * b_ - a_ * a_ / b_) * cosa) * csca;

      ws_vert_[ 42] = -0.50 * b_ * c_ * cosa / a_ + 0.25 * (b_ * b_ / a_ - a_);
      ws_vert_[ 43] =  0.50 * (b_ - c_ * cosa);
      ws_vert_[ 44] = -0.50 * c_ * sina;

      ws_vert_[ 46] = -0.25 * a_ * a_ / b_ + 0.75 * b_ - c_ * cosa;
      ws_vert_[ 47] =  0.25 * (2.0 * c_ * (2.0 * cosa * cosa - 1.0)
                               + (b_ * b_ - a_ * a_) / c_ -
                               (3.0 * b_ - a_ * a_ / b_) * cosa) * csca;

      ws_vert_[ 48] = -0.50 * b_ * c_ * cosa / a_ + 0.25 * (b_ * b_ / a_ + a_);
      ws_vert_[ 49] = -0.50 * c_ * cosa;
      ws_vert_[ 50] = -0.50 * c_ * sina;

      ws_vert_[ 52] = -c_ * cosa + 0.25 * (b_ + a_ * a_ / b_);
      ws_vert_[ 53] =  0.50 * c_ * (2.0 * cosa * cosa - 1.0) * csca
                       - 0.25 * (a_ * a_ / b_ + b_) * cosa * csca;

      ws_vert_[ 54] =  0.50 * b_ * c_ * cosa / a_ - 0.25 * (b_ * b_ / a_ + a_);
      ws_vert_[ 55] = -0.50 * c_ * cosa;
      ws_vert_[ 56] = -0.50 * c_ * sina;

      ws_vert_[ 58] = -0.25 * ( b_ + a_ * a_ / b_);
      ws_vert_[ 59] = -0.50 * (c_ - 0.50 * ( b_ + a_ * a_ / b_ ) * cosa) * csca;

      ws_vert_[ 60] = -0.50 * b_ * c_ * cosa / a_ + 0.25 * (b_ * b_ / a_ + a_);
      ws_vert_[ 61] =  0.50 * c_ * cosa;
      ws_vert_[ 62] =  0.50 * c_ * sina;

      ws_vert_[ 64] =  0.25 * ( b_ + a_ * a_ / b_);
      ws_vert_[ 65] =  0.50 * (c_ - 0.50 * ( b_ + a_ * a_ / b_ ) * cosa) * csca;

      ws_vert_[ 66] =  0.50 * b_ * c_ * cosa / a_ - 0.25 * (b_ * b_ / a_ + a_);
      ws_vert_[ 67] =  0.50 * c_ * cosa;
      ws_vert_[ 68] =  0.50 * c_ * sina;

      ws_vert_[ 70] =  c_ * cosa - 0.25 * (b_ + a_ * a_ / b_);
      ws_vert_[ 71] = -0.50 * c_ * (2.0 * cosa * cosa - 1.0) * csca
                      + 0.25 * (a_ * a_ / b_ + b_) * cosa * csca;

      ws_vert_[ 72] =  0.50 * b_ * c_ * cosa / a_ - 0.25 * (b_ * b_ / a_ + a_);
      ws_vert_[ 73] = -0.50 * c_ * cosa;
      ws_vert_[ 74] =  (0.5 * c_ * (cosa * cosa + 1.0)
                        + 0.25 * (b_ * b_ - a_ * a_) / c_ - b_ * cosa) * csca;

      ws_vert_[ 75] = -0.50 * b_ * c_ * cosa / a_ + 0.25 * (b_ * b_ / a_ - a_);
      ws_vert_[ 76] =  0.50 * (c_ * cosa - a_ * a_ / b_);
      ws_vert_[ 77] =  0.50 * (c_ + (a_ * a_ / b_ - b_ - c_ * cosa) * cosa
                               + 0.5 * (b_ * b_ - a_ * a_) / c_) * csca;

      ws_vert_[ 78] =  0.50 * b_ * c_ * cosa / a_ - 0.25 * (b_ * b_ / a_ + a_);
      ws_vert_[ 79] = -0.50 * (c_ * cosa - b_ + a_ * a_ / b_);
      ws_vert_[ 80] = -0.50 * (c_ - (a_ * a_ / b_ - b_ + c_ * cosa) * cosa
                               - 0.5 * (b_ * b_ - a_ * a_) / c_) * csca;

      ws_vert_[ 81] = -0.50 * b_ * c_ * cosa / a_ + 0.25 * (b_ * b_ / a_ - a_);
      ws_vert_[ 82] =  0.50 * (b_ - c_ * cosa);
      ws_vert_[ 83] =  (0.5 * c_ * (cosa * cosa + 1.0)
                        + 0.25 * (b_ * b_ - a_ * a_) / c_ - b_ * cosa) * csca;

      ws_vert_[ 84] = -0.50 * b_ * c_ * cosa / a_ + 0.25 * (b_ * b_ / a_ + a_);
      ws_vert_[ 85] = -0.50 * c_ * cosa;
      ws_vert_[ 86] =  (0.5 * c_ * (cosa * cosa + 1.0)
                        + 0.25 * (b_ * b_ - a_ * a_) / c_ - b_ * cosa) * csca;

      ws_vert_[ 87] =  0.50 * b_ * c_ * cosa / a_ - 0.25 * (b_ * b_ / a_ - a_);
      ws_vert_[ 88] =  0.50 * (c_ * cosa - a_ * a_ / b_);
      ws_vert_[ 89] =  0.50 * (c_ + (a_ * a_ / b_ - b_ - c_ * cosa) * cosa
                               + 0.5 * (b_ * b_ - a_ * a_) / c_) * csca;

      ws_vert_[ 90] = -0.50 * b_ * c_ * cosa / a_ + 0.25 * (b_ * b_ / a_ + a_);
      ws_vert_[ 91] = -0.50 * (c_ * cosa - b_ + a_ * a_ / b_);
      ws_vert_[ 92] = -0.50 * (c_ - (a_ * a_ / b_ - b_ + c_ * cosa) * cosa
                               - 0.5 * (b_ * b_ - a_ * a_) / c_) * csca;

      ws_vert_[ 93] =  0.50 * b_ * c_ * cosa / a_ - 0.25 * (b_ * b_ / a_ - a_);
      ws_vert_[ 94] =  0.50 * (b_ - c_ * cosa);
      ws_vert_[ 95] =  (0.5 * c_ * (cosa * cosa + 1.0)
                        + 0.25 * (b_ * b_ - a_ * a_) / c_ - b_ * cosa) * csca;

      ws_vert_[ 96] =  0.50 * b_ * c_ * cosa / a_ - 0.25 * (b_ * b_ / a_ + a_);
      ws_vert_[ 97] = -0.50 * (c_ * cosa - b_ + a_ * a_ / b_);
      ws_vert_[ 98] =  0.50 * (c_ + (b_ * b_ - a_ * a_) / c_
                               + (a_ * a_ / b_ - 3.0 * b_ + c_ * cosa) * cosa
                              ) * csca;

      ws_vert_[ 99] = -0.50 * b_ * c_ * cosa / a_ + 0.25 * (b_ * b_ / a_ + a_);
      ws_vert_[100] = -0.50 * (c_ * cosa - b_ + a_ * a_ / b_);
      ws_vert_[101] =  0.50 * (c_ + (b_ * b_ - a_ * a_) / c_
                               + (a_ * a_ / b_ - 3.0 * b_ + c_ * cosa) * cosa
                              ) * csca;

      ws_vert_[103] =  0.25 * (b_ - 3.0 * a_ * a_ / b_);
      ws_vert_[104] =  0.50 * (c_ + (b_ * b_ - a_ * a_) / c_
                               + (1.5 * a_ * a_ / b_ - 2.5 * b_) * cosa) * csca;

      ws_vert_[106] =  0.75 * b_ - 0.25 * a_ * a_/b_ - c_ * cosa;
      ws_vert_[107] =  0.25 * (2.0 * ( c_ + (b_ * b_ - a_ * a_ ) / c_)
                               + (a_ * a_ / b_ - 7.0 * b_ + 4.0 * c_ * cosa
                                 ) * cosa
                              ) * csca;

      ws_vert_[109] =  0.25 * (b_ + a_ * a_ / b_) - c_ * cosa;
      ws_vert_[110] =  0.25 * (( 2.0 * c_ + (b_ * b_ - a_ * a_ ) / c_)
                               - (a_ * a_ / b_ + 5.0 * b_ - 4.0 * c_ * cosa
                                 ) * cosa
                              ) * csca;

      ws_vert_[111] = -0.5 * b_ * (b_ - 2.0 * c_ * cosa) / a_;
      ws_vert_[112] =  0.25 * (b_ - a_ * a_ / b_);
      ws_vert_[113] =  0.25 * (2.0 * c_ + (b_ * b_ - a_ * a_) / c_ -
                               (3.0 * b_ - a_ * a_ / b_) * cosa) * csca;
   }
   else
   {
      ws_vert_[  0] = -0.25 * (a_ - b_* b_ / a_);
      ws_vert_[  1] = -0.50 * b_;
      ws_vert_[  2] = -0.50 * (c_ - b_ * cosa) * csca;

      ws_vert_[  3] = -0.25 * (a_ - b_* b_ / a_);
      ws_vert_[  4] = -0.50 * b_;
      ws_vert_[  5] =  0.50 * (c_ - b_ * cosa) * csca;

      ws_vert_[  6] =  0.25 * (a_ - b_* b_ / a_);
      ws_vert_[  7] = -0.50 * b_;
      ws_vert_[  8] =  0.50 * (c_ - b_ * cosa) * csca;

      ws_vert_[  9] =  0.25 * (a_ - b_* b_ / a_);
      ws_vert_[ 10] = -0.50 * b_;
      ws_vert_[ 11] = -0.50 * (c_ - b_ * cosa) * csca;

      ws_vert_[ 12] = -0.25 * (a_ - b_* b_ / a_);
      ws_vert_[ 13] =  0.50 * b_;
      ws_vert_[ 14] = -0.50 * (c_ - b_ * cosa) * csca;

      ws_vert_[ 15] = -0.25 * (a_ - b_* b_ / a_);
      ws_vert_[ 16] =  0.50 * b_;
      ws_vert_[ 17] =  0.50 * (c_ - b_ * cosa) * csca;

      ws_vert_[ 18] =  0.25 * (a_ - b_* b_ / a_);
      ws_vert_[ 19] =  0.50 * b_;
      ws_vert_[ 20] =  0.50 * (c_ - b_ * cosa) * csca;

      ws_vert_[ 21] =  0.25 * (a_ - b_* b_ / a_);
      ws_vert_[ 22] =  0.50 * b_;
      ws_vert_[ 23] = -0.50 * (c_ - b_ * cosa) * csca;

      ws_vert_[ 24] =  0.25 * (a_ + b_* b_ / a_ - 2.0 * b_ * c_ * cosa / a_);
      ws_vert_[ 25] =  0.50 * c_ * cosa;
      ws_vert_[ 26] =  0.50 * c_ * sina;

      ws_vert_[ 27] =  0.25 * (a_ - b_* b_ / a_);
      ws_vert_[ 28] = -0.50 * b_ + c_ * cosa;
      ws_vert_[ 29] = -0.50 * (c_ * (2.0 * cosa * cosa - 1.0) - b_ * cosa
                              ) * csca;

      ws_vert_[ 30] =  0.25 * (a_ - b_* b_ / a_ + 2.0 * b_ * c_ * cosa / a_);
      ws_vert_[ 31] =  0.50 * (c_ * cosa - b_);
      ws_vert_[ 32] =  0.50 * c_ * sina;

      ws_vert_[ 33] =  0.25 * (a_ + b_* b_ / a_);
      ws_vert_[ 35] =  0.50 * (c_ - b_ * cosa) * csca;

      ws_vert_[ 36] = -0.25 * (a_ - b_* b_ / a_ + 2.0 * b_ * c_ * cosa / a_);
      ws_vert_[ 37] =  0.50 * (b_ - c_ * cosa);
      ws_vert_[ 38] = -0.50 * c_ * sina;

      ws_vert_[ 39] = -0.25 * (a_ + b_* b_ / a_);
      ws_vert_[ 41] = -0.50 * (c_ - b_ * cosa) * csca;

      ws_vert_[ 42] = -0.25 * (a_ + b_* b_ / a_ - 2.0 * b_ * c_ * cosa / a_);
      ws_vert_[ 43] = -0.50 * c_ * cosa;
      ws_vert_[ 44] = -0.50 * c_ * sina;

      ws_vert_[ 45] = -0.25 * (a_ - b_* b_ / a_);
      ws_vert_[ 46] =  0.50 * (b_ - 2.0 * c_ * cosa);
      ws_vert_[ 47] =  0.50 * (c_ * (2.0 * cosa * cosa - 1.0) - b_ * cosa
                              ) * csca;

      ws_vert_[ 48] =  0.25 * (a_ - b_* b_ / a_ + 2.0 * b_ * c_ * cosa / a_);
      ws_vert_[ 49] =  0.50 * (b_ - c_ * cosa);
      ws_vert_[ 50] = -0.50 * c_ * sina;

      ws_vert_[ 51] =  0.25 * (a_ - b_* b_ / a_);
      ws_vert_[ 52] =  0.50 * (b_ - 2.0 * c_ * cosa);
      ws_vert_[ 53] =  0.50 * (c_ * (2.0 * cosa * cosa - 1.0) - b_ * cosa
                              ) * csca;

      ws_vert_[ 54] =  0.25 * (a_ + b_* b_ / a_ - 2.0 * b_ * c_ * cosa / a_);
      ws_vert_[ 55] = -0.50 * c_ * cosa;
      ws_vert_[ 56] = -0.50 * c_ * sina;

      ws_vert_[ 57] =  0.25 * (a_ + b_* b_ / a_);
      ws_vert_[ 59] = -0.50 * (c_ - b_ * cosa) * csca;

      ws_vert_[ 60] = -0.25 * (a_ + b_* b_ / a_ - 2.0 * b_ * c_ * cosa / a_);
      ws_vert_[ 61] =  0.50 * c_ * cosa;
      ws_vert_[ 62] =  0.50 * c_ * sina;

      ws_vert_[ 63] = -0.25 * (a_ + b_* b_ / a_);
      ws_vert_[ 65] =  0.50 * (c_ - b_ * cosa) * csca;

      ws_vert_[ 66] = -0.25 * (a_ - b_* b_ / a_ + 2.0 * b_ * c_ * cosa / a_);
      ws_vert_[ 67] =  0.50 * (c_ * cosa - b_);
      ws_vert_[ 68] =  0.50 * c_ * sina;

      ws_vert_[ 69] = -0.25 * (a_ - b_* b_ / a_);
      ws_vert_[ 70] = -0.50 * (b_ - 2.0 * c_ * cosa);
      ws_vert_[ 71] = -0.50 * (c_ * (2.0 * cosa * cosa - 1.0) - b_ * cosa
                              ) * csca;

      ws_vert_[ 72] =  0.25 * (a_ + b_* b_ / a_ - 2.0 * b_ * c_ * cosa / a_);
      ws_vert_[ 73] = -0.50 * c_ * cosa;
      ws_vert_[ 74] =  0.50 * (c_ * (1.0 + cosa * cosa) - 2.0 * b_ * cosa
                              ) * csca;

      ws_vert_[ 75] = -0.25 * (a_ - b_* b_ / a_ - 2.0 * b_ * c_ * cosa / a_);
      ws_vert_[ 76] =  0.50 * (c_ * cosa - b_);
      ws_vert_[ 77] =  0.50 * c_ * sina;

      ws_vert_[ 78] = -0.25 * (a_ - 3.0 * b_* b_ / a_
                               + 2.0 * b_ * c_ * cosa / a_);
      ws_vert_[ 79] = -0.50 * c_ * cosa;
      ws_vert_[ 80] = -0.50 * c_ * sina;

      ws_vert_[ 81] = -0.25 * (a_ + b_* b_ / a_ - 2.0 * b_ * c_ * cosa / a_);
      ws_vert_[ 82] = -0.50 * c_ * cosa;
      ws_vert_[ 83] =  0.50 * (c_ * (1.0 + cosa * cosa) - 2.0 * b_ * cosa
                              ) * csca;

      ws_vert_[ 84] =  0.25 * (a_ - b_* b_ / a_ + 2.0 * b_ * c_ * cosa / a_);
      ws_vert_[ 85] =  0.50 * (b_ - c_ * cosa);
      ws_vert_[ 86] =  0.50 * (c_ * (1.0 + cosa * cosa) - 2.0 * b_ * cosa
                              ) * csca;

      ws_vert_[ 87] = -0.25 * (a_ - 3.0 * b_* b_ / a_
                               + 2.0 * b_ * c_ * cosa / a_);
      ws_vert_[ 88] =  0.50 * c_ * cosa;
      ws_vert_[ 89] =  0.50 * c_ * sina;

      ws_vert_[ 90] = -0.25 * (a_ - b_* b_ / a_ - 2.0 * b_ * c_ * cosa / a_);
      ws_vert_[ 91] =  0.50 * (b_ - c_ * cosa);
      ws_vert_[ 92] = -0.50 * c_ * sina;

      ws_vert_[ 93] = -0.25 * (a_ - b_* b_ / a_ + 2.0 * b_ * c_ * cosa / a_);
      ws_vert_[ 94] =  0.50 * (b_ - c_ * cosa);
      ws_vert_[ 95] =  0.50 * (c_ * (1.0 + cosa * cosa) - 2.0 * b_ * cosa
                              ) * csca;

      ws_vert_[ 96] = -0.25 * (a_ - 3.0 * b_* b_ / a_
                               + 2.0 * b_ * c_ * cosa / a_);
      ws_vert_[ 97] = -0.50 * c_ * cosa;
      ws_vert_[ 98] =  0.50 * (c_ * (1.0 + cosa * cosa) - 2.0 * b_ * cosa
                              ) * csca;

      ws_vert_[ 99] = -0.25 * (a_ - b_* b_ / a_ - 2.0 * b_ * c_ * cosa / a_);
      ws_vert_[100] =  0.50 * (b_ - c_ * cosa);
      ws_vert_[101] =  0.50 * (c_ * (1.0 + cosa * cosa) - 2.0 * b_ * cosa
                              ) * csca;

      ws_vert_[102] = -0.25 * (a_ - 3.0 * b_* b_ / a_);
      ws_vert_[104] =  0.50 * (c_ - b_ * cosa) * csca;

      ws_vert_[105] = -0.25 * (a_ - b_* b_ / a_);
      ws_vert_[106] =  0.50 * (b_ - 2.0 * c_ * cosa);
      ws_vert_[107] =  0.50 * (c_ * (1.0 + 2.0 * cosa * cosa)
                               - 3.0 * b_ * cosa) * csca;

      ws_vert_[108] =  0.25 * (a_ - b_* b_ / a_);
      ws_vert_[109] =  0.50 * (b_ - 2.0 * c_ * cosa);
      ws_vert_[110] =  0.50 * (c_ * (1.0 + 2.0 * cosa * cosa)
                               - 3.0 * b_ * cosa) * csca;
      ws_vert_[111] = -0.25 * (a_ - 3.0 * b_* b_ / a_
                               + 4.0 * b_ * c_ * cosa / a_);
      ws_vert_[113] =  0.50 * (c_ - b_ * cosa) * csca;

   }

   ws_e2v_[  0] =  0; ws_e2v_[  1] =  1; ws_e2v_[  2] =  2; ws_e2v_[  3] =  3;
   ws_e2v_[  4] = 26; ws_e2v_[  5] = 32; ws_e2v_[  6] = 24; ws_e2v_[  7] = 18;

   ws_e2v_[  8] = 26; ws_e2v_[  9] = 32; ws_e2v_[ 10] = 24; ws_e2v_[ 11] = 18;
   ws_e2v_[ 12] = 15; ws_e2v_[ 13] = 35; ws_e2v_[ 14] = 36; ws_e2v_[ 15] = 17;

   ws_e2v_[ 16] = 15; ws_e2v_[ 17] = 35; ws_e2v_[ 18] = 36; ws_e2v_[ 19] = 17;
   ws_e2v_[ 20] = 30; ws_e2v_[ 21] = 33; ws_e2v_[ 22] = 28; ws_e2v_[ 23] = 16;

   ws_e2v_[ 24] = 30; ws_e2v_[ 25] = 33; ws_e2v_[ 26] = 28; ws_e2v_[ 27] = 16;
   ws_e2v_[ 28] =  4; ws_e2v_[ 29] =  5; ws_e2v_[ 30] =  6; ws_e2v_[ 31] =  7;

   ws_e2v_[ 32] =  9; ws_e2v_[ 33] =  8; ws_e2v_[ 34] = 11; ws_e2v_[ 35] = 10;
   ws_e2v_[ 36] = 23; ws_e2v_[ 37] = 29; ws_e2v_[ 38] = 34; ws_e2v_[ 39] = 25;

   ws_e2v_[ 40] = 23; ws_e2v_[ 41] = 29; ws_e2v_[ 42] = 34; ws_e2v_[ 43] = 25;
   ws_e2v_[ 44] = 22; ws_e2v_[ 45] = 37; ws_e2v_[ 46] = 32; ws_e2v_[ 47] =  1;

   ws_e2v_[ 48] = 22; ws_e2v_[ 49] = 37; ws_e2v_[ 50] = 32; ws_e2v_[ 51] =  1;
   ws_e2v_[ 52] = 21; ws_e2v_[ 53] = 31; ws_e2v_[ 54] = 35; ws_e2v_[ 55] = 27;

   ws_e2v_[ 56] = 21; ws_e2v_[ 57] = 31; ws_e2v_[ 58] = 35; ws_e2v_[ 59] = 27;
   ws_e2v_[ 60] = 13; ws_e2v_[ 61] = 12; ws_e2v_[ 62] = 15; ws_e2v_[ 63] = 14;

   ws_e2v_[ 64] = 16; ws_e2v_[ 65] = 17; ws_e2v_[ 66] = 18; ws_e2v_[ 67] = 19;
   ws_e2v_[ 68] = 28; ws_e2v_[ 69] = 36; ws_e2v_[ 70] = 24; ws_e2v_[ 71] = 11;

   ws_e2v_[ 72] = 28; ws_e2v_[ 73] = 36; ws_e2v_[ 74] = 24; ws_e2v_[ 75] = 11;
   ws_e2v_[ 76] = 33; ws_e2v_[ 77] = 35; ws_e2v_[ 78] = 32; ws_e2v_[ 79] = 34;

   ws_e2v_[ 80] = 33; ws_e2v_[ 81] = 35; ws_e2v_[ 82] = 32; ws_e2v_[ 83] = 34;
   ws_e2v_[ 84] =  5; ws_e2v_[ 85] = 31; ws_e2v_[ 86] = 37; ws_e2v_[ 87] = 29;

   ws_e2v_[ 88] =  5; ws_e2v_[ 89] = 31; ws_e2v_[ 90] = 37; ws_e2v_[ 91] = 29;
   ws_e2v_[ 92] = 20; ws_e2v_[ 93] = 21; ws_e2v_[ 94] = 22; ws_e2v_[ 95] = 23;

   ws_e2v_[ 96] = 24; ws_e2v_[ 97] = 11; ws_e2v_[ 98] = 34; ws_e2v_[ 99] = 32;
   ws_e2v_[100] =  2; ws_e2v_[101] = 10; ws_e2v_[102] = 25; ws_e2v_[103] =  1;

   ws_e2v_[104] = 11; ws_e2v_[105] = 28; ws_e2v_[106] = 33; ws_e2v_[107] = 34;
   ws_e2v_[108] =  8; ws_e2v_[109] =  6; ws_e2v_[110] =  5; ws_e2v_[111] = 29;

   ws_e2v_[112] = 30; ws_e2v_[113] = 15; ws_e2v_[114] = 35; ws_e2v_[115] = 33;
   ws_e2v_[116] =  4; ws_e2v_[117] = 12; ws_e2v_[118] = 31; ws_e2v_[119] =  5;

   ws_e2v_[120] = 15; ws_e2v_[121] = 26; ws_e2v_[122] = 32; ws_e2v_[123] = 35;
   ws_e2v_[124] = 14; ws_e2v_[125] =  0; ws_e2v_[126] =  1; ws_e2v_[127] = 27;

   for (int i=0; i<16; i++) { ws_elem_att_[i] = 1; }

   ws_be2v_[ 0] =  3; ws_be2v_[ 1] =  2; ws_be2v_[ 2] =  1; ws_be2v_[ 3] =  0;
   ws_be2v_[ 4] =  2; ws_be2v_[ 5] =  3; ws_be2v_[ 6] = 18; ws_be2v_[ 7] = 24;
   ws_be2v_[ 8] =  3; ws_be2v_[ 9] =  0; ws_be2v_[10] = 26; ws_be2v_[11] = 18;
   ws_be2v_[12] =  4; ws_be2v_[13] =  5; ws_be2v_[14] =  6; ws_be2v_[15] =  7;
   ws_be2v_[16] = 16; ws_be2v_[17] = 30; ws_be2v_[18] =  4; ws_be2v_[19] =  7;
   ws_be2v_[20] = 16; ws_be2v_[21] =  7; ws_be2v_[22] =  6; ws_be2v_[23] = 28;
   ws_be2v_[24] = 17; ws_be2v_[25] = 15; ws_be2v_[26] = 30; ws_be2v_[27] = 16;
   ws_be2v_[28] = 15; ws_be2v_[29] = 17; ws_be2v_[30] = 18; ws_be2v_[31] = 26;
   ws_be2v_[32] =  9; ws_be2v_[33] = 10; ws_be2v_[34] = 11; ws_be2v_[35] =  8;
   ws_be2v_[36] =  9; ws_be2v_[37] =  8; ws_be2v_[38] = 29; ws_be2v_[39] = 23;
   ws_be2v_[40] =  9; ws_be2v_[41] = 23; ws_be2v_[42] = 25; ws_be2v_[43] = 10;
   ws_be2v_[44] =  1; ws_be2v_[45] = 25; ws_be2v_[46] = 23; ws_be2v_[47] = 22;
   ws_be2v_[48] =  1; ws_be2v_[49] = 22; ws_be2v_[50] = 21; ws_be2v_[51] = 27;
   ws_be2v_[52] = 13; ws_be2v_[53] = 12; ws_be2v_[54] = 15; ws_be2v_[55] = 14;
   ws_be2v_[56] = 21; ws_be2v_[57] = 31; ws_be2v_[58] = 12; ws_be2v_[59] = 13;
   ws_be2v_[60] = 21; ws_be2v_[61] = 13; ws_be2v_[62] = 14; ws_be2v_[63] = 27;
   ws_be2v_[64] = 16; ws_be2v_[65] = 19; ws_be2v_[66] = 18; ws_be2v_[67] = 17;
   ws_be2v_[68] = 11; ws_be2v_[69] = 24; ws_be2v_[70] = 18; ws_be2v_[71] = 19;
   ws_be2v_[72] = 11; ws_be2v_[73] = 19; ws_be2v_[74] = 16; ws_be2v_[75] = 28;
   ws_be2v_[76] = 20; ws_be2v_[77] = 21; ws_be2v_[78] = 22; ws_be2v_[79] = 23;
   ws_be2v_[80] =  5; ws_be2v_[81] = 31; ws_be2v_[82] = 21; ws_be2v_[83] = 20;
   ws_be2v_[84] =  5; ws_be2v_[85] = 20; ws_be2v_[86] = 23; ws_be2v_[87] = 29;
   ws_be2v_[88] =  2; ws_be2v_[89] = 10; ws_be2v_[90] = 25; ws_be2v_[91] =  1;
   ws_be2v_[92] = 24; ws_be2v_[93] = 11; ws_be2v_[94] = 10; ws_be2v_[95] =  2;
   ws_be2v_[96] =  8; ws_be2v_[97] =  6; ws_be2v_[98] =  5; ws_be2v_[99] = 29;
   ws_be2v_[100] = 11; ws_be2v_[101] = 28; ws_be2v_[102] =  6;
   ws_be2v_[103] =  8;
   ws_be2v_[104] =  4; ws_be2v_[105] = 12; ws_be2v_[106] = 31;
   ws_be2v_[107] =  5;
   ws_be2v_[108] = 30; ws_be2v_[109] = 15; ws_be2v_[110] = 12;
   ws_be2v_[111] =  4;
   ws_be2v_[112] = 15; ws_be2v_[113] = 26; ws_be2v_[114] =  0;
   ws_be2v_[115] = 14;
   ws_be2v_[116] = 14; ws_be2v_[117] =  0; ws_be2v_[118] =  1;
   ws_be2v_[119] = 27;

   for (int i=0; i<30; i++) { ws_belem_att_[i] = 1; }
}

bool
BaseCenteredMonoclinicLattice::MapToFundamentalDomain(const Vector & pt,
                                                      Vector & ipt) const
{
   bool map = false;
   /*
   ipt = pt;
   if ( ipt[2] < 0.0 )
   {
      ipt[2] *= -1.0;
      map = true;
   }
   if ( ipt[1] > 0.0 )
   {
      ipt[1] *= -1.0;
      map = true;
   }
   if ( ipt[0] < 0.0 )
   {
      ipt[0] *= -1.0;
      map = true;
   }
   if ( ipt[1] < -sqrt(3.0) * ipt[0] )
   {
      double d0 = ipt[0], d1 = ipt[1];
      ipt[0] = -0.5 * (d0 + sqrt(3.0) * d1);
      ipt[1] =  0.5 * (d1 - sqrt(3.0) * d0);
      map = true;
   }
   if ( ipt[0] > -sqrt(3.0) * ipt[1] )
   {
      double d0 = ipt[0], d1 = ipt[1];
      ipt[0] =  0.5 * (d0 - sqrt(3.0) * d1);
      ipt[1] = -0.5 * (d1 + sqrt(3.0) * d0);
      map = true;
   }
   */
   return map;
}

Mesh *
BaseCenteredMonoclinicLattice::GetWignerSeitzMesh(bool tetMesh) const
{
   bool ed = 2.0 * b_ * c_ * cos(alpha_) + a_ * a_ - b_ * b_ < 0.0;
   Mesh * mesh = new Mesh((double*)ws_vert_, ed?24:38,
                          (int*)ws_e2v_, Geometry::CUBE,
                          (int*)ws_elem_att_, ed?8:16,
                          (int*)ws_be2v_, Geometry::SQUARE,
                          (int*)ws_belem_att_, ed?20:30,
                          3, 3);
   mesh->Finalize();

   return mesh;
}

Mesh *
BaseCenteredMonoclinicLattice::GetPeriodicWignerSeitzMesh(bool tetMesh) const
{
   Mesh * mesh = this->GetWignerSeitzMesh(tetMesh);

   {
      std::ofstream ofs("mclc.mesh");
      mesh->Print(ofs);
      ofs.close();
   }

   mesh->UniformRefinement();

   Mesh * per_mesh = MakePeriodicMesh(mesh, trn_vecs_);

   delete mesh;

   {
      std::ofstream ofs("per-mclc.mesh");
      per_mesh->Print(ofs);
      ofs.close();
   }

   return per_mesh;
}

TriclinicLattice::TriclinicLattice(double a, double b, double c,
                                   double alpha, double beta, double gamma)
   : BravaisLattice3D(a, b, c, alpha, beta, gamma)
{
   label_ = "TRI";
   type_  = PRIMITIVE_TRICLINIC;

   double ca = cos(alpha_);
   double cb = cos(beta_);
   double cg = cos(gamma_);
   double sg = sin(gamma_);
   double ca2 = ca * ca;
   double cb2 = cb * cb;
   double sg2 = sg * sg;
   double cabg = ca * cb * cg;

   // Set Lattice Vectors
   lat_vecs_[0][0] = a_; lat_vecs_[0][1] = 0.0; lat_vecs_[0][2] = 0.0;
   lat_vecs_[1][0] = b_ * cg; lat_vecs_[1][1] = b_ * sg; lat_vecs_[1][2] = 0.0;
   lat_vecs_[2][0] = c_ * cb; lat_vecs_[2][1] = c_ * (ca - cb * cg) / sg;
   lat_vecs_[2][2] = c_ * sqrt(sg2 - ca2 - cb2 + 2.0 * cabg) / sg;

   // Set Reciprocal Lattice Vectors
   rec_vecs_[0][0] = 1.0 / a_;
   rec_vecs_[0][1] = -cg / (a_ * sg);
   rec_vecs_[0][2] = (ca * cg - cb) /
                     (a_ * sg * sqrt(sg2 - ca2 - cb2 + 2.0 * cabg));

   rec_vecs_[1][0] = 0.0;
   rec_vecs_[1][1] = 1.0 / (b_ * sg);
   rec_vecs_[1][2] = (cb * cg - ca) /
                     (b_ * sg * sqrt(sg2 - ca2 - cb2 + 2.0 * cabg));

   rec_vecs_[2][0] = 0.0;
   rec_vecs_[2][1] = 0.0;
   rec_vecs_[2][2] = sg / (c_ * sqrt(sg2 - ca2 - cb2 + 2.0 * cabg));

   bool kalpha_acute = rec_vecs_[1] * rec_vecs_[2] > 0.0;
   bool  kbeta_acute = rec_vecs_[0] * rec_vecs_[2] > 0.0;
   MFEM_VERIFY(kalpha_acute == kbeta_acute, "Unrecognized Triclinic variation");
   double ckgamma = rec_vecs_[0] * rec_vecs_[1] /
                    (rec_vecs_[0].Norml2() * rec_vecs_[1].Norml2());
   t12ab_ = (fabs(ckgamma) < 1e-4) ? 2 : 0;
   if (kalpha_acute && kbeta_acute)
   {
      t12ab_++;
   }

   switch (t12ab_)
   {
      case 0:
         label_ = "TRI1a";
         break;
      case 1:
         label_ = "TRI1b";
         break;
      case 2:
         label_ = "TRI2a";
         break;
      case 3:
         label_ = "TRI2b";
         break;
   };

   // Set Translation Vectors
   trn_vecs_.resize(7 - t12ab_ / 2);
   for (unsigned int i=0; i<trn_vecs_.size(); i++)
   {
      trn_vecs_[i].SetSize(3);
   }
   /*
   trn_vecs_[0][0] =  a_; trn_vecs_[0][1] = 0.0; trn_vecs_[0][2] = 0.0;
   trn_vecs_[1][0] = 0.0; trn_vecs_[1][1] =  b_; trn_vecs_[1][2] = 0.0;
   trn_vecs_[2][0] = 0.0; trn_vecs_[2][1] = 0.0; trn_vecs_[2][2] =  c_;
   */

   // Set the face radii
   face_radii_.resize(7 - t12ab_ / 2);
   /*
   face_radii_[0] = 0.5 * min(b_,c_);
   face_radii_[1] = 0.5 * min(c_,a_);
   face_radii_[2] = 0.5 * min(a_,b_);
   */

   this->SetCellVolumes();

   // Allocate sl_, sp_, and path_ vectors
   this->SetVectorSizes();

   // Set Symmetry Points and Labels
   sl_[0] = "Gamma";
   sp_[0] = 0.0;

   sl_[1] = "L";
   add(0.5,rec_vecs_[0],0.5 - double(t12ab_ % 2),rec_vecs_[2],sp_[1]);

   sl_[2] = "M";
   sp_[2].Set(0.5,rec_vecs_[2]);
   if (t12ab_ % 2 == 0) { sp_[2].Add(0.5,rec_vecs_[1]); }

   sl_[3] = "N";
   add(0.5,rec_vecs_[0],0.5,rec_vecs_[2],sp_[3]);
   if (t12ab_ % 2 == 1)
   {
      sp_[3].Add(0.5,rec_vecs_[1]);
      sp_[3] *= -1.0;
   }

   sl_[4] = "R";
   sp_[4].Set(0.5,rec_vecs_[2]);
   if (t12ab_ % 2 == 0)
   {
      sp_[4].Add(0.5,rec_vecs_[0]);
      sp_[4].Add(0.5,rec_vecs_[1]);
   }
   else
   {
      sp_[4].Add(-0.5,rec_vecs_[1]);
   }

   sl_[5] = "X";
   sp_[5].Set((t12ab_%2 == 0) ? 0.5 : -0.5,rec_vecs_[t12ab_%2]);

   sl_[6] = "Y";
   sp_[6].Set(0.5,rec_vecs_[1 - (t12ab_ % 2)]);

   sl_[7] = "Z";
   sp_[7].Set(0.5,rec_vecs_[2]);
   if (t12ab_%2 == 1) { sp_[7].Add(-0.5,rec_vecs_[0]); }

   for (unsigned int i=0; i<sl_.size(); i++)
   {
      si_[sl_[i]] = i;
   }

   // Define Paths
   path_[0][0] = 5;
   path_[0][1] = 0;
   path_[0][2] = 6;

   path_[1][0] = 1;
   path_[1][1] = 0;
   path_[1][2] = 7;

   path_[2][0] = 3;
   path_[2][1] = 0;
   path_[2][2] = 2;

   path_[3][0] = 4;
   path_[3][1] = 0;

   // Set Intermediate Symmetry Points
   this->SetIntermediatePoints();

   // Set Intermediate Symmetry Point Labels
   il_[0][0] = "XGamma"; // X     -> Gamma
   il_[0][1] = "GammaY"; // Gamma -> Y

   il_[1][0] = "LGamma"; // L     -> Gamma
   il_[1][1] = "GammaZ"; // Gamma -> Z

   il_[2][0] = "NGamma"; // N     -> Gamma
   il_[2][1] = "GammaM"; // Gamma -> M

   il_[3][0] = "RGamma"; // R     -> Gamma
}

bool
TriclinicLattice::MapToFundamentalDomain(const Vector & pt,
                                         Vector & ipt) const
{
   bool map = false;
   return map;
}

const DenseMatrix &
TriclinicLattice::GetTransformation(int ti) const
{
   T_ = 0.0;

   T_(0, 0) = 1.0 - 2.0 * ti;
   T_(1, 1) = 1.0 - 2.0 * ti;
   T_(2, 2) = 1.0 - 2.0 * ti;

   return T_;
}

Mesh *
TriclinicLattice::GetFundamentalDomainMesh() const
{
   /*
    Mesh * mesh = new Mesh((double*)fd_vert_, 8,
                           (int*)fd_e2v_, Geometry::CUBE,
                           (int*)fd_elem_att_, 1,
                           (int*)fd_be2v_, Geometry::SQUARE,
                           (int*)fd_belem_att_, 6,
                           3, 3);

    mesh->Finalize();

    return mesh;
   */
   return NULL;
}

BravaisLattice *
BravaisLatticeFactory(BRAVAIS_LATTICE_TYPE lattice_type,
                      double a, double b, double c,
                      double alpha, double beta, double gamma,
                      int logging)
{
   BravaisLattice * bravais = NULL;

   switch (lattice_type)
   {
      case PRIMITIVE_SEGMENT:
         // Primitive Linear Lattice
         // lattice_label = "SQR";
         if ( a <= 0.0 ) { a = 1.0; }
         if ( logging > 0 )
         {
            cout << "Calling LinearLattice(" << a << ")" << endl;
         }
         bravais = new LinearLattice(a);
         break;
      case PRIMITIVE_SQUARE:
         // Primitive square Lattice
         // lattice_label = "SQR";
         if ( a <= 0.0 ) { a = 1.0; }
         if ( logging > 0 )
         {
            cout << "Calling SquareLattice(" << a << ")" << endl;
         }
         bravais = new SquareLattice(a);
         break;
      case PRIMITIVE_HEXAGONAL:
         // Hexagonal Lattice
         // lattice_label = "HEX2D";
         if ( a <= 0.0 ) { a = 1.0; }
         if ( logging > 0 )
         {
            cout << "Calling HexagonalLattice(" << a << ")" << endl;
         }
         bravais = new HexagonalLattice(a);
         break;
      case PRIMITIVE_RECTANGULAR:
         // Rectangular Lattice
         // lattice_label = "RECT";
         if ( a <= 0.0 ) { a = 0.5; }
         if ( b <= 0.0 ) { b = 1.0; }
         bravais = new RectangularLattice(a, b);
         break;
      case CENTERED_RECTANGULAR:
         // Centered Rectangular Lattice
         // lattice_label = "RECTI";
         if ( a <= 0.0 ) { a = 0.5; }
         if ( b <= 0.0 ) { b = 1.0; }
         bravais = new CenteredRectangularLattice(a, b);
         break;
      case PRIMITIVE_OBLIQUE:
         // Oblique Lattice
         // lattice_label = "OBL";
         if ( a <= 0.0 ) { a = 0.5; }
         if ( b <= 0.0 ) { b = 1.0; }
         if ( gamma <= 0.0 ) { gamma = 0.4 * M_PI; }
         bravais = new ObliqueLattice(a, b, gamma);
         break;
      case PRIMITIVE_CUBIC:
         // Cubic Lattice
         // lattice_label = "CUB";
         if ( a <= 0.0 ) { a = 1.0; }
         if ( logging > 0 )
         {
            cout << "Calling CubicLattice(" << a << ")" << endl;
         }
         bravais = new CubicLattice(a);
         break;
      case FACE_CENTERED_CUBIC:
         // Face-Centered Cubic Lattice
         // lattice_label = "FCC";
         if ( a <= 0.0 ) { a = 1.0; }
         if ( logging > 0 )
         {
            cout << "Calling FaceCenteredCubicLattice(" << a << ")" << endl;
         }
         bravais = new FaceCenteredCubicLattice(a);
         break;
      case BODY_CENTERED_CUBIC:
         // Body-Centered Cubic Lattice
         // lattice_label = "BCC";
         if ( a <= 0.0 ) { a = 1.0; }
         if ( logging > 0 )
         {
            cout << "Calling BodyCenteredCubicLattice(" << a << ")" << endl;
         }
         bravais = new BodyCenteredCubicLattice(a);
         break;
      case PRIMITIVE_TETRAGONAL:
         // Tetragonal Lattice
         // lattice_label = "TET";
         if ( a <= 0.0 ) { a = 1.0; }
         if ( c <= 0.0 ) { c = 0.5; }
         if ( logging > 0 )
         {
            cout << "Calling TetragonalLattice("
                 << a << ", " << c << ")" << endl;
         }
         bravais = new TetragonalLattice(a, c);
         break;
      case BODY_CENTERED_TETRAGONAL:
         // Body-centered Tetragonal Lattice
         // lattice_label = "BCT";
         if ( a <= 0.0 ) { a = 1.0; }
         if ( c <= 0.0 ) { c = 0.5; }
         if ( logging > 0 )
         {
            cout << "Calling BodyCenteredTetragonalLattice("
                 << a << ", " << c << ")" << endl;
         }
         bravais = new BodyCenteredTetragonalLattice(a, c);
         break;
      case PRIMITIVE_ORTHORHOMBIC:
         // Orthorhombic Lattice
         // lattice_label = "ORC";
         if ( a <= 0.0 ) { a = 0.5; }
         if ( b <= 0.0 ) { b = 0.8; }
         if ( c <= 0.0 ) { c = 1.0; }
         if ( logging > 0 )
         {
            cout << "Calling OrthorhombicLattice("
                 << a << ", " << b << ", " << c << ")" << endl;
         }
         bravais = new OrthorhombicLattice(a, b, c);
         break;
      case FACE_CENTERED_ORTHORHOMBIC:
         // Face-centered Orthorhombic Lattice
         // lattice_label = "ORCF";
         if ( a <= 0.0 ) { a = 0.5; }
         if ( b <= 0.0 ) { b = 0.8; }
         if ( c <= 0.0 ) { c = 1.0; }
         if ( logging > 0 )
         {
            cout << "Calling FaceCenteredOrthorhombicLattice("
                 << a << ", " << b << ", " << c << ")" << endl;
         }
         bravais = new FaceCenteredOrthorhombicLattice(a, b, c);
         break;
      case BODY_CENTERED_ORTHORHOMBIC:
         // Body-centered Orthorhombic Lattice
         // lattice_label = "ORCI";
         if ( a <= 0.0 ) { a = 0.5; }
         if ( b <= 0.0 ) { b = 0.8; }
         if ( c <= 0.0 ) { c = 1.0; }
         if ( logging > 0 )
         {
            cout << "Calling BodyCenteredOrthorhombicLattice("
                 << a << ", " << b << ", " << c << ")" << endl;
         }
         bravais = new BodyCenteredOrthorhombicLattice(a, b, c);
         break;
      case BASE_CENTERED_ORTHORHOMBIC:
         // C-Centered Orthorhombic Lattice
         // lattice_label = "ORCC";
         if ( a <= 0.0 ) { a = 0.5; }
         if ( b <= 0.0 ) { b = 1.0; }
         if ( c <= 0.0 ) { c = 1.0; }
         if ( logging > 0 )
         {
            cout << "Calling BaseCenteredOrthorhombicLattice("
                 << a << ", " << b << ", " << c << ")" << endl;
         }
         bravais = new BaseCenteredOrthorhombicLattice(a, b, c);
         break;
      case PRIMITIVE_HEXAGONAL_PRISM:
         // Hexagonal Prism Lattice
         // lattice_label = "HEX";
         if ( a <= 0.0 ) { a = 1.0; }
         if ( c <= 0.0 ) { c = 1.0; }
         if ( logging > 0 )
         {
            cout << "Calling HexagonalPrismLattice(" << a << ")" << endl;
         }
         bravais = new HexagonalPrismLattice(a, c);
         break;
      case PRIMITIVE_RHOMBOHEDRAL:
         // Rhombohedral Lattice
         // lattice_label = "RHL";
         if (     a <= 0.0 ) { a = 1.0; }
         if ( alpha <= 0.0 ) { alpha = 0.25 * M_PI; }
         if ( logging > 0 )
         {
            cout << "Calling RhombohedralLattice("
                 << a << ", " << alpha << ")" << endl;
         }
         bravais = new RhombohedralLattice(a, alpha);
         break;
      case PRIMITIVE_MONOCLINIC:
         // Monoclinic Lattice
         // lattice_label = "MCL";
         if (     a <= 0.0 ) { a = 1.0; }
         if (     b <= 0.0 ) { b = 1.0; }
         if (     c <= 0.0 ) { c = 1.0; }
         if ( alpha <= 0.0 ) { alpha = 0.25 * M_PI; }
         if ( logging > 0 )
         {
            cout << "Calling MonoclinicLattice("
                 << a << ", " << b << ", " << c << ", " << alpha << ")" << endl;
         }
         bravais = new MonoclinicLattice(a, b, c, alpha);
         break;
      case BASE_CENTERED_MONOCLINIC:
         // C-centered Monoclinic Lattice
         // lattice_label = "MCLC";
         if (     a <= 0.0 ) { a = 1.0; }
         if (     b <= 0.0 ) { b = 1.0; }
         if (     c <= 0.0 ) { c = 1.0; }
         if ( alpha <= 0.0 ) { alpha = 0.25 * M_PI; }
         if ( logging > 0 )
         {
            cout << "Calling BaseCenteredMonoclinicLattice("
                 << a << ", " << b << ", " << c << ", " << alpha << ")" << endl;
         }
         bravais = new BaseCenteredMonoclinicLattice(a, b, c, alpha);
         break;
      case PRIMITIVE_TRICLINIC:
         // Triclinic Lattice
         // lattice_label = "TRI";
         if (     a <= 0.0 ) { a = 1.0; }
         if (     b <= 0.0 ) { b = 1.0; }
         if (     c <= 0.0 ) { c = 1.0; }
         if ( alpha <= 0.0 ) { alpha = 0.25 * M_PI; }
         if (  beta <= 0.0 ) { beta = 0.25 * M_PI; }
         if ( gamma <= 0.0 ) { gamma = 0.25 * M_PI; }
         bravais = new TriclinicLattice(a, b, c, alpha, beta, gamma);
         break;
      default:

         MFEM_ASSERT(false, "Unsupported Lattice Type:  " << lattice_type);

         return NULL;
         break;
   }

   if ( bravais != NULL )
   {
      MFEM_ASSERT( lattice_type == bravais->GetLatticeType(),
                   "Object's lattice type does not match the requested type.");
   }

   return bravais;
}

void
ModeCoefficient::SetAmplitude(double a)
{
   a_ = a;
}

void
ModeCoefficient::SetModeIndices(int n0, int n1, int n2)
{
   n0_ = n0;
   n1_ = n1;
   n2_ = n2;
}

void
ModeCoefficient::SetReciprocalLatticeVectors(const vector<Vector> & rec_vecs)
{
   rec_vecs_.resize(rec_vecs.size());
   for (unsigned int i=0; i<rec_vecs_.size(); i++)
   {
      rec_vecs_[i] = rec_vecs[i];
   }
}

double
ModeCoefficient::Eval(ElementTransformation & T,
                      const IntegrationPoint & ip)
{
   MFEM_ASSERT( rec_vecs_.size() > 0 ,
                "ModeCoefficient Error: Reciprocal Vectors not set");

   double x[3];
   Vector transip(x, 3);

   T.Transform(ip, transip);

   double phase = rec_vecs_[0] * transip * n0_;
   if ( rec_vecs_.size() > 1 )
   {
      phase += rec_vecs_[1] * transip * n1_;
   }
   if ( rec_vecs_.size() > 2 )
   {
      phase += rec_vecs_[2] * transip * n2_;
   }

   return a_ * func_(2.0 * M_PI * phase);
}

void
PhaseCoefficient::SetAmplitude(double a)
{
   a_ = a;
}

void
PhaseCoefficient::SetKappa(const Vector & kappa)
{
   kappa_ = kappa;
}

double
PhaseCoefficient::Eval(ElementTransformation &T,
                       const IntegrationPoint &ip)
{
   double x[3];
   Vector transip(x, 3);

   T.Transform(ip, transip);

   double phase = kappa_ * transip;

   return a_ * func_(phase);
}

BravaisFourierSeries::BravaisFourierSeries(
   const BravaisLattice & bravais,
   ParFiniteElementSpace & fes)
   : br_(NULL),
     bi_(NULL),
     n0_(0),
     n1_(0),
     n2_(0)
{
   br_  = new ParLinearForm(&fes);
   bi_  = new ParLinearForm(&fes);

   vol_ = bravais.GetUnitCellVolume();

   bravais.GetReciprocalLatticeVectors(rec_vecs_);

   coefr_.SetAmplitude( 1.0);
   coefi_.SetAmplitude(-1.0);

   coefr_.SetReciprocalLatticeVectors(rec_vecs_);
   coefi_.SetReciprocalLatticeVectors(rec_vecs_);
}

BravaisFourierSeries::~BravaisFourierSeries()
{
   delete br_;
   delete bi_;
}

void
BravaisFourierSeries::SetMode(int n0, int n1, int n2)
{
   n0_ = n0; n1_ = n1; n2_ = n2;

   coefr_.SetModeIndices(n0_, n1_, n2_);
   coefi_.SetModeIndices(n0_, n1_, n2_);

   this->init();
}

ScalarFourierSeries::ScalarFourierSeries(const BravaisLattice & bravais,
                                         ParFiniteElementSpace & fes)
   : BravaisFourierSeries(bravais, fes),
     Br_(NULL),
     Bi_(NULL)
{
   Br_ = new HypreParVector(&fes);
   Bi_ = new HypreParVector(&fes);
}

ScalarFourierSeries::~ScalarFourierSeries()
{
   delete Br_;
   delete Bi_;
}

void
ScalarFourierSeries::init()
{
   br_->Assemble();
   bi_->Assemble();

   br_->ParallelAssemble(*Br_);
   bi_->ParallelAssemble(*Bi_);
}

void
ScalarFourierSeries::GetCoefficient(HypreParVector & v,
                                    double & a_r, double & a_i)
{
   a_r = InnerProduct(*Br_, v) / vol_;
   a_i = InnerProduct(*Bi_, v) / vol_;
}
/*
VectorFourierSeries::VectorFourierSeries(const BravaisLattice & bravais,
                                         ParFiniteElementSpace & fes)
   : BravaisFourierSeries(bravais, fes),
     vec_(3),
     vecCoef_(vec_, coefr_)
{
   for (int i=0; i<3; i++)
   {
      Br_[i] = new HypreParVector(&fes);
      Bi_[i] = new HypreParVector(&fes);
   }
}

VectorFourierSeries::~VectorFourierSeries()
{
   for (int i=0; i<3; i++)
   {
      delete Br_[i];
      delete Bi_[i];
   }
}

void
VectorFourierSeries::init()
{
   for (int i=0; i<3; i++)
   {
      vec_ = 0.0; vec_[i] = 1.0;

      vecCoef_.SetConstantVector(vec_);

      vecCoef_.SetScalarCoefficient(&coefr_);
      br_->Assemble();
      br_->ParallelAssemble(*Br_[i]);

      vecCoef_.SetScalarCoefficient(&coefi_);
      bi_->Assemble();
      bi_->ParallelAssemble(*Bi_[i]);
   }
}

void
VectorFourierSeries::GetCoefficient(HypreParVector & v,
                                    Vector & a_r, Vector & a_i)
{
   a_r.SetSize(3);
   a_i.SetSize(3);

   for (int i=0; i<3; i++)
   {
      a_r[i] = InnerProduct(*Br_[i], v) / vol_;
      a_i[i] = InnerProduct(*Bi_[i], v) / vol_;
   }
}
*/
H1FourierSeries::H1FourierSeries(const BravaisLattice & bravais,
                                 mfem::common::H1_ParFESpace & fes)
   : ScalarFourierSeries(bravais, fes)
{
   br_->AddDomainIntegrator(new DomainLFIntegrator(coefr_));
   bi_->AddDomainIntegrator(new DomainLFIntegrator(coefi_));
}
/*
HCurlFourierSeries::HCurlFourierSeries(const BravaisLattice & bravais,
                                       mfem::common::ND_ParFESpace & fes)
   : VectorFourierSeries(bravais, fes)
{
   br_->AddDomainIntegrator(new VectorFEDomainLFIntegrator(vecCoef_));
   bi_->AddDomainIntegrator(new VectorFEDomainLFIntegrator(vecCoef_));
}

HDivFourierSeries::HDivFourierSeries(const BravaisLattice & bravais,
                                     mfem::common::RT_ParFESpace & fes)
   : VectorFourierSeries(bravais, fes)
{
   br_->AddDomainIntegrator(new VectorFEDomainLFIntegrator(vecCoef_));
   bi_->AddDomainIntegrator(new VectorFEDomainLFIntegrator(vecCoef_));
}
*/
L2FourierSeries::L2FourierSeries(const BravaisLattice & bravais,
                                 mfem::common::L2_ParFESpace & fes)
   : ScalarFourierSeries(bravais, fes)
{
   br_->AddDomainIntegrator(new DomainLFIntegrator(coefr_));
   bi_->AddDomainIntegrator(new DomainLFIntegrator(coefi_));
}

int toint(int d, double v)
{
   return (int)copysign(round(fabs(v)*pow(10.0,d)),v);
}

double todouble(int d, int v)
{
   return pow(10.0,-d)*v;
}
/*
Mesh *
OldMakePeriodicMesh(Mesh * mesh, const vector<Vector> & trans_vecs, int logging)
{
   int dim  = mesh->Dimension();
   int sdim = mesh->SpaceDimension();

   if ( logging > 0 )
      cout << "Euler Number of Initial Mesh:  "
           << ((dim==3)?mesh->EulerNumber():mesh->EulerNumber2D()) << endl;

   map<int,map<int,map<int,int> > > c2v;
   set<int> v;

   int d = 5;
   Vector xMax(sdim), xMin(sdim);
   xMax = xMin = 0.0;

   for (int be=0; be<mesh->GetNBE(); be++)
   {
      Array<int> dofs;
      mesh->GetBdrElementVertices(be,dofs);

      for (int i=0; i<dofs.Size(); i++)
      {
         double * coords = mesh->GetVertex(dofs[i]);
         for (int j=0; j<sdim; j++)
         {
            xMax[j] = max(xMax[j],coords[j]);
            xMin[j] = min(xMin[j],coords[j]);
         }
         c2v[toint(d,coords[0])][toint(d,coords[1])][toint(d,coords[2])] =
            dofs[i];
         v.insert(dofs[i]);
      }
   }
   if ( logging > 0 )
   {
      cout << "Number of Boundary Vertices:  " << v.size() << endl;

      cout << "xMin: ";
      xMin.Print(cout,sdim);
      cout << "xMax: ";
      xMax.Print(cout,sdim);
   }

   set<int> xc;
   set<int> yc;
   set<int> zc;

   map<int,map<int,map<int,int> > >::iterator mxyz;
   map<int,map<int,int> >::iterator myz;
   map<int,int>::iterator mz;

   if ( logging > 0 )
   {
      int c = 0;
      for (mxyz=c2v.begin(); mxyz!=c2v.end(); mxyz++)
      {
         for (myz=mxyz->second.begin(); myz!=mxyz->second.end(); myz++)
         {
            for (mz=myz->second.begin(); mz!=myz->second.end(); mz++)
            {
               cout << mz->second << ":  "
                    << mxyz->first << " " << myz->first << " " << mz->first
                    << endl;
               c++;
            }
         }
      }
   }

   map<int,int>        slaves;
   map<int,set<int> > masters;

   set<int>::iterator sit;
   for (sit=v.begin(); sit!=v.end(); sit++) { masters[*sit]; }

   Vector at(sdim);

   for (unsigned int i=0; i<trans_vecs.size(); i++)
   {
      int c = 0;
      if ( logging > 0 )
      {
         cout << "trans_vecs = ";
         trans_vecs[i].Print(cout,sdim);
         cout << "toint(trans_vecs) = " << toint(d,trans_vecs[i][0]) << " "
              << toint(d,trans_vecs[i][1]) << " "
              << toint(d,trans_vecs[i][2]) << endl;
      }
      for (mxyz=c2v.begin(); mxyz!=c2v.end(); mxyz++)
      {
         for (myz=mxyz->second.begin(); myz!=mxyz->second.end(); myz++)
         {
            for (mz=myz->second.begin(); mz!=myz->second.end(); mz++)
            {
               double * coords = mesh->GetVertex(mz->second);

               at = trans_vecs[i];
               for (int j=0; j<sdim; j++) { at[j] += coords[j]; }

               int xi = 0, yi = 0, zi = 0;

               xi = toint(d,at[0]);
               if ( sdim > 1 ) { yi = toint(d,at[1]); }
               if ( sdim > 2 ) { zi = toint(d,at[2]); }

               if (c2v.find(xi) == c2v.end()) { continue; }
               if (c2v[xi].find(yi) == c2v[xi].end()) { continue; }
               if (c2v[xi][yi].find(zi) == c2v[xi][yi].end()) { continue; }

               int master = mz->second;
               int slave  = c2v[xi][yi][zi];

               bool mInM = masters.find(master) != masters.end();
               bool sInM = masters.find(slave)  != masters.end();

               if ( mInM && sInM )
               {
                  // Both vertices are currently masters
                  //   Demote "slave" to be a slave of master
                  if ( logging > 0 )
                  {
                     cout << "Both " << master << " and " << slave
                          << " are masters." << endl;
                  }
                  masters[master].insert(slave);
                  slaves[slave] = master;
                  for (sit=masters[slave].begin();
                       sit!=masters[slave].end(); sit++)
                  {
                     masters[master].insert(*sit);
                     slaves[*sit] = master;
                  }
                  masters.erase(slave);
                  if ( masters.find(0) != masters.end() ) { cout << "*** masters contains 0 ***" << endl; }
               }
               else if ( mInM && !sInM )
               {
                  // "master" is already a master and "slave" is already a slave
                  // Make "master" and its slaves slaves of "slave"'s master
                  if ( logging > 0 )
                  {
                     cout << master << " is already a master and " << slave
                          << " is already a slave." << endl;
                  }
                  if ( master != slaves[slave] )
                  {
                     masters[slaves[slave]].insert(master);
                     slaves[master] = slaves[slave];
                     for (sit=masters[master].begin();
                          sit!=masters[master].end(); sit++)
                     {
                        masters[slaves[slave]].insert(*sit);
                        slaves[*sit] = slaves[slave];
                     }
                     masters.erase(master);
                  }
                  if ( masters.find(0) != masters.end() ) { cout << "*** masters contains 0 ***" << endl; }
               }
               else if ( !mInM && sInM )
               {
                  // "master" is currently a slave and
                  // "slave" is currently a master
                  // Make "slave" and its slaves slaves of "master"'s master
                  if ( logging > 0 )
                  {
                     cout << master << " is currently a slave and " << slave
                          << " is currently a master." << endl;
                  }
                  if ( slave != slaves[master] )
                  {
                     masters[slaves[master]].insert(slave);
                     slaves[slave] = slaves[master];
                     for (sit=masters[slave].begin();
                          sit!=masters[slave].end(); sit++)
                     {
                        masters[slaves[master]].insert(*sit);
                        slaves[*sit] = slaves[master];
                     }
                     masters.erase(slave);
                  }
                  if ( masters.find(0) != masters.end() ) { cout << "*** masters contains 0 ***" << endl; }
               }
               else
               {
                  // Both vertices are currently slaves
                  // Make "slave" and its fellow slaves slaves
                  // of "master"'s master
                  if ( logging > 0 )
                  {
                     cout << "Both " << master << " and " << slave
                          << " are slaves." << endl;
                  }
                  int master_of_master = slaves[master];
                  int master_of_slave  = slaves[slave];

                  // Move slave and its fellow slaves to master_of_master
                  if ( slaves[master] != slaves[slave] )
                  {
                     for (sit=masters[master_of_slave].begin();
                          sit!=masters[master_of_slave].end(); sit++)
                     {
                        masters[master_of_master].insert(*sit);
                        slaves[*sit] = master_of_master;
                     }
                     masters.erase(master_of_slave);
                  }
                  // if ( slaves[master] != slaves[slave] )
                  // {
		  //    cout << "foo" << endl;
                  //    masters[slaves[master]].insert(slaves[slave]);
		  //    slaves[slaves[slave]] = slaves[master];
		  // }
		  // for (sit=masters[slaves[slave]].begin();
		  //      sit!=masters[slaves[slave]].end(); sit++)
		  // {
		  //    // cout << "bar " << *sit << endl;
                  //    masters[slaves[master]].insert(*sit);
		  //    slaves[*sit] = slaves[master];
		  // }
		  // if ( slaves[master] != slaves[slave] )
		  // {
		  //    cout << "foobar" << endl;
		  //    masters.erase(slaves[slave]);
		  // }
                  if ( masters.find(0) != masters.end() ) { cout << "*** masters contains 0 ***" << endl; }
               }
               c++;
            }
         }
      }
      if ( logging > 0 )
      {
         cout << "Found " << c << " possible node";
         if ( c != 1 ) { cout << "s"; }
         cout <<" to project." << endl;
      }
   }
   if ( logging > 0 )
   {
      cout << "Number of Master Vertices:  " << masters.size() << endl;
      cout << "Number of Slave Vertices:   " << slaves.size() << endl;
   }

   Array<int> v2v(mesh->GetNV());

   for (int i=0; i<v2v.Size(); i++)
   {
      v2v[i] = i;
   }

   map<int,int>::iterator mit;
   for (mit=slaves.begin(); mit!=slaves.end(); mit++)
   {
      v2v[mit->first] = mit->second;
   }

   Mesh *per_mesh = new Mesh(*mesh, true);

   per_mesh->SetCurvature(1,true);

   // renumber elements
   for (int i = 0; i < per_mesh->GetNE(); i++)
   {
      Element *el = per_mesh->GetElement(i);
      int *v = el->GetVertices();
      int nv = el->GetNVertices();
      for (int j = 0; j < nv; j++)
      {
         v[j] = v2v[v[j]];
      }
   }
   // renumber boundary elements
   for (int i = 0; i < per_mesh->GetNBE(); i++)
   {
      Element *el = per_mesh->GetBdrElement(i);
      int *v = el->GetVertices();
      int nv = el->GetNVertices();
      for (int j = 0; j < nv; j++)
      {
         v[j] = v2v[v[j]];
      }
   }

   per_mesh->RemoveUnusedVertices();
   // per_mesh->RemoveInternalBoundaries();

   if ( logging > 0 )
   {
      cout << "Euler Number of Final Mesh:    "
           << ((dim==3)?per_mesh->EulerNumber():per_mesh->EulerNumber2D())
           << endl;
   }
   return per_mesh;
}
*/
void
MergeMeshNodes(Mesh * mesh, int logging)
{
   int dim  = mesh->Dimension();
   int sdim = mesh->SpaceDimension();

   double tol = 1.0e-8;
   // double dia = -1.0;

   if ( logging > 0 )
      cout << "Euler Number of Initial Mesh:  "
           << ((dim==3)?mesh->EulerNumber() :
               ((dim==2)?mesh->EulerNumber2D() :
                mesh->GetNV() - mesh->GetNE())) << endl;

   vector<int> v2v(mesh->GetNV());

   Vector vd(sdim);

   for (int i = 0; i < mesh->GetNV(); i++)
   {
      Vector vi(mesh->GetVertex(i), sdim);

      v2v[i] = -1;

      for (int j = 0; j < i; j++)
      {
         Vector vj(mesh->GetVertex(j), sdim);
         add(vi, -1.0, vj, vd);

         if ( vd.Norml2() < tol )
         {
            v2v[i] = j;
            break;
         }
      }
      if ( v2v[i] < 0 ) { v2v[i] = i; }
   }

   // renumber elements
   for (int i = 0; i < mesh->GetNE(); i++)
   {
      Element *el = mesh->GetElement(i);
      int *v = el->GetVertices();
      int nv = el->GetNVertices();
      for (int j = 0; j < nv; j++)
      {
         v[j] = v2v[v[j]];
      }
   }
   // renumber boundary elements
   for (int i = 0; i < mesh->GetNBE(); i++)
   {
      Element *el = mesh->GetBdrElement(i);
      int *v = el->GetVertices();
      int nv = el->GetNVertices();
      for (int j = 0; j < nv; j++)
      {
         v[j] = v2v[v[j]];
      }
   }

   mesh->RemoveUnusedVertices();

   if ( logging > 0 )
   {
      cout << "Euler Number of Final Mesh:    "
           << ((dim==3) ? mesh->EulerNumber() :
               ((dim==2) ? mesh->EulerNumber2D() :
                mesh->GetNV() - mesh->GetNE()))
           << endl;
   }
}

Mesh *
MakePeriodicMesh(Mesh * mesh, const vector<Vector> & trans_vecs, int logging)
{
   int dim  = mesh->Dimension();
   int sdim = mesh->SpaceDimension();

   double tol = 1.0e-8;
   double dia = -1.0;

   if ( logging > 0 )
      cout << "Euler Number of Initial Mesh:  "
           << ((dim==3)?mesh->EulerNumber():mesh->EulerNumber2D()) << endl;

   // map<int,map<int,map<int,int> > > c2v;
   set<int> v;
   set<int>::iterator si, sj, sk;
   map<int,int>::iterator mi;
   map<int,set<int> >::iterator msi;

   Vector coord(NULL, sdim);

   // map<int,vector<double> > bnd_vtx;
   // map<int,vector<double> > shft_bnd_vtx;

   // int d = 5;
   Vector xMax(sdim), xMin(sdim), xDiff(sdim);
   xMax = xMin = xDiff = 0.0;

   for (int be=0; be<mesh->GetNBE(); be++)
   {
      Array<int> dofs;
      mesh->GetBdrElementVertices(be,dofs);

      for (int i=0; i<dofs.Size(); i++)
      {
         v.insert(dofs[i]);

         coord.SetData(mesh->GetVertex(dofs[i]));
         for (int j=0; j<sdim; j++)
         {
            xMax[j] = max(xMax[j],coord[j]);
            xMin[j] = min(xMin[j],coord[j]);
         }
      }
   }
   add(xMax, -1.0, xMin, xDiff);
   dia = xDiff.Norml2();

   if ( logging > 0 )
   {
      cout << "Number of Boundary Vertices:  " << v.size() << endl;

      cout << "xMin: ";
      xMin.Print(cout,sdim);
      cout << "xMax: ";
      xMax.Print(cout,sdim);
      cout << "xDiff: ";
      xDiff.Print(cout,sdim);
   }

   if ( logging > 0 )
   {
      for (si=v.begin(); si!=v.end(); si++)
      {
         cout << *si << ": ";
         coord.SetData(mesh->GetVertex(*si));
         coord.Print(cout);
      }
   }

   map<int,int>        slaves;
   map<int,set<int> > masters;

   for (si=v.begin(); si!=v.end(); si++) { masters[*si]; }

   Vector at(sdim);
   Vector dx(sdim);

   for (unsigned int i=0; i<trans_vecs.size(); i++)
   {
      int c = 0;
      if ( logging > 0 )
      {
         cout << "trans_vecs = ";
         trans_vecs[i].Print(cout,sdim);
      }

      for (si=v.begin(); si!=v.end(); si++)
      {
         coord.SetData(mesh->GetVertex(*si));

         add(coord, trans_vecs[i], at);

         for (sj=v.begin(); sj!=v.end(); sj++)
         {
            coord.SetData(mesh->GetVertex(*sj));
            add(at, -1.0, coord, dx);

            if ( dx.Norml2() > dia * tol )
            {
               continue;
            }

            int master = *si;
            int slave  = *sj;

            bool mInM = masters.find(master) != masters.end();
            bool sInM = masters.find(slave)  != masters.end();

            if ( mInM && sInM )
            {
               // Both vertices are currently masters
               //   Demote "slave" to be a slave of master
               if ( logging > 0 )
               {
                  cout << "Both " << master << " and " << slave
                       << " are masters." << endl;
               }
               masters[master].insert(slave);
               slaves[slave] = master;
               for (sk=masters[slave].begin();
                    sk!=masters[slave].end(); sk++)
               {
                  masters[master].insert(*sk);
                  slaves[*sk] = master;
               }
               masters.erase(slave);
            }
            else if ( mInM && !sInM )
            {
               // "master" is already a master and "slave" is already a slave
               // Make "master" and its slaves slaves of "slave"'s master
               if ( logging > 0 )
               {
                  cout << master << " is already a master and " << slave
                       << " is already a slave of " << slaves[slave]
                       << "." << endl;
               }
               if ( master != slaves[slave] )
               {
                  masters[slaves[slave]].insert(master);
                  slaves[master] = slaves[slave];
                  for (sk=masters[master].begin();
                       sk!=masters[master].end(); sk++)
                  {
                     masters[slaves[slave]].insert(*sk);
                     slaves[*sk] = slaves[slave];
                  }
                  masters.erase(master);
               }
            }
            else if ( !mInM && sInM )
            {
               // "master" is currently a slave and
               // "slave" is currently a master
               // Make "slave" and its slaves slaves of "master"'s master
               if ( logging > 0 )
               {
                  cout << master << " is currently a slave of "
                       << slaves[master]<< " and " << slave
                       << " is currently a master." << endl;
               }
               if ( slave != slaves[master] )
               {
                  masters[slaves[master]].insert(slave);
                  slaves[slave] = slaves[master];
                  for (sk=masters[slave].begin();
                       sk!=masters[slave].end(); sk++)
                  {
                     masters[slaves[master]].insert(*sk);
                     slaves[*sk] = slaves[master];
                  }
                  masters.erase(slave);
               }
            }
            else
            {
               // Both vertices are currently slaves
               // Make "slave" and its fellow slaves slaves
               // of "master"'s master
               if ( logging > 0 )
               {
                  cout << "Both " << master << " and " << slave
                       << " are slaves of " << slaves[master] << " and "
                       << slaves[slave] << " respectively." << endl;
               }

               int master_of_master = slaves[master];
               int master_of_slave  = slaves[slave];

               // Move slave and its fellow slaves to master_of_master
               if ( slaves[master] != slaves[slave] )
               {
                  for (sk=masters[master_of_slave].begin();
                       sk!=masters[master_of_slave].end(); sk++)
                  {
                     masters[master_of_master].insert(*sk);
                     slaves[*sk] = master_of_master;
                  }
                  masters.erase(master_of_slave);
                  slaves[master_of_slave] = master_of_master;
               }
            }
            c++;
            break;
         }
      }
      if ( logging > 0 )
      {
         cout << "Found " << c << " possible node";
         if ( c != 1 ) { cout << "s"; }
         cout <<" to project." << endl;
      }
   }
   if ( logging > 0 )
   {
      cout << "Number of Master Vertices:  " << masters.size() << endl;
      cout << "Number of Slave Vertices:   " << slaves.size() << endl;
      cout << "Master to slave mapping:" << endl;
      for (msi=masters.begin(); msi!=masters.end(); msi++)
      {
         cout << msi->first << " ->";
         for (si=msi->second.begin(); si!=msi->second.end(); si++)
         {
            cout << " " << *si;
         }
         cout << endl;
      }
      cout << "Slave to master mapping:" << endl;
      for (mi=slaves.begin(); mi!=slaves.end(); mi++)
      {
         cout << mi->first << " <- " << mi->second << endl;
      }
   }

   Array<int> v2v(mesh->GetNV());

   for (int i=0; i<v2v.Size(); i++)
   {
      v2v[i] = i;
   }

   for (mi=slaves.begin(); mi!=slaves.end(); mi++)
   {
      v2v[mi->first] = mi->second;
   }

   Mesh *per_mesh = new Mesh(*mesh, true);

   per_mesh->SetCurvature(1,true);

   // renumber elements
   for (int i = 0; i < per_mesh->GetNE(); i++)
   {
      Element *el = per_mesh->GetElement(i);
      int *v = el->GetVertices();
      int nv = el->GetNVertices();
      for (int j = 0; j < nv; j++)
      {
         v[j] = v2v[v[j]];
      }
   }
   // renumber boundary elements
   for (int i = 0; i < per_mesh->GetNBE(); i++)
   {
      Element *el = per_mesh->GetBdrElement(i);
      int *v = el->GetVertices();
      int nv = el->GetNVertices();
      for (int j = 0; j < nv; j++)
      {
         v[j] = v2v[v[j]];
      }
   }

   per_mesh->RemoveUnusedVertices();
   // per_mesh->RemoveInternalBoundaries();

   if ( logging > 0 )
   {
      cout << "Euler Number of Final Mesh:    "
           << ((dim==3)?per_mesh->EulerNumber():per_mesh->EulerNumber2D())
           << endl;
   }
   return per_mesh;
}

LatticeCoefficient::LatticeCoefficient(const BravaisLattice & bl,
                                       double frac, double val0, double val1)
   : frac_(frac),
     a0_(val0),
     a1_(val1)
{
   bl.GetTranslationVectors(axes_);
   bl.GetFaceRadii(radii_);

   for (unsigned int i=0; i<axes_.size(); i++)
   {
      axes_[i] /= axes_[i].Norml2();
   }
}

double
LatticeCoefficient::Eval(ElementTransformation &T,
                         const IntegrationPoint &ip)
{
   T.Transform(ip, x_);

   for (unsigned int i=0; i<axes_.size(); i++)
   {
      double xa = x_ * axes_[i];
      xp_ = x_; xp_.Add(-xa, axes_[i]);
      if ( xp_.Norml2() < frac_ * radii_[i] ) { return a1_; }
   }
   return a0_;
}

} // namespace bravais
} // namespace mfem
