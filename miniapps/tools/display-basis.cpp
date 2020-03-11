// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.
//
//      ----------------------------------------------------------------
//      Display Basis Miniapp:  Visualize finite element basis functions
//      ----------------------------------------------------------------
//
// This miniapp visualizes various types of finite element basis functions on a
// single mesh element in 1D, 2D and 3D. The order and the type of finite
// element space can be changed, and the mesh element is either the reference
// one, or a simple transformation of it. Dynamic creation and interaction with
// multiple GLVis windows is demonstrated.
//
// Compile with: make display-basis
//
// Sample runs:  display-basis
//               display_basis -e 2 -b 3 -o 3
//               display-basis -e 5 -b 1 -o 1
//               display-basis -e 3 -b 7 -o 3
//               display-basis -e 3 -b 7 -o 5 -only 16

#include "mfem.hpp"
#include "../common/mfem-common.hpp"
#include <vector>
#include <iostream>

using namespace std;
using namespace mfem;
using namespace mfem::common;

// Data structure used to collect visualization window layout parameters
struct VisWinLayout
{
   int nx;
   int ny;
   int w;
   int h;
};

// Data structure used to define simple coordinate transformations
struct DeformationData
{
   double uniformScale;

   int    squeezeAxis;
   double squeezeFactor;

   int    shearAxis;
   Vector shearVec;
};

/** The Deformation class implements three simple coordinate transformations:
    Uniform Scaling:
      u = a v for a scalar constant 'a'

    Compression or Squeeze (along a coordinate axis):
          / 1/b 0 \            / 1/b 0  0 \     for a scalar constant b
      u = \ 0   b / v   or u = |  0  c  0 | v,  and c = sqrt(b)
                               \  0  0  c /     the axis can also be chosen

    Shear:
      u = v + v_i * s where 's' is the shear vector
                        and 'i' is the shear axis
*/
class Deformation : public VectorCoefficient
{
public:

   enum DefType {INVALID, UNIFORM, SQUEEZE, SHEAR};

   Deformation(int dim, DefType dType, const DeformationData & data)
      : VectorCoefficient(dim), dim_(dim), dType_(dType), data_(data) {}

   void Eval(Vector &v, ElementTransformation &T, const IntegrationPoint &ip);
   using VectorCoefficient::Eval;
private:
   void Def1D(const Vector & u, Vector & v);
   void Def2D(const Vector & u, Vector & v);
   void Def3D(const Vector & u, Vector & v);

   int     dim_;
   DefType dType_;
   const DeformationData & data_;
};

string   elemTypeStr(const Element::Type & eType);
inline bool elemIs1D(const Element::Type & eType);
inline bool elemIs2D(const Element::Type & eType);
inline bool elemIs3D(const Element::Type & eType);

string   basisTypeStr(char bType);
inline bool basisIs1D(char bType);
inline bool basisIs2D(char bType);
inline bool basisIs3D(char bType);

string mapTypeStr(int mType);

int update_basis(vector<socketstream*> & sock, const VisWinLayout & vwl,
                 Element::Type e, char bType, int bOrder, int mType,
                 Deformation::DefType dType, const DeformationData & defData,
                 bool visualization, int &onlySome);

int main(int argc, char *argv[])
{
   // Parse command-line options.
   Element::Type eType  = Element::TRIANGLE;
   char          bType  = 'h';
   int           bOrder = 2;
   int           mType  = 0;

   int eInt = -1;
   int bInt = -1;

   VisWinLayout vwl;
   vwl.nx = 5;
   vwl.ny = 3;
   vwl.w  = 250;
   vwl.h  = 250;

   Deformation::DefType dType = Deformation::INVALID;
   DeformationData defData;

   bool visualization = true;
   int onlySome = -1;

   vector<socketstream*> sock;

   OptionsParser args(argc, argv);
   args.AddOption(&eInt, "-e", "--elem-type",
                  "Element Type: (1-Segment, 2-Triangle, 3-Quadrilateral, "
                  "4-Tetrahedron, 5-Hexahedron)");
   args.AddOption(&bInt, "-b", "--basis-type",
                  "Basis Function Type (0-H1, 1-Nedelec, 2-Raviart-Thomas, "
                  "3-L2, 4-Fixed Order Cont.,\n\t5-Gaussian Discontinuous (2D),"
                  " 6-Crouzeix-Raviart, 7-Serendipity)");
   args.AddOption(&bOrder, "-o", "--order", "Basis function order");
   args.AddOption(&vwl.nx, "-nx", "--num-win-x",
                  "Number of Viz windows in X");
   args.AddOption(&vwl.ny, "-ny", "--num-win-y",
                  "Number of Viz windows in y");
   args.AddOption(&vwl.w, "-w", "--width",
                  "Width of Viz windows");
   args.AddOption(&vwl.h, "-h", "--height",
                  "Height of Viz windows");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&onlySome, "-only", "--onlySome",
                  "Only view 10 dofs, starting with the specified one.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   {
      args.PrintOptions(cout);
   }
   if ( eInt > 0 && eInt < 6 )
   {
      eType = (Element::Type)eInt;
   }
   switch (bInt)
   {
      case 0:
         bType = 'h';
         break;
      case 1:
         bType = 'n';
         break;
      case 2:
         bType = 'r';
         break;
      case 3:
         bType = 'l';
         break;
      case 4:
         bType = 'f';
         break;
      case 5:
         bType = 'g';
         break;
      case 6:
         bType = 'c';
         break;
      case 7:
         bType = 's';
         break;
      default:
         bType = 'h';
   }

   // Collect user input
   bool print_char = true;
   while (true)
   {
      if (print_char)
      {
         cout << endl;
         cout << "Element Type:          " << elemTypeStr(eType) << endl;
         cout << "Basis Type:            " << basisTypeStr(bType) << endl;;
         cout << "Basis function order:  " << bOrder << endl;
         cout << "Map Type:              " << mapTypeStr(mType) << endl;
      }
      if ( update_basis(sock, vwl, eType, bType, bOrder, mType,
                        dType, defData, visualization, onlySome) )
      {
         cerr << "Invalid combination of basis info (try again)" << endl;
      }

      if (!visualization) { break; }

      print_char = false;
      cout << endl;
      cout << "What would you like to do?\n"
           "q) Quit\n"
           "c) Close Windows and Quit\n"
           "e) Change Element Type\n"
           "b) Change Basis Type\n";
      if ( bType == 'h' || bType == 'p' || bType == 'n' || bType == 'r' ||
           bType == 'l' || bType == 'f' || bType == 'g' || bType == 's')
      {
         cout << "o) Change Basis Order\n";
      }
      // The following is disabled pending updates to GLVis
      if ( bType == 'l' && false )
      {
         cout << "m) Change Map Type\n";
      }
      cout << "t) Transform Element\n";
      cout << "--> " << flush;
      char mk;
      cin >> mk;

      if (mk == 'q')
      {
         break;
      }
      if (mk == 'c')
      {
         for (unsigned int i=0; i<sock.size(); i++)
         {
            *sock[i] << "keys q";
         }
         break;
      }
      if (mk == 'e')
      {
         eInt = 0;
         cout << "valid element types:\n";
         if ( basisIs1D(bType) )
         {
            cout <<
                 "1) Segment\n";
         }
         if ( basisIs2D(bType) )
         {
            cout <<
                 "2) Triangle\n"
                 "3) Quadrilateral\n";
         }
         if ( basisIs3D(bType) )
         {
            cout <<
                 "4) Tetrahedron\n"
                 "5) Hexahedron\n";
         }
         cout << "enter new element type --> " << flush;
         cin >> eInt;
         if ( eInt <= 0 || eInt > 5 )
         {
            cout << "invalid element type \"" << eInt << "\"" << endl << flush;
         }
         else if ( (elemIs1D((Element::Type)eInt) && basisIs1D(bType)) ||
                   (elemIs2D((Element::Type)eInt) && basisIs2D(bType)) ||
                   (elemIs3D((Element::Type)eInt) && basisIs3D(bType)) )
         {
            if ( (elemIs1D((Element::Type)eInt) && !elemIs1D(eType)) ||
                 (elemIs2D((Element::Type)eInt) && !elemIs2D(eType)) ||
                 (elemIs3D((Element::Type)eInt) && !elemIs3D(eType)) )
            {
               dType = Deformation::INVALID;
            }
            eType = (Element::Type)eInt;

            print_char = true;
         }
         else
         {
            cout << "invalid element type \"" << eInt <<
                 "\" for basis type \"" << basisTypeStr(bType) << "\"." << endl;
         }
      }
      if (mk == 'b')
      {
         char bChar = 0;
         cout << "valid basis types:\n";
         cout << "h) H1 Finite Element\n";
         cout << "p) H1 Positive Finite Element\n";
         if ( elemIs2D(eType) || elemIs3D(eType) )
         {
            cout << "s) H1 Serendipity Finite Element\n";
            cout << "n) Nedelec Finite Element\n";
            cout << "r) Raviart-Thomas Finite Element\n";
         }
         cout << "l) L2 Finite Element\n";
         if ( elemIs1D(eType) || elemIs2D(eType) )
         {
            cout << "c) Crouzeix-Raviart Finite Element\n";
         }
         cout << "f) Fixed Order Continuous Finite Element\n";
         if ( elemIs2D(eType) )
         {
            cout << "g) Gauss Discontinuous Finite Element\n";
         }
         cout << "enter new basis type --> " << flush;
         cin >> bChar;
         if (bChar == 'h' || bChar == 'p' || bChar == 'l' || bChar == 'f' ||
             bChar == 's' ||
             ((bChar == 'n' || bChar == 'r') && (elemIs2D(eType) || elemIs3D(eType))) ||
             (bChar == 'c' && (elemIs1D(eType) || elemIs2D(eType))) ||
             (bChar == 'g' && elemIs2D(eType)))
         {
            bType = bChar;
            if ( bType == 'h' )
            {
               mType = FiniteElement::VALUE;
            }
            else if ( bType == 'p' )
            {
               mType = FiniteElement::VALUE;
            }
            else if (bType == 's')
            {
               mType = FiniteElement::VALUE;
            }
            else if ( bType == 'n' )
            {
               mType = FiniteElement::H_CURL;
            }
            else if ( bType == 'r' )
            {
               mType = FiniteElement::H_DIV;
            }
            else if ( bType == 'l' )
            {
               if ( mType != FiniteElement::VALUE &&
                    mType != FiniteElement::INTEGRAL )
               {
                  mType = FiniteElement::VALUE;
               }
            }
            else if ( bType == 'c' )
            {
               bOrder = 1;
               mType  = FiniteElement::VALUE;
            }
            else if ( bType == 'f' )
            {
               if ( bOrder < 1 || bOrder > 3)
               {
                  bOrder = 1;
               }
               mType  = FiniteElement::VALUE;
            }
            else if ( bType == 'g' )
            {
               if ( bOrder < 1 || bOrder > 2)
               {
                  bOrder = 1;
               }
               mType  = FiniteElement::VALUE;
            }
            print_char = true;
         }
         else
         {
            cout << "invalid basis type \"" << bChar << "\"." << endl;
         }
      }
      if (mk == 'm' && bType == 'l')
      {
         int mInt = 0;
         cout << "valid map types:\n"
              "0) VALUE\n"
              "1) INTEGRAL\n";
         cout << "enter new map type --> " << flush;
         cin >> mInt;
         if (mInt >=0 && mInt <= 1)
         {
            mType = mInt;
            print_char = true;
         }
         else
         {
            cout << "invalid map type \"" << mInt << "\"." << endl;
         }
      }
      if (mk == 'o')
      {
         int oInt = 1;
         int oMin = ( bType == 'h' || bType == 'p' || bType == 'n' ||
                      bType == 'f' || bType == 'g' || bType == 's')?1:0;
         int oMax = -1;
         switch (bType)
         {
            case 'g':
               oMax = 2;
               break;
            case 'f':
               oMax = 3;
               break;
            default:
               oMax = -1;
         }
         cout << "basis function order must be >= " << oMin;
         if ( oMax >= 0 )
         {
            cout << " and <= " << oMax;
         }
         cout << endl;
         cout << "enter new basis function order --> " << flush;
         cin >> oInt;
         if ( oInt >= oMin && oInt <= (oMax>=0)?oMax:oInt )
         {
            bOrder = oInt;
            print_char = true;
         }
         else
         {
            cout << "invalid basis order \"" << oInt << "\"." << endl;
         }
      }
      if (mk == 't')
      {
         cout << "transformation options:\n";
         cout << "r) reset to reference element\n";
         cout << "u) uniform scaling\n";
         if ( elemIs2D(eType) || elemIs3D(eType) )
         {
            cout << "c) compression\n";
            cout << "s) shear\n";
         }
         cout << "enter transformation type --> " << flush;
         char tk;
         cin >> tk;
         if (tk == 'r')
         {
            dType = Deformation::INVALID;
         }
         else if (tk == 'u')
         {
            cout << "enter scaling constant --> " << flush;
            cin >> defData.uniformScale;
            if ( defData.uniformScale > 0.0 )
            {
               dType = Deformation::UNIFORM;
            }
         }
         else if (tk == 'c' && !elemIs1D(eType))
         {
            int dim = elemIs2D(eType)?2:3;
            cout << "enter compression factor --> " << flush;
            cin >> defData.squeezeFactor;
            cout << "enter compression axis (0-" << dim-1 << ") --> " << flush;
            cin >> defData.squeezeAxis;

            if ( defData.squeezeFactor > 0.0 &&
                 (defData.squeezeAxis >= 0 && defData.squeezeAxis < dim))
            {
               dType = Deformation::SQUEEZE;
            }
         }
         else if (tk == 's' && !elemIs1D(eType))
         {
            int dim = elemIs2D(eType)?2:3;
            cout << "enter shear vector (components separated by spaces) --> "
                 << flush;
            defData.shearVec.SetSize(dim);
            for (int i=0; i<dim; i++)
            {
               cin >> defData.shearVec[i];
            }
            cout << "enter shear axis (0-" << dim-1 << ") --> " << flush;
            cin >> defData.shearAxis;

            if ( defData.shearAxis >= 0 && defData.shearAxis < dim )
            {
               dType = Deformation::SHEAR;
            }
         }
      }
   }

   // Cleanup
   for (unsigned int i=0; i<sock.size(); i++)
   {
      delete sock[i];
   }

   // Exit
   return 0;
}

string elemTypeStr(const Element::Type & eType)
{
   switch (eType)
   {
      case Element::POINT:
         return "POINT";
      case Element::SEGMENT:
         return "SEGMENT";
      case Element::TRIANGLE:
         return "TRIANGLE";
      case Element::QUADRILATERAL:
         return "QUADRILATERAL";
      case Element::TETRAHEDRON:
         return "TETRAHEDRON";
      case Element::HEXAHEDRON:
         return "HEXAHEDRON";
      default:
         return "INVALID";
   };
}

bool
elemIs1D(const Element::Type & eType)
{
   return eType == Element::SEGMENT;
}

bool
elemIs2D(const Element::Type & eType)
{
   return eType == Element::TRIANGLE || eType == Element::QUADRILATERAL;
}

bool
elemIs3D(const Element::Type & eType)
{
   return eType == Element::TETRAHEDRON || eType == Element::HEXAHEDRON;
}

string
basisTypeStr(char bType)
{
   switch (bType)
   {
      case 'h':
         return "Continuous (H1)";
      case 'p':
         return "Continuous Positive (H1)";
      case 's':
         return "Continuous Serendipity (H1)";
      case 'n':
         return "Nedelec";
      case 'r':
         return "Raviart-Thomas";
      case 'l':
         return "Discontinuous (L2)";
      case 'f':
         return "Fixed Order Continuous";
      case 'g':
         return "Gaussian Discontinuous";
      case 'c':
         return "Crouzeix-Raviart";
      default:
         return "INVALID";
   };
}

bool
basisIs1D(char bType)
{
   return bType == 'h' || bType == 'p' || bType == 'l' || bType == 'c' ||
          bType == 'f';
}

bool
basisIs2D(char bType)
{
   return bType == 'h' || bType == 'p' || bType == 'n' || bType == 'r' ||
          bType == 'l' || bType == 'c' || bType == 'f' || bType == 'g' ||
          bType == 's';
}

bool
basisIs3D(char bType)
{
   return bType == 'h' || bType == 'p' || bType == 'n' || bType == 'r' ||
          bType == 'f' || bType == 'l';
}

string
mapTypeStr(int mType)
{
   switch (mType)
   {
      case FiniteElement::VALUE:
         return "VALUE";
      case FiniteElement::H_CURL:
         return "H_CURL";
      case FiniteElement::H_DIV:
         return "H_DIV";
      case FiniteElement::INTEGRAL:
         return "INTEGRAL";
      default:
         return "INVALID";
   }
}

void
Deformation::Eval(Vector &v, ElementTransformation &T,
                  const IntegrationPoint &ip)
{
   Vector u(dim_);
   T.Transform(ip, u);

   switch (dim_)
   {
      case 1:
         Def1D(u, v);
         break;
      case 2:
         Def2D(u, v);
         break;
      case 3:
         Def3D(u, v);
         break;
   }
}

void
Deformation::Def1D(const Vector & u, Vector & v)
{
   v = u;
   if ( dType_ == UNIFORM )
   {
      v *= data_.uniformScale;
   }
}

void
Deformation::Def2D(const Vector & u, Vector & v)
{
   switch (dType_)
   {
      case UNIFORM:
         v = u;
         v *= data_.uniformScale;
         break;
      case SQUEEZE:
         v = u;
         v[ data_.squeezeAxis     ] /= data_.squeezeFactor;
         v[(data_.squeezeAxis+1)%2] *= data_.squeezeFactor;
         break;
      case SHEAR:
         v = u;
         v.Add(v[data_.shearAxis], data_.shearVec);
         break;
      default:
         v = u;
   }
}

void
Deformation::Def3D(const Vector & u, Vector & v)
{
   switch (dType_)
   {
      case UNIFORM:
         v = u;
         v *= data_.uniformScale;
         break;
      case SQUEEZE:
         v = u;
         v[ data_.squeezeAxis     ] /= data_.squeezeFactor;
         v[(data_.squeezeAxis+1)%2] *= sqrt(data_.squeezeFactor);
         v[(data_.squeezeAxis+2)%2] *= sqrt(data_.squeezeFactor);
         break;
      case SHEAR:
         v = u;
         v.Add(v[data_.shearAxis], data_.shearVec);
         break;
      default:
         v = u;
   }
}

int
update_basis(vector<socketstream*> & sock,  const VisWinLayout & vwl,
             Element::Type e, char bType, int bOrder, int mType,
             Deformation::DefType dType, const DeformationData & defData,
             bool visualization, int &onlySome)
{
   bool vec = false;

   Mesh *mesh;
   ElementMeshStream imesh(e);
   if (!imesh)
   {
      {
         cerr << "\nProblem with meshstream object\n" << endl;
      }
      return 2;
   }
   mesh = new Mesh(imesh, 1, 1);
   int dim = mesh->Dimension();

   if ( dType != Deformation::INVALID )
   {
      Deformation defCoef(dim, dType, defData);
      mesh->Transform(defCoef);
   }

   FiniteElementCollection * FEC = NULL;
   switch (bType)
   {
      case 'h':
         FEC = new H1_FECollection(bOrder, dim);
         vec = false;
         break;
      case 'p':
         FEC = new H1Pos_FECollection(bOrder, dim);
         vec = false;
         break;
      case 's':
         if (bOrder == 1)
         {
            FEC = new H1_FECollection(bOrder, dim);
         }
         else
         {
            FEC = new H1Ser_FECollection(bOrder, dim);
         }
         vec = false;
         break;
      case 'n':
         FEC = new ND_FECollection(bOrder, dim);
         vec = true;
         break;
      case 'r':
         FEC = new RT_FECollection(bOrder-1, dim);
         vec = true;
         break;
      case 'l':
         FEC = new L2_FECollection(bOrder, dim, BasisType::GaussLegendre,
                                   mType);
         vec = false;
         break;
      case 'c':
         FEC = new CrouzeixRaviartFECollection();
         break;
      case 'f':
         if ( bOrder == 1 )
         {
            FEC = new LinearFECollection();
         }
         else if ( bOrder == 2 )
         {
            FEC = new QuadraticFECollection();
         }
         else if ( bOrder == 3 )
         {
            FEC = new CubicFECollection();
         }
         break;
      case 'g':
         if ( bOrder == 1 )
         {
            FEC = new GaussLinearDiscont2DFECollection();
         }
         else if ( bOrder == 2 )
         {
            FEC = new GaussQuadraticDiscont2DFECollection();
         }
         break;
   }
   if ( FEC == NULL)
   {
      delete mesh;
      return 1;
   }

   FiniteElementSpace FESpace(mesh, FEC);

   int ndof = FESpace.GetVSize();

   Array<int> vdofs;
   FESpace.GetElementVDofs(0,vdofs);

   char vishost[] = "localhost";
   int  visport   = 19916;

   int offx = vwl.w+10, offy = vwl.h+45; // window offsets

   for (unsigned int i=0; i<sock.size(); i++)
   {
      *sock[i] << "keys q";
      delete sock[i];
   }

   sock.resize(ndof);
   for (int i=0; i<ndof; i++)
   {
      sock[i] = new socketstream; sock[i]->precision(8);
   }

   GridFunction ** x = new GridFunction*[ndof];
   for (int i=0; i<ndof; i++)
   {
      x[i]  = new GridFunction(&FESpace);
      *x[i] = 0.0;
      if ( vdofs[i] < 0 )
      {
         (*x[i])(-1-vdofs[i]) = -1.0;
      }
      else
      {
         (*x[i])(vdofs[i]) = 1.0;
      }
   }

   int ref = 0;
   int exOrder = 0;
   if ( bType == 'n' ) { exOrder++; }
   if ( bType == 'r' ) { exOrder += 2; }
   while ( 1<<ref < bOrder + exOrder || ref == 0 )
   {
      mesh->UniformRefinement();
      FESpace.Update();

      for (int i=0; i<ndof; i++)
      {
         x[i]->Update();
      }
      ref++;
   }

   int stopAt = ndof;
   if (ndof > 25 && onlySome == -1)
   {
      cout << endl;
      cout << "There are more than 25 windows to open.\n"
           << "Only showing Dofs 1-10 to avoid crashing.\n"
           << "Use the option -only N to show Dofs N to N+9 instead.\n";
      onlySome = 1;
   }
   for (int i = 0; i < stopAt; i++)
   {
      if (i ==0 && onlySome > 0 && onlySome <ndof)
      {
         i = onlySome-1;
         stopAt = min(ndof,onlySome+9);
      }

      ostringstream oss;
      oss << "DoF " << i + 1;
      if (visualization)
      {
         VisualizeField(*sock[i], vishost, visport, *x[i], oss.str().c_str(),
                        (i % vwl.nx) * offx, ((i / vwl.nx) % vwl.ny) * offy,
                        vwl.w, vwl.h,
                        "aaAc", vec);
      }
   }

   for (int i=0; i<ndof; i++)
   {
      delete x[i];
   }
   delete [] x;

   delete FEC;
   delete mesh;

   return 0;
}
