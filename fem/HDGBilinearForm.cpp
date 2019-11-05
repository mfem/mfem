// Implementation of class HDGBilinearForm

#include "HDGBilinearForm.hpp"
#include "fem.hpp"
#include <cmath>
#include <fstream>
#include <iostream>

using namespace std;

namespace mfem
{


HDGBilinearForm::HDGBilinearForm (Array<FiniteElementSpace*> &_fes1,
                                  Array<FiniteElementSpace*> &_fes2,
                                  bool _parallel)
{
   NInteriorFES = _fes1.Size();
   NBdrFES = _fes2.Size();
   fes1.SetSize(NInteriorFES);
   fes2.SetSize(NBdrFES);
   for(int fes=0; fes < NInteriorFES; fes++)
   {
            fes1[fes] = _fes1[fes];
   }
   for(int fes=0; fes < NBdrFES; fes++)
   {
      fes2[fes] = _fes2[fes];
   }

   parallel = _parallel;
   mat = NULL;
   rhs_SC = NULL;
   el_to_face = NULL;

   A_data = NULL; B_data = NULL;
   elements_A = elements_B = 0;
}

HDGBilinearForm::HDGBilinearForm(FiniteElementSpace *_fes1,
                                 FiniteElementSpace *_fes2,
                                 bool _parallel)
{
   NInteriorFES = 1;
   NBdrFES = 1;
   fes1.SetSize(NInteriorFES);
   fes2.SetSize(NBdrFES);

   fes1[0] = _fes1;
   
   fes2[0] = _fes2;
   
   parallel = _parallel;
   mat = NULL;
   rhs_SC = NULL;
   el_to_face = NULL;

   A_data = NULL; B_data = NULL;
   elements_A = elements_B = 0;
}

HDGBilinearForm::HDGBilinearForm(FiniteElementSpace *_fes1,
                                 FiniteElementSpace *_fes2,
                                 FiniteElementSpace *_fes3,
                                 bool _parallel)
{
   NInteriorFES = 2;
   NBdrFES = 1;
   fes1.SetSize(NInteriorFES);
   fes2.SetSize(NBdrFES);

   fes1[0] = _fes1;
   fes1[1] = _fes2;
   
   fes2[0] = _fes3;
   
   parallel = _parallel;
   mat = NULL;
   rhs_SC = NULL;
   el_to_face = NULL;

   A_data = NULL; B_data = NULL;
   elements_A = elements_B = 0;
}

HDGBilinearForm::~HDGBilinearForm()
{
   delete mat;
   delete rhs_SC;
   delete A_data;
   delete B_data;

   int k;
   for (k=0; k < hdg_dbfi.Size(); k++) { delete hdg_dbfi[k]; }
   for (k=0; k < hdg_fbfi.Size(); k++) { delete hdg_fbfi[k]; }
}

void HDGBilinearForm::Finalize (int skip_zeros)
{
    mat->Finalize(skip_zeros);
}

void HDGBilinearForm::AddHDGDomainIntegrator(BilinearFormIntegrator * bfi)
{
   hdg_dbfi.Append (bfi);
}

void HDGBilinearForm::AddHDGFaceIntegrator(BilinearFormIntegrator * bfi)
{
   hdg_fbfi.Append (bfi);
}

void HDGBilinearForm::Update(FiniteElementSpace *nfes1, FiniteElementSpace *nfes2)
{
   if (nfes1)
   {
      fes1 = nfes1;
      fes2 = nfes2;
   }

   delete mat;
   mat = NULL;

   delete rhs_SC;
   rhs_SC = NULL;

   delete [] A_data;
   A_data = NULL;

   delete [] B_data;
   B_data = NULL;

   elements_A = elements_B = 0;

}



void HDGBilinearForm::GetInteriorVDofs(int i, Array<int> &vdofs) const
{
   vdofs.SetSize(0);
   Array<int> vdofs_fes;
   for(int fes=0; fes < NInteriorFES; fes++)
   {
      fes1[fes]->GetElementVDofs(i, vdofs_fes);
      vdofs.Append(vdofs_fes);
   }
}

void HDGBilinearForm::GetInteriorSubVector(const Array<GridFunction*> &rhs_vector,
                                           int i, int ndof, Vector &SubVector) const
{
   SubVector.SetSize(ndof);
   Vector LocalVector;
   Array<int> vdofs_fes;

   int counter = 0;
   for(int fes=0; fes < NInteriorFES; fes++)
   {
      fes1[fes]->GetElementVDofs(i, vdofs_fes);
      rhs_vector[fes]->GetSubVector(vdofs_fes, LocalVector);
      for(int k = 0; k<vdofs_fes.Size(); k++)
         SubVector(counter + k) = LocalVector(k);

      counter += vdofs_fes.Size();
   }
}

void HDGBilinearForm::SetInteriorSubVector(Array<GridFunction*> &rhs_vector,
                                           int i, int ndof, Vector &SubVector)
{
   SubVector.SetSize(ndof);
   Vector LocalVector;
   Array<int> vdofs_fes;

   int counter = 0;
   for(int fes=0; fes < NInteriorFES; fes++)
   {
      fes1[fes]->GetElementVDofs(i, vdofs_fes);
      LocalVector.SetSize(vdofs_fes.Size());
      for(int k = 0; k<vdofs_fes.Size(); k++)
         LocalVector(k) = SubVector(counter + k);

      rhs_vector[fes]->SetSubVector(vdofs_fes, LocalVector);

      counter += vdofs_fes.Size();
   }
}

void HDGBilinearForm::GetBdrVDofs(int i, Array<int> &vdofs) const
{
   vdofs.SetSize(0);
   Array<int> vdofs_fes;
   for(int fes=0; fes < NBdrFES; fes++)
   {
      fes2[fes]->GetFaceVDofs(i, vdofs_fes);
      vdofs.Append(vdofs_fes);
   }
}

// compute all the domain based integrals in one loop over the quadrature nodes
void HDGBilinearForm::compute_domain_integrals(const int elem, DenseMatrix *A_local)
{
    // get the element transformation and the finite elements for the variables
    ElementTransformation *eltrans;
    eltrans = fes1[0]->GetElementTransformation(elem);
    const FiniteElement &fe_u1 = *fes1[0]->GetFE(elem);

    switch (NInteriorFES)
    {
    case 1:
    {
        // compute the integrals
        hdg_dbfi[0]->AssembleElementMatrix(fe_u1, *eltrans, elemmat1);
        break;
    }
    case 2:
    {
        const FiniteElement &fe_u2 = *fes1[1]->GetFE(elem);
        // compute the integrals
        hdg_dbfi[0]->AssembleElementMatrix2FES(fe_u1, fe_u2, *eltrans, elemmat1);
        break;
    }
    default:
    {
        mfem_error("HDGBilinearForm::compute_domain_integrals is defined only for 1 or 2 interior FES");
        break;
    }
    }

    // add them to the right matrices
    A_local->Add(1.0, elemmat1);
}

// Compute the face integrals for B and C
/* They both contain only TraceInteriorFaceIntegrators and TraceBoundaryFaceIntegrators.
 * The bool onlyB should be false when creating the Schur complement, true when reconstructing u from ubar.
*/
void HDGBilinearForm::compute_face_integrals(const int elem, const int edge,
                                             const int isboundary,
                                             const bool onlyB,
                                             DenseMatrix *A_local,
                                             DenseMatrix *B_local,
                                             DenseMatrix *C_local,
                                             DenseMatrix *D_local)
{
    Mesh *mesh = fes1[0] -> GetMesh();
    FaceElementTransformations *tr;

    // see comment above
    tr = mesh->GetFaceElementTransformations(edge);

    const FiniteElement &trial_face_fe = *fes2[0]->GetFaceElement(edge);
    const FiniteElement &testu1_fe1 = *fes1[0]->GetFE(tr->Elem1No);

    // this is a setup for the case of the boundary
    const FiniteElement &testu1_fe2 = *fes1[0]->GetFE(tr->Elem1No);
    if (isboundary == -1)
    {
        // if not a boundary edge use the neighbouring element
        const FiniteElement &testu1_fe2 = *fes1[0]->GetFE(tr->Elem2No);
    }
    // Compute the integrals depending on which element do we need to use
    int elem_1or2 = 1 + (tr->Elem2No == elem);
    switch (NInteriorFES)
    {
    case 1:
    {
        hdg_fbfi[0]->AssembleFaceMatrixOneElement1and1FES(testu1_fe1, testu1_fe2,
                            trial_face_fe, *tr,
                            elem_1or2, onlyB,
                            elemmat1, elemmat2,
                            elemmat3, elemmat4);
        break;
    }
    case 2:
    {
        const FiniteElement &testu2_fe1 = *fes1[1]->GetFE(tr->Elem1No);
        // this is a setup for the case of the boundary
        const FiniteElement &testu2_fe2 = *fes1[1]->GetFE(tr->Elem1No);
        if (isboundary == -1)
        {
            // if not a boundary edge use the neighbouring element
            const FiniteElement &testu2_fe2 = *fes1[0]->GetFE(tr->Elem2No);
        }
        hdg_fbfi[0]->AssembleFaceMatrixOneElement2and1FES(testu1_fe1, testu1_fe2,
                            testu2_fe1, testu2_fe2,
                            trial_face_fe,
                            *tr, elem_1or2, onlyB,
                            elemmat1, elemmat2,
                            elemmat3, elemmat4);
        break;
    }
    default:
    {
        mfem_error("HDGBilinearForm::compute_face_integrals is defined only for 1 or 2 interior FES");
        break;
    }

    }

    // If it is not reconstruction C and D are needed
    A_local->Add(1.0, elemmat1);
    B_local->Add(1.0, elemmat2);
    if (!onlyB)
    {
        C_local->Add(1.0, elemmat3);
        D_local->Add(1.0, elemmat4);
    }

}

/* To allocate the sparse matrix and the right hand side vector and to create the Edge_to_be and el_to_face tables.
 * This is also called for the parallel, since these information are important on every processor
 * Edge_to_be is an Array with size of the number of edges.
 * The entry Edge_to_be[i] is -1 if the i-th face is interior or shared. It is greater then -1 if the i-th face lies on the boundary
 * Moreover, Edge_to_be[i] = n means that the n-th boundary face is the i-th face.
 * el_to_faces has number of element rows and the i-th row contains the faces of the i-th element
 */
void HDGBilinearForm::Allocate(const Array<int> &bdr_attr_is_ess,
        const double memA, const double memB)
{
   Mesh *mesh = fes1[0] -> GetMesh();
   mesh->GetEdgeToBdrFace(Edge_to_Be);


   // Get the list of the faces of every element
   if (mesh->Dimension() == 2)
       el_to_face = new Table(mesh->ElementToEdgeTable());
   else if (mesh->Dimension() == 3)
       el_to_face = new Table(mesh->ElementToFaceTable());


#ifdef MFEM_USE_MPI
   int nedge = Edge_to_Be.Size();

   Edge_to_SharedEdge.SetSize(nedge);
   Edge_to_SharedEdge = -1;
   // ExchangeFaceNbrData to be able to use shared faces
   ParFiniteElementSpace* pfes1 = dynamic_cast<ParFiniteElementSpace*>(fes1[0]);

   if (parallel)
   {
       pfes1->ExchangeFaceNbrData();
       int nsharedfaces = pfes1 -> GetParMesh()->GetNSharedFaces();

       // Create an array to identify the shared faces. The entry is one of the face is not shared,
       // otherwise, is gives the number of the face in the shared face list, so that GetSharedFaceTransformation
       // can be used.
       for (int i = 0; i < nsharedfaces; i++)
       {
           Edge_to_SharedEdge[pfes1 -> GetParMesh()->GetSharedFace(i)] = i;
       }
   }
#endif

   if (mat == NULL)
   {
       mat = new SparseMatrix(fes2[0]->GetVSize());
   }

   if (rhs_SC == NULL)
   {
       rhs_SC = new Vector(fes2[0]->GetVSize());
       *rhs_SC = 0.0;
   }

   fes2[0]->GetEssentialVDofs(bdr_attr_is_ess, ess_dofs);

   elements_A = (int)(memA * fes1[0]->GetNE());
   elements_B = (int)(memB * fes1[0]->GetNE());

//   std::cout << "elements_B " << elements_B << std::endl << std::flush;

   // Set the offset vectors
   A_offsets.SetSize(elements_A+1);
   B_offsets.SetSize(elements_B+1);
   A_offsets[0] = 0;
   B_offsets[0] = 0;

   Array<int> vdofs_u, fcs;
   int ndof_u;

   // loop over the elements to find the offset entries
   for(int i=0; i< fes1[0]->GetNE(); i++)
   {
       // Get the local number of dof for u
       GetInteriorVDofs(i, vdofs_u);
       ndof_u  = vdofs_u.Size();

       // A will have the size (ndof_q + ndof_u)*(ndof_q + ndof_u)
       // The next offset entry can be set
       if (i < elements_A)
       {
           A_offsets[i+1] = A_offsets[i] + ndof_u * ndof_u;
       }

       // To find the next offset entry of B the local number of dofs
       // are needed
       el_to_face->GetRow(i, fcs);
       int no_faces = fcs.Size();
       int ndof_edge_all = 0;

       // Sum up the face dofs for all faces
       if (i < elements_B)
       {
           for(int edge1=0; edge1<no_faces; edge1++)
           {
               GetBdrVDofs(fcs[edge1], vdofs3);
               ndof_edge_all += vdofs3.Size();
           }

           B_offsets[i+1] = B_offsets[i] + ndof_u*ndof_edge_all;
       }

       // If i >= elements_A then i >= elements_B also, so the for loop can be cancelled
       if (i >= elements_A)
       {
           break;
       }
   }

   // Create A_data and B_data as a vector with the proper size
   delete A_data;
   A_data = new double[A_offsets[elements_A]];
   delete B_data;
   B_data = new double[B_offsets[elements_B]];

}

void HDGBilinearForm::AssembleSC(GridFunction *F,
                                 const double memA, const double memB,
                                 int skip_zeros)
{
   Array<GridFunction*> RHSGridFunctions;
   RHSGridFunctions.SetSize(1);
   RHSGridFunctions[0] = F;
   Array<int> ess_bdr;
   ess_bdr.SetSize(1);
   ess_bdr = 0; // dummy
   
   GridFunction sol(fes2[0]);
   sol = 0.0;
   
   AssembleSC(RHSGridFunctions, ess_bdr, sol, 0, memA, memB, skip_zeros);
}

void HDGBilinearForm::AssembleSC(GridFunction *F1, 
                                 GridFunction *F2,
                                 Array<int> &bdr_attr_is_ess,
                                 GridFunction &sol,
                                 const double memA, const double memB,
                                 int skip_zeros)
{
   Array<GridFunction*> RHSGridFunctions;
   RHSGridFunctions.SetSize(2);
   RHSGridFunctions[0] = F1;
   RHSGridFunctions[1] = F2;
   AssembleSC(RHSGridFunctions, bdr_attr_is_ess, sol, 1, memA, memB, skip_zeros);
}

void HDGBilinearForm::AssembleSC(Array<GridFunction*> &rhs_F,
        const Array<int> &bdr_attr_is_ess,
        const GridFunction &sol,
        const int elimBC,
        const double memA, const double memB,
        int skip_zeros)
{
    Allocate(bdr_attr_is_ess, memA, memB);

    DenseMatrix A_local, AC_local, SC_local;
    Vector F_local, F1_local, F2_local, CAinvF, AinvF;

    Array<int> fcs;
    Array<int> be_to_face;

    Array<int> vdofs_u, vdofs_e1, vdofs_e2;
    int ndof_u, ndof_e1, ndof_e2;

    double *A_local_data, *B_local_data;

    DenseMatrix *B_local;
    DenseMatrix *C_local;
    DenseMatrix *D_local;
    Vector G_local;

    for(int i=0; i< fes1[0]->GetNE(); i++)
    {
//           std::cout << "i " << i << endl << flush;

        GetInteriorVDofs(i, vdofs_u);
        ndof_u  = vdofs_u.Size();
//        std::cout << "ndof_u " << ndof_u << endl << flush;


        // Set A_local and compute the domain integrals
        A_local.SetSize(ndof_u, ndof_u);
        A_local = 0.0;
        compute_domain_integrals(i, &A_local);

        // Get the element faces
        el_to_face->GetRow(i, fcs);

        int no_faces = fcs.Size();
        B_local = new DenseMatrix[no_faces];
        C_local = new DenseMatrix[no_faces];
        D_local = new DenseMatrix[no_faces];

//        F_local.SetSize(ndof_u);
//        F_local = 0.0;
//        rhs_F1.GetSubVector(vdofs_u, F_local);

        GetInteriorSubVector(rhs_F, i, ndof_u, F_local);

        // compute the face integrals for A, B, C and D
        for(int edge1=0; edge1<no_faces; edge1++)
        {
            GetBdrVDofs(fcs[edge1], vdofs_e1);
            ndof_e1 = vdofs_e1.Size();
//            std::cout << "ndof_e1 " << ndof_e1 << endl << flush;

            B_local[edge1].SetSize(ndof_u, ndof_e1);
            C_local[edge1].SetSize(ndof_e1, ndof_u);
            D_local[edge1].SetSize(ndof_e1, ndof_e1);

            B_local[edge1] = 0.0;
            C_local[edge1] = 0.0;
            D_local[edge1] = 0.0;
#ifdef MFEM_USE_MPI
            // compute the face integrals
            if ( Edge_to_SharedEdge[fcs[edge1]] > -1 )
            {
                compute_face_integrals_shared(i, fcs[edge1], Edge_to_SharedEdge[fcs[edge1]], false,
                                    &A_local, &B_local[edge1], &C_local[edge1], &D_local[edge1]);
            }
            else
            {
#endif
                compute_face_integrals(i, fcs[edge1], Edge_to_Be[fcs[edge1]], false,
                        &A_local, &B_local[edge1], &C_local[edge1], &D_local[edge1]);
#ifdef MFEM_USE_MPI
            }
#endif
        }

        A_local.Neg();
        A_local.Invert();

        // Save A and B if necessary
        if (i<elements_A)
        {
            A_local_data = A_local.GetData();

            for(int j = 0; j<ndof_u*ndof_u; j++)
                A_data[A_offsets[i] + j] = A_local_data[j];
        }

        if (i<elements_B)
        {
            int size_B_copied = 0;
            for(int edge1=0; edge1<no_faces; edge1++)
            {
                B_local_data = B_local[edge1].GetData();
                GetBdrVDofs(fcs[edge1], vdofs_e1);

                for(int j = 0; j<(ndof_u*(vdofs_e1.Size())); j++)
                    B_data[B_offsets[i] + size_B_copied + j] = B_local_data[j];

                size_B_copied += ndof_u*(vdofs_e1.Size());
            }
        }

        for(int edge1=0; edge1<no_faces; edge1++)
        {
            GetBdrVDofs(fcs[edge1], vdofs_e1);
            ndof_e1 = vdofs_e1.Size();
            G_local.SetSize(ndof_e1);
            G_local = 0.0;
            if (elimBC == 1)
            {
                Eliminate_BC(vdofs_e1, ndof_u, sol, &F_local, &G_local,
                        &B_local[edge1], &C_local[edge1], &D_local[edge1]);
            }

            rhs_SC->AddElementVector(vdofs_e1, 1.0, G_local);
        }

        AinvF.SetSize(ndof_u);
        A_local.Mult(F_local, AinvF);

        // Loop over all the possible face pairs
        for(int edge1=0; edge1<no_faces; edge1++)
        {
            GetBdrVDofs(fcs[edge1], vdofs_e1);
            ndof_e1 = vdofs_e1.Size();
            (D_local[edge1]).Threshold(1.0e-16);
            mat->AddSubMatrix(vdofs_e1, vdofs_e1, D_local[edge1], skip_zeros);

            CAinvF.SetSize(ndof_e1);
            (C_local[edge1]).Mult(AinvF, CAinvF);
            rhs_SC->AddElementVector(vdofs_e1, 1.0, CAinvF);

            AC_local.SetSize(ndof_e1, ndof_u);
            Mult(C_local[edge1], A_local, AC_local);

            for(int edge2=0; edge2<no_faces; edge2++)
            {
                // Get the unknowns belonging to the edge
                GetBdrVDofs(fcs[edge2], vdofs_e2);
                ndof_e2 = vdofs_e2.Size();

                SC_local.SetSize(ndof_e1, ndof_e2);

                // Compute the product that will be added to the Schur complement
                Mult(AC_local, B_local[edge2], SC_local);

                SC_local.Threshold(1.0e-16);
                mat->AddSubMatrix(vdofs_e1, vdofs_e2, SC_local, skip_zeros);
            }
        }

        delete [] B_local;
        delete [] C_local;
        delete [] D_local;
    }

}

// Eliminate the boundary condition from B, C and D
void HDGBilinearForm::Eliminate_BC(const Array<int> &vdofs_e1,
        const int ndof_u, const GridFunction &sol,
        Vector *rhs_F, Vector *rhs_G, DenseMatrix *B_local,
        DenseMatrix *C_local, DenseMatrix *D_local)
{
    int ndof_e1 = vdofs_e1.Size();
    double solution;

    // First we set the BC on the rhs vector for the unknowns on the boundary
    for(int j = 0; j < ndof_e1; j++) // j is the column
    {
        if (ess_dofs[vdofs_e1[j]] < 0)
        {
            (*rhs_G)(j) += sol(vdofs_e1[j]);
            for(int i = 0; i < ndof_e1; i++)
            {
                (*D_local)(j,i) = (i == j);
            }
        }
    }

    // Eliminate BC from B, C and D
    // From D we have to eliminate only the rows that do not belong to a boundary unknown,
    // since those values or the RHS are already set.
    for(int j = 0; j < ndof_e1; j++) // j is the column
    {
        if (ess_dofs[vdofs_e1[j]] < 0)
        {
            solution = sol(vdofs_e1[j]);
            for(int i = 0; i < ndof_e1; i++)
            {
                if (!(ess_dofs[vdofs_e1[i]] < 0))
                    (*rhs_G)(i) -= solution * (*D_local)(i,j);

                (*D_local)(i,j) = (i==j);
            }

            for(int i = 0; i < ndof_u; i++)
            {
                (*rhs_F)(i) -= solution * (*B_local)(i,j);
                (*B_local)(i,j) = 0.0;
                (*C_local)(j,i) = 0.0;
            }
        }
    }
}


void HDGBilinearForm::Reconstruct(GridFunction *F,
                                  const GridFunction *ubar,
                                  GridFunction *u)
{
   Array<GridFunction*> RHSGridFunctions, FacetGridFunctions, SolGridFunctions;
   RHSGridFunctions.SetSize(1);
   RHSGridFunctions[0] = F;
   SolGridFunctions.SetSize(1);
   SolGridFunctions[0] = u;
   Reconstruct(RHSGridFunctions, ubar, SolGridFunctions);
}

void HDGBilinearForm::Reconstruct(GridFunction *F1,
                                  GridFunction *F2,
                                  const GridFunction *ubar,
                                  GridFunction *q, 
                                  GridFunction *u)
{
   Array<GridFunction*> RHSGridFunctions, FacetGridFunctions, SolGridFunctions;
   RHSGridFunctions.SetSize(1);
   RHSGridFunctions[0] = F1;
   RHSGridFunctions[1] = F2;
   SolGridFunctions.SetSize(2);
   SolGridFunctions[0] = q;
   SolGridFunctions[1] = u;
   Reconstruct(RHSGridFunctions, ubar, SolGridFunctions);
}


   
// Reconstruct u from the facet solution
void HDGBilinearForm::Reconstruct(Array<GridFunction*> &F, 
                                  const GridFunction *ubar,
                                  Array<GridFunction*> &u)
{
    DenseMatrix A_local;
    Vector u_local, F_local, ubar_local, Bubar_local;

    Array<int> fcs;

    Array<int> vdofs_u, vdofs_e1;
    int ndof_u, ndof_e1;

    DenseMatrix dummy_DM;
    DenseMatrix *B_local;

    for(int i=0; i< fes1[0]->GetNE(); i++)
    {
        GetInteriorVDofs(i, vdofs_u);
        ndof_u  = vdofs_u.Size();

        // Set A_local and compute the domain integrals
        A_local.SetSize(ndof_u, ndof_u);
        A_local = 0.0;
        // If A is not saved then compute the domain integrals
        if (i>=elements_A)
            compute_domain_integrals(i, &A_local);


        // Get the element faces
        el_to_face->GetRow(i, fcs);

        int no_faces = fcs.Size();
        B_local = new DenseMatrix[no_faces];

        Bubar_local.SetSize(ndof_u);

        int B_values_read = 0;

//        F_local.SetSize(ndof_u);
//        F_local = 0.0;
//        F->GetSubVector(vdofs_u, F_local);

        GetInteriorSubVector(F, i, ndof_u, F_local);


        for(int edge1=0; edge1<no_faces; edge1++)
        {
            GetBdrVDofs(fcs[edge1], vdofs_e1);
            ndof_e1 = vdofs_e1.Size();

            B_local[edge1].SetSize(ndof_u, ndof_e1);
            B_local[edge1] = 0.0;
            // If B is not saved then compute the face integrals
            // otherwise load the matrices
            if (i>=elements_B)
            {
#ifdef MFEM_USE_MPI
                if ( Edge_to_SharedEdge[fcs[edge1]] > -1 )
                {
                    compute_face_integrals_shared(i, fcs[edge1], Edge_to_SharedEdge[fcs[edge1]], true,
                                        &A_local, &B_local[edge1], &dummy_DM, &dummy_DM);
                }
                else
                {
#endif
                    compute_face_integrals(i, fcs[edge1], Edge_to_Be[fcs[edge1]], true,
                            &A_local, &B_local[edge1], &dummy_DM, &dummy_DM);
#ifdef MFEM_USE_MPI
                }
#endif
            }
            else
            {
                for(int row = 0; row < ndof_e1; row++)
                    for(int col = 0; col < (ndof_u); col++)
                    {
                        (B_local[edge1])(col,row) = B_data[B_offsets[i] + B_values_read + row*ndof_u + col];
                    }

                B_values_read += ndof_u*ndof_e1;
            }

            ubar_local.SetSize(ndof_e1);

            ubar->GetSubVector(vdofs_e1, ubar_local);
            (B_local[edge1]).Mult(ubar_local, Bubar_local);

            F_local.Add(-1.0, Bubar_local);
        }

        F_local *= -1.0;
        // Compute A^{-1} if it is not saved or just load it
        if (i>=elements_A)
        {
            A_local.Invert();
            A_local.Neg();
        }
        else
        {
            for(int row = 0; row < ndof_u; row++)
                for(int col = 0; col < ndof_u; col++)
                {
                    A_local(col,row) = A_data[A_offsets[i] + row*ndof_u + col];
                }
        }

        u_local.SetSize(ndof_u);
        A_local.Mult(F_local, u_local);

//        u->SetSubVector(vdofs_u, u_local);

        SetInteriorSubVector(u, i, ndof_u, u_local);

        delete [] B_local;
    }

}


#ifdef MFEM_USE_MPI
// Compute the face integrals for B and C
/* They both contain only TraceInteriorFaceIntegrators and TraceBoundaryFaceIntegrators.
 * The bool onlyB should be false when creating the Schur complement, true when reconstructing u from ubar.
*/
void HDGBilinearForm::compute_face_integrals_shared(const int elem, const int edge,
                                             const int sf,
                                             const bool onlyB,
                                             DenseMatrix *A_local,
                                             DenseMatrix *B_local,
                                             DenseMatrix *C_local,
                                             DenseMatrix *D_local)
{
    ParFiniteElementSpace* pfes1 = dynamic_cast<ParFiniteElementSpace*>(fes1[0]);
    ParMesh *pmesh = pfes1 -> GetParMesh();
    FaceElementTransformations *tr;

    tr = pmesh->GetSharedFaceTransformations(sf);

    const FiniteElement &trial_face_fe = *fes2[0]->GetFaceElement(edge);
    const FiniteElement &testu1_fe1 = *fes1[0]->GetFE(tr->Elem1No);
    const FiniteElement &testu1_fe2 = *fes1[0]->GetFE(tr->Elem2No);

    // For parallel the element the processor owns is tr->Elem1No
//    hdg_fbfi[0]->AssembleFaceMatrixOneElement1and1FES(test_fe1, test_fe2,
//                            trial_face_fe, *tr,
//                            1, onlyB,
//                            elemmat1, elemmat2,
//                            elemmat3, elemmat4);
    switch (NInteriorFES)
    {
    case 1:
    {
        hdg_fbfi[0]->AssembleFaceMatrixOneElement1and1FES(testu1_fe1, testu1_fe2,
                            trial_face_fe, *tr,
                            1, onlyB,
                            elemmat1, elemmat2,
                            elemmat3, elemmat4);
        break;
    }
    case 2:
    {
        const FiniteElement &testu2_fe1 = *fes1[1]->GetFE(tr->Elem1No);
        const FiniteElement &testu2_fe2 = *fes1[1]->GetFE(tr->Elem2No);
        hdg_fbfi[0]->AssembleFaceMatrixOneElement2and1FES(testu1_fe1, testu1_fe2,
                            testu2_fe1, testu2_fe2,
                            trial_face_fe,
                            *tr, 1, onlyB,
                            elemmat1, elemmat2,
                            elemmat3, elemmat4);
        break;
    }
    default:
    {
        mfem_error("HDGBilinearForm::compute_face_integrals is defined only for 1 or 2 interior FES");
        break;
    }

    }

    A_local->Add(1.0, elemmat1);
    B_local->Add(1.0, elemmat2);
    // If it is not reconstruction C and D are needed
    if (!onlyB)
    {
        C_local->Add(1.0, elemmat3);
        D_local->Add(1.0, elemmat4);
    }
}

HypreParMatrix *HDGBilinearForm::ParallelAssembleSC(SparseMatrix *m)
{
   if (m == NULL) { return NULL; }

   MFEM_VERIFY(m->Finalized(), "local matrix needs to be finalized for "
               "ParallelAssemble3");

   ParFiniteElementSpace* pfes2 = dynamic_cast<ParFiniteElementSpace*>(fes2[0]);

   int lvsize = pfes2->GetVSize();
   const HYPRE_Int *face_nbr_glob_ldof = pfes2->GetFaceNbrGlobalDofMap();
   HYPRE_Int ldof_offset = pfes2->GetMyDofOffset();

   Array<HYPRE_Int> glob_J(m->NumNonZeroElems());
   int *J = m->GetJ();
   for (int i = 0; i < glob_J.Size(); i++)
   {
        if (J[i] < lvsize)
        {
           glob_J[i] = J[i] + ldof_offset;
        }
        else
        {
           glob_J[i] = face_nbr_glob_ldof[J[i] - lvsize];
        }
   }

   HypreParMatrix *A = new HypreParMatrix(pfes2->GetComm(), lvsize, pfes2->GlobalVSize(),
                                          pfes2->GlobalVSize(), m->GetI(), glob_J,
                                          m->GetData(), pfes2->GetDofOffsets(),
                                          pfes2->GetDofOffsets());

   HypreParMatrix *rap = RAP(A, pfes2->Dof_TrueDof_Matrix());

   delete A;

   return rap;
}

HypreParVector *HDGBilinearForm::ParallelVectorSC()
{
   ParFiniteElementSpace* pfes2 = dynamic_cast<ParFiniteElementSpace*>(fes2[0]);
   HypreParVector *tv = pfes2->NewTrueDofVector();

   pfes2->Dof_TrueDof_Matrix()->MultTranspose(*rhs_SC, *tv);
   return tv;
}

#endif

}
