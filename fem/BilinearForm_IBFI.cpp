#include "bilinearform_IBFI.hpp"

using namespace mfem;

BilinearFormIBFI::BilinearFormIBFI(FiniteElementSpace* f) : BilinearForm(f)
{
    MFEM_VERIFY(interior_boundary_face_integs_marker != NULL,
        "interior_boundary_face_integs_marker is NULL.");
}

void BilinearFormIBFI::AddInteriorBoundaryFaceIntegrator(BilinearFormIntegrator* bfi,
    Array<int>& int_bdr_marker)
{
    interior_boundary_face_integs.Append(bfi);
    interior_boundary_face_integs_marker.Append(&int_bdr_marker);
}

void BilinearFormIBFI::Assemble(int skip_zeros) {
    
    Mesh* mesh = fes->GetMesh();
    
    BilinearForm::Assemble(skip_zeros);

    if (interior_boundary_face_integs.Size())
    {
        FaceElementTransformations* tr;
        Array<int> vdofs2;
        const FiniteElement* fe1, * fe2;

        // Which interior boundary attributes need to be processed?
        Array<int> int_bdr_attr_marker(mesh->bdr_attributes.Size() ?
            mesh->bdr_attributes.Max() : 0);
        int_bdr_attr_marker = 0;
        for (int k = 0; k < interior_boundary_face_integs.Size(); k++)
        {
            if (interior_boundary_face_integs_marker[k] == NULL)
            {
                int_bdr_attr_marker = 1;
                break;
            }
            Array<int>& int_bdr_marker = *interior_boundary_face_integs_marker[k];
            MFEM_ASSERT(int_bdr_marker.Size() == int_bdr_attr_marker.Size(),
                "invalid boundary marker for boundary face integrator #"
                << k << ", counting from zero");
            for (int i = 0; i < int_bdr_attr_marker.Size(); i++)
            {
                int_bdr_attr_marker[i] |= int_bdr_marker[i];
            }
        }

        for (int i = 0; i < fes->GetNBE(); i++)
        {
            const int mesh_bdr_attr = mesh->GetBdrAttribute(i);
            if (int_bdr_attr_marker[mesh_bdr_attr - 1] == 0) { continue; }

            tr = mesh->GetInteriorFaceTransformations(i);
            if (tr != NULL)
            {
                fes->GetElementVDofs(tr->Elem1No, vdofs);
                fes->GetElementVDofs(tr->Elem2No, vdofs2);
                vdofs.Append(vdofs2);
                fe1 = fes->GetFE(tr->Elem1No);
                fe2 = fes->GetFE(tr->Elem2No);
                for (int k = 0; k < interior_boundary_face_integs.Size(); k++)
                {
                    if (interior_boundary_face_integs_marker[k] &&
                        (*interior_boundary_face_integs_marker[k])[mesh_bdr_attr - 1] == 0)
                    {
                        continue;
                    }

                    interior_boundary_face_integs[k]->AssembleFaceMatrix(*fe1, *fe2, *tr,
                        elemmat);
                    mat->AddSubMatrix(vdofs, vdofs, elemmat, skip_zeros);
                }
            }
        }
    }

#ifdef MFEM_USE_LEGACY_OPENMP
    if (free_element_matrices)
    {
        FreeElementMatrices();
    }
#endif
}