// SaveSolidSubmesh: write the solid part of a topology-optimization design to
// its own ParaView collection.
//
// The optimizer's ParaView output carries the whole mesh, so the void shows up
// as near-zero density draped over every element and the final shape has to be
// thresholded in the viewer.  This writes the shape directly: every element
// whose design density peaks above `threshold` is retagged, a ParSubMesh is cut
// from those elements, and the design/physical density fields are transferred
// onto it and saved under ParaView_fsol/<run_tag>.
//
// Thresholding is per ELEMENT on the max over the element's nodes, so an
// element survives if any part of it is solid.  That also drops the Option C
// passive void (rho~ ~ 0) without needing to know which attributes are passive.
//
// NOTE: this mutates `pmesh` -- the surviving elements keep attribute 1000 and
// the mesh's attribute table is rebuilt.  Call it at the very end of a run, or
// restore the attributes yourself afterwards.

#pragma once

#include "mfem.hpp"
#include <string>

namespace mfem
{

inline void SaveSolidSubmesh(ParMesh &pmesh, ParGridFunction &desi_density,
                             ParGridFunction &phys_density,
                             const std::string &run_tag,
                             int order, real_t threshold = 0.1)
{
    const int sol_attr = 1000;

    for (int i = 0; i < pmesh.GetNE(); i++)
    {
        real_t elem_max = -infinity();

        ElementTransformation *T = pmesh.GetElementTransformation(i);
        const FiniteElement *fe = desi_density.FESpace()->GetFE(i);
        const IntegrationRule &ir = fe->GetNodes();

        for (int j = 0; j < ir.GetNPoints(); j++)
        {
            const IntegrationPoint &ip = ir.IntPoint(j);
            T->SetIntPoint(&ip);
            real_t val = desi_density.GetValue(*T, ip);
            elem_max = std::max(elem_max, val);
        }

        if (elem_max > threshold)
        {
            pmesh.SetAttribute(i, sol_attr);
        }
    }
    pmesh.SetAttributes();

    Array<int> sol_mesh_attrs(1);
    sol_mesh_attrs[0] = sol_attr;

    ParSubMesh sol_submesh = ParSubMesh::CreateFromDomain(pmesh, sol_mesh_attrs);
    ParFiniteElementSpace filter_subfes(&sol_submesh,
                                        desi_density.ParFESpace()->FEColl());

    ParGridFunction desi_density_sub(&filter_subfes);
    ParGridFunction phys_density_sub(&filter_subfes);

    ParSubMesh::Transfer(desi_density, desi_density_sub);
    ParSubMesh::Transfer(phys_density, phys_density_sub);

    // save in separate paraview
    ParaViewDataCollection dc(run_tag, &sol_submesh);
    dc.SetPrefixPath("ParaView_fsol");
    dc.SetLevelsOfDetail(order);
    dc.SetDataFormat(VTKFormat::BINARY);
    dc.SetHighOrderOutput(true);
    dc.RegisterField("density", &phys_density_sub);
    dc.RegisterField("rho_filter", &desi_density_sub);
    dc.Save();
}

} // namespace mfem
