// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "multiapp.hpp"

#ifdef MFEM_USE_MPI

namespace mfem
{

void Field::GetDerivative(Field* x, Vector &x0, Vector &dydx)
{
    // If y = x, dy/dx = 1, the identity operator
    if(GetID() == x->GetID())
    {
        dydx.SetSize(Data()->Size());
        dydx = 1.0;
        return;
    }

    if(IsSource() || IsTarget())
    {
        GraphNode *op = GetSourceNode();
        MFEM_ASSERT(op != nullptr, "Field: " << GetID()
                    << " does not have an associated source GraphNode.");
        op->GetDerivative(this, x, x0, dydx);
    }
    else
    {
        MFEM_ABORT("Field::GetDerivative() not implemented for fields that "
                   << "are neither Field::Type::SOURCE nor Field::Type::TARGET."
                   << " Field IDs: " << GetID() << " and " << x->GetID());
    }
}

DAGraph::~DAGraph()
{
    for(int i=0; i < nnodes; i++)
    {
        if(nodes_owned[i] && nodes[i]) delete nodes[i];
    }
    if(grad) delete grad;
}

void DAGraph::Mult(const Vector &x, Vector &y) const
{
    BlockVector xb(x.GetData(), in_offsets);
    BlockVector yb(y.GetData(), out_offsets);

    int i_in = 0, i_out = 0;
    for (int i=0; i < nnodes; i++)
    {
        bool has_input = (nodes[i]->Height() > 0);
        bool has_output = (nodes[i]->Width() > 0);

        auto in_field = fields.GetLinkedFields("x" + std::to_string(i));
        if (in_field && has_input)
        {   // Update the field with the input data for this operator
            in_field->GetSource()->SetData(&xb.GetBlock(i_in));
            in_field->UpdateTargets();
            i_in++;
        }

        auto out_field = nodes[i]->LinkedField("output");
        if (out_field && has_output)
        {   // Update the output field with the output data for this operator
            out_field->GetSource()->SetData(&yb.GetBlock(i_out));
            out_field->UpdateTargets();
            i_out++;
        }
    }

    i_in = 0;
    i_out = 0;
    for (int i=0; i < nnodes; i++)
    {
        bool has_input = (nodes[i]->Height() > 0);
        bool has_output = (nodes[i]->Width() > 0);

        Vector &xi = (has_input) ? xb.GetBlock(i_in) : tmp;
        Vector &yi = (has_output) ? yb.GetBlock(i_out) : tmp;
        i_in += has_input;
        i_out += has_output;

        nodes[i]->Mult(xi, yi);
    }
}

void DAGraph::Execute(const Vector &x, Vector &y)
{
    BlockVector xb(x.GetData(), in_offsets);
    BlockVector yb(y.GetData(), out_offsets);

    int i_in = 0, i_out = 0;
    for (int i=0; i < nnodes; i++)
    {
        bool has_input = (nodes[i]->Height() > 0);
        bool has_output = (nodes[i]->Width() > 0);

        auto in_field = fields.GetLinkedFields("x" + std::to_string(i));
        if (in_field && has_input)
        {   // Update the field with the input data for this operator
            in_field->GetSource()->SetData(&xb.GetBlock(i_in));
            in_field->UpdateTargets();
            i_in++;
        }

        auto out_field = nodes[i]->LinkedField("output");
        if (out_field && has_output)
        {   // Update the output field with the output data for this operator
            out_field->GetSource()->SetData(&yb.GetBlock(i_out));
            out_field->UpdateTargets();
            i_out++;
        }
    }

    i_in = 0;
    i_out = 0;
    for (int i=0; i < nnodes; i++)
    {
        bool has_input = (nodes[i]->Height() > 0);
        bool has_output = (nodes[i]->Width() > 0);

        Vector &xi = (has_input) ? xb.GetBlock(i_in) : tmp;
        Vector &yi = (has_output) ? yb.GetBlock(i_out) : tmp;
        i_in += has_input;
        i_out += has_output;

        nodes[i]->Execute(xi, yi);
    }
}

Operator& DAGraph::GetGradient(const Vector &x) const
{
    if(grad_mode == GradMode::FD)
    {
        if(!grad)
        {
            grad = new future::FDJacobian(*this, x, 1e-6);
        }
        else
        {
            auto *fdj = dynamic_cast<future::FDJacobian*>(grad);
            fdj->Update(x);
        }
        return *grad;
    }

    BlockVector xb(x.GetData(), in_offsets);
    BlockOperator *block_grad;

    if(!grad)
    {
        block_grad = new BlockOperator(out_offsets, in_offsets);
        grad = block_grad;
    }
    else
    {
        block_grad = dynamic_cast<BlockOperator*>(grad);
    }

    // Forward pass to to populate (intermediate) fields for differentiation
    tmp.SetSize(out_offsets.Last());
    Mult(x, tmp);

    block_grad->owns_blocks = OwnsGradientOperators();
    int i_in = 0, i_out = 0;
    for (int i = 0; i < nnodes; i++)
    {
        std::string out_name = "y" + std::to_string(i);
        auto yfield = fields.GetField(out_name);
        if(yfield)
        {
            auto yfield_src = yfield->GetSource();
            if(yfield_src) // Has a source field
            {
                auto y_node = yfield_src->GetNode(); // Get node for src field

                MFEM_ASSERT(y_node != nullptr, "Source field: " << yfield_src->GetID() 
                            << " for target field: " << out_name 
                            << " does not have an associated GraphNode.");

                i_in = 0;
                for (int j = 0; j < nnodes; j++)
                {
                    std::string in_name = "x" + std::to_string(j);
                    auto x_lf = fields.GetLinkedFields(in_name);
                    bool has_targets = (x_lf) ? x_lf->HasTargets() : false;
                    
                    if(x_lf && has_targets) // Has a linked field with at least one target
                    {
                        auto xfield = x_lf->GetSource(); // Get source field for this input

                        MFEM_ASSERT(xfield != nullptr, "LinkedFields:" << in_name 
                                    << " has target fields but no source field!");

                        // Differentiate y_node with respect to xfield (dy/dx)
                        Operator *dydx = nullptr;
                        bool ownership = y_node->GetDerivative(xfield, xb.GetBlock(i_in), dydx);
                        if(dydx)
                        {
                            block_grad->SetBlock(i_out,i_in,dydx);
                            // block_grad->BlockOwnership(i_out,i_in) = ownership; // Set ownership flag for this block
                        }
                        i_in++;
                    }
                }
                i_out++;
            }
        }
    }
    return *grad;
}

void DAGraph::GetDerivative(Field* y, Field* x, Vector &x0, Vector &dydx)
{
    if(y->GetSourceNode()->GetID() == x->GetSourceNode()->GetID())
    {   
        // Fields are sources (i.e., input to graph) and independent, so dy/dx = 0
        dydx.SetSize(y->Data()->Size());
        dydx = 0.0;
        return;
    }
    y->GetDerivative(x, x0, dydx);
}

} // namespace mfem
#endif // MFEM_USE_MPI