#include "cut_marking.hpp"

namespace mfem {

void ElementMarker::SetLevelSetFunction(Coefficient &ls_fun)
{
    FiniteElementCollection* fec=new H1_FECollection(h1_order,smesh->Dimension());
    FiniteElementSpace* pfes_sltn=new FiniteElementSpace(smesh,fec);

    Vector vals;
    Array<int> vdofs;

    if(use_cut_marks==false){
        if(include_cut_elements){
            elgf=(double)(ElementType::INSIDE);
            for(int e=0;e<smesh->GetNE();e++){
                const IntegrationRule &ir = pfes_sltn->GetFE(e)->GetNodes();
                {
                    int n = ir.GetNPoints();
                    vals.SetSize(n);
                    ElementTransformation *Tr = pfes_sltn->GetElementTransformation(e);
                    for(int k=0;k<n;k++){
                        Tr->SetIntPoint(&ir.IntPoint(k));
                        vals[k]=ls_fun.Eval(*Tr,ir.IntPoint(k));
                    }
                }


                int countp = 0;
                int countn = 0;
                for (int j = 0; j < ir.GetNPoints(); j++){
                    if (vals(j)>0.0) { countp++; }
                    else{countn++;}
                }
                if (countn == ir.GetNPoints()) // completely outside
                {
                    elfes->GetElementVDofs(e,vdofs);
                    elgf[vdofs[0]] = ElementType::OUTSIDE;
                }
            }
        }else{//DEFAULT - do not include cuts
            elgf=(double)(ElementType::OUTSIDE);
            for(int e=0;e<smesh->GetNE();e++){
                const IntegrationRule &ir = pfes_sltn->GetFE(e)->GetNodes();
                {
                    int n = ir.GetNPoints();
                    vals.SetSize(n);
                    ElementTransformation *Tr = pfes_sltn->GetElementTransformation(e);
                    for(int k=0;k<n;k++){
                        Tr->SetIntPoint(&ir.IntPoint(k));
                        vals[k]=ls_fun.Eval(*Tr,ir.IntPoint(k));
                    }
                }

                int countp = 0;
                int countn = 0;
                for (int j = 0; j < ir.GetNPoints(); j++){
                    if (vals(j)>0.0) { countp++; }
                    else{countn++;}
                }
                if (countp == ir.GetNPoints()) // completely inside
                {
                    elfes->GetElementVDofs(e,vdofs);
                    elgf[vdofs[0]] = ElementType::INSIDE;
                }
            }
        }
    }else{// use CUT mark
        elgf=(double)(ElementType::INSIDE);
        for(int e=0;e<smesh->GetNE();e++){
            const IntegrationRule &ir = pfes_sltn->GetFE(e)->GetNodes();
            {
                int n = ir.GetNPoints();
                vals.SetSize(n);
                ElementTransformation *Tr = pfes_sltn->GetElementTransformation(e);
                for(int k=0;k<n;k++){
                    Tr->SetIntPoint(&ir.IntPoint(k));
                    vals[k]=ls_fun.Eval(*Tr,ir.IntPoint(k));
                }
            }

            int countp = 0;
            int countn = 0;
            for (int j = 0; j < ir.GetNPoints(); j++){
                if (vals(j)>0) {countp++;}
                else {countn++;}
            }
            if (countn == ir.GetNPoints()) // completely outside
            {
                elfes->GetElementVDofs(e,vdofs);
                elgf[vdofs[0]] = ElementType::OUTSIDE;
            }else
                if ((countp>0)&&(countn>0))
                {
                    elfes->GetElementVDofs(e,vdofs);
                    elgf[vdofs[0]] = ElementType::CUT;
                }
        }
    }

    delete pfes_sltn;
    delete fec;
}

void ElementMarker::SetLevelSetFunction(const GridFunction &ls_fun)
{
    const FiniteElementSpace* pfes_sltn=ls_fun.FESpace();
    Vector vals;
    Array<int> vdofs;

    if(use_cut_marks==false){
        if(include_cut_elements){
            elgf=(double)(ElementType::INSIDE);
            for(int e=0;e<smesh->GetNE();e++){
                const IntegrationRule &ir = pfes_sltn->GetFE(e)->GetNodes();
                ls_fun.GetValues(e, ir, vals);
                int countn = 0;
                for (int j = 0; j < ir.GetNPoints(); j++){
                    if (vals(j)>0.0) {}
                    else{ countn++; }
                }
                if (countn == ir.GetNPoints()) // completely outside
                {
                    elfes->GetElementVDofs(e,vdofs);
                    elgf[vdofs[0]] = ElementType::OUTSIDE;
                }
            }
        }else{//DEFAULT - do not include cuts
            elgf=(double)(ElementType::OUTSIDE);
            for(int e=0;e<smesh->GetNE();e++){
                const IntegrationRule &ir = pfes_sltn->GetFE(e)->GetNodes();
                ls_fun.GetValues(e, ir, vals);
                int countp = 0;
                for (int j = 0; j < ir.GetNPoints(); j++){
                    if (vals(j)>0.0) { countp++; }
                }
                if (countp == ir.GetNPoints()) // completely inside
                {
                    elfes->GetElementVDofs(e,vdofs);
                    elgf[vdofs[0]] = ElementType::INSIDE;
                }
            }
        }
    }else{//use CUT marks
        elgf=(double)(ElementType::INSIDE);
        for(int e=0;e<smesh->GetNE();e++){
            const IntegrationRule &ir = pfes_sltn->GetFE(e)->GetNodes();
            ls_fun.GetValues(e, ir, vals);
            int countp = 0;
            int countn = 0;
            for (int j = 0; j < ir.GetNPoints(); j++){
                if (vals(j)>0) {countp++;}
                else {countn++;}
            }
            if (countn == ir.GetNPoints()) // completely outside
            {
                elfes->GetElementVDofs(e,vdofs);
                elgf[vdofs[0]] = ElementType::OUTSIDE;
            }else
                if ((countp>0)&&(countn>0))
                {
                    elfes->GetElementVDofs(e,vdofs);
                    elgf[vdofs[0]] = ElementType::CUT;
                }
        }
    }
}

void ElementMarker::MarkElements(Array<int> &elem_marker)
{
    elem_marker.SetSize(smesh->GetNE());
    for(int e=0;e<smesh->GetNE();e++)
    {
        ElementTransformation* tr=elfes->GetElementTransformation(e);
        IntegrationPoint ip; ip.Init(0);
        elem_marker[e] = elgf.GetValue(*tr, ip);
    }
}

void ElementMarker::MarkGhostPenaltyFaces(Array<int> &face_marker)
{
    face_marker.SetSize(smesh->GetNumFaces());
    face_marker=FaceType::UNDEFINED;
    IntegrationPoint ip; ip.Init(0);

    for(int f=0;f<smesh->GetNumFaces();f++){
        auto *ft = smesh->GetFaceElementTransformations(f, 3);
        if (ft->Elem2No < 0) { continue; } //do not mark boundary faces
        const int attr1 = elgf.GetValue(*ft->Elem1,ip);
        const int attr2 = elgf.GetValue(*ft->Elem2,ip);
        if((attr1==ElementType::CUT)&&(attr2!=ElementType::OUTSIDE))
        {
            face_marker[f]=FaceType::GHOSTP;
        }else
            if((attr1!=ElementType::OUTSIDE)&&(attr2==ElementType::CUT))
            {
                face_marker[f]=FaceType::GHOSTP;
            }
    }
}

void ElementMarker::MarkFaces(Array<int> &face_marker)
{
    face_marker.SetSize(smesh->GetNumFaces());
    face_marker=FaceType::UNDEFINED;
    IntegrationPoint ip; ip.Init(0);

    if(include_cut_elements==true){
        for(int f=0;f<smesh->GetNumFaces();f++){
            auto *ft = smesh->GetFaceElementTransformations(f, 3);
            if (ft->Elem2No < 0) { continue; } //do not mark boundary faces
            const int attr1 = elgf.GetValue(*ft->Elem1,ip);
            const int attr2 = elgf.GetValue(*ft->Elem2,ip);
            if((attr1==ElementType::OUTSIDE)||(attr2==ElementType::OUTSIDE)){
                if(attr1!=attr2){
                    face_marker[f]=FaceType::SURROGATE;
                }
            }
        }
    }else{
        for(int f=0;f<smesh->GetNumFaces();f++){
            auto *ft = smesh->GetFaceElementTransformations(f, 3);
            if (ft->Elem2No < 0) { continue; } //do not mark boundary faces
            const int attr1 = elgf.GetValue(*ft->Elem1,ip);
            const int attr2 = elgf.GetValue(*ft->Elem2,ip);
            if((attr1==ElementType::INSIDE)||(attr2==ElementType::INSIDE)){
                if(attr1!=attr2){
                    face_marker[f]=FaceType::SURROGATE;
                }
            }
        }
    }
}

void ElementMarker::ListEssentialTDofs(const Array<int> &elem_marker,
                                       FiniteElementSpace &lfes,
                                       Array<int> &ess_tdof_list) const
{
    Array<int> dofs;

    mfem::Vector vvdof; vvdof.SetSize(lfes.GetVSize()); vvdof=0.0;

    for(int i=0;i<lfes.GetNE();i++)
    {
        if(elem_marker[i]==ElementType::INSIDE){
            lfes.GetElementVDofs(i,dofs);
            for(int j=0;j<dofs.Size();j++){
                vvdof[dofs[j]]=1.0;
            }
        }

        if(include_cut_elements==true){
            if(elem_marker[i]==ElementType::CUT){
                lfes.GetElementVDofs(i,dofs);
                for(int j=0;j<dofs.Size();j++){
                    vvdof[dofs[j]]=1.0;
                }
            }
        }
    }

    Array<int> tdof_mark; tdof_mark.SetSize(lfes.GetTrueVSize());
    Vector vtdof; vtdof.SetSize(lfes.GetTrueVSize()); vtdof=0.0;
    lfes.GetProlongationMatrix()->MultTranspose(vvdof,vtdof);
    for(int i=0;i<vtdof.Size();i++){
        if(vtdof[i]<1.0){tdof_mark[i]=1;}
        else{tdof_mark[i]=0;}
    }

    lfes.MarkerToList(tdof_mark, ess_tdof_list);
}

#ifdef MFEM_USE_MPI

void ParElementMarker::SetLevelSetFunction(Coefficient &ls_fun)
{
    FiniteElementCollection* fec=new H1_FECollection(h1_order,pmesh->Dimension());
    ParFiniteElementSpace* pfes_sltn=new ParFiniteElementSpace(pmesh,fec);

    Vector vals;
    Array<int> vdofs;

    if(use_cut_marks==false){
        if(include_cut_elements){
            elgf=(double)(ElementMarker::ElementType::INSIDE);
            for(int e=0;e<pmesh->GetNE();e++){
                const IntegrationRule &ir = pfes_sltn->GetFE(e)->GetNodes();
                {
                    int n = ir.GetNPoints();
                    vals.SetSize(n);
                    ElementTransformation *Tr = pfes_sltn->GetElementTransformation(e);
                    for(int k=0;k<n;k++){
                        Tr->SetIntPoint(&ir.IntPoint(k));
                        vals[k]=ls_fun.Eval(*Tr,ir.IntPoint(k));
                    }
                }


                int countp = 0;
                int countn = 0;
                for (int j = 0; j < ir.GetNPoints(); j++){
                    if (vals(j)>0.0) { countp++; }
                    else{countn++;}
                }
                if (countn == ir.GetNPoints()) // completely outside
                {
                    elfes->GetElementVDofs(e,vdofs);
                    elgf[vdofs[0]] = ElementMarker::ElementType::OUTSIDE;
                }
            }
        }else{//DEFAULT - do not include cuts
            elgf=(double)(ElementMarker::ElementType::OUTSIDE);
            for(int e=0;e<pmesh->GetNE();e++){
                const IntegrationRule &ir = pfes_sltn->GetFE(e)->GetNodes();
                {
                    int n = ir.GetNPoints();
                    vals.SetSize(n);
                    ElementTransformation *Tr = pfes_sltn->GetElementTransformation(e);
                    for(int k=0;k<n;k++){
                        Tr->SetIntPoint(&ir.IntPoint(k));
                        vals[k]=ls_fun.Eval(*Tr,ir.IntPoint(k));
                    }
                }

                int countp = 0;
                int countn = 0;
                for (int j = 0; j < ir.GetNPoints(); j++){
                    if (vals(j)>0.0) { countp++; }
                    else{countn++;}
                }
                if (countp == ir.GetNPoints()) // completely inside
                {
                    elfes->GetElementVDofs(e,vdofs);
                    elgf[vdofs[0]] = ElementMarker::ElementType::INSIDE;
                }
            }
        }
    }else{// use CUT mark
        elgf=(double)(ElementMarker::ElementType::INSIDE);
        for(int e=0;e<pmesh->GetNE();e++){
            const IntegrationRule &ir = pfes_sltn->GetFE(e)->GetNodes();
            {
                int n = ir.GetNPoints();
                vals.SetSize(n);
                ElementTransformation *Tr = pfes_sltn->GetElementTransformation(e);
                for(int k=0;k<n;k++){
                    Tr->SetIntPoint(&ir.IntPoint(k));
                    vals[k]=ls_fun.Eval(*Tr,ir.IntPoint(k));
                }
            }

            int countp = 0;
            int countn = 0;
            for (int j = 0; j < ir.GetNPoints(); j++){
                if (vals(j)>0) {countp++;}
                else {countn++;}
            }
            if (countn == ir.GetNPoints()) // completely outside
            {
                elfes->GetElementVDofs(e,vdofs);
                elgf[vdofs[0]] = ElementMarker::ElementType::OUTSIDE;
            }else
                if ((countp>0)&&(countn>0))
                {
                    elfes->GetElementVDofs(e,vdofs);
                    elgf[vdofs[0]] = ElementMarker::ElementType::CUT;
                }
        }
    }

    if(pmesh->GetNRanks()>0){
        elgf.ExchangeFaceNbrData();
    }

    delete pfes_sltn;
    delete fec;
}


void ParElementMarker::SetLevelSetFunction(const ParGridFunction &ls_fun)
{
    ParFiniteElementSpace* pfes_sltn=ls_fun.ParFESpace();
    Vector vals;
    Array<int> vdofs;
    if(use_cut_marks==false){
        if(include_cut_elements){
            elgf=(double)(ElementMarker::ElementType::INSIDE);
            for(int e=0;e<pmesh->GetNE();e++){
                const IntegrationRule &ir = pfes_sltn->GetFE(e)->GetNodes();
                ls_fun.GetValues(e, ir, vals);
                int countn = 0;
                for (int j = 0; j < ir.GetNPoints(); j++){
                    if (vals(j)>0.0) {}
                    else{ countn++; }
                }
                if (countn == ir.GetNPoints()) // completely outside
                {
                    elfes->GetElementVDofs(e,vdofs);
                    elgf[vdofs[0]] = ElementMarker::ElementType::OUTSIDE;
                }
            }
        }else{//DEFAULT - do not include cuts
            elgf=(double)(ElementMarker::ElementType::OUTSIDE);
            for(int e=0;e<pmesh->GetNE();e++){
                const IntegrationRule &ir = pfes_sltn->GetFE(e)->GetNodes();
                ls_fun.GetValues(e, ir, vals);
                int countp = 0;
                for (int j = 0; j < ir.GetNPoints(); j++){
                    if (vals(j)>0.0) { countp++; }
                }
                if (countp == ir.GetNPoints()) // completely inside
                {
                    elfes->GetElementVDofs(e,vdofs);
                    elgf[vdofs[0]] = ElementMarker::ElementType::INSIDE;
                }
            }
        }}else{//use CUT marks
        elgf=(double)(ElementMarker::ElementType::INSIDE);
        for(int e=0;e<pmesh->GetNE();e++){
            const IntegrationRule &ir = pfes_sltn->GetFE(e)->GetNodes();
            ls_fun.GetValues(e, ir, vals);
            int countp = 0;
            int countn = 0;
            for (int j = 0; j < ir.GetNPoints(); j++){
                if (vals(j)>0) {countp++;}
                else {countn++;}
            }
            if (countn == ir.GetNPoints()) // completely outside
            {
                elfes->GetElementVDofs(e,vdofs);
                elgf[vdofs[0]] = ElementMarker::ElementType::OUTSIDE;
            }else
                if ((countp>0)&&(countn>0))
                {
                    elfes->GetElementVDofs(e,vdofs);
                    elgf[vdofs[0]] = ElementMarker::ElementType::CUT;
                }
        }
    }

    if(pmesh->GetNRanks()>0){
        elgf.ExchangeFaceNbrData();}
}

void ParElementMarker::MarkElements(Array<int> &elem_marker)
{
    elem_marker.SetSize(pmesh->GetNE());
    for(int e=0;e<pmesh->GetNE();e++)
    {
        ElementTransformation* tr=elfes->GetElementTransformation(e);
        IntegrationPoint ip; ip.Init(0);
        elem_marker[e] = elgf.GetValue(*tr, ip);
    }
}

void ParElementMarker::MarkGhostPenaltyFaces(Array<int> &face_marker)
{
    face_marker.SetSize(pmesh->GetNumFaces());
    face_marker=ElementMarker::FaceType::UNDEFINED;
    IntegrationPoint ip; ip.Init(0);

    for(int f=0;f<pmesh->GetNumFaces();f++){
        auto *ft = pmesh->GetFaceElementTransformations(f, 3);
        if (ft->Elem2No < 0) { continue; } //do not mark boundary faces
        const int attr1 = elgf.GetValue(*ft->Elem1,ip);
        const int attr2 = elgf.GetValue(*ft->Elem2,ip);
        if((attr1==ElementMarker::ElementType::CUT)&&(attr2!=ElementMarker::ElementType::OUTSIDE))
        {
            face_marker[f]=ElementMarker::FaceType::GHOSTP;
        }else
            if((attr1!=ElementMarker::ElementType::OUTSIDE)&&(attr2==ElementMarker::ElementType::CUT))
            {
                face_marker[f]=ElementMarker::FaceType::GHOSTP;
            }
    }

    if(pmesh->GetNRanks()>0){
        elgf.ExchangeFaceNbrData();
    }

    for (int f = 0; f < pmesh->GetNSharedFaces(); f++)
    {
        auto *ftr = pmesh->GetSharedFaceTransformations(f, true);
        const int attr1 = elgf.GetValue(*ftr->Elem1, ip);
        const int attr2 = elgf.GetValue(*ftr->Elem2, ip);
        int faceno = pmesh->GetSharedFace(f);
        if((attr1==ElementMarker::ElementType::CUT)&&(attr2!=ElementMarker::ElementType::OUTSIDE))
        {
            face_marker[faceno]=ElementMarker::FaceType::GHOSTP;
        }else
            if((attr1!=ElementMarker::ElementType::OUTSIDE)&&(attr2==ElementMarker::ElementType::CUT))
            {
                face_marker[faceno]=ElementMarker::FaceType::GHOSTP;
            }
    }
}

void ParElementMarker::MarkFaces(Array<int> &face_marker)
{
    face_marker.SetSize(pmesh->GetNumFaces());
    face_marker=ElementMarker::FaceType::UNDEFINED;
    IntegrationPoint ip; ip.Init(0);

    if(include_cut_elements==true){
        for(int f=0;f<pmesh->GetNumFaces();f++){
            auto *ft = pmesh->GetFaceElementTransformations(f, 3);
            if (ft->Elem2No < 0) { continue; } //do not mark boundary faces
            const int attr1 = elgf.GetValue(*ft->Elem1,ip);
            const int attr2 = elgf.GetValue(*ft->Elem2,ip);
            if((attr1==ElementMarker::ElementType::OUTSIDE)||(attr2==ElementMarker::ElementType::OUTSIDE)){
                if(attr1!=attr2){
                    face_marker[f]=ElementMarker::FaceType::SURROGATE;
                }
            }
        }
    }else{
        for(int f=0;f<pmesh->GetNumFaces();f++){
            auto *ft = pmesh->GetFaceElementTransformations(f, 3);
            if (ft->Elem2No < 0) { continue; } //do not mark boundary faces
            const int attr1 = elgf.GetValue(*ft->Elem1,ip);
            const int attr2 = elgf.GetValue(*ft->Elem2,ip);
            if((attr1==ElementMarker::ElementType::INSIDE)||(attr2==ElementMarker::ElementType::INSIDE)){
                if(attr1!=attr2){
                    face_marker[f]=ElementMarker::FaceType::SURROGATE;
                }
            }
        }
    }

    if(pmesh->GetNRanks()>0){
        elgf.ExchangeFaceNbrData();
    }

    if(include_cut_elements==true){
        for (int f = 0; f < pmesh->GetNSharedFaces(); f++)
        {
            auto *ftr = pmesh->GetSharedFaceTransformations(f, true);
            const int attr1 = elgf.GetValue(*ftr->Elem1, ip);
            const int attr2 = elgf.GetValue(*ftr->Elem2, ip);
            int faceno = pmesh->GetSharedFace(f);
            if((attr1==ElementMarker::ElementType::OUTSIDE)||(attr2==ElementMarker::ElementType::OUTSIDE)){
                if(attr1!=attr2){
                    face_marker[faceno]=ElementMarker::FaceType::SURROGATE;
                }
            }
        }
    }else{
        for (int f = 0; f < pmesh->GetNSharedFaces(); f++)
        {
            auto *ftr = pmesh->GetSharedFaceTransformations(f, true);
            const int attr1 = elgf.GetValue(*ftr->Elem1, ip);
            const int attr2 = elgf.GetValue(*ftr->Elem2, ip);
            int faceno = pmesh->GetSharedFace(f);
            if((attr1==ElementMarker::ElementType::INSIDE)||(attr2==ElementMarker::ElementType::INSIDE)){
                if(attr1!=attr2){
                    face_marker[faceno]=ElementMarker::FaceType::SURROGATE;
                }
            }
        }

    }
}

void ParElementMarker::ListEssentialTDofs(const Array<int> &elem_marker,
                                       ParFiniteElementSpace &lfes,
                                       Array<int> &ess_tdof_list) const
{
    Array<int> dofs;

    mfem::Vector vvdof; vvdof.SetSize(lfes.GetVSize()); vvdof=0.0;

    for(int i=0;i<lfes.GetNE();i++)
    {
        if(elem_marker[i]==ElementMarker::ElementType::INSIDE){
            lfes.GetElementVDofs(i,dofs);
            for(int j=0;j<dofs.Size();j++){
                vvdof[dofs[j]]=1.0;
            }
        }

        if(include_cut_elements==true){
            if(elem_marker[i]==ElementMarker::ElementType::CUT){
                lfes.GetElementVDofs(i,dofs);
                for(int j=0;j<dofs.Size();j++){
                    vvdof[dofs[j]]=1.0;
                }
            }
        }
    }

    Array<int> tdof_mark; tdof_mark.SetSize(lfes.GetTrueVSize());
    Vector vtdof; vtdof.SetSize(lfes.GetTrueVSize()); vtdof=0.0;
    lfes.GetProlongationMatrix()->MultTranspose(vvdof,vtdof);
    for(int i=0;i<vtdof.Size();i++){
        if(vtdof[i]<1.0){tdof_mark[i]=1;}
        else{tdof_mark[i]=0;}
    }

    lfes.MarkerToList(tdof_mark, ess_tdof_list);
}


#endif

}
