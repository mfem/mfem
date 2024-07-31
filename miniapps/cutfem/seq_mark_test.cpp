

#include "mfem.hpp"
#include <iostream>
#include "cut_marking.hpp"

using namespace mfem;
using namespace std;



class GyroidCoeff:public Coefficient
{
public:
    GyroidCoeff(double cell_size=1.0){
        ll=cell_size;
    }

    virtual
        double Eval(ElementTransformation &T,
             const IntegrationPoint &ip)
    {
        //evaluate the true coordinate of the ip
        Vector xx; xx.SetSize(T.GetDimension());
        T.Transform(ip,xx);
        double x = xx[0]*ll;
        double y = xx[1]*ll;
        double z = (xx.Size()==3) ? xx[2]*ll : 0.0;

        double r=std::sin(x)*std::cos(y) +
                   std::sin(y)*std::cos(z) +
                   std::sin(z)*std::cos(x) ;

        return r;
    }

private:
    double ll;
};


class BinaryGyroidCoeff:public GyroidCoeff
{
public:
    BinaryGyroidCoeff(double cell_size=1.0):GyroidCoeff(cell_size)
    {

    }

    virtual
    double Eval(ElementTransformation &T,
             const IntegrationPoint &ip)
    {
        double r=GyroidCoeff::Eval(T,ip);
        if(r>0.0){return 1.0;}
        return -1.0;
    }

};


int main(int argc, char *argv[])
{
    // 1. Parse command line options.
    string mesh_file = "../../data/star.mesh";
    int order = 3;
    int rs_levels = 2;


    OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
    args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
    args.AddOption(&rs_levels, "-rs", "--refine-serial",
                   "Number of times to refine the mesh uniformly in serial.");
    args.ParseCheck();

    Mesh mesh(mesh_file);
    for (int lev = 0; lev < rs_levels; lev++) { mesh.UniformRefinement(); }


    H1_FECollection fec(order, mesh.Dimension());
    FiniteElementSpace fespace(&mesh, &fec);
    cout << "Number of unknowns: " << fespace.GetTrueVSize() << endl;

    GridFunction cgf(&fespace);
    // project the Gyroid coefficient onto the grid function
    GyroidCoeff gco(2.0*M_PI);
    cgf.ProjectCoefficient(gco);

    ElementMarker* elmark=new ElementMarker(mesh,false,true);
    elmark->SetLevelSetFunction(cgf);

    Array<int> marks;
    elmark->MarkElements(marks);
    Array<int> ghost_penalty_marks;
    elmark->MarkGhostPenaltyFaces(ghost_penalty_marks);


    //Create L2 field for marking
    L2_FECollection* l2fec=new L2_FECollection(0,mesh.Dimension());
    FiniteElementSpace* l2fes=new FiniteElementSpace(&mesh,l2fec,1);
    GridFunction mgf(l2fes);
    for(int i=0;i<marks.Size();i++){
        mgf[i]=marks[i];
    }

    delete elmark;


    // ParaView output.
    ParaViewDataCollection dacol("ParaViewMarking", &mesh);
    dacol.SetLevelsOfDetail(order);
    dacol.SetHighOrderOutput(true);
    dacol.RegisterField("marks", &mgf);
    dacol.RegisterField("gyroid",&cgf);
    dacol.SetTime(1.0);
    dacol.SetCycle(1);
    dacol.Save();

    delete l2fes;
    delete l2fec;

    return 0;

}
