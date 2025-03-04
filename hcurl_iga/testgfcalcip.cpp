#include <iostream>
#include "mfem.hpp"
using namespace mfem;
using namespace std;

double func(Vector& x)
{
    return x[0] + x[1];
}

int main()
{
    Mesh mesh(10, 10, Element::TRIANGLE, true, 1.0, 1.0);

    int p_order = 1;
    H1_FECollection h1_fec(p_order, mesh.Dimension());
    FiniteElementSpace h1_space(&mesh, &h1_fec);

    GridFunction gf(&h1_space);
    FunctionCoefficient coeff(func);
    gf.ProjectCoefficient(coeff);

    Vector phy_point(2);
    phy_point(0) = 0.15;
    phy_point(1) = 0.25;
    phy_point.Print(cout << "physical point: ");

    IntegrationPoint ip;
    int elem_idx;
    ElementTransformation* tran;
    for (int i=0; i<mesh.GetNE(); ++i)
    {
        tran = mesh.GetElementTransformation(i);
        InverseElementTransformation invtran(tran);
        int ret = invtran.Transform(phy_point, ip);
        if (ret == 0)
        {
            elem_idx = i;
            break;
        }
    }

    cout << elem_idx << "-th element\n"
         << "reference point: " << ip.x << ", " << ip.y << endl;

    DenseMatrix phy_point_mat(2, 1);
    phy_point_mat(0, 0) = 0.15;
    phy_point_mat(1, 0) = 0.25;
    Array<int> elem_ids;
    Array<IntegrationPoint> ips;
    mesh.FindPoints(phy_point_mat, elem_ids, ips);

    elem_ids.Print(cout << "element ids: ");

    for (int i=0; i<ips.Size(); ++i)
    {
        cout << i << "-th integration point: " << ips[i].x << ", " << ips[i].y << endl;
    }

    return 0;
}