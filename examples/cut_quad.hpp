#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "algoim_quad.hpp"
#include <list>
/// Class for domain integration L(v) := (f, v)
using namespace mfem;
using namespace std;
template <int N, int ls1>
struct vortex
{
    double xscale;
    double yscale;
    double xmin;
    double ymin;
    double radius;
    double ax;
    double ay;
    double xc;
    double yc;
    template <typename T>
    T operator()(const blitz::TinyVector<T, N> &x) const
    {
        // level-set function to work in physical space
        // return -1 * (((x[0] - 5) * (x[0] - 5)) +
        //               ((x[1]- 5) * (x[1] - 5)) - (0.5 * 0.5));
        // level-set function for reference elements
        if (ls1 == 1)
        {
            return -1 * ((((x[0] * xscale) + xmin) * ((x[0] * xscale) + xmin)) +
                         (((x[1] * yscale) + ymin) * ((x[1] * yscale) + ymin)) - (radius * radius));
        }
        else if (ls1 == 2)
        {
            return 1 * ((((x[0] * xscale) + xmin) * ((x[0] * xscale) + xmin)) +
                        (((x[1] * yscale) + ymin) * ((x[1] * yscale) + ymin)) - (radius * radius));
        }
        else if (ls1==3)
        {
            return -1 * (((((x[0] * xscale) + xmin - xc) * ((x[0] * xscale) + xmin - xc)) / (ax * ax)) +
                         ((((x[1] * yscale) + ymin - yc) * ((x[1] * yscale) + ymin - yc)) / (ay * ay)) - (radius * radius));
        }

        else 
        {
            return -1 * ((((x[0] * xscale) + xmin) * ((x[0] * xscale) + xmin - xc)) +
                        (((x[1] * yscale) + ymin) * ((x[1] * yscale) + ymin - yc )) - (radius * radius));
        }
    }

    template <typename T>
    blitz::TinyVector<T, N> grad(const blitz::TinyVector<T, N> &x) const
    {
        // return blitz::TinyVector<T, N>(-1 * (2.0 * (x(0) - 5)), -1 * (2.0 * (x(1) - 5)));
        if (ls1 == 1)
        {
            return blitz::TinyVector<T, N>(-1 * (2.0 * xscale * ((x(0) * xscale) + xmin)),
                                           -1 * (2.0 * yscale * ((x(1) * yscale) + ymin)));
        }
        else if (ls1 == 2)
        {
            return blitz::TinyVector<T, N>(1 * (2.0 * xscale * ((x(0) * xscale) + xmin)),
                                           1 * (2.0 * yscale * ((x(1) * yscale) + ymin)));
        }
        else if (ls1 ==3)
        {
            return blitz::TinyVector<T, N>(-1 * (2.0 * xscale * ((x(0) * xscale) + xmin - xc)) / (ax * ax),
                                           -1 * (2.0 * yscale * ((x(1) * yscale) + ymin - yc)) / (ay * ay));
        }

        else
        {
            return blitz::TinyVector<T, N>(-1 * (2.0 * xscale * ((x(0) * xscale) + xmin - xc)),
                                           -1 * (2.0 * yscale * ((x(1) * yscale) + ymin -yc)));
        }

    }
};

///function that checks if an element is `cut` by `embedded geometry` or  not
template <int ls1>
bool cutByGeom(Mesh *mesh, int &elemid)
{
    Element *el = mesh->GetElement(elemid);
    Array<int> v;
    el->GetVertices(v);
    int k, l, n;
    k = 0;
    l = 0;
    n = 0;
    double xc = 0.0;
    double yc = 0.0;
    double ax = 0.0;
    double ay = 0.0;
    double r;
    for (int i = 0; i < v.Size(); ++i)
    {
        double *coord = mesh->GetVertex(v[i]);
        Vector lvsval(v.Size());
        if (ls1 == 1)
        {
            r = 1.0;
            lvsval(i) = ((coord[0] - xc) * (coord[0] - xc)) + ((coord[1] - yc) * (coord[1] - yc)) - (r * r);
        }
        else if (ls1 == 2)
        {
            r = 3.0;
            lvsval(i) = -1 * (((coord[0] - xc) * (coord[0] - xc)) + ((coord[1] - yc) * (coord[1] - yc)) - (r * r));
        }

        else if (ls1 == 3)
        {
            r = 1.0;
            xc = 20.0;
            yc = 20.0;
            ax = 0.5;
            ay = ax / 10.0;
            lvsval(i) = 1 * ((((coord[0] - xc) * (coord[0] - xc)) / (ax * ax)) + (((coord[1] - yc) * (coord[1] - yc)) / (ay * ay)) - (r * r));
        }

        else
        {
            r = 0.5;
            lvsval(i) = 1 * (((coord[0] - xc) * (coord[0] - xc)) + ((coord[1] - yc) * (coord[1] - yc)) - (r * r));
        }

        if ((lvsval(i) < 0) && (abs(lvsval(i)) > 1e-16))
        {
            k = k + 1;
        }
        if ((lvsval(i) > 0))
        {
            l = l + 1;
        }
        if ((lvsval(i) == 0) || (abs(lvsval(i)) < 1e-16))
        {
            n = n + 1;
        }
    }
    if ((k == v.Size()) || (l == v.Size()))
    {
        return false;
    }
    if (((k == 3) || (l == 3)) && (n == 1))
    {
        return false;
    }
    else
    {
        return true;
    }
}

///function that checks if an element is inside the `embedded geometry` or  not
template <int ls1>
bool insideBoundary(Mesh *mesh, int &elemid)
{
    Element *el = mesh->GetElement(elemid);
    Array<int> v;
    el->GetVertices(v);
    int k;
    k = 0;
    double xc = 0.0;
    double yc = 0.0;
    double ax = 0.0;
    double ay = 0.0;
    double r;
    for (int i = 0; i < v.Size(); ++i)
    {
        double *coord = mesh->GetVertex(v[i]);
        Vector lvsval(v.Size());
        if (ls1 == 1)
        {
            r = 1.0;
            lvsval(i) = ((coord[0] - xc) * (coord[0] - xc)) + ((coord[1] - yc) * (coord[1] - yc)) - (r * r);
        }
        else if (ls1 == 2)
        {
            r = 3.0;
            lvsval(i) = -1 * (((coord[0] - xc) * (coord[0] - xc)) + ((coord[1] - yc) * (coord[1] - yc)) - (r * r));
        }
        else if (ls1==3)
        {
            r = 1.0;
            xc = 20.0;
            yc = 20.0;
            ax = 0.5;
            ay = ax / 10.0;
            lvsval(i) = 1 * ((((coord[0] - xc) * (coord[0] - xc)) / (ax * ax)) + (((coord[1] - yc) * (coord[1] - yc)) / (ay * ay)) - (r * r));
        }
        else
        {
            r = 0.5;
            lvsval(i) = 1 * (((coord[0] - xc) * (coord[0] - xc)) + ((coord[1] - yc) * (coord[1] - yc)) - (r * r));
        }

        if ((lvsval(i) < 0) || (lvsval(i) == 0))
        {
            k = k + 1;
        }
    }
    if (k == v.Size())
    {
        return true;
    }
    else
    {
        return false;
    }
}

/// function to get element center
void GetElementCenter(Mesh *mesh, int id, mfem::Vector &cent)
{
    cent.SetSize(mesh->Dimension());
    int geom = mesh->GetElement(id)->GetGeometryType();
    ElementTransformation *eltransf = mesh->GetElementTransformation(id);
    eltransf->Transform(Geometries.GetCenter(geom), cent);
}

/// find bounding box for a given cut element
template <int N>
void findBoundingBox(Mesh *mesh, int id, blitz::TinyVector<double, N> &xmin, blitz::TinyVector<double, N> &xmax)
{
    Element *el = mesh->GetElement(id);
    Array<int> v;
    Vector min, max;
    min.SetSize(N);
    max.SetSize(N);
    for (int d = 0; d < N; d++)
    {
        min(d) = infinity();
        max(d) = -infinity();
    }
    el->GetVertices(v);
    for (int iv = 0; iv < v.Size(); ++iv)
    {
        double *coord = mesh->GetVertex(v[iv]);
        for (int d = 0; d < N; d++)
        {
            if (coord[d] < min(d))
            {
                min(d) = coord[d];
            }
            if (coord[d] > max(d))
            {
                max(d) = coord[d];
            }
        }
    }
    xmin = {min[0], min[1]};
    xmax = {max[0], max[1]};
}

/// find worst cut element size
void GetCutsize(Mesh *mesh, vector<int> cutelems, std::map<int, IntegrationRule *> &cutSquareIntRules,
                double &cutsize) {}

/// get integration rule for cut elements
template <int N, int ls>
void GetCutElementIntRule(Mesh *mesh, vector<int> cutelems, int order, double radius,
                          std::map<int, IntegrationRule *> &cutSquareIntRules)
{
    double tol = 1e-16;
    for (int k = 0; k < cutelems.size(); ++k)
    {
        IntegrationRule *ir;
        blitz::TinyVector<double, N> xmin;
        blitz::TinyVector<double, N> xmax;
        blitz::TinyVector<double, N> xupper;
        blitz::TinyVector<double, N> xlower;
        // standard reference element
        xlower = {0, 0};
        xupper = {1, 1};
        int dir = -1;
        int side = -1;
        int elemid = cutelems.at(k);
        findBoundingBox<N>(mesh, elemid, xmin, xmax);
        vortex<N, ls> phi;
        phi.xscale = xmax[0] - xmin[0];
        phi.yscale = xmax[1] - xmin[1];
        phi.xmin = xmin[0];
        phi.ymin = xmin[1];
        phi.radius = radius;
        if (ls == 3)
        {
            phi.ax = 0.5;
            phi.ay = phi.ax / 10.0;
            phi.xc = 20.0;
            phi.yc = 20.0;
        }
        
        if (ls == 4)
        {
            phi.xc = 20.0;
            phi.yc = 20.0;
        }
        auto q = Algoim::quadGen<N>(phi, Algoim::BoundingBox<double, N>(xlower, xupper), dir, side, order);
        int i = 0;
        ir = new IntegrationRule(q.nodes.size());
        //cout << "for element " << elemid << endl;
        for (const auto &pt : q.nodes)
        {
            IntegrationPoint &ip = ir->IntPoint(i);
            ip.x = pt.x[0];
            ip.y = pt.x[1];
            ip.weight = pt.w;
            i = i + 1;
            // scaled to original element space
            double xq = (pt.x[0] * phi.xscale) + phi.xmin;
            double yq = (pt.x[1] * phi.yscale) + phi.ymin;
            //cout << "int rule " << xq << " , " << yq << " : " << pt.w << endl;
            MFEM_ASSERT(ip.weight > 0, "integration point weight is negative in domain integration from Saye's method");
            MFEM_ASSERT((phi(pt.x) < tol), " phi = " << phi(pt.x) << " : "
                                                     << " levelset function positive at the quadrature point domain integration (Saye's method)");
        }
        cutSquareIntRules[elemid] = ir;
    }
}

/// get integration rule for cut segments
template <int N, int ls>
void GetCutSegmentIntRule(Mesh *mesh, vector<int> cutelems, vector<int> cutinteriorFaces,
                          int order, double radius, std::map<int, IntegrationRule *> &cutSegmentIntRules,
                          std::map<int, IntegrationRule *> &cutInteriorFaceIntRules)
{
    for (int k = 0; k < cutelems.size(); ++k)
    {
        IntegrationRule *ir;
        blitz::TinyVector<double, N> xmin;
        blitz::TinyVector<double, N> xmax;
        blitz::TinyVector<double, N> xupper;
        blitz::TinyVector<double, N> xlower;
        int side;
        int dir;
        double tol = 1e-16;
        // standard reference element
        xlower = {0, 0};
        xupper = {1, 1};
        int elemid = cutelems.at(k);
        findBoundingBox<N>(mesh, elemid, xmin, xmax);
        // cout << "bounding box " << endl;
        // cout << xmin[0] << " , " << xmax[0] << endl;
        // cout << xmin[1] << " , " << xmax[1] << endl;
        vortex<N, ls> phi;
        phi.xscale = xmax[0] - xmin[0];
        phi.yscale = xmax[1] - xmin[1];
        phi.xmin = xmin[0];
        phi.ymin = xmin[1];
        phi.radius = radius;
        if (ls == 3)
        {
            phi.ax = 0.5;
            phi.ay = phi.ax / 10.0;
            phi.xc = 20.0;
            phi.yc = 20.0;
        }
        if (ls == 4)
        {
            phi.xc = 20.0;
            phi.yc = 20.0;
        }
        
        dir = N;
        side = -1;
        auto q = Algoim::quadGen<N>(phi, Algoim::BoundingBox<double, N>(xlower, xupper), dir, side, order);
        MFEM_ASSERT(q.nodes.size() > 0, "int rule not defined by Saye's method");
        int i = 0;
        ir = new IntegrationRule(q.nodes.size());
        //cout << "curved face int rule for " << elemid << endl;
        for (const auto &pt : q.nodes)
        {
            IntegrationPoint &ip = ir->IntPoint(i);
            ip.x = pt.x[0];
            ip.y = pt.x[1];
            ip.weight = pt.w;
            i = i + 1;
            // cout << "elem " << elemid << " , " << ip.weight << endl;
            double xqp = (pt.x[0] * phi.xscale) + phi.xmin;
            double yqp = (pt.x[1] * phi.yscale) + phi.ymin;
            // cout << "int rule " << xqp << " , " << yqp << " : " << pt.w << endl;
            //cout << "level_set func value " << phi(pt.x) << endl;
            MFEM_ASSERT(ip.weight > 0, "integration point weight is negative in curved surface int rule from Saye's method");
        }
        cutSegmentIntRules[elemid] = ir;
        Array<int> orient;
        Array<int> fids;
        mesh->GetElementEdges(elemid, fids, orient);
        int fid;
        //cout << setprecision(11) << endl;
        for (int c = 0; c < fids.Size(); ++c)
        {
            fid = fids[c];
            if (find(cutinteriorFaces.begin(), cutinteriorFaces.end(), fid) != cutinteriorFaces.end())
            {
                if (cutInteriorFaceIntRules[fid] == NULL)
                {
                    //cout << "face int rule for " << fid << endl;
                    Array<int> v;
                    mesh->GetEdgeVertices(fid, v);
                    double *v1coord, *v2coord;
                    v1coord = mesh->GetVertex(v[0]);
                    v2coord = mesh->GetVertex(v[1]);
                    if (abs(v1coord[0] - v2coord[0]) < 1e-15)
                    {
                        dir = 0;
                        //if (v1coord[0] > v2coord[1])
                        if (abs(v1coord[0] - xmax[0]) > 1e-15)
                        {
                            side = 0;
                        }
                        else
                        {
                            side = 1;
                        }
                    }
                    else
                    {
                        dir = 1;
                        if (abs(v1coord[1] - xmax[1]) > 1e-15)
                        {
                            side = 0;
                        }
                        else
                        {
                            side = 1;
                        }
                    }

                    // cout << "dir " << dir << endl;
                    // cout << "side " << side << endl;

                    auto q = Algoim::quadGen<N>(phi, Algoim::BoundingBox<double, N>(xlower, xupper), dir, side, order);

                    int i = 0;
                    ir = new IntegrationRule(q.nodes.size());

                    for (const auto &pt : q.nodes)
                    {

                        IntegrationPoint &ip = ir->IntPoint(i);
                        ip.y = 0.0;
                        if (dir == 0)
                        {
                            if (v1coord[1] < v2coord[1])
                            {
                                if (-1 == orient[c])
                                {
                                    ip.x = 1 - pt.x[1];
                                }
                                else
                                {
                                    ip.x = pt.x[1];
                                }
                            }
                            else
                            {
                                if (1 == orient[c])
                                {
                                    ip.x = 1 - pt.x[1];
                                }
                                else
                                {
                                    ip.x = pt.x[1];
                                }
                            }
                        }
                        else if (dir == 1)
                        {
                            if (v1coord[0] < v2coord[0])
                            {
                                if (-1 == orient[c])
                                {
                                    ip.x = 1.0 - pt.x[0];
                                }
                                else
                                {
                                    ip.x = pt.x[0];
                                }
                            }
                            else
                            {
                                if (1 == orient[c])
                                {
                                    ip.x = 1.0 - pt.x[0];
                                }
                                else
                                {
                                    ip.x = pt.x[0];
                                }
                            }
                        }
                        ip.weight = pt.w;
                        i = i + 1;
                        FaceElementTransformations *trans;
                        trans = mesh->GetInteriorFaceTransformations(fid);
                        IntegrationPoint eip1;
                        IntegrationPoint eip2;
                        trans->Loc1.Transform(ip, eip1);
                        trans->Loc2.Transform(ip, eip2);

                        Vector x1, x2;

                        //cout << "--- x1 ---" << endl;
                        trans->Elem1->Transform(eip1, x1);
                        //x1.Print();
                        //cout << "--- x2 ---" << endl;
                        trans->Elem2->Transform(eip2, x2);
                        // x2.Print();
                         double r = 1.0;
                         double xc = 20.0;
                         double yc = 20.0;
                         double ax = 0.5;
                         double ay = ax / 10.0;
                         double phi_inner = -1 * ((((x1(0) - xc) * (x1(0) - xc)) / (ax * ax)) + (((x1(1) - yc) * (x1(1) - yc)) / (ay * ay)) - (r * r));
                        //  double phi_inner = (x1(0) * x1(0)) + (x1(1) * x1(1)) - 9.0;
                         MFEM_ASSERT((phi_inner < 0), " phi = " << phi_inner << " : "
                                                                << "levelset function positive at the quadrature point (Saye's method)");

                         // scaled to original element space
                         double xq = (pt.x[0] * phi.xscale) + phi.xmin;
                         double yq = (pt.x[1] * phi.yscale) + phi.ymin;
                         // cout << setprecision(
                         //             11)
                         //      << "int rule " << xq << " , " << yq << " : " << pt.w << endl;
                         // cout << "ymax " << v2coord[1] << " , ymin " << v1coord[1] << endl;
                        //  cout << "phi_inner " << phi_inner << endl;
                        //  cout << "phi(pt.x) " << phi(pt.x) << endl;

                         // MFEM_ASSERT(abs(phi_inner - phi(pt.x)) < 1e-13, "level set funcs not same ");

                         MFEM_ASSERT(ip.weight > 0, "integration point weight is negative from Saye's method");
                         MFEM_ASSERT((phi(pt.x) < tol), " phi = " << phi(pt.x) << " : "
                                                                  << "levelset function positive at the quadrature point (Saye's method)");
                         // MFEM_ASSERT((xq <= (max(v1coord[0], v2coord[0]))) && (xq >= (min(v1coord[0], v2coord[0]))),
                         //             "integration point (xcoord) not on element face (Saye's rule)");
                         MFEM_ASSERT((yq <= (max(v1coord[1], v2coord[1]))) && (yq >= (min(v1coord[1], v2coord[1]))),
                                     "integration point (ycoord) not on element face (Saye's rule)");
                    }

                    cutInteriorFaceIntRules[fid] = ir;
                }
            }
        }
    }
}

/// get integration rule for cut segments
template <int N, int ls>
void GetCutBdrSegmentIntRule(Mesh *mesh, vector<int> cutelems, vector<int> cutBdrFaces,
                             int order, double radius, std::map<int, IntegrationRule *> &cutBdrFaceIntRules)
{
    for (int k = 0; k < cutelems.size(); ++k)
    {
        IntegrationRule *ir;
        blitz::TinyVector<double, N> xmin;
        blitz::TinyVector<double, N> xmax;
        blitz::TinyVector<double, N> xupper;
        blitz::TinyVector<double, N> xlower;
        int side;
        int dir;
        double tol = 1e-16;
        // standard reference element
        xlower = {0, 0};
        xupper = {1, 1};
        int elemid = cutelems.at(k);
        findBoundingBox<N>(mesh, elemid, xmin, xmax);
        vortex<N, ls> phi;
        phi.xscale = xmax[0] - xmin[0];
        phi.yscale = xmax[1] - xmin[1];
        phi.xmin = xmin[0];
        phi.ymin = xmin[1];
        phi.radius = radius;
        Array<int> orient;
        Array<int> fids;
        mesh->GetElementEdges(elemid, fids, orient);
        int fid;
        for (int c = 0; c < fids.Size(); ++c)
        {
            fid = fids[c];
            if (find(cutBdrFaces.begin(), cutBdrFaces.end(), fid) != cutBdrFaces.end())
            {
                if (cutBdrFaceIntRules[elemid] == NULL)
                {
                    //cout << "bdr face int rule for " << fid << endl;
                    Array<int> v;
                    mesh->GetEdgeVertices(fid, v);
                    double *v1coord, *v2coord;
                    v1coord = mesh->GetVertex(v[0]);
                    v2coord = mesh->GetVertex(v[1]);
                    // cout << " x vert " << v1coord[0] << " , " << v2coord[0] << endl;
                    // cout << " y vert " << v1coord[1] << " , " << v2coord[1] << endl;
                    // cout << abs(v1coord[0] - v2coord[0]) << endl;
                    if (abs(v1coord[0] - v2coord[0]) < 1e-15)
                    {
                        dir = 0;

                        if (abs(v1coord[0] - xmax[0]) > 1e-15)
                        {
                            side = 0;
                        }
                        else
                        {
                            side = 1;
                        }
                    }
                    else
                    {
                        dir = 1;
                        if (abs(v1coord[1] - xmax[1]) > 1e-15)
                        {
                            side = 0;
                        }
                        else
                        {
                            side = 1;
                        }
                    }

                    // cout << "dir " << dir << endl;
                    // cout << "side " << side << endl;

                    auto q = Algoim::quadGen<N>(phi, Algoim::BoundingBox<double, N>(xlower, xupper), dir, side, order);
                    int i = 0;
                    ir = new IntegrationRule(q.nodes.size());
                    for (const auto &pt : q.nodes)
                    {
                        IntegrationPoint &ip = ir->IntPoint(i);
                        ip.y = 0.0;
                        if (dir == 0)
                        {
                            if (v1coord[1] < v2coord[1])
                            {
                                if (-1 == orient[c])
                                {
                                    ip.x = 1 - pt.x[1];
                                }
                                else
                                {
                                    ip.x = pt.x[1];
                                }
                            }
                            else
                            {
                                if (1 == orient[c])
                                {
                                    ip.x = 1 - pt.x[1];
                                }
                                else
                                {
                                    ip.x = pt.x[1];
                                }
                            }
                        }
                        else if (dir == 1)
                        {
                            if (v1coord[0] < v2coord[0])
                            {
                                if (-1 == orient[c])
                                {
                                    ip.x = 1.0 - pt.x[0];
                                }
                                else
                                {
                                    ip.x = pt.x[0];
                                }
                            }
                            else
                            {
                                if (1 == orient[c])
                                {
                                    ip.x = 1.0 - pt.x[0];
                                }
                                else
                                {
                                    ip.x = pt.x[0];
                                }
                            }
                        }
                        ip.weight = pt.w;
                        i = i + 1;
                                    // scaled to original element space

                        double xq = (pt.x[0] * phi.xscale) + phi.xmin;
                        double yq = (pt.x[1] * phi.yscale) + phi.ymin;
                        // cout << setprecision(
                        //             11)
                        //      << "int rule " << xq << " , " << yq << " : " << pt.w << endl;
                        // cout << "ymax " << v2coord[1] << " , ymin " << v1coord[1] << endl;

                        // cout << "phi(pt.x) " << phi(pt.x) << endl;

                        MFEM_ASSERT(ip.weight > 0, "integration point weight is negative from Saye's method");
                        MFEM_ASSERT((phi(pt.x) < tol), " phi = " << phi(pt.x) << " : "
                                                                 << "levelset function positive at the quadrature point (Saye's method)");
                        MFEM_ASSERT((xq <= (max(v1coord[0], v2coord[0]))) && (xq >= (min(v1coord[0], v2coord[0]))),
                                    "integration point (xcoord) not on element face (Saye's rule)");
                        MFEM_ASSERT((yq <= (max(v1coord[1], v2coord[1]))) && (yq >= (min(v1coord[1], v2coord[1]))),
                                    "integration point (ycoord) not on element face (Saye's rule)");
                    }
                    cutBdrFaceIntRules[elemid] = ir;
                }
            }
        }
    }
}