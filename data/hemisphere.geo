// Gmsh script: Solid Hemisphere (Half Ball)
// Center: (0, 0, 0.4), Radius: 0.4, lower hemisphere (pointing down)
//
// Uses a single clean volume (no wedge splitting) with the HXT mesher,
// which produces high-quality, well-distributed meshes that are as
// symmetric as an unstructured mesher can achieve without constraints.

SetFactory("OpenCASCADE");

R  = 0.4;   // Radius
lc = 0.05;  // Target mesh size

// Full sphere centered at (0, 0, 0.4)
Sphere(1) = {0, 0, 0.4, R};

// Box covering the lower half: z in [0.4-R, 0.4]
Box(2) = {-R, -R, 0.4-R, 2*R, 2*R, R};

// Intersect: keep only the lower hemisphere
BooleanIntersection{ Volume{1}; Delete; }{ Volume{2}; Delete; }

// --- Physical attributes ---
eps = 1e-6;

// Attribute 2: flat top — entirely at z = 0.4
flat_surfs[] = Surface In BoundingBox {
    -R-eps, -R-eps, 0.4-eps,
     R+eps,  R+eps, 0.4+eps };
Physical Surface(2) = { flat_surfs[] };

// Attribute 1: curved dome — everything else
all_surfs[]  = Surface{:};
dome_surfs[] = {};
For i In {0:#all_surfs[]-1}
    s = all_surfs[i];
    is_flat = 0;
    For j In {0:#flat_surfs[]-1}
        If (s == flat_surfs[j])
            is_flat = 1;
        EndIf
    EndFor
    If (!is_flat)
        dome_surfs[] += {s};
    EndIf
EndFor
Physical Surface(1) = { dome_surfs[] };

// Volume attribute (required for MFEM to write interior elements)
Physical Volume(1) = { Volume{:} };

// --- Mesh settings ---
MeshSize{ PointsOf{ Volume{:}; } } = lc;

// HXT is the fastest and produces the most isotropic, well-distributed
// tetrahedra — giving the best symmetry for an unstructured mesh
Mesh.Algorithm3D = 10;  // HXT

Mesh.ElementOrder = 1;
Mesh.SaveAll = 0;

// Optimize mesh quality after generation
Mesh.Optimize = 1;
Mesh.OptimizeNetgen = 1;
