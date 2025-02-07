#paraview.compatibility.major = 5
#paraview.compatibility.minor = 11
from paraview.simple import *


boundpath = '/Users/mittal3/LLNL/mfem-detJ/mfem/scripts/bounds_max_M4_r3.vtu'
lboundpath = '/Users/mittal3/LLNL/mfem-detJ/mfem/scripts/bounds_min_M4_r3.vtu'
solpath = '/Users/mittal3/LLNL/mfem-detJ/mfem/scripts/sol.vtu'

paraview.simple._DisableFirstRenderCameraReset()
solvtu = XMLUnstructuredGridReader(FileName=[solpath])

renderView1 = GetActiveViewOrCreate('RenderView')
renderView1.OrientationAxesVisibility = 0
renderView1.UseColorPaletteForBackground = 0
renderView1.Background = [1.0, 1.0, 1.0]

solvtuDisplay = Show(solvtu, renderView1, 'UnstructuredGridRepresentation')
solvtuDisplay.Ambient = 0.25
solvtuDisplay.SetScalarBarVisibility(renderView1, False)

ColorBy(solvtuDisplay, ('POINTS', 'u'))
uLUT = GetColorTransferFunction('u')
uLUT.ApplyPreset('Rainbow Uniform', True)
uLUT.NumberOfTableValues = 10

# create a new 'Contour'
contour1 = Contour(Input=solvtu)
contour1.Isosurfaces = [-7.60548e-05, 0.08307875068, 0.16623355616, 0.24938836164, 0.33254316712,
                        0.41569797259999997, 0.49885277808, 0.58200758356, 0.66516238904, 0.74831719452, 0.831472]

contour1Display = Show(contour1, renderView1, 'GeometryRepresentation')
ColorBy(contour1Display, None)
contour1Display.LineWidth = 2.0

bounds = XMLUnstructuredGridReader(FileName=[boundpath])
bounds_Display = Show(bounds, renderView1, 'UnstructuredGridRepresentation')
bounds_Display.NonlinearSubdivisionLevel = 3
bounds_Display.Opacity = 0.5

bounds_2 = XMLUnstructuredGridReader(FileName=[boundpath])
bounds_2Display = Show(bounds_2, renderView1, 'UnstructuredGridRepresentation')
bounds_2Display.NonlinearSubdivisionLevel = 3
bounds_2Display.SetRepresentationType('Wireframe')
bounds_2Display.AmbientColor = [0.0, 0.0, 0.0]
bounds_2Display.DiffuseColor = [0.0, 0.0, 0.0]
bounds_2Display.LineWidth = 1.5

bounds_3 = XMLUnstructuredGridReader(FileName=[boundpath])
bounds_3Display = Show(bounds_3, renderView1, 'UnstructuredGridRepresentation')
bounds_3Display.NonlinearSubdivisionLevel = 1
bounds_3Display.SetRepresentationType('Points')
bounds_3Display.AmbientColor = [0.0, 0.0, 0.0]
bounds_3Display.DiffuseColor = [0.0, 0.0, 0.0]
bounds_3Display.PointSize = 4

lbounds = XMLUnstructuredGridReader(FileName=[lboundpath])
lbounds_Display = Show(lbounds, renderView1, 'UnstructuredGridRepresentation')
lbounds_Display.NonlinearSubdivisionLevel = 3
lbounds_Display.Opacity = 0.5

lbounds_2 = XMLUnstructuredGridReader(FileName=[lboundpath])
lbounds_2Display = Show(lbounds_2, renderView1, 'UnstructuredGridRepresentation')
lbounds_2Display.NonlinearSubdivisionLevel = 3
lbounds_2Display.SetRepresentationType('Wireframe')
lbounds_2Display.AmbientColor = [0.0, 0.0, 0.0]
lbounds_2Display.DiffuseColor = [0.0, 0.0, 0.0]
lbounds_2Display.LineWidth = 1.5

lbounds_3 = XMLUnstructuredGridReader(FileName=[lboundpath])
lbounds_3Display = Show(lbounds_3, renderView1, 'UnstructuredGridRepresentation')
lbounds_3Display.NonlinearSubdivisionLevel = 1
lbounds_3Display.SetRepresentationType('Points')
lbounds_3Display.AmbientColor = [0.0, 0.0, 0.0]
lbounds_3Display.DiffuseColor = [0.0, 0.0, 0.0]
lbounds_3Display.PointSize = 4

renderView1.CameraParallelScale = 0.8
renderView1.CameraPosition = [0.5, 0.5, 3.2037576157783714]
renderView1.CameraFocalPoint = [0.5, 0.5, 0.47777678503189236]
renderView1.CameraViewUp = [0,1,0]
renderView1.CameraParallelScale = 0.8536982642537119

outpath = boundpath.replace('.vtu', f'_bound_recursion.jpg')
SaveScreenshot(outpath, renderView1, ImageResolution=[1800, 1800])

renderView1.CameraPosition = [-0.6950484258590709, -1.8916546047568064, -0.05403051182923655]
renderView1.CameraFocalPoint = [0.5000000000000003, 0.5000000000000008, 0.4777767850318916]
renderView1.CameraViewUp = [-0.04890250236941325, -0.1934526212408308, 0.9798901104700778]
renderView1.CameraParallelScale = 0.8
outpath = boundpath.replace('.vtu', f'_bound_recursion_3D.jpg')
SaveScreenshot(outpath, renderView1, ImageResolution=[1800, 1800])