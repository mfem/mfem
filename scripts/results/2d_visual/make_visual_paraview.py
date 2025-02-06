#paraview.compatibility.major = 5
#paraview.compatibility.minor = 11
from paraview.simple import *

M = 6
r = 1
boundpath = f'C:\\Users\\Tarik-Personal\\Documents\\Research\\MeshBounding\\bounds_max_M{M}_r{r}.vtu'
loboundpath = f'C:\\Users\\Tarik-Personal\\Documents\\Research\\MeshBounding\\bounds_max_M{M}_r{r}_LO.vtu'
solpath = 'C:\\Users\\Tarik-Personal\\Documents\\Research\\MeshBounding\\sol.vtu'

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
uLUT.ApplyPreset('Rainbow Desaturated', True)
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

bounds_3 = XMLUnstructuredGridReader(FileName=[loboundpath])
bounds_3Display = Show(bounds_3, renderView1, 'UnstructuredGridRepresentation')
bounds_3Display.NonlinearSubdivisionLevel = 1
bounds_3Display.SetRepresentationType('Points')
bounds_3Display.AmbientColor = [0.0, 0.0, 0.0]
bounds_3Display.DiffuseColor = [0.0, 0.0, 0.0]
bounds_3Display.PointSize = 4
