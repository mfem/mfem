from paraview.simple import *
paraview.simple._DisableFirstRenderCameraReset()
path='/Users/mittal3/LLNL/mfem-detJ/mfem/examples/ParaView/mesh_detj_bounds_comparison/mesh_detj_bounds_comparison'

solution = PVDReader(FileName=path+'.pvd')

solution2 = PVDReader(FileName=path+'.pvd')

outpath = path+'.jpg'

renderView1 = GetActiveViewOrCreate('RenderView')
solutionDisplay = GetDisplayProperties(solution, view=renderView1)
Show(solution, renderView1)
ColorBy(solutionDisplay, ('POINTS', 'cusboundmin'))
solutionDisplay.NonlinearSubdivisionLevel = 3

solution2Display = GetDisplayProperties(solution2, view=renderView1)
Show(solution2, renderView1)
solution2Display.SetRepresentationType('Wireframe')
solution2Display.AmbientColor = [0.0, 0.0, 0.0]
solution2Display.DiffuseColor = [0.0, 0.0, 0.0]
solution2Display.LineWidth = 1.5
solution2Display.NonlinearSubdivisionLevel = 3

LUT = GetColorTransferFunction('cusboundmin')
LUT.ApplyPreset('Cool to Warm', True)

LUTColorBar = GetScalarBar(LUT, renderView1)
LUTColorBar.RangeLabelFormat = '%-#6.0g'
LUTColorBar.TextPosition = 'Ticks right/top, annotations left/bottom'
LUTColorBar.LabelColor = [1.0, 1.0, 1.0]
LUTColorBar.TitleColor = [1.0, 1.0, 1.0]
LUTColorBar.LabelBold = 0
LUTColorBar.TitleBold = 0
LUTColorBar.LabelFontSize = 16
LUTColorBar.TitleFontSize = 20
LUTColorBar.Title = '$\\rho$'
LUTColorBar.ComponentTitle = ''
LUTColorBar.Orientation = 'Vertical'
LUTColorBar.WindowLocation = 'Any Location'
LUTColorBar.ScalarBarLength = 0.4
LUTColorBar.ScalarBarThickness = 12
LUTColorBar.Position = [0.88, 0.55]
LUTColorBar.HorizontalTitle = False
LUTColorBar.AddRangeLabels = 1
LUTColorBar.RangeLabelFormat = '%g'


renderView1.OrientationAxesVisibility = 0
renderView1.CameraPosition = [3.4999999999999996, 1.4021470621217202, 14.71254386022332]
renderView1.CameraFocalPoint = [3.4999999999999996, 1.4021470621217202, 0.0]
renderView1.CameraParallelScale = 2.24
renderView1.Background = [1, 1, 1]
renderView1.CameraViewUp = [0.0, 1.0, 0.0]

SaveScreenshot(outpath, renderView1, ImageResolution=[2100, 900])