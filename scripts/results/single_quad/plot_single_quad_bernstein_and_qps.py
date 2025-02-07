#paraview.compatibility.major = 5
#paraview.compatibility.minor = 11
from paraview.simple import *

path = '/Users/mittal3/LLNL/mfem-detJ/mfem/examples/ParaView/single_quad_bernstein/single_quad_bernstein.pvd'
solution = PVDReader(FileName=path)
solution2 = PVDReader(FileName=path)

renderView1 = GetActiveViewOrCreate('RenderView')
solutionDisplay = GetDisplayProperties(solution, view=renderView1)
Show(solution, renderView1)
ColorBy(solutionDisplay, ('POINTS', 'solution'))
solutionDisplay.NonlinearSubdivisionLevel=3
solutionDisplay.SetScalarBarVisibility(renderView1, True)
renderView1 = GetActiveViewOrCreate('RenderView')

solutionDisplay2 = GetDisplayProperties(solution2, view=renderView1)
Show(solution2, renderView1)
solutionDisplay2.NonlinearSubdivisionLevel=3
# ColorBy(solutionDisplay2, None)
solutionDisplay2.SetRepresentationType('Wireframe')

renderView1.OrientationAxesVisibility = 0
renderView1.UseColorPaletteForBackground = 0
renderView1.Background = [1.0, 1.0, 1.0]
renderView1.CameraParallelScale = 0.8

LUT = GetColorTransferFunction('solution')
LUT.ApplyPreset('Rainbow Uniform', True)
LUT.NumberOfTableValues = 1024
LUT.RescaleTransferFunctionToDataRange()
LUTColorBar = GetScalarBar(LUT, renderView1)
LUTColorBar.RangeLabelFormat = '%-#6.0g'
LUTColorBar.TextPosition = 'Ticks right/top, annotations left/bottom'
LUTColorBar.LabelColor = [0, 0, 0]
LUTColorBar.TitleColor = [1, 1, 1]
LUTColorBar.LabelBold = 1
LUTColorBar.TitleBold = 1
LUTColorBar.LabelFontSize = 12
LUTColorBar.TitleFontSize = 14
LUTColorBar.ComponentTitle = ''
LUTColorBar.Orientation = 'Horizontal'
LUTColorBar.WindowLocation = 'Any Location'
LUTColorBar.ScalarBarLength = 0.6
LUTColorBar.ScalarBarThickness = 12
LUTColorBar.Position = [0.2, 0.02]
LUTColorBar.HorizontalTitle = False
LUTColorBar.AddRangeLabels = 1
LUTColorBar.RangeLabelFormat = '%g'
LUTColorBar.Title = 'min det(J)'
# LUT.RescaleTransferFunction(-0.0002, 2.1)
solutionDisplay.SetScalarBarVisibility(renderView1, False)
outpath = path.replace('.pvd', f'_detj.jpg')
SaveScreenshot(outpath, renderView1, ImageResolution=[600, 600])


csv_file = CSVReader(FileName=['/Users/mittal3/LLNL/mfem-detJ/mfem/examples/single_quad_nodes.txt'])
tableToPoints = TableToPoints(Input=csv_file)
tableToPoints.XColumn = "x"
tableToPoints.YColumn = "y"
tableToPoints.ZColumn = ""
tableToPoints.a2DPoints = 1
pointDisplay = Show(tableToPoints, renderView1)
pointDisplay.SetRepresentationType("Points")
pointDisplay.PointSize = 10
pointDisplay.RenderPointsAsSpheres = 1
ColorBy(pointDisplay, ("POINTS", "color"))
outpath = path.replace('.pvd', f'_nodes.jpg')
SaveScreenshot(outpath, renderView1, ImageResolution=[600, 600])
Hide(tableToPoints, renderView1)


for qo in range(2,20):
    txt_path = f'/Users/mittal3/LLNL/mfem-detJ/mfem/examples/single_quad_qps_{qo}.txt'
    csv_file2 = CSVReader(FileName=[txt_path])
    tableToPoints2 = TableToPoints(Input=csv_file2)
    tableToPoints2.XColumn = "x"
    tableToPoints2.YColumn = "y"
    tableToPoints2.ZColumn = ""
    tableToPoints2.a2DPoints = 1
    pointDisplay2 = Show(tableToPoints2, renderView1)
    pointDisplay2.SetRepresentationType("Points")
    pointDisplay2.PointSize = 10
    pointDisplay2.RenderPointsAsSpheres = 1
    ColorBy(pointDisplay2, ("POINTS", "color"))
    outpath = path.replace('.pvd', f'_qp_{qo}.jpg')
    SaveScreenshot(outpath, renderView1, ImageResolution=[600, 600])
    Hide(tableToPoints2, renderView1)


animationScene1 = GetAnimationScene()
for t in range(1,4):
    animationScene1.GoToNext()
    LUT.RescaleTransferFunctionToDataRange()
    outpath = path.replace('.pvd', f'_detj_time_{t}.jpg')
    SaveScreenshot(outpath, renderView1, ImageResolution=[600, 600])