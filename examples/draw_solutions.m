meshfiles = dir('mesh*.mesh');
ufiles = dir('u*.gf');
rhofiles = dir('rho*.gf');
frhofiles = dir('f_rho*.gf');

ax = arrayfun(@(i) axes(figure()), 1:3);
figs = arrayfun(@(ax) ax.Parent, ax);
recorders = arrayfun(@(name, fig) FigureRecorder(name, fig), ["u.mp4", "rho.mp4", "frho.mp4"], figs);
for i = 1 : length(meshfiles)
    mesh = read_mfem_mesh([meshfiles(i).folder filesep meshfiles(i).name]);

    contour_mfem_gf(ax(1), mesh, [ufiles(i).folder filesep ufiles(i).name], 11);
    title(ax(1), sprintf('Iteration %d: u', i))

    contour_mfem_gf(ax(2), mesh, [rhofiles(i).folder filesep rhofiles(i).name], 0.1:0.1:0.9);
    title(ax(2), sprintf('Iteration %d: ρ', i))
    set(ax(2), 'Clim', [0, 1]);
    colormap(ax(2), flipud(gray));

    contour_mfem_gf(ax(3), mesh, [frhofiles(i).folder filesep frhofiles(i).name], 0.1:0.1:0.9);
    title(ax(3), sprintf('Iteration %d: ρ̃', i))
    set(ax(3), 'Clim', [0, 1]);
    colormap(ax(3), flipud(gray));

    arrayfun(@(a) view(a, 2), ax);
    arrayfun(@(a) set(a, 'FontSize', 16), ax);
    arrayfun(@(a) colorbar(a), ax);
    arrayfun(@(a) axis(a, 'off'), ax)
    arrayfun(@(recorder) recorder.capture(), recorders);
end
arrayfun(@(recorder) recorder.close(), recorders);