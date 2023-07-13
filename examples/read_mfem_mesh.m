function mesh = read_mfem_mesh(filename)
fid = fopen(filename, 'r');
str = textscan(fid, '%s', 'Delimiter', '\n');
str = str{1};

%% Find Element starting point
for i = 1 : length(str)
    cur_line = str{i};
    if strcmp(cur_line, 'elements')
        break;
    end
end
%% Convert Element Data
i = i + 1;
cur_line = str{i};
nrElem = str2double(cur_line);
i0 = i;
elements = str2num(strjoin(str(i0 + 1 : i0 + nrElem), ';')); %#ok
elements = elements(:,3:end).' + 1;
i0 = i0 + nrElem;

%% Find Vertex Starting Point
for i = i0 + 1 : length(str)
    cur_line = str{i};
    if strcmp(cur_line, 'vertices')
        break;
    end
end
%% Convert Vertex Data
i = i + 1;
cur_line = str{i};
nrVerts = str2double(cur_line);
i = i + 1;
% cur_line = str{i};
% dim = str2double(cur_line);
i0 = i;
V = str2num(strjoin(str(i0 + 1 : i0 + nrVerts), ';')); %#ok

%% Create SDGMesh
V = num2cell(V, 1);
mesh = SDGMesh(elements, V{:});