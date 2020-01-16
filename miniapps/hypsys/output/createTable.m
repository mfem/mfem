function createTable()

filename = 'p0e0o';
geom = 'tri';

q = 1; % q=1: L1, q=2: L2, q=3 LInf errors
data = zeros(7,4);

for j = 1:4
  file = fopen([filename num2str(j) '-' geom '.txt'], 'r');
  for i = 1:8-j
    aux = str2num(fgets(file));
    data(i,j) = aux(q);
  end
  fclose(file);
end

numElPerDim = [48 64 96 128 192 256 384]';
numLvls = length(numElPerDim);
eoc = zeros(numLvls-1, 3);
for i=1:4
  eoc(:,i) = log(data(2:end,i) ./ data(1:end-1,i)) ./ log(numElPerDim(1:end-1) ./ numElPerDim(2:end));
end

tab = [data(2:end,1), eoc(:,1), data(2:end,2), eoc(:,2), data(2:end,3), eoc(:,3), data(2:end,4), eoc(:,4)];

file = fopen('table.txt','wt');
fprintf(file, '\\begin{table}[h!]\n\\footnotesize{\n\\begin{tabular}{||c||c|c||c|c||c|c||c|c||}\n\\hline\n$1/h$ & $p=1$ & EOC & $p=2$ & EOC & $p=3$ & EOC & $p=4$ & EOC\\\\\n\\hline\n');
fprintf(file, '%d  & %1.2E &      & %1.2E &      & %1.2E &      & %1.2E & \\\\\n', numElPerDim(1), data(1,:));
for i = 2 : numLvls
  switch i
    case {2,3,4}
      if numElPerDim(i) < 100
        fprintf(file, '%d  & %1.2E & %1.2f & %1.2E & %1.2f & %1.2E & %1.2f & %1.2E & %1.2f \\\\\n', numElPerDim(i), tab(i-1,:));
      else
        fprintf(file, '%d & %1.2E & %1.2f & %1.2E & %1.2f & %1.2E & %1.2f & %1.2E & %1.2f \\\\\n', numElPerDim(i), tab(i-1,:));
      end
    case 5
      if numElPerDim(i) < 100
        fprintf(file, '%d  & %1.2E & %1.2f & %1.2E & %1.2f & %1.2E & %1.2f &&\\\\\n', numElPerDim(i), tab(i-1,1:end-2));
      else
        fprintf(file, '%d & %1.2E & %1.2f & %1.2E & %1.2f & %1.2E & %1.2f &&\\\\\n', numElPerDim(i), tab(i-1,1:end-2));
      end
    case 6
      if numElPerDim(i) < 100
        fprintf(file, '%d  & %1.2E & %1.2f & %1.2E & %1.2f &&&&\\\\\n', numElPerDim(i), tab(i-1,1:end-4));
      else
        fprintf(file, '%d & %1.2E & %1.2f & %1.2E & %1.2f &&&&\\\\\n', numElPerDim(i), tab(i-1,1:end-4));
      end
    case 7
      if numElPerDim(i) < 100
        fprintf(file, '%d  & %1.2E & %1.2f &&&&&&\\\\\n', numElPerDim(i), tab(i-1,1:end-6));
      else
        fprintf(file, '%d & %1.2E & %1.2f &&&&&&\\\\\n', numElPerDim(i), tab(i-1,1:end-6));
      end
    otherwise
      error('Change of configuration requires modification of output formatting.');
  end
end
e = num2str(q);
if q==3
  e = '\\infty';
end
fprintf(file, ['\\hline\n\\end{tabular}\n}\n\\caption{$\\|\\cdot\\|_{L^' e '(\\Omega)}$~errors and corresponding EOC for $p \\in \\{1,\\hdots,4\\}$.}\\label{tab:}\n\\end{table}']);
fclose(file);
end
