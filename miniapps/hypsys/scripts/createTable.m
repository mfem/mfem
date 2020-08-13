function createTable()

q = 1; % q=1: L1, q=2: L2, q=3 LInf errors
domainSize = 2;

numElPerDim = [48 64 96 128 192 256 384]' / domainSize;
numLvls = length(numElPerDim);
maxOrd = 4;
data = zeros(numLvls, maxOrd);
eoc = zeros(numLvls-1, 3);

for j = 1:maxOrd
  file = fopen([num2str(j) '.txt'], 'r');
  for i = 1:numLvls+1-j
    aux = str2num(fgets(file));
    data(i,j) = aux(q);
  end
  fclose(file);
end

tab = [];
for i=1:maxOrd
  eoc(:,i) = log(data(2:end,i) ./ data(1:end-1,i)) ./ log(numElPerDim(1:end-1) ./ numElPerDim(2:end));
  tab = [tab data(2:end,i), eoc(:,i)];
end

% tab
% return

file = fopen('table.txt','wt');
fprintf(file, '\\begin{table}[ht!]\n\\centering\n\\begin{tabular}{||c||c|c||c|c||c|c||c|c||}\n\\hline\n$1/h$ & $p=1$ & EOC & $p=2$ & EOC & $p=3$ & EOC & $p=4$ & EOC\\\\\n\\hline\n');
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
fprintf(file, ['\\hline\n\\end{tabular}\n\\caption{The $\\|\\cdot\\|_{L^' e '(\\Omega)}$~errors and corresponding EOC of ...$\\mathbb Q_p$, $p \\in \\{1,\\hdots,4\\}$ solutions to the 1D ... equation with initial condition ...}\\label{tab:}\n\\end{table}']);
fclose(file);
end
