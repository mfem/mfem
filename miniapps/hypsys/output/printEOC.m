function printEOC(filename)

file = fopen(filename, 'r');
data = zeros(0,3);
while true
  aux = fgets(file);
  if aux == -1
    break;
  end
  data(end+1,:) = str2num(aux);
end

fclose(file);

% numElPerDim = [48 64 96 128 192 256 384]';
numElPerDim = [12 24 48 96 192 384]';
numElPerDim = numElPerDim(1:size(data,1));
numLvls = length(numElPerDim)-1;
eoc = zeros(numLvls, 3);
for i=1:3
  eoc(:,i) = log(data(2:end,i) ./ data(1:end-1,i)) ./ log(numElPerDim(1:end-1) ./ numElPerDim(2:end));
end

tab = [data(2:end,1), eoc(:,1), data(2:end,2), eoc(:,2), data(2:end,3), eoc(:,3)];

fprintf('\n1/%d  & %1.2E &      & %1.2E &      & %1.2E & \\\\\n', numElPerDim(1), data(1,:));
for i = 2:length(numElPerDim)
  if numElPerDim(i) < 100
    fprintf('1/%d  & %1.2E & %1.2f & %1.2E & %1.2f & %1.2E & %1.2f \\\\\n', numElPerDim(i), tab(i-1,:));
  else
    fprintf('1/%d & %1.2E & %1.2f & %1.2E & %1.2f & %1.2E & %1.2f \\\\\n', numElPerDim(i), tab(i-1,:));
  end
end
fprintf('\n');
end
