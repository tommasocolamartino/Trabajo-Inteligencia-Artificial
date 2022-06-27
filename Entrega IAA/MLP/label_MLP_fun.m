function [label_MLP] = label_MLP_fun(label)
% Realize an RxQ matrix from a vector of labels
l = length(label);
label_MLP = zeros(10,l);
for i = 1:l
    pos = label(i)+1;
    label_MLP(pos,i) = 1;
end
end

