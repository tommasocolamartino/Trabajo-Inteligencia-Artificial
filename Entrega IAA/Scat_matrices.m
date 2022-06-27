function [Sb,Sw] = Scat_matrices(data,label)
% Separación de los datos por clase
% in = 1;
[r,~] = size(data);
Sw = zeros(r,r);
Sb = zeros(r,r);

% Count the number of differrent labels
Num_label = 0;
Type_labels = [];
for i = 1:length(label)
    if find(Type_labels==label(i))
        
    else
        Num_label = Num_label+1;
        Type_labels(Num_label) = label(i);
    end        
end

for i = 0:Num_label-1
    pos = find(label==i); %devuelve pos. columnas de img con num. i
%   fin = in + length(pos)-1;
%   data_ord(:, in:fin) = data(:,pos);
%   n(i) = fin-in+1;
    data_clase = data(:,pos);
    n = length(pos);
    m_c = mean(data_clase')';
    
    Sc = cov(data_clase')*(n-1);
    Sw = Sw+Sc;
    Sb = Sb + n*(m_c*m_c');
%   in = length(data_ord)+1;
end

end

