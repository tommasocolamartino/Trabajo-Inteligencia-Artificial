function [data_norm,pos_data_red,media,std_dev] = normalizacion(data)
media = mean(data')';
std_dev = std(data')'; 
% Para la normalización utilizo solo datos con std ~= 0
pos_data_red = find(std_dev~=0);
data_red = data(pos_data_red,:);
% Normalización
data_norm = (data_red - media(pos_data_red))./(std_dev(pos_data_red));
end

