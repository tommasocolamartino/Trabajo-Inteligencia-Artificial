clc
clearvars
close all
addpath ('C:\Users\tomma\MATLAB Drive\Inteligencia Artificial Aplicada\Progetto')
load Trainnumbers
%% PCA on the trainnumbers images

Data = Trainnumbers.image;
[Pixeles,N_im] = size(Data);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Normalización de los datos %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[datos_procesar,pos_data_red,media,std_dev] = normalizacion(Data);
n = length(pos_data_red);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calcular las componentes principales %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N_comp_princ = 100; % Cuantas componentes principales quiero
[Wc,Diag] = eig(cov(datos_procesar'));
pos = 1:1:N_comp_princ;
lambda = diag(Diag);
W = PCA_fun(Wc,lambda,N_comp_princ);

PCA_data = W'*datos_procesar; %Datos_proyectados
Datos_reconstruidos_n = W*PCA_data;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Desnormalizar los datos reconstruidos %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Data_reconstruidos_red = Datos_reconstruidos_n.*std_dev(pos_data_red)+media(pos_data_red);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Anadir los pixeles sobre los cuales no se ha trbajado %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Datos_reconstruidos = Data;
Datos_reconstruidos(pos_data_red,:) = Data_reconstruidos_red; 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Imprimir las nuevas imagenes %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
print_9_numbers_range (Datos_reconstruidos,1:9);
print_9_numbers_range (Data,1:9);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%

lambda_ord = sort(lambda,'descend');

tot = sum(lambda_ord);
perc_lambda = lambda_ord/tot*100;
for i = 1:length(lambda_ord)
    lambda_perc_dist(i) = sum(perc_lambda(1:i));
end
figure
plot(lambda_perc_dist)
grid on
xlabel ('N° componentes principales')
ylabel ('Porcentaje información original')
title('N° componentes principales VS (1-MSE)%')
ylim([0,100])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Notice that the rank of the Diag is 664
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

save PCA_data Wc lambda datos_procesar