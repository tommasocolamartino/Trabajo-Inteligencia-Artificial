clearvars
clc
close all
addpath ('C:\Users\tomma\MATLAB Drive\Inteligencia Artificial Aplicada\Progetto')
addpath ('C:\Users\tomma\MATLAB Drive\Inteligencia Artificial Aplicada\Progetto\PCA')
addpath ('C:\Users\tomma\MATLAB Drive\Inteligencia Artificial Aplicada\Progetto\LDA')
load Trainnumbers
load 'Test_numbers_HW2.mat'
load net_25_15.mat

% Save the data
Data_train = Trainnumbers.image;
Data_test = Test_numbers.image;

% Normalización 
[data_norm_train,data_norm_test] = norm_train_test(Data_train,Data_test);

% PCA
cp = 30;
[Wc,Diag] = eig(cov(data_norm_train'));
lambda = diag(Diag);
W = PCA_fun(Wc,lambda,cp);

PCA_data_train = W'*data_norm_train; %Datos_proyectados
PCA_data_test = W'*data_norm_test;

% MLP 
label_nn_MLP = net(PCA_data_test);
label_nn = (vec2ind(label_nn_MLP)-1)';
class = label_nn;
%%
T= table(class);
filename = 'data_testmlp.xlsx';
writetable(T,filename)

PCA = 30;
name = {'TommasoC'};
save('M21025_mlp.mat', "name", "PCA", "class")