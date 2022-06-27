% 1. Take the pre-processed data
clearvars
clc
close all
addpath ('..\')

load Trainnumbers
load 'Test_numbers_HW2.mat'
load Bayes_model_test.mat
Data_train = Trainnumbers.image;
Data_test = Test_numbers.image;

[data_norm_train,data_norm_test] = norm_train_test(Data_train,Data_test);
LDA_data_test = W'*data_norm_test;

class = predict(bayMdl,LDA_data_test');
class = floor(class/2);

T= table(class);
filename = 'data_testbay.xlsx';
writetable(T,filename)
%% Guardado
PCA = 19;
name = {'TommasoC'};
save('M21025_bay.mat', "name", "PCA", "class")