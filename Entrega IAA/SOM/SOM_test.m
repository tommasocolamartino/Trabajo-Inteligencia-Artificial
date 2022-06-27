%% SOM TEST
close all
clc
clearvars

addpath('..\')
load Trainnumbers
load Test_numbers_HW2
load SOM_net

% 1:
train_data = Trainnumbers.image;
test_data = Test_numbers.image;
label_train_data = Trainnumbers.label;

% 2:
[data_norm_train,data_norm_test] = norm_train_test(train_data,test_data);

% 3:
[Wc,lambda] = eig(cov(data_norm_train'), 'vector');

% 4:
cp = 25;
W = PCA_fun(Wc,lambda,cp);
PCA_data_train = W'*data_norm_train;
PCA_data_test = W'*data_norm_test;

% 5:
Test_neur_activ = vec2ind(net(PCA_data_test));
class = (neuron_labels(Test_neur_activ))';

%%
T= table(class);
filename = 'data_testsom.xlsx';
writetable(T,filename)

PCA = 25;
name = {'TommasoC'};
save('M21025_som.mat', "name", "PCA", "class")