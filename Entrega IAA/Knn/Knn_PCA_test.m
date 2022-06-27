%% KNN CLASSIFIER
% STEPS: 
% 1. Divide between test and train data
% 2. Normalize the train data and with mean and std of these normalize
%    the test data
% 3. Compute the eigvect and eigval of the covariance matrix of the
%    training data
% 4. Compute W and reduce the dimensionality of train and test data
% 5. train and test the model
close all
clearvars
clc
addpath ('..\')
load Trainnumbers
load Test_numbers_HW2.mat

% 1:
train_data = Trainnumbers.image;
test_data = Test_numbers.image;
label_train_data = Trainnumbers.label;

% 2:
[data_norm_train,data_norm_test] = norm_train_test(train_data,test_data);

% 3:
[Wc,lambda] = eig(cov(data_norm_train'), 'vector');

% 4:
cp = 45;
W = PCA_fun(Wc,lambda,cp);
PCA_data_train = W'*data_norm_train;
PCA_data_test = W'*data_norm_test;

% 5:
knnMdl = fitcknn(PCA_data_train',label_train_data','NumNeighbors',5);
class = predict(knnMdl,PCA_data_test');      

%%

T= table(class);
filename = 'data_testknn.xlsx';
writetable(T,filename)

PCA = 46;
name = {'TommasoC'};
save('M21025_knn.mat', "name", "PCA", "class")