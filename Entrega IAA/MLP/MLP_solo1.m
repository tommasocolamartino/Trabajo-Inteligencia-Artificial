%% MLP
% 1. PreProcessing data

% PCA
clc
clearvars
close all
addpath ('C:\Users\tomma\MATLAB Drive\Inteligencia Artificial Aplicada\Progetto')
addpath ('C:\Users\tomma\MATLAB Drive\Inteligencia Artificial Aplicada\Progetto\PCA')
load Trainnumbers
load PCA_data
label = Trainnumbers.label;

% 2. Dividing in training and test
% 3. Training and Classification

% Initialize out of the cycle the non-changing parameters
hiddenSizes = [25 15];
trainFcn = 'trainlm';
grupo = 1000;%length(datos_procesar)/10; %datos_procesar en este caso son los datos normalizados
N_comp_princ = 30;

% Arranging the labels in a RxQ matrix, where R are the possible outputs
% and Q the batch size
[true_label_MLP] = label_MLP_fun(label);

W = PCA_fun(Wc,lambda,N_comp_princ);
PCA_data = W'*datos_procesar;

for i = 1:10
% Clear the net each time you test it again
clear net

% Set the net parameters
net = feedforwardnet(hiddenSizes,trainFcn);
net.layers{1}.transferFcn = 'tansig';
net.layers{2}.transferFcn = 'tansig';
% net.layers{3}.transferFcn = 'tansig';
net.layers{3}.transferFcn = 'softmax';
net.performFcn = 'mse';

net.trainParam.min_grad = 1e-7;
net.trainParam.showWindow = true;
net.trainParam.epochs = 30; %30 top

% Select Train and Test data
ind_random = randperm(length(datos_procesar));
pos_test_data = ind_random(1:grupo);
pos_train_data = ind_random(grupo+1:end);        

train_data = PCA_data(:,pos_train_data);
label_train_data = true_label_MLP(:,pos_train_data);
test_data = PCA_data(:,pos_test_data);
label_test_data = true_label_MLP(:,pos_test_data);

% Train and Test
net = train(net,train_data,label_train_data);
label_nn_MLP = net(test_data);

label_nn = vec2ind(label_nn_MLP)-1;

n_errors(i)=length(find(label_nn~=label(:,pos_test_data)))
end

var_error = std(n_errors); 
misclassification = 100-mean(n_errors)/grupo*100;


%%
clear net

% Set the net parameters
net = feedforwardnet(hiddenSizes,trainFcn);
net.layers{1}.transferFcn = 'tansig';
net.layers{2}.transferFcn = 'tansig';
% net.layers{3}.transferFcn = 'tansig';
net.layers{3}.transferFcn = 'softmax';
net.performFcn = 'mse';

net.trainParam.min_grad = 1e-7;
net.trainParam.showWindow = true;
net.trainParam.epochs = 30; %30 top

% Select Train and Test data
ind_random = randperm(length(datos_procesar));
pos_test_data = ind_random(1:grupo);
pos_train_data = ind_random(grupo+1:end);        

train_data = PCA_data(:,pos_train_data);
label_train_data = true_label_MLP(:,pos_train_data);
test_data = PCA_data(:,pos_test_data);
label_test_data = true_label_MLP(:,pos_test_data);

% Train and Test
net = train(net,train_data,label_train_data);
label_nn_MLP = net(test_data);

label_nn = vec2ind(label_nn_MLP)-1;

n_errors=length(find(label_nn~=label(:,pos_test_data)))

figure
cm = confusionchart(label(pos_test_data),label_nn);
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
cm.Title = 'MLP Model Confusion Matrix'; 