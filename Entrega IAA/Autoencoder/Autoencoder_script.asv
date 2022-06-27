%autoencoder_data
clearvars
clc
close all
addpath ('..\');
addpath ('..\SOM');
load SOM_net
load Trainnumbers
load autoencoder60 
load autoenc_SOM
Data = Trainnumbers.image;
label = Trainnumbers.label;
[data_norm,~,~,~] = normalizacion(Data);
grupo = 1000;

ind_random = randperm(length(data_norm));
pos_test_data = ind_random(1:grupo);
pos_train_data = ind_random(grupo+1:end);

XTrain = data_norm(:,pos_train_data);
XTest = data_norm(:,pos_train_data);
hiddenSize = 25;
% autoenc = trainAutoencoder(XTrain,hiddenSize,...
%         'L2WeightRegularization',0.004,...
%         'SparsityRegularization',4,...
%         'SparsityProportion',0.15);
    
XReconstructed = predict(autoenc,XTest);
mseError = mse(XTest-XReconstructed)
%% Bayes
Z = encode(autoenc,data_norm);

ind_random = randperm(length(data_norm));
pos_test_data = ind_random(1:grupo);
pos_train_data = ind_random(grupo+1:end);

bayMdl= fitcnb(Z(:,pos_train_data)',label(pos_train_data)');
bayclass= predict(bayMdl,Z(:,pos_test_data)');

n_errors=length(find(bayclass~=label(pos_test_data)'));

misclassification_2 = 100-n_errors/grupo*100

%% Knn
Z = encode(autoenc,data_norm);
KnnMdl=fitcknn(Z(:,pos_train_data)',label(pos_train_data)','NumNeighbors',5);
KnnClass= predict(KnnMdl,Z(:,pos_test_data)');
n_errors=length(find(KnnClass~=label(pos_test_data)'));
misclassification = 100-n_errors/grupo*100

%% SOM
% Data
Data = Trainnumbers.image;
Label = Trainnumbers.label;
[datos_norm,pos_data_red,media,std_dev] = normalizacion(Data);
Z = encode(autoenc,data_norm);

% Choose test and train
ind_random = randperm(length(data_norm));
pos_test_data = ind_random(1:grupo);
pos_train_data = ind_random(grupo+1:end);

train_data = Z(:,pos_train_data);
label_train_data = Label(:,pos_train_data);
test_data = Z(:,pos_test_data);
label_test_data = Label(:,pos_test_data);

% Red SOM
neuronas = [36 30];
num_neuronas = prod (neuronas);
% dimensions = neuronas;
% coverSteps = 100; %default = 100
% initNeighbor = 5; %default = 3
% topologyFcn = 'randtop'; %default = 'hextop'
% distanceFcn = 'linkdist'; %default = 'linkdist'
% net = selforgmap(dimensions,coverSteps,initNeighbor,topologyFcn,distanceFcn);
% net.trainParam.epochs = 200;
% net = train(net,train_data);

% Label the neurons
activation_table = zeros(10,num_neuronas);
for i = 1:length(train_data)
    activated_neuron = vec2ind(net(train_data(:,i)));
    image_label = label_train_data(i);
    activation_table(image_label+1,activated_neuron) = activation_table(image_label+1,activated_neuron)+1;
end

[~,neuron_labels] = max(activation_table);
neuron_labels = neuron_labels-1;

% Test the SOM
Test_neur_activ = vec2ind(net(test_data));
label_test_som = neuron_labels(Test_neur_activ);

n_errors = length(find(label_test_som-label_test_data))