%% KNN CLASSIFIER
close all
clearvars
clc

% 1. Pre-process the data
addpath ('..\');
addpath ('..\PCA');
load Trainnumbers
Data = Trainnumbers.image;
label = Trainnumbers.label;
[data_norm,pos_data_red,media,std_dev] = normalizacion(Data);
[Wc,Diag] = eig(cov(data_norm'));
lambda = diag(Diag);

grupo = length(data_norm)/10; 
N_comp_princ = 100;
for i = 1:N_comp_princ %Componentes principales de la PCA
    W = PCA_fun(Wc,lambda,i);
    PCA_data = W'*data_norm;
    pos_test_data = 1:1:grupo;
    pos_train_data = [1:1:(pos_test_data(1)-1),(pos_test_data(grupo)+1):1:length(data_norm)];
    for j = 1:(length(data_norm)/grupo)    
            % 3. Train and test the Knn network  
            knnMdl = fitcknn(PCA_data(:,pos_train_data)',label(pos_train_data)','NumNeighbors',5);
            knnclass= predict(knnMdl,PCA_data(:,pos_test_data)');      
            n_errors(j,1)=length(find(knnclass~=label(pos_test_data)'));
            pos_test_data = pos_test_data+grupo;
            pos_train_data = [1:1:(pos_test_data(1)-1),(pos_test_data(grupo)+1):1:length(data_norm)];
    end
    var_error (i) = std(n_errors); 
    misclassification(i) = 100-mean(n_errors)/grupo*100
end

plot(1:1:N_comp_princ,misclassification,'b', 'LineWidth',2), grid on, xlabel('Num. componentes principales'), ylabel ('Accuracy %'); title('Knn Classifier: Accuracy VS N° of Pricipal Components')

%% 4. Plot the Confusion Matrix of the best result: 46 componentes principales.
addpath ('..\');
addpath ('..\PCA');
load Trainnumbers
Data = Trainnumbers.image;
label = Trainnumbers.label;
[data_norm,pos_data_red,media,std_dev] = normalizacion(Data);
[Wc,Diag] = eig(cov(data_norm'));
lambda = diag(Diag);

grupo = 1000;
cp = 46;
W = PCA_fun(Wc,lambda,cp);
PCA_data = W'*data_norm;

ind_random = randperm(length(data_norm));
pos_test_data_46 = ind_random(1:grupo);
pos_train_data_46 = ind_random(grupo+1:end);

knnMdl_46 = fitcknn(PCA_data(:,pos_train_data_46)',label(pos_train_data_46)','NumNeighbors',5);
%fitcknn(PCA_data(:,pos_train_data_46)',label(pos_train_data_46)','OptimizeHyperparameters','auto',...
%     'HyperparameterOptimizationOptions',...
%     struct('AcquisitionFunctionName','expected-improvement-plus'))
knnclass_46= predict(knnMdl_46,PCA_data(:,pos_test_data_46)');
n_errors_46=length(find(knnclass_46~=label(pos_test_data_46)'));
misclassification_46 = 100-n_errors_46/grupo*100

figure
cm = confusionchart(label(pos_test_data_46),knnclass_46);
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
cm.Title = 'KNN Model Confusion Matrix'; 
