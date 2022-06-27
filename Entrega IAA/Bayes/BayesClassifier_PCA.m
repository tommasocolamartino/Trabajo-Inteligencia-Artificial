%% BAYESIAN CLASSIFIER
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

% 2. Divide them in training and test samples; perform the ten fold cross validation.
grupo = length(data_norm)/10; %datos_procesar en este caso son los datos normalizados
N_comp_princ = 100;
for i = 1:N_comp_princ %Componentes principales de la PCA
    W = PCA_fun(Wc,lambda,i);
    PCA_data = W'*data_norm;
    pos_test_data = 1:1:grupo;
    pos_train_data = [1:1:(pos_test_data(1)-1),(pos_test_data(grupo)+1):1:length(data_norm)];
    for j = 1:10    
            % 3. Train and test the Bayesian network    
            bayMdl= fitcnb(PCA_data(:,pos_train_data)',label(pos_train_data)');
            bayclass= predict(bayMdl,PCA_data(:,pos_test_data)');
            n_errors(j,1)=length(find(bayclass~=label(pos_test_data)'));
            pos_test_data = pos_test_data+1000;
            pos_train_data = [1:1:(pos_test_data(1)-1),(pos_test_data(grupo)+1):1:length(data_norm)];
    end
    var_error (i) = std(n_errors); 
    misclassification(i) = 100-mean(n_errors)/grupo*100
end

plot(1:1:N_comp_princ,misclassification,'b', 'LineWidth',2), grid on, xlabel('Num. componentes principales'), ylabel ('Accuracy %'); title('Bayesian Classifier: Accuracy VS N° of Pricipal Components')

% 4. Plot the Confusion Matrix of the best result: 20 componentes principales.
cp = 20;
W = PCA_fun(Wc,lambda,cp);
PCA_data = W'*data_norm;
pos_test_data = randi(10000,1,grupo);
j = 1;
for i = 1:10000
    if find(pos_test_data~=i) 
        pos_train_data(j) = i;
        j = j+1;
    end
end
bayMdl_20= fitcnb(PCA_data(:,pos_train_data)',label(pos_train_data)');
bayclass_20= predict(bayMdl_20,PCA_data(:,pos_test_data)');
n_errors_20=length(find(bayclass_20~=label(pos_test_data)'));
misclassification_20 = 100-n_errors_20/grupo*100

figure
cm = confusionchart(label(pos_test_data),bayclass_20);
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
cm.Title = 'Bayes Model Confusion Matrix'; 
