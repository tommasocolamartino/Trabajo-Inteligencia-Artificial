%% BAYESIAN CLASSIFIER
close all
clearvars
clc

% 1. Pre-process the data
addpath ('..\');
load Trainnumbers
Data = Trainnumbers.image;
[data_norm,pos_data_red,media,std_dev] = normalizacion(Data);

% 2. Divide them in training and test samples; perform the ten fold cross validation.
grupo = length(data_norm)/10;
K_max = 4;
for k = 1:K_max %Cuantos subconjuntos de cada clase
    % K-means y luego LDA 
    % los datos tienen la dimensionalidad de 10*k
    [datos_procesar,label] = k_means_new(data_norm,Trainnumbers.label,k);
    [Sb,Sw] = Scat_matrices(datos_procesar,label);
    [Wc,Diag]=eig(pinv(Sw)*Sb);
    Wc = real(Wc);
    Diag = real(Diag);
    [pos_eigvect,~] = find(Diag>1e-5);
    W = Wc(:,pos_eigvect);
    LDA_data = W'*datos_procesar;
    
    % Inicialización ``ten fold cross validation''
    ind_random = randperm(length(data_norm));
    pos_test_data = ind_random(1:grupo);
    pos_train_data = ind_random(grupo+1:end);
    n_errors=zeros(10,1);
    for j = 1:10    
            % 3. Train and test the Bayesian network    
            knnMdl = fitcknn(LDA_data(:,pos_train_data)',label(pos_train_data)','NumNeighbors',5);
            knnClass= predict(knnMdl,LDA_data(:,pos_test_data)');      
           
            actual_class = floor(label(pos_test_data)/k)';
            predicted_class = floor(knnClass/k);
            n_errors(j,1)=length(find(predicted_class~=actual_class));
            in = grupo*j+1;
            fin = in+grupo-1;
            if j < 9
                pos_test_data = ind_random(in:fin);
                pos_train_data = [ind_random(1:find(ind_random==pos_test_data(1))-1),ind_random(find(ind_random==pos_test_data(grupo))+1:end)];
            elseif j==9
                pos_test_data = ind_random(in:fin);
                pos_train_data = ind_random(1:9000);
            else
                break
            end
    end
    var_error (k) = std(n_errors); 
    misclassification(k) = 100-mean(n_errors)/grupo*100
end

plot(1:1:K_max,misclassification,'b', 'LineWidth',2), grid on, xlabel('N° of K-menas clusters'), ylabel ('Accuracy of the LDA algorithm %'); title('KNN Classifier: Accuracy VS N° of K-menas clusters')

%% 4. Plot the Confusion Matrix of the best result: 20 componentes principales.
clearvars
clc
close all
load Trainnumbers
load LDA_data
grupo = length(data_norm)/10;
k = 2;
[datos_procesar,label] = k_means_new(data_norm,Trainnumbers.label,k);
[Sb,Sw] = Scat_matrices(datos_procesar,label);
[Wc,Diag]=eig(pinv(Sw)*Sb);
Wc = real(Wc);
Diag = real(Diag);
[pos_eigvect,~] = find(Diag>1e-5);
W = Wc(:,pos_eigvect);
LDA_data = W'*datos_procesar;
    
ind_random = randperm(length(data_norm));
pos_test_data = ind_random(1:grupo);
pos_train_data = ind_random(grupo+1:end);

KnnMdl_2=fitcknn(LDA_data(:,pos_train_data)',label(pos_train_data)','NumNeighbors',10);
%fitcknn(LDA_data(:,pos_train_data)',label(pos_train_data)','OptimizeHyperparameters','auto',...
%     'HyperparameterOptimizationOptions',...
%     struct('AcquisitionFunctionName','expected-improvement-plus'));
%fitcknn(LDA_data(:,pos_train_data)',label(pos_train_data)');
%   3          cityblock
KnnClass_2= predict(KnnMdl_2,LDA_data(:,pos_test_data)');
actual_class_2 = floor(label(pos_test_data)/k)';
predicted_class_2 = floor(KnnClass_2/k);
n_errors_2 =length(find(predicted_class_2~=actual_class_2));
misclassification_2 = 100-n_errors_2/grupo*100

figure
cm = confusionchart(actual_class_2,predicted_class_2);
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
cm.Title = 'KNN Model Confusion Matrix'; 
