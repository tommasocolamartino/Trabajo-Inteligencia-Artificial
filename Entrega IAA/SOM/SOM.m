clc
clearvars
close all

addpath ('..\');
addpath ('..\PCA');
load Trainnumbers
Data = Trainnumbers.image;
label = Trainnumbers.label;
[data_norm,pos_data_red,media,std_dev] = normalizacion(Data);
[Wc,Diag] = eig(cov(data_norm'));
lambda = diag(Diag);

% PCA
grupo = 1000;
N_comp_princ = 25;
W = PCA_fun(Wc,lambda,N_comp_princ);
PCA_data = W'*data_norm;

% Select Train and Test data
ind_random = randperm(length(data_norm));
pos_test_data = ind_random(1:grupo);
pos_train_data = ind_random(grupo+1:end);        

train_data = PCA_data(:,pos_train_data);
label_train_data = label(:,pos_train_data);
test_data = PCA_data(:,pos_test_data);
label_test_data = label(:,pos_test_data);

% Create and train SOM

neuronas = [30 36];
num_neuronas = prod (neuronas);
% net.trainParam.epochs = 200;
dimensions = neuronas;
coverSteps = 100; %default = 100
initNeighbor = 5; %default = 3
topologyFcn = 'randtop'; %default = 'hextop'
distanceFcn = 'linkdist'; %default = 'linkdist'
net = selforgmap(dimensions,coverSteps,initNeighbor,topologyFcn,distanceFcn);
net.trainParam.epochs = 200;
net = train(net,train_data);

%% Label the neurons
activation_table = zeros(10,num_neuronas);
image_neuron = zeros(40,2*num_neuronas);
comp = ones(1,num_neuronas);
for i = 1:length(train_data)
    activated_neuron = vec2ind(net(train_data(:,i)));
    image_label = label_train_data(i);
    activation_table(image_label+1,activated_neuron) = activation_table(image_label+1,activated_neuron)+1;
    
    image_neuron(comp(activated_neuron),(2*activated_neuron-1):(2*activated_neuron)) = [image_label,i];
    comp(activated_neuron) = comp(activated_neuron)+1;
end

[~,neuron_labels] = max(activation_table);
neuron_labels = neuron_labels-1;
%% Print the map
dim = 39;
SOM_map = ones((dim*36)+35,(dim*30)+29);
figure
for i = 1:num_neuronas
    a = find (image_neuron(:,2*i-1)==neuron_labels(i));
    b = image_neuron(a(1),2*i)
    digit = Data(:,pos_train_data(b));
    for j=1:28
        for k=1:28
            digito(j,k)=digit((j-1)*28+k);
        end
    end
    
    riga = 36-(floor(i/30));
    if riga == 36-(i/30)
        riga = riga+1;
    end
    
    col = mod(i,30);
    if col == 0
        col = 30;
    end
    
    digito_rid = imresize(digito,[dim,dim]);
%     p = (riga-1)*30+col;
    in_r = (dim*(riga-1)+riga);
    fin_r = in_r + (dim-1);
    in_c = (dim*(col-1)+col);
    fin_c = in_c+(dim-1);
    SOM_map(in_r:fin_r,in_c:fin_c) = digito_rid;
end
imshow(SOM_map)
% figure 
% image = zeros()
% for i = 1:num_neuronas
%     
% end
%% Test the SOM
Test_neur_activ = vec2ind(net(test_data));
label_test_som = neuron_labels(Test_neur_activ);

n_errors = length(find(label_test_som-label_test_data))
% % view(net)
% % y = net(x);
% classes = vec2ind(train_data); % vettore con numero neurone attivato in ordine 

%save SOM_net net neuron_labels
figure
cm = confusionchart(label_test_data,label_test_som);
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
cm.Title = 'SOM Model Confusion Matrix'; 