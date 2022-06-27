%% DNN for image classification

clc
clearvars
close all

imds = imageDatastore('../Trainnumbers_image', ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

numTrainFiles = 750;
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');

% Layers of the DNN
layers = [
    imageInputLayer([28 28 1])
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer

    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];

% Trainng options
    options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.05, ...
    'MaxEpochs',16, ...
    'ValidationData',imdsValidation,...
    'Shuffle','every-epoch', ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');

% Train the network
net = trainNetwork(imdsTrain,layers,options);

% Test the network
YPred = classify(net,imdsValidation);
YValidation = imdsValidation.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation)

%save DNN_model net