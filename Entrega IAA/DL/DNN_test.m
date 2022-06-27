%% DNN TEST
load DNN_model
Data_test = imageDatastore('../Test_image');
YPred = classify(net,Data_test);

Label_Data_test = zeros(length(YPred),1);
% Ordinar los datos
for i = 1:length(Data_test.Files)
    a = Data_test.Files{i};
    l = length(a);
    num = a(59:(59+l-64));
%     num = a(85:(85+l-90));
    num = str2double(num);
    
    Label_Data_test(num) = double(YPred(i))-1;
end

class = Label_Data_test;

T= table(class);
filename = 'data_testdnn.xlsx';
writetable(T,filename)


%% Guardado
PCA = 0;
name = {'TommasoC'};
save('M21025_dln.mat', "name", "PCA", "class")