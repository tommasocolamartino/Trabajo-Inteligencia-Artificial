function [data_norm_train,data_norm_test] = norm_train_test(Data_train,Data_test)

[data_norm_train,pos_data_red,media,std_dev] = normalizacion(Data_train);
Data_test = Data_test(pos_data_red,:);
data_norm_test = (Data_test - media(pos_data_red))./(std_dev(pos_data_red));

end

