%% Guarda imagenes para el test del deep
close all
clc
clearvars

addpath ('..\')
load Test_numbers_HW2

Data = Test_numbers.image;
count = 0;
l = length(Data);

for i = 1:l
    count = count+1;
    image = Data(:,i);
    digito = zeros(28,28);
    for j=1:28
        for k=1:28
            digito(j,k)=image((j-1)*28+k);
        end
    end
    grayImage = uint8(digito);
    imwrite(grayImage, sprintf('../Test_image/FIG%d.jpeg',count));
end