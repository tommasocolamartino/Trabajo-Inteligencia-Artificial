%% Guarda imagenes para el entrenamiento del deep
addpath ('..\')
load Trainnumbers

Data = Trainnumbers.image;
Labels = Trainnumbers.label;
count = 0;
for i = 0:9
    p = find(Labels == i);
    l = length(p);
    for j = 1:l
        count = count+1;
        pos = p(j);
        image = Data(:,pos);
        digito = zeros(28,28);
        for a=1:28
            for b=1:28
                digito(a,b)=image((a-1)*28+b);
            end
        end
        grayImage = uint8(digito);
        imwrite(grayImage, sprintf('../Trainnumbers_image/%d/FIG%d.jpeg',i,count));
    end
end