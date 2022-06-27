function  print_9_numbers_range(Datos,range1_9)

immagine = zeros(28*3,28*3);
pixels = zeros(28,28);
image = range1_9;
for i = 1:9
    if i<=3
        riga = 1;
        col = i;
    elseif i<=6
        riga = 2;
        col = i-3;
    else
        riga = 3;
        col = i-6;
    end
    im = image(i);
    for j = 1:28
        for k = 1:28
            pixels(j,k) = Datos((j-1)*28+k,im);
        end
    end
    in_r = 28*(riga-1)+1;
    fin_r = 28*(riga-1)+28;
    in_c = 28*(col-1)+1;
    fin_c = 28*(col-1)+28;
    immagine(in_r:fin_r,in_c:fin_c) = pixels;
end
figure
imshow(immagine,[]),hold on
end

