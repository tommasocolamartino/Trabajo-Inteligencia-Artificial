function  print_400_numbers_range(Datos,range)

immagine = zeros(28*20,28*20);
pixels = zeros(28,28);
for i = 1:400
    riga = floor(range(i)/20)+1;
    if floor(range(i)/20) == (range(i)/20)
        riga =(range(i)/20);
    end
    col = mod(range(i),20);
    if col == 0
        col = 20;
    end
    im = range(i);
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

