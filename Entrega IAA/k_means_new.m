function [datos_kmeans,label_kmeans] = k_means_new(data_norm,label,k)
% Esta función te da como resultado las etiquetas obtenidas despues del
% kmeans y los datos ordenados según el vector label_kmeans.

[data_ord,~,pos] = ordenar_datos(data_norm,label);
num_labels = zeros(10,k);
for i=0:9
    ok = 0;
    rep = 0;
    in = pos(i+1,1);
    fin = pos(i+1,2);
    while (ok~=1)&&(rep<100)
        rep=rep+1;
        [label_kmeans_i]=kmeans(data_ord(:,in:fin)',k,'replicates', 5);
        for j = 1:k
            ok = 1;
            s = sum(label_kmeans_i==j);
            if s <=5
                ok = 0;
                break
            end
            num_labels(i+1,j) = s;
        end
    end
    label_kmeans(in:fin)=label_kmeans_i+k*i-1;
end
datos_kmeans = data_ord;
end