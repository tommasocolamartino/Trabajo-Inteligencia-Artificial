function label_nn = mat2vect(label_nn_MLP)
l = length(label_nn_MLP);
label_nn = zeros(1,l);
for i = 1:l
    [~,I] = max(label_nn_MLP(:,i));
    label_nn(i) = I-1;
end
end

