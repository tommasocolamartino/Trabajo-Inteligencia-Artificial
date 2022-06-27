function [framed_image] = add_border(image)
[rg,c] = size(image);
int_image = uint8(image);
r = randi(255,1,1);
g = randi(255,1,1);
b = randi(255,1,1);
h = imshow(int_image)
vect = ['r';'g';'b';'c';'m';'y';'#7E2F8E';'#EDB120';'#77AC30'];
rh = rectangle('position',[ ...
    h.XData(1)-.5,...
    h.YData(1)-.5,...
    range(h.XData)+1,...
    range(h.YData)+1],...
    'EdgeColor', vect(neuron_labels(i),:),'linewidth',2);
% r_image = zeros(rg+4,c+4)+r;
% % r_image(3:rg+2,3:c+2) = int_image;
% g_image = zeros(rg+4,c+4)+g;
% % g_image(3:rg+2,3:c+2) = int_image;
% b_image = zeros(rg+4,c+4)+b;
% % b_image(3:rg+2,3:c+2) = int_image;
% framed_image = imshow(cat(3, r_image, g_image, b_image));
% imshow(framed_image)
end

