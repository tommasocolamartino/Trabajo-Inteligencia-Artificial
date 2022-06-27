function [W] = PCA_fun(Wc,lambda,N_comp_princ)
% This function allows to obtain the matrix W that can be used to obtain a
% reduction of the space of the original data to N principal comonents
% INPUT:
% - Wc: eigenvectors of the covariance matrix of the normalized data
% - lambda: eigenvalues of the covariance matrix of the normalized data
% - N_comp_princ: number of principal components we are interested in
% OUTPUT
% - W: matrix for the reduction of data dimension to N_comp_princ dim.

lambda_ord = sort(lambda,'descend');
for i = 1:N_comp_princ
    pos(i) = find(lambda==lambda_ord(i));
end
W = Wc(:,pos);
end

