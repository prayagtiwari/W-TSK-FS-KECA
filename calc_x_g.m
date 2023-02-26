function [mu_m] = calc_x_g(x,v,b)

% Calculate the X_g by x * fire-level

% x: the original data -- n_examples * n_features
% v: clustering centers of the fuzzy rule base -- k * n_features
% b: kernel width of the corresponding centers of the fuzzy rule base


n_examples = size(x,1);
x_e = [x,ones(n_examples,1)];
[k,d] = size(v); % k: number of rules of TSK; d: number of dimensions
mu_m=[];
for i=1:k
    v1 = repmat(v(i,:),n_examples,1);
    bb = repmat(b(i,:),n_examples,1);
    wt(:,i) = exp(-sum((x-v1).^2./bb,2));
end

wt2 = sum(wt,2);

% To avoid the situation that zeros are exist in the matrix wt2
ss = wt2==0;
wt2(ss,:) = eps;
wt = wt./repmat(wt2,1,k);


mu_m = wt;


end