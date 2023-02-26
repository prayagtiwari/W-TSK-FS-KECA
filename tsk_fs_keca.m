function [predict_label,score_s] = tsk_fs_keca(X_train,Y_train,X_test)

% X_train: n_tr * n_features
% Y_train: n_tr * 1
% X_test: n_te * n_features
% Y_test: n_te * 1
% options.lambda: % regularization parameter for ridge regression TSK
% options.k: number of fuzzy rules
% options.h: adjustable parameter for fcm used for generating antecedent
%             parameters.
% C : Dimension of feature space data (number of clusters)
% sigma:Parameters of RBF kernel function
%seed = 12345678;
%rand('seed', seed);
y_pred_test=[];
[n_tr,~] = size(X_train);
X_uni = [X_train',X_test'];
X_uni = X_uni*diag(sparse(1./sqrt(sum(X_uni.^2))));
X_uni_train = X_uni(:,1:n_tr)';
X_uni_test = X_uni(:,(n_tr+1):end)';
D_train = X_train;
D_test = X_test;
options.k=2;
options.lambda=1;
options.h=50;

% Get antecedent parameters and the transformed data in fuzzy feature space
% There are two ways to generate the antecedent parameters: 'deter' or 'fcm'
% 'deter' generate the ant7cedent parameters deterministically and 'fcm' 
% generate the antecedent parameters with randomness.

[v_train,b_train] = gene_ante_fcm(D_train,options);
[v_test,b_test] = gene_ante_fcm(D_test,options);

%[v_train,b_train] = gene_ante_deter(D_train,options);
%[v_test,b_test] = gene_ante_deter(D_test,options);

G_train = calc_x_g(D_train,v_train,b_train);
G_test = calc_x_g(D_test,v_test,b_test);

P = lms_l2(G_train,Y_train,options.lambda); % can be replaced with lms

y_pred_test = G_test*P;


score_s = y_pred_test;
predict_label = sign(score_s);
end


function c = lms_l2(x_g,y,omega)
% calculate the least square solution of the x_g and y
% ridge regression is used here

C=10;
sigma=14;
%x=X_train;
[fuzzy_membership_scores,~]=computing_fuzzy_scores_keca(x_g,C,sigma);
S=fuzzy_membership_scores;
A = x_g'*x_g;
B = eye(size(A,1));
S=S'*S;
c = (A*S+omega*B)\(x_g'*y);
if sum(isnan(c))>0
    error('calculate results contains NaN!!!!!!!!!!!!');  
end
end


function [v,b] = gene_ante_fcm(data,options)
% Generate the antecedents parameters of TSK FS by FCM

% data: n_example * n_features
% options.k: the number of rules
% options.h: the adjustable parameter of kernel width 
% of Gaussian membership function.

% return::v: the clustering centers -- k * n_features
% return::b: kernel width of corresponding clustering centers

k = options.k;
h = options.h;
[n_examples, d] = size(data);
% options: exponent for partition matrix & iterations & threshold & display
[v,U,~] = fcm(data,k,[2,NaN,1.0e-6,0]);

for i=1:k
    v1 = repmat(v(i,:),n_examples,1);
    u = U(i,:);
    uu = repmat(u',1,d);
    b(i,:) = sum((data-v1).^2.*uu,1)./sum(uu)./1;
end
b = b*h+eps;


end


function [x_g] = calc_x_g(x,v,b)
% Calculate the X_g by x * fire-level
% x: the original data -- n_examples * n_features
% v: clustering centers of the fuzzy rule base -- k * n_features
% b: kernel width of the corresponding centers of the fuzzy rule base
% x_g: data in the new fuzzy feature space -- n_examples * (n_features+1)k

n_examples = size(x,1);
x_e = [x,ones(n_examples,1)];
[k,d] = size(v); % k: number of rules of TSK; d: number of dimensions

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

x_g = [];
for i=1:k
    wt1 = wt(:,i);
    wt2 = repmat(wt1,1,d+1);
    x_g = [x_g,x_e.*wt2];
end

end


function [ data_y_ooh ] = y2ooh( y_label, num_classes )
% Transform the labels to the form of 'one-of-hot'
n_examples = size(y_label, 1);
data_y_ooh = zeros(n_examples, num_classes);
for i=1:n_examples
    index = y_label(i, :);
    data_y_ooh(i, index) = 1;
end
end