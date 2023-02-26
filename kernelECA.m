%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MATLAB function: kernelECA.m
% 
% PURPOSE:  Maps an input data set to a C-dimensional 
%           feature space using kernel entropy component analysis
%           (kernel ECA).
% 
% OUTPUT
% Phi         : Kernel ECA feature space data set
% 
% INPUT
% K           : Data affinity matrix
% C           : Dimension of feature space data (number of clusters)
% center           : Centering parameter (only makes sense in a PCA context)
% n           : Normalized to unit length (n == 1)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Phi,newData] = kernelECA(data,K,C,center,n); 
    % Center the feature space data if c=1
    if (center == 1);
        jj = ones(N,1);
        K = K - (1/N)*jj*jj'*K - (1/N)*K*jj*jj' + (1/N^2)*(jj'*K*jj)*jj*jj';
    end
   
    numSamples= size(data, 1);
    % Eigendecompose K
    %[E,D] = eig(K);
    %rng('default')
    [E, D, ~] = svd(K/numSamples, 'econ');
    [D,E] = sort_eigenvalues(D,E);
    % Create a vector of sorted eigenvalues
    d = diag(D)';
    
    % Compute entropy components and sort them
    [sorted_entropy_index,entropy] = ECA(D,E);
    
    % Rearrange descending entropy components
    Es = E(:,sorted_entropy_index);
    ds = d(sorted_entropy_index);
    
    % Map the input data using kernelECA
    for i = 1 : C;
        Phi(:,i) = sqrt(ds(i)) * Es(:,i);
    end;
    
    % Transformed data as colum vectors
    Phi = Phi';
    
    % Normalize to unit length if n=1
    if (n == 1);
        for i = 1 : size(Phi,2);
            Phi(:,i) = Phi(:,i)/(norm(Phi(:,i)));
        end;
    end;
    
    lambda= diag(D);
    numFeatures= size(data, 2);
    alpha = 1;
    try
        coefficient = E./sqrt(numSamples*lambda)';
    catch
         coefficient = zeros(numSamples, numSamples);
         for i = 1:numSamples
             coefficient(:, i) = E(:, i)/sqrt(numSamples*lambda(i, 1));
         end
    end
    
    score = K* coefficient(:, 1:numFeatures);
    K_1 =  kernel(score', 20);
    K_1_ = K_1;
    for i = 1:numSamples
        K_1(i, i) = K_1(i, i)+alpha;
    end
    dual_coef = mldivide(K_1, data);
    K_2 =  K_1_;
    newData = K_2*dual_coef;

%% Computing kernel matrix
function [K] = kernel(data,sigma);
    
    % Creating matrix of Euclidean distances
    X = squareform(pdist(data'));
    
    % Using RBF kernel to obtain  matrix
    alpha = 1/(2*sigma^2);
    K = exp(-1*alpha*X.^2);
    
%% Sorting eigenvalues and eigenvectors
function [D,E] = sort_eigenvalues(D,E);
    
    % Ectract eigenvalues from diagonal of D
    d = diag(D);
    
    % Sort the eigenvalues
    [d_sorted d_index] = sort(d,'descend');
    
    % Create new matrix D
    D = zeros(length(d_sorted));
    for i = 1 : length(d_sorted);
        D(i,i) = d_sorted(i);
    end;
    
    % Create new matrix E
    E = E(:,d_index);    
    
%% Computing sorted kernel ECA entropy terms   
function [sorted_entropy_index,entropy] = ECA(D,E);

N = size(E,2);

% Computes the sorted Renyi entropy estimate 
entropy = diag(D)' .* (ones(1,N)*E).^2;
[sorted_entropy,sorted_entropy_index] = sort(entropy,'descend');

