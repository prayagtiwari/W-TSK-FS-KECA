%% Computing kernel matrix
function [K] = kernel(data,sigma);
    
    % Creating matrix of Euclidean distances
    X = squareform(pdist(data'));
    
    % Using RBF kernel to obtain  matrix
    alpha = 1/(2*sigma^2);
    K = exp(-1*alpha*X.^2);

  
