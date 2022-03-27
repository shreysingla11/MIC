function [eigvals,eigvecs] = computeVariation(mean,pointSets)
%COMPUTEVARIATION Summary of this function goes here
%   Detailed explanation goes here
    size_mean = size(mean);
    mean = reshape(mean,numel(mean),1);
    pointSets = reshape(pointSets,numel(mean),[]);
    
    cov = (pointSets - mean)*(pointSets - mean)';
    [V, D] = eig(cov,'vector');
    [D, ind] = sort(D,'descend');
    V = V(:, ind);
    eigvecs = zeros(size_mean(1),size_mean(2),length(V));
    for i = 1: length(V)
        eigvecs(:,:,i) = reshape(V(:,i),size_mean);
    end
    eigvals = D;
end

