function [R] = Code1(x,y,w)
%CODE1 Summary of this function goes here
%   Detailed explanation goes here
    numDims = size(y,1);

    Mat = eye(numDims);

    [u,s,v] = svd(y*diag(w)*x');
    
    [~,idx] = min(diag(s));
    
    Mat(idx(1),idx(1)) = -1;

    R = v*u';
    if det(R) == -1
        R = v*Mat*u';
    end
end

