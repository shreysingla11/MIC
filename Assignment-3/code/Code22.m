function [z_new, alignedPointSets] = Code22(z,pointSets)
%CODE11 Summary of this function goes here
%   Detailed explanation goes here
    numPointSets = size(pointSets,3);
    numPoints = size(pointSets,2);
    alignedPointSets = zeros(size(pointSets));
    
    for i = 1:numPointSets
        pointSet = pointSets(:,:,i);
        [sR, T] = Code2(z,pointSet,ones(numPoints,1));
        alignedPointSets(:,:,i) =  sR*pointSet + T;
    end
    z_new = mean(alignedPointSets,3);
    z_new = z_new / norm(z_new,'fro');
end
