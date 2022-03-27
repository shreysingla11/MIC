function [] = makePlots(z, alignedPointSets, str)
%MAKEPLOTS Summary of this function goes here
%   Detailed explanation goes here
    figure("Name",sprintf("[%s] Aligned pointsets and mean", str));
    pause(1);
    plotPointSets(alignedPointSets);
    hold on;
    plotPointSet(z);
    hold off;
    
    [eigvals, eigvecs] = computeVariation(z,alignedPointSets);
    
    figure("Name",sprintf("[%s] Plot of eigenvalues", str));
    pause(1);
    plot(eigvals);
    
    for i = 1:3
        figure("Name",sprintf("[%s] Mode of variation %d", str, i));
        pause(1);
        plotPointSets(alignedPointSets);
        hold on;
        plotPointSet(z);
        plotPointSet(z + 2*sqrt(eigvals(i))*eigvecs(:,:,i));
        plotPointSet(z - 2*sqrt(eigvals(i))*eigvecs(:,:,i));
        hold off;
        legend('mean','mean + 2*std','mean - 2*std');
    end
end

