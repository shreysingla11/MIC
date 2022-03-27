function [] = plotPointSets(pointSets)
%PLOTPOINTSETS Summary of this function goes here
pause(1);
hold on;
for i = 1:size(pointSets,3)
    scatter(pointSets(1,:,i),pointSets(2,:,i),5,rand(1,3),"filled");
end
hold off;
end

