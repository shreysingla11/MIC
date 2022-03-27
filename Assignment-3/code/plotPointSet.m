function [] = plotPointSet(pointSet)
%PLOTPOINTSET Summary of this function goes here
%   Detailed explanation goes here
pause(1);
patch('XData',pointSet(1,:),'YData',pointSet(2,:),'FaceColor','none','LineWidth',2, 'EdgeColor', rand(1,3));
end

