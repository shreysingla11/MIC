function [sR,T] = Code2(x,y,w)
%CODE2 Summary of this function goes here
%   Detailed explanation goes here
temp = x*w;
X1 = temp(1);
Y1 = temp(2);


temp = y*w;
X2 = temp(1);
Y2 = temp(2);

W = sum(w, "all");
Z = (y(1,:).^2 + y(2,:).^2)*w;
C1 = (x(1,:).*y(1,:) + x(2,:).*y(2,:))*w;
C2 = (x(2,:).*y(1,:) - x(1,:).*y(2,:))*w;

A = [X2, -Y2, W, 0;Y2, X2, 0, W;Z, 0, X2, Y2;0, Z, -Y2, X2];    
b = [X1, Y1, C1, C2]';

answer = A\b;
sR = [answer(1), -answer(2);answer(2), answer(1)];
T = answer(3:end);
end

