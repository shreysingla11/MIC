function [f,der] = likelihood(x, y,weight)
    f = weight*sum((x - y).^2,'all');  % This is iid gaussian noise
    der = weight*2*(x-y);
end