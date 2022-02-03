function [f,der] = likelihood(x, y,weight)
    global likelihood_type;
    if strcmp(likelihood_type,'gaussian')
        f = weight*sum((x - y).^2,'all');  % This is iid gaussian noise
        der = weight*2*(x-y);
    end
end