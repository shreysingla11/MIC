function [f,der] = likelihood(x, y,weight)
    % Function to compute the value f and derivative der of the likelihood 
    % at a point
    
    global likelihood_type;

    if strcmp(likelihood_type,'gaussian')
        f = weight*sum((x - y).^2,'all');
        der = weight*2*(x-y);
    end
end