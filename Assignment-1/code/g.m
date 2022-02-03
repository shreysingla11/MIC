function [f,der] = g(im, gamma)
    % This is quadratic function
    %res = im.^2;   

    % This is huber function
    global prior_type;
    
    if strcmp(prior_type,'huber')
        ind_lin_pos = im > gamma;
        ind_lin_neg = im < -gamma;
        ind_lin = ind_lin_pos | ind_lin_neg;
        ind_quad = abs(im) <= gamma;
        
        f = zeros(size(im));
        der = zeros(size(im));
    
        f(ind_quad) = 0.5*(im(ind_quad).^2);
        f(ind_lin) = gamma*abs(im(ind_lin)) - 0.5*gamma*gamma;
        f = sum(f,'all');
    
        der(ind_quad) = im(ind_quad);
        der(ind_lin_pos) = gamma;
        der(ind_lin_neg) = -gamma;
    elseif strcmp(prior_type,'gaussian')
        f = sum(im.^2,'all');
        der = 2*im;
    elseif strcmp(prior_type,'disadp')
        f = sum(gamma*abs(im) - gamma*gamma*log(1 + abs(im)/gamma),'all');
        der = gamma*sign(im) - gamma*gamma*(sign(im)./(gamma + abs(im)));

end