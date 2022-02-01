function [f,der] = g(im, gamma)
    % This is quadratic function
    %res = im.^2;   

    % This is huber function
    
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
   
end