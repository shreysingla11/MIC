function [f,der] = g(im, gamma, prior_ind)
    
    % This is quadratic function
    if prior_ind == 0
        f = im.^2;   
        der = 2*im;
        f = sum(f, 'all');
    end
    
    % This is huber function
    if prior_ind == 1
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
    
    % This is discontinuity adaptive function
    if prior_ind == 2
        f = gamma*abs(im)-gamma^2*log(1+abs(im)/gamma);
        f = sum(f, 'all');
        der = gamma*sign(im) - gamma*(1./(1+abs(im)/gamma)).*sign(im);
    end

end