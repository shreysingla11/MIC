function [f,der] = g(im, gamma)
    % Function to compute the value f and derivative der of g at a point

    global prior_type;
    
    % This is the huber function
    if strcmp(prior_type,'huber')
        mask_less = abs(im) <= gamma;
        mask_more = abs(im) > gamma;
        f = sum((0.5*im.^2).*mask_less + (gamma*abs(im) - 0.5*gamma*gamma).*mask_more,"all");
        der = im.*mask_less + gamma*sign(im).*mask_more;
    
    % This is the quadratic prior
    elseif strcmp(prior_type,'quadratic')
        f = sum(im.^2,'all');
        der = 2*im;
    
    % This is the discontinuity adaptive prior
    elseif strcmp(prior_type,'discontinuity_adaptive')
        f = sum(gamma*abs(im) - gamma*gamma*log(1 + abs(im)/gamma),'all');
        der = gamma*sign(im) - gamma*gamma*(sign(im)./(gamma + abs(im)));
    end

end