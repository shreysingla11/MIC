function [f,der] = g(im, gamma)
    % This is quadratic function
    %res = im.^2;   

    % This is huber function
    global prior_type;
    
    if strcmp(prior_type,'huber')
        mask_less = abs(im) <= gamma;
        mask_more = abs(im) > gamma;
        f = sum((0.5*im.^2).*mask_less + (gamma*abs(im) - 0.5*gamma*gamma).*mask_more,"all");
        der = im.*mask_less + gamma*sign(im).*mask_more;
    elseif strcmp(prior_type,'gaussian')
        f = sum(im.^2,'all');
        der = 2*im;
    elseif strcmp(prior_type,'disadp')
        f = sum(gamma*abs(im) - gamma*gamma*log(1 + abs(im)/gamma),'all');
        der = gamma*sign(im) - gamma*gamma*(sign(im)./(gamma + abs(im)));
    end

end