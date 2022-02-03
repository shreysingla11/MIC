function [f,der] = prior(x, weight, gamma)
    % Function to compute the value f and derivative der of the prior 
    % at a point

    shape = size(x);
    h = shape(1);
    w = shape(2);
    temp = zeros([h+2, w+2]);

    % Use 'same' type padding to x to ease neighbour computation
    temp(2:end-1,2:end-1) = x;
    temp(1,2:end-1) = x(1,:);
    temp(end,2:end-1) = x(end,:);
    temp(2:end-1,1) = x(:,1);
    temp(2:end-1,end) = x(:,end);
    
    f = 0;
    der = zeros(size(x));

    for offset=[-1,1]
        % Sum of the values for g
        [gx,gx_der] = g(x-temp(2+offset:end-1+offset,2:end-1), gamma);
        [gy,gy_der] = g(x-temp(2:end-1,2+offset:end-1+offset), gamma);
        f = f + weight*(gx+gy);
        der = der + weight*(gx_der + gy_der);
    end
end