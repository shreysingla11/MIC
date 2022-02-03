function [f,der] = prior(x, weight, gamma, prior_ind)
    shape = size(x);
    h = shape(1);
    w = shape(2);
    temp = zeros([h+2, w+2]);
    temp(2:end-1,2:end-1) = x;
    temp(1,2:end-1) = x(1,:);
    temp(end,2:end-1) = x(end,:);
    temp(2:end-1,1) = x(:,1);
    temp(2:end-1,end) = x(:,end);
    
    f = 0;
    der = zeros(size(x));
    for offset=[-1,1]
        [gx,gx_der] = g(x-temp(2+offset:end-1+offset,2:end-1), gamma, prior_ind);
        [gy,gy_der] = g(x-temp(2:end-1,2+offset:end-1+offset), gamma, prior_ind);
        f = f + weight*(gx+gy);
        der = der + weight*(gx_der + gy_der);
    end
end