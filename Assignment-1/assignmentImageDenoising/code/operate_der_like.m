function res = operate_der_like(noisy_im, present)
    shape = size(noisy_im);
    h = shape(1);
    w = shape(2);
    res = zeros(shape);
    for i=1:h
        for j=1:w
            res(i,j) = like_dash(noisy_im(i,j), present(i,j));
        end
    end
end