function res = operate_der_prior(image)
     shape = size(image);
%      disp(image);
    h = shape(1);
    w = shape(2);
    temp = zeros([h+2, w+2]);
    temp(2:end-1,2:end-1) = image;
    temp(1,2:end-1) = image(1,:);
    temp(end,2:end-1) = image(end,:);
    temp(2:end-1,1) = image(:,1);
    temp(2:end-1,end) = image(:,end);
    res = zeros(shape);
    for i=2:h+1
        for j=2:w+1
            res(i-1,j-1) = g_dash(temp(i,j)-temp(i+1,j)) + g_dash(temp(i,j)-temp(i,j+1))+g_dash(temp(i,j)-temp(i-1,j))+g_dash(temp(i,j)-temp(i,j-1));
        end
    end
end