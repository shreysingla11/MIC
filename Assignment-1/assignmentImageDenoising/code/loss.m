function res = loss(noisy_im, image)
    shape = size(image);
    h = shape(1);
    w = shape(2);
    temp = zeros([h+2, w+2]);
    temp(2:end-1,2:end-1) = image;
    temp(1,2:end-1) = image(1,:);
    temp(end,2:end-1) = image(end,:);
    temp(2:end-1,1) = image(:,1);
    temp(2:end-1,end) = image(:,end);
    res = 0;
    for i=2:h+1
        for j=2:w+1
            res = res + g(temp(i,j)-temp(i+1,j)) + g(temp(i,j)-temp(i,j+1))+g(temp(i,j)-temp(i-1,j))+g(temp(i,j)-temp(i,j-1));
        end
    end
    for i=1:h
        for j=1:w
            res = res + like(noisy_im(i,j), image(i,j));
        end
    end
end