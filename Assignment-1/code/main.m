clf;
clear;
close all;
images = load("../data/assignmentImageDenoisingPhantom.mat");
noisy_im = images.imageNoisy;
clear_im = images.imageNoiseless;
eta = 0.01;
alphas = linspace(0.993333,1,5);
gammas = linspace(0.00001,0.001120,5);
rmses = zeros(length(alphas),length(gammas));

for i = 1:length(alphas)
    for j = 1:length(gammas)
        [~, rmse] = denoise(noisy_im,clear_im,alphas(i),eta,gammas(j),0);
        rmses(i,j) = rmse;
        fprintf("Tuning iteration - %d,%d\n",i,j);
    end
end

[~, index] = min(rmses(:));
[opt_i, opt_j] = ind2sub(size(rmses),index);

fprintf('Best alpha = %f\n',alphas(opt_i));
fprintf('Best gamma = %f\n',gammas(opt_j));

[denoised_image,rmse] = denoise(noisy_im,clear_im,alphas(opt_i),eta,gammas(opt_j),1);
fprintf('RMSE at alpha*, gamma* = %f\n',rmse);

[~,rmse] = denoise(noisy_im,clear_im,0.8*alphas(opt_i),eta,gammas(opt_j),0);
fprintf('RMSE at 0.8alpha*, gamma* = %f\n',rmse);
[~,rmse] = denoise(noisy_im,clear_im,min(1.2*alphas(opt_i),1),eta,gammas(opt_j),0);
fprintf('RMSE at 1.2alpha*, gamma* = %f\n',rmse);
[~,rmse] = denoise(noisy_im,clear_im,alphas(opt_i),eta,0.8*gammas(opt_j),0);
fprintf('RMSE at alpha*, 0.8gamma* = %f\n',rmse);
[~,rmse] = denoise(noisy_im,clear_im,alphas(opt_i),eta,1.2*gammas(opt_j),0);
fprintf('RMSE at alpha*, 1.2gamma* = %f\n',rmse);

figure;
pause(1);
imshow(noisy_im);
title('Noisy image');
colormap('jet');

figure;
pause(1);
imshow(denoised_image);
title('Denoised image');
colormap('jet');

figure;
pause(1);
imshow(clear_im);
title('Original image')
colormap('jet');
