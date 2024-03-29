clf;
clear;
close all;

image = 'phantom'; % Image to perform denoising on
global likelihood_type; % Variable to store type of noise model to use 
global prior_type; % Variable to store type of prior to use 

prior_type = 'discontinuity_adaptive';
likelihood_type = 'gaussian';

if strcmp(image,'phantom') 
    images = load("../data/assignmentImageDenoisingPhantom.mat");
    noisy_im = images.imageNoisy;
    clear_im = images.imageNoiseless;
else
    images = load("../data/brainMRIslice.mat");
    noisy_im = images.brainMRIsliceNoisy;
    clear_im = images.brainMRIsliceOrig;
end

fprintf('Initial RRMSE = %f\n',RRMSE(clear_im,noisy_im));

% Set hyperparameters
eta = 0.01; % Inital value for learning rate in gradient descent
alphas = linspace(0.985,0.995,3); % Array of alpha values to search over
gammas = linspace(0.0009,0.0011,3);  % Array of gamma values to search over

rmses = zeros(length(alphas),length(gammas));

% Search over the hyperparameters for best hyperparameters 
for i = 1:length(alphas)
    for j = 1:length(gammas)
        [~, rmse] = denoise(noisy_im,clear_im,alphas(i),eta,gammas(j),0);
        rmses(i,j) = rmse;
        fprintf("Tuning iteration with (Alpha =  %f, Gamma = %f) - RRMSE = %f \n",alphas(i),gammas(j),rmse);
    end
end

% Calculate the optimal hyperparameters
[~, index] = min(rmses(:));
[opt_i, opt_j] = ind2sub(size(rmses),index);

fprintf('Best alpha = %f\n',alphas(opt_i));
fprintf('Best gamma = %f\n',gammas(opt_j));

% Calculate final results for display
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

% Display figures
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
