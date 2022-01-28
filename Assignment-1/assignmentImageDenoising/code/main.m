% clf;
clear;
images = load("../data/assignmentImageDenoisingPhantom.mat");
noisy_im = images.imageNoisy;
clear_im = images.imageNoiseless;
% figure;
% imshow(noisy_im);
% figure;
% imshow(clear_im);

present = noisy_im;
shape = size(noisy_im);
h = shape(1);
w = shape(2);
alpha = 0.5;
eta = 0.1;
for k=1:10
    der = (1-alpha)*operate_der_like(noisy_im, present) + alpha*operate_der_prior(present);
    present = present - eta*der;
end 
% imshow(noisy_im);
% figure;
% imshow(present);
% figure;
% imshow(clear_im);