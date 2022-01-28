clf;
clear;
images = load("../data/assignmentImageDenoisingPhantom.mat");
noisy_im = images.imageNoisy;
clear_im = images.imageNoiseless;

present = noisy_im;
shape = size(noisy_im);
h = shape(1);
w = shape(2);
alpha = 0.8;
eta = 0.01;
iters = 100;
Losses = zeros(iters,1);
for k=1:iters
    Losses(k) = loss(noisy_im, present);
    der = (1-alpha)*operate_der_like(noisy_im, present) + alpha*operate_der_prior(present);
    present = present - eta*der;
end 
imshow(noisy_im);
figure;
imshow(present);
figure;
imshow(clear_im);
figure;
plot(Losses);