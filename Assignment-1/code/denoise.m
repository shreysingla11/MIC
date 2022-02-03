function [denoised_image, rmse] = denoise(noisy_im,clear_im,alpha,eta,gamma, plot_im)
    
    present = noisy_im;
    min_iters = 100; % Minimum iterations for gradient descent
    max_iters = 1000; % Maximum iterations for gradient descent
    br = 0.001; % Loss threshold to stop gradient descent

    losses = zeros(max_iters + 1); % Array to store the losses at each iteration
    losses(1) = loss(noisy_im, present,alpha,gamma);

    for k = 1:max_iters
        [curr_loss, curr_loss_dash] = loss(noisy_im, present,alpha,gamma);

        next_loss = loss(noisy_im,present-eta*curr_loss_dash,alpha,gamma);
        
        % Update the present estimate only if loss decreased
        if (curr_loss < next_loss) 
            eta = eta/2;
            next_loss = curr_loss;
        else
            present = present - eta*curr_loss_dash;
            eta = eta*1.1;
        end
        
        losses(k+1) = next_loss;
        
        % Stopping codition for gradient descent
        if abs(next_loss - curr_loss) > 0 && abs(next_loss-curr_loss) < br && k > min_iters
            break
        end
    end

    denoised_image = present;
    rmse = RRMSE(clear_im, present); % Calculate RRMSE between final denoised image and actual image
    
    % Code to plot the loss values at each iteration if plot_im flag is set
    if plot_im == 1 
        figure;
        plot(losses(1:k+1));
        title(sprintf('Loss - Gamma %f - Alpha %f',gamma, alpha));
    end

end