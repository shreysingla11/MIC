function [denoised_image, rmse] = denoise(noisy_im,clear_im,alpha,eta,gamma, plot_im)
    
    present = noisy_im;
    min_iters = 100;
    br = 0.001;

    losses = [loss(noisy_im, present,alpha,gamma)];
    %rmses = [];
    %etas = [];
    for k = 1:10000
        [curr_loss, curr_loss_dash] = loss(noisy_im, present,alpha,gamma);

        %rmses = [rmses, RRMSE(clear_im, present)];
        next_loss = loss(noisy_im,present-eta*curr_loss_dash,alpha,gamma);
        
        if (curr_loss < next_loss)
            eta = eta/2;
            next_loss = curr_loss;
        else
            present = present - eta*curr_loss_dash;
            eta = eta*1.1;
        end
        
        losses = [losses,next_loss];
        %etas = [etas, eta];
        
        if abs(next_loss - curr_loss) > 0 && abs(next_loss-curr_loss) < br && k > min_iters
            break
        end
    end

    denoised_image = present;
    rmse = RRMSE(clear_im, present);
    
    if plot_im == 1
        figure;
        plot(losses);
        title(sprintf('Loss - Gamma %f - Alpha %f',gamma, alpha));
        %figure;
        %plot(rmses);
        %title(sprintf('RMSE - Gamma %f - Alpha %f',gamma, alpha));
        %figure;
        %plot(etas(2:end));
        %title(sprintf('Step size - Gamma %f - Alpha %f',gamma, alpha));
        %close all;
    end

end