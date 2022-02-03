function [f,der] = loss(noisy_im, image, alpha,gamma, prior_ind)
    [li,li_dash] = likelihood(image,noisy_im,1-alpha);
    [pr,pr_dash] = prior(image,alpha,gamma, prior_ind);
    f = pr + li;
    der = pr_dash + li_dash;
end