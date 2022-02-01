function res = RRMSE(A, B)
    res = sqrt(sum((abs(A) - abs(B)).^2, 'all'))/sqrt(sum(abs(A).^2, 'all'));
end