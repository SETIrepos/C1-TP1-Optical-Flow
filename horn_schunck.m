function [u, v] = horn_schunck(I1, I2, alpha, n_iter, gauss_sigma)
% HORN_SCHUNCK  Implémentation du flot optique Horn & Schunck (1981)
%
% Inputs :
%   I1, I2       : images successives (grayscale, type double, [0..1])
%   alpha        : paramètre de régularisation (ex: 10)
%   n_iter       : nombre d'itérations (ex: 200)
%   gauss_sigma  : sigma du filtrage gaussien (recommandé 0.5–1.0)
%
% Outputs :
%   u, v         : champs de déplacement horizontal et vertical

    if nargin < 5
        gauss_sigma = 0;
    end

    % === (optionnel) filtrage gaussien pour réduire le bruit ===
    if gauss_sigma > 0
        h = fspecial('gaussian', [5 5], gauss_sigma);
        I1 = imfilter(I1, h, 'replicate');
        I2 = imfilter(I2, h, 'replicate');
    end

    % === 1) Calcul des dérivées Ex, Ey, Et (Section 7) ===
    kx = 0.25 * [-1 1; -1 1];
    ky = 0.25 * [-1 -1; 1 1];
    kt = 0.25 * [1 1; 1 1];

    Ex = conv2(I1, kx, 'same') + conv2(I2, kx, 'same');
    Ey = conv2(I1, ky, 'same') + conv2(I2, ky, 'same');
    Et = conv2(I2, kt, 'same') - conv2(I1, kt, 'same');

    % === 2) Initialisation ===
    u = zeros(size(I1));
    v = zeros(size(I1));

    % === Noyau de moyenne (Section 8) ===
    avg_k = [1/12 1/6 1/12;
             1/6   0  1/6;
             1/12 1/6 1/12];

    % === 3) Mise à jour itérative (Section 12) ===
    for it = 1:n_iter
        u_bar = conv2(u, avg_k, 'same');
        v_bar = conv2(v, avg_k, 'same');

        num = Ex .* u_bar + Ey .* v_bar + Et;
        den = alpha^2 + Ex.^2 + Ey.^2;

        u = u_bar - Ex .* (num ./ den);
        v = v_bar - Ey .* (num ./ den);
    end
end
