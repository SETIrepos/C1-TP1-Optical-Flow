clear; close all; clc;

% Charger les images
I1 = im2double(imread('image_001.jpg'));
I2 = im2double(imread('image_002.jpg'));

% Convertir si couleur
if size(I1,3)==3
    I1 = rgb2gray(I1);
    I2 = rgb2gray(I2);
end

alpha = 10;         % recommandé
n_iter = 100;       % nombre d'itérations
sigma = 0.8;        % filtrage gaussien

% Flot optique Horn & Schunck
[u, v] = horn_schunck(I1, I2, alpha, n_iter, sigma);

mag = sqrt(u.^2 + v.^2);

% seuil adaptatif
th = mean(mag(:)) + 0.5 * std(mag(:));

mask = mag > th;

% Visualisation (ta fonction)
flow_img = flowToColor(u, v);

figure;
subplot(1,3,1); imshow(I1); title('Image 1');
subplot(1,3,2); imshow(flow_img); title('Flot optique (HS)');
% subplot(1,3,3); quiver(u, v); axis ij; title('Vecteurs (u,v)');
subplot(1,3,3); imshow(mask);title('Pixels en mouvement');

[u1, v1] = horn_schunck(I1, I2, 10, 1000, sigma);

figure;
imshow(I1);
hold on;
plotFlowArrows(v, u, 10, 1);

figure;
imshow(I1);
hold on;
plotFlowArrows(v1, u1, 10, 1);



