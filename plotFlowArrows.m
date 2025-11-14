function plotFlowArrows(u, v, stride, scale)
% plotFlowArrows(u, v, stride, scale)
%
% Affiche le flow optique (u, v) sous forme d'un champ de flèches
% similaire aux figures de l’article de Horn & Schunck.
%
% Paramètres :
%   u, v  : champs de flot calculés
%   stride : facteur d'échantillonnage, ex: 5 ou 10 (par défaut 8)
%   scale  : échelle des flèches (par défaut 1)
%
% Exemple :
%   plotFlowArrows(u, v, 8, 1);

    if nargin < 3
        stride = 8; % sous-échantillonnage par défaut
    end
    if nargin < 4
        scale = 1;  % échelle par défaut
    end

    [h, w] = size(u);

    % grille sous-échantillonnée
    [X, Y] = meshgrid(1:stride:w, 1:stride:h);

    % champs sous-échantillonnés
    u_sub = u(1:stride:end, 1:stride:end);
    v_sub = v(1:stride:end, 1:stride:end);

    % affichage des flèches
    quiver(X, Y, u_sub, v_sub, scale, 'color', 'cyan', 'linewidth', 1.2);

    axis ij;        % inverser l’axe vertical pour correspondre à l’image
    axis equal;     % pixels carrés
    axis tight;     % zoom sur le domaine
    title('Champ de flot optique (Horn & Schunck)');
end
