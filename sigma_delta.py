"""
Algorithme Sigma-Delta pour la détection de mouvement
Implémentation basique selon la littérature
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


class SigmaDeltaBasic:
    """
    Détecteur de mouvement basé sur l'algorithme Sigma-Delta basique
    
    Références :
    - Manzanera & Richefeu (2007)
    - L'algorithme maintient deux variables par pixel :
      M : estimation de la moyenne (fond)
      V : estimation de l'écart-type (variance)
    """
    
    def __init__(self, height, width, N=2, Vmin=2):
        """
        Initialisation du détecteur
        
        Args:
            height: Hauteur de l'image
            width: Largeur de l'image
            N: Facteur multiplicatif pour le seuil de détection (N * V)
            Vmin: Valeur minimale de V pour éviter division par zéro et trop de sensibilité
        """
        self.height = height
        self.width = width
        self.N = N
        self.Vmin = Vmin
        
        # M : estimation de la moyenne du fond (initialisée à 0)
        self.M = np.zeros((height, width), dtype=np.uint8)
        
        # V : estimation de l'écart-type (initialisée à une valeur raisonnable)
        self.V = np.ones((height, width), dtype=np.uint8) * Vmin
        
        # Pour la première frame
        self.first_frame = True
    
    def update(self, frame):
        """
        Mise à jour du modèle de fond avec une nouvelle frame
        
        L'algorithme Sigma-Delta met à jour M et V de manière incrémentale :
        - M converge vers la valeur moyenne du pixel
        - V converge vers l'écart-type du pixel
        
        Args:
            frame: Image en niveaux de gris (numpy array uint8)
        
        Returns:
            mask: Masque binaire de détection (255 = mouvement, 0 = fond)
        """
        frame = frame.astype(np.uint8)
        
        # Initialisation avec la première frame
        if self.first_frame:
            self.M = frame.copy()
            self.first_frame = False
            # Retourner un masque vide pour la première frame
            return np.zeros((self.height, self.width), dtype=np.uint8)
        
        # ===== Mise à jour de M (estimation de la moyenne) =====
        # Si pixel actuel > M, on incrémente M de 1
        # Si pixel actuel < M, on décrémente M de 1
        # Sinon M reste inchangé
        M_new = self.M.copy()
        M_new[frame > self.M] += 1
        M_new[frame < self.M] -= 1
        self.M = M_new
        
        # ===== Calcul de la différence absolue =====
        diff = np.abs(frame.astype(np.int16) - self.M.astype(np.int16))
        
        # ===== Mise à jour de V (estimation de l'écart-type) =====
        # Si |frame - M| > V, on incrémente V
        # Si |frame - M| <= V, on décrémente V (mais pas en-dessous de Vmin)
        V_new = self.V.copy()
        V_new[diff > self.V] += 1
        V_new[diff <= self.V] = np.maximum(V_new[diff <= self.V] - 1, self.Vmin)
        self.V = V_new
        
        # ===== Détection =====
        # Un pixel est détecté comme mouvement si : |frame - M| > N * V
        threshold = self.N * self.V
        mask = (diff > threshold).astype(np.uint8) * 255
        
        return mask
    
    def get_background(self):
        """Retourne le modèle de fond actuel (M)"""
        return self.M.copy()
    
    def get_variance(self):
        """Retourne l'estimation de la variance actuelle (V)"""
        return self.V.copy()


def process_sequence(image_dir, n_images, N_values, output_dir):
    """
    Traite une séquence d'images avec différentes valeurs de N
    
    Args:
        image_dir: Dossier contenant les images
        n_images: Nombre d'images à traiter
        N_values: Liste des valeurs de N à tester
        output_dir: Dossier de sortie
    """
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    for N in N_values:
        print(f"\n=== Traitement avec N = {N} ===")
        
        detector = None
        masks = []
        frames_loaded = []
        
        for i in range(1, n_images + 1):
            img_path = os.path.join(image_dir, f'image_{i:03d}.jpg')
            
            if not os.path.exists(img_path):
                print(f"  Image {img_path} non trouvée")
                continue
            
            # Charger l'image en niveaux de gris
            frame = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if frame is None:
                print(f"  Erreur de chargement : {img_path}")
                continue
            
            # Initialiser le détecteur avec la première image
            if detector is None:
                height, width = frame.shape
                detector = SigmaDeltaBasic(height, width, N=N, Vmin=2)
                print(f"  Détecteur initialisé : {height}x{width}, N={N}")
            
            # Mettre à jour et obtenir le masque
            mask = detector.update(frame)
            masks.append(mask)
            frames_loaded.append(frame)
        
        print(f"  {len(masks)} images traitées")
        
        # Sauvegarder quelques résultats intermédiaires (images 20, 50, 100)
        for idx in [19, 49, 99]:
            if idx < len(masks):
                output_path = os.path.join(output_dir, f'sigmadelta_N{N}_img{idx+1:03d}.png')
                cv2.imwrite(output_path, masks[idx])
                print(f"  Sauvegarde : sigmadelta_N{N}_img{idx+1:03d}.png")
        
        # Garder les résultats finaux
        if len(masks) > 0:
            results[N] = {
                'final_mask': masks[-1],
                'M': detector.get_background(),
                'V': detector.get_variance(),
                'n_frames': len(masks),
                'last_frame': frames_loaded[-1]
            }
            
            # Sauvegarder M (fond estimé)
            cv2.imwrite(
                os.path.join(output_dir, f'sigmadelta_M_N{N}.png'),
                detector.get_background()
            )
            
            # Sauvegarder V (variance estimée) - multiplié pour la visualisation
            V_vis = np.clip(detector.get_variance() * 10, 0, 255).astype(np.uint8)
            cv2.imwrite(
                os.path.join(output_dir, f'sigmadelta_V_N{N}.png'),
                V_vis
            )
            
            print(f"  M : min={detector.M.min()}, max={detector.M.max()}, mean={detector.M.mean():.1f}")
            print(f"  V : min={detector.V.min()}, max={detector.V.max()}, mean={detector.V.mean():.1f}")
            pct_motion = (masks[-1] > 0).sum() / masks[-1].size * 100
            print(f"  Mouvement détecté (dernière frame) : {pct_motion:.2f}%")
    
    return results


def visualize_results(results, output_dir):
    """
    Visualise les résultats pour différentes valeurs de N
    
    Args:
        results: Dictionnaire des résultats
        output_dir: Dossier de sortie
    """
    n_values = sorted(results.keys())
    
    # Figure 1 : Image originale + masques de détection
    fig1, axes1 = plt.subplots(1, len(n_values) + 1, figsize=(18, 5))
    
    # Afficher l'image originale (dernière frame)
    axes1[0].imshow(results[n_values[0]]['last_frame'], cmap='gray')
    axes1[0].set_title('Image originale\n(dernière frame)', fontsize=12)
    axes1[0].axis('off')
    
    # Afficher les masques pour chaque N
    for idx, N in enumerate(n_values):
        axes1[idx + 1].imshow(results[N]['final_mask'], cmap='gray', vmin=0, vmax=255)
        axes1[idx + 1].set_title(f'Sigma-Delta\nN={N}', fontsize=12)
        axes1[idx + 1].axis('off')
        
        # Calculer le pourcentage de pixels détectés
        pct = (results[N]['final_mask'] > 0).sum() / results[N]['final_mask'].size * 100
        axes1[idx + 1].text(0.5, -0.1, f'{pct:.1f}% mouvement',
                          transform=axes1[idx + 1].transAxes,
                          ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sigmadelta_comparaison.png'), dpi=150, bbox_inches='tight')
    print(f"\nSauvegarde : sigmadelta_comparaison.png")
    
    # Figure 2 : Modèles de fond (M)
    fig2, axes2 = plt.subplots(1, len(n_values), figsize=(15, 5))
    if len(n_values) == 1:
        axes2 = [axes2]
    
    for idx, N in enumerate(n_values):
        axes2[idx].imshow(results[N]['M'], cmap='gray', vmin=0, vmax=255)
        axes2[idx].set_title(f'Fond estimé M (N={N})', fontsize=12)
        axes2[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sigmadelta_fond.png'), dpi=150, bbox_inches='tight')
    print(f"Sauvegarde : sigmadelta_fond.png")
    
    plt.show()


def main():
    """
    Fonction principale
    """
    # Configuration
    image_dir = 'Images'
    output_dir = 'resultats_soustraction_fond'
    n_images = 100
    N_values = [1, 2, 4]  # Différentes valeurs de N à tester
    
    print("=" * 60)
    print("ALGORITHME SIGMA-DELTA BASIQUE")
    print("=" * 60)
    print(f"Dossier d'images : {image_dir}")
    print(f"Nombre d'images  : {n_images}")
    print(f"Valeurs de N     : {N_values}")
    print("=" * 60)
    
    # Vérifier que le dossier existe
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Le dossier {image_dir} n'existe pas")
    
    # Traitement
    results = process_sequence(image_dir, n_images, N_values, output_dir)
    
    # Visualisation
    visualize_results(results, output_dir)
    
    print("\n" + "=" * 60)
    print("TRAITEMENT TERMINÉ")
    print("=" * 60)
    print(f"Résultats dans : {output_dir}")
    print(f"  - Images de comparaison des masques")
    print(f"  - Images des fonds estimés (M)")
    print(f"  - Images des variances estimées (V)")
    print(f"  - Résultats intermédiaires (images 20, 50, 100)")
    print("=" * 60)


if __name__ == "__main__":
    main()



if __name__ == "__main__":
    main()
