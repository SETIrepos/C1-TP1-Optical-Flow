"""
Comparaison des résultats de seuillage Sigma-Delta
dans les espaces niveaux de gris, HSV et LAB
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import sys

# Importer la classe SigmaDelta
sys.path.insert(0, os.path.dirname(__file__))
from sigma_delta import SigmaDeltaBasic


class SigmaDeltaColor:
    """Version simplifiée pour les espaces couleur"""
    
    def __init__(self, height, width, color_space='RGB', N=2, Vmin=2, channels_to_use=None):
        self.height = height
        self.width = width
        self.color_space = color_space
        self.N = N
        self.Vmin = Vmin
        self.channels_to_use = channels_to_use if channels_to_use else [0, 1, 2]
        
        self.M = [np.zeros((height, width), dtype=np.uint8) for _ in self.channels_to_use]
        self.V = [np.ones((height, width), dtype=np.uint8) * Vmin for _ in self.channels_to_use]
        self.first_frame = True
    
    def convert_color_space(self, frame):
        if self.color_space == 'HSV':
            return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        elif self.color_space == 'LAB':
            return cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        else:
            return frame
    
    def update(self, frame):
        frame_color = self.convert_color_space(frame)
        
        if self.first_frame:
            for i, ch in enumerate(self.channels_to_use):
                self.M[i] = frame_color[:, :, ch].copy()
            self.first_frame = False
            return np.zeros((self.height, self.width), dtype=np.uint8)
        
        combined_mask = np.zeros((self.height, self.width), dtype=np.bool_)
        
        for i, ch in enumerate(self.channels_to_use):
            channel = frame_color[:, :, ch].astype(np.uint8)
            
            M_new = self.M[i].copy()
            M_new[channel > self.M[i]] += 1
            M_new[channel < self.M[i]] -= 1
            self.M[i] = M_new
            
            if self.color_space == 'HSV' and ch == 0:
                diff1 = np.abs(channel.astype(np.int16) - self.M[i].astype(np.int16))
                diff2 = 180 - diff1
                diff = np.minimum(diff1, diff2)
            else:
                diff = np.abs(channel.astype(np.int16) - self.M[i].astype(np.int16))
            
            V_new = self.V[i].copy()
            V_new[diff > self.V[i]] += 1
            V_new[diff <= self.V[i]] = np.maximum(V_new[diff <= self.V[i]] - 1, self.Vmin)
            self.V[i] = V_new
            
            threshold = self.N * self.V[i]
            channel_mask = diff > threshold
            combined_mask = np.logical_or(combined_mask, channel_mask)
        
        return (combined_mask.astype(np.uint8) * 255)


def compare_color_spaces(image_dir, n_images, output_dir):
    """
    Compare les résultats de Sigma-Delta dans différents espaces
    
    Args:
        image_dir: Dossier contenant les images
        n_images: Nombre d'images à traiter
        output_dir: Dossier de sortie
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Configurations à tester
    configs = [
        {'name': 'Niveaux de gris', 'detector': None, 'type': 'gray'},
        {'name': 'HSV (H+S)', 'detector': None, 'type': 'hsv_hs'},
        {'name': 'LAB (a+b)', 'detector': None, 'type': 'lab_ab'}
    ]
    
    last_frame = None
    
    print("\nTraitement des images...")
    for i in range(1, n_images + 1):
        img_path = os.path.join(image_dir, f'image_{i:03d}.jpg')
        
        if not os.path.exists(img_path):
            continue
        
        # Charger l'image
        frame_color = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if frame_color is None:
            continue
        
        frame_gray = cv2.cvtColor(frame_color, cv2.COLOR_BGR2GRAY)
        last_frame = frame_color
        
        # Initialiser les détecteurs
        if configs[0]['detector'] is None:
            h, w = frame_gray.shape
            configs[0]['detector'] = SigmaDeltaBasic(h, w, N=2, Vmin=2)
            configs[1]['detector'] = SigmaDeltaColor(h, w, 'HSV', N=2, Vmin=2, channels_to_use=[0, 1])
            configs[2]['detector'] = SigmaDeltaColor(h, w, 'LAB', N=2, Vmin=2, channels_to_use=[1, 2])
        
        # Mettre à jour chaque détecteur
        configs[0]['mask'] = configs[0]['detector'].update(frame_gray)
        configs[1]['mask'] = configs[1]['detector'].update(frame_color)
        configs[2]['mask'] = configs[2]['detector'].update(frame_color)
        
        if i % 20 == 0:
            print(f"  {i}/{n_images} images traitées")
    
    print(f"  {n_images}/{n_images} images traitées")
    
    # Créer la figure de comparaison
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Image originale
    img_rgb = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)
    axes[0].imshow(img_rgb)
    axes[0].set_title('Image originale', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Masques de détection
    for idx, config in enumerate(configs):
        mask = config['mask']
        axes[idx + 1].imshow(mask, cmap='gray', vmin=0, vmax=255)
        axes[idx + 1].set_title(config['name'], fontsize=14, fontweight='bold')
        axes[idx + 1].axis('off')
        
        # Calculer le pourcentage
        pct = (mask > 0).sum() / mask.size * 100
        axes[idx + 1].text(0.5, -0.08, f'{pct:.1f}% mouvement',
                          transform=axes[idx + 1].transAxes,
                          ha='center', fontsize=11)
    
    plt.tight_layout()
    
    # Sauvegarder
    output_path = os.path.join(output_dir, 'comparaison_espaces_couleur_simple.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSauvegarde : {output_path}")
    
    plt.show()
    plt.close()
    
    # Afficher les statistiques
    print("\nStatistiques de détection :")
    for config in configs:
        pct = (config['mask'] > 0).sum() / config['mask'].size * 100
        print(f"  {config['name']:20s} : {pct:.2f}% de mouvement détecté")


def main():
    """
    Fonction principale
    """
    # Configuration
    image_dir = 'TP2-Hugo'
    output_dir = 'resultats_soustraction_fond'
    n_images = 100
    
    print("=" * 70)
    print("COMPARAISON DES ESPACES COLORIMÉTRIQUES")
    print("Algorithme Sigma-Delta appliqué dans différents espaces")
    print("=" * 70)
    print(f"Dossier : {image_dir}")
    print(f"Images  : {n_images}")
    print(f"Sortie  : {output_dir}")
    print("=" * 70)
    
    # Générer la comparaison
    compare_color_spaces(image_dir, n_images, output_dir)
    
    print("\n" + "=" * 70)
    print("TERMINÉ")
    print("=" * 70)


if __name__ == "__main__":
    main()
