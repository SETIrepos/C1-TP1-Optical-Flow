"""
Application de Sigma-Delta sur les canaux a+b de l'espace LAB
pour réduire la détection des ombres
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from sigma_delta import SigmaDeltaBasic


def lab_ab_to_gray(lab_image):
    """
    Convertit les canaux a et b de LAB en une image en niveaux de gris
    en calculant la magnitude de la chrominance
    
    Args:
        lab_image: Image LAB (3 canaux)
    
    Returns:
        Image en niveaux de gris basée sur a et b
    """
    # Extraire les canaux a et b
    a = lab_image[:, :, 1].astype(np.float32)
    b = lab_image[:, :, 2].astype(np.float32)
    
    # Calculer la magnitude de la chrominance: sqrt(a^2 + b^2)
    # Normaliser pour avoir des valeurs entre 0 et 255
    chroma_magnitude = np.sqrt(a**2 + b**2)
    
    # Normaliser
    chroma_normalized = cv2.normalize(chroma_magnitude, None, 0, 255, cv2.NORM_MINMAX)
    
    return chroma_normalized.astype(np.uint8)


def compare_gray_vs_lab_ab(image_dir, n_images, output_dir):
    """
    Compare Sigma-Delta en niveaux de gris vs LAB (a+b)
    
    Args:
        image_dir: Dossier des images
        n_images: Nombre d'images à traiter
        output_dir: Dossier de sortie
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Deux détecteurs
    detector_gray = None
    detector_lab_ab = None
    
    last_frame_bgr = None
    last_gray = None
    last_lab_ab_gray = None
    
    print("\nTraitement des images...")
    for i in range(1, n_images + 1):
        img_path = os.path.join(image_dir, f'image_{i:03d}.jpg')
        
        if not os.path.exists(img_path):
            continue
        
        # Charger l'image en couleur
        frame_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            continue
        
        last_frame_bgr = frame_bgr
        
        # Convertir en niveaux de gris
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        last_gray = frame_gray
        
        # Convertir en LAB et extraire a+b comme niveaux de gris
        frame_lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        frame_lab_ab_gray = lab_ab_to_gray(frame_lab)
        last_lab_ab_gray = frame_lab_ab_gray
        
        # Initialiser les détecteurs
        if detector_gray is None:
            h, w = frame_gray.shape
            detector_gray = SigmaDeltaBasic(h, w, N=2, Vmin=2)
            detector_lab_ab = SigmaDeltaBasic(h, w, N=2, Vmin=2)
            print(f"  Détecteurs initialisés : {h}x{w}")
        
        # Mettre à jour les deux détecteurs
        mask_gray = detector_gray.update(frame_gray)
        mask_lab_ab = detector_lab_ab.update(frame_lab_ab_gray)
        
        if i % 20 == 0:
            print(f"  {i}/{n_images} images traitées")
    
    print(f"  {n_images}/{n_images} images traitées")
    
    # Récupérer les masques finaux
    final_mask_gray = mask_gray
    final_mask_lab_ab = mask_lab_ab
    
    # Créer la visualisation
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Ligne 1 : Images originales / transformées
    # Image couleur originale
    img_rgb = cv2.cvtColor(last_frame_bgr, cv2.COLOR_BGR2RGB)
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title('Image originale (RGB)', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Niveaux de gris
    axes[0, 1].imshow(last_gray, cmap='gray')
    axes[0, 1].set_title('Niveaux de gris', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # LAB a+b en niveaux de gris
    axes[0, 2].imshow(last_lab_ab_gray, cmap='gray')
    axes[0, 2].set_title('LAB (a+b) → Niveaux de gris\n(magnitude chrominance)', fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Ligne 2 : Masques de détection
    # Masque vide pour alignement
    axes[1, 0].axis('off')
    
    # Masque niveaux de gris
    axes[1, 1].imshow(final_mask_gray, cmap='gray', vmin=0, vmax=255)
    axes[1, 1].set_title(f'Sigma-Delta (Niveaux de gris)', 
                         fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Masque LAB a+b
    axes[1, 2].imshow(final_mask_lab_ab, cmap='gray', vmin=0, vmax=255)
    axes[1, 2].set_title(f'Sigma-Delta (LAB a+b)', 
                         fontsize=14, fontweight='bold', color='green')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    # Sauvegarder
    output_path = os.path.join(output_dir, 'comparaison_seuillage.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSauvegarde : {output_path}")
    
    plt.show()
    plt.close()
    
    # Créer une deuxième figure plus simple pour le compte-rendu
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
    
    # Image originale
    axes2[0].imshow(img_rgb)
    axes2[0].set_title('Image originale', fontsize=14, fontweight='bold')
    axes2[0].axis('off')
    
    # Masque niveaux de gris
    axes2[1].imshow(final_mask_gray, cmap='gray', vmin=0, vmax=255)
    axes2[1].set_title(f'Niveaux de gris', 
                      fontsize=14, fontweight='bold')
    axes2[1].axis('off')
    
    # Masque LAB a+b
    axes2[2].imshow(final_mask_lab_ab, cmap='gray', vmin=0, vmax=255)
    axes2[2].set_title(f'LAB (a+b)', 
                      fontsize=14, fontweight='bold', color='green')
    axes2[2].axis('off')
    
    plt.tight_layout()
    
    output_path2 = os.path.join(output_dir, 'comparaison_seuillage_simple.png')
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"Sauvegarde : {output_path2}")
    
    plt.show()
    plt.close()


def main():
    """
    Fonction principale
    """
    image_dir = 'TP2-Hugo'
    output_dir = 'resultats_soustraction_fond'
    n_images = 100
    
    print("=" * 70)
    print("SIGMA-DELTA SUR LAB (a+b) vs NIVEAUX DE GRIS")
    print("=" * 70)
    print(f"Dossier : {image_dir}")
    print(f"Images  : {n_images}")
    print("=" * 70)
    
    # Traitement
    compare_gray_vs_lab_ab(image_dir, n_images, output_dir)
    
    print("\n" + "=" * 70)
    print("TERMINÉ")
    print("=" * 70)
    print("Images générées :")
    print("  - comparaison_seuillage.png (vue complète)")
    print("  - comparaison_seuillage_simple.png (pour compte-rendu)")
    print("=" * 70)


if __name__ == "__main__":
    main()
