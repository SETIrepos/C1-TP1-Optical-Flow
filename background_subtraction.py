"""
Détection par soustraction de fond
Calcul de la moyenne et de l'écart-type pour N images
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from pathlib import Path

def load_images(image_dir, n_images):
    """
    Charge les N premières images en niveaux de gris
    
    Args:
        image_dir: Dossier contenant les images
        n_images: Nombre d'images à charger
    
    Returns:
        numpy array de dimension (n_images, height, width)
    """
    images = []
    
    for i in range(1, n_images + 1):
        # Format: image_001.jpg, image_002.jpg, etc.
        img_path = os.path.join(image_dir, f'image_{i:03d}.jpg')
        
        if not os.path.exists(img_path):
            print(f"Attention: {img_path} n'existe pas")
            continue
        
        # Charger en niveaux de gris
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is not None:
            images.append(img)
        else:
            print(f"Erreur lors du chargement de {img_path}")
    
    if len(images) == 0:
        raise ValueError("Aucune image n'a pu être chargée")
    
    print(f"{len(images)} images chargées sur {n_images} demandées")
    
    # Convertir en array numpy (N, H, W)
    return np.array(images, dtype=np.float32)


def compute_mean_std(images):
    """
    Calcule la moyenne et l'écart-type pixel par pixel
    
    Args:
        images: numpy array (N, H, W)
    
    Returns:
        mean_image: moyenne (H, W)
        std_image: écart-type (H, W)
    """
    # Calcul de la moyenne le long de l'axe 0 (les N images)
    mean_image = np.mean(images, axis=0)
    
    # Calcul de l'écart-type le long de l'axe 0
    std_image = np.std(images, axis=0)
    
    return mean_image, std_image


def process_and_save(image_dir, n_values, output_dir):
    """
    Traite les images pour différentes valeurs de N et sauvegarde les résultats
    
    Args:
        image_dir: Dossier contenant les images source
        n_values: Liste des valeurs de N à tester
        output_dir: Dossier de sortie pour les résultats
    """
    # Créer le dossier de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    for N in n_values:
        print(f"\n=== Traitement pour N = {N} ===")
        
        # Charger les images
        images = load_images(image_dir, N)
        
        # Calculer moyenne et écart-type
        mean_img, std_img = compute_mean_std(images)
        
        # Sauvegarder les résultats
        results[N] = {
            'mean': mean_img,
            'std': std_img,
            'n_images': len(images)
        }
        
        # Sauvegarder les images
        cv2.imwrite(
            os.path.join(output_dir, f'mean_N{N}.png'),
            mean_img.astype(np.uint8)
        )
        cv2.imwrite(
            os.path.join(output_dir, f'std_N{N}.png'),
            std_img.astype(np.uint8)
        )
        
        print(f"Moyenne - min: {mean_img.min():.2f}, max: {mean_img.max():.2f}")
        print(f"Écart-type - min: {std_img.min():.2f}, max: {std_img.max():.2f}")
    
    return results


def visualize_results(results, output_dir):
    """
    Crée une visualisation des résultats pour les différentes valeurs de N
    
    Args:
        results: Dictionnaire des résultats
        output_dir: Dossier de sortie
    """
    n_values = sorted(results.keys())
    
    # Figure pour les moyennes
    fig1, axes1 = plt.subplots(1, len(n_values), figsize=(15, 5))
    if len(n_values) == 1:
        axes1 = [axes1]
    
    for idx, N in enumerate(n_values):
        axes1[idx].imshow(results[N]['mean'], cmap='gray', vmin=0, vmax=255)
        axes1[idx].set_title(f'Moyenne (N={N})')
        axes1[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparaison_moyennes.png'), dpi=150)
    print(f"\nSauvegarde: {os.path.join(output_dir, 'comparaison_moyennes.png')}")
    
    # Figure pour les écarts-types
    fig2, axes2 = plt.subplots(1, len(n_values), figsize=(15, 5))
    if len(n_values) == 1:
        axes2 = [axes2]
    
    for idx, N in enumerate(n_values):
        im = axes2[idx].imshow(results[N]['std'], cmap='jet', vmin=0, vmax=50)
        axes2[idx].set_title(f'Écart-type (N={N})')
        axes2[idx].axis('off')
        plt.colorbar(im, ax=axes2[idx], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparaison_ecarts_types.png'), dpi=150)
    print(f"Sauvegarde: {os.path.join(output_dir, 'comparaison_ecarts_types.png')}")
    
    # Figure pour les seuillages
    # On choisit un seuil raisonnable (par exemple 15-20)
    threshold = 30
    
    fig3, axes3 = plt.subplots(1, len(n_values), figsize=(15, 5))
    if len(n_values) == 1:
        axes3 = [axes3]
    
    for idx, N in enumerate(n_values):
        # Créer le masque binaire : 1 (blanc) pour mouvement, 0 (noir) pour fond
        mask = (results[N]['std'] > threshold).astype(np.uint8) * 255
        
        axes3[idx].imshow(mask, cmap='gray', vmin=0, vmax=255)
        axes3[idx].set_title(f'Seuillage (N={N}, seuil={threshold})')
        axes3[idx].axis('off')
        
        # Calculer le pourcentage de pixels détectés comme mouvement
        pct_movement = (mask > 0).sum() / mask.size * 100
        axes3[idx].text(0.5, -0.1, f'{pct_movement:.1f}% mouvement', 
                       transform=axes3[idx].transAxes, 
                       ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparaison_seuillage_ecarts_types.png'), dpi=150)
    print(f"Sauvegarde: {os.path.join(output_dir, 'comparaison_seuillage_ecarts_types.png')}")
    print(f"Seuil utilisé: σ > {threshold}")
    
    plt.show()


def main():
    """
    Fonction principale
    """
    # Configuration
    image_dir = 'TP2-Hugo'
    output_dir = 'resultats_soustraction_fond'
    n_values = [5, 20, 100]
    
    print("=== Détection par soustraction de fond ===")
    print(f"Dossier d'images: {image_dir}")
    print(f"Valeurs de N testées: {n_values}")
    
    # Vérifier que le dossier existe
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Le dossier {image_dir} n'existe pas")
    
    # Traitement
    results = process_and_save(image_dir, n_values, output_dir)
    
    # Visualisation
    visualize_results(results, output_dir)
    
    print(f"\n=== Traitement terminé ===")
    print(f"Résultats sauvegardés dans: {output_dir}")
    print(f"  - 6 images individuelles (3 moyennes + 3 écarts-types)")
    print(f"  - 3 images de comparaison (moyennes, écarts-types, seuillage)")


if __name__ == "__main__":
    main()
