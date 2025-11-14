"""
Analyse temporelle d'un pixel spécifique
Trace l'évolution du niveau de gris du pixel (200, 150) au cours du temps
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def analyze_pixel_temporal(image_dir, pixel_coords, n_images=100):
    """
    Analyse l'évolution temporelle d'un pixel spécifique
    
    Args:
        image_dir: Dossier contenant les images
        pixel_coords: (y, x) coordonnées du pixel à analyser
        n_images: Nombre d'images à analyser
    
    Returns:
        array des valeurs du pixel au cours du temps
    """
    y, x = pixel_coords
    pixel_values = []
    
    for i in range(1, n_images + 1):
        img_path = os.path.join(image_dir, f'image_{i:03d}.jpg')
        
        if not os.path.exists(img_path):
            continue
        
        # Charger en niveaux de gris
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is not None:
            pixel_values.append(img[y, x])
    
    return np.array(pixel_values)


def plot_pixel_evolution(pixel_values, pixel_coords, output_dir):
    """
    Trace l'évolution temporelle du pixel
    
    Args:
        pixel_values: Valeurs du pixel au cours du temps
        pixel_coords: (y, x) coordonnées du pixel
        output_dir: Dossier de sortie
    """
    os.makedirs(output_dir, exist_ok=True)
    
    y, x = pixel_coords
    
    # Calculer les statistiques
    mean_val = np.mean(pixel_values)
    std_val = np.std(pixel_values)
    
    # Créer la figure
    plt.figure(figsize=(14, 6))
    
    # Tracer l'évolution
    plt.plot(pixel_values, linewidth=1.5, color='blue', label='Niveau de gris')
    plt.axhline(y=mean_val, color='red', linestyle='--', linewidth=2, 
                label=f'Moyenne μ = {mean_val:.1f}')
    plt.axhline(y=mean_val + std_val, color='orange', linestyle=':', linewidth=1.5,
                label=f'μ ± σ (σ = {std_val:.1f})')
    plt.axhline(y=mean_val - std_val, color='orange', linestyle=':', linewidth=1.5)
    
    plt.xlabel('Numéro d\'image (temps)', fontsize=12)
    plt.ylabel('Niveau de gris', fontsize=12)
    plt.title(f'Évolution temporelle du pixel ({y}, {x})', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.ylim(0, 255)
    
    plt.tight_layout()
    
    # Sauvegarder
    output_path = os.path.join(output_dir, f'evolution_pixel_{y}_{x}.png')
    plt.savefig(output_path, dpi=150)
    print(f"Sauvegarde: {output_path}")
    print(f"Statistiques du pixel ({y}, {x}):")
    print(f"  - Moyenne μ = {mean_val:.2f}")
    print(f"  - Écart-type σ = {std_val:.2f}")
    print(f"  - Min = {pixel_values.min()}")
    print(f"  - Max = {pixel_values.max()}")
    
    # Compter les variations brusques (changement > 30 entre deux images consécutives)
    diff = np.abs(np.diff(pixel_values))
    sharp_changes = np.sum(diff > 30)
    print(f"  - Nombre de variations brusques (Δ > 30) : {sharp_changes}")
    
    plt.show()
    
    return mean_val, std_val


def main():
    """
    Fonction principale
    """
    # Configuration
    image_dir = 'Images'
    output_dir = 'resultats_soustraction_fond'
    pixel_coords = (200, 150)  # (y, x)
    n_images = 600
    
    print("=== Analyse temporelle d'un pixel ===")
    print(f"Pixel analysé: {pixel_coords}")
    print(f"Nombre d'images: {n_images}")
    
    # Vérifier que le dossier existe
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Le dossier {image_dir} n'existe pas")
    
    # Analyser le pixel
    pixel_values = analyze_pixel_temporal(image_dir, pixel_coords, n_images)
    
    print(f"\n{len(pixel_values)} images analysées")
    
    # Tracer l'évolution
    plot_pixel_evolution(pixel_values, pixel_coords, output_dir)
    
    print(f"\n=== Analyse terminée ===")


if __name__ == "__main__":
    main()
