"""
Application de Sigma-Delta dans les espaces colorimétriques HSV et LAB
pour réduire la détection des ombres
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


class SigmaDeltaColor:
    """
    Détecteur de mouvement Sigma-Delta pour images couleur
    Peut fonctionner dans différents espaces colorimétriques
    """
    
    def __init__(self, height, width, color_space='RGB', N=2, Vmin=2, channels_to_use=None):
        """
        Initialisation du détecteur
        
        Args:
            height: Hauteur de l'image
            width: Largeur de l'image
            color_space: 'RGB', 'HSV', ou 'LAB'
            N: Facteur multiplicatif pour le seuil
            Vmin: Valeur minimale de V
            channels_to_use: Liste des canaux à utiliser (None = tous)
                            Ex: [0, 2] pour H et V dans HSV
        """
        self.height = height
        self.width = width
        self.color_space = color_space
        self.N = N
        self.Vmin = Vmin
        self.channels_to_use = channels_to_use
        
        # Déterminer le nombre de canaux
        if channels_to_use is None:
            self.n_channels = 3
            self.active_channels = [0, 1, 2]
        else:
            self.n_channels = len(channels_to_use)
            self.active_channels = channels_to_use
        
        # M et V pour chaque canal actif
        self.M = [np.zeros((height, width), dtype=np.uint8) for _ in self.active_channels]
        self.V = [np.ones((height, width), dtype=np.uint8) * Vmin for _ in self.active_channels]
        
        self.first_frame = True
    
    def convert_color_space(self, frame):
        """
        Convertit l'image dans l'espace colorimétrique souhaité
        
        Args:
            frame: Image BGR (format OpenCV)
        
        Returns:
            Image convertie
        """
        if self.color_space == 'RGB':
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif self.color_space == 'HSV':
            return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        elif self.color_space == 'LAB':
            return cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        else:
            raise ValueError(f"Espace colorimétrique non supporté: {self.color_space}")
    
    def update(self, frame):
        """
        Mise à jour du modèle avec une nouvelle frame couleur
        
        Args:
            frame: Image BGR (format OpenCV)
        
        Returns:
            mask: Masque binaire de détection
        """
        # Convertir dans l'espace colorimétrique souhaité
        frame_color = self.convert_color_space(frame)
        
        # Initialisation avec la première frame
        if self.first_frame:
            for i, ch in enumerate(self.active_channels):
                self.M[i] = frame_color[:, :, ch].copy()
            self.first_frame = False
            return np.zeros((self.height, self.width), dtype=np.uint8)
        
        # Masque combiné pour tous les canaux
        combined_mask = np.zeros((self.height, self.width), dtype=np.bool_)
        
        # Traiter chaque canal actif
        for i, ch in enumerate(self.active_channels):
            channel = frame_color[:, :, ch].astype(np.uint8)
            
            # Mise à jour de M
            M_new = self.M[i].copy()
            M_new[channel > self.M[i]] += 1
            M_new[channel < self.M[i]] -= 1
            self.M[i] = M_new
            
            # Calcul de la différence
            # Traitement spécial pour le canal H (Hue) qui est circulaire
            if self.color_space == 'HSV' and ch == 0:
                # Pour Hue, on doit gérer la circularité (0° = 180° en OpenCV car 0-179)
                diff1 = np.abs(channel.astype(np.int16) - self.M[i].astype(np.int16))
                diff2 = 180 - diff1
                diff = np.minimum(diff1, diff2)
            else:
                diff = np.abs(channel.astype(np.int16) - self.M[i].astype(np.int16))
            
            # Mise à jour de V
            V_new = self.V[i].copy()
            V_new[diff > self.V[i]] += 1
            V_new[diff <= self.V[i]] = np.maximum(V_new[diff <= self.V[i]] - 1, self.Vmin)
            self.V[i] = V_new
            
            # Détection pour ce canal
            threshold = self.N * self.V[i]
            channel_mask = diff > threshold
            
            # Combiner avec OR (mouvement si au moins un canal détecte)
            combined_mask = np.logical_or(combined_mask, channel_mask)
        
        return (combined_mask.astype(np.uint8) * 255)
    
    def get_background(self):
        """Retourne le modèle de fond (M) pour tous les canaux"""
        # Reconstruire une image avec tous les canaux
        bg = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        for i, ch in enumerate(self.active_channels):
            bg[:, :, ch] = self.M[i]
        return bg


def process_sequence_color(image_dir, n_images, color_configs, output_dir):
    """
    Traite une séquence avec différentes configurations colorimétriques
    
    Args:
        image_dir: Dossier des images
        n_images: Nombre d'images à traiter
        color_configs: Liste de dictionnaires de configuration
                      [{'name': 'HSV', 'space': 'HSV', 'N': 2, 'channels': None}, ...]
        output_dir: Dossier de sortie
    """
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    for config in color_configs:
        name = config['name']
        color_space = config['space']
        N = config.get('N', 2)
        channels = config.get('channels', None)
        
        print(f"\n=== {name} ===")
        print(f"  Espace: {color_space}, N={N}, Canaux={channels}")
        
        detector = None
        masks = []
        frames_loaded = []
        
        for i in range(1, n_images + 1):
            img_path = os.path.join(image_dir, f'image_{i:03d}.jpg')
            
            if not os.path.exists(img_path):
                continue
            
            # Charger l'image en couleur (BGR)
            frame = cv2.imread(img_path, cv2.IMREAD_COLOR)
            
            if frame is None:
                continue
            
            # Initialiser le détecteur
            if detector is None:
                height, width = frame.shape[:2]
                detector = SigmaDeltaColor(height, width, color_space=color_space, 
                                          N=N, Vmin=2, channels_to_use=channels)
                print(f"  Détecteur initialisé : {height}x{width}")
            
            # Mettre à jour
            mask = detector.update(frame)
            masks.append(mask)
            frames_loaded.append(frame)
        
        print(f"  {len(masks)} images traitées")
        
        # Sauvegarder quelques résultats
        for idx in [19, 49, 99]:
            if idx < len(masks):
                output_path = os.path.join(output_dir, f'{name}_img{idx+1:03d}.png')
                cv2.imwrite(output_path, masks[idx])
        
        # Résultats finaux
        if len(masks) > 0:
            results[name] = {
                'final_mask': masks[-1],
                'background': detector.get_background(),
                'n_frames': len(masks),
                'last_frame': frames_loaded[-1],
                'color_space': color_space
            }
            
            pct_motion = (masks[-1] > 0).sum() / masks[-1].size * 100
            print(f"  Mouvement détecté (dernière frame) : {pct_motion:.2f}%")
    
    return results


def visualize_results_color(results, output_dir):
    """
    Visualise les résultats pour différents espaces colorimétriques
    
    Args:
        results: Dictionnaire des résultats
        output_dir: Dossier de sortie
    """
    configs = list(results.keys())
    
    # Figure 1: Comparaison des masques
    n_cols = len(configs) + 1
    fig1, axes1 = plt.subplots(1, n_cols, figsize=(5*n_cols, 5))
    
    # Image originale
    first_result = results[configs[0]]
    original = cv2.cvtColor(first_result['last_frame'], cv2.COLOR_BGR2RGB)
    axes1[0].imshow(original)
    axes1[0].set_title('Image originale\n(dernière frame)', fontsize=12, fontweight='bold')
    axes1[0].axis('off')
    
    # Masques pour chaque configuration
    for idx, name in enumerate(configs):
        axes1[idx + 1].imshow(results[name]['final_mask'], cmap='gray', vmin=0, vmax=255)
        axes1[idx + 1].set_title(f'{name}', fontsize=12, fontweight='bold')
        axes1[idx + 1].axis('off')
        
        pct = (results[name]['final_mask'] > 0).sum() / results[name]['final_mask'].size * 100
        axes1[idx + 1].text(0.5, -0.08, f'{pct:.1f}% mouvement',
                           transform=axes1[idx + 1].transAxes,
                           ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparaison_espaces_couleur.png'), 
                dpi=150, bbox_inches='tight')
    print(f"\nSauvegarde : comparaison_espaces_couleur.png")
    plt.close()
    
    # Figure 2: Fonds estimés (seulement pour HSV et LAB)
    color_results = {k: v for k, v in results.items() if v['color_space'] in ['HSV', 'LAB']}
    
    if len(color_results) > 0:
        fig2, axes2 = plt.subplots(1, len(color_results), figsize=(7*len(color_results), 5))
        if len(color_results) == 1:
            axes2 = [axes2]
        
        for idx, name in enumerate(color_results.keys()):
            bg = color_results[name]['background']
            
            # Convertir le fond en RGB pour l'affichage
            if color_results[name]['color_space'] == 'HSV':
                bg_rgb = cv2.cvtColor(bg, cv2.COLOR_HSV2RGB)
            elif color_results[name]['color_space'] == 'LAB':
                bg_rgb = cv2.cvtColor(bg, cv2.COLOR_LAB2RGB)
            else:
                bg_rgb = bg
            
            axes2[idx].imshow(bg_rgb)
            axes2[idx].set_title(f'Fond estimé ({name})', fontsize=12, fontweight='bold')
            axes2[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'fonds_espaces_couleur.png'), 
                    dpi=150, bbox_inches='tight')
        print(f"Sauvegarde : fonds_espaces_couleur.png")
        plt.close()


def main():
    """
    Fonction principale
    """
    # Configuration
    image_dir = 'Images'
    output_dir_hsv = 'HSV_images'
    output_dir_lab = 'LAB_images'
    output_dir = 'resultats_soustraction_fond'
    n_images = 100
    
    print("=" * 70)
    print("SIGMA-DELTA DANS ESPACES COLORIMÉTRIQUES HSV ET LAB")
    print("=" * 70)
    
    # Configurations à tester
    color_configs = [
        {
            'name': 'Niveaux de gris',
            'space': 'RGB',  # On utilisera qu'un seul canal
            'N': 2,
            'channels': [0]  # Juste le canal R (équivalent au gris si RGB=BGR converti)
        },
        {
            'name': 'HSV (H+S+V)',
            'space': 'HSV',
            'N': 2,
            'channels': None  # Tous les canaux
        },
        {
            'name': 'HSV (H+S)',
            'space': 'HSV',
            'N': 2,
            'channels': [0, 1]  # Hue et Saturation uniquement (invariant à la luminosité)
        },
        {
            'name': 'LAB (L+a+b)',
            'space': 'LAB',
            'N': 2,
            'channels': None  # Tous les canaux
        },
        {
            'name': 'LAB (a+b)',
            'space': 'LAB',
            'N': 2,
            'channels': [1, 2]  # Chrominance uniquement (invariant à la luminosité)
        }
    ]
    
    # Créer les dossiers de sortie
    os.makedirs(output_dir_hsv, exist_ok=True)
    os.makedirs(output_dir_lab, exist_ok=True)
    
    # Traitement
    results = process_sequence_color(image_dir, n_images, color_configs, output_dir)
    
    # Visualisation
    visualize_results_color(results, output_dir)
    
    print("\n" + "=" * 70)
    print("TRAITEMENT TERMINÉ")
    print("=" * 70)
    print(f"Résultats dans : {output_dir}")
    print("  - comparaison_espaces_couleur.png")
    print("  - fonds_espaces_couleur.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
