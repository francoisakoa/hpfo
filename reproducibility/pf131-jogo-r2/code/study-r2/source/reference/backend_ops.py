#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
backend_ops.py - Module Backend Unifié GPU/CPU

Ce module fournit des wrappers unifiés pour:
- Conversions array (numpy <-> cupy)
- Clustering (KMeans, DBSCAN, GMM) via cuML ou sklearn
- Calcul de distances (pdist, cdist) via CuPy ou scipy

Configuration via backend_cfg dict:
    backend_cfg = {
        "use_gpu": True,           # Master switch
        "gpu_clustering": True,    # KMeans/GMM/DBSCAN via cuML
        "gpu_pdist": True,         # pdist via CuPy
        "gpu_dtype": "float32",    # dtype pour GPU (économie VRAM)
        "pdist_chunk_size": 2000,  # Seuil pour chunking pdist
        "verbose": False,          # Logs détaillés
    }
"""

import numpy as np
import warnings
from typing import Optional, Union, Literal, Tuple

# ==============================================================================
# Configuration par défaut
# ==============================================================================
DEFAULT_BACKEND_CFG = {
    "use_gpu": True,
    "gpu_clustering": True,
    "gpu_pdist": True,
    "gpu_dtype": "float32",
    "pdist_chunk_size": 2000,
    "verbose": False,
}

# ==============================================================================
# Détection et import GPU avec fallback automatique
# ==============================================================================

# Flags de disponibilité
_CUPY_AVAILABLE = False
_CUML_AVAILABLE = False
_CUPYX_AVAILABLE = False

# Modules (seront assignés si disponibles)
cp = None
cuml_kmeans = None
cuml_dbscan = None
cuml_gmm = None

try:
    import cupy as cp
    _CUPY_AVAILABLE = True
except ImportError:
    cp = None
    _CUPY_AVAILABLE = False

try:
    import cupyx
    import cupyx.scipy.spatial.distance as cupyx_distance
    _CUPYX_AVAILABLE = True
except ImportError:
    cupyx = None
    cupyx_distance = None
    _CUPYX_AVAILABLE = False

try:
    from cuml.cluster import KMeans as cuMLKMeans
    from cuml.cluster import DBSCAN as cuMLDBSCAN
    cuml_kmeans = cuMLKMeans
    cuml_dbscan = cuMLDBSCAN
    _CUML_AVAILABLE = True
except ImportError:
    cuml_kmeans = None
    cuml_dbscan = None
    _CUML_AVAILABLE = False

# NOTE: cuML n'a PAS de GaussianMixture - GMM reste toujours sur sklearn (CPU)
cuml_gmm = None

# Imports CPU (toujours disponibles)
from sklearn.cluster import KMeans as SKLearnKMeans
from sklearn.cluster import DBSCAN as SKLearnDBSCAN
from sklearn.mixture import GaussianMixture as SKLearnGMM
from scipy.spatial.distance import pdist as scipy_pdist
from scipy.spatial.distance import cdist as scipy_cdist
from scipy.spatial.distance import squareform as scipy_squareform


def get_backend_info() -> dict:
    """Retourne les informations sur les backends disponibles."""
    info = {
        "cupy_available": _CUPY_AVAILABLE,
        "cuml_available": _CUML_AVAILABLE,
        "cupyx_available": _CUPYX_AVAILABLE,
    }
    if _CUPY_AVAILABLE:
        try:
            info["cuda_version"] = cp.cuda.runtime.runtimeGetVersion()
            info["gpu_name"] = cp.cuda.Device().name
            info["gpu_memory_total"] = cp.cuda.Device().mem_info[1]
            info["gpu_memory_free"] = cp.cuda.Device().mem_info[0]
        except Exception as e:
            info["cuda_error"] = str(e)
    return info


def print_backend_status(cfg: dict = None):
    """Affiche le statut des backends."""
    info = get_backend_info()
    print("=" * 60)
    print("BACKEND STATUS")
    print("=" * 60)
    print(f"  CuPy disponible:  {info['cupy_available']}")
    print(f"  cuML disponible:  {info['cuml_available']}")
    print(f"  cupyx disponible: {info['cupyx_available']}")
    if info.get('gpu_name'):
        print(f"  GPU: {info['gpu_name']}")
        print(f"  VRAM: {info['gpu_memory_free'] / 1e9:.2f} GB libre / {info['gpu_memory_total'] / 1e9:.2f} GB total")
    if cfg:
        print("-" * 60)
        print("Configuration active:")
        for k, v in cfg.items():
            print(f"  {k}: {v}")
    print("=" * 60)


# ==============================================================================
# Conversions Array
# ==============================================================================

def to_xp(array: np.ndarray, cfg: dict = None) -> Union[np.ndarray, "cp.ndarray"]:
    """
    Convertit un array vers le backend actif (CuPy si GPU, NumPy sinon).
    
    Args:
        array: Array numpy à convertir
        cfg: Configuration backend (utilise DEFAULT si None)
        
    Returns:
        Array dans le backend approprié
    """
    if cfg is None:
        cfg = DEFAULT_BACKEND_CFG
        
    if not cfg.get("use_gpu", True) or not _CUPY_AVAILABLE:
        return np.asarray(array)
    
    dtype = cfg.get("gpu_dtype", "float32")
    return cp.asarray(array, dtype=dtype)


def to_np(array) -> np.ndarray:
    """
    Convertit un array vers NumPy (depuis CuPy si nécessaire).
    
    Args:
        array: Array (numpy ou cupy)
        
    Returns:
        Array numpy
    """
    if _CUPY_AVAILABLE and isinstance(array, cp.ndarray):
        return cp.asnumpy(array)
    return np.asarray(array)


def ensure_contiguous(array, cfg: dict = None):
    """Assure que l'array est contigu en mémoire."""
    if cfg is None:
        cfg = DEFAULT_BACKEND_CFG
        
    if _CUPY_AVAILABLE and cfg.get("use_gpu", True) and isinstance(array, cp.ndarray):
        return cp.ascontiguousarray(array)
    return np.ascontiguousarray(array)


# ==============================================================================
# Clustering Unifié
# ==============================================================================

def cluster_fit_predict(
    X: np.ndarray,
    method: Literal["kmeans", "gmm", "dbscan"] = "kmeans",
    n_clusters: Optional[int] = None,
    random_state: int = 42,
    cfg: dict = None,
    **kwargs
) -> np.ndarray:
    """
    Clustering unifié avec switch GPU/CPU automatique.
    
    Args:
        X: Données (n_samples, n_features)
        method: "kmeans", "gmm", ou "dbscan"
        n_clusters: Nombre de clusters (ignoré pour DBSCAN)
        random_state: Graine aléatoire
        cfg: Configuration backend
        **kwargs: Arguments supplémentaires pour le clusterer
        
    Returns:
        labels: np.ndarray de shape (n_samples,) avec les labels de cluster
    """
    if cfg is None:
        cfg = DEFAULT_BACKEND_CFG
        
    use_gpu = cfg.get("use_gpu", True) and cfg.get("gpu_clustering", True)
    verbose = cfg.get("verbose", False)
    
    n_samples = X.shape[0]
    
    # Auto-détermination du nombre de clusters si non spécifié
    if n_clusters is None and method != "dbscan":
        n_clusters = max(2, int(np.sqrt(n_samples)))
    
    method = method.lower()
    
    # ========== GMM (toujours CPU - cuML n'a pas de GMM) ==========
    if method == "gmm":
        # Filtrer les kwargs pour GMM (exclure les params DBSCAN)
        gmm_kwargs = {k: v for k, v in kwargs.items() if k not in ["eps", "min_samples"]}
        
        if verbose:
            print(f"[Backend] GMM clustering (CPU/sklearn) - {n_clusters} clusters, {n_samples} samples")
        gmm = SKLearnGMM(n_components=n_clusters, random_state=random_state, **gmm_kwargs)
        gmm.fit(X)
        labels = gmm.predict(X)
        return labels.astype(np.int32)
    
    # ========== KMeans ==========
    if method == "kmeans":
        # Filtrer les kwargs pour KMeans (exclure les params DBSCAN)
        kmeans_kwargs = {k: v for k, v in kwargs.items() if k not in ["eps", "min_samples", "max_iter"]}
        
        if use_gpu and _CUML_AVAILABLE:
            if verbose:
                print(f"[Backend] KMeans clustering (GPU/cuML) - {n_clusters} clusters, {n_samples} samples")
            try:
                X_gpu = to_xp(X, cfg)
                kmeans = cuml_kmeans(
                    n_clusters=n_clusters,
                    random_state=random_state,
                    max_iter=kwargs.get("max_iter", 300),
                    **kmeans_kwargs
                )
                labels = kmeans.fit_predict(X_gpu)
                return to_np(labels).astype(np.int32)
            except Exception as e:
                if verbose:
                    print(f"[Backend] KMeans GPU failed ({e}), falling back to CPU")
        
        # CPU fallback
        if verbose:
            print(f"[Backend] KMeans clustering (CPU) - {n_clusters} clusters, {n_samples} samples")
        kmeans = SKLearnKMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init="auto",
            **kmeans_kwargs
        )
        labels = kmeans.fit_predict(X)
        return labels.astype(np.int32)
    
    # ========== DBSCAN ==========
    if method == "dbscan":
        eps = kwargs.get("eps", 0.5)
        min_samples = kwargs.get("min_samples", 2)
        
        if use_gpu and _CUML_AVAILABLE:
            if verbose:
                print(f"[Backend] DBSCAN clustering (GPU/cuML) - eps={eps}, min_samples={min_samples}")
            try:
                X_gpu = to_xp(X, cfg)
                dbscan = cuml_dbscan(
                    eps=eps,
                    min_samples=min_samples
                )
                labels = dbscan.fit_predict(X_gpu)
                return to_np(labels).astype(np.int32)
            except Exception as e:
                if verbose:
                    print(f"[Backend] DBSCAN GPU failed ({e}), falling back to CPU")
        
        # CPU fallback
        if verbose:
            print(f"[Backend] DBSCAN clustering (CPU) - eps={eps}, min_samples={min_samples}")
        dbscan = SKLearnDBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X)
        return labels.astype(np.int32)
    
    raise ValueError(f"Unknown clustering method: {method}")


# ==============================================================================
# Distances / pdist Unifié
# ==============================================================================

def _pdist_gpu_chunked(X: "cp.ndarray", metric: str = "euclidean", chunk_size: int = 2000) -> "cp.ndarray":
    """
    Calcul pdist par chunks sur GPU pour éviter OOM.
    
    Retourne un vecteur condensé comme scipy.pdist.
    """
    n = X.shape[0]
    
    # Nombre de paires
    n_pairs = n * (n - 1) // 2
    
    # Si assez petit, calcul direct
    if n <= chunk_size:
        # Calcul de la matrice de distances complète
        if metric == "euclidean":
            # ||a - b||² = ||a||² + ||b||² - 2*a·b
            norms_sq = cp.sum(X ** 2, axis=1, keepdims=True)
            D_sq = norms_sq + norms_sq.T - 2 * cp.dot(X, X.T)
            D_sq = cp.maximum(D_sq, 0)  # Éviter les valeurs négatives dues aux erreurs numériques
            D = cp.sqrt(D_sq)
        else:
            raise NotImplementedError(f"Metric {metric} not implemented for GPU")
        
        # Extraire le triangle supérieur (format condensé)
        indices = cp.triu_indices(n, k=1)
        return D[indices]
    
    # Chunking pour grosses matrices
    result = cp.zeros(n_pairs, dtype=X.dtype)
    idx = 0
    
    for i in range(0, n, chunk_size):
        i_end = min(i + chunk_size, n)
        chunk_i = X[i:i_end]
        
        for j in range(i, n, chunk_size):
            j_end = min(j + chunk_size, n)
            chunk_j = X[j:j_end]
            
            # Distance entre les chunks
            if metric == "euclidean":
                norms_i = cp.sum(chunk_i ** 2, axis=1, keepdims=True)
                norms_j = cp.sum(chunk_j ** 2, axis=1, keepdims=True)
                D_chunk = cp.sqrt(cp.maximum(norms_i + norms_j.T - 2 * cp.dot(chunk_i, chunk_j.T), 0))
            
            # Extraire les paires valides (i < j uniquement)
            for ii in range(chunk_i.shape[0]):
                global_i = i + ii
                for jj in range(chunk_j.shape[0]):
                    global_j = j + jj
                    if global_i < global_j:
                        # Calculer l'index dans le vecteur condensé
                        pair_idx = n * global_i - (global_i * (global_i + 1)) // 2 + (global_j - global_i - 1)
                        if pair_idx < n_pairs:
                            result[pair_idx] = D_chunk[ii, jj]
    
    return result


def pdist_like(
    X: np.ndarray,
    metric: str = "euclidean",
    cfg: dict = None
) -> np.ndarray:
    """
    Calcul pdist unifié avec switch GPU/CPU.

    Args:
        X: Données (n_samples, n_features)
        metric: Métrique de distance ("euclidean" supportée)
        cfg: Configuration backend

    Returns:
        Vecteur condensé des distances pairwise (comme scipy.pdist)
    """
    if cfg is None:
        cfg = DEFAULT_BACKEND_CFG

    use_gpu = cfg.get("use_gpu", True) and cfg.get("gpu_pdist", True)
    chunk_size = cfg.get("pdist_chunk_size", 2000)
    verbose = cfg.get("verbose", False)

    X = np.asarray(X)
    n = X.shape[0]
    if n < 2:
        return np.array([], dtype=X.dtype)

    # GPU path
    if use_gpu and _CUPY_AVAILABLE:
        try:
            X_gpu = to_xp(X, cfg)

            # Chemin direct cupyx si dispo
            if _CUPYX_AVAILABLE and cupyx_distance is not None:
                if verbose:
                    print(f"[Backend] pdist (GPU/cupyx) - n={n}, metric={metric}")
                return to_np(cupyx_distance.pdist(X_gpu, metric=metric))

            # Fallback GPU : chunking euclidean
            if metric == "euclidean":
                if verbose:
                    print(f"[Backend] pdist (GPU/chunked) - n={n}, chunk_size={chunk_size}")
                return to_np(_pdist_gpu_chunked(X_gpu, metric=metric, chunk_size=chunk_size))

            raise NotImplementedError(f"Metric {metric} not implemented for GPU pdist")
        except Exception as e:
            if verbose:
                print(f"[Backend] pdist GPU failed ({e}), falling back to CPU")

    # CPU fallback
    if verbose:
        print(f"[Backend] pdist (CPU) - n={n}, metric={metric}")
    return scipy_pdist(X, metric=metric)


def squareform_like(X: np.ndarray, cfg: dict = None) -> np.ndarray:
    """
    Conversion squareform unifiée.
    
    Args:
        X: Vecteur condensé ou matrice carrée
        cfg: Configuration backend
        
    Returns:
        Matrice carrée ou vecteur condensé
    """
    # squareform est léger, pas besoin de GPU
    return scipy_squareform(X)


def compute_mean_pairwise_distance(X: np.ndarray, cfg: dict = None) -> float:
    """
    Calcule la distance pairwise moyenne (optimisé).
    
    Args:
        X: Données (n_samples, n_features)
        cfg: Configuration backend
        
    Returns:
        Distance moyenne entre toutes les paires
    """
    if cfg is None:
        cfg = DEFAULT_BACKEND_CFG
        
    n = X.shape[0]
    if n < 2:
        return float('inf')
    
    distances = pdist_like(X, metric="euclidean", cfg=cfg)
    return float(np.mean(distances))


def compute_distance_matrix(X: np.ndarray, cfg: dict = None) -> np.ndarray:
    """
    Calcule la matrice de distances complète.
    
    Args:
        X: Données (n_samples, n_features)
        cfg: Configuration backend
        
    Returns:
        Matrice de distances (n_samples, n_samples)
    """
    distances = pdist_like(X, metric="euclidean", cfg=cfg)
    return squareform_like(distances, cfg=cfg)


# ==============================================================================
# Fonctions Helper pour PF
# ==============================================================================

def compute_sharing_factors_backend(
    particles: np.ndarray,
    sigma_share: float,
    cfg: dict = None
) -> np.ndarray:
    """
    Calcule les facteurs de sharing pour la fitness sharing.
    Version unifiée GPU/CPU - OPTIMISÉE pour éviter matrice NxN sur CPU.
    
    Args:
        particles: Array (n_particles, n_features)
        sigma_share: Rayon de sharing
        cfg: Configuration backend
        
    Returns:
        sharing: Array (n_particles,) des facteurs de sharing
    """
    if cfg is None:
        cfg = DEFAULT_BACKEND_CFG
    
    arr = np.asarray(particles)
    n = len(arr)
    if n < 2:
        return np.zeros(n)
    
    use_gpu = cfg.get("use_gpu", True) and cfg.get("gpu_pdist", True)
    
    # === GPU PATH: calcul direct sans matérialiser sur CPU ===
    if use_gpu and _CUPY_AVAILABLE:
        try:
            X_gpu = to_xp(arr, cfg)
            
            # Calcul de la matrice de distances sur GPU
            norms_sq = cp.sum(X_gpu ** 2, axis=1, keepdims=True)
            D_sq = norms_sq + norms_sq.T - 2 * cp.dot(X_gpu, X_gpu.T)
            D = cp.sqrt(cp.maximum(D_sq, 0))
            
            # Calcul du sharing directement sur GPU
            sharing_matrix = cp.maximum(0, 1 - D / sigma_share)
            sharing = cp.sum(sharing_matrix, axis=1)
            
            # Retourner seulement le vecteur (n,) sur CPU
            return to_np(sharing)
        except Exception as e:
            if cfg.get("verbose", False):
                print(f"[Backend] sharing GPU failed ({e}), falling back to CPU")
    
    # === CPU FALLBACK ===
    # Utiliser pdist + squareform (moins optimal mais fonctionnel)
    D = compute_distance_matrix(arr, cfg={"use_gpu": False})
    sharing = np.maximum(0, 1 - D / sigma_share)
    return np.sum(sharing, axis=1)


def compute_diversity_backend(particles: np.ndarray, cfg: dict = None) -> float:
    """
    Calcule la diversité (distance pairwise moyenne).
    Version unifiée GPU/CPU.
    
    Args:
        particles: Array (n_particles, n_features)
        cfg: Configuration backend
        
    Returns:
        diversity: float, distance moyenne
    """
    return compute_mean_pairwise_distance(particles, cfg=cfg)


# ==============================================================================
# Fonctions pour la standardisation des labels
# ==============================================================================

def standardize_cluster_labels(
    labels: np.ndarray,
    particles: np.ndarray,
    objectives: np.ndarray
) -> np.ndarray:
    """
    Standardise les labels de cluster pour reproductibilité.
    
    Renuméro les clusters par ordre d'énergie moyenne croissante.
    Cela rend les décisions PF comparables entre runs GPU et CPU.
    
    Args:
        labels: Labels originaux
        particles: Particules correspondantes
        objectives: Valeurs objectives correspondantes
        
    Returns:
        labels_new: Labels renumérotés de manière stable
    """
    unique_labels = np.unique(labels)
    # Enlever le label -1 (bruit DBSCAN) si présent
    unique_labels = unique_labels[unique_labels >= 0]
    
    if len(unique_labels) <= 1:
        return labels.copy()
    
    # Calculer l'énergie moyenne par cluster
    cluster_energies = []
    for label in unique_labels:
        mask = labels == label
        mean_energy = np.mean(objectives[mask])
        cluster_energies.append((label, mean_energy))
    
    # Trier par énergie moyenne
    cluster_energies.sort(key=lambda x: x[1])
    
    # Créer le mapping old -> new
    label_mapping = {old_label: new_label for new_label, (old_label, _) in enumerate(cluster_energies)}
    
    # Appliquer le mapping
    labels_new = labels.copy()
    for old_label, new_label in label_mapping.items():
        labels_new[labels == old_label] = new_label
    
    # Garder -1 pour le bruit
    labels_new[labels == -1] = -1
    
    return labels_new.astype(np.int32)


# ==============================================================================
# Test du module
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TEST DU MODULE backend_ops.py")
    print("=" * 60)
    
    # Afficher le statut
    cfg = DEFAULT_BACKEND_CFG.copy()
    cfg["verbose"] = True
    print_backend_status(cfg)
    
    # Test avec données synthétiques
    np.random.seed(42)
    n_samples, n_features = 500, 30
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    
    print("\n--- Test Clustering ---")
    
    # Test KMeans
    labels_kmeans = cluster_fit_predict(X, method="kmeans", n_clusters=5, cfg=cfg)
    print(f"KMeans: {len(np.unique(labels_kmeans))} clusters, labels shape: {labels_kmeans.shape}")
    
    # Test DBSCAN
    labels_dbscan = cluster_fit_predict(X, method="dbscan", eps=1.5, min_samples=5, cfg=cfg)
    print(f"DBSCAN: {len(np.unique(labels_dbscan))} clusters, labels shape: {labels_dbscan.shape}")
    
    # Test GMM
    labels_gmm = cluster_fit_predict(X, method="gmm", n_clusters=5, cfg=cfg)
    print(f"GMM: {len(np.unique(labels_gmm))} clusters, labels shape: {labels_gmm.shape}")
    
    print("\n--- Test Distances ---")
    
    # Test pdist
    import time
    t0 = time.time()
    distances = pdist_like(X, cfg=cfg)
    t_gpu = time.time() - t0
    print(f"pdist: {len(distances)} paires, temps: {t_gpu:.4f}s")
    
    # Test diversité
    diversity = compute_diversity_backend(X, cfg=cfg)
    print(f"Diversité moyenne: {diversity:.4f}")
    
    # Test sharing factors
    sigma_share = 0.5 * np.linalg.norm(X.max(axis=0) - X.min(axis=0))
    sharing = compute_sharing_factors_backend(X, sigma_share, cfg=cfg)
    print(f"Sharing factors: mean={sharing.mean():.4f}, std={sharing.std():.4f}")
    
    # Comparaison CPU vs GPU
    if _CUPY_AVAILABLE:
        print("\n--- Comparaison CPU vs GPU ---")
        cfg_cpu = cfg.copy()
        cfg_cpu["use_gpu"] = False
        
        t0 = time.time()
        distances_cpu = pdist_like(X, cfg=cfg_cpu)
        t_cpu = time.time() - t0
        
        cfg_gpu = cfg.copy()
        cfg_gpu["use_gpu"] = True
        
        t0 = time.time()
        distances_gpu = pdist_like(X, cfg=cfg_gpu)
        t_gpu = time.time() - t0
        
        diff = np.abs(distances_cpu - distances_gpu).max()
        print(f"CPU: {t_cpu:.4f}s, GPU: {t_gpu:.4f}s, speedup: {t_cpu/t_gpu:.2f}x")
        print(f"Différence max CPU-GPU: {diff:.2e}")
    
    print("\n" + "=" * 60)
    print("TESTS TERMINÉS")
    print("=" * 60)
