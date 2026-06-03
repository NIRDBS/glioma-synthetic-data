import h5py
import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

try:
    import hdbscan
except ImportError:
    hdbscan = None

class ClinicalAnnotator:
    """
    Spatio-Spectral Annotator for Raman Biopsies.
    Performs per-sample unsupervised clustering to separate Tumor from Healthy tissue.
    Supports GMM and HDBSCAN.
    """
    def __init__(self, data_path, algorithm='gmm', pca_variance=0.99, spatial_weight=2.0):
        self.data_path = data_path
        self.algorithm = algorithm.lower()
        self.pca_variance = pca_variance
        self.spatial_weight = spatial_weight

        if self.algorithm == 'hdbscan' and hdbscan is None:
            raise ImportError("HDBSCAN is not installed. Please run: pip install hdbscan")

    def _cluster_gmm(self, features):
        gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
        return gmm.fit_predict(features)

    def _cluster_hdbscan(self, features):
        # min_cluster_size is a hyperparameter you might need to tune based on image size
        clusterer = hdbscan.HDBSCAN(min_cluster_size=50, metric='euclidean')
        return clusterer.fit_predict(features)

    def annotate(self):
        print(f" Starting Spatio-Spectral Annotation (Algorithm: {self.algorithm.upper()})...")

        with h5py.File(self.data_path, 'r+') as f:
            X = f['spectra_smoothed'][:]
            if 'coords' not in f:
                raise KeyError(" 'coords' not found in HDF5. Did you run the spatial preprocessor?")
            coords = f['coords'][:]
            wn = f['wavenumbers'][:]
            
            # Safe string decoding for samples
            raw_samples = f['sample_ids'][:]
            if isinstance(raw_samples[0], (bytes, np.bytes_)):
                samples = np.char.decode(raw_samples.astype('S'), 'utf-8')
            else:
                samples = raw_samples.astype(str)

            y_tumor_global = np.zeros(X.shape[0], dtype='int32')
            idx_1004 = np.argmin(np.abs(wn - 1004)) # Phenylalanine peak index

            unique_samples = np.unique(samples)
            print(f" Found {len(unique_samples)} unique slides to process.")

            for sample in tqdm(unique_samples, desc="Clustering Slides"):
                mask = (samples == sample)
                X_slide = X[mask]
                c_slide = coords[mask]

                if len(X_slide) < 10:
                    continue # Skip tiny artifacts

                # 1. Spectral PCA
                pca = PCA(n_components=self.pca_variance, random_state=42)
                X_pca = pca.fit_transform(X_slide)

                # 2. Scale Features
                X_pca_scaled = StandardScaler().fit_transform(X_pca)
                c_scaled = StandardScaler().fit_transform(c_slide)

                # 3. Combine into Spatio-Spectral Space
                features = np.hstack([X_pca_scaled, c_scaled * self.spatial_weight])

                # 4. Cluster
                if self.algorithm == 'gmm':
                    labels = self._cluster_gmm(features)
                elif self.algorithm == 'hdbscan':
                    labels = self._cluster_hdbscan(features)
                else:
                    raise ValueError(f"Unknown algorithm: {self.algorithm}")

                # 5. Identify Tumor Cluster (Highest Phe Signal)
                y_tumor_slide = np.zeros_like(labels, dtype='int32')
                unique_clusters = np.unique(labels)
                
                # Exclude HDBSCAN noise (-1) from being considered as tumor
                valid_clusters = [c for c in unique_clusters if c != -1]
                
                if valid_clusters:
                    # Find the cluster with the highest mean Phenylalanine signal
                    mean_signals = {c: np.mean(X_slide[labels == c, idx_1004]) for c in valid_clusters}
                    tumor_cluster = max(mean_signals, key=mean_signals.get)
                    
                    # Set the winning cluster to 1 (Tumor), everything else remains 0 (Healthy)
                    y_tumor_slide[labels == tumor_cluster] = 1

                y_tumor_global[mask] = y_tumor_slide

            # 6. Save to HDF5
            print("\n Writing 'y_tumor' labels directly to HDF5...")
            if 'y_tumor' in f:
                del f['y_tumor']
                
            f.create_dataset("y_tumor", data=y_tumor_global, dtype='int32')
            
            n_tumor = np.sum(y_tumor_global == 1)
            n_total = len(y_tumor_global)
            print(f" Annotation Complete!")
            print(f" Final Dataset: {n_tumor} Tumor spectra ({n_tumor/n_total:.1%}) out of {n_total} total tissue spectra.")