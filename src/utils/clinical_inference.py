import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion

# --- 1. LOAD CALIBRATION ---
# Load the frequency table (assuming no header, just Index, Shift)
calib_df = pd.read_csv('data/RamanFreqTable.xlsx - hf-448-v5b-1.csv', header=None, names=['Index', 'Shift'])

# Extract the Wavenumber vector
wavenumbers = calib_df['Shift'].values

# Handle length mismatch (CSV has 1667, Data has 1738)
# We pad the end with the extrapolated trend if needed, or just map the first 1667
if len(wavenumbers) < 1738:
    # Extrapolate the last few points to match 1738 dimension of the model
    diff = np.mean(np.diff(wavenumbers[-10:])) # Average step size at the end
    padding = [wavenumbers[-1] + diff * i for i in range(1, 1738 - len(wavenumbers) + 1)]
    wavenumbers = np.concatenate([wavenumbers, padding])

print(f"Calibration Loaded. X-axis shape: {wavenumbers.shape}")
print(f"Range: {wavenumbers[0]:.2f} cm⁻¹ to {wavenumbers[-1]:.2f} cm⁻¹")

# --- 2. FIND PHENYLALANINE INDEX ---
target_peak = 1004 # Phenylalanine
phe_idx = np.argmin(np.abs(wavenumbers - target_peak))

print(f"TARGET: Phenylalanine ({target_peak} cm⁻¹) is at Pixel Index: {phe_idx}")
print(f"        (Actual value at index: {wavenumbers[phe_idx]:.2f} cm⁻¹)")

# --- 3. THE CLINICAL PIPELINE (The "Blob" Logic) ---
def run_clinical_pipeline(raman_image, model, threshold_mult=1.5):
    """
    raman_image: (H, W, 1738)
    model: Your trained CVAE or Classifier
    """
    H, W, D = raman_image.shape
    
    # A. BIOLOGICAL FILTER
    # Extract the 1004 cm^-1 band
    phe_band = raman_image[:, :, phe_idx]
    
    # Define Local Baseline (e.g., average of 980-990 and 1010-1020)
    # We use indices +/- 10 pixels around the peak
    baseline_left = raman_image[:, :, phe_idx-15 : phe_idx-5]
    baseline_right = raman_image[:, :, phe_idx+5 : phe_idx+15]
    local_noise = np.mean(np.concatenate([baseline_left, baseline_right], axis=2), axis=2)
    
    # Mask: Signal must be significantly higher than local baseline
    # AND have absolute intensity (avoid dividing by zero in empty air)
    bio_mask = (phe_band > threshold_mult * local_noise) & (phe_band > 50)
    
    # B. MORPHOLOGICAL EROSION
    # "Eliminate the edges" -> Shrinks blobs to remove glass-tissue boundary
    clean_mask = binary_erosion(bio_mask, iterations=2)
    
    # C. EXTRACTION
    valid_spectra = raman_image[clean_mask]
    
    if len(valid_spectra) == 0:
        return {"diagnosis": "No Tissue", "confidence": 0.0}
    
    # D. INFERENCE (Majority Vote)
    # Normalize inputs (StandardScaler logic per sample)
    spectra_norm = (valid_spectra - np.mean(valid_spectra, axis=1, keepdims=True)) / \
                   (np.std(valid_spectra, axis=1, keepdims=True) + 1e-6)
    
    # Predict (Using CVAE Encoder Head)
    # Model output is likely [z_mean, z_log_var, z, class_pred]
    _, _, _, preds = model(spectra_norm)
    pred_classes = np.argmax(preds, axis=1)
    
    # Voting
    counts = np.bincount(pred_classes, minlength=6)
    final_class = np.argmax(counts)
    confidence = counts[final_class] / np.sum(counts)
    
    return {
        "diagnosis": final_class,
        "confidence": confidence,
        "n_pixels": len(valid_spectra),
        "mask": clean_mask,
        "bio_mask_raw": bio_mask
    }

# --- 4. VISUALIZE THE FILTER ---
# (Run this if you have a sample image 'img_cube')
# result = run_clinical_pipeline(img_cube, model)
# plt.imshow(result['mask'])
# plt.title("Tissue Mask (Phenylalanine > Noise)")
# plt.show()