import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from collections import defaultdict

# ── Load composite data (per participant × per photo) ─────────────────────────
AOI_COLS = [
    'Aquatic plants (%)', 'Artificial water features (%)', 'Buildings (%)',
    'Economic crops (%)', 'Flower beds / trellises (%)', 'Flowers / ornamental grass (%)',
    'Herbaceous plants (%)', 'Landscape bridges (%)',
    'Landscape seating / recreational facilities (%)', 'Landscape stones (%)',
    'Landscape walls (%)', 'Mountains (%)', 'Other elements (%)',
    'Parking lots / vehicles (%)', 'Paved surfaces / plazas (%)',
    'Pavilions / corridors (%)', 'Roads (%)', 'Sculptures / landmarks (%)',
    'Shrubs (%)', 'Sky (%)', 'Tall trees (%)', 'Water bodies (%)'
]

EXTRA_COLS = ['Total Dwell (ms)', 'AOIs Attended', 'Attention Entropy']

# Group by participant
participant_data = defaultdict(lambda: {'aoi': [], 'extra': []})

with open('perception_features_composite.csv', encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        name = row['Student Name']
        aoi_vals   = [float(row.get(c, 0) or 0) for c in AOI_COLS]
        extra_vals = [float(row.get(c, 0) or 0) for c in EXTRA_COLS]
        participant_data[name]['aoi'].append(aoi_vals)
        participant_data[name]['extra'].append(extra_vals)

# Only keep participants in clean file
clean_names = []
clean_genders = []
with open('student_information_clean.csv', encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        clean_names.append(row['English Name'])
        clean_genders.append(row['Gender'])

names   = [n for n in clean_names if n in participant_data]
genders = [clean_genders[clean_names.index(n)] for n in names]

# Build rich feature matrix:
# For each participant: mean + std of each AOI across all photos
# + mean of extra cols (dwell, entropy)
# This captures not just WHERE they look but HOW CONSISTENTLY they do so
feature_matrix = []
for name in names:
    aoi_arr   = np.array(participant_data[name]['aoi'])    # (n_photos, 22)
    extra_arr = np.array(participant_data[name]['extra'])  # (n_photos, 3)
    means = aoi_arr.mean(axis=0)   # 22 values
    stds  = aoi_arr.std(axis=0)    # 22 values — consistency across scenes
    extra_means = extra_arr.mean(axis=0)  # 3 values
    feature_matrix.append(np.concatenate([means, stds, extra_means]))

X = np.array(feature_matrix)

# Feature names
feat_names = (
    [f'{c.replace(" (%)","")}_mean' for c in AOI_COLS] +
    [f'{c.replace(" (%)","")}_std'  for c in AOI_COLS] +
    ['AvgDwell_ms', 'AvgAOIsAttended', 'AvgEntropy']
)

print(f'Feature matrix: {X.shape[0]} participants x {X.shape[1]} features')
print(f'(22 AOI means + 22 AOI stds + 3 extra = {X.shape[1]} total features)')

# ── Standardise ───────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── PCA ───────────────────────────────────────────────────────────────────────
pca_full = PCA()
pca_full.fit(X_scaled)
explained  = pca_full.explained_variance_ratio_ * 100
cumulative = np.cumsum(explained)

pca2 = PCA(n_components=2)
coords = pca2.fit_transform(X_scaled)

pca5 = PCA(n_components=5)
pca5.fit(X_scaled)

print(f'\nPCA results:')
print(f'  PC1: {explained[0]:.1f}%')
print(f'  PC2: {explained[1]:.1f}%')
print(f'  PC1+PC2: {cumulative[1]:.1f}%')
print(f'  Components for 80%: {next(i+1 for i,c in enumerate(cumulative) if c>=80)}')

# Save PCA summary
with open('ml_pca_summary.csv', 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.writer(f)
    w.writerow(['Component','Explained Variance (%)','Cumulative (%)'])
    for i,(e,c) in enumerate(zip(explained, cumulative), 1):
        w.writerow([f'PC{i}', round(e,2), round(c,2)])

# Save PCA loadings
with open('ml_pca_loadings.csv', 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.writer(f)
    w.writerow(['Feature'] + [f'PC{i+1}' for i in range(5)])
    for feat, load in zip(feat_names, pca5.components_.T):
        w.writerow([feat] + [round(v,4) for v in load])

# ── K-Means ───────────────────────────────────────────────────────────────────
inertias, silhouettes = [], []
k_range = range(2, 9)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))

best_k = k_range[np.argmax(silhouettes)]
print(f'\nK-Means best k={best_k}, silhouette={max(silhouettes):.3f}')

km_final  = KMeans(n_clusters=best_k, random_state=42, n_init=10)
km_labels = km_final.fit_predict(X_scaled)

# ── Hierarchical Clustering ───────────────────────────────────────────────────
hc        = AgglomerativeClustering(n_clusters=best_k, linkage='ward')
hc_labels = hc.fit_predict(X_scaled)

agree    = sum(1 for a,b in zip(km_labels, hc_labels) if a==b)
disagree = [names[i] for i in range(len(names)) if km_labels[i] != hc_labels[i]]
print(f'K-Means vs Hierarchical agreement: {agree}/{len(names)}')
if disagree:
    print(f'  Disagreed on: {disagree}')

# ── Save cluster assignments ──────────────────────────────────────────────────
with open('ml_cluster_assignments.csv', 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.writer(f)
    w.writerow(['Student Name','Gender','PC1','PC2','KMeans_Cluster','Hierarchical_Cluster'])
    for i,name in enumerate(names):
        w.writerow([name, genders[i], round(coords[i,0],4), round(coords[i,1],4),
                    km_labels[i]+1, hc_labels[i]+1])

# ── Cluster profiles (AOI means) ──────────────────────────────────────────────
aoi_means = X[:, :22]  # just the mean columns
with open('ml_cluster_profiles.csv', 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.writer(f)
    short = [c.replace(' (%)','') for c in AOI_COLS]
    w.writerow(['Cluster','Size'] + short + ['Avg Dwell (ms)','Avg Entropy'])
    for k in range(best_k):
        mask  = km_labels == k
        means = aoi_means[mask].mean(axis=0)
        dwell   = X[mask, 44].mean()
        entropy = X[mask, 46].mean()
        w.writerow([f'Cluster {k+1}', mask.sum()] +
                   [round(v,2) for v in means] +
                   [round(dwell,1), round(entropy,3)])

# ── Print cluster breakdown ───────────────────────────────────────────────────
for k in range(best_k):
    mask    = km_labels == k
    members = [names[i] for i in range(len(names)) if mask[i]]
    top_idx = aoi_means[mask].mean(axis=0).argmax()
    top_aoi = AOI_COLS[top_idx].replace(' (%)','')
    dwell   = X[mask, 44].mean()
    entropy = X[mask, 46].mean()
    print(f'\nCluster {k+1} ({mask.sum()} people)')
    print(f'  Top AOI: {top_aoi}')
    print(f'  Avg dwell: {dwell:.0f} ms | Avg entropy: {entropy:.3f}')
    print(f'  Members: {", ".join(members)}')

# ── Plots ─────────────────────────────────────────────────────────────────────
colors = ['#e74c3c','#3498db','#2ecc71','#f39c12','#9b59b6']

# Scree plot
fig, ax = plt.subplots(figsize=(8,4))
ax.bar(range(1,len(explained)+1), explained, color='steelblue', alpha=0.7)
ax.plot(range(1,len(explained)+1), cumulative, 'o-', color='red', label='Cumulative')
ax.axhline(80, color='gray', linestyle='--', label='80% threshold')
ax.set_xlabel('Principal Component')
ax.set_ylabel('Explained Variance (%)')
ax.set_title('PCA Scree Plot — Rich Feature Set (47 features)')
ax.legend()
plt.tight_layout()
plt.savefig('ml_pca_scree.png', dpi=150)
plt.close()

# PCA scatter
fig, ax = plt.subplots(figsize=(9,6))
for k in range(best_k):
    mask = km_labels == k
    ax.scatter(coords[mask,0], coords[mask,1], c=colors[k], label=f'Cluster {k+1}', s=80, alpha=0.8)
for i,name in enumerate(names):
    ax.annotate(name.split()[0], (coords[i,0], coords[i,1]), fontsize=6, alpha=0.6)
ax.set_xlabel(f'PC1 ({explained[0]:.1f}%)')
ax.set_ylabel(f'PC2 ({explained[1]:.1f}%)')
ax.set_title('PCA Scatter — Participants by Cluster (Rich Features)')
ax.legend()
plt.tight_layout()
plt.savefig('ml_pca_scatter.png', dpi=150)
plt.close()

# Silhouette
fig, ax = plt.subplots(figsize=(6,4))
ax.plot(list(k_range), silhouettes, 'o-', color='steelblue')
ax.axvline(best_k, color='red', linestyle='--', label=f'Best k={best_k}')
ax.set_xlabel('Number of Clusters (k)')
ax.set_ylabel('Silhouette Score')
ax.set_title('K-Means — Optimal Number of Clusters')
ax.legend()
plt.tight_layout()
plt.savefig('ml_kmeans_silhouette.png', dpi=150)
plt.close()

# Dendrogram
fig, ax = plt.subplots(figsize=(14,5))
Z = linkage(X_scaled, method='ward')
dendrogram(Z, labels=names, leaf_rotation=90, leaf_font_size=7, ax=ax)
ax.set_title('Hierarchical Clustering Dendrogram (Rich Features)')
ax.set_ylabel('Distance')
plt.tight_layout()
plt.savefig('ml_dendrogram.png', dpi=150)
plt.close()

print('\nAll files saved.')
