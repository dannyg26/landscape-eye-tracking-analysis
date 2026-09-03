# Where Do People Look?

Eye-tracking analysis of visual attention in landscape photography.

This study used Tobii eye-tracking recordings from 48 valid participants to examine two questions:

1. Is visual attention explained more by the landscape being viewed or by the person viewing it?
2. Are there distinct viewing styles among participants?

The analysis maps gaze samples to 22 landscape Areas of Interest (AOIs), summarizes dwell time and gaze counts, compares participant and site effects with ANOVA, and explores viewing patterns with PCA, K-Means, and hierarchical clustering.

## Key findings

- The landscape image was the dominant source of variation for 21 of 22 AOI categories.
- Sky was the exception: individual differences were stronger than site differences.
- Tall trees received 33.47% of all AOI dwell time, the largest share of any category.
- Two broad viewing styles appeared: a ground-level/natural-feature pattern and a vertical/tree-and-sky pattern.
- The two clusters were weakly separated (best silhouette score: 0.192), so they should be interpreted as tendencies rather than fixed viewer types.
- Attention was strongly nature-oriented overall.

## Results

### Overall visual attention

Tall trees were the strongest visual anchor, receiving approximately 14.49 million ms of dwell time and 33.47% of all attention assigned to an AOI. Buildings and herbaceous plants ranked second and third.

| Rank | AOI category | Total dwell time (ms) | Gaze samples | Share of AOI dwell time |
|---:|---|---:|---:|---:|
| 1 | Tall trees | 14,493,637.7 | 822,594 | 33.47% |
| 2 | Buildings | 5,290,279.5 | 301,420 | 12.22% |
| 3 | Herbaceous plants | 4,415,903.6 | 231,872 | 10.20% |
| 4 | Shrubs | 2,586,532.2 | 140,553 | 5.97% |
| 5 | Roads | 2,240,755.5 | 115,464 | 5.17% |
| 6 | Water bodies | 2,072,592.5 | 112,891 | 4.79% |

The complete ranking of all 22 AOIs is available in [`data/aggregate/attention_by_category.csv`](data/aggregate/attention_by_category.csv).

### Participant effect versus site effect

ANOVA showed that the image or site being viewed had a larger effect than the participant for 21 of the 22 individual AOI categories. Sky was the only exception.

| Example AOI | Participant F | Site F | Dominant factor |
|---|---:|---:|---|
| Economic crops | 0.172 | 346.912 | Site |
| Aquatic plants | 0.625 | 139.244 | Site |
| Buildings | 1.691 | 120.745 | Site |
| Tall trees | 8.330 | 40.067 | Site |
| Sky | 22.221 | 7.420 | Participant |

The six broader AOI groups produced the same general result: site dominated Vegetation, Built, Water, Natural, and Amenity, while participant differences dominated Sky. Full results are in [`results/anova`](results/anova).

### PCA and viewer clusters

PC1 explained 38.9% of participant variation and PC2 explained 14.2%, for 53.1% combined. K-Means testing from two through eight clusters selected two clusters with a silhouette score of 0.192. This low score indicates broad tendencies with considerable overlap, not sharply separated viewer types.

| Measure | Cluster 1 | Cluster 2 |
|---|---:|---:|
| Participants | 26 | 22 |
| Tall trees | 28.57% | 39.13% |
| Sky | 1.87% | 6.72% |
| Buildings | 10.15% | 14.49% |
| Herbaceous plants | 12.43% | 7.65% |
| Roads | 6.76% | 3.63% |
| Water bodies | 5.81% | 3.45% |
| Average dwell time | 19,032.5 ms | 18,549.3 ms |
| Average attention entropy | 2.081 | 2.007 |

Cluster 1 distributed relatively more attention across ground-level features such as herbaceous plants, roads, shrubs, and water. Cluster 2 concentrated more strongly on tall trees, sky, buildings, and mountains. Hierarchical clustering recovered a broadly similar division, supporting the existence of these two loose viewing styles.

## Design implications and final takeaway

The clearest practical implication is that tall trees and other prominent vertical greenery can serve as reliable visual anchors in urban landscapes. Tall trees captured approximately one-third of all AOI dwell time and attracted substantial attention across participants, while the site itself explained visual attention better than individual differences for nearly every landscape category.

For landscape architects and urban designers, the findings suggest:

- incorporating tall, visible trees into plazas, streetscapes, campuses, parks, and other public spaces where attracting attention to greenery is a design goal;
- preserving clear sightlines to tree canopies and vertical vegetation rather than obscuring them behind built elements;
- combining vertical greenery with ground-level planting, water, and other natural features to support different viewing tendencies; and
- treating the composition and placement of landscape elements as especially important, because the scene influenced attention more strongly than the identity of the viewer.

The final recommendation is not simply to add more trees everywhere, but to intentionally use tall trees and layered vegetation as focal elements within urban design. Because this study measured visual attention rather than preference, comfort, health, or long-term behavior, further research would be needed to determine whether greater attention also produces a more positive experience.

## Repository contents

```text
.
|-- data/
|   `-- aggregate/                 # Public, non-participant-level summaries
|-- docs/
|   `-- eye-tracking-poster.pptx   # Research poster
|-- results/
|   |-- anova/                     # Category and grouped ANOVA results
|   `-- machine-learning/          # Cluster profiles and PCA loadings
|-- src/
|   |-- analyse_attention.py       # Tobii parsing and AOI hit testing
|   |-- extract_student_data.py    # Participant metadata extraction utility
|   `-- ml_analysis.py             # PCA and clustering workflow
|-- CITATION.cff
|-- LICENSE
`-- requirements.txt
```

See [data/README.md](data/README.md) for a description of each published dataset and the fields intentionally withheld.

## Methodology

### 1. Gaze extraction and AOI matching

`analyse_attention.py` reads Tobii Studio binary gaze records, stimulus XML, and the project's SQLite AOI definitions. Valid left/right-eye coordinates are combined, transformed from normalized screen coordinates into image pixels, and tested against AOI polygons. Sample intervals are capped at 200 ms when estimating dwell time so recording gaps are not counted as continuous attention.

### 2. Feature construction

Each participant was represented by 47 features:

- mean attention percentage for each of 22 AOIs;
- standard deviation across images for each of 22 AOIs; and
- average dwell time, number of AOIs attended, and attention entropy.

Features were standardized before machine-learning analysis.

### 3. Statistical and machine-learning analysis

- One-way ANOVA compared the effects of participant and site for each AOI and for six grouped landscape categories.
- PCA summarized major dimensions of variation; PC1 explained 38.9% and PC2 explained 14.2% (53.1% combined).
- K-Means was evaluated for `k = 2` through `8` using silhouette score. The selected solution was `k = 2`.
- Ward-linkage hierarchical clustering provided an independent comparison.

## Reproducing the analysis

Create a virtual environment and install the Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

The full raw Tobii project and participant-level tables are not included because they contain restricted research data. Consequently, the repository documents the complete analytical code and publishes non-identifying aggregate outputs, but the participant-level pipeline cannot run from a fresh clone alone.

To rerun it with authorized data:

1. Place the Tobii `Data` directory where `src/analyse_attention.py` expects it, or update the path constants near the top of that script.
2. Run `python src/analyse_attention.py` to produce recording-, stimulus-, and category-level attention tables.
3. Construct the participant-level composite and cleaned metadata inputs described in [data/README.md](data/README.md).
4. Run `src/ml_analysis.py` from the directory containing `perception_features_composite.csv` and `student_information_clean.csv`.

`extract_student_data.py` is an archival extraction utility and contains paths from the original workstation. Update its input and output paths before use. Its output contains personally identifiable information and must not be committed.

## Data privacy and research ethics

This public version deliberately excludes names, participant IDs, gender-linked assignments, individual recording identifiers, the participant workbook, and row-level participant results. Do not upload those files without the appropriate participant consent, institutional approval, and de-identification review.

The included stimulus-level data aggregate observations across recordings. The repository contains no raw gaze coordinates.

## Limitations

- The sample contains 48 valid recordings and may not represent broader populations.
- The clustering separation is weak and is descriptive, not a diagnostic classification.
- Dwell estimates depend on Tobii validity flags, coordinate transformations, AOI definitions, and the 200 ms gap cap.
- Observational attention does not by itself establish preference, emotion, or design quality.
- Statistical results should be interpreted in the context of the specific image set and study protocol.

## Acknowledgments

Project by Danny Galvis, with faculty mentorship from Dr. Feng Qi. The work reported in the project poster was supported by the Kean University Scholars Academy and NSF Award #2247157, *Fostering Communities of Practice through Research and Peer Mentoring*.

## License

No open-source license has been granted yet. See [LICENSE](LICENSE). Add an appropriate license before inviting reuse or contributions.
