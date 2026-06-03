# MTech Final Thesis — Writing Instructions

## Overview

This folder contains two IEEE-format research papers that together form the basis of the MTech final thesis. The thesis must synthesize, expand, and narrate the work presented in both papers into a single cohesive academic document.

A **previous Sem-3 project report** (`SEM-3_M.Tech _Project_ Report_ 324103210 - Copy.docx`) also exists in this folder. It covers **Paper 1 only** (the visual hybrid deep learning detector) and provides ready-made chapters, tables, figures, and 19 verified references that can be used as a foundation and reference for the writing style, depth, and level of detail expected.

---

## Source Materials

| # | File | What it covers |
|---|------|----------------|
| 1 | `paper1.tex` | *Hybrid Deep Learning for Robust Deepfake Video Detection with Enhanced Generalization and Fairness* — ResNet50 + Bi-GRU + Multi-Head Attention, visual-only |
| 2 | `paper2.tex` | *Synchronicity-Aware Multimodal Deepfake Detection via Cross-Modal Attention and Bi-Directional Temporal Modeling* — dual-stream audio-visual framework, NEW for Sem-4 |
| 3 | `SEM-3_M.Tech _Project_ Report_ 324103210 - Copy.docx` | Sem-3 project report — full written chapters for Paper 1; use as a style and structure reference |

---

## What Already Exists in the Sem-3 Report (Use as Reference)

The Sem-3 report is a **complete draft** for Paper 1. The following chapters are already written and can be adapted/expanded for the final thesis:

| Chapter / Section | Status in Sem-3 Report | Action for Final Thesis |
|---|---|---|
| Abstract | Written (visual detection only) | Rewrite to cover both papers |
| Ch. 1 — Introduction | Complete (1.1–1.5) | Expand section 1.3 and 1.4 to include multimodal scope |
| Ch. 2 — Motivation | Written | Extend with multimodal motivation from Paper 2 |
| Ch. 3 — Literature Review | Complete (3.1–3.4) | Add subsections on audio deepfakes and audio-visual synchrony |
| Ch. 4 — Methodology | Complete (Paper 1 only) | Add new chapter for Paper 2 methodology |
| Ch. 5 — Results & Discussion | Complete (Paper 1 only) | Add Paper 2 results; add cross-paper comparison |
| Ch. 6 — Conclusion & Future Work | Written | Update to reflect both contributions |
| References | 19 IEEE references | Extend with ≥ 15 more references for Paper 2 topics |
| Appendix | Source code | Extend with Paper 2 code |

> Key tables already available in Sem-3 report: Table 3.1 (state-of-the-art comparison), Table 3.2 (generation techniques), Table 4.1 (architecture breakdown), Table 5.1–5.4 (performance, comparison, ablation, fairness).

---

## Existing References from Sem-3 Report (Do NOT Re-number; Continue from [19])

The following 19 IEEE-style references are already established. They must all appear in the final thesis. Add new references starting from [20].

```
[1]  A. Kaur, A. Hoshyar, V. Saikrishna, S. Firmin, and F. Xia, "Deepfake video detection: challenges 
     and opportunities," Artificial Intelligence Review, vol. 57, 2024, doi: 10.1007/s10462-024-10810-6.

[2]  A. Rössler, D. Cozzolino, L. Verdoliva, C. Riess, J. Thies, and M. Niessner, "FaceForensics++: 
     Learning to Detect Manipulated Facial Images," in Proc. IEEE/CVF ICCV, Seoul, 2019, pp. 1–11, 
     doi: 10.1109/ICCV.2019.00009.

[3]  H. H. Nguyen, J. Yamagishi, and I. Echizen, "Capsule-forensics: Using Capsule Networks to Detect 
     Forged Images and Videos," in Proc. ICASSP 2019, Brighton, UK, 2019, pp. 2307–2311, 
     doi: 10.1109/ICASSP.2019.8682602.

[4]  J. K. Lewis et al., "Deepfake Video Detection Based on Spatial, Spectral, and Temporal 
     Inconsistencies Using Multimodal Deep Learning," in Proc. IEEE AIPR 2020, Washington DC, 2020, 
     pp. 1–9, doi: 10.1109/AIPR50011.2020.9425167.

[5]  D. Afchar, V. Nozick, J. Yamagishi, and I. Echizen, "MesoNet: a Compact Facial Video Forgery 
     Detection Network," in Proc. IEEE WIFS, Hong Kong, 2018, pp. 1–7, 
     doi: 10.1109/WIFS.2018.8630761.

[6]  D. Güera and E. J. Delp, "Deepfake Video Detection Using Recurrent Neural Networks," in Proc. 
     IEEE AVSS 2018, Auckland, 2018, pp. 1–6, doi: 10.1109/AVSS.2018.8639163.

[7]  P. Zhou, X. Han, V. I. Morariu, and L. S. Davis, "Two-Stream Neural Networks for Tampered Face 
     Detection," in Proc. CVPRW 2017, Honolulu, 2017, pp. 1831–1839, 
     doi: 10.1109/CVPRW.2017.229.

[8]  A. Parikh, K. Pereira, P. Kumar, and K. Devadkar, "Audio-Visual Deepfake Detection System Using 
     Multimodal Deep Learning," in Proc. CONIT 2023, Hubli, India, 2023, pp. 1–6, 
     doi: 10.1109/CONIT59222.2023.10205804.

[9]  M. Huang, Z. Liang, P. Zhang, H. Li, D. Zhan, S. Wang, and M. Chan, "Spatiotemporal 
     Attention-Based Deepfake Detection," in Proc. CAIT 2024, pp. 47–52, 
     doi: 10.1109/CAIT64506.2024.10962871.

[10] X. Fu, Z. Yan, T. Yao, S. Chen, and X. Li, "Exploring Unbiased Deepfake Detection via 
     Token-Level Shuffling and Mixing," arXiv:2501.04376, 2025.

[11] R. Mubarak et al., "A Survey on the Detection and Impacts of Deepfakes in Visual, Audio, and 
     Textual Formats," in IEEE Access, vol. 11, pp. 144497–144529, 2023, 
     doi: 10.1109/ACCESS.2023.3344653.

[12] S. Antad et al., "A Hybrid approach for Deepfake Detection using CNN-RNN," in Proc. OTCON 2024, 
     Raigarh, India, 2024, pp. 1–6, doi: 10.1109/OTCON60325.2024.10687890.

[13] R. Dey and F. M. Salem, "Gate-variants of Gated Recurrent Unit (GRU) neural networks," in Proc. 
     IEEE MWSCAS 2017, Boston, 2017, pp. 1597–1600, doi: 10.1109/MWSCAS.2017.8053243.

[14] Y. Li, X. Yang, P. Sun, H. Qi, and S. Lyu, "Celeb-DF: A Large-Scale Challenging Dataset for 
     DeepFake Forensics," in Proc. CVPR 2020, Seattle, 2020, pp. 3204–3213, 
     doi: 10.1109/CVPR42600.2020.00327.

[15] L. Gong and X. Li, "A Contemporary Survey on Deepfake Detection: Datasets, Algorithms, and 
     Challenges," Electronics, vol. 13, no. 3, p. 585, 2024, doi: 10.3390/electronics13030585.

[16] Y. Xu, M. Pedersen, and K. Raja, "VoD: Learning Volume of Differences for Video-Based Deepfake 
     Detection," arXiv:2503.07607, 2025.

[17] M. S. Rana, B. Murali, and A. H. Sung, "Deepfake Detection Using Machine Learning Algorithms," 
     in Proc. IIAI-AAI 2021, Niigata, 2021, pp. 458–463, doi: 10.1109/IIAI-AAI53430.2021.00079.

[18] T. Dissanayake Mohottalalage, D. Saha, and M. Schmidt, "A Comprehensive Review of Deepfake 
     Detection Methods and Challenges in Digital Forensics," 2025, doi: 10.92298/2025732438.

[19] E. Prashnani, M. Goebel, and B. Manjunath, "Generalizable Deepfake Detection With Phase-Based 
     Motion Analysis," IEEE Trans. Image Process., pp. 1–1, 2024, doi: 10.1109/TIP.2024.3441821.
```

> Add new references for Paper 2 topics (audio-visual synchrony, multimodal fusion, FakeAVCeleb, AV-Deepfake1M++, cross-modal attention) starting at **[20]**.

---

## Formatting Requirements

| Property | Specification |
|----------|---------------|
| Layout | Single-column |
| Font | Times New Roman, 12 pt |
| Line spacing | 1.5 |
| Section start | Every major section begins on a new page |
| Minimum length | 25 pages (excluding references and appendices) |
| Reference style | IEEE citation style (numbered, e.g., [1]) |
| Language | Formal academic English; third-person voice |

---

## Quality Requirements

- **Plagiarism-free**: All content must be original. Do not copy-paste sentences from the source papers verbatim. Paraphrase, expand, and synthesize.
- **AI-detection-free**: Writing must read as authentic scholarly prose — well-argued, precise, and evidence-based. Avoid generic or padded sentences.
- **High academic quality**: Arguments must be logically structured, technically accurate, and supported by cited literature. Every claim needs evidence.

---

## Thesis Structure

Write the thesis in the following order. Each section starts on a new page.

### 1. Title Page
- Thesis title: *"A Multimodal Hybrid Framework for Robust Deepfake Detection: From Spatial-Temporal Analysis to Audio-Visual Synchronicity"*
- Author name: Paras Garg
- Supervisor: Prof. Mayank Dave
- Department of Computer Engineering, NIT Kurukshetra
- Degree: Master of Technology in Computer Engineering (Cyber Security)
- Year: 2026

### 2. Certificate / Declaration Page
- Standard NIT Kurukshetra declaration of original work
- Supervisor endorsement block

### 3. Acknowledgements
- Formal acknowledgements (supervisor, institution, funding if any, datasets used)

### 4. Abstract (≈ 400–500 words)
- Summarise both research contributions in a unified narrative
- State the problem (deepfake proliferation), methods (hybrid CNN-RNN-Attention for visual; dual-stream multimodal with cross-modal attention), and key results (accuracy, AUC-ROC figures from both papers)
- End with the significance of the work

### 5. List of Abbreviations
- Define all acronyms used in the thesis (CNN, RNN, GAN, Bi-GRU, AUC-ROC, DFDC, etc.)

### 6. List of Figures

### 7. List of Tables

### 8. Table of Contents

---

### Chapter 1 — Introduction (≥ 8 pages)
**Goal**: Motivate the thesis, frame both research problems, and state contributions.

Subsections to include:
1. **Background**: The rise of generative AI (GANs, VAEs, diffusion models) and synthetic media
2. **The Deepfake Threat**: Societal, political, and forensic implications
3. **Limitations of Existing Detection Methods**: Unimodal detectors, generalisation failures, demographic bias
4. **Research Gaps**: Visual-only deepfakes vs. audio-visual deepfakes; cross-dataset generalisation; fairness
5. **Thesis Objectives**: List the 5–6 concrete objectives that map to both papers
6. **Thesis Contributions**: Clearly numbered list of original contributions
7. **Thesis Organisation**: Brief paragraph describing each chapter

> Draw from both paper abstracts and introductions. Expand every point with explanations, context, and recent examples. Do not use bullet lists excessively — favour well-developed paragraphs.

---

### Chapter 2 — Literature Review (≥ 6 pages)
**Goal**: Survey the state of the art, identify research gaps, and position the proposed work.

Subsections:
1. **Generative Models for Synthetic Media**: GANs, VAEs, diffusion models
2. **Visual Deepfake Detection**: CNN-based, RNN-based, attention-based, transformer-based methods
3. **Audio Deepfake Detection**: Vocoder artifact detection, spectral methods
4. **Multimodal Deepfake Detection**: Fusion strategies, audio-visual synchrony
5. **Benchmark Datasets**: FaceForensics++, Celeb-DF, DFDC, FakeAVCeleb, AV-Deepfake1M++
6. **Fairness and Generalisation in Detection**: Cross-dataset performance, demographic analysis
7. **Research Gap Summary**: Tabulate gaps addressed by this thesis

> All claims must be supported by IEEE-style citations. Aim for ≥ 30 references total in the thesis.

---

### Chapter 3 — Methodology I: Hybrid Visual Deepfake Detection (≥ 5 pages)
**Goal**: Fully describe the architecture, dataset, and training from Paper 1.

Subsections:
1. **Problem Formulation**: Define the visual deepfake detection task formally
2. **Proposed Architecture**: ResNet50 (spatial feature extraction) → Bi-GRU (temporal modelling) → Multi-Head Attention (frame-level focus) → classification head
3. **Dataset Description**: FaceForensics++, Celeb-DF, DFDC — preprocessing, splits, augmentation
4. **Training Protocol**: Loss function, optimiser, hyperparameters, hardware setup
5. **Ablation Study Design**: Describe the component-wise ablation experiments

> Include architecture diagrams (TikZ figures from paper1.tex can be reused or redrawn). Use numbered equations for key mathematical formulations.

---

### Chapter 4 — Methodology II: Synchronicity-Aware Multimodal Deepfake Detection (≥ 5 pages)
**Goal**: Fully describe the architecture, dataset, and training from Paper 2.

Subsections:
1. **Problem Formulation**: Define the multimodal deepfake detection task; explain audio-visual synchrony as a forensic cue
2. **Proposed Architecture**: Dual-stream (ResNet50 visual encoder + convolutional audio encoder) → per-stream Bi-GRUs → 8-head cross-modal attention → fusion and classification
3. **Biological Motivation**: Viseme–phoneme temporal correlation as an invariant cue
4. **Dataset Description**: FakeAVCeleb, AV-Deepfake1M++ — 28,461 videos, four manipulation categories
5. **Training Protocol**: Multi-task objectives, hyperparameters, compression robustness testing
6. **Fairness Evaluation Design**: Demographic subgroup analysis methodology

---

### Chapter 5 — Experiments and Results (≥ 5 pages)
**Goal**: Report all experimental results for both papers clearly and comparatively.

Subsections:
1. **Evaluation Metrics**: Accuracy, AUC-ROC, F1-score, EER — formally define each
2. **Results of Paper 1 (Visual Detection)**: Tables comparing with baselines; ablation results
3. **Results of Paper 2 (Multimodal Detection)**: Tables comparing with unimodal and naive fusion baselines; compression robustness results; fairness results (92.5% accuracy, AUC 0.94)
4. **Cross-Paper Comparison**: How the multimodal system improves over the visual-only system
5. **Error Analysis**: Failure cases, challenging scenarios, limitations encountered

> Every table and figure must be numbered and have a descriptive caption.

---

### Chapter 6 — Discussion (≥ 2 pages)
**Goal**: Interpret results, relate back to research gaps, and critically evaluate the work.

Subsections:
1. **Interpretation of Results**: What the numbers mean in practice
2. **Addressing the Research Gaps**: Map each gap (Ch. 2) to how the proposed methods addressed it
3. **Wider Implications**: Deployment considerations, societal impact, ethical considerations
4. **Limitations**: Computational cost, dataset scope, adversarial robustness limitations

---

### Chapter 7 — Conclusion and Future Work (≥ 1.5 pages)
Subsections:
1. **Summary of Contributions**: Restate in plain language what was accomplished
2. **Future Work**: At least 4 concrete directions (e.g., transformer-based detectors, real-time deployment, adversarial training, cross-lingual audio deepfakes)

---

### References
- IEEE numbered citation style: `[1] Author, "Title," *Journal/Conf.*, vol., no., pp., year.`
- References [1]–[19] are already established — see the list above. **Do not change their numbering.**
- Add new references for Paper 2 starting at **[20]**. Minimum additional references needed: 15 (covering FakeAVCeleb, AV-Deepfake1M++, cross-modal attention, audio deepfake detection, viseme-phoneme synchrony, diffusion-based audio synthesis, etc.)
- Total thesis references: ≥ 34

### Appendices (optional)
- A: Detailed dataset statistics
- B: Full hyperparameter tables
- C: Additional ablation figures

---

## Writing Guidelines

1. **Each chapter must open with a short paragraph** (3–5 sentences) introducing what that chapter covers and how it connects to the rest of the thesis.
2. **Use transitional sentences** between subsections to guide the reader.
3. **Equations**: Number all equations. Use LaTeX `equation` environment.
4. **Figures**: Every architectural diagram, graph, and confusion matrix must be included with a caption.
5. **Tables**: Results tables must include baseline comparisons and must cite the baseline methods.
6. **Voice**: Write in third person (*"The proposed framework..."*, not *"We propose..."* unless using academic *"we"* consistently).
7. **Tense**: Past tense for specific experiments; present tense for general truths and descriptions of the system.
8. **Never pad**: Every sentence must add information. Do not repeat the same point in different words.

---

## Checklist Before Submission

- [ ] Minimum 25 pages of body content
- [ ] Single-column, Times New Roman 12 pt, 1.5 line spacing
- [ ] Every section starts on a new page
- [ ] All figures and tables are numbered and captioned
- [ ] All equations are numbered
- [ ] ≥ 30 IEEE-style references
- [ ] No plagiarised content
- [ ] Formal academic English throughout
- [ ] Title page, certificate, acknowledgements present
- [ ] Spelling and grammar checked
- [ ] Consistent notation across all chapters

---

*Both source papers are in this folder. Analyse them carefully before writing. The thesis must go beyond the papers — add deeper explanation, extended related work, and richer discussion.*