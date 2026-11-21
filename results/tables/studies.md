
### Minority-group recovery (clustering only, groups 20/10/10/m)

| Method | m | Minority recall | Minority purity | ARI | Conflict merge % |
|---|---|---|---|---|---|
| DisCo | 1 | 1.00 | 1.00 | 1.00 | 0.0 |
| DisCo | 2 | 1.00 | 1.00 | 1.00 | 0.0 |
| DisCo | 3 | 1.00 | 1.00 | 1.00 | 0.0 |
| DisCo | 5 | 1.00 | 1.00 | 1.00 | 0.0 |
| FL+HC | 1 | 0.00 | 0.00 | 0.64 | 10.4 |
| FL+HC | 2 | 0.33 | 0.67 | 0.73 | 8.5 |
| FL+HC | 3 | 0.89 | 1.00 | 0.81 | 9.1 |
| FL+HC | 5 | 0.67 | 0.67 | 0.71 | 8.6 |
| PACFL | 1 | 0.00 | 0.00 | 0.43 | 23.5 |
| PACFL | 2 | 0.00 | 0.00 | 0.43 | 23.0 |
| PACFL | 3 | 0.00 | 0.00 | 0.43 | 22.6 |
| PACFL | 5 | 0.13 | 0.17 | 0.46 | 20.7 |

### Explanation fidelity (divergent classes between clusters)

| Scenario | pairs | precision | recall | exact set | diagnosis correct |
|---|---|---|---|---|---|
| Mixed + QS | 18 | 1.00 | 0.78 | 0.33 | 0.33 |
| Rotation | 21 | 1.00 | 0.75 | 0.43 | 0.43 |
| Label swap | 18 | 1.00 | 1.00 | 1.00 | 1.00 |

### Assignment certificates (accountability)

| Scenario | clients | confident | correct among confident | correct among uncertain |
|---|---|---|---|---|
| Mixed + QS | 120 | 0.77 | 1.000 | 1.000 |
| Rotation | 120 | 0.76 | 0.923 | 0.690 |
| Label swap | 120 | 0.87 | 1.000 | 1.000 |

### Newcomers (assigned with the frozen reference model)

| Scenario | known-concept newcomers correctly assigned | novel-concept newcomers flagged |
|---|---|---|
| Rotation | 0.96 (24) | 1.00 (12) |
| Label swap | 0.96 (24) | 1.00 (12) |

### Scalability

| Clients | signature time / client (s) | server clustering time (s) | upload floats / client | model dim | ARI |
|---|---|---|---|---|---|
| 40 | 0.09 | 0.09 | 12237 | 46730 | 0.68 |
| 80 | 0.09 | 0.36 | 12237 | 46730 | 0.60 |
| 160 | 0.08 | 1.23 | 12134 | 46730 | 0.82 |
| 320 | 0.08 | 5.95 | 11821 | 46730 | 0.59 |

### Differential privacy (ARI, mean over 3 seeds)

| Scenario | samples/client | σ | ε (δ=1e-5) | DisCo | raw-cosine ablation |
|---|---|---|---|---|---|
| Rotation | 400 | 0.0 | ∞ | 0.93 | 0.89 |
| Rotation | 400 | 0.5 | 10.00 | 0.43 | 0.01 |
| Rotation | 400 | 1.0 | 4.38 | 0.19 | 0.00 |
| Rotation | 400 | 2.0 | 1.99 | 0.07 | 0.00 |
| Rotation | 400 | 4.0 | 0.93 | 0.01 | 0.00 |
| Rotation | 1500 | 0.0 | ∞ | 1.00 | 0.97 |
| Rotation | 1500 | 0.5 | 10.00 | 0.75 | 0.05 |
| Rotation | 1500 | 1.0 | 4.38 | 0.58 | 0.00 |
| Rotation | 1500 | 2.0 | 1.99 | 0.41 | 0.01 |
| Rotation | 1500 | 4.0 | 0.93 | 0.41 | 0.00 |
| Label swap | 400 | 0.0 | ∞ | 0.98 | 0.91 |
| Label swap | 400 | 0.5 | 10.00 | 0.43 | 0.00 |
| Label swap | 400 | 1.0 | 4.38 | 0.34 | 0.00 |
| Label swap | 400 | 2.0 | 1.99 | 0.10 | 0.00 |
| Label swap | 400 | 4.0 | 0.93 | 0.04 | 0.02 |
| Label swap | 1500 | 0.0 | ∞ | 0.99 | 0.99 |
| Label swap | 1500 | 0.5 | 10.00 | 0.76 | 0.03 |
| Label swap | 1500 | 1.0 | 4.38 | 0.65 | 0.01 |
| Label swap | 1500 | 2.0 | 1.99 | 0.55 | 0.00 |
| Label swap | 1500 | 4.0 | 0.93 | 0.30 | 0.00 |
