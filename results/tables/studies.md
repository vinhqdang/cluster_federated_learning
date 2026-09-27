
### Minority-group recovery (clustering only, groups 20/10/10/m)

| Method | m | Minority recall | Minority purity | ARI | Conflict merge % |
|---|---|---|---|---|---|
| DisCo | 1 | 1.00 | 1.00 | 0.98 | 0.0 |
| DisCo | 2 | 1.00 | 1.00 | 0.98 | 0.0 |
| DisCo | 3 | 1.00 | 1.00 | 1.00 | 0.0 |
| DisCo | 5 | 1.00 | 1.00 | 0.98 | 0.0 |
| FL+HC | 1 | 1.00 | 1.00 | 0.09 | 81.5 |
| FL+HC | 2 | 1.00 | 1.00 | 0.11 | 81.0 |
| FL+HC | 3 | 1.00 | 1.00 | 0.12 | 80.6 |
| FL+HC | 5 | 0.60 | 1.00 | 0.09 | 82.9 |
| PACFL | 1 | 0.00 | 0.00 | 0.43 | 23.5 |
| PACFL | 2 | 0.00 | 0.00 | 0.43 | 23.0 |
| PACFL | 3 | 0.00 | 0.00 | 0.43 | 22.6 |
| PACFL | 5 | 0.13 | 0.17 | 0.46 | 20.7 |

### Explanation fidelity (divergent classes between clusters)

| Scenario | pairs | precision | recall | exact set | diagnosis correct |
|---|---|---|---|---|---|
| Mixed + QS | 18 | 1.00 | 0.92 | 0.39 | 0.39 |
| Rotation | 18 | 1.00 | 0.93 | 0.67 | 0.67 |
| Label swap | 18 | 1.00 | 1.00 | 1.00 | 1.00 |

### Assignment certificates (accountability)

| Scenario | clients | confident | correct among confident | correct among uncertain |
|---|---|---|---|---|
| Mixed + QS | 120 | 0.93 | 0.919 | 0.778 |
| Rotation | 120 | 0.95 | 1.000 | 1.000 |
| Label swap | 120 | 0.97 | 1.000 | 1.000 |

### Newcomers (assigned with the frozen reference model)

| Scenario | known-concept newcomers correctly assigned | novel-concept newcomers flagged |
|---|---|---|
| Rotation | 1.00 (24) | 1.00 (12) |
| Label swap | 0.96 (24) | 1.00 (12) |

### Scalability

| Clients | signature time / client (s) | server clustering time (s) | upload floats / client | model dim | ARI |
|---|---|---|---|---|---|
| 40 | 0.16 | 0.05 | 12237 | 46730 | 1.00 |
| 80 | 0.14 | 0.07 | 12237 | 46730 | 1.00 |
| 160 | 0.14 | 0.29 | 12134 | 46730 | 0.98 |
| 320 | 0.14 | 2.28 | 11821 | 46730 | 0.98 |

### Differential privacy (ARI, mean over 3 seeds)

| Scenario | samples/client | σ | ε (δ=1e-5) | DisCo | raw-cosine ablation |
|---|---|---|---|---|---|
| Rotation | 400 | 0.0 | ∞ | 1.00 | 0.87 |
| Rotation | 400 | 0.5 | 10.00 | 0.64 | 0.01 |
| Rotation | 400 | 1.0 | 4.38 | 0.32 | 0.01 |
| Rotation | 400 | 2.0 | 1.99 | 0.09 | 0.00 |
| Rotation | 400 | 4.0 | 0.93 | 0.00 | 0.02 |
| Rotation | 1500 | 0.0 | ∞ | 0.99 | 0.81 |
| Rotation | 1500 | 0.5 | 10.00 | 0.89 | 0.02 |
| Rotation | 1500 | 1.0 | 4.38 | 0.92 | 0.01 |
| Rotation | 1500 | 2.0 | 1.99 | 0.68 | 0.02 |
| Rotation | 1500 | 4.0 | 0.93 | 0.30 | 0.00 |
| Label swap | 400 | 0.0 | ∞ | 1.00 | 0.87 |
| Label swap | 400 | 0.5 | 10.00 | 0.72 | 0.02 |
| Label swap | 400 | 1.0 | 4.38 | 0.37 | 0.01 |
| Label swap | 400 | 2.0 | 1.99 | 0.14 | 0.00 |
| Label swap | 400 | 4.0 | 0.93 | 0.01 | 0.01 |
| Label swap | 1500 | 0.0 | ∞ | 1.00 | 0.88 |
| Label swap | 1500 | 0.5 | 10.00 | 0.84 | 0.03 |
| Label swap | 1500 | 1.0 | 4.38 | 0.77 | 0.01 |
| Label swap | 1500 | 2.0 | 1.99 | 0.60 | 0.01 |
| Label swap | 1500 | 4.0 | 0.93 | 0.33 | 0.01 |
