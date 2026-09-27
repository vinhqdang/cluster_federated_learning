
### Minority-group recovery (clustering only, groups 20/10/10/m)

| Method | m | Minority recall | Minority purity | ARI | Conflict merge % |
|---|---|---|---|---|---|
| DisCo | 1 | 0.67 | 0.67 | 0.89 | 0.1 |
| DisCo | 2 | 1.00 | 1.00 | 0.81 | 0.0 |
| DisCo | 3 | 1.00 | 1.00 | 0.90 | 0.0 |
| DisCo | 5 | 1.00 | 1.00 | 0.90 | 0.0 |
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
| Mixed + QS | 31 | 0.96 | 0.85 | 0.26 | 0.35 |
| Rotation | 27 | 1.00 | 0.90 | 0.59 | 0.59 |
| Label swap | 18 | 1.00 | 1.00 | 1.00 | 1.00 |

### Assignment certificates (accountability)

| Scenario | clients | confident | correct among confident | correct among uncertain |
|---|---|---|---|---|
| Mixed + QS | 120 | 0.53 | 0.937 | 0.825 |
| Rotation | 120 | 0.66 | 0.949 | 0.854 |
| Label swap | 120 | 0.91 | 1.000 | 1.000 |

### Newcomers (assigned with the frozen reference model)

| Scenario | known-concept newcomers correctly assigned | novel-concept newcomers flagged |
|---|---|---|
| Rotation | 0.88 (24) | 1.00 (12) |
| Label swap | 0.96 (24) | 1.00 (12) |
