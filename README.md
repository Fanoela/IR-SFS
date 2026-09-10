# IR-SFS
Iterative Replacement Sequential Feature Replacement (IR-SFS) is a novel feature selection algorithm as an extension to the classical SFS to overcome its greedy nature.

# IR-SFS — Iterative Replacement Sequential Forward Selection

Feature selection algorithm proposed in:

> Fanoela R. (2025) *Amélioration de l'algorithme de sélection séquentielle avant par remplacement itératif en apprentissage automatique : Application à l'estimation du redshift photométrique.*
> Master thesis, ESPA — Université d'Antananarivo.

---

## Motivation

Sequential Forward Selection (SFS) is a greedy algorithm: once a feature is added, it is never reconsidered. This can trap the search in a suboptimal subset when early choices interact poorly with later ones.

**IR-SFS** adds a post-SFS refinement phase: for each selected feature, it tries replacing it with every unselected feature and keeps the swap if it reduces validation error. Two stopping criteria prevent unnecessary computation:

1. **Max-no-improvement** — stop after *N* consecutive positions with no gain.
2. **Moving-average criterion** — stop when the average absolute gain over the last *k* steps falls below a threshold *δ*.

---

## Project structure

```
ir_sfs/
├── ir_sfs/
│   ├── __init__.py       # public API
│   ├── algorithm.py      # sfs(), ir_sfs(), fit()
│   ├── metrics.py        # RMSE, R², σ_NMAD, f_out
│   └── data.py           # loading, colour indices, normalisation
├── scripts/
│   └── run_experiment.py # reproduces the thesis experiment
├── tests/
│   └── test_algorithm.py # pytest unit tests
├── requirements.txt
└── README.md
```

---

## Experimentation

We used astronomical data about photometric redshift estimation using colours and magnitudes. Experimental data can be found in Curran et al. (2021) *QSO photometric redshifts using machine learning and neural networks* <https://doi.org/10.1093/mnras/stab485>
