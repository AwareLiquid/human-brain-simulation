# Human Brain Simulation

**Bio-inspired sparse spiking neural network** — extracting the statistical laws of the real *Drosophila* connectome and turning them into a trainable, event-driven SNN that matches dense MLP accuracy at **2 orders of magnitude lower energy**.

Part of the **AwareLiquid** bio-inspired AI research family (see [M1: MT-LNN](https://github.com/AwareLiquid/M1), [M2](https://github.com/AwareLiquid/M2)).

---

## Core result

Sparse spiking SNN (surrogate-gradient trained, long-tail + small-world + E/I masks derived from the real MaleCNS connectome) vs dense MLP:

| Task | Dense MLP | Sparse SNN | Gap | Energy saved |
|---|---|---|---|---|
| MNIST | 98.32% | 96.69% | 1.6 pts | **105×** |
| Fashion-MNIST | 87.56% | 87.17% | **0.4 pts** | **106×** |

**Takeaway**: brain-inspired sparsity + event-driven spikes deliver **near-lossless accuracy at ~2 orders of magnitude lower energy**, and the gap keeps shrinking with training.

---

## What this repo proves (and disproves)

A systematic investigation into whether the real connectome topology is a "naturally optimal sparse compute graph":

1. **Disproved**: MaleCNS topology, as a *static* structure, carries **no measurable advantage** over random / structured-sparse graphs (5 experiments: classification, temporal, robustness, sample-efficiency, plasticity — all indistinguishable).
2. **Extracted**: the topology's *statistical laws* that DO matter for architecture design:
   - sparsity **0.09%**, long-tail degree distribution (scale-free, max/mean ≈ 75)
   - strong **small-worldness** (clustering 6.65× random, path length 2.39)
   - **E/I ratio 60/40**
3. **Built**: a trainable sparse SNN from these laws → near-lossless accuracy at 105–106× energy savings.

---

## Project structure

```
pipeline.py            MaleCNS feather → signed sparse adjacency matrix
engine.py              LIF reservoir engine + 3 topology generators
bakeoff.py             Experiment 1: static classification bake-off
bakeoff_temporal.py    Experiment 2a: temporal (Poisson spike) bake-off
bakeoff_properties.py  Experiment A/B: robustness + sample efficiency
bakeoff_plasticity.py  Experiment C: Hebbian plasticity
graph_stats.py         Topology statistical laws (sparsity, small-world, E/I)
energy.py              Energy model (45nm CMOS, Horowitz 2014)
sparse_snn.py          Trainable sparse SNN (surrogate gradient)
recurrent_snn.py       Recurrent SNN (small-world recurrent connections)
capability.py          Capability ceiling (larger training)
full_train.py          Full-data MNIST training
fashion_snn.py         Fashion-MNIST (harder task) bake-off
EXPERIMENTS.md         Full experiment report (all result tables)
download_data.py       Download all datasets
```

---

## Quick start

```bash
# 1. environment
uv venv --python 3.12 .venv
uv pip install --python .venv numpy scipy pandas pyarrow torch

# 2. download data (~1.1 GB MaleCNS connectome + MNIST + Fashion-MNIST)
python download_data.py

# 3. build the connectome adjacency matrix
python pipeline.py

# 4. run the core bake-off (static classification)
python bakeoff.py

# 5. the headline result (trainable sparse SNN vs dense MLP)
python full_train.py          # MNIST, full data
python fashion_snn.py         # Fashion-MNIST, harder task
```

---

## Data

- **MaleCNS v1.0** — full male *Drosophila* CNS connectome (165,122 neurons, ~24.5M connections), CC-BY 4.0, HHMI Janelia / Cambridge / Google Research. Source: <https://male-cns.janelia.org/>
- MNIST, Fashion-MNIST — standard benchmarks.

## Method notes

- LIF neurons, hard reset, α-decay. Parameters follow Shiu et al. *Nature* 2024 (the canonical fly-brain LIF model).
- Surrogate-gradient training (straight-through estimator with sigmoid gradient) for the trainable SNN.
- Energy model: 45nm CMOS (Horowitz 2014) — MAC 3.7 pJ, addition 0.9 pJ. Sparse SNN does event-driven **additions only**; dense MLP does **MACs** per connection.

## License

MIT. MaleCNS data is CC-BY 4.0 (see source).
