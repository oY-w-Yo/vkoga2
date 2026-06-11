# vkoga2

VKOGA-style greedy kernel solver core v2.  Pure NumPy, no scikit-learn dependency.

## Install

Clone the repo then run from the project root:

```bash
pip install -e .
```

After installation, `from vkoga2 import VKOGARebuilt` works from anywhere.

For local development inside the umbrella `Codebase` workspace:

```bash
pip install -e /path/to/vkoga2
```

## Quick start

```python
import numpy as np
from vkoga2 import VKOGARebuilt, BetaGreedyPointSelectionRule, kernel_from_config

# training data
X = np.random.randn(200, 3)
y = np.sin(X[:, 0])

# solver
model = VKOGARebuilt(
    kernel=kernel_from_config("matern32", gamma=1.0),
    point_selection_rule=BetaGreedyPointSelectionRule(beta=1.0),
    max_centers=50,
)
model.fit(X, y)

# predict
pred = model.predict(X)
rmse = np.sqrt(np.mean((pred - y) ** 2))
print(f"RMSE: {rmse:.6e}, centers: {model.n_selected_}")
```

## Dependencies

- Python >= 3.10
- NumPy
