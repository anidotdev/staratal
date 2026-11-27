# stratal

**Machine learning written from scratch, one layer on top of the last.**

stratal is a Python library where we write ML models from scratch. It starts with the models you know from scikit-learn and keeps going: trees and boosting first, then attention and embeddings. The code sits in five layers, and a layer may only import from the ones below it. The gradient step you write for logistic regression is the same one gradient boosting calls later.

It's early. Some models are missing, the API moves, and the tests are the only part we trust.

```python
import numpy as np

from stratal import LinearRegression, Pipeline, StandardScaler
from stratal.pipeline import train_test_split

rng = np.random.default_rng(0)
X = rng.normal(size=(200, 3))
y = 4.0 * X[:, 0] - 2.5 * X[:, 1] + 0.7 + rng.normal(scale=0.1, size=200)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = Pipeline([
    ("scale", StandardScaler()),
    ("reg", LinearRegression()),
])

model.fit(X_train, y_train)
model.predict(X_test)[:3]
model.score(X_test, y_test)  # R^2
```

## Why this exists

We wanted to know what `fit` actually does. Reading scikit-learn's source helps, but it isn't the same as writing the loop and watching the loss fall.

Most from-scratch ML repos we found are one file per algorithm. Each file grows its own gradient step, its own train/test split, and its own distance function. Nothing is shared, so nothing carries over when you move to the next model.

We wanted the pieces to stack. We borrow scikit-learn's API because it's the one people already know, and we check our output against it because it's usually right. The later layers go where scikit-learn stops.

## Philosophy

- NumPy is the only runtime dependency. scikit-learn, SciPy and PyTorch are allowed in tests and benchmarks, never inside `stratal/`. `tests/test_layers.py` walks the imports and fails if one gets in.
- Every model is tested against a reference implementation on real data. Real means a dataset that ships with scikit-learn or a CSV in `tests/fixtures/`, not noise we generated until the numbers matched. Tolerances are written down (`rtol=1e-8` unless the derivation says otherwise).
- Readable beats fast until a benchmark says otherwise. If a loop is clearer than the vectorized version, keep the loop and leave a note. When something is too slow, add a benchmark first, then optimize, then paste the before and after numbers in the PR.
- Every model ships with a short derivation in `docs/derivations/`. One page: the objective, the update rule, the assumptions. If we can't write it, we don't understand the model well enough to merge it.
- One model per PR. The reviewer has to explain the model back to the author before merging. If they can't, neither of us understands it yet.

## The layers

```
┌────────────────────────────────────────────────────────────────┐
│ 5  applications    end-to-end examples, notebooks, write-ups   │
├────────────────────────────────────────────────────────────────┤
│ 4  pipeline        preprocessing, metrics, CV, search          │
├────────────────────────────────────────────────────────────────┤
│ 3  models          linear, trees, neighbors, clustering        │
├────────────────────────────────────────────────────────────────┤
│ 2  blocks          losses, optimizers, kernels, initializers   │
├────────────────────────────────────────────────────────────────┤
│ 1  foundations     arrays, linalg, distances, autograd         │
└────────────────────────────────────────────────────────────────┘
```

A layer imports from the layers below it and from nothing above. `stratal/models/` can use `stratal/blocks/` and `stratal/foundations/`. `stratal/blocks/` cannot import a model. `import-linter` enforces this in CI, so a backwards import is a failed test rather than a code review argument.

## Roadmap

Rough order, no dates.

### Foundations and building blocks

- [ ] input checks (`check_X_y`, `check_array`) and one dtype policy (`float64` unless a model says otherwise)
- [ ] numerical helpers: `logsumexp`, stable softmax, one-hot, a seeded RNG wrapper
- [ ] distances: euclidean, manhattan, cosine, plus pairwise versions with a block-wise loop
- [ ] activations and derivatives: sigmoid, tanh, relu, softmax
- [ ] losses: MSE, MAE, binary and categorical cross-entropy, hinge
- [ ] gradient checking with finite differences, used by every model that has a gradient
- [ ] optimizers: full-batch GD, minibatch SGD, momentum, Adam, behind one `Optimizer` protocol
- [ ] kernels: linear, polynomial, RBF, plus a Gram matrix cache
- [ ] reverse-mode autodiff over NumPy arrays, small enough to read in one sitting
- [ ] a tiny MLP on top of the autodiff engine, as the first thing that uses the whole stack

### Preprocessing and evaluation

- [ ] `StandardScaler`, `MinMaxScaler`, `RobustScaler`
- [ ] `OneHotEncoder`, `OrdinalEncoder`, `LabelEncoder`
- [ ] `SimpleImputer` with mean, median and most-frequent strategies
- [ ] `train_test_split`, `KFold`, `StratifiedKFold`
- [ ] `cross_val_score`, `cross_validate`
- [ ] `GridSearchCV`, `RandomizedSearchCV`
- [ ] `learning_curve`, `validation_curve`
- [ ] classification metrics: accuracy, precision, recall, F1, confusion matrix, ROC AUC
- [ ] regression metrics: MSE, RMSE, MAE, R^2
- [ ] `Pipeline` and `ColumnTransformer`

### sklearn-parity models

- [ ] `LinearRegression` with the normal equation, and a GD path for comparison
- [ ] `Ridge`, `Lasso` (coordinate descent), `ElasticNet`
- [ ] `LogisticRegression` with IRLS and with SGD
- [ ] `KNeighborsClassifier`, `KNeighborsRegressor`
- [ ] `GaussianNB`, `MultinomialNB`
- [ ] `DecisionTreeClassifier`, `DecisionTreeRegressor` (CART, gini and entropy)
- [ ] `RandomForestClassifier`, `RandomForestRegressor`
- [ ] `GradientBoostingClassifier`, `GradientBoostingRegressor`
- [ ] `PCA`, `TruncatedSVD`
- [ ] `KMeans`, `DBSCAN`, `AgglomerativeClustering`
- [ ] `SVC` with SMO
- [ ] `GaussianMixture` with EM

### Beyond scikit-learn

- [ ] histogram-based gradient boosting (bin the features, scan the bins)
- [ ] leaf-wise tree growth
- [ ] `word2vec` with negative sampling
- [ ] one transformer block: attention, layer norm, feed-forward, built on our autodiff
- [ ] ALS matrix factorization for recommendations
- [ ] isolation forest
- [ ] t-SNE
- [ ] epsilon-greedy, UCB and Thompson sampling for bandits

### Vertical layers

- [ ] `applications/spam.py`: SMS spam from raw text to a scored pipeline
- [ ] `applications/housing.py`: regression with missing values and a `ColumnTransformer`
- [ ] `applications/movielens.py`: recommender on MovieLens 100k
- [ ] `applications/attention.py`: train the transformer block on a tiny character-level corpus
- [ ] a notebook per application that shows the intermediate steps
- [ ] `docs/derivations/` complete for every model that ships
- [ ] `docs/layers.md` explaining the import rules with a dependency graph

## Project layout

```
stratal/
├── stratal/
│   ├── foundations/
│   │   ├── checks.py
│   │   ├── linalg.py
│   │   ├── distances.py
│   │   ├── activations.py
│   │   └── autograd.py
│   ├── blocks/
│   │   ├── losses.py
│   │   ├── optimizers.py
│   │   └── kernels.py
│   ├── models/
│   │   ├── linear.py
│   │   ├── tree.py
│   │   ├── neighbors.py
│   │   ├── cluster.py
│   │   └── ensemble.py
│   ├── pipeline/
│   │   ├── preprocessing.py
│   │   ├── metrics.py
│   │   ├── selection.py
│   │   └── pipeline.py
│   └── applications/
│       ├── spam.py
│       └── movielens.py
├── tests/
│   ├── unit/
│   ├── reference/
│   └── test_layers.py
├── benchmarks/
│   └── bench_kmeans.py
├── docs/
│   ├── derivations/
│   │   ├── linear_regression.md
│   │   └── pca.md
│   └── layers.md
├── examples/
├── pyproject.toml
├── CONTRIBUTING.md
├── LICENSE
└── README.md
```

`tests/reference/` holds the comparisons against scikit-learn. `tests/unit/` holds everything that doesn't need a reference.

## Install for development

```bash
git clone https://github.com/stratal-ml/stratal.git
cd stratal
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pre-commit install
pytest
```

The dev extra brings in pytest, ruff, mypy, scikit-learn and SciPy. `pip install -e .` on its own installs NumPy and nothing else.

Reference tests are the slow ones. `pytest tests/unit` runs the fast set.

## How to add a model

1. Open an issue with the model, the reference you'll test against, and the dataset. Wait for a thumbs up so two people don't write the same thing.
2. Write `docs/derivations/<model>.md` first: objective, update rule, assumptions. If the derivation is hand-wavy, stop and read until it isn't.
3. Put the model in the right layer. Reusable maths goes down into `blocks/` or `foundations/`; the model file keeps only what's specific to it.
4. Add `tests/reference/test_<model>.py`. Fit both implementations on the same real dataset with the same seed and assert they match within a stated tolerance. Include one edge case: a single feature, duplicate rows, or a constant column.
5. Run `pytest` and `ruff check .`. If the model is slow, add a benchmark under `benchmarks/` and put the number in the PR description.
6. Open one PR for one model. In review we'll ask you to explain it back to us, and we do the same for our own PRs.

## Conventions

- The API follows scikit-learn. `fit` returns `self`, learned attributes end with `_`, `X` is `(n_samples, n_features)`, and `predict` doesn't modify its input.
- Randomness goes through `random_state=None` and `np.random.default_rng(random_state)`. No global `np.random` calls.
- Validate at the public boundary with `check_X_y`. Don't reshape silently inside a loop.
- Public functions get type hints. `ruff` and `mypy` run in CI, line length 100.
- One module per model family: `linear.py`, `tree.py`, `neighbors.py`. Class names match scikit-learn's.
- Tests: one unit test per branch, one reference test per model, no network access. Anything slow gets `@pytest.mark.slow`.
- Docstrings say what the function does, its parameters and its return value. The maths lives in `docs/derivations/`.
- No dependency beyond NumPy. If you think you need one, open an issue and make the case.

## Contributors

stratal is built by Animesh and [co-maintainer], two students who wanted to know how this stuff works. Issues and PRs are open to anyone. If you want one of the roadmap items, comment on the issue first so we don't duplicate work. Add yourself here after your first merged PR.

## License

MIT. See [LICENSE](LICENSE).
