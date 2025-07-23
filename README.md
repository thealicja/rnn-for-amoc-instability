# RNN for AMOC Instability 🌊

**Goal:** Use a lightweight **Reservoir Computing (Echo State Network)** to learn, detect, and forecast tipping behavior (transient chaos → collapse) in a simplified model of the **Atlantic Meridional Overturning Circulation (AMOC)**.

Workflow overview:

1. **Generate synthetic time series** from a 5‑dimensional climate box model (Van Veen–Cessi style).
2. **Train an ESN** on those trajectories at several freshwater‑flux (bifurcation) values.
3. **Pick the best reservoir** (best‑of strategy).
4. **Use the trained ESN** to predict at new parameter values faster than solving ODEs again.


## 🔎 Repository Structure

.
├── climate_model/
│   ├── VV_model.m          % 5D RHS of the coupled dynamical system
│   ├── vv_params.m         % Default physical parameters (a, b, Fs, etc.)
│   ├── ode4.m              % 4th-order Runge–Kutta solver (constant step)
│   └── generate_data.m     % Integrates VV_model for a list of Fs values and packs data
│
├── rnn_reservoir/
│   ├── rnn_params.m        % ESN hyperparameters (n, k, eig_rho, etc.) and lengths
│   ├── rnn_train.m         % One-shot ESN training + validation on multi-param data
│   ├── rnn_predict.m       % Forecasting with a trained ESN (warm-up + closed loop)
│   └── run_training.m      % End-to-end script: generate data → train ESN (best-of) → save results
│
├── results/                % (created at runtime) saved .mat with best model and predictions
└── data/                   % (optional) large .mat/.csv datasets if you export them


## 🚀 Quickstart (MATLAB)

Requires MATLAB (or Octave with sparse/eigs compatibility). Tested with R202x.

1. **Clone or download** this repo.
2. Open MATLAB in the repo root.
3. Run:
>> run_training


This will:

* Integrate the VV climate model for `Fs_list = [0.97 0.98 0.99]`
* Train the ESN `bo` times (default 5) with different random seeds
* Save the best model to `results/best_reservoir.mat`
* Plot one validation trace vs. prediction

---

### 🔁 Predict at a New Parameter (Example)

After training, you can load the model and predict at a new freshwater flux, e.g. `Fs = 1.01`:

load('results/best_reservoir.mat', 'best', 'R');

% Warm-up series: use one of the validation trajectories, for instance
warmup_series = squeeze(best.x_real(1, :, :));  % [time x dim]

% Parameter channel (dim_tp = 1 here)
new_Fs = 1.01;
tp_vec = new_Fs;

% Prediction settings
warmup_len  = 100;    % steps for warm-up
predict_cut = 0;      % drop initial transient (if desired)
predict_len = 500;    % steps to predict

flag_pred = [R.n, R.dim, R.a, warmup_len, predict_cut, predict_len];

y_hat = rnn_predict(warmup_series, tp_vec, best.W_in, best.W_r, best.W_out, flag_pred);

plot(y_hat(:,1)); title('Predicted state dim 1'); xlabel('step');


## 🧠 Why a Neural Reservoir if We Already Have the Equations?

* **Speed:** Once trained, the ESN predicts future states without integrating ODEs step-by-step.
* **Generalization:** A single ESN can emulate system dynamics across multiple parameter values.
* **Early warning:** Query the ESN as parameters drift to anticipate a tipping event.
* **Scalability:** Great for parameter sweeps, Monte Carlo runs, or real-time systems.

The ESN acts as a learned surrogate of the ODE system—capturing nonlinear transitions while being computationally cheap at run time.

## 🧩 Model Details

* **Dynamics:** 5D system `(x, y, z, T, S)`

  * `(x, y, z)` ≈ Lorenz-84-like atmospheric subsystem
  * `(T, S)` ≈ temperature/salinity boxes (Cessi-style)
  * Coupling + freshwater flux (`Fs`) drive chaotic and transient regimes
* **Integration:** `ode4.m` (fixed-step RK4). Swap in `ode45` if you prefer adaptive steps.
* **Data tensor shape:** `(n_params, n_time_steps, dim + param_dim)` where `param_dim = 1` (Fs channel).

---

## 📊 Validation Metrics

Set in `R.validation_type`:

1. **Max RMSE** over parameter trials
2. **Success length** (time until error exceeds a threshold)
3. **Product of RMSEs** (penalizes any single bad trial)
4. **Average RMSE**

---

## 🛠 Tech Stack

* **MATLAB** (core implementation)
* (Optional) **Python** ports/notebooks can be added for broader accessibility

---

## 🙋 Contributing / Using This

* Open an issue if you spot bugs or want to add features (e.g., Bayesian hyperparameter search, PyTorch port, visualization notebooks).
* Pull Requests are welcome—clear code and comments appreciated!
* This repo serves as a learning+collab space for ML + climate dynamics.

---

## 📄 License

Released under the **MIT License**. See `LICENSE` for details.

---

## 📬 Contact

Made with curiosity (and a bit of chaos) in mind.
If you’re into climate tipping points, reservoir computing, or dynamical systems, feel free to reach out or start a discussion!

```

::contentReference[oaicite:0]{index=0}
```
