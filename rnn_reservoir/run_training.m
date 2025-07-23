%% run_training.m
% End-to-end script:
% 1. Generate VV model data for several Fs values
% 2. Train ESN/Reservoir multiple times with different random seeds (best-of)
% 3. Save the best model and a sample plot

clear; clc; close all;

%% ------------------- Paths -------------------
addpath('climate_model');
addpath('rnn_reservoir');

if ~exist('results','dir'); mkdir results; end

%% ------------------- 1. Data generation -------------------
% Bifurcation / freshwater flux values used for training
Fs_list = [0.97 0.98 0.99];      % match tp_train_set

% VV model integration settings
tspan = [0 4000];                % [start end] in model time units
dt    = 1;                       % time step in data
x0    = [0.1; 0; 0; 1; 0.1];     % initial condition (5x1)
add_param_channel = true;        % append Fs as extra input channel

fprintf('Generating data for Fs_list = [%s] ...\n', num2str(Fs_list));
[train_data, t] = generate_data(Fs_list, tspan, dt, x0, add_param_channel);

[n_tp, n_time, data_dim] = size(train_data);
fprintf('Generated data: %d parameter trials, %d time steps, %d dims\n', n_tp, n_time, data_dim);

%% ------------------- 2. Reservoir parameters -------------------
R = rnn_params();   % your struct with hyperparams/lengths

% Sanity checks
needed_steps = R.train_r_step_length + R.validate_r_step_length + 10 + R.train_r_step_cut;
if n_time < needed_steps
    error('Not enough time steps (%d) for your train/validate lengths (%d). Increase tspan or reduce lengths.', ...
           n_time, needed_steps);
end
if R.reservoir_tstep ~= dt
    warning('reservoir_tstep (%g) != dt from generate_data (%g). Make sure this is intended.', R.reservoir_tstep, dt);
end

%% ------------------- 3. Best-of training loop -------------------
bo = 5;                          % number of random trials for W_in & W_r
best_perf = Inf;
best = struct();

fprintf('Training reservoir %d times (best-of)...\n', bo);
for trial = 1:bo
    rng(trial*12345,'twister');  % different random seeds

    [val_perf, W_in, W_r, W_out, t_val, x_real, x_pred] = rnn_train(train_data, Fs_list, R);

    fprintf('  Trial %d/%d: validation metric = %.6f\n', trial, bo, val_perf);

    if val_perf < best_perf
        best_perf = val_perf;
        best.W_in = W_in;
        best.W_r  = W_r;
        best.W_out= W_out;
        best.t_val= t_val;
        best.x_real = x_real;
        best.x_pred = x_pred;
        best.trial  = trial;
    end
end

fprintf('\nBest validation metric = %.6f (trial %d)\n', best_perf, best.trial);

%% ------------------- 4. Save results -------------------
save('results/best_reservoir.mat', 'best', 'R', 'Fs_list', 't', 'train_data', 'best_perf');
fprintf('Saved best model to results/best_reservoir.mat\n');

%% ------------------- 5. Quick visualization (optional) -------------------
% Plot one variable for one parameter value to eyeball the fit
tp_plot = 1;          % index into Fs_list
dim_plot = 1;         % 1..R.dim (x,y,z,T,S). Change to what you want to see.

figure;
plot(best.t_val, squeeze(best.x_real(tp_plot,:,dim_plot)), 'k', 'LineWidth', 1.5); hold on;
plot(best.t_val, squeeze(best.x_pred(tp_plot,:,dim_plot)), '--', 'LineWidth', 1.5);
xlabel('time'); ylabel(sprintf('State dim %d', dim_plot));
title(sprintf('Validation: Fs = %.3f', Fs_list(tp_plot)));
legend('Truth','Prediction');
box on; grid on;

fprintf('Done.\n');
