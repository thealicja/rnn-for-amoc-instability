function R = rnn_params()
% Hyperparameters + lengths for the reservoir

R.n      = 900;
R.k      = 4;
R.eig_rho= 2.3;
R.W_in_a = 3.6;
R.a      = 0.30;
R.beta   = 3e-5;

R.W_in_type     = 1;   % 1: dense, 2/3: sparse layouts
R.res_net_type  = 1;   % 1: symmetric, 2: asymmetric
R.validation_type = 1; % 1: max RMSE, 2: success length, etc.

% Time/length settings (be consistent with generate_data.m)
R.reservoir_tstep        = 1;
R.train_r_step_cut       = 50*40;
R.train_r_step_length    = 80*40;
R.validate_r_step_length = 30*40;
R.success_threshold      = 0; % only used if validation_type == 2

% Target dimension (x,y,z,T,S)
R.dim = 5;
end
