function [val_perf, W_in, W_r, W_out, t_val, x_real, x_pred] = rnn_train(udata, tp_train_set, R)
%RNN_TRAIN Train a reservoir (Echo State Network) on multiple parameter trials.
%
% Inputs:
%   udata         - (tp_length, time_steps, dim + tp_dim) training+validation data
%   tp_train_set  - vector of parameter values (length = tp_length)
%   R             - struct from rnn_params.m
%
% Outputs:
%   val_perf  - validation performance (RMSE max/avg/product or success length)
%   W_in, W_r, W_out - trained ESN matrices
%   t_val     - time vector for validation
%   x_real    - ground truth validation data (tp_length x val_len x dim)
%   x_pred    - ESN predicted validation data (tp_length x val_len x dim)

% ---------- Unpack parameters ----------
n        = R.n;
k        = R.k;
eig_rho  = R.eig_rho;
W_in_a   = R.W_in_a;
a        = R.a;
beta     = R.beta;

train_length    = R.train_r_step_length;
validate_length = R.validate_r_step_length;
tstep           = R.reservoir_tstep;
dim             = R.dim;

W_in_type       = R.W_in_type;
res_net_type    = R.res_net_type;
validation_type = R.validation_type;

if validation_type == 2
    success_threshold = R.success_threshold;
else
    success_threshold = 0;
end

len_washout     = 10;                 % number of initial states to drop
tp_length       = numel(tp_train_set);
tp_dim          = size(udata,3) - dim;
validate_start  = train_length + 2;   % matches your original offset

% ---------- Define W_in ----------
switch W_in_type
    case 1
        W_in = W_in_a*(2*rand(n,dim+tp_dim)-1);

    case 2
        W_in = zeros(n,dim+tp_dim);
        n_win = n - mod(n,dim);
        idx = randperm(n_win); idx = reshape(idx, n_win/dim, dim);
        for d_i = 1:dim
            W_in(idx(:,d_i), d_i) = W_in_a*(2*rand(n_win/dim,1)-1);
        end
        W_in(:, dim+1:dim+tp_dim) = W_in_a*(2*rand(n,tp_dim)-1);

    case 3
        dim_sep = dim + tp_dim - 1;
        W_in = zeros(n, dim_sep+1);
        n_win = n - mod(n,dim_sep);
        idx = randperm(n_win); idx = reshape(idx, n_win/dim_sep, dim_sep);
        for d_i = 1:dim_sep
            W_in(idx(:,d_i), d_i) = W_in_a*(2*rand(n_win/dim_sep,1)-1);
        end
        W_in(:,end) = W_in_a*(2*rand(n,1)-1);

    otherwise
        error('W_in type error');
end

% ---------- Define W_r (reservoir network) ----------
switch res_net_type
    case 1  % symmetric, normal
        W_r = sprandsym(n, k/n);
    case 2  % asymmetric, uniform
        k = round(k);
        index1 = repmat((1:n)', k, 1);
        index2 = randperm(n*k)';
        index2(:,2) = repmat((1:n)', k, 1);
        index2      = sortrows(index2,1);
        index1(:,2) = index2(:,2);
        W_r = sparse(index1(:,1), index1(:,2), rand(size(index1,1),1), n, n);
    otherwise
        error('res_net type error');
end

% Rescale spectral radius
eig_D = eigs(W_r, 1);
W_r   = (eig_rho/abs(eig_D)) .* W_r;
W_r   = full(W_r);

% ---------- Training ----------
n_cols = tp_length * (train_length - len_washout);
r_reg  = zeros(n,   n_cols);
y_reg  = zeros(dim, n_cols);

r_end  = zeros(n, tp_length);

for tp_i = 1:tp_length
    train_x = udata(tp_i,1:train_length,:);      % size: [train_length, dim+tp_dim]
    train_y = udata(tp_i,2:train_length+1,:);    % next-step targets

    train_x = train_x.';  % (dim+tp_dim) x train_length
    train_y = train_y.';  % (dim+tp_dim) x train_length

    % Run reservoir for this trial
    r_all = zeros(n, train_length+1);
    % r_all(:,1) = 2*rand(n,1)-1;   % you can randomize initial state if desired

    for ti = 1:train_length
        r_all(:,ti+1) = (1-a)*r_all(:,ti) + a*tanh( W_r*r_all(:,ti) + W_in*train_x(:,ti) );
    end

    % Discard washout
    r_out = r_all(:, len_washout+2 : end);     % n x (train_length-len_washout)
    r_out(2:2:end, :) = r_out(2:2:end, :).^2;  % square even rows

    r_end(:,tp_i) = r_all(:, end);

    col_start = (tp_i-1)*(train_length-len_washout) + 1;
    col_end   = tp_i*(train_length-len_washout);
    r_reg(:, col_start:col_end) = r_out;
    y_reg(:, col_start:col_end) = train_y(1:dim, len_washout+1:end);  % no tp in target
end

% Ridge regression for readout
W_out = y_reg * r_reg' / (r_reg*r_reg' + beta*eye(n));

% ---------- Validation ----------
rmse_set            = zeros(1, tp_length);
success_length_set  = zeros(1, tp_length);
validate_predict_y_set = zeros(tp_length, validate_length, dim);
validate_real_y_set    = zeros(tp_length, validate_length, dim);

for tp_i = 1:tp_length
    validate_real_y_set(tp_i,:,:) = udata(tp_i, validate_start:(validate_start+validate_length-1), 1:dim);

    r = r_end(:, tp_i);
    u = zeros(dim+tp_dim, 1);
    u(1:dim) = udata(tp_i, train_length+1, 1:dim);

    for t_i = 1:validate_length
        u(dim+1:end) = udata(tp_i, train_length + t_i, dim+1:end);
        r = (1-a)*r + a*tanh( W_r*r + W_in*u );
        r_out = r;
        r_out(2:2:end) = r_out(2:2:end).^2;

        predict_y = W_out * r_out;
        validate_predict_y_set(tp_i,t_i,:) = predict_y;
        u(1:dim) = predict_y;  % closed-loop
    end

    % --- Error & metrics ---
    err  = squeeze(validate_predict_y_set(tp_i,:,:)) - squeeze(validate_real_y_set(tp_i,:,:)); % [val_len x dim]
    se_ts = sum(err.^2, 2);

    success_length_set(tp_i) = validate_length * tstep;
    if validation_type == 2
        for t_i = 1:validate_length
            if se_ts(t_i) > success_threshold
                success_length_set(tp_i) = t_i * tstep;
                break;
            end
        end
    end

    if any(isnan(validate_predict_y_set(tp_i,:,:)),'all')
        success_length_set(tp_i) = 0;
        se_ts = 10;    % big error if NaN
    end

    rmse_set(tp_i) = sqrt(mean(se_ts));
end

% ---------- Aggregate validation performance ----------
switch validation_type
    case 1
        val_perf = max(rmse_set);
    case 2
        val_perf = min(success_length_set);
    case 3
        rmse_set = max(rmse_set, 1e-3);
        val_perf = prod(rmse_set);
    case 4
        val_perf = mean(rmse_set);
    otherwise
        error('validation type error');
end

% ---------- Outputs ----------
t_val  = tstep:tstep:(tstep*validate_length);
x_pred = validate_predict_y_set;
x_real = validate_real_y_set;

end
