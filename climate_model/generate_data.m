function [train_data, t] = generate_data(Fs_list, tspan, dt, x0, add_param_channel)
%GENERATE_DATA Integrate VV model for a set of Fs values.
%
% Inputs:
%   Fs_list            - vector of bifurcation (freshwater flux) values, e.g. [0.97 0.98 0.99]
%   tspan              - [t0 tf]
%   dt                 - time step
%   x0                 - initial condition (5x1)
%   add_param_channel  - true/false, append Fs as extra column (default: true)
%
% Outputs:
%   train_data  - (n_Fs x n_time x (dim + param_dim)) array
%   t           - time vector

    if nargin < 5, add_param_channel = true; end

    P   = vv_params();
    t   = tspan(1):dt:tspan(2);
    dim = numel(x0);
    x0  = x0(:);                         % ensure column vector

    nF         = numel(Fs_list);
    param_dim  = double(add_param_channel);  % 1 if true, else 0
    train_data = zeros(nF, numel(t), dim + param_dim);

    for i = 1:nF
        P.Fs = Fs_list(i);
        f    = @(tt,uu) VV_model(tt, uu, P);
        [~, X] = ode4(f, t, x0);         % X is (n_time x dim)

        train_data(i, :, 1:dim) = X;     % fill state variables
        if add_param_channel
            train_data(i, :, dim+1) = P.Fs;   % parameter channel
        end
    end
end
