function [t,x] = ode4(f, t, x0)
% 4th order Runge-Kutta solver with constant step length
% Inputs:
%   f  - function handle @(t,x) ...
%   t  - time vector
%   x0 - initial condition column vector

x = zeros(length(x0), length(t));
x(:,1) = x0;

h = t(2) - t(1);

for i = 1:length(t)-1
    k1 = f(t(i),           x(:,i));
    k2 = f(t(i) + h/2,     x(:,i) + h/2 * k1);
    k3 = f(t(i) + h/2,     x(:,i) + h/2 * k2);
    k4 = f(t(i) + h,       x(:,i) + h   * k3);
    x(:,i+1) = x(:,i) + h/6 * (k1 + 2*k2 + 2*k3 + k4);
end

x = x.';  % return as (time x state_dim), like your original
end
