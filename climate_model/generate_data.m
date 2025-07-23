P = vv_params();
P.Fs = 0.97;              % e.g. loop over this
f = @(tt,uu) VV_model(tt, uu, P);

[t, x] = ode4(f, tspan, x0);
