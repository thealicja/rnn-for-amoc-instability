function P = vv_params()
% Default parameters for the Van Veen + Cessi model

P.a     = 0.25;
P.b     = 4.0;
P.Fa    = 6;
P.Ga    = 1;

P.alp   = 300;
P.mu    = 6.25;
P.Fs    = 0.99;   % bifurcation / freshwater flux
P.del   = 0.01;
P.gamma = 0.1;

P.delf  = 0;
P.tmax  = 1000;
end
