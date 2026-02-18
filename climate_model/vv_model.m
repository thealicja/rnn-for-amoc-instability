function F = VV_model(t, u, P)
% Van Veen + Cessi style 5D model
% u = [x y z T S]

F = zeros(5,1);

F(1) = -u(2)^2 - u(3)^2 - P.a*u(1) + P.a*P.Fa;
F(2) =  u(1)*u(2) - P.b*u(1)*u(3) - u(2) + P.Ga;
F(3) =  P.b*u(1)*u(2) + u(1)*u(3) - u(3);
F(4) = -P.alp*(u(4) - 1 + P.gamma*u(1)) - u(4)*(1 + P.mu*(u(4) - u(5))^2);
F(5) = P.Fs*(1 + P.delf*t/P.tmax) + P.del*(u(3)^2 + u(2)^2)- u(5)*(1 + P.mu*(u(4) - u(5))^2);
end
