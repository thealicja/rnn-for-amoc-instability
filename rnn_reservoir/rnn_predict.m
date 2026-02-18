function predict = rnn_predict(x_warmup, tp, W_in, W_r, W_out, flag)
% flag = [n, dim, a, warmup_len, predict_cut, predict_len]

n            = flag(1);
dim          = flag(2);
a            = flag(3);
warmup_len   = flag(4);
predict_cut  = flag(5);
predict_len  = flag(6);

dim_tp = numel(tp);

r = zeros(n,1);
u = zeros(dim + dim_tp, 1);
u(dim+1:end) = tp;

% Warm-up
if warmup_len > 0
    x_warmup = x_warmup(1:warmup_len,:);
    for t_i = 1:(warmup_len-1)
        u(1:dim) = x_warmup(t_i,:);
        r = (1-a)*r + a*tanh(W_r*r + W_in*u);
    end
else
    x_warmup = zeros(1,dim);
end

% Predict
predict = zeros(predict_cut + predict_len, dim);
u(1:dim) = x_warmup(end,:);

for t_i = 1:(predict_cut + predict_len)
    r = (1-a)*r + a*tanh(W_r*r + W_in*u);
    r_out = r;
    r_out(2:2:end) = r_out(2:2:end).^2;
    predict(t_i,:) = W_out * r_out;
    u(1:dim) = predict(t_i,:);
end

predict = predict(predict_cut+1:end, :);
end
