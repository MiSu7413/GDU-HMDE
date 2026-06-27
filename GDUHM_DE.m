% =========================================================================
%  GDU-HMDE: Differential Evolution with Gated Dimension Update and Hierarchical Memory
%  Version:1.8
% -------------------------------------------------------------------------
%  INPUTS
%  functionnum : Index of the benchmark function
%  fhd         : Function handle of the objective function
%  dim         : Problem dimension
%  xmin        : Lower bound of decision variables
%  xmax        : Upper bound of decision variables
% -------------------------------------------------------------------------
%  OUTPUTS
%  bestx       : Best solution found
%  bestf       : Best objective function value
%  hist        : Convergence history (best fitness over function evaluations)
% -------------------------------------------------------------------------
%  NOTES
%  max_nfe = 10000 * dim function evaluations
% -------------------------------------------------------------------------
%  If you use this code, please cite the following paper:
%
%  "Differential evolution with gated dimension update and hierarchical
%  memory for numerical optimization."
%
% -------------------------------------------------------------------------
% =========================================================================
function [bestx,bestf,hist] = GDUHM_DE(functionnum,fhd,dim,xmin,xmax)
hist=[];
max_nfe=10000*dim;
p=0.11;
pminN = 2;
eta = 0.1;
Archfactor = 1.6;
memory_size = 6;
memory_pos = 1;
base_update_num = -1;
ps_ini = round(25*log(dim)*sqrt(dim));
ps_min = 4;
ps = ps_ini;

X = xmin + (xmax - xmin) * rand(ps, dim);
FX = feval(fhd, X', functionnum)';
nfe = ps;

[bestf, gbestid] = min(FX);
bestx = X(gbestid, :);
hist = append_hist(hist, nfe, bestf, max_nfe);

memory_F=0.5*ones(memory_size,1);
X_Arc = [];
Arc_size = 0;

X_H  = repmat(X, 1, 1, 3);
FX_H = repmat(FX, 1, 3);
FX_H(:,3) = inf(ps,1);
level_indicator=3*ones(ps,1);
stagnation_count=zeros(ps,1);
max_stagnation_count=floor(log(max_nfe)^(3/2)/sqrt(log(ps_ini)));
label=zeros(ps,dim);
for i = 1:ps
    j = mod(i-1, dim) + 1;
    idx = randperm(dim, j);
    label(i, idx) = 1;
end
d_vector = (1:dim)';
one_vector=ones(dim,1);

while nfe < max_nfe
    [~, order] = sort(FX);
    pbest_num = max(round(ps*p),pminN);
    pbest_idx_relative = max(ceil(pbest_num*rand(1,ps)),1);
    pbest_idx_true=order(pbest_idx_relative);
    pbestX = X(pbest_idx_true,:);
    if base_update_num ~= -1
        real_update_num = round(base_update_num + randn(ps, 1) * dim/10);
        real_update_num = min(max(real_update_num,1),dim);
        label = zeros(ps, dim);
        for ii = 1:ps
            k = real_update_num(ii);
            idx = randperm(dim, k);
            label(ii, idx) = 1;
        end
    end
    memory_rand_idx = ceil(rand(1, ps) * memory_size);
    mu_F = memory_F(memory_rand_idx);
    F = randCauchy(mu_F, 0.01);
    while any(F <= 0)
        idx = (F <= 0);
        F(idx) = randCauchy(mu_F(idx), 0.01);
    end
    F = min(1, F);
    FMat=repmat(F,1,dim);
    X_all = [X; X_Arc];
    ps_all = ps + Arc_size;
    shuffle_idx = ceil(rand(ps,1)*ps);
    shuffle_idx_all = ceil(rand(ps,1)*ps_all);
    for i = 1:ps
        while shuffle_idx(i)==i
            shuffle_idx(i) = ceil(rand()*ps);
        end
        while shuffle_idx_all(i)==shuffle_idx(i) || shuffle_idx_all(i)==i
            shuffle_idx_all(i) = ceil(rand()*ps_all);
        end
    end
    Xr1 = X(shuffle_idx, :);
    Xr2 = X_all(shuffle_idx_all, :);


    Xnew = (X + FMat.*(pbestX - X) + FMat.*(Xr1 - Xr2)) .* label+ X .* (1-label);

    Xnew = ((Xnew>=xmin)&(Xnew<=xmax)).*Xnew...
        +(Xnew<xmin).*((xmin + X).*rand(ps,dim)/2) ...
        +(Xnew>xmax).*((xmax + X).*rand(ps,dim)/2);

    FXnew = feval(fhd, Xnew', functionnum)';
    nfe = nfe + ps;
    hist_interval=ps;

    Xdiff = (Xnew-X);
    FXdiff = FX - FXnew;

    success = (FX>FXnew);
    successF = F(success);
    one_vector_s=ones(sum(success),1);
    one_vector_f=ones(sum(~success),1);

    Xdiff = Xdiff(success,:);
    Xdiff = std(Xdiff,0,2);
    w_x = Xdiff/sum(Xdiff);

    FXdiff = FXdiff(success);
    w_f = FXdiff/sum(FXdiff);
    w=w_x.*w_f;

    if ~isempty(successF)
        memory_F(memory_pos) = w' * (successF.^2) / (w' * successF);
        memory_pos = mod(memory_pos, memory_size) + 1;
        l_s = sum(label(success, :), 2);
        l_f = sum(label(~success, :), 2);
        indicator_s=(l_s*one_vector'==one_vector_s*d_vector')';
        indicator_f=(l_f*one_vector'==one_vector_f*d_vector')';
        g_s=indicator_s*one_vector_s;
        g_f=indicator_f*one_vector_f;
        d_f  = indicator_s*FXdiff;
        d_x  = indicator_s*w_x;
        g_t = g_s+g_f;
        valid = g_s > 0 & g_t > 0;
        g_t(g_t==0)=1;
        r_s=valid.*g_s./g_t;
        r_f=valid.*g_f./g_t;
        w_g=valid.*(d_f.*d_x.*(1-(r_f-r_s)));
        w_g=w_g/sum(w_g);
        base_update_num_hat=(w_g'*(d_vector.*d_vector))/(w_g'*d_vector);
        if base_update_num ~= -1
            base_update_num = eta * base_update_num_hat + (1-eta)* base_update_num;
        else
            base_update_num = base_update_num_hat;
        end
        base_update_num = max(1, min(dim, base_update_num));
    end

    X_Arc= [X_Arc;X(success,:)];
    Arc_size=size(X_Arc,1);
    if Arc_size>round(Archfactor*ps)
        shuffle_A_idx = randperm(Arc_size)';
        shuffle_A_idx = shuffle_A_idx(round(Archfactor*ps)+1:Arc_size);
        X_Arc(shuffle_A_idx,:) = [];
        Arc_size=size(X_Arc,1);
    end

    for h = 1:3
        switch h
            case 1
                mask = FXnew < FX_H(:,1);
            case 2
                mask = (FXnew < FX_H(:,2)) & (FXnew >= FX);
            case 3
                mask = (FXnew < FX_H(:,3)) & (FXnew >= FX_H(:,2));
        end
        FX_H(mask,h) = FXnew(mask);
        X_H(mask,:,h) = Xnew(mask,:);
    end

    X(success, :) = Xnew(success, :);
    FX(success) = FXnew(success);

    stagnation_count = stagnation_count + 1;
    stagnation_count(success) = 0;
    level_indicator(success) = 3;
    is_stagnation = stagnation_count > max_stagnation_count;
    masks = (is_stagnation & (level_indicator == (1:3)))';
    for level=1:3
        mask=masks(level,:);
        mask2=mask;
        if any(mask)
            if level==1
                num=sum(mask);
                shuffle_idx = randperm(ps);
                mask2 = shuffle_idx(1:num);
            end
            X(mask, :) = X_H(mask2, :, level);
            FX(mask) = FX_H(mask2, level);
            stagnation_count(mask) = 0;
            level_indicator(mask) = mod(level_indicator(mask)+1,3)+1;
        end
    end

    ps = max(round((ps_min - ps_ini)/(max_nfe) * nfe + ps_ini), 4);
    [old_size, ~] = size(X);
    reduction_num = old_size - ps;
    [~, order] = sort(FX_H(1:old_size,1));
    delete_idx = order(end-reduction_num+1:end);
    X(delete_idx,:) = [];
    FX(delete_idx) = [];
    X_H(delete_idx,:,:) = [];
    FX_H(delete_idx,:)   = [];
    level_indicator(delete_idx) = [];
    stagnation_count(delete_idx) = [];
    label(delete_idx, :) = [];
    max_Arc_size = ceil(ps * Archfactor);
    if Arc_size > max_Arc_size
        shuffle_idx = randperm(Arc_size)';
        keep_idx = shuffle_idx(1 : max_Arc_size);
        X_Arc = X_Arc(keep_idx, :);
        Arc_size=size(X_Arc,1);
    end
    [bestf, gbestid] = min(FX_H(:,1));
    bestx = X_H(gbestid,:,1);
    hist = append_hist(hist, hist_interval, bestf, max_nfe);
end
end
function result = randCauchy(mu, sigma)
[m,n] = size(mu);
result = mu + sigma*tan(pi*(rand(m,n)-0.5));
end
