function [hist] = append_hist(hist, neval, val, maxfes)
curr = numel(hist);
if curr >= maxfes || neval == 0
    hist = hist(1:maxfes);
    return;
end
n_written = min(neval, maxfes - curr);
hist(end + (1:n_written)) = val;
end

