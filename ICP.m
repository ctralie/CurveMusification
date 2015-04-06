%Do ICP from P1 to P2
function [ T, P, iters ] = ICP( P1, P2, MaxIter, DOPLOT )
    if nargin < 3
        MaxIter = 1000;
    end
    if nargin < 4
        DOPLOT = 0;
    end
    dim = size(P1, 2);
    T = eye(dim+1);
    N1 = size(P1, 1);
    lastidx = ones(N1, 1);
    
    AllP = [P1; P2];
    xmin = min(AllP(:, 2));
    xmax = max(AllP(:, 2));
    ymin = min(AllP(:, 1));
    ymax = max(AllP(:, 2));
    iters = zeros(1, MaxIter);
    
    for iter = 1:MaxIter
        P = [P1 ones(N1, 1)];
        P = (T*P')';
        D = pdist2(P(:, 1:dim), P2);
        [dists, idx] = min(D, [], 2);
        iters(iter) = sum(dists);
        if sum(idx == lastidx) == N1
            %If it's converged
            break;
        end
        lastidx = idx;
        
        if DOPLOT
            clf;
            plot(P(:, 1), P(:, 2), 'r.');
            hold on;
            plot(P2(:, 1), P2(:, 2), 'b.');
            for kk = 1:size(P, 1)
                thisP = [P(kk, 1:dim); P2(idx(kk), :)];
                plot(thisP(:, 1), thisP(:, 2), 'g');
            end
            xlim([xmin, xmax]);
            ylim([ymin, ymax]);
            title(sprintf('Iteration %i', iter));
            print('-dpng', '-r100', sprintf('%i.png', iter));
        end
        
        T = getRigidTransformation(P1, P2(idx, :));
    end
    iters = iters(1:iter);
    P = P(:, 1:dim);
end

