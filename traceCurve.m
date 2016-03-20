function [ XOut, idx ] = traceCurve( Y, X, K, DOPLOT )
    DTarget = squareform(pdist(Y));
    NX = size(X, 1);
    NY = size(Y, 1);
    D = zeros(NY, NY);
    idx = randperm(NX, NY);
    XSumT = sum(X.^2, 2)';
    for ii = 2:NY
        fprintf(1, '%i of %i\n', ii, NY);
        bestMatch = inf;
        %Get the K nearest neighbors to the last point chosen
        lastX = X(idx(ii-1), :);
        neighbs = bsxfun(@plus, sum(lastX.^2), XSumT) - 2*lastX*X';
        [neighbsdist, neighbs] = sort(neighbs);
        %Make sure to points that are far enough away are included
        n = find(neighbsdist > DTarget(ii-1, ii), 1);
        if isempty(n)
            n = NX;
        end
        n = max(n, K+1);
        neighbs = neighbs(1:n); %Allow to stay at the same point
        lastXs = X(idx(1:ii-1), :);
        DTargetSub = DTarget(ii, 1:ii-1);
        for kk = 1:length(neighbs)
            thisX = X(neighbs(kk), :);
            DRow = bsxfun(@plus, sum(thisX.^2), sum(lastXs.^2, 2)') - 2*thisX*lastXs';
            DRow = sqrt(DRow);
            thisMatch = sqrt(sum( (DRow(:) - DTargetSub(:)).^2 ));
            if thisMatch < bestMatch
                bestMatch = thisMatch;
                idx(ii) = neighbs(kk);
            end
        end
        if DOPLOT
            XSofar = X(idx(1:ii), :);
            clf;
            subplot(2, 2, 1);
            plot(X(:, 1), X(:, 2), 'b.');
            hold on;
            plot(XSofar(:, 1), XSofar(:, 2), 'r', 'LineWidth', 3);
            title(sprintf('%i of %i', ii, NY));
            subplot(2, 2, 2);
            DSoFar = zeros(NY, NY);
            DSoFar(1:ii, 1:ii) = squareform(pdist(X(idx(1:ii), :)));
            imagesc(DSoFar);
            subplot(2, 2, 3);
            plot(Y(:, 1), Y(:, 2), '.');
            hold on;
            plot(Y(1:ii, 1), Y(1:ii, 2), 'r');
            subplot(2, 2, 4);
            imagesc(DTarget);
            print('-dpng', sprintf('%i.png', ii));
        end
    end
    XOut = X(idx, :);
end

