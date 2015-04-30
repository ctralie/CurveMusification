function [ XOut ] = traceCurve( Y, X, K, DOPLOT )
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
        [~, neighbs] = sort(neighbs);
        neighbs = neighbs(1:1+K); %Allow to stay at the same point
        lastXs = X(idx(1:ii-1), :);
        DTargetSub = DTarget(1:ii, 1:ii);
        for kk = 1:length(neighbs)
            D2 = D(1:ii, 1:ii);
            thisX = X(neighbs(kk), :);
            DRow = bsxfun(@plus, sum(thisX.^2), sum(lastXs.^2, 2)') - 2*thisX*lastXs';
            DRow = sqrt(DRow);
            DRow(end+1) = 0;
            D2(1:ii, ii) = DRow;
            D2(ii, 1:ii) = DRow;
            %D2 = (max(DTarget(:))/max(D2(:)))*D2;
            thisMatch = sqrt(sum( (D2(:) - DTargetSub(:)).^2 ));
            if thisMatch < bestMatch
                bestMatch = thisMatch;
                idx(ii) = neighbs(kk);
            end
        end
        XSofar = X(idx(1:ii), :);
        if DOPLOT || ii == NY
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

