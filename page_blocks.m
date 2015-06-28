load data/data
testIndex = 3:3:length(data);
testSet = data(testIndex, :);
trSet = setdiff(data, testSet, 'rows');
trSet = [trSet; min(data); max(data)];
xTrSet = trSet(:, 1:10);
yTrSet = trSet(:, 11);
xTestSet = testSet(:, 1:10);
yTestSet = testSet(:, 11);

% Distribution data
for i = 1:size(trSet, 2)
    meanTr(i) = mean(trSet(:, i));
    stdTr(i) = std(trSet(:, i));
end

for i = 1:size(testSet, 2)
    meanTest(i) = mean(testSet(:, i));
    stdTest(i) = std(testSet(:, i));
end
%----------
    

% Synthesis of decision trees
T = classregtree(xTrSet, yTrSet, 'method', 'classification');
view(T)
%----------

% Testing decision tree
[numNodeTr, errorTr] = test_tree(T, xTrSet, yTrSet);
[numNodeTest, errorTest] = test_tree(T, xTestSet, yTestSet);
figure(1);
plot(numNodeTr, errorTr, 'b-o', numNodeTest, errorTest, 'g-o');
xlabel('Number of nodes');
ylabel('Number of erros');
legend('tr set', 'test set');
%--------------------

% Optimal decision tree
[minError, minInd] = min(errorTest);
optimalTree = prune(T, 'level', minInd);
view(T, 'prunelevel', minInd);
%----------

% Comparison of different tree synthesis algorithms
method = [{'gdi'}, {'deviance'}, {'twoing'}];
for i = 1:length(method)
 Tree{i} = classregtree(xTrSet, yTrSet, 'method', 'classification', 'splitcriterion', method{i});
 [numNodeTr1{i}, errorTr1{i}] = test_tree(Tree{i}, xTrSet, yTrSet);
 [numNodeTest1{i}, errorTest1{i}] = test_tree(Tree{i}, xTestSet, yTestSet);
 subplot(2, 2, i);
 plot(numNodeTr1{i}, errorTr1{i}, 'b-o', numNodeTest1{i}, errorTest1{i}, 'g-o');
 title(method{i});
 legend('tr set', 'test set');
 xlabel('Number of nodes');
 ylabel('Number of erros');
end
%---------
 
% Confusion matrix
yfit = eval(optimalTree, xTestSet);
if iscell(yfit)
    yfit = cell2mat(yfit);
    yfit = str2num(yfit);
end
CM_test = confusionmat(yfit, yTestSet);
 
yfit = eval(optimalTree, xTrSet);
if iscell(yfit)
    yfit = cell2mat(yfit);
    yfit = str2num(yfit);
end
CM_tr = confusionmat(yfit, yTrSet);
%--------

% Distribution classes
for i = 1:length(yTrSet)
    switch (yTrSet(i))
        case 1
            class = 'text';
        case 2
            class = 'horiz. line';
        case 3
            class = 'graphic';
        case 4
            class = 'vert. line';
        case 5
            class = 'picture';
    end
    group(i, 1) = { class };
end
figure(2);
gplotmatrix(xTrSet, [], group);
%--------------
 
