%% CLASSIFICATION (FAULTY-HEALTHY ENGINE)
%% 1. Setup and Data Loading
clear; clc; close all;
rng(42);

% Load Data
train_tbl = readtable('train_selected.csv');
test_tbl = readtable('test_selected.csv');
truth_tbl = readtable('PM_truth.txt', 'ReadVariableNames', false);
y_truth = table2array(truth_tbl);

%% 2. Preprocessing (Smoothing - Trailing Window)
w_size = 3;
ids_train = unique(train_tbl.id);
ids_test = unique(test_tbl.id);
cols_to_smooth = {'s1', 's2', 's3', 's4'};

% Train Smoothing
for i = 1:length(ids_train)
    idx = train_tbl.id == ids_train(i);
    for c = 1:length(cols_to_smooth)
        vn = cols_to_smooth{c};
        train_tbl.(vn)(idx) = movmean(train_tbl.(vn)(idx), [w_size-1, 0]);
    end
end

% Test Smoothing
for i = 1:length(ids_test)
    idx = test_tbl.id == ids_test(i);
    for c = 1:length(cols_to_smooth)
        vn = cols_to_smooth{c};
        test_tbl.(vn)(idx) = movmean(test_tbl.(vn)(idx), [w_size-1, 0]);
    end
end

%% 3. Feature Selection & Normalization
THR = 30;
y_train = double(train_tbl.ttf <= THR);
y_test = double(y_truth <= THR);

% Select features
feat_vars = {'cycle','s1','s2','s3','s4'};
X_train = train_tbl(:, feat_vars);

% Extract last row for Test
X_test = array2table(zeros(length(ids_test), length(feat_vars)), 'VariableNames', feat_vars);
for i = 1:length(ids_test)
    idx = test_tbl.id == ids_test(i);
    block = test_tbl(idx, feat_vars);
    X_test(i, :) = block(end, :);
end

% Z-Score Normalization 
mu = mean(table2array(X_train));
sigma = std(table2array(X_train));
X_train_norm = array2table((table2array(X_train) - mu) ./ sigma, 'VariableNames', feat_vars);
X_test_norm = array2table((table2array(X_test) - mu) ./ sigma, 'VariableNames', feat_vars);

%% 4. Logistic Regression (10-Fold CV Optimized)
K_FOLDS = 10;
cv = cvpartition(length(ids_train), 'KFold', K_FOLDS);
thresholds = zeros(K_FOLDS, 1);


for k = 1:K_FOLDS
    % Train/Val Split by Engine ID
    train_ids = ids_train(training(cv, k));
    val_ids = ids_train(test(cv, k));
    
    is_train = ismember(train_tbl.id, train_ids);
    is_val = ismember(train_tbl.id, val_ids);
    
    % Train Model
    mdl = fitglm(X_train_norm(is_train,:), y_train(is_train), 'Distribution', 'binomial');
    probs = predict(mdl, X_train_norm(is_val,:));
    
    % Find Safety Threshold (Recall >= 0.95)
    [~, Y_roc, T_roc, ~] = perfcurve(y_train(is_val), probs, 1);
    candidates = find(Y_roc >= 0.95);
    
    if ~isempty(candidates)
        thresholds(k) = T_roc(candidates(1));
    else
        [~, max_idx] = max(Y_roc);
        thresholds(k) = T_roc(max_idx);
    end
    
    if isinf(thresholds(k)), thresholds(k) = 0; end
end
FINAL_THRESH = mean(thresholds);


% Final Logistic Model
mdl_log = fitglm(X_train_norm, y_train, 'Distribution', 'binomial');
prob_log_tr = predict(mdl_log, X_train_norm);
prob_log_te = predict(mdl_log, X_test_norm);
pred_log_tr = double(prob_log_tr >= FINAL_THRESH);
pred_log_te = double(prob_log_te >= FINAL_THRESH);
[~, ~, ~, AUC_log_tr] = perfcurve(y_train, prob_log_tr, 1);
[~, ~, ~, AUC_log_te] = perfcurve(y_test, prob_log_te, 1);

%% 5. KNN (K Optimization)
k_range = 35:55;
cv_scores = zeros(length(k_range), 1);

for i = 1:length(k_range)
    k_val = k_range(i);
    fold_f1 = zeros(K_FOLDS, 1);
    
    for f = 1:K_FOLDS
        train_ids = ids_train(training(cv, f));
        val_ids = ids_train(test(cv, f));
        
        is_train = ismember(train_tbl.id, train_ids);
        is_val = ismember(train_tbl.id, val_ids);
        
        mdl_k = fitcknn(X_train_norm(is_train,:), y_train(is_train), 'NumNeighbors', k_val);
        pred_k = predict(mdl_k, X_train_norm(is_val,:));
        
        % F1 Calculation
        tp = sum(pred_k==1 & y_train(is_val)==1);
        fp = sum(pred_k==1 & y_train(is_val)==0);
        fn = sum(pred_k==0 & y_train(is_val)==1);
        fold_f1(f) = 2*tp / (2*tp + fp + fn + eps);
    end
    cv_scores(i) = mean(fold_f1);
end
[max_f1, best_idx] = max(cv_scores);
BEST_K = k_range(best_idx);


% Final KNN Model
mdl_knn = fitcknn(X_train_norm, y_train, 'NumNeighbors', BEST_K);

% Optimize KNN Threshold on Train Set
[~, sc_tr] = predict(mdl_knn, X_train_norm);
prob_knn_tr = sc_tr(:,2);
[Xk, Yk, Tk, ~] = perfcurve(y_train, prob_knn_tr, 1);
[~, opt_idx] = max(Yk - Xk); 
THRESH_KNN = Tk(opt_idx);

pred_knn_tr = double(prob_knn_tr >= THRESH_KNN);
[~, sc_te] = predict(mdl_knn, X_test_norm);
prob_knn_te = sc_te(:,2);
pred_knn_te = double(prob_knn_te >= THRESH_KNN);

[~, ~, ~, AUC_knn_tr] = perfcurve(y_train, prob_knn_tr, 1);
[~, ~, ~, AUC_knn_te] = perfcurve(y_test, prob_knn_te, 1);

%% 6. Results Summary
calc_mets = @(y, p) [mean(y==p), ...
                     sum(y==1 & p==1)/(sum(p==1)+eps), ...
                     sum(y==1 & p==1)/(sum(y==1)+eps), ...
                     2*sum(y==1 & p==1)/(sum(y==1)+sum(p==1)+eps)];

% Calculate Metrics for Train and Test
m_log_tr = [calc_mets(y_train, pred_log_tr), AUC_log_tr];
m_log_te = [calc_mets(y_test, pred_log_te), AUC_log_te];
m_knn_tr = [calc_mets(y_train, pred_knn_tr), AUC_knn_tr];
m_knn_te = [calc_mets(y_test, pred_knn_te), AUC_knn_te];

fprintf('\n%s\n', repmat('=', 65));
fprintf(' TRAIN vs TEST PERFORMANCE SUMMARY \n');
fprintf('%s\n', repmat('=', 65));
fprintf('%-10s | %-12s %-12s || %-12s %-12s\n', 'Metric', 'Log(TRAIN)', 'Log(TEST)', 'KNN(TRAIN)', 'KNN(TEST)');
fprintf('%s\n', repmat('-', 65));
fprintf('%-10s | %.4f       %.4f       || %.4f       %.4f\n', 'Accuracy', m_log_tr(1), m_log_te(1), m_knn_tr(1), m_knn_te(1));
fprintf('%-10s | %.4f       %.4f       || %.4f       %.4f\n', 'Precision', m_log_tr(2), m_log_te(2), m_knn_tr(2), m_knn_te(2));
fprintf('%-10s | %.4f       %.4f       || %.4f       %.4f\n', 'Recall', m_log_tr(3), m_log_te(3), m_knn_tr(3), m_knn_te(3));
fprintf('%-10s | %.4f       %.4f       || %.4f       %.4f\n', 'F1 Score', m_log_tr(4), m_log_te(4), m_knn_tr(4), m_knn_te(4));
fprintf('%-10s | %.4f       %.4f       || %.4f       %.4f\n', 'AUC', m_log_tr(5), m_log_te(5), m_knn_tr(5), m_knn_te(5));
fprintf('%s\n', repmat('=', 65));

%% 7. Visualizations

% Figure 1: Logistic Confidence
figure('Color','w','Name','Logistic Confidence');
[p_sorted, idx] = sort(prob_log_tr);

scatter(find(y_train(idx)==0), p_sorted(y_train(idx)==0), 10, [0.6 0.8 1], 'filled'); hold on;
scatter(find(y_train(idx)==1), p_sorted(y_train(idx)==1), 10, [1 0.7 0.7], 'filled');

yline(FINAL_THRESH, 'k--', 'LineWidth', 1.5);

ylabel('Probability P(Faulty)');  
xlabel('Samples (Sorted)');        
legend({'True Healthy', 'True Faulty'}, 'Location', 'southeast');
% ---------------------------------------------------

grid on; ylim([-0.05 1.05]); hold off;
% Figure 2: Feature Importance
figure('Color','w','Name','Feature Importance');
bar(abs(mdl_log.Coefficients.Estimate(2:end)), 'FaceColor', [0.3 0.5 0.7]);
xticklabels(feat_vars); ylabel('Magnitude'); grid on;

% Figure 3: ROC Curves
figure('Color','w','Name','ROC Analysis');
[xl, yl] = perfcurve(y_test, prob_log_te, 1);
[xk, yk] = perfcurve(y_test, prob_knn_te, 1);
plot(xl, yl, 'LineWidth', 2); hold on;
plot(xk, yk, '--', 'LineWidth', 2);
plot([0 1],[0 1], 'k:');
legend('Logistic', 'KNN'); xlabel('False Positive Rate'); ylabel('True Positive Rate'); grid on;

% Figure 4: Confusion Matrices
figure('Color','w','Name','Confusion Matrices');
t = tiledlayout(1, 2, 'Padding', 'compact');
nexttile; confusionchart(y_test, pred_log_te); title('Logistic Regression');
nexttile; confusionchart(y_test, pred_knn_te); title(['KNN (K=' num2str(BEST_K) ')']);

% Figure 5: Test Performance Summary 
figure('Color','w','Name','Test Performance');
b = bar([m_log_te; m_knn_te]', 'grouped');
b(1).FaceColor = [0.6 0.8 1]; 
b(2).FaceColor = [1 0.7 0.7]; 
legend('Logistic', 'KNN');
xticklabels({'Accuracy', 'Precision', 'Recall', 'F1', 'AUC'});
ylabel('Score (0-1)'); xlabel('Metric');

% Figure 6: Overfitting Check 
figure('Color','w','Name','Overfitting Analysis');
f1_tr_val = [m_log_tr(4); m_knn_tr(4)];
f1_te_val = [m_log_te(4); m_knn_te(4)];
b2 = bar([f1_tr_val, f1_te_val], 'grouped');
b2(1).FaceColor = [0.6 0.8 1]; % Pastel Blue
b2(2).FaceColor = [1 0.7 0.7]; % Pastel Pink
legend('Train', 'Test'); xticklabels({'Logistic', 'KNN'});
ylabel('F1 Score'); xlabel('Model');
