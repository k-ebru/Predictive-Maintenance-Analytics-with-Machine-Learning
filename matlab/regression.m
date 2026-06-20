%% REGRESSION (TIME-TO-FAILURE/TTF)
%% 1. Setup and Data Loading
clear; clc; close all;

% Load data files
train_tbl = readtable('train_selected.csv');
test_tbl  = readtable('test_selected.csv');
truth_tbl = readtable('PM_truth.txt', 'ReadVariableNames', false);
y_true_test = table2array(truth_tbl);


%% 2. Preprocessing (Smoothing) - Trailing Window
w_size = 3; 

% Smooth Training Data
train_ids = unique(train_tbl.id);
for i = 1:length(train_ids)
    idx = train_tbl.id == train_ids(i);
    train_tbl.s1(idx) = movmean(train_tbl.s1(idx), [w_size-1, 0]);
    train_tbl.s2(idx) = movmean(train_tbl.s2(idx), [w_size-1, 0]);
    train_tbl.s3(idx) = movmean(train_tbl.s3(idx), [w_size-1, 0]);
    train_tbl.s4(idx) = movmean(train_tbl.s4(idx), [w_size-1, 0]);
end

% Smooth Test Data
test_ids = unique(test_tbl.id);
for i = 1:length(test_ids)
    idx = test_tbl.id == test_ids(i);
    test_tbl.s1(idx) = movmean(test_tbl.s1(idx), [w_size-1, 0]);
    test_tbl.s2(idx) = movmean(test_tbl.s2(idx), [w_size-1, 0]);
    test_tbl.s3(idx) = movmean(test_tbl.s3(idx), [w_size-1, 0]);
    test_tbl.s4(idx) = movmean(test_tbl.s4(idx), [w_size-1, 0]);
end
% Select Features and Target
feats = {'cycle','s1','s2','s3','s4'};
X_train = train_tbl(:, feats);
y_train = train_tbl.ttf;
X_test  = test_tbl(:, feats);

% Target Clipping (Training)
y_train(y_train > 125) = 125; 

%% 3. Normalization
% Calculate Mean and Std from Training Data
mean_vals = mean(table2array(X_train));
std_vals  = std(table2array(X_train));

% Create copies for Normalized Data
X_train_norm = X_train;
X_test_norm  = X_test;

% Apply Normalization (X - Mean) / Std
for i = 1:length(feats)
    X_train_norm.(feats{i}) = (X_train.(feats{i}) - mean_vals(i)) / std_vals(i);
    X_test_norm.(feats{i})  = (X_test.(feats{i})  - mean_vals(i)) / std_vals(i);
end

%% 4. Model 1: Stepwise Regression
mdl_sw = stepwiselm(X_train_norm, y_train, 'constant', 'Upper', 'quadratic', ...
                    'ResponseVar', 'ttf', 'Verbose', 0);

% Extract formula with coefficients
coeffs = mdl_sw.Coefficients.Estimate; 
names = mdl_sw.CoefficientNames;       

formula_str = 'TTF = ';

for i = 1:length(names)
    c_val = coeffs(i);
    term_name = names{i};
    
    if strcmp(term_name, '(Intercept)')
        formula_str = [formula_str, sprintf('%.4f', c_val)];
    else
        term_name = strrep(term_name, ':', '*');
        
        if c_val >= 0
            formula_str = [formula_str, sprintf(' + (%.4f * %s)', c_val, term_name)];
        else
            formula_str = [formula_str, sprintf(' - (%.4f * %s)', abs(c_val), term_name)];
        end
    end
end

disp('--------------------------------------------------');
disp('STEPWISE REGRESSION CALCULATION FORMULA:');
disp(formula_str);
disp('--------------------------------------------------');

% Prediction
pred_sw_train = predict(mdl_sw, X_train_norm); 
pred_sw_train(pred_sw_train < 0) = 0;
pred_sw_test = predict(mdl_sw, X_test_norm); 
pred_sw_test(pred_sw_test < 0) = 0;

%% 5. Model 2: Random Forest 
mdl_rf = fitrensemble(X_train, y_train, 'Method', 'Bag', ...
                      'NumLearningCycles',500 , ... 
                      'Learners', templateTree('MinLeafSize', 120));

% Prediction
pred_rf_train = predict(mdl_rf, X_train); 
pred_rf_train(pred_rf_train < 0) = 0;
pred_rf_test = predict(mdl_rf, X_test); 
pred_rf_test(pred_rf_test < 0) = 0;

%% 6. Evaluation Calculations
% Clip Ground Truth for Evaluation
y_true_test_clipped = y_true_test;
y_true_test_clipped(y_true_test_clipped > 125) = 125;

% Stepwise metrics
err_sw_tr = pred_sw_train - y_train;
rmse_sw_tr = sqrt(mean(err_sw_tr.^2));
mae_sw_tr = mean(abs(err_sw_tr));
r2_sw_tr = 1 - (sum(err_sw_tr.^2) / sum((y_train - mean(y_train)).^2));

err_sw_ts = pred_sw_test - y_true_test_clipped; 
rmse_sw_ts = sqrt(mean(err_sw_ts.^2));
mae_sw_ts = mean(abs(err_sw_ts));
r2_sw_ts = 1 - (sum(err_sw_ts.^2) / sum((y_true_test_clipped - mean(y_true_test_clipped)).^2));

% Random forest metrics
err_rf_tr = pred_rf_train - y_train;
rmse_rf_tr = sqrt(mean(err_rf_tr.^2));
mae_rf_tr = mean(abs(err_rf_tr));
r2_rf_tr = 1 - (sum(err_rf_tr.^2) / sum((y_train - mean(y_train)).^2));

err_rf_ts = pred_rf_test - y_true_test_clipped; 
rmse_rf_ts = sqrt(mean(err_rf_ts.^2));
mae_rf_ts = mean(abs(err_rf_ts));
r2_rf_ts = 1 - (sum(err_rf_ts.^2) / sum((y_true_test_clipped - mean(y_true_test_clipped)).^2));

%% 7. Display Results
fprintf('\nSTEPWISE QUADRATIC REGRESSION RESULTS:\n');
fprintf('Train -> MAE: %.4f, RMSE: %.4f, R^2: %.4f\n', mae_sw_tr, rmse_sw_tr, r2_sw_tr);
fprintf('Test  -> MAE: %.4f, RMSE: %.4f, R^2: %.4f\n', mae_sw_ts, rmse_sw_ts, r2_sw_ts);

fprintf('\nRANDOM FOREST RESULTS:\n');
fprintf('Train -> MAE: %.4f, RMSE: %.4f, R^2: %.4f\n', mae_rf_tr, rmse_rf_tr, r2_rf_tr);
fprintf('Test  -> MAE: %.4f, RMSE: %.4f, R^2: %.4f\n', mae_rf_ts, rmse_rf_ts, r2_rf_ts);
fprintf('\n');

%% 8. Visualization
lbls = {'Cycle', 'S_{1}', 'S_{2}', 'S_{3}', 'S_{4}'};
sensor_names = {'s1', 's2', 's3', 's4'};

% A. Sensor degradation path for one engine
figure('Name', 'Single Unit Degradation', 'Color', 'w', 'NumberTitle', 'off');
sample_id = 11; 
idx = train_tbl.id == sample_id; 

for k = 1:4
    subplot(2, 2, k);
    x_val = train_tbl.ttf(idx);       
    y_val = train_tbl.(sensor_names{k})(idx); 
    
    plot(x_val, y_val, 'b-', 'LineWidth', 1);
    hold on;
    xline(125, 'r--', 'Threshold (125)', 'LineWidth', 1, 'LabelVerticalAlignment', 'bottom');
    
    xlabel('TTF (Time-to-failure)');
    ylabel(['Sensor Value (' sensor_names{k} ')']);
    title(['Sensor ' sensor_names{k} ' (Engine #' num2str(sample_id) ')']);
    grid on;
    set(gca, 'XDir', 'reverse'); 
    hold off;
end

% B. Sensor degradation paths for all engines
figure('Name', 'Sensor Degradation Paths', 'Color', 'w', 'NumberTitle', 'off');
unique_ids = unique(train_tbl.id);

for k = 1:4
    subplot(2, 2, k);
    hold on;
    for i = 1:length(unique_ids)
        idx = train_tbl.id == unique_ids(i);
        x_val = train_tbl.ttf(idx);       
        y_val = train_tbl.(sensor_names{k})(idx); 
        plot(x_val, y_val, '-', 'Color', [0 0.447 0.741 0.3], 'LineWidth', 0.5);
    end
    
    xlabel('TTF (Time-to-failure)');
    ylabel(['Sensor Value (' sensor_names{k} ')']);
    title(['Sensor ' sensor_names{k} ' Degradation Paths']);
    grid on;
    set(gca, 'XDir', 'reverse'); 
    hold off;
end

% --- C. Correlation Heatmap ---
figure('Name', 'Correlation', 'NumberTitle', 'off');
c_mat = corr(table2array(X_train));
h = heatmap(feats, feats, c_mat);
h.Title = ' ';
h.XDisplayLabels = lbls;
h.YDisplayLabels = lbls;
h.Colormap = [linspace(1, 0, 256)', linspace(1, 0, 256)', ones(256, 1)];
% --- D. Stepwise Feature Importance
figure('Name', 'Stepwise Imp', 'NumberTitle', 'off');

coeffs = mdl_sw.Coefficients.Estimate(2:end); 
c_names = mdl_sw.CoefficientNames(2:end);

c_names = strrep(c_names, ':', ' * '); 

bar(abs(coeffs), 'FaceColor', [0.2 0.6 0.5]); 

xticks(1:length(c_names));
xticklabels(c_names);
xtickangle(45);

ylabel('Absolute Coefficient Value (Stepwise Regression)');
grid on; box on;
% --- E. Random Forest Feature Importance ---
figure('Name', 'RF Imp', 'NumberTitle', 'off');
imp = predictorImportance(mdl_rf);
bar(imp, 'FaceColor', [0.2 0.6 0.5]); 
xticks(1:length(lbls));
xticklabels(lbls);
ylabel('Importance Score (Random Forest Regression)');
grid on; box on;

% --- F. Stepwise Predictions ---
figure('Color','w','Name', 'Stepwise Pred', 'NumberTitle', 'off');
hold on;
x = (1:length(y_true_test))'; 
y_pred = pred_sw_test; 
rmse_val = rmse_sw_ts;

fill([x; flipud(x)], ...
     [y_pred - rmse_val; flipud(y_pred + rmse_val)], ...
     [0.8 0.85 1], 'EdgeColor','none', 'FaceAlpha',0.5);
plot(x, y_pred, 'b-', 'LineWidth', 1.5);
scatter(x, y_true_test_clipped, 25, 'ko', 'filled', 'MarkerFaceAlpha', 0.6);
xlabel('Sample Index'); 
ylabel('TTF');
legend({'\pm1 RMSE region', 'Stepwise Prediction', 'True Values'}, 'Location', 'best');
grid on; hold off;

% --- G. Random Forest Predictions ---
figure('Color','w','Name', 'RF Pred', 'NumberTitle', 'off');
hold on;
x = (1:length(y_true_test))'; 
y_pred = pred_rf_test; 
rmse_val = rmse_rf_ts;

fill([x; flipud(x)], ...
     [y_pred - rmse_val; flipud(y_pred + rmse_val)], ...
     [1 0.85 0.85], 'EdgeColor','none', 'FaceAlpha',0.5);
plot(x, y_pred, 'r-', 'LineWidth', 1.5);
scatter(x, y_true_test_clipped, 25, 'ko', 'filled', 'MarkerFaceAlpha', 0.6);
xlabel('Sample Index'); 
ylabel('TTF');
legend({'\pm1 RMSE region', 'RF Prediction', 'True Values'}, 'Location', 'best');
grid on; hold off;

% --- H. Residual Analysis ---
figure('Name', 'Residuals', 'NumberTitle', 'off');
[f_sw, xi_sw] = ksdensity(err_sw_ts);
[f_rf, xi_rf] = ksdensity(err_rf_ts);
plot(xi_sw, f_sw, 'b-', 'LineWidth', 2); hold on;
plot(xi_rf, f_rf, 'r-', 'LineWidth', 2);
xline(0, 'k--', 'LineWidth', 1);
xlabel('Residual Error (Predicted - True)');
ylabel('Density');
legend('Stepwise Regression', 'Random Forest', 'Zero Error');
grid on;
