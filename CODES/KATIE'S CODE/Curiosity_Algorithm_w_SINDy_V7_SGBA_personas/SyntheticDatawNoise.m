clc;
close all;
clear;
% Define the range of FIM values
FIM = 57:0.1:62; % Example range of FIM values

% Calculate BERG values using the equation
BERG = 10.12 * FIM - 571.5;

% (Optional) Add random noise to simulate variability
noise = randn(size(FIM)) * 5; % Adjust the noise level as needed
BERG_noisy = BERG + noise;

% Plot the simulated data
figure;
plot(FIM, BERG, 'b-', 'LineWidth', 2); % Plot without noise
hold on;
plot(FIM, BERG_noisy, 'ro', 'MarkerSize', 8); % Plot with noise
xlabel('FIM Score', 'FontSize', 12);
ylabel('BERG', 'FontSize', 12);
title('Simulated Data for BERG vs. FIM', 'FontSize', 14);
legend('Exact BERG', 'Noisy BERG', 'Location', 'NorthWest');
grid on;
