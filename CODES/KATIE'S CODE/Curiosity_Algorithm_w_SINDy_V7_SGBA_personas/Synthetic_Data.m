% FIM vector
x = [57, 57, 57, 57, 57, 57, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, ...
       59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, ...
       60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, ...
       61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, ...
       62, 62, 62, 62, 62, 62, 62];

% BERG vector
y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 5, 9, 12, 15, 18, 20, 23, 25, ...
        27, 29, 30, 32, 33, 35, 36, 37, 38, 38, 39, 40, 40, ...
        41, 41, 41, 42, 42, 42, 42, 42, 42, 42, ...
        43, 43, 43, 43, 43, 44, 44, 44, 45, 45, 46, ...
        47, 47, 48, 49, 50, 50, 50];

pt = [0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, ...
          1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, ...
          0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0];

%plot(FIM, BERG)
%plot (BERG, pt)

%BERG = 10.12*FIM-571.5;

% Create the cubic polynomial fit
p = polyfit(x, y, 1); % Cubic fit
x_fit = linspace(min(x), max(x), 100); % Generate finer points for smooth curve
y_fit = polyval(p, x_fit); % Evaluate polynomial at finer points

% Display the cubic equation
fprintf('Linear Polynomial Equation: y = %.3fx^3 + %.3fx^2 + %.3fx + %.3f\n', p(1), p(2));

% Create the plot
figure;
hold on;
plot(x, y, 'o', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'Original Data'); % Original data
plot(x_fit, y_fit, '-', 'LineWidth', 2, 'DisplayName', 'Linear Fit'); % Polynomial fit

% Add labels and title
xlabel('FIM Score', 'FontSize', 12);
ylabel('Berg Balance Score', 'FontSize', 12);
title('BERG vs FIM with Linear Fit', 'FontSize', 14);

% Customize the grid and axis
grid on;
xlim([min(x) - 1, max(x) + 1]); % Add some margin to x-axis
ylim([min(y) - 5, max(y) + 5]); % Add some margin to y-axis

% Add legend
legend('show', 'Location', 'best');

hold off;
