function [x_hat_all_PF_Matlab, P_x_all_PF_Matlab] = ParticleFilter_Matlab_function(dt, Dynamics_PF, Measurement_PF, z_true_all, x_hat, P_x, P_w, P_v, N_particles)

% July 2021 changes:
% - moved predict before correct
% - passed DiscreteStateDynamics_PF P_w instead of zeros and then adding P_w impact after

% N_particles = 1000;
% x_hat = [2;0];
%P_x = 0.01*eye(2);

pf = particleFilter(@ParticleFilterStateFcn, @PFMeasurementLikelihoodFcn);
initialize(pf, N_particles, x_hat, P_x);

pf.StateEstimationMethod = 'mean';
pf.ResamplingMethod = 'multinomial';
%pf.ResamplingPolicy

% Estimate
%x_hat_all_PF_Matlab = zeros(size(z_true_all));
%P_x_all_PF_Matlab = zeros(size([z_true_all, z_true_all]));
for k=1:size(z_true_all,1)
    
    predict(pf, dt, P_w, Dynamics_PF); % Filter updates and stores Particles[k+1|k]
    [stateCorrected, covCorrected] = correct(pf, z_true_all(k, :), P_v, Measurement_PF); % Filter updates and stores Particles[k|k], Weights[k|k]
    
    % save
    %x_hat_all_PF_Matlab(k, :) = stateCorrected;
    %P_x_all_PF_Matlab(k, :) = covCorrected(:);
    
    [State,StateCovariance] = getStateEstimate(pf);
    x_hat_all_PF_Matlab(k, :) = State;
%     P_x_all_PF_Matlab(k, :) = StateCovariance(:);
P_x_all_PF_Matlab = StateCovariance;
end

end


function particles = ParticleFilterStateFcn(particles, dt, P_w, Dynamics_PF) 
% Discrete-time approximation to van der Pol ODEs for mu = 1. 
% Sample time is 0.05s.

[numberOfStates, numberOfParticles] = size(particles);
% 
% dt = 0.05; % [s] Sample time
% for kk=1:numberOfParticles
%     particles(:,kk) = particles(:,kk) + vdpStateFcnContinuous(particles(:,kk))*dt;
% end
%dt = 0.1;

%particle_noise = 0*particles;
% 
% particle_noise0 = mvnrnd(zeros(numberOfStates, 1), P_w, numberOfParticles)'; 
% mean0 =  mean(particle_noise0, 2);
% cov0 = cov(particle_noise0');
% 
% % This gives x_hat nearly exactly equal to other filters, but P_x is off
% particle_noise1 = mvnrnd(zeros(numberOfStates, 1), sqrt(P_w), numberOfParticles)'; 
% mean1 =  mean(particle_noise1, 2);
% cov1 = cov(particle_noise1');
% 
% particle_noise1b = [normrnd(0, sqrt(P_w(1, 1)), [1, numberOfParticles]); normrnd(0, sqrt(P_w(2, 2)), [1, numberOfParticles])];
% mean1b =  mean(particle_noise1b, 2);
% cov1b = cov(particle_noise1b');
% 
% particle_noise1c = [sqrt(P_w(1, 1))*randn([1, numberOfParticles]); sqrt(P_w(2, 2))*randn([1, numberOfParticles])];
% mean1c =  mean(particle_noise1c, 2);
% cov1c = cov(particle_noise1c');
% %particle_noise5 = mvnrnd(zeros(numberOfStates, 1), chol(P_w), numberOfParticles)'; 
% 
% % Inbetween 1 and 2, same as 4
% particle_noise3 = mvnrnd(zeros(numberOfStates, 1), P_w, numberOfParticles)'; 
% 
% % This give P_x equal to Bayes, but x_hat is off
% particle_noise2 = P_w* randn(size(particles));
% mean2 = mean(particle_noise2, 2);
% cov2 = cov(particle_noise2');

% Inbetween 1 and 2, same as 3, but this is how Matlab says randn should be used
sqrtP_w = chol(P_w);
particle_noise4 = sqrtP_w* randn(size(particles));
% mean4 = mean(particle_noise4, 2);
% cov4 = cov(particle_noise4');


particle_noise = particle_noise4;

% figure
% hold on
% plot(particle_noise1(1, :), particle_noise1(2, :), 'o')
% plot(particle_noise2(1, :), particle_noise2(2, :), 'x')


for p = 1:numberOfParticles
    particles(:, p) = Dynamics_PF(dt, particles(:, p), particle_noise(:, p));
end

% Add Gaussian noise on each state variable
% processNoise = P_w*eye(numberOfStates);
% particles = particles + processNoise * randn(size(particles));
end

% function likelihood = vdpExamplePFMeasurementLikelihoodFcn(particles, measurement, P_v)
% % vdpExamplePFMeasurementLikelihoodFcn Example measurement likelihood function
% %
% % The measurement is the first state.
% %
% % likelihood = vdpParticleFilterMeasurementLikelihoodFcn(particles, measurement)
% %
% % Inputs:
% %    particles - NumberOfStates-by-NumberOfParticles matrix that holds 
% %                the particles
% %
% % Outputs:
% %    likelihood - A vector with NumberOfParticles elements whose n-th
% %                 element is the likelihood of the n-th particle
% 
% measurement = measurement';
% % Validate the sensor measurement
% numberOfMeasurements = 2; % Expected number of measurements
% validateattributes(measurement, {'double'}, {'vector', 'numel', numberOfMeasurements}, ...
%     'vdpExamplePFMeasurementLikelihoodFcn', 'measurement');
% 
% % The measurement is first state. Get all measurement hypotheses from particles
% predictedMeasurement = particles;
% 
% % Assume the ratio of the error between predicted and actual measurements
% % follow a Gaussian distribution with zero mean, variance 0.2
% mu = 0; % mean
% sigma = P_v; % variance
% 
% % Use multivariate Gaussian probability density function, calculate
% % likelihood of each particle
% numParticles = size(particles,2);
% likelihood = zeros(numParticles,1);
% C = det(2*pi*sigma) ^ (-0.5);
% for kk=1:numParticles
%     errorRatio = (predictedMeasurement(:, kk)-measurement)/predictedMeasurement(:, kk);
%     v = errorRatio-mu;
%     likelihood(kk) = C * exp(-0.5 * (v' / sigma * v) );
% end
% end


function  likelihood = PFMeasurementLikelihoodFcn(predictParticles, z, P_v, Measurement_PF)


% Assume that measurements are subject to Gaussian distributed noise with
% variance 0.016
% Specify noise as covariance matrix
%measurementNoise = 0.016 * eye(numberOfMeasurements);
  
% The measurement contains the first state variable. Get the first state of
% all particles

%predictedMeasurement = predictParticles; % I should actually run these through the measurement function with zero noise here...

[numberOfStates, numberOfParticles] = size(predictParticles);

% Instead of assuming no noise as above, add noise as if was run through measurement function
% sqrtP_v = chol(P_v);
% particle_noise = sqrtP_v* randn(size(predictParticles));
% particle_noise = 0.*particle_noise;

particle_noise = zeros([length(P_v), numberOfParticles]);

% predictedMeasurement = predictParticles + particle_noise;

for p = 1:numberOfParticles
    predictedMeasurement(:, p) = Measurement_PF(predictParticles(:, p), particle_noise(:, p));
end
%predictedMeasurement = Measurement(predictParticles, particle_noise);

%numberOfMeasurements = length(z); % Expected number of measurements

% Calculate error between predicted and actual measurement
%measurementError = bsxfun(@minus, predictedMeasurement, measurement');

% Use measurement noise and take inner product
% Andrew modified this to be for loop as orginially coded for 1D: measurementErrorProd = dot(measurementError, measurementNoiseCovariance \ measurementError, numberOfMeasurements);
% measurementErrorProd = 0.*measurementError;
% for i = 1:length(predictParticles)
%     measurementErrorProd(:, i) = dot(measurementError(:, i), measurementNoiseCovariance \ measurementError(:, i), numberOfMeasurements);
% end


% Convert error norms into likelihood measure. 
% Evaluate the PDF of the multivariate normal distribution. A measurement
% error of 0 results in the highest possible likelihood.
%likelihood = 1/sqrt((2*pi).^numberOfMeasurements * det(measurementNoiseCovariance)) * exp(-0.5 * measurementErrorProd);
       
%likelihood = mvnpdf(measurement, predictedMeasurement', sqrt(measurementNoiseCovariance));

likelihood = mvnpdf(z - predictedMeasurement', zeros(size(predictedMeasurement')), P_v(1:length(z), 1:length(z)));
%likelihood = mvnpdf(z, predictedMeasurement', P_v); % same as above
       
   end