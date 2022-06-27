function [out4lcell] = lcell4(DurationInSeconds)
%% Setup Session, Add Channels and Configure Parameters
% Create the data acquisition session
close all

daqSession = daq.createSession('ni');
% Create analog input channel with board ID 'Dev2', Channel 'ai0', measuring 'Voltage'
daqSession.addAnalogInputChannel('Dev2', 'ai0', 'Voltage');
% Create analog input channel with board ID 'Dev1', Channel 'ai1', measuring 'Voltage'
daqSession.addAnalogInputChannel('Dev2', 'ai1', 'Voltage');
% Create analog input channel with board ID 'Dev1', Channel 'ai2', measuring 'Voltage'
daqSession.addAnalogInputChannel('Dev2', 'ai2', 'Voltage');
% Create analog input channel with board ID 'Dev1', Channel 'ai3', measuring 'Voltage'
daqSession.addAnalogInputChannel('Dev2', 'ai3', 'Voltage');
% Set property value
daqSession.DurationInSeconds = DurationInSeconds;
%% Data Acquisition and Plotting
% Start the acquisition
disp(['Wait ',num2str(DurationInSeconds),' seconds'])
disp('Acquiring data...');
[data, time] = daqSession.startForeground();
out4lcell = [time data];
disp('Acquisition complete.');

% Plot the acquired data in a new figure window
figure
plot(time, data);
title('RAW DATA')
legend('load cell 1','load cell 2','load cell 3','load cell 4')
grid on
ylabel('Voltage')
xlabel('Time [s]')

%% Clean up and release hardware
daqSession.release();
delete(daqSession);
clear daqSession;

dlmwrite(['data_',num2str(round(clock)),'.csv'],out4lcell,'-append','precision',20);
end
