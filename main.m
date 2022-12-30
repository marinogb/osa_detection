%% Detection of obstructive sleep apnea from time series data
% 
% This script generates the data for training and testing
% 
% Marino Gavidia,. et al.

addpath("lib/") % Add functions to path 
addpath("samples/") % Add functions to path 

%% Parameters
tw=60; % window's size

%% Input folder
inputf="samples/";

%% Output folder
outputf="data/";
mkdir(outputf);

%% Load Samples
load("meta_data")

%% Generate data for training/testing
pr_dat(tw,meta_data,outputf);

