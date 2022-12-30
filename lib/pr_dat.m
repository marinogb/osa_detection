function [] = pr_dat(tw,meta_data,output_folder)
%% This script processes the data
warning('off')

%% CHANNELS
sensor_target{1}={'EKG','ECGL','ECG','ECGLECGR'};% ECG
sensor_target{2}={'ABDO','ABDOMINAL','ABD','ABDORES','ABDOMEN'}; % ABDOMINAL
sensor_target{3}={'THOR','THORACIC','CHEST','THORRES','THORAX'}; % CHEST
sensor_target{4}={'FLOW','AUX','CANNULAFLOW','NASALFLOW','NEWAIR','AIRFLOW'}; % NASSAL   AirFLow   
sensor_target{5}={'SPO2','SAO2','SPO2'}; % O2

for tt = 1:height(meta_data)
disp("file_" + num2str(tt))
% INPUT FILE
file_name=meta_data.File{tt};
duration=string(meta_data.Duration(tt));
duration=split(duration," ");
duration=str2double(duration{1});

% FIND SENSORS
sensors=meta_data.Channels(tt,:);
[~,inx] =find_channel_indx(sensor_target,sensors) ;
samp_fs=meta_data.Fs(tt,inx);

% CREATE OUTPUT FILE
outputf=strcat( output_folder,'/',num2str(tt) ,'.hdf5');
h5create(outputf,'/y', [1 Inf ],'Datatype','double','ChunkSize', [1 1]);

% READ DATA
[DATA,~] = edf_read(file_name);

% GET LABELS
label=[];
label=zeros(1,duration);
for jj = 1:length(meta_data.Label{tt})
    osa_ini= meta_data.Label{tt}{jj,1} ;
    if osa_ini<1
        osa_ini=1;
    end
    osa_end=meta_data.Label{tt}{jj,2}  ;
    % %END OF FILE?
    if osa_end < duration
        label(osa_ini:osa_end)=meta_data.Label{tt}{jj,3};
    else
        label(osa_ini:duration)=meta_data.Label{tt}{jj,3};
    end
end
normal=find(label(1:end-tw)==0);
normal=normal(randperm( length(normal)));

% Go through channels
for ch=length(inx):-1:1
    sen_inx=inx(ch);
    dim=samp_fs{ch}*(tw);
    h5create(outputf,"/x"+num2str(ch), [1  dim*3 Inf ],'Datatype','double','ChunkSize', [1  dim 1]);
    DATA_s=DATA{:,sen_inx};
    if samp_fs{ch}>1
        DATA_s=cat(1,DATA_s{:,1});
    end
    DATA_s=DATA_s(1:duration*samp_fs{ch});

    % GEN DAT_OUT
    gen_dat(ch, DATA_s,outputf,tw,samp_fs{ch},dim,label,normal);

end

end
end

