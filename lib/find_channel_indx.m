function [all_channels,inx] =find_channel_indx(channels,current_channel)
warning('off')
inx=[];
count=0;
for zz=1:length(channels)
    flag=0;
    sensors=strrep(current_channel,' ','');
    sensors=strrep(sensors,'.','');
    sensors=strrep(sensors,'-','');
    sensors=strrep(sensors,'_','');
    sensors=upper(sensors);
    for pp=1:length(current_channel)
        for ss=1:length(channels{zz})
            if any(strcmpi(sensors, channels{zz}{ss}) )
                inx=[inx  find( ismember(sensors,channels{zz}{ss}) )];
                flag=1;
                break
            end
        end
        if flag
            count=count+1;
            break
        end
    end
end
if count==zz && length(inx)==zz
    all_channels=1;
else
    all_channels=0;
end

end