function [] = gen_dat(ch, signal,outputf,tw,fs,dim,label,normal)
%% This script generates the data for the CNN
warning('off')
ini=1;
ini_n=1;
ini_data=tw;
stride=0.9;
flagg=1;
N=floor(length(label));
output_vec=[];

%% LABELS
% {'obstapnea','oa','obsthypopnea'};%           1
% {'centralapnea','ca','centralhypopnea'}; %    2
% {'hypopnea',}; %                              3
% {'mixedapnea','ma'};%                         4
% {'label'}; %                                  5
% {'snorea'}; %

while(flagg)
    current_label=label( ini_data:ini_data+tw );
    if sum(current_label)>0
        lab=mode(current_label(current_label>0));
        if isnan(lab)
            lab=0;
        end
        if lab>1
            % OSA sample
            x=( signal(ini_data*fs: ini_data*fs + tw*fs -1) );
            x=imresize(x', [1,dim*3], "bilinear" );
            info = h5info(outputf, "/x"+num2str(ch));
            curSize = info.Dataspace.Size;
            h5write(outputf,"/x"+num2str(ch), x,[ 1 1 curSize(end)+1] , [1  dim*3 1]);
            delete(gca);
            close;
            ini=ini+1;

            if ch ==5
                h5write(outputf,'/y', lab,[1 curSize(end)+1] , [1 1]);
            end

            % Normal sample
            flag_n=1;
            while flag_n
                idn=normal(ini_n);
                x=( signal(idn*fs: idn*fs + tw*fs -1)  );
                x=imresize(x', [1,dim*3], "bilinear" );
                info = h5info(outputf, "/x"+num2str(ch));
                curSize = info.Dataspace.Size;
                h5write(outputf,"/x"+num2str(ch), x,[ 1 1 curSize(end)+1] , [1  dim*3 1]);
                ini=ini+1;
                if ch ==5
                    h5write(outputf,'/y', 0,[1 curSize(end)+1] , [1 1]);
                end
                flag_n=0;
                ini_n=ini_n+1;
            end
        end
    end
    ini_data=ini_data+round(stride*tw);
    a=ini_data + tw;

    if ( a>=N)
        flagg=0;
    end
end
end