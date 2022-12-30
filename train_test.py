import tensorflow as tf
import h5py
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import random
import numpy as np
import pandas as pd
import glob
import joblib
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
import os
from lightgbm import LGBMClassifier  
from sklearn.metrics import accuracy_score,confusion_matrix,recall_score,precision_score
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
from tensorflow.python.platform import tf_logging as logging
np.random.seed(42)
random.seed(42)

class ReduceLRBacktrack(ReduceLROnPlateau):
    def __init__(self, best_path, *args, **kwargs):
        super(ReduceLRBacktrack, self).__init__(*args, **kwargs)
        self.best_path = best_path
    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            logging.warning('Reduce LR on plateau conditioned on metric `%s` '
                            'which is not available. Available metrics are: %s',
                             self.monitor, ','.join(list(logs.keys())))
        if not self.monitor_op(current, self.best):
            if not self.in_cooldown(): 
                if self.wait+1 >= self.patience: 
                    # load best model so far
                    print("Backtracking to best model before reducting LR")
                    self.model.load_weights(self.best_path)

        super().on_epoch_end(epoch, logs) # actually reduce LR
        
        
def generator(files,batch_size,chan,flag): 
    last_file=1
    while True:
        if not(last_file):          
            for tt in range(files.size):
                if files.size==1:
                    current_file=files
                else:
                    current_file=files[tt]
                try:
                    with h5py.File(current_file, 'r') as hf:

                            y=hf["y"][:,0]
                            y[y>0]=1                   
                            y = to_categorical(y[:])
                            y= tf.convert_to_tensor( y ) 
                            x= tf.convert_to_tensor(hf["x"+(chan)][:,:] )
      
                            if len(y)<batch_size:
                                yield x[:], y[:] 
                            else:
                                for ii in range( round(len(y)/batch_size) + 1):
                                    if (ii+1)*batch_size >= len(y):
                                        yield  x[ii*batch_size:], y[ii*batch_size:] 
                                        break
                                    else:
                                        yield x[ii*batch_size:(ii+1)*batch_size], y[ii*batch_size:(ii+1)*batch_size] 
                except:
                    pass
            last_file=1
        else:
            
            if flag==0 and not files.size==1:
                random.shuffle(files)
            last_file=0 
      

def get_model(train_f,tw,chan):
    if train_f.size >1:
        train_f=train_f[0]
    with h5py.File(train_f, 'r') as hf:
        x=hf["x" + (chan)][:]
    dim=x.shape[1]
    model = tf.keras.applications.EfficientNetV2S(
        include_top=False,
        weights=None,
        input_tensor=tf.keras.layers.Input(shape=(dim, 1,1))
    )
    output = model.layers[-1].output
    output= tf.keras.layers.GlobalAveragePooling2D()(output)
    output= tf.keras.layers.Dropout(0.2)(output)
    output=tf.keras.layers.Dense(2, activation="softmax")(output)
    model_out = tf.keras.models.Model(inputs=model.input, outputs=output) 
    model_out.compile(Adam(learning_rate=lr), loss='categorical_crossentropy',metrics=['accuracy'])
    return model_out,callbacks_list


def extract_features(seg,files,src_o,model1,model2,model3,model4,model5,train_f,val_f,test_f,dim): 
    if seg ==1:
        files=train_f
        f1 = h5py.File(src_o+'train.hdf5', mode='w') 

    if seg ==2:
        f1 = h5py.File(src_o+'test.hdf5', mode='w') 
        files=test_f
        
    if seg ==3:
        f1 = h5py.File(src_o+'val.hdf5', mode='w') 
        files=val_f      
    f1.create_dataset("x", (1, dim*5 ), np.float32,maxshape=(None,dim*5))
    f1.create_dataset("y", (1,2), np.float32,maxshape=(None,2))
    indx=0
    count_p=0
    n_feat=5
    for tt in range(files.size):
        if files.size==1:
            current_file=files
        else:
            current_file=files[tt]
        with h5py.File(current_file, 'r') as hf:
            y=hf["y"][:,0]
            y[y>0]=1
            y = to_categorical(y[:])
            feats=np.empty(shape=[len(y),dim*5])
            for lll in range(1,n_feat+1):
                x= tf.convert_to_tensor(hf["x"+str(lll)][:,:] )
                cnn_output=np.array( eval("model"+str(lll)+".predict(x)")   )
                feats[list(range(len(y))), (lll-1)*dim:lll*dim] = cnn_output[:]
            kk = list(range(indx, indx + len(y)))    
            f1["x"].resize(indx+ len(y),axis=0)
            f1["y"].resize(indx+ len(y),axis=0)
            f1["x"][kk, ...] = feats[:]
            f1["y"][kk, ...] = y[:]
            indx += len(y)
            count_p+=1
            if files.size==1:
                break
    return

if __name__ == '__main__':
## TRAIN NETWROKS
    # # Parameters
    batch_size=1
    tw=60
    epochs=3
    lr=1e-3
    train_s=100
    val_s =20
    src="./"
    src_d=src+"data/"
    src_w=src+"/weights/"
    
    # Files 
    files=np.sort(np.array( glob.glob(src_d+'*.hdf5')))
    train_f=files[0]
    val_f=files[1]
    test_f=files[2]
        
        
    for chan in list(['1','2','3','4','5']):
        try: 
            os.mkdir(src_w) 
        except: 
            pass
        reduce_lr = ReduceLRBacktrack(best_path=src_w+"w"+"_"+(chan)+".hdf5", monitor='val_loss', factor=0.1, patience=6, min_lr=1e-6) 
        checkpoint = ModelCheckpoint(src_w+"w"+"_"+(chan)+".hdf5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [reduce_lr,checkpoint] 
        
        model, callbacks_list= get_model(train_f,tw,chan )
    
        history_train=model.fit(
            generator(train_f,batch_size,chan,0), steps_per_epoch=train_s,
            epochs=epochs,
            shuffle=False,
            validation_data=generator(val_f,batch_size,chan,1), validation_steps=val_s,
            callbacks=callbacks_list,
            verbose=1)

## EXTRACT FEATURES

    src_o=src+"cnn_feat/"
    try:
        os.mkdir(src_o)    
    except:
        pass  

    # Channel 1
    model_name =src_w+"w_1.hdf5"
    model = tf.keras.models.load_model(model_name)
    model1= Model(inputs=model.input, outputs=model.layers[-3].output)
    # Channel 2
    model_name =src_w+"w_2.hdf5"
    model = tf.keras.models.load_model(model_name)
    model2= Model(inputs=model.input, outputs=model.layers[-3].output)   
    # Channel 3
    model_name =src_w+"w_3.hdf5"
    model = tf.keras.models.load_model(model_name)
    model3= Model(inputs=model.input, outputs=model.layers[-3].output)    
    # Channel 4
    model_name =src_w+"w_4.hdf5"
    model = tf.keras.models.load_model(model_name)
    model4= Model(inputs=model.input, outputs=model.layers[-3].output)   
    # Channel 5
    model_name =src_w+"w_5.hdf5"
    model = tf.keras.models.load_model(model_name)
    model5= Model(inputs=model.input, outputs=model.layers[-3].output)
    dim=model5.layers[-3].output_shape[-1]
    
    for seg in range(1,4):
        extract_features(seg,files,src_o,model1,model2,model3,model4,model5,train_f,val_f,test_f,dim)

## TRAIN CLASSIFIER

    # channel 1 ECG
    # channel 2 ABDOMINAL
    # channel 3 CHEST
    # channel 4 NASSAL
    # channel 5 SPO2   

    sens=list([[1,  2,  3,  4,  5],
                [2,5]])

    for current_sens in sens:
        sen_names=str(current_sens).replace("  ", "")
        sen_names=sen_names.replace(" ", "")
        sen_names=sen_names.replace("[", "_")
        sen_names=sen_names.replace("]", "_")
        sen_names=sen_names.replace(",", "_")
        
        inx=np.array([] ,dtype=int)
        
        for l in current_sens:
            inx= np.append(inx, list(range(  (l-1) * dim, l * dim  )) )  

            # # Get Data 
            data=src_o+'train.hdf5'
            with h5py.File(data, 'r') as hf: 
                x_train = pd.DataFrame(hf["x"][:,inx] )
                y_train = pd.Series(hf["y"][:,0])  
            data=src_o+'test.hdf5'
            with h5py.File(data, 'r') as hf:           
                x_test = pd.DataFrame(hf["x"][:,inx] )
                y_test = pd.Series(hf["y"][:,0] )  
            data=src_o+'val.hdf5'
            with h5py.File(data, 'r') as hf:           
                x_val = pd.DataFrame(hf["x"][:,inx] )
                y_val = pd.Series(hf["y"][:,0] )  
                
        # LGBM 
        model = LGBMClassifier(class_weight='balanced', colsample_bytree=0.72, first_metric_only = True,
                                                  min_child_samples=221, min_child_weight=10.0,
                                                  n_estimators= 610 , num_leaves=39, random_state=14, reg_alpha=0,
                                                  reg_lambda=0.1, subsample=0.99,importance_type='gain',objective='binary', metric =None )        
        # train
        model.fit(x_train, y_train , eval_metric='aucpr' , eval_set=[(x_val, y_val)] )   
        # test
        y_pred=model.predict(x_test)
        
        # Metrics            
        accuracy =accuracy_score(y_test,y_pred)
        recall=recall_score(y_test,y_pred )
        precision=precision_score(y_test,y_pred )
        confusion=confusion_matrix(y_test,y_pred)      

        # save results
        with open(src_o+"LGB_OUT_"+str(sen_names)+'.txt', 'a') as file:
            file.write("Confusion_=\n")
            file.write(str(confusion))
            file.write("\nAccu="+str(accuracy))
            file.write("\nRecall="+str(recall))
            file.write("\nPrecision="+str(precision))      
        
        #save model
        joblib.dump(model, src_o+"LGBD_MODEL_"+str(sen_names)+'.pkl')
        # # load model
        # model = joblib.load( src_o+"LGB_MODEL_f"+str(fold)+'_'+str(sen_names)+'.pkl')
        