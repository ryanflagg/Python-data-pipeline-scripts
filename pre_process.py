import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dir = 'AReMv1'

def createLists(dir):
    data_dir = dir
    labels = []
    data = []
    print('concatenating files')
    for label_type in ['bending1', 'bending2', 'cycling', 'lying', 'sitting', 'standing', 'walking']:
        dir_name = os.path.join(data_dir, label_type)
        #print(dir_name)
        for fname in os.listdir(dir_name):
            #print(fname)
            file = os.path.join(dir_name, fname)
            print(file)
            features = list(csv.reader(open(file)))
            #print(np.shape(features))
            float_features = []
            for feature in features:
                #print('.', end='')
                feature_row = []
                for i in feature:
                    i = float(i)
                    feature_row.append(i)
                float_features.append(feature_row)
            data.append(float_features)
            #print(np.shape(data))
            labels.append(label_type)
    return data, labels
 
 
def feature_MAX(data):
    #iterates over a 3d array to find the highest value.
    feature_MAX = 0
    for sample in data:
        for time_step in sample:
            for feature in time_step:
                if feature > feature_MAX:
                    feature_MAX = feature
    return feature_MAX
    
def vectorise_list(data):
    MAX = feature_MAX(data)
    vectorised_data = []
    for sample in data:
        new_sample = []
        for time_step in sample:
            new_timestep = []
            for feature in time_step:
                vfeature = feature/MAX
                new_timestep.append(vfeature)
            new_sample.append(new_timestep)
        vectorised_data.append(new_sample)
    return vectorised_data
    
def tokenise(labels):
    token_list = []
    print('Tokenising labels')
    for label in labels:
        if label == "bending1":
            token_list.append(0)
            print(label + ' = 0')
        elif label == "bending2":
            token_list.append(1)
            print(label + ' = 1')
        elif label == "cycling":
            token_list.append(2)
            print(label + ' = 2')
        elif label == "lying":
            token_list.append(3)
            print(label + ' = 3')
        elif label == "sitting":
            token_list.append(4)
            print(label + ' = 4')
        elif label == "standing":
            token_list.append(5)
            print(label + ' = 5')
        elif label == "walking":
            token_list.append(6)
            print(label + ' = 6')
    return(token_list)
    
def pre_process():
    train_data_list, train_label_list = createLists('AReMv1')
    training_data = vectorise_list(train_data_list)
    training_labels = tokenise(train_label_list)
    data = np.array(training_data)
    print('data shape:')
    print(np.shape(training_data))
    labels = np.array(training_labels)
    print('labels shape:')
    print(np.shape(training_labels))
    return data, labels

def cluster_shuffle(data, labels):

    #after the pre-process function is run, cluster_shuffle will shuffle the data set in clusters and then split the 
    #dataset into train val & test, in a way which ensures an even spread of activities across the splits.
    
    act0_labels = []
    act1_labels = []
    act2_labels = []
    act3_labels = []
    act4_labels = []
    act5_labels = []
    act6_labels = []
    act0_data = []
    act1_data = []
    act2_data = []
    act3_data = []
    act4_data = []
    act5_data = []
    act6_data = []
    
    n = 0
    
    for label in labels:
        if n < len(labels):
            if label == 0:
                act0_labels.append(label) 
                act_data = data[n]
                act0_data.append(act_data)
            
         
            if label == 1:
                act1_labels.append(label)
                act_data = data[n]
                act1_data.append(act_data)
                      

            if label == 2:
                act2_labels.append(label)
                act_data = data[n]
                act2_data.append(act_data)
                     

            if label == 3:
                act3_labels.append(label)
                act_data = data[n]
                act3_data.append(act_data)
                   

            if label == 4:
                act4_labels.append(label)
                act_data = data[n]
                act4_data.append(act_data)
                        

            if label == 5:
                act5_labels.append(label)
                act_data = data[n]
                act5_data.append(act_data)
                        

            if label == 6:
                act6_labels.append(label)
                act_data = data[n]
                act6_data.append(act_data)

        n = n + 1            
            
    # CONVERT THE LISTS TO ARRAYS
    print('SHUFFLING ARRAYS') 
    
    act0_data = np.asarray(act0_data)
    act0_labels  = np.asarray(act0_labels)
    idx = np.random.permutation(len(act0_labels))
    act0_data, act0_labels = act0_data[idx], act0_labels[idx]
    print('act0 shape: ', np.shape(act0_data))
    
    act1_data = np.asarray(act1_data)
    act1_labels  = np.asarray(act1_labels)    
    idx = np.random.permutation(len(act1_labels))
    act1_data, act1_labels = act1_data[idx], act1_labels[idx]
    print('act1 shape: ', np.shape(act1_data))
    
    act2_data = np.asarray(act2_data)
    act2_labels  = np.asarray(act2_labels)
    idx = np.random.permutation(len(act2_labels))
    act2_data, act2_labels = act2_data[idx], act2_labels[idx]
    print('act2 shape: ', np.shape(act2_data))
    
    act3_data = np.asarray(act3_data)
    act3_labels  = np.asarray(act3_labels)    
    idx = np.random.permutation(len(act3_labels))
    act3_data, act3_labels = act3_data[idx], act3_labels[idx]
    print('act3 shape: ', np.shape(act3_data))
    
    act4_data = np.asarray(act4_data)
    act4_labels  = np.asarray(act4_labels)    
    idx = np.random.permutation(len(act4_labels))
    act4_data, act4_labels = act4_data[idx], act4_labels[idx]
    print('act4 shape: ', np.shape(act4_data))
    
    act5_data = np.asarray(act5_data)
    act5_labels  = np.asarray(act5_labels)    
    idx = np.random.permutation(len(act5_labels))
    act5_data, act5_labels = act5_data[idx], act5_labels[idx]
    print('act5 shape: ', np.shape(act5_data))
    
    act6_data = np.asarray(act6_data)
    act6_labels  = np.asarray(act6_labels)    
    idx = np.random.permutation(len(act6_labels))
    act6_data, act6_labels = act6_data[idx], act6_labels[idx]
    print('act6 shape: ', np.shape(act6_data))
    
    # the splits act0 = 4-2-1. act1 = 3-1-1 act2-6 = 9-3-3
    print('SLICING ARRAYS')
    #ACT0 splits
    act0_train_lab = act0_labels[:4]
    act0_val_lab = act0_labels[4:6]
    act0_test_lab = act0_labels[[6],]
    act0_train_data = act0_data[:4]
    print(np.shape(act0_train_data))    
    act0_val_data = act0_data[4:6]
    #print(act0_val_data)
    print(np.shape(act0_val_data))
    act0_test_data = act0_data[[6],]
    print(np.shape(act0_test_data))    

    #ACT1 splits
    act1_train_lab = act1_labels[:3]
    act1_val_lab = act1_labels[[3],]
    act1_test_lab = act1_labels[[4],]
    act1_train_data = act1_data[:3]
    act1_val_data = act1_data[[3],]
    #print(act1_val_data)
    #print(np.shape(act1_val_data))
    act1_test_data = act1_data[[4],]    
    
     #ACT2 splits
    act2_train_lab = act2_labels[:9]
    act2_val_lab = act2_labels[9:12]
    act2_test_lab = act2_labels[12:]
    act2_train_data = act2_data[:9]
    act2_val_data = act2_data[9:12]
    act2_test_data = act2_data[12:]   
    
     #ACT3 splits
    act3_train_lab = act3_labels[:9]
    act3_val_lab = act3_labels[9:12]
    act3_test_lab = act3_labels[12:]
    act3_train_data = act3_data[:9]
    act3_val_data = act3_data[9:12]
    act3_test_data = act3_data[12:]  

     #ACT4 splits
    act4_train_lab = act4_labels[:9]
    act4_val_lab = act4_labels[9:12]
    act4_test_lab = act4_labels[12:]
    act4_train_data = act4_data[:9]
    act4_val_data = act4_data[9:12]
    act4_test_data = act4_data[12:] 

     #ACT5 splits
    act5_train_lab = act5_labels[:9]
    act5_val_lab = act5_labels[9:12]
    act5_test_lab = act5_labels[12:]
    act5_train_data = act5_data[:9]
    act5_val_data = act5_data[9:12]
    act5_test_data = act5_data[12:] 

     #ACT6 splits
    act6_train_lab = act6_labels[:9]
    act6_val_lab = act6_labels[9:12]
    act6_test_lab = act6_labels[12:]
    act6_train_data = act6_data[:9]
    act6_val_data = act6_data[9:12]
    act6_test_data = act6_data[12:]  

    # USE CONCAT TO JOIN ARRAYS

    train_data = np.concatenate((act0_train_data, act1_train_data, act2_train_data, act3_train_data, act4_train_data, act5_train_data, act6_train_data))
    train_labels = np.concatenate((act0_train_lab, act1_train_lab, act2_train_lab, act3_train_lab, act4_train_lab, act5_train_lab, act6_train_lab))
    val_data = np.concatenate((act0_val_data, act1_val_data, act2_val_data, act3_val_data, act4_val_data, act5_val_data, act6_val_data))
    val_labels = np.concatenate((act0_val_lab, act1_val_lab, act2_val_lab, act3_val_lab, act4_val_lab, act5_val_lab, act6_val_lab))
    test_data = np.concatenate((act0_test_data, act1_test_data, act2_test_data, act3_test_data, act4_test_data, act5_test_data, act6_test_data))
    test_labels = np.concatenate((act0_test_lab, act1_test_lab, act2_test_lab, act3_test_lab, act4_test_lab, act5_test_lab, act6_test_lab))
    
    return train_data, train_labels, val_data, val_labels, test_data, test_labels
    
def cluster_shuffle_2splits(data, labels):

    #after the pre-process function is run, cluster_shuffle will shuffle the data set in clusters and then split the 
    #dataset into train val & test, in a way which ensures an even spread of activities across the splits.
    
    act0_labels = []
    act1_labels = []
    act2_labels = []
    act3_labels = []
    act4_labels = []
    act5_labels = []
    act6_labels = []
    act0_data = []
    act1_data = []
    act2_data = []
    act3_data = []
    act4_data = []
    act5_data = []
    act6_data = []
    
    n = 0
    
    for label in labels:
        if n < len(labels):
            if label == 0:
                act0_labels.append(label) 
                act_data = data[n]
                act0_data.append(act_data)
            
         
            if label == 1:
                act1_labels.append(label)
                act_data = data[n]
                act1_data.append(act_data)
                      

            if label == 2:
                act2_labels.append(label)
                act_data = data[n]
                act2_data.append(act_data)
                     

            if label == 3:
                act3_labels.append(label)
                act_data = data[n]
                act3_data.append(act_data)
                   

            if label == 4:
                act4_labels.append(label)
                act_data = data[n]
                act4_data.append(act_data)
                        

            if label == 5:
                act5_labels.append(label)
                act_data = data[n]
                act5_data.append(act_data)
                        

            if label == 6:
                act6_labels.append(label)
                act_data = data[n]
                act6_data.append(act_data)

        n = n + 1            
            
    # CONVERT THE LISTS TO ARRAYS
    print('SHUFFLING ARRAYS') 
    
    act0_data = np.asarray(act0_data)
    act0_labels  = np.asarray(act0_labels)
    idx = np.random.permutation(len(act0_labels))
    act0_data, act0_labels = act0_data[idx], act0_labels[idx]
    print('act0 shape: ', np.shape(act0_data))
    
    act1_data = np.asarray(act1_data)
    act1_labels  = np.asarray(act1_labels)    
    idx = np.random.permutation(len(act1_labels))
    act1_data, act1_labels = act1_data[idx], act1_labels[idx]
    print('act1 shape: ', np.shape(act1_data))
    
    act2_data = np.asarray(act2_data)
    act2_labels  = np.asarray(act2_labels)
    idx = np.random.permutation(len(act2_labels))
    act2_data, act2_labels = act2_data[idx], act2_labels[idx]
    print('act2 shape: ', np.shape(act2_data))
    
    act3_data = np.asarray(act3_data)
    act3_labels  = np.asarray(act3_labels)    
    idx = np.random.permutation(len(act3_labels))
    act3_data, act3_labels = act3_data[idx], act3_labels[idx]
    print('act3 shape: ', np.shape(act3_data))
    
    act4_data = np.asarray(act4_data)
    act4_labels  = np.asarray(act4_labels)    
    idx = np.random.permutation(len(act4_labels))
    act4_data, act4_labels = act4_data[idx], act4_labels[idx]
    print('act4 shape: ', np.shape(act4_data))
    
    act5_data = np.asarray(act5_data)
    act5_labels  = np.asarray(act5_labels)    
    idx = np.random.permutation(len(act5_labels))
    act5_data, act5_labels = act5_data[idx], act5_labels[idx]
    print('act5 shape: ', np.shape(act5_data))
    
    act6_data = np.asarray(act6_data)
    act6_labels  = np.asarray(act6_labels)    
    idx = np.random.permutation(len(act6_labels))
    act6_data, act6_labels = act6_data[idx], act6_labels[idx]
    print('act6 shape: ', np.shape(act6_data))
    
    # the splits act0 = 5-2. act1 = 3-2 act2-6 = 12-3
    print('SLICING ARRAYS')
    #ACT0 splits
    act0_train_lab = act0_labels[:6]
    act0_test_lab = act0_labels[[6],]
    act0_train_data = act0_data[:6]   
    act0_test_data = act0_data[[6],]
    

    #ACT1 splits
    act1_train_lab = act1_labels[:4]
    act1_test_lab = act1_labels[[4],]
    act1_train_data = act1_data[:4]
    act1_test_data = act1_data[[4],]    
    
     #ACT2 splits
    act2_train_lab = act2_labels[:12]
    act2_test_lab = act2_labels[12:]
    act2_train_data = act2_data[:12]
    act2_test_data = act2_data[12:]   
    
     #ACT3 splits
    act3_train_lab = act3_labels[:12]
    act3_test_lab = act3_labels[12:]
    act3_train_data = act3_data[:12]
    act3_test_data = act3_data[12:]  

     #ACT4 splits
    act4_train_lab = act4_labels[:12]
    act4_test_lab = act4_labels[12:]
    act4_train_data = act4_data[:12]
    act4_test_data = act4_data[12:] 

     #ACT5 splits
    act5_train_lab = act5_labels[:13]
    act5_test_lab = act5_labels[13:]
    act5_train_data = act5_data[:13]
    act5_test_data = act5_data[13:] 

     #ACT6 splits
    act6_train_lab = act6_labels[:13]
    act6_test_lab = act6_labels[13:]
    act6_train_data = act6_data[:13]
    act6_test_data = act6_data[13:]  

    # USE CONCAT TO JOIN ARRAYS

    train_data = np.concatenate((act0_train_data, act1_train_data, act2_train_data, act3_train_data, act4_train_data, act5_train_data, act6_train_data))
    train_labels = np.concatenate((act0_train_lab, act1_train_lab, act2_train_lab, act3_train_lab, act4_train_lab, act5_train_lab, act6_train_lab))
    test_data = np.concatenate((act0_test_data, act1_test_data, act2_test_data, act3_test_data, act4_test_data, act5_test_data, act6_test_data))
    test_labels = np.concatenate((act0_test_lab, act1_test_lab, act2_test_lab, act3_test_lab, act4_test_lab, act5_test_lab, act6_test_lab))
    
    return train_data, train_labels, test_data, test_labels

def shuffle(data, labels):    
    idx = np.random.permutation(len(labels))
    shuf_data, shuf_labels = data[idx], labels[idx]
    return shuf_data, shuf_labels
    
def shuffle_split(data, labels):
    training_data, training_labels = shuffle(data, labels)
    t_data = training_data[:49]
    t_labels = training_labels[:49]
    v_data = training_data[50:69]
    v_labels = training_labels[50:69]
    test_data = training_data[70:]
    test_labels = training_labels[70:]
    return t_data, t_labels, v_data, v_labels, test_data, test_labels
    
def plot_loss(hist_obj):
    plt.plot(hist_obj.history['loss'])
    plt.plot(hist_obj.history['val_loss'])
    plt.ylabel('loss')
    plt.xlabel('Epoch')
    plt.legend(['loss','validation loss'], loc='upper right')
    plt.show()
    
def plot_accuracy(hist_obj):
    plt.plot(hist_obj.history['accuracy'])
    plt.plot(hist_obj.history['val_accuracy'])
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['Accuracy','Validation'], loc='best')
    plt.show()
    
def plot_hist(hist_obj):
    plt.plot(hist_obj.history['accuracy'])
    plt.plot(hist_obj.history['val_accuracy'])
    plt.plot(hist_obj.history['loss'])
    plt.plot(hist_obj.history['val_loss'])
    plt.ylabel('accuracy/loss')
    plt.xlabel('epoch')
    plt.legend(['train_acc','val_acc','train_loss','val_loss'], loc='best')
    plt.show()
    
def results_to_df(results):
    df = pd.DataFrame.from_records(results)
    df = df.sort_values(by=[0,1], ascending=True)
    df = df.rename(columns={2:'Test_Loss',3:'Test_Accuracy'})
    return df