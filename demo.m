clc
clear all
close all

load 'Data_4m_C';

nfolds = 10; 
ACC=[];SN=[];Spec=[];PE=[];NPV=[];F_score=[];MCC=[];

X=x;
X(isnan(X)) = 0;
X = line_map(X);
KP = 1:1:length(y);
crossval_idx = crossvalind('Kfold',KP,nfolds);

X_Y_test_label=[];
X_Y_dis=[];

for fold=1:nfolds
 
 train_idx = find(crossval_idx~=fold);
 test_idx  = find(crossval_idx==fold);
 
 train_x = X(train_idx,:);
 train_y = y(train_idx,1);
 
 test_x = X(test_idx,:);
 test_y = y(test_idx,1);
 
 [predict_label,score_s] = tsk_fs_keca(train_x,train_y,test_x);

 [ACC_i,SN_i,Spec_i,PE_i,NPV_i,F_score_i,MCC_i] = roc( predict_label,test_y );
 ACC=[ACC,ACC_i];SN=[SN,SN_i];Spec=[Spec,Spec_i];PE=[PE,PE_i];NPV=[NPV,NPV_i];F_score=[F_score,F_score_i];MCC=[MCC,MCC_i];
 acc = length(find(predict_label==test_y))/length(test_y);
 X_Y_test_label=[X_Y_test_label;test_y];
 
 fprintf('- FOLD %d - ACC: %f \n', fold, ACC_i)
 end

mean_acc=mean(ACC)
mean_sn=mean(SN)
mean_sp=mean(Spec)
mean_mcc=mean(MCC)
 
