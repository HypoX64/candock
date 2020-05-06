import numpy as np
import matplotlib.pyplot as plt
import util
import os
from . import heatmap

def label_statistics(labels):
    #for sleep label: N3->0  N2->1  N1->2  REM->3  W->4
    s = set()
    for label in labels:
        s.add(label)
    label_num = len(list(s))

    label_cnt=np.zeros(label_num,dtype=np.int64)
    for i in range(len(labels)):
        label_cnt[labels[i]] += 1
    label_cnt_per = label_cnt/len(labels)
    return label_cnt,label_cnt_per,label_num

def mat2predtrue(mat):
    y_pred = [];y_true = []
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            for x in range(mat[i][j]):
                y_true.append(i)
                y_pred.append(j)
    return y_true,y_pred

def predtrue2mat(y_true,y_pred):
    label_num = label_statistics(y_true)[2]
    mat = np.zeros((label_num,label_num), dtype=np.int64)
    for i in range(len(y_true)):
        mat[y_true[i]][y_pred[i]] +=1
    return mat

def mergemat(mat,mergemethod):
    y_true,y_pred = mat2predtrue(mat)
    new_true = np.zeros(len(y_true), dtype=np.int64)
    new_pred = np.zeros(len(y_true), dtype=np.int64)
    for i in range(len(y_true)):
        for j in range(len(mergemethod)):
            if y_true[i] in mergemethod[j]:
                new_true[i]=j
            if y_pred[i] in mergemethod[j]:
                new_pred[i]=j
    return predtrue2mat(new_true, new_pred)

def Kappa(mat):
    mat=mat/10000 # avoid overflow
    mat_length=np.sum(mat)
    wide=mat.shape[0]
    po=0.0;pe=0.0
    for i in range(wide):
        po=po+mat[i][i]
        pe=pe+np.sum(mat[:,i])*np.sum(mat[i,:])
    po=po/mat_length
    pe=pe/(mat_length*mat_length)
    k=(po-pe)/(1-pe)
    return k

def report(mat,print_sub=False):
    wide=mat.shape[0]
    sub_recall = np.zeros(wide)
    sub_precision = np.zeros(wide)
    sub_F1 = np.zeros(wide)
    sub_acc = np.zeros(wide)
    _err = 0

    for i in range(wide):
        TP = mat[i,i]
        FN = np.sum(mat[i])- mat[i,i]
        TN = (np.sum(mat)-np.sum(mat[i])-np.sum(mat[:,i])+mat[i,i])
        FP = np.sum(mat[:,i]) - mat[i,i]

        _err += mat[i,i]
        sub_acc[i]=(TP+TN)/(TP+FN+TN+FP)
        sub_precision[i] = TP/np.clip((TP+FP), 1e-5, 1e10)
        sub_recall[i]=(TP)/np.clip((TP+FN), 1e-5, 1e10) 
        #F1 score = 2 * P * R / (P + R)
        sub_F1[i] = 2*sub_precision[i]*sub_recall[i] / np.clip((sub_precision[i]+sub_recall[i]),1e-5,1e10)

    if print_sub == True:
        print('sub_recall:',sub_recall,'\nsub_acc:',sub_acc,'\nsub_sp:',sub_sp)

    err = 1-_err/np.sum(mat)
    Macro_precision = np.mean(sub_precision)
    Macro_recall = np.mean(sub_recall)
    Macro_F1 = np.mean(sub_F1)
    Macro_acc = np.mean(sub_acc)

    k = Kappa(mat)
    return round(Macro_precision,4),round(Macro_recall,4),round(Macro_F1,4),round(err,4),round(k, 4)

def plotloss(plot_result,epoch,opt):
    train = np.array(plot_result['train'])
    test = np.array(plot_result['test'])
    plt.figure('running recall')
    plt.clf()
    train_x = np.linspace(0,epoch,len(train))
    test_x = np.linspace(0,int(epoch),len(test))
    plt.xlabel('Epoch')
    plt.ylabel('%')
    plt.ylim((0,100))
    if epoch <10:
        plt.xlim((0,10))
    else:
        plt.xlim((0,epoch))
    plt.plot(train_x,train*100,label='train',linewidth = 1.5,color = 'red')
    plt.plot(test_x,test*100,label='test', linewidth = 1.5,color = 'blue')
    plt.legend(loc=1)
    plt.title('Running err.',fontsize='large')
    plt.savefig(os.path.join(opt.save_dir,'running_err.png'))


def statistics(mat,opt,logname,heatmapname):
    util.writelog('------------------------------ '+logname+' result ------------------------------',opt,True)
    util.writelog(logname+' -> macro-prec,reca,F1,err,kappa: '+str(report(mat)),opt,True)
    util.writelog('confusion_mat:\n'+str(mat)+'\n',opt,True)
    heatmap.draw(mat,opt,name = heatmapname)


def main():
    mat=[[37980,1322,852,2,327],[3922,8784,3545,0,2193],[1756,5136,99564,1091,991],[18,1,7932,4063,14],[1361,1680,465,0,23931]]
    mat = np.array(mat)
    avg_recall,avg_acc,avg_sp,err,kappa = result(mat)
    print(avg_recall,avg_acc,avg_sp,err,kappa)
if __name__ == '__main__':
    main()