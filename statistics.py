import numpy as np
import matplotlib.pyplot as plt
import util

def stage(stages):	
    #N3->0  N2->1  N1->2  REM->3  W->4
    stage_cnt=np.array([0,0,0,0,0])
    for i in range(len(stages)):
        stage_cnt[stages[i]] += 1
    stage_cnt_per = stage_cnt/len(stages) 
    util.writelog('     dataset statistics [S3 S2 S1 R W]: '+str(stage_cnt),True)
    return stage_cnt,stage_cnt_per

def reversal_label(mat):
    new_mat = np.zeros(mat.shape,dtype='int')
    new_mat[0]=mat[4]
    new_mat[1]=mat[2]
    new_mat[2]=mat[1]
    new_mat[3]=mat[0]
    new_mat[4]=mat[3]

    mat=new_mat.copy()

    new_mat[:,0]=mat[:,4]
    new_mat[:,1]=mat[:,2]
    new_mat[:,2]=mat[:,1]
    new_mat[:,3]=mat[:,0]
    new_mat[:,4]=mat[:,3]

    return new_mat

def class_5to4(mat):
    #[W N1 N2 N3 R] to [W N1+N2 N3 R]
    new_mat=np.zeros((4,5),dtype='int')
    new_mat[0] = mat[0]
    new_mat[1] = mat[1]+mat[2]
    new_mat[2] = mat[3]
    new_mat[3] = mat[4]
    mat = new_mat.copy()
    new_mat=np.zeros((4,4),dtype='int')
    new_mat[:,0] = mat[:,0]
    new_mat[:,1] = mat[:,1]+mat[:,2]
    new_mat[:,2] = mat[:,3]
    new_mat[:,3] = mat[:,4]
    return new_mat


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

def result(mat,print_sub=False):
    wide=mat.shape[0]
    sub_acc = np.zeros(wide)
    sub_recall = np.zeros(wide)
    sub_sp = np.zeros(wide)
    err = 0
    for i in range(wide):
        TP = mat[i,i]
        FN = np.sum(mat[i])- mat[i,i]
        TN = (np.sum(mat)-np.sum(mat[i])-np.sum(mat[:,i])+mat[i,i])
        FP = np.sum(mat[:,i]) - mat[i,i]

        err += mat[i,i]
        sub_acc[i]=(TP+TN)/(TP+FN+TN+FP)
        sub_recall[i]=(TP)/np.clip((TP+FN), 1e-5, 1e10) 
        sub_sp[i] = TN/np.clip((TN+FP), 1e-5, 1e10)
    if print_sub == True:
        print('sub_recall:',sub_recall,'\nsub_acc:',sub_acc,'\nsub_sp:',sub_sp)
    avg_recall = np.mean(sub_recall)
    avg_acc = np.mean(sub_acc)
    avg_sp = np.mean(sub_sp)
    err = 1-err/np.sum(mat)
    k = Kappa(mat)
    return round(avg_recall,4),round(avg_acc,4),round(avg_sp,4),round(err,4),round(k, 4)

def stagefrommat(mat):
    wide=mat.shape[0]
    stage_num = np.zeros(wide,dtype='int')
    for i in range(wide):
        stage_num[i]=np.sum(mat[i])
    util.writelog('statistics of dataset [S3 S2 S1 R W]:\n'+str(stage_num),True)

def show(plot_result,epoch):
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
    plt.plot(train_x,train*100,label='train',linewidth = 2.0,color = 'red')
    plt.plot(test_x,test*100,label='test', linewidth = 2.0,color = 'blue')
    plt.legend(loc=1)
    plt.title('Running err.',fontsize='large')
    plt.savefig('./running_err.png')

    # plt.draw()
    # plt.pause(0.01)


def main():
    mat=[[37980,1322,852,2,327],[3922,8784,3545,0,2193],[1756,5136,99564,1091,991],[18,1,7932,4063,14],[1361,1680,465,0,23931]]
    mat = np.array(mat)
    avg_recall,avg_acc,err = result(mat)
    print(avg_recall,avg_acc,err)
if __name__ == '__main__':
    main()

