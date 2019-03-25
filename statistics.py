import numpy as np
import matplotlib.pyplot as plt

def stage(stages):	
    #N3->0  N2->1  N1->2  REM->3  W->4
    stage_cnt=np.array([0,0,0,0,0])
    for i in range(len(stages)):
        stage_cnt[stages[i]] += 1
    stage_cnt_per = stage_cnt/len(stages) 
    return stage_cnt,stage_cnt_per

def result(mat):
    wide=mat.shape[0]
    sub_acc = np.zeros(wide)
    sub_recall = np.zeros(wide)
    err = 0
    for i in range(wide):
        sub_recall[i]=mat[i,i]/np.sum(mat[i])
        err += mat[i,i]
        sub_acc[i] = (np.sum(mat)-((np.sum(mat[i])+np.sum(mat[:,i]))-2*mat[i,i]))/np.sum(mat)
    avg_recall = np.mean(sub_recall)
    avg_acc = np.mean(sub_acc)
    err = 1-err/np.sum(mat)
    return avg_recall,avg_acc,err


def show(plot_result,epoch):
    train_recall = np.array(plot_result['train'])
    test_recall = np.array(plot_result['test'])
    plt.figure('running recall')
    plt.clf()
    train_recall_x = np.linspace(0,epoch,len(train_recall))
    test_recall_x = np.linspace(0,int(epoch),len(test_recall))
    plt.xlabel('Epoch')
    plt.ylabel('%')
    plt.ylim((0,1))
    if epoch <10:
        plt.xlim((0,10))
    else:
        plt.xlim((0,epoch))
    plt.plot(train_recall_x,train_recall,label='train',linewidth = 2.0,color = 'red')
    plt.plot(test_recall_x,test_recall,label='test', linewidth = 2.0,color = 'blue')
    plt.legend(loc=4)
    plt.savefig('./running_recall.png')

    # plt.draw()
    # plt.pause(0.01)


def main():
    plot_result={'train': [0.2303303787268332, 0.2119345588626961, 0.20542007990053074, 0.20353191245282734, 0.2032570804016917, 0.20269640625503033, 0.2020943574651975, 0.2108357726067258, 0.21750990713964172, 0.23142651474994708, 0.2318236991596459, 0.22924187151697578, 0.22830716248841004, 0.2331831179181414, 0.23604422314519158, 0.23734486777406488, 0.23929925551037354, 0.2451802483014293, 0.24753448439761755, 0.24964581836870603, 0.2506097959967858, 0.2497704229822455], 'test': [0.28670433145009416, 0.29533625933982305, 0.2927783086111587, 0.28665535025585603, 0.2884532914652956]}
    show(plot_result,10)
if __name__ == '__main__':
    main()

