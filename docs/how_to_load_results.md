Use```torch.load(path)```to load results.pth<br>
Just like a Dict.<br>
```python
    results:{
        0:{                                    #dict,index->fold
            'F1':[0.1,0.2...],                 #list,index->epoch
            'err':[0.9,0.8...],                #list,index->epoch
            'loss':[1.1,1.0...],               #list,index->epoch
            'confusion_mat':[
                [[1204  133  763  280]
                 [ 464  150  477  152]
                 [ 768   66 1276  308]
                 [ 159   23  293 2145]],
                 [[2505  251 1322  667]
                 [1010  283  834  353]
                 [1476  174 2448  766]
                 [ 376   46  446 4365]],
                 ......
            ],                                 #list,index->epoch
            'eval_detail':[                    #list,index->epoch
                {
                    'sequences':[],
                    'ture_labels':[],
                    'pre_labels':[]
                },
                {
                    'sequences':[],
                    'ture_labels':[],
                    'pre_labels':[]
                }
                ...
            ], 
            'best_epoch':0                     #int
        }
        1:{

        ...

        }
    }
```

