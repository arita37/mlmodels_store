## Original File URL: https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_dataloader.py


### Error 1, [Traceback at line 104](https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_dataloader.py#L104)<br />104..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/1f9d17efb377b015a330ae693d8b5624cdd08eef/mlmodels/dataloader.py", line 445, in test_json_list
<br />    loader.compute()
<br />  File "https://github.com/arita37/mlmodels/tree/1f9d17efb377b015a330ae693d8b5624cdd08eef/mlmodels/dataloader.py", line 253, in compute
<br />    log("cls_name :", cls_name, flush=True)
<br />TypeError: log() got an unexpected keyword argument 'flush'



### Error 2, [Traceback at line 110](https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_dataloader.py#L110)<br />110..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/1f9d17efb377b015a330ae693d8b5624cdd08eef/mlmodels/dataloader.py", line 445, in test_json_list
<br />    loader.compute()
<br />  File "https://github.com/arita37/mlmodels/tree/1f9d17efb377b015a330ae693d8b5624cdd08eef/mlmodels/dataloader.py", line 253, in compute
<br />    log("cls_name :", cls_name, flush=True)
<br />TypeError: log() got an unexpected keyword argument 'flush'



### Error 3, [Traceback at line 119](https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_dataloader.py#L119)<br />119..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/1f9d17efb377b015a330ae693d8b5624cdd08eef/mlmodels/dataloader.py", line 510, in <module>
<br />    main()
<br />  File "https://github.com/arita37/mlmodels/tree/1f9d17efb377b015a330ae693d8b5624cdd08eef/mlmodels/dataloader.py", line 498, in main
<br />    test_dataloader('dataset/json/refactor/')   
<br />  File "https://github.com/arita37/mlmodels/tree/1f9d17efb377b015a330ae693d8b5624cdd08eef/mlmodels/dataloader.py", line 415, in test_dataloader
<br />    test_json_list(data_pars_list)
<br />  File "https://github.com/arita37/mlmodels/tree/1f9d17efb377b015a330ae693d8b5624cdd08eef/mlmodels/dataloader.py", line 453, in test_json_list
<br />    log("Error", f,  e, flush=True)    
<br />TypeError: log() got an unexpected keyword argument 'flush'
