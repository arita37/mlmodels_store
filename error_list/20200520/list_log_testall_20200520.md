## Original File URL: https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall.py


### Error 1, [Traceback at line 39](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall.py#L39)<br />39..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/model_keras//keras_gan.py", line 31, in <module>
<br />    'AAE' : kg.aae.aae,
<br />AttributeError: module 'mlmodels.model_keras.raw.keras_gan' has no attribute 'aae'



### Error 2, [Traceback at line 85](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall.py#L85)<br />85..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/model_keras//nbeats.py", line 315, in <module>
<br />    test(pars_choice="test01")
<br />  File "https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/model_keras//nbeats.py", line 278, in test
<br />    Xtuple = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/model_keras//nbeats.py", line 172, in get_dataset
<br />    train_data = Data(data_source= path_norm( data_pars["train_data_source"]) ,
<br />NameError: name 'Data' is not defined



### Error 3, [Traceback at line 5055](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall.py#L5055)<br />5055..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/model_keras//01_deepctr.py", line 541, in <module>
<br />    test(pars_choice=5, **{"model_name": model_name})
<br />  File "https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/model_keras//01_deepctr.py", line 517, in test
<br />    module, model = module_load_full("model_keras.01_deepctr", model_pars, data_pars, compute_pars, dataset=dataset)
<br />  File "https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/models.py", line 101, in module_load_full
<br />    model = module.Model(model_pars=model_pars, data_pars=data_pars, compute_pars=compute_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/model_keras/01_deepctr.py", line 155, in __init__
<br />    self.model = modeli(feature_columns, **MODEL_PARAMS[model_name])
<br />TypeError: PNN() got an unexpected keyword argument 'embedding_size'



### Error 4, [Traceback at line 5112](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall.py#L5112)<br />5112..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/model_keras//textvae.py", line 356, in <module>
<br />    test(pars_choice="test01")
<br />  File "https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/model_keras//textvae.py", line 327, in test
<br />    xtuple = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/model_keras//textvae.py", line 269, in get_dataset
<br />    with codecs.open(data_pars["train_data_path"], encoding='utf-8') as f:
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/codecs.py", line 897, in open
<br />    file = builtins.open(filename, mode, buffering)
<br />FileNotFoundError: [Errno 2] No such file or directory: 'https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/dataset/text/quora/train.csv'



### Error 5, [Traceback at line 5161](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall.py#L5161)<br />5161..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/model_keras//namentity_crm_bilstm_dataloader.py", line 306, in <module>
<br />    test_module(model_uri=MODEL_URI, param_pars=param_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/models.py", line 257, in test_module
<br />    model_pars, data_pars, compute_pars, out_pars = module.get_params(param_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/model_keras/namentity_crm_bilstm_dataloader.py", line 197, in get_params
<br />    cf = json.load(open(data_path, mode="r"))
<br />FileNotFoundError: [Errno 2] No such file or directory: 'https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/dataset/json/refactor/namentity_crm_bilstm_dataloader.json'



### Error 6, [Traceback at line 5202](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall.py#L5202)<br />5202..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/model_keras//Autokeras.py", line 12, in <module>
<br />    import autokeras as ak
<br />ModuleNotFoundError: No module named 'autokeras'



### Error 7, [Traceback at line 5342](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall.py#L5342)<br />5342..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/model_keras//charcnn_zhang.py", line 284, in <module>
<br />    test(pars_choice="json", data_path= f"{root_path}/model_keras/charcnn_zhang.json")
<br />  File "https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/model_keras//charcnn_zhang.py", line 268, in test
<br />    model2 = load(out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/model_keras//charcnn_zhang.py", line 118, in load
<br />    model = load_keras(load_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/util.py", line 602, in load_keras
<br />    model.model = load_model(path_file)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/keras/saving/save.py", line 146, in load_model
<br />    loader_impl.parse_saved_model(filepath)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/saved_model/loader_impl.py", line 83, in parse_saved_model
<br />    constants.SAVED_MODEL_FILENAME_PB))
<br />OSError: SavedModel file does not exist at: ztest/ml_keras/charcnn_zhang//model.h5/{saved_model.pbtxt|saved_model.pb}



### Error 8, [Traceback at line 5397](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall.py#L5397)<br />5397..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/model_keras//charcnn.py", line 357, in <module>
<br />    test(pars_choice="test01")
<br />  File "https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/model_keras//charcnn.py", line 320, in test
<br />    Xtuple = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/model_keras//charcnn.py", line 216, in get_dataset
<br />    if data_pars['type'] == "npz":
<br />KeyError: 'type'



### Error 9, [Traceback at line 5442](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall.py#L5442)<br />5442..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/model_keras//namentity_crm_bilstm.py", line 348, in <module>
<br />    test(pars_choice="json", data_path=f"model_keras/namentity_crm_bilstm.json")
<br />  File "https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/model_keras//namentity_crm_bilstm.py", line 311, in test
<br />    Xtuple = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/model_keras//namentity_crm_bilstm.py", line 193, in get_dataset
<br />    raise Exception(f"Not support dataset yet")
<br />Exception: Not support dataset yet
<br />
<br />   cd /home/runner/work/mlmodels/mlmodels_store/ ;            pip3 freeze > deps.txt ;            ls ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all  &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
<br />Logs
<br />README.md
<br />README_actions.md
<br />create_error_file.py
<br />create_github_issues.py
<br />deps.txt
<br />error_list
<br />log_benchmark
<br />log_dataloader
<br />log_import
<br />log_json
<br />log_jupyter
<br />log_pullrequest
<br />log_test_cli
<br />log_testall
<br />test_jupyter
<br />Fetching origin
<br />Already up to date.
