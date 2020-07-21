
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/061c074e0ea8d028c7c86b76298dc9fc3ebb6845', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '061c074e0ea8d028c7c86b76298dc9fc3ebb6845', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/061c074e0ea8d028c7c86b76298dc9fc3ebb6845

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/061c074e0ea8d028c7c86b76298dc9fc3ebb6845

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/061c074e0ea8d028c7c86b76298dc9fc3ebb6845

 ************************************************************************************************************************

  ############Check model ################################ 





 ********************************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py --do test  

  




 #################################################################################################### 

  /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/charcnn.json 
 

  #####  Load JSON data_pars 

  {
  "data_info":{
    "dataset":"mlmodels\/dataset\/text\/ag_news_csv",
    "train":true,
    "alphabet_size":69,
    "alphabet":"abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"\/\\|_@#$%^&*~`+-=<>()[]{}",
    "input_size":1014,
    "num_of_classes":4
  },
  "preprocessors":[
    {
      "name":"loader",
      "uri":"mlmodels.preprocess.generic:pandasDataset",
      "args":{
        "colX":[
          "colX"
        ],
        "coly":[
          "coly"
        ],
        "encoding":"'ISO-8859-1'",
        "read_csv_parm":{
          "usecols":[
            0,
            1
          ],
          "names":[
            "coly",
            "colX"
          ]
        }
      }
    },
    {
      "name":"tokenizer",
      "uri":"mlmodels.model_keras.raw.char_cnn.data_utils:Data",
      "args":{
        "data_source":"",
        "alphabet":"abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"\/\\|_@#$%^&*~`+-=<>()[]{}",
        "input_size":1014,
        "num_of_classes":4
      }
    }
  ]
} 

  
 #####  Load DataLoader  

  
 #####  compute DataLoader  

  URL:  mlmodels.preprocess.generic:pandasDataset {'colX': ['colX'], 'coly': ['coly'], 'encoding': "'ISO-8859-1'", 'read_csv_parm': {'usecols': [0, 1], 'names': ['coly', 'colX']}} 

  
###### load_callable_from_uri LOADED <class 'mlmodels.preprocess.generic.pandasDataset'> 
cls_name : pandasDataset
Error /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/charcnn.json [Errno 2] File b'mlmodels/dataset/text/ag_news_csv/train/mlmodels/dataset/text/ag_news_csv.csv' does not exist: b'mlmodels/dataset/text/ag_news_csv/train/mlmodels/dataset/text/ag_news_csv.csv'

  




 #################################################################################################### 

  /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/charcnn_zhang.json 
 

  #####  Load JSON data_pars 

  {
  "data_info":{
    "dataset":"mlmodels\/dataset\/text\/ag_news_csv",
    "train":true,
    "alphabet_size":69,
    "alphabet":"abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"\/\\|_@#$%^&*~`+-=<>()[]{}",
    "input_size":1014,
    "num_of_classes":4
  },
  "preprocessors":[
    {
      "name":"loader",
      "uri":"mlmodels.preprocess.generic:pandasDataset",
      "args":{
        "colX":[
          "colX"
        ],
        "coly":[
          "coly"
        ],
        "encoding":"'ISO-8859-1'",
        "read_csv_parm":{
          "usecols":[
            0,
            1
          ],
          "names":[
            "coly",
            "colX"
          ]
        }
      }
    },
    {
      "name":"tokenizer",
      "uri":"mlmodels.model_keras.raw.char_cnn.data_utils:Data",
      "args":{
        "data_source":"",
        "alphabet":"abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"\/\\|_@#$%^&*~`+-=<>()[]{}",
        "input_size":1014,
        "num_of_classes":4
      }
    }
  ]
} 

  
 #####  Load DataLoader  

  
 #####  compute DataLoader  

  URL:  mlmodels.preprocess.generic:pandasDataset {'colX': ['colX'], 'coly': ['coly'], 'encoding': "'ISO-8859-1'", 'read_csv_parm': {'usecols': [0, 1], 'names': ['coly', 'colX']}} 

  
###### load_callable_from_uri LOADED <class 'mlmodels.preprocess.generic.pandasDataset'> 
cls_name : pandasDataset
Error /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/charcnn_zhang.json [Errno 2] File b'mlmodels/dataset/text/ag_news_csv/train/mlmodels/dataset/text/ag_news_csv.csv' does not exist: b'mlmodels/dataset/text/ag_news_csv/train/mlmodels/dataset/text/ag_news_csv.csv'

  




 #################################################################################################### 

  /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/textcnn.json 
 

  #####  Load JSON data_pars 

  {
  "data_info":{
    "dataset":"mlmodels\/dataset\/text\/imdb",
    "pass_data_pars":false,
    "train":true,
    "maxlen":40,
    "max_features":5
  },
  "preprocessors":[
    {
      "name":"loader",
      "uri":"mlmodels.preprocess.generic:NumpyDataset",
      "args":{
        "numpy_loader_args":{
          "allow_pickle":true
        },
        "encoding":"'ISO-8859-1'"
      }
    },
    {
      "name":"imdb_process",
      "uri":"mlmodels.preprocess.text_keras:IMDBDataset",
      "args":{
        "num_words":5
      }
    }
  ]
} 

  
 #####  Load DataLoader  

  
 #####  compute DataLoader  

  URL:  mlmodels.preprocess.generic:NumpyDataset {'numpy_loader_args': {'allow_pickle': True}, 'encoding': "'ISO-8859-1'"} 

  
###### load_callable_from_uri LOADED <class 'mlmodels.preprocess.generic.NumpyDataset'> 
cls_name : NumpyDataset
Dataset File path :  train/mlmodels/dataset/text/imdb.npz
Error /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/textcnn.json [Errno 2] No such file or directory: 'train/mlmodels/dataset/text/imdb.npz'

  




 #################################################################################################### 

  /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/namentity_crm_bilstm.json 
 

  #####  Load JSON data_pars 

  {
  "data_info":{
    "data_path":"dataset\/text\/",
    "dataset":"ner_dataset.csv",
    "pass_data_pars":false,
    "train":true
  },
  "preprocessors":[
    {
      "name":"loader",
      "uri":"mlmodels.preprocess.generic:pandasDataset",
      "args":{
        "read_csv_parm":{
          "encoding":"ISO-8859-1"
        },
        "colX":[

        ],
        "coly":[

        ]
      }
    },
    {
      "uri":"mlmodels.preprocess.text_keras:Preprocess_namentity",
      "args":{
        "max_len":75
      },
      "internal_states":[
        "word_count"
      ]
    },
    {
      "name":"split_xy",
      "uri":"mlmodels.dataloader:split_xy_from_dict",
      "args":{
        "col_Xinput":[
          "X"
        ],
        "col_yinput":[
          "y"
        ]
      }
    },
    {
      "name":"split_train_test",
      "uri":"sklearn.model_selection:train_test_split",
      "args":{
        "test_size":0.5
      }
    },
    {
      "name":"saver",
      "uri":"mlmodels.dataloader:pickle_dump",
      "args":{
        "path":"mlmodels\/ztest\/ml_keras\/namentity_crm_bilstm\/data.pkl"
      }
    }
  ],
  "output":{
    "shape":[
      [
        75
      ],
      [
        75,
        18
      ]
    ],
    "max_len":75
  }
} 

  
 #####  Load DataLoader  

  
 #####  compute DataLoader  

  URL:  mlmodels.preprocess.generic:pandasDataset {'read_csv_parm': {'encoding': 'ISO-8859-1'}, 'colX': [], 'coly': []} 

  
###### load_callable_from_uri LOADED <class 'mlmodels.preprocess.generic.pandasDataset'> 
cls_name : pandasDataset

  URL:  mlmodels.preprocess.text_keras:Preprocess_namentity {'max_len': 75} 

  
###### load_callable_from_uri LOADED <class 'mlmodels.preprocess.text_keras.Preprocess_namentity'> 
cls_name : Preprocess_namentity

  
 Object Creation 

  
 Object Compute 

  
 Object get_data 

  URL:  mlmodels.dataloader:split_xy_from_dict {'col_Xinput': ['X'], 'col_yinput': ['y']} 

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f9b2ac449d8> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f9b2ac449d8> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f9b958af510> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f9b958af510> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f9bb4afeea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f9bb4afeea0> 
Error /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/namentity_crm_bilstm.json [Errno 2] No such file or directory: 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'

  




 #################################################################################################### 

  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor/torchhub_cnn_dataloader.json 
 

  #####  Load JSON data_pars 

  {
  "data_info":{
    "data_path":"dataset\/vision\/MNIST",
    "dataset":"MNIST",
    "data_type":"tch_dataset",
    "batch_size":10,
    "train":true
  },
  "preprocessors":[
    {
      "name":"tch_dataset_start",
      "uri":"mlmodels.preprocess.generic:get_dataset_torch",
      "args":{
        "dataloader":"torchvision.datasets:MNIST",
        "to_image":true,
        "transform":{
          "uri":"mlmodels.preprocess.image:torch_transform_mnist",
          "pass_data_pars":false,
          "arg":{
            "fixed_size":256,
            "path":"dataset\/vision\/MNIST\/"
          }
        },
        "shuffle":true,
        "download":true
      }
    }
  ]
} 

  
 #####  Load DataLoader  

  
 #####  compute DataLoader  

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {'fixed_size': 256, 'path': 'dataset/vision/MNIST/'}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f9b42be3950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f9b42be3950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f9b42be3950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {'fixed_size': 256, 'path': 'dataset/vision/MNIST/'}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 452, in test_json_list
    loader.compute()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 258, in compute
    obj_preprocessor = preprocessor_func(**args, data_info=self.data_info)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/preprocess/generic.py", line 502, in __init__
    df = pd.read_csv(file_path, **args.get("read_csv_parm",{}))
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 685, in parser_f
    return _read(filepath_or_buffer, kwds)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 457, in _read
    parser = TextFileReader(fp_or_buf, **kwds)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 895, in __init__
    self._make_engine(self.engine)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1135, in _make_engine
    self._engine = CParserWrapper(self.f, **self.options)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1917, in __init__
    self._reader = parsers.TextReader(src, **kwds)
  File "pandas/_libs/parsers.pyx", line 382, in pandas._libs.parsers.TextReader.__cinit__
  File "pandas/_libs/parsers.pyx", line 689, in pandas._libs.parsers.TextReader._setup_parser_source
FileNotFoundError: [Errno 2] File b'mlmodels/dataset/text/ag_news_csv/train/mlmodels/dataset/text/ag_news_csv.csv' does not exist: b'mlmodels/dataset/text/ag_news_csv/train/mlmodels/dataset/text/ag_news_csv.csv'
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 452, in test_json_list
    loader.compute()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 258, in compute
    obj_preprocessor = preprocessor_func(**args, data_info=self.data_info)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/preprocess/generic.py", line 502, in __init__
    df = pd.read_csv(file_path, **args.get("read_csv_parm",{}))
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 685, in parser_f
    return _read(filepath_or_buffer, kwds)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 457, in _read
    parser = TextFileReader(fp_or_buf, **kwds)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 895, in __init__
    self._make_engine(self.engine)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1135, in _make_engine
    self._engine = CParserWrapper(self.f, **self.options)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1917, in __init__
    self._reader = parsers.TextReader(src, **kwds)
  File "pandas/_libs/parsers.pyx", line 382, in pandas._libs.parsers.TextReader.__cinit__
  File "pandas/_libs/parsers.pyx", line 689, in pandas._libs.parsers.TextReader._setup_parser_source
FileNotFoundError: [Errno 2] File b'mlmodels/dataset/text/ag_news_csv/train/mlmodels/dataset/text/ag_news_csv.csv' does not exist: b'mlmodels/dataset/text/ag_news_csv/train/mlmodels/dataset/text/ag_news_csv.csv'
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 452, in test_json_list
    loader.compute()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 258, in compute
    obj_preprocessor = preprocessor_func(**args, data_info=self.data_info)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/preprocess/generic.py", line 607, in __init__
    data            = np.load( file_path,**args.get("numpy_loader_args", {}))
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/numpy/lib/npyio.py", line 416, in load
    fid = stack.enter_context(open(os_fspath(file), "rb"))
FileNotFoundError: [Errno 2] No such file or directory: 'train/mlmodels/dataset/text/imdb.npz'
Using TensorFlow backend.
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 452, in test_json_list
    loader.compute()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 301, in compute
    out_tmp = preprocessor_func(input_tmp, **args)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 93, in pickle_dump
    with open(kwargs["path"], "wb") as fi:
FileNotFoundError: [Errno 2] No such file or directory: 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s]  0%|          | 49152/9912422 [00:00<00:33, 296909.37it/s]  2%|▏         | 212992/9912422 [00:00<00:25, 383890.46it/s]  9%|▉         | 876544/9912422 [00:00<00:17, 531141.65it/s] 31%|███       | 3031040/9912422 [00:00<00:09, 749540.02it/s] 58%|█████▊    | 5783552/9912422 [00:00<00:03, 1052687.75it/s] 89%|████████▉ | 8814592/9912422 [00:01<00:00, 1477966.33it/s]9920512it [00:01, 8860384.85it/s]                             
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 135552.47it/s]           
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:05, 296013.25it/s] 13%|█▎        | 212992/1648877 [00:00<00:03, 382183.28it/s] 53%|█████▎    | 876544/1648877 [00:00<00:01, 529626.86it/s]1654784it [00:00, 2690724.43it/s]                           
0it [00:00, ?it/s]8192it [00:00, 92448.89it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-labels-idx1-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/train-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/t10k-images-idx3-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/t10k-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw/t10k-labels-idx1-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/t10k-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Processing...
Done!

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f9b411495c0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f9b41151828>), {}) 

  




 #################################################################################################### 

  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor/model_list_CIFAR.json 
 

  #####  Load JSON data_pars 

  {
  "data_info":{
    "data_path":"dataset\/vision\/cifar10\/",
    "dataset":"CIFAR10",
    "data_type":"tf_dataset",
    "batch_size":10,
    "train":true
  },
  "preprocessors":[
    {
      "name":"tf_dataset_start",
      "uri":"mlmodels.preprocess.generic:tf_dataset_download",
      "arg":{
        "train_samples":2000,
        "test_samples":500,
        "shuffle":true,
        "download":true
      }
    },
    {
      "uri":"mlmodels.preprocess.generic:get_dataset_torch",
      "args":{
        "dataloader":"mlmodels.preprocess.generic:NumpyDataset",
        "to_image":true,
        "transform":{
          "uri":"mlmodels.preprocess.image:torch_transform_generic",
          "pass_data_pars":false,
          "arg":{
            "fixed_size":256
          }
        },
        "shuffle":true,
        "download":true
      }
    }
  ]
} 

  
 #####  Load DataLoader  

  
 #####  compute DataLoader  

  URL:  mlmodels.preprocess.generic:tf_dataset_download {} 

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f9b42be3598> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f9b42be3598> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f9b42be3598> , (data_info, **args) 

  CIFAR10 

  Dataset Name is :  cifar10 

Dl Completed...: 0 url [00:00, ? url/s]
Dl Size...: 0 MiB [00:00, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...: 0 MiB [00:00, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/urllib3/connectionpool.py:986: InsecureRequestWarning: Unverified HTTPS request is being made to host 'www.cs.toronto.edu'. Adding certificate verification is strongly advised. See: https://urllib3.readthedocs.io/en/latest/advanced-usage.html#ssl-warnings
  InsecureRequestWarning,
Dl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   0%|          | 0/162 [00:00<?, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   1%|          | 1/162 [00:00<02:10,  1.23 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<02:10,  1.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<02:09,  1.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<02:08,  1.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   2%|▏         | 4/162 [00:00<01:31,  1.73 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:31,  1.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:30,  1.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:30,  1.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:29,  1.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<01:29,  1.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:01<01:28,  1.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   6%|▌         | 10/162 [00:01<01:02,  2.43 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:01<01:02,  2.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:01<01:02,  2.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:01<01:01,  2.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:01<01:01,  2.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:01<01:00,  2.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   9%|▉         | 15/162 [00:01<00:43,  3.40 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:01<00:43,  3.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:01<00:42,  3.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:01<00:42,  3.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:01<00:42,  3.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:01<00:42,  3.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:01<00:41,  3.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  13%|█▎        | 21/162 [00:01<00:29,  4.72 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:01<00:29,  4.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:01<00:29,  4.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:01<00:29,  4.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:01<00:29,  4.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:01<00:29,  4.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:01<00:28,  4.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  17%|█▋        | 27/162 [00:01<00:20,  6.47 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:01<00:20,  6.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:01<00:20,  6.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:01<00:20,  6.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:01<00:20,  6.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:01<00:20,  6.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:01<00:20,  6.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  20%|██        | 33/162 [00:01<00:14,  8.77 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:01<00:14,  8.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:14,  8.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:14,  8.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:14,  8.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:14,  8.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:14,  8.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  24%|██▍       | 39/162 [00:01<00:10, 11.66 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:10, 11.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:10, 11.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:10, 11.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:10, 11.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:10, 11.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:10, 11.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  28%|██▊       | 45/162 [00:01<00:07, 15.17 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:07, 15.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:07, 15.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:07, 15.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:07, 15.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:07, 15.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  31%|███       | 50/162 [00:01<00:05, 19.14 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:05, 19.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:05, 19.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:05, 19.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:05, 19.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:05, 19.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  34%|███▍      | 55/162 [00:01<00:04, 23.24 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:04, 23.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:04, 23.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:04, 23.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:02<00:04, 23.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:02<00:04, 23.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:02<00:04, 23.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  38%|███▊      | 61/162 [00:02<00:03, 27.71 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:02<00:03, 27.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:02<00:03, 27.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:02<00:03, 27.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:02<00:03, 27.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:02<00:03, 27.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:02<00:03, 27.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  41%|████▏     | 67/162 [00:02<00:02, 32.11 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:02<00:02, 32.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:02<00:02, 32.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:02<00:02, 32.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:02<00:02, 32.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:02<00:02, 32.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  44%|████▍     | 72/162 [00:02<00:02, 35.89 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:02<00:02, 35.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:02<00:02, 35.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:02<00:02, 35.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:02<00:02, 35.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:02<00:02, 35.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:02<00:02, 35.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  48%|████▊     | 78/162 [00:02<00:02, 39.63 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:02<00:02, 39.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:02<00:02, 39.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:02<00:02, 39.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:02<00:02, 39.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:02<00:02, 39.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:02<00:01, 39.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  52%|█████▏    | 84/162 [00:02<00:01, 41.89 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:02<00:01, 41.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:02<00:01, 41.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:02<00:01, 41.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:02<00:01, 41.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:02<00:01, 41.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:02<00:01, 41.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  56%|█████▌    | 90/162 [00:02<00:01, 44.22 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:02<00:01, 44.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:02<00:01, 44.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:02<00:01, 44.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:02<00:01, 44.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:02<00:01, 44.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:02<00:01, 44.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  59%|█████▉    | 96/162 [00:02<00:01, 46.07 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:02<00:01, 46.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:02<00:01, 46.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:02<00:01, 46.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:02<00:01, 46.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:02<00:01, 46.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:02<00:01, 46.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  63%|██████▎   | 102/162 [00:02<00:01, 47.43 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:02<00:01, 47.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:02<00:01, 47.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:02<00:01, 47.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:02<00:01, 47.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:02<00:01, 47.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:02<00:01, 47.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  67%|██████▋   | 108/162 [00:03<00:01, 48.70 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:03<00:01, 48.70 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:03<00:01, 48.70 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:03<00:01, 48.70 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:03<00:01, 48.70 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:03<00:01, 48.70 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:03<00:01, 48.70 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  70%|███████   | 114/162 [00:03<00:00, 48.94 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:03<00:00, 48.94 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:03<00:00, 48.94 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:03<00:00, 48.94 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:03<00:00, 48.94 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:03<00:00, 48.94 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:03<00:00, 48.94 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  74%|███████▍  | 120/162 [00:03<00:00, 49.13 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:03<00:00, 49.13 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:03<00:00, 49.13 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:03<00:00, 49.13 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:03<00:00, 49.13 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:03<00:00, 49.13 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:03<00:00, 49.13 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  78%|███████▊  | 126/162 [00:03<00:00, 49.53 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:03<00:00, 49.53 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:03<00:00, 49.53 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:03<00:00, 49.53 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:03<00:00, 49.53 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:03<00:00, 49.53 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:03<00:00, 49.53 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  81%|████████▏ | 132/162 [00:03<00:00, 50.28 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:03<00:00, 50.28 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:03<00:00, 50.28 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:03<00:00, 50.28 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:03<00:00, 50.28 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:03<00:00, 50.28 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:03<00:00, 50.28 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  85%|████████▌ | 138/162 [00:03<00:00, 50.37 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:03<00:00, 50.37 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:03<00:00, 50.37 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:03<00:00, 50.37 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:03<00:00, 50.37 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:03<00:00, 50.37 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:03<00:00, 50.37 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  89%|████████▉ | 144/162 [00:03<00:00, 50.19 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:03<00:00, 50.19 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:03<00:00, 50.19 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:03<00:00, 50.19 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:03<00:00, 50.19 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:03<00:00, 50.19 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:03<00:00, 50.19 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  93%|█████████▎| 150/162 [00:03<00:00, 49.85 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:03<00:00, 49.85 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:03<00:00, 49.85 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:03<00:00, 49.85 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:03<00:00, 49.85 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:03<00:00, 49.85 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:03<00:00, 49.85 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  96%|█████████▋| 156/162 [00:03<00:00, 50.12 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:03<00:00, 50.12 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:03<00:00, 50.12 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:03<00:00, 50.12 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:04<00:00, 50.12 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:04<00:00, 50.12 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:04<00:00, 50.12 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 50.46 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 50.46 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.08s/ url]Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.08s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 50.46 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.08s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 50.46 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:04<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:06<00:00,  6.17s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:06<00:00,  4.08s/ url]
Dl Size...: 100%|██████████| 162/162 [00:06<00:00, 50.46 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:06<00:00,  6.17s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:06<00:00,  6.17s/ file]
Dl Size...: 100%|██████████| 162/162 [00:06<00:00, 26.26 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:06<00:00,  6.17s/ url]
0 examples [00:00, ? examples/s]2020-07-21 00:08:53.278712: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-21 00:08:53.294388: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095190000 Hz
2020-07-21 00:08:53.294592: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5568c7e1f5d0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-21 00:08:53.294611: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
51 examples [00:00, 503.11 examples/s]156 examples [00:00, 596.19 examples/s]270 examples [00:00, 695.71 examples/s]382 examples [00:00, 783.52 examples/s]491 examples [00:00, 854.67 examples/s]606 examples [00:00, 925.34 examples/s]719 examples [00:00, 976.83 examples/s]832 examples [00:00, 1016.03 examples/s]936 examples [00:00, 1013.19 examples/s]1046 examples [00:01, 1035.26 examples/s]1160 examples [00:01, 1062.70 examples/s]1272 examples [00:01, 1078.40 examples/s]1385 examples [00:01, 1091.27 examples/s]1495 examples [00:01, 1093.11 examples/s]1605 examples [00:01, 1084.07 examples/s]1718 examples [00:01, 1097.30 examples/s]1829 examples [00:01, 1098.49 examples/s]1940 examples [00:01, 1097.51 examples/s]2052 examples [00:01, 1103.81 examples/s]2163 examples [00:02, 1104.21 examples/s]2274 examples [00:02, 1104.20 examples/s]2387 examples [00:02, 1109.66 examples/s]2501 examples [00:02, 1117.94 examples/s]2614 examples [00:02, 1119.02 examples/s]2726 examples [00:02, 1100.57 examples/s]2837 examples [00:02, 1053.87 examples/s]2946 examples [00:02, 1063.08 examples/s]3055 examples [00:02, 1068.80 examples/s]3163 examples [00:02, 1072.01 examples/s]3271 examples [00:03, 1061.36 examples/s]3383 examples [00:03, 1076.23 examples/s]3497 examples [00:03, 1094.05 examples/s]3610 examples [00:03, 1103.26 examples/s]3723 examples [00:03, 1110.71 examples/s]3835 examples [00:03, 1110.49 examples/s]3950 examples [00:03, 1119.81 examples/s]4065 examples [00:03, 1126.31 examples/s]4178 examples [00:03, 1100.94 examples/s]4293 examples [00:03, 1113.41 examples/s]4405 examples [00:04, 1106.76 examples/s]4518 examples [00:04, 1111.57 examples/s]4630 examples [00:04, 1108.31 examples/s]4741 examples [00:04, 1099.68 examples/s]4853 examples [00:04, 1102.80 examples/s]4964 examples [00:04, 1103.35 examples/s]5076 examples [00:04, 1106.54 examples/s]5187 examples [00:04, 1088.01 examples/s]5296 examples [00:04, 1084.70 examples/s]5405 examples [00:04, 1076.94 examples/s]5513 examples [00:05, 1073.16 examples/s]5625 examples [00:05, 1086.42 examples/s]5738 examples [00:05, 1098.79 examples/s]5852 examples [00:05, 1109.57 examples/s]5966 examples [00:05, 1116.31 examples/s]6078 examples [00:05, 1108.80 examples/s]6189 examples [00:05, 1035.40 examples/s]6294 examples [00:05, 992.37 examples/s] 6398 examples [00:05, 1004.87 examples/s]6502 examples [00:06, 1014.76 examples/s]6611 examples [00:06, 1034.96 examples/s]6719 examples [00:06, 1045.78 examples/s]6824 examples [00:06, 1046.54 examples/s]6937 examples [00:06, 1069.00 examples/s]7045 examples [00:06, 1055.42 examples/s]7157 examples [00:06, 1071.01 examples/s]7267 examples [00:06, 1078.45 examples/s]7381 examples [00:06, 1095.72 examples/s]7491 examples [00:06, 1094.20 examples/s]7601 examples [00:07, 1087.68 examples/s]7710 examples [00:07, 1086.08 examples/s]7824 examples [00:07, 1101.12 examples/s]7935 examples [00:07, 1094.26 examples/s]8045 examples [00:07, 1094.99 examples/s]8155 examples [00:07, 1084.87 examples/s]8264 examples [00:07, 1074.19 examples/s]8372 examples [00:07, 1030.68 examples/s]8477 examples [00:07, 1025.20 examples/s]8580 examples [00:07, 966.75 examples/s] 8681 examples [00:08, 977.34 examples/s]8780 examples [00:08, 930.17 examples/s]8874 examples [00:08, 916.03 examples/s]8967 examples [00:08, 901.22 examples/s]9058 examples [00:08, 892.02 examples/s]9154 examples [00:08, 909.12 examples/s]9246 examples [00:08, 908.62 examples/s]9338 examples [00:08, 897.92 examples/s]9444 examples [00:08, 940.13 examples/s]9558 examples [00:09, 991.62 examples/s]9667 examples [00:09, 1019.13 examples/s]9770 examples [00:09, 1008.10 examples/s]9877 examples [00:09, 1023.79 examples/s]9990 examples [00:09, 1053.15 examples/s]10096 examples [00:09, 1023.18 examples/s]10206 examples [00:09, 1044.28 examples/s]10311 examples [00:09, 1008.00 examples/s]10417 examples [00:09, 1022.52 examples/s]10529 examples [00:09, 1049.71 examples/s]10637 examples [00:10, 1058.23 examples/s]10751 examples [00:10, 1080.62 examples/s]10864 examples [00:10, 1094.41 examples/s]10979 examples [00:10, 1109.38 examples/s]11091 examples [00:10, 1093.82 examples/s]11206 examples [00:10, 1107.68 examples/s]11317 examples [00:10, 1108.21 examples/s]11433 examples [00:10, 1121.53 examples/s]11549 examples [00:10, 1131.40 examples/s]11663 examples [00:10, 1133.31 examples/s]11778 examples [00:11, 1136.38 examples/s]11892 examples [00:11, 1132.12 examples/s]12006 examples [00:11, 1102.60 examples/s]12117 examples [00:11, 1102.72 examples/s]12228 examples [00:11, 1097.53 examples/s]12343 examples [00:11, 1112.25 examples/s]12455 examples [00:11, 1108.63 examples/s]12568 examples [00:11, 1112.04 examples/s]12680 examples [00:11, 1091.94 examples/s]12790 examples [00:11, 1093.83 examples/s]12902 examples [00:12, 1100.41 examples/s]13013 examples [00:12, 1101.90 examples/s]13124 examples [00:12, 1101.49 examples/s]13237 examples [00:12, 1108.55 examples/s]13351 examples [00:12, 1115.80 examples/s]13464 examples [00:12, 1118.16 examples/s]13576 examples [00:12, 1115.04 examples/s]13688 examples [00:12, 1114.48 examples/s]13800 examples [00:12, 1112.40 examples/s]13912 examples [00:13, 1092.43 examples/s]14022 examples [00:13, 1079.66 examples/s]14132 examples [00:13, 1085.47 examples/s]14242 examples [00:13, 1089.44 examples/s]14351 examples [00:13, 1048.41 examples/s]14463 examples [00:13, 1068.00 examples/s]14574 examples [00:13, 1079.11 examples/s]14687 examples [00:13, 1091.66 examples/s]14802 examples [00:13, 1106.88 examples/s]14916 examples [00:13, 1114.61 examples/s]15029 examples [00:14, 1117.70 examples/s]15141 examples [00:14, 1115.07 examples/s]15253 examples [00:14, 1105.72 examples/s]15364 examples [00:14, 1105.39 examples/s]15475 examples [00:14, 1097.59 examples/s]15588 examples [00:14, 1106.76 examples/s]15699 examples [00:14, 1088.95 examples/s]15808 examples [00:14, 1080.19 examples/s]15921 examples [00:14, 1092.05 examples/s]16035 examples [00:14, 1105.35 examples/s]16149 examples [00:15, 1113.12 examples/s]16261 examples [00:15, 1114.91 examples/s]16373 examples [00:15, 1106.90 examples/s]16484 examples [00:15, 1096.38 examples/s]16594 examples [00:15, 1093.15 examples/s]16704 examples [00:15, 1073.63 examples/s]16812 examples [00:15, 1018.26 examples/s]16922 examples [00:15, 1040.11 examples/s]17032 examples [00:15, 1056.56 examples/s]17145 examples [00:15, 1076.81 examples/s]17257 examples [00:16, 1088.25 examples/s]17368 examples [00:16, 1094.00 examples/s]17478 examples [00:16, 1068.93 examples/s]17586 examples [00:16, 1059.77 examples/s]17696 examples [00:16, 1071.09 examples/s]17809 examples [00:16, 1087.00 examples/s]17922 examples [00:16, 1096.70 examples/s]18032 examples [00:16, 1095.22 examples/s]18142 examples [00:16, 1081.40 examples/s]18251 examples [00:17, 1072.51 examples/s]18359 examples [00:17, 1070.80 examples/s]18467 examples [00:17, 1068.66 examples/s]18574 examples [00:17, 1061.65 examples/s]18687 examples [00:17, 1080.25 examples/s]18799 examples [00:17, 1090.13 examples/s]18909 examples [00:17, 1084.95 examples/s]19018 examples [00:17, 1071.15 examples/s]19128 examples [00:17, 1078.19 examples/s]19241 examples [00:17, 1091.01 examples/s]19353 examples [00:18, 1098.54 examples/s]19466 examples [00:18, 1107.24 examples/s]19579 examples [00:18, 1113.53 examples/s]19691 examples [00:18, 1114.86 examples/s]19804 examples [00:18, 1117.67 examples/s]19916 examples [00:18, 1118.22 examples/s]20028 examples [00:18, 1063.83 examples/s]20142 examples [00:18, 1083.22 examples/s]20256 examples [00:18, 1097.88 examples/s]20369 examples [00:18, 1105.62 examples/s]20481 examples [00:19, 1106.70 examples/s]20592 examples [00:19, 1106.30 examples/s]20703 examples [00:19, 1101.85 examples/s]20814 examples [00:19, 1097.01 examples/s]20925 examples [00:19, 1098.87 examples/s]21036 examples [00:19, 1101.28 examples/s]21150 examples [00:19, 1111.82 examples/s]21262 examples [00:19, 1069.45 examples/s]21374 examples [00:19, 1083.70 examples/s]21487 examples [00:19, 1096.85 examples/s]21602 examples [00:20, 1110.07 examples/s]21717 examples [00:20, 1120.83 examples/s]21830 examples [00:20, 1117.00 examples/s]21942 examples [00:20, 1111.67 examples/s]22055 examples [00:20, 1114.94 examples/s]22169 examples [00:20, 1121.28 examples/s]22284 examples [00:20, 1127.51 examples/s]22397 examples [00:20, 1123.95 examples/s]22510 examples [00:20, 1119.36 examples/s]22623 examples [00:20, 1121.74 examples/s]22736 examples [00:21, 1117.28 examples/s]22848 examples [00:21, 1114.83 examples/s]22961 examples [00:21, 1116.59 examples/s]23074 examples [00:21, 1119.60 examples/s]23188 examples [00:21, 1125.18 examples/s]23303 examples [00:21, 1131.29 examples/s]23418 examples [00:21, 1135.89 examples/s]23532 examples [00:21, 1118.57 examples/s]23644 examples [00:21, 1112.70 examples/s]23756 examples [00:21, 1098.79 examples/s]23866 examples [00:22, 1093.58 examples/s]23976 examples [00:22, 1092.30 examples/s]24087 examples [00:22, 1095.09 examples/s]24199 examples [00:22, 1100.39 examples/s]24311 examples [00:22, 1106.04 examples/s]24422 examples [00:22, 1100.87 examples/s]24533 examples [00:22, 1102.58 examples/s]24644 examples [00:22, 1072.91 examples/s]24754 examples [00:22, 1079.29 examples/s]24864 examples [00:23, 1083.99 examples/s]24977 examples [00:23, 1096.46 examples/s]25089 examples [00:23, 1100.68 examples/s]25200 examples [00:23, 1079.17 examples/s]25311 examples [00:23, 1085.73 examples/s]25425 examples [00:23, 1101.08 examples/s]25537 examples [00:23, 1106.24 examples/s]25653 examples [00:23, 1119.92 examples/s]25766 examples [00:23, 1121.41 examples/s]25879 examples [00:23, 1080.94 examples/s]25991 examples [00:24, 1091.83 examples/s]26103 examples [00:24, 1097.53 examples/s]26216 examples [00:24, 1105.25 examples/s]26327 examples [00:24, 1090.79 examples/s]26437 examples [00:24, 1093.16 examples/s]26547 examples [00:24, 1092.07 examples/s]26657 examples [00:24, 1075.38 examples/s]26765 examples [00:24, 1072.29 examples/s]26873 examples [00:24, 1073.54 examples/s]26981 examples [00:24, 1074.92 examples/s]27090 examples [00:25, 1077.73 examples/s]27198 examples [00:25, 1076.58 examples/s]27306 examples [00:25, 1058.56 examples/s]27412 examples [00:25, 1048.69 examples/s]27522 examples [00:25, 1062.50 examples/s]27634 examples [00:25, 1077.20 examples/s]27742 examples [00:25, 1064.81 examples/s]27854 examples [00:25, 1079.60 examples/s]27964 examples [00:25, 1082.87 examples/s]28073 examples [00:25, 1082.26 examples/s]28182 examples [00:26, 1084.26 examples/s]28294 examples [00:26, 1093.81 examples/s]28405 examples [00:26, 1097.56 examples/s]28515 examples [00:26, 1095.55 examples/s]28625 examples [00:26, 1074.67 examples/s]28737 examples [00:26, 1086.92 examples/s]28848 examples [00:26, 1092.32 examples/s]28958 examples [00:26, 1085.87 examples/s]29068 examples [00:26, 1089.15 examples/s]29177 examples [00:26, 1084.47 examples/s]29286 examples [00:27, 1072.90 examples/s]29395 examples [00:27, 1077.10 examples/s]29508 examples [00:27, 1091.62 examples/s]29620 examples [00:27, 1097.74 examples/s]29730 examples [00:27, 1089.92 examples/s]29842 examples [00:27, 1098.59 examples/s]29955 examples [00:27, 1106.99 examples/s]30066 examples [00:27, 1055.00 examples/s]30176 examples [00:27, 1066.83 examples/s]30287 examples [00:27, 1077.72 examples/s]30399 examples [00:28, 1086.82 examples/s]30508 examples [00:28, 1048.34 examples/s]30618 examples [00:28, 1062.64 examples/s]30731 examples [00:28, 1081.33 examples/s]30840 examples [00:28, 1073.55 examples/s]30951 examples [00:28, 1083.97 examples/s]31060 examples [00:28, 1083.99 examples/s]31175 examples [00:28, 1100.89 examples/s]31287 examples [00:28, 1106.08 examples/s]31398 examples [00:29, 1102.88 examples/s]31511 examples [00:29, 1109.37 examples/s]31623 examples [00:29, 1110.44 examples/s]31735 examples [00:29, 1109.70 examples/s]31847 examples [00:29, 1097.69 examples/s]31957 examples [00:29, 1089.86 examples/s]32069 examples [00:29, 1098.58 examples/s]32182 examples [00:29, 1105.26 examples/s]32294 examples [00:29, 1108.89 examples/s]32406 examples [00:29, 1110.97 examples/s]32518 examples [00:30, 1105.59 examples/s]32631 examples [00:30, 1110.17 examples/s]32743 examples [00:30, 1094.79 examples/s]32855 examples [00:30, 1100.89 examples/s]32969 examples [00:30, 1110.34 examples/s]33081 examples [00:30, 1101.12 examples/s]33192 examples [00:30, 1100.44 examples/s]33306 examples [00:30, 1111.49 examples/s]33422 examples [00:30, 1123.01 examples/s]33535 examples [00:30, 1124.16 examples/s]33648 examples [00:31, 1120.25 examples/s]33763 examples [00:31, 1128.54 examples/s]33876 examples [00:31, 1111.82 examples/s]33988 examples [00:31, 1112.45 examples/s]34100 examples [00:31, 1095.74 examples/s]34210 examples [00:31, 1088.97 examples/s]34319 examples [00:31, 1085.83 examples/s]34431 examples [00:31, 1093.24 examples/s]34543 examples [00:31, 1100.22 examples/s]34654 examples [00:31, 1096.14 examples/s]34764 examples [00:32, 1048.79 examples/s]34871 examples [00:32, 1053.15 examples/s]34977 examples [00:32, 1047.59 examples/s]35082 examples [00:32, 1023.05 examples/s]35185 examples [00:32, 1022.27 examples/s]35291 examples [00:32, 1030.54 examples/s]35400 examples [00:32, 1047.09 examples/s]35507 examples [00:32, 1053.10 examples/s]35616 examples [00:32, 1061.81 examples/s]35723 examples [00:32, 1062.59 examples/s]35830 examples [00:33, 1028.59 examples/s]35941 examples [00:33, 1050.16 examples/s]36053 examples [00:33, 1068.60 examples/s]36165 examples [00:33, 1081.37 examples/s]36275 examples [00:33, 1084.91 examples/s]36387 examples [00:33, 1092.52 examples/s]36497 examples [00:33, 1094.71 examples/s]36608 examples [00:33, 1096.40 examples/s]36719 examples [00:33, 1098.23 examples/s]36831 examples [00:34, 1101.77 examples/s]36944 examples [00:34, 1108.08 examples/s]37055 examples [00:34, 1106.17 examples/s]37169 examples [00:34, 1113.95 examples/s]37282 examples [00:34, 1117.73 examples/s]37394 examples [00:34, 1078.79 examples/s]37505 examples [00:34, 1086.63 examples/s]37617 examples [00:34, 1094.14 examples/s]37727 examples [00:34, 1091.95 examples/s]37837 examples [00:34, 1091.08 examples/s]37949 examples [00:35, 1099.53 examples/s]38060 examples [00:35, 1102.28 examples/s]38173 examples [00:35, 1109.48 examples/s]38285 examples [00:35, 1111.07 examples/s]38397 examples [00:35, 1092.53 examples/s]38509 examples [00:35, 1100.61 examples/s]38621 examples [00:35, 1105.79 examples/s]38732 examples [00:35, 1065.89 examples/s]38839 examples [00:35, 1066.29 examples/s]38949 examples [00:35, 1075.18 examples/s]39057 examples [00:36, 1074.60 examples/s]39170 examples [00:36, 1090.40 examples/s]39281 examples [00:36, 1095.37 examples/s]39392 examples [00:36, 1098.38 examples/s]39502 examples [00:36, 1094.04 examples/s]39612 examples [00:36, 1086.40 examples/s]39721 examples [00:36, 1073.51 examples/s]39834 examples [00:36, 1087.88 examples/s]39946 examples [00:36, 1096.45 examples/s]40056 examples [00:36, 1046.56 examples/s]40162 examples [00:37, 994.63 examples/s] 40263 examples [00:37, 989.85 examples/s]40376 examples [00:37, 1026.35 examples/s]40488 examples [00:37, 1050.40 examples/s]40599 examples [00:37, 1067.25 examples/s]40709 examples [00:37, 1074.34 examples/s]40822 examples [00:37, 1088.66 examples/s]40933 examples [00:37, 1092.88 examples/s]41044 examples [00:37, 1096.89 examples/s]41154 examples [00:38, 1097.32 examples/s]41264 examples [00:38, 1093.34 examples/s]41376 examples [00:38, 1100.85 examples/s]41487 examples [00:38, 1102.06 examples/s]41599 examples [00:38, 1105.87 examples/s]41712 examples [00:38, 1111.72 examples/s]41824 examples [00:38, 1109.71 examples/s]41935 examples [00:38, 1105.54 examples/s]42046 examples [00:38, 1071.71 examples/s]42159 examples [00:38, 1087.84 examples/s]42271 examples [00:39, 1095.45 examples/s]42381 examples [00:39, 1093.32 examples/s]42491 examples [00:39, 1089.00 examples/s]42600 examples [00:39, 1080.13 examples/s]42709 examples [00:39, 1076.99 examples/s]42817 examples [00:39, 1075.08 examples/s]42930 examples [00:39, 1088.43 examples/s]43041 examples [00:39, 1092.52 examples/s]43155 examples [00:39, 1105.22 examples/s]43270 examples [00:39, 1116.37 examples/s]43385 examples [00:40, 1124.57 examples/s]43498 examples [00:40, 1121.90 examples/s]43611 examples [00:40, 1120.43 examples/s]43724 examples [00:40, 1121.71 examples/s]43837 examples [00:40, 1104.26 examples/s]43952 examples [00:40, 1115.86 examples/s]44065 examples [00:40, 1117.27 examples/s]44178 examples [00:40, 1119.36 examples/s]44290 examples [00:40, 1115.47 examples/s]44403 examples [00:40, 1119.01 examples/s]44516 examples [00:41, 1119.66 examples/s]44628 examples [00:41, 1113.78 examples/s]44740 examples [00:41, 1108.47 examples/s]44851 examples [00:41, 1105.10 examples/s]44962 examples [00:41, 1086.50 examples/s]45075 examples [00:41, 1096.41 examples/s]45185 examples [00:41, 1057.40 examples/s]45299 examples [00:41, 1079.06 examples/s]45410 examples [00:41, 1087.04 examples/s]45522 examples [00:41, 1095.62 examples/s]45636 examples [00:42, 1108.28 examples/s]45748 examples [00:42, 1104.94 examples/s]45862 examples [00:42, 1115.14 examples/s]45974 examples [00:42, 1114.44 examples/s]46086 examples [00:42, 1107.53 examples/s]46200 examples [00:42, 1115.35 examples/s]46314 examples [00:42, 1121.42 examples/s]46432 examples [00:42, 1138.03 examples/s]46548 examples [00:42, 1141.69 examples/s]46663 examples [00:42, 1097.54 examples/s]46777 examples [00:43, 1109.58 examples/s]46889 examples [00:43, 1104.98 examples/s]47002 examples [00:43, 1110.04 examples/s]47114 examples [00:43, 1110.76 examples/s]47228 examples [00:43, 1117.96 examples/s]47341 examples [00:43, 1120.55 examples/s]47454 examples [00:43, 1117.28 examples/s]47567 examples [00:43, 1119.25 examples/s]47680 examples [00:43, 1121.12 examples/s]47793 examples [00:44, 1120.88 examples/s]47908 examples [00:44, 1129.13 examples/s]48023 examples [00:44, 1134.45 examples/s]48137 examples [00:44, 1125.43 examples/s]48250 examples [00:44, 1118.20 examples/s]48362 examples [00:44, 1108.62 examples/s]48473 examples [00:44, 1106.24 examples/s]48584 examples [00:44, 1103.50 examples/s]48697 examples [00:44, 1109.87 examples/s]48812 examples [00:44, 1119.20 examples/s]48924 examples [00:45, 1102.67 examples/s]49036 examples [00:45, 1107.32 examples/s]49149 examples [00:45, 1112.50 examples/s]49261 examples [00:45, 1110.66 examples/s]49373 examples [00:45, 1111.17 examples/s]49485 examples [00:45, 1110.94 examples/s]49597 examples [00:45, 1076.20 examples/s]49705 examples [00:45, 1046.22 examples/s]49812 examples [00:45, 1051.90 examples/s]49922 examples [00:45, 1064.45 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 13%|█▎        | 6476/50000 [00:00<00:00, 64758.67 examples/s] 38%|███▊      | 19148/50000 [00:00<00:00, 75890.62 examples/s] 65%|██████▍   | 32373/50000 [00:00<00:00, 87014.50 examples/s] 92%|█████████▏| 45821/50000 [00:00<00:00, 97318.89 examples/s]                                                               0 examples [00:00, ? examples/s]88 examples [00:00, 874.65 examples/s]201 examples [00:00, 937.44 examples/s]311 examples [00:00, 980.03 examples/s]424 examples [00:00, 1019.54 examples/s]539 examples [00:00, 1052.91 examples/s]652 examples [00:00, 1073.62 examples/s]760 examples [00:00, 1074.85 examples/s]873 examples [00:00, 1088.80 examples/s]985 examples [00:00, 1097.14 examples/s]1092 examples [00:01, 1084.15 examples/s]1203 examples [00:01, 1091.22 examples/s]1316 examples [00:01, 1099.42 examples/s]1425 examples [00:01, 1096.43 examples/s]1537 examples [00:01, 1103.09 examples/s]1650 examples [00:01, 1108.73 examples/s]1763 examples [00:01, 1112.27 examples/s]1875 examples [00:01, 1111.52 examples/s]1986 examples [00:01, 1065.30 examples/s]2098 examples [00:01, 1080.41 examples/s]2208 examples [00:02, 1083.74 examples/s]2318 examples [00:02, 1088.34 examples/s]2431 examples [00:02, 1099.26 examples/s]2544 examples [00:02, 1106.90 examples/s]2655 examples [00:02, 1038.13 examples/s]2765 examples [00:02, 1055.94 examples/s]2881 examples [00:02, 1083.06 examples/s]2994 examples [00:02, 1093.74 examples/s]3104 examples [00:02, 1082.16 examples/s]3219 examples [00:02, 1101.36 examples/s]3334 examples [00:03, 1115.29 examples/s]3450 examples [00:03, 1127.52 examples/s]3563 examples [00:03, 1126.88 examples/s]3677 examples [00:03, 1128.95 examples/s]3791 examples [00:03, 1131.81 examples/s]3905 examples [00:03, 1132.44 examples/s]4019 examples [00:03, 1134.50 examples/s]4133 examples [00:03, 1132.08 examples/s]4247 examples [00:03, 1130.86 examples/s]4361 examples [00:03, 1132.59 examples/s]4476 examples [00:04, 1136.70 examples/s]4590 examples [00:04, 1134.34 examples/s]4704 examples [00:04, 1133.08 examples/s]4818 examples [00:04, 1113.38 examples/s]4932 examples [00:04, 1119.49 examples/s]5045 examples [00:04, 1111.70 examples/s]5158 examples [00:04, 1115.83 examples/s]5270 examples [00:04, 1116.35 examples/s]5383 examples [00:04, 1119.52 examples/s]5496 examples [00:04, 1119.93 examples/s]5610 examples [00:05, 1125.23 examples/s]5724 examples [00:05, 1129.29 examples/s]5837 examples [00:05, 1121.67 examples/s]5953 examples [00:05, 1130.79 examples/s]6067 examples [00:05, 1125.06 examples/s]6180 examples [00:05, 1119.91 examples/s]6294 examples [00:05, 1125.55 examples/s]6407 examples [00:05, 1122.78 examples/s]6520 examples [00:05, 1122.33 examples/s]6633 examples [00:05, 1119.16 examples/s]6745 examples [00:06, 1110.94 examples/s]6860 examples [00:06, 1121.50 examples/s]6973 examples [00:06, 1115.72 examples/s]7085 examples [00:06, 1113.38 examples/s]7197 examples [00:06, 1107.37 examples/s]7308 examples [00:06, 1048.76 examples/s]7419 examples [00:06, 1065.36 examples/s]7527 examples [00:06, 1065.72 examples/s]7635 examples [00:06, 1069.91 examples/s]7743 examples [00:07, 1053.64 examples/s]7852 examples [00:07, 1063.08 examples/s]7959 examples [00:07, 1032.32 examples/s]8068 examples [00:07, 1047.30 examples/s]8177 examples [00:07, 1055.06 examples/s]8288 examples [00:07, 1070.12 examples/s]8403 examples [00:07, 1090.81 examples/s]8517 examples [00:07, 1103.42 examples/s]8630 examples [00:07, 1109.31 examples/s]8744 examples [00:07, 1117.81 examples/s]8857 examples [00:08, 1120.08 examples/s]8970 examples [00:08, 1119.95 examples/s]9085 examples [00:08, 1128.17 examples/s]9198 examples [00:08, 1119.37 examples/s]9312 examples [00:08, 1124.25 examples/s]9425 examples [00:08, 1097.53 examples/s]9535 examples [00:08, 1065.66 examples/s]9644 examples [00:08, 1071.41 examples/s]9756 examples [00:08, 1084.66 examples/s]9871 examples [00:08, 1101.45 examples/s]9984 examples [00:09, 1106.96 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteLIEXQX/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteLIEXQX/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f9b42be3950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f9b42be3950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f9b42be3950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f9b40cbdbe0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f9accf633c8>), {}) 

  




 #################################################################################################### 

  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor/resnet34_benchmark_mnist.json 
 

  #####  Load JSON data_pars 

  {
  "data_info":{
    "data_path":"dataset\/vision\/MNIST\/",
    "dataset":"MNIST",
    "data_type":"tch_dataset",
    "batch_size":10,
    "train":true
  },
  "preprocessors":[
    {
      "name":"tch_dataset_start",
      "uri":"mlmodels.preprocess.generic:get_dataset_torch",
      "args":{
        "dataloader":"torchvision.datasets:MNIST",
        "to_image":true,
        "transform":{
          "uri":"mlmodels.preprocess.image:torch_transform_mnist",
          "pass_data_pars":false,
          "arg":{

          }
        },
        "shuffle":true,
        "download":true
      }
    }
  ]
} 

  
 #####  Load DataLoader  

  
 #####  compute DataLoader  

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f9b42be3950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f9b42be3950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f9b42be3950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f9b41151208>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f9b411516d8>), {}) 

  




 #################################################################################################### 

  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor/textcnn.json 
 

  #####  Load JSON data_pars 

  {
  "data_info":{
    "data_path":"dataset\/recommender\/",
    "dataset":"IMDB_sample.txt",
    "data_type":"csv_dataset",
    "batch_size":64,
    "train":true
  },
  "preprocessors":[
    {
      "uri":"mlmodels.model_tch.textcnn:split_train_valid",
      "args":{
        "frac":0.99
      }
    },
    {
      "uri":"mlmodels.model_tch.textcnn:create_tabular_dataset",
      "args":{
        "lang":"en",
        "pretrained_emb":"glove.6B.300d"
      }
    }
  ]
} 

  
 #####  Load DataLoader  

  
 #####  compute DataLoader  

  URL:  mlmodels.model_tch.textcnn:split_train_valid {'frac': 0.99} 

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f9abb583e18> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f9abb583e18> 

  function with postional parmater data_info <function split_train_valid at 0x7f9abb583e18> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f9abb583f28> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f9abb583f28> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f9abb583f28> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.1
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.1) (2.3.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (4.48.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.19.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.1.3)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.24.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.7.1)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.0)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (7.4.1)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (45.2.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.4.1)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.0.3)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2020.6.20)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.25.9)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.10)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.7.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.1-py3-none-any.whl size=12047105 sha256=92e50ab3cca352fe44bb986880372137f5cf463e294954231e2f3325ba6f66b3
  Stored in directory: /tmp/pip-ephem-wheel-cache-lfl8qcb3/wheels/10/6f/a6/ddd8204ceecdedddea923f8514e13afb0c1f0f556d2c9c3da0
Successfully built en-core-web-sm
Installing collected packages: en-core-web-sm
Successfully installed en-core-web-sm-2.3.1
[38;5;2m✔ Download and installation successful[0m
You can now load the model via spacy.load('en_core_web_sm')
[38;5;2m✔ Linking successful[0m
/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/en_core_web_sm
-->
/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/spacy/data/en
You can now load the model via spacy.load('en')
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<7:51:12, 30.5kB/s].vector_cache/glove.6B.zip:   0%|          | 352k/862M [00:00<5:30:58, 43.4kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.62M/862M [00:00<3:50:55, 62.0kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.4M/862M [00:00<2:40:02, 88.5kB/s].vector_cache/glove.6B.zip:   3%|▎         | 22.0M/862M [00:00<1:50:48, 126kB/s] .vector_cache/glove.6B.zip:   4%|▎         | 30.4M/862M [00:00<1:16:50, 180kB/s].vector_cache/glove.6B.zip:   4%|▍         | 36.8M/862M [00:00<53:26, 257kB/s]  .vector_cache/glove.6B.zip:   5%|▌         | 43.1M/862M [00:00<37:11, 367kB/s].vector_cache/glove.6B.zip:   6%|▌         | 50.9M/862M [00:01<25:50, 523kB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.9M/862M [00:01<19:41, 685kB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.0M/862M [00:03<15:38, 858kB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.5M/862M [00:03<11:52, 1.13MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.1M/862M [00:04<08:23, 1.59MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.2M/862M [00:05<1:24:19, 158kB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.5M/862M [00:05<1:00:02, 222kB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.3M/862M [00:07<43:49, 303kB/s]  .vector_cache/glove.6B.zip:   8%|▊         | 65.8M/862M [00:07<31:25, 422kB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.4M/862M [00:09<23:56, 552kB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.9M/862M [00:09<17:40, 747kB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.6M/862M [00:11<14:18, 918kB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.2M/862M [00:11<10:39, 1.23MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.7M/862M [00:13<09:30, 1.37MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.4M/862M [00:13<07:12, 1.81MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.8M/862M [00:15<07:09, 1.82MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.5M/862M [00:15<05:37, 2.31MB/s].vector_cache/glove.6B.zip:  10%|▉         | 86.0M/862M [00:17<05:59, 2.16MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.6M/862M [00:17<04:49, 2.68MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.1M/862M [00:19<05:24, 2.38MB/s].vector_cache/glove.6B.zip:  11%|█         | 90.8M/862M [00:19<04:19, 2.98MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.3M/862M [00:21<05:06, 2.50MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.7M/862M [00:21<04:39, 2.75MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.4M/862M [00:23<05:09, 2.47MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.7M/862M [00:23<04:51, 2.62MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:23<03:30, 3.62MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:25<16:48, 753kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:25<12:53, 981kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:25<09:05, 1.38MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:27<24:05, 523kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:27<17:58, 700kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:27<12:37, 992kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:29<47:25, 264kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:29<35:02, 357kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:29<24:31, 508kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:31<28:50, 432kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:31<21:52, 569kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:31<15:21, 807kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:33<22:01, 562kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:33<17:11, 720kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:33<12:05, 1.02MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:35<17:43, 695kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:35<13:25, 917kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:35<09:27, 1.30MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:37<23:28, 522kB/s] .vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:37<18:11, 673kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:37<13:02, 937kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<11:15, 1.08MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<08:53, 1.37MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:41<08:00, 1.51MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<06:47, 1.78MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:42<04:50, 2.48MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:43<17:31, 685kB/s] .vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<13:58, 859kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:44<09:51, 1.21MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:45<15:14, 784kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:46<12:19, 969kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:46<08:43, 1.36MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:47<11:59, 990kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:48<09:25, 1.26MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:48<06:40, 1.77MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:49<18:33, 636kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:49<13:58, 845kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:50<09:50, 1.19MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:51<17:35, 667kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:51<13:58, 840kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:52<09:50, 1.19MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:52<07:47, 1.50MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:54<12:02:26, 16.1kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:54<8:26:46, 23.0kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:54<5:52:59, 32.9kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:56<4:39:37, 41.5kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:56<3:17:04, 58.8kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:56<2:17:22, 84.0kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:57<1:51:01, 104kB/s] .vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:58<1:18:31, 147kB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [00:58<54:47, 209kB/s]  .vector_cache/glove.6B.zip:  20%|██        | 175M/862M [00:59<1:13:38, 156kB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:00<52:27, 218kB/s]  .vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:00<36:39, 311kB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:01<36:28, 312kB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:02<26:33, 429kB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:02<18:34, 609kB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:03<1:14:03, 153kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:03<52:40, 215kB/s]  .vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:04<36:47, 306kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:05<37:28, 300kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:05<26:59, 416kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:06<18:54, 592kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:07<24:43, 452kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:07<18:43, 597kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:08<13:09, 846kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:09<17:45, 626kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:09<13:20, 832kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:10<09:51, 1.12MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:11<12:05, 914kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:11<09:20, 1.18MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:11<06:36, 1.66MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:13<20:24, 538kB/s] .vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:13<15:03, 728kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:13<10:34, 1.03MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:15<32:36, 334kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:15<24:15, 449kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:15<16:58, 639kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:17<36:55, 293kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:17<26:41, 406kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:17<18:40, 577kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:19<2:32:30, 70.6kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:19<1:47:23, 100kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:21<1:16:22, 140kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:21<54:48, 195kB/s]  .vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:21<38:15, 278kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:23<39:44, 268kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:23<29:18, 363kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:23<20:30, 515kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:25<21:51, 483kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:25<16:23, 644kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:25<11:29, 913kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:27<45:46, 229kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:27<33:22, 314kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:27<23:18, 447kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:29<58:30, 178kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:29<42:02, 248kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:29<29:22, 353kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:31<29:21, 353kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:31<21:19, 485kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:31<14:56, 689kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:33<30:01, 343kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:33<21:49, 471kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:33<15:16, 669kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:35<11:12:52, 15.2kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:35<7:52:23, 21.6kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:35<5:28:58, 30.9kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:37<3:58:09, 42.6kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:37<2:47:56, 60.4kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:37<1:56:56, 86.2kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:39<1:42:12, 98.6kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:39<1:12:56, 138kB/s] .vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:39<50:51, 197kB/s]  .vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:41<47:08, 212kB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:41<34:28, 290kB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:41<24:05, 413kB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:43<26:44, 372kB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:43<19:58, 498kB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:43<14:12, 696kB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:45<8:05:44, 20.3kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:45<5:39:43, 29.0kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<3:58:20, 41.2kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:47<2:47:47, 58.5kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:47<1:56:56, 83.4kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:48<1:28:27, 110kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:49<1:02:31, 156kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [01:49<43:42, 222kB/s]  .vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:50<34:30, 280kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:51<25:30, 379kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:51<17:50, 539kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:52<20:20, 472kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:53<15:26, 622kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:53<10:50, 881kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:54<15:00, 635kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:55<11:30, 828kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:55<08:05, 1.17MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:56<14:17, 663kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:56<10:54, 867kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:57<07:42, 1.22MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:58<09:47, 960kB/s] .vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:58<08:14, 1.14MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 302M/862M [01:59<05:49, 1.60MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:00<09:59, 934kB/s] .vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:00<07:37, 1.22MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:01<05:23, 1.72MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:02<13:17, 697kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:02<09:53, 936kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:04<08:16, 1.11MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:04<06:38, 1.38MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:04<04:42, 1.94MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:06<09:46, 933kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:06<08:11, 1.11MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:06<05:47, 1.57MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:08<10:20, 876kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:08<07:56, 1.14MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:08<05:36, 1.61MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:10<18:50, 477kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 323M/862M [02:10<13:56, 644kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:10<09:46, 913kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:12<31:57, 279kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:12<23:04, 386kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:12<16:07, 549kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:14<55:20, 160kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:14<39:50, 222kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:14<27:47, 316kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:16<29:45, 295kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:16<21:44, 404kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:16<15:10, 574kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:18<1:13:18, 119kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:18<52:15, 167kB/s]  .vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:18<36:25, 237kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:20<34:20, 252kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:20<24:36, 351kB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:20<17:10, 499kB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:22<33:23, 257kB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:22<24:31, 349kB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:22<17:12, 496kB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:24<14:55, 570kB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:24<11:10, 760kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:26<09:01, 934kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:26<06:57, 1.21MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:28<06:06, 1.37MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:28<04:53, 1.71MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:30<04:39, 1.78MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:30<04:21, 1.90MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:30<03:06, 2.66MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:32<12:01, 685kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:32<09:10, 896kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:32<06:27, 1.27MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:34<18:03, 452kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:34<13:40, 597kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:34<09:34, 846kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:34<10:37, 762kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:36<6:38:02, 20.3kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:36<4:38:11, 29.0kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:37<3:15:03, 41.2kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:38<2:17:32, 58.4kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:38<1:35:41, 83.3kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:39<1:13:30, 108kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:40<52:02, 153kB/s]  .vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:41<37:20, 211kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:42<27:14, 290kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:42<19:01, 412kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:43<19:10, 408kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:44<14:28, 540kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:44<10:07, 767kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:45<14:09, 548kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:45<10:57, 707kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:46<07:42, 999kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:47<08:45, 877kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:47<07:16, 1.06MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:48<05:07, 1.49MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:49<10:35, 719kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:50<07:30, 1.01MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:51<07:01, 1.08MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:51<05:57, 1.27MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:51<04:15, 1.76MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:53<05:10, 1.44MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:53<04:43, 1.58MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:53<03:23, 2.19MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:55<04:40, 1.58MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:55<04:23, 1.69MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:55<03:09, 2.33MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:57<04:25, 1.66MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:57<04:09, 1.76MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:57<03:00, 2.43MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [02:59<04:20, 1.68MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [02:59<04:05, 1.78MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [02:59<02:56, 2.46MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:01<04:25, 1.63MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:01<04:08, 1.74MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:01<02:58, 2.41MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:03<04:29, 1.59MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:03<04:09, 1.71MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:03<02:57, 2.39MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:05<07:15, 973kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:05<05:40, 1.25MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:07<04:58, 1.40MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:07<04:02, 1.73MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:07<02:51, 2.42MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:09<20:40, 335kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:09<15:02, 460kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:11<11:27, 598kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:11<08:31, 805kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:11<05:58, 1.14MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:13<18:22, 370kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:13<13:38, 498kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:13<09:30, 707kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:15<29:34, 227kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:15<21:15, 316kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:17<15:44, 423kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:17<11:36, 573kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:19<09:01, 730kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:19<06:58, 944kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:21<05:47, 1.13MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:21<04:43, 1.38MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:21<03:20, 1.93MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:23<07:58, 808kB/s] .vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:23<06:03, 1.06MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:23<04:26, 1.44MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:25<5:01:19, 21.2kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:25<3:30:41, 30.2kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:27<2:27:17, 42.9kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:27<1:43:35, 60.9kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:27<1:11:56, 87.0kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:28<56:42, 110kB/s]   .vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:29<40:24, 155kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:29<28:04, 220kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:30<27:10, 227kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:31<19:30, 317kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:31<13:34, 450kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [03:32<21:52, 279kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:33<16:02, 381kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:33<11:10, 541kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:34<24:49, 243kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:35<17:51, 338kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:36<13:16, 450kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:36<09:50, 606kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:38<07:40, 768kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:38<05:52, 1.00MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:39<04:08, 1.41MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:40<05:58, 978kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:40<04:42, 1.24MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:41<03:19, 1.74MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:42<07:34, 761kB/s] .vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:42<05:47, 994kB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:42<04:03, 1.40MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:44<18:12, 313kB/s] .vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:44<13:26, 424kB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:44<09:21, 602kB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:46<11:16, 499kB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:46<08:25, 667kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:46<05:53, 945kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:48<10:39, 522kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:48<07:46, 714kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:50<06:14, 881kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:50<04:50, 1.13MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:50<03:23, 1.60MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:52<09:41, 560kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:52<07:14, 749kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:54<05:47, 923kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:54<04:30, 1.19MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:56<03:54, 1.35MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:56<03:08, 1.68MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:58<02:57, 1.77MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:58<02:46, 1.88MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [03:58<01:57, 2.62MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:00<13:14, 389kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:00<09:39, 532kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:02<07:26, 682kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:02<05:28, 927kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:04<04:34, 1.10MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:04<03:36, 1.39MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:06<03:14, 1.53MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:06<02:34, 1.92MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:06<01:49, 2.67MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:08<08:07, 599kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:08<06:05, 799kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:10<04:55, 975kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:10<03:51, 1.25MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:12<03:22, 1.41MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:12<02:40, 1.77MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:14<02:33, 1.82MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:14<02:09, 2.16MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:14<01:37, 2.83MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:16<3:44:44, 20.5kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:16<2:36:43, 29.2kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:17<1:49:26, 41.4kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:18<1:17:08, 58.7kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:18<53:16, 83.9kB/s]  .vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:19<1:35:05, 47.0kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:20<1:07:04, 66.5kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:21<46:54, 93.7kB/s]  .vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:22<33:13, 132kB/s] .vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:22<22:58, 189kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:23<28:15, 153kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:24<20:05, 215kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:25<14:29, 294kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:25<10:27, 406kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:27<07:51, 534kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:27<05:56, 705kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:28<04:07, 998kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:29<20:40, 199kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:29<14:45, 279kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:31<10:47, 376kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:31<07:54, 512kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:31<05:28, 727kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:33<46:49, 85.1kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:33<33:02, 120kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:35<23:21, 168kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:35<16:40, 234kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:37<12:03, 319kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:37<08:47, 437kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:39<06:36, 571kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:39<04:51, 776kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:39<03:22, 1.10MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:41<18:23, 202kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:41<13:17, 279kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:43<09:40, 376kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:43<07:08, 509kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:45<05:25, 658kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:45<04:05, 871kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:47<03:20, 1.05MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:47<02:34, 1.36MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:49<02:17, 1.50MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:49<01:50, 1.87MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:49<01:18, 2.59MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:51<02:50, 1.19MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:51<02:15, 1.49MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:51<01:34, 2.09MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:53<1:12:13, 45.6kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:53<50:39, 64.9kB/s]  .vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:55<35:18, 91.4kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:55<25:03, 129kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:55<17:17, 183kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:57<14:14, 222kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:57<10:18, 306kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 677M/862M [04:59<07:31, 411kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [04:59<05:28, 563kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [04:59<03:46, 799kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:01<41:47, 72.3kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:01<29:23, 103kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:03<20:36, 143kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 685M/862M [05:03<14:36, 202kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:03<10:07, 285kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:05<2:25:52, 19.8kB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:05<1:41:29, 28.3kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:06<1:10:22, 40.1kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:07<49:17, 57.1kB/s]  .vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:08<34:11, 80.5kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:09<24:12, 114kB/s] .vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:10<16:56, 158kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:11<12:00, 223kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:12<08:37, 303kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:13<06:11, 420kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:14<04:37, 550kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:14<03:22, 751kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:16<02:41, 920kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:16<02:03, 1.20MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:18<01:46, 1.36MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:18<01:31, 1.58MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:20<01:22, 1.70MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:20<01:07, 2.07MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:22<01:07, 2.03MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:22<00:54, 2.48MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:22<00:38, 3.43MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:24<08:22, 263kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:24<05:58, 367kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:26<04:23, 485kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:26<03:13, 659kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:28<02:30, 825kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:28<01:52, 1.09MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:28<01:17, 1.54MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:30<05:31, 361kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:30<03:59, 499kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:32<03:00, 642kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:32<02:13, 862kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:34<01:47, 1.04MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:34<01:27, 1.27MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:36<01:14, 1.44MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:36<00:59, 1.80MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:38<00:56, 1.84MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:38<00:46, 2.21MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:40<00:46, 2.14MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:40<00:39, 2.51MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:42<00:40, 2.33MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:42<00:37, 2.50MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:42<00:26, 3.47MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:44<16:36, 91.2kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:44<11:40, 129kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:46<08:03, 179kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:46<05:42, 252kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:46<03:52, 359kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:48<03:31, 391kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:48<02:32, 538kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:48<01:43, 763kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:50<03:55, 333kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:50<02:49, 460kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:52<02:04, 597kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:52<01:31, 811kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:54<01:11, 983kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:54<00:54, 1.28MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:54<00:38, 1.74MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:56<55:01, 20.1kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:56<37:56, 28.7kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:57<25:30, 40.8kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:58<17:51, 57.9kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [05:59<11:52, 81.7kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:00<08:23, 115kB/s] .vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:01<05:36, 161kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:02<03:56, 226kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:03<02:42, 308kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:04<01:57, 423kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:05<01:22, 555kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:05<01:03, 723kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:06<00:41, 1.02MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:07<00:54, 766kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:07<00:43, 964kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:08<00:27, 1.36MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:09<01:09, 537kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:09<00:50, 730kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:11<00:37, 900kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:11<00:29, 1.12MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:11<00:18, 1.57MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:13<04:44, 103kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:13<03:21, 145kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:13<02:02, 206kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:15<05:18, 79.1kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:15<03:39, 112kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:17<02:14, 157kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:17<01:33, 221kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:19<00:56, 301kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:19<00:39, 417kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 849M/862M [06:21<00:23, 546kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:21<00:17, 719kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:23<00:09, 893kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:23<00:07, 1.12MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:25<00:03, 1.30MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:25<00:02, 1.62MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:27<00:00, 1.72MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:27<00:00, 2.06MB/s].vector_cache/glove.6B.zip: 862MB [06:27, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<51:43:35,  2.15it/s]  0%|          | 917/400000 [00:00<36:07:45,  3.07it/s]  0%|          | 1822/400000 [00:00<25:14:12,  4.38it/s]  1%|          | 2728/400000 [00:00<17:37:45,  6.26it/s]  1%|          | 3622/400000 [00:00<12:18:59,  8.94it/s]  1%|          | 4501/400000 [00:00<8:36:21, 12.77it/s]   1%|▏         | 5388/400000 [00:01<6:00:52, 18.23it/s]  2%|▏         | 6272/400000 [00:01<4:12:15, 26.01it/s]  2%|▏         | 7179/400000 [00:01<2:56:23, 37.12it/s]  2%|▏         | 8077/400000 [00:01<2:03:24, 52.93it/s]  2%|▏         | 8971/400000 [00:01<1:26:24, 75.42it/s]  2%|▏         | 9879/400000 [00:01<1:00:33, 107.36it/s]  3%|▎         | 10760/400000 [00:01<42:31, 152.57it/s]   3%|▎         | 11641/400000 [00:01<29:55, 216.24it/s]  3%|▎         | 12531/400000 [00:01<21:07, 305.73it/s]  3%|▎         | 13402/400000 [00:01<14:58, 430.08it/s]  4%|▎         | 14265/400000 [00:02<10:41, 601.27it/s]  4%|▍         | 15142/400000 [00:02<07:41, 834.43it/s]  4%|▍         | 16077/400000 [00:02<05:34, 1148.11it/s]  4%|▍         | 16966/400000 [00:02<04:06, 1554.12it/s]  4%|▍         | 17852/400000 [00:02<03:05, 2063.75it/s]  5%|▍         | 18737/400000 [00:02<02:22, 2680.09it/s]  5%|▍         | 19628/400000 [00:02<01:52, 3391.41it/s]  5%|▌         | 20514/400000 [00:02<01:31, 4153.72it/s]  5%|▌         | 21397/400000 [00:02<01:17, 4886.18it/s]  6%|▌         | 22264/400000 [00:02<01:08, 5526.12it/s]  6%|▌         | 23110/400000 [00:03<01:01, 6159.74it/s]  6%|▌         | 23955/400000 [00:03<00:56, 6671.09it/s]  6%|▌         | 24795/400000 [00:03<00:53, 7066.83it/s]  6%|▋         | 25676/400000 [00:03<00:49, 7510.70it/s]  7%|▋         | 26542/400000 [00:03<00:47, 7820.51it/s]  7%|▋         | 27466/400000 [00:03<00:45, 8197.32it/s]  7%|▋         | 28341/400000 [00:03<00:44, 8343.96it/s]  7%|▋         | 29232/400000 [00:03<00:43, 8505.34it/s]  8%|▊         | 30120/400000 [00:03<00:42, 8613.27it/s]  8%|▊         | 31011/400000 [00:04<00:42, 8698.58it/s]  8%|▊         | 31895/400000 [00:04<00:42, 8611.66it/s]  8%|▊         | 32767/400000 [00:04<00:42, 8549.92it/s]  8%|▊         | 33630/400000 [00:04<00:42, 8543.07it/s]  9%|▊         | 34490/400000 [00:04<00:42, 8517.61it/s]  9%|▉         | 35368/400000 [00:04<00:42, 8592.97it/s]  9%|▉         | 36282/400000 [00:04<00:41, 8748.98it/s]  9%|▉         | 37164/400000 [00:04<00:41, 8767.82it/s] 10%|▉         | 38043/400000 [00:04<00:41, 8766.57it/s] 10%|▉         | 38933/400000 [00:04<00:41, 8804.24it/s] 10%|▉         | 39815/400000 [00:05<00:41, 8607.06it/s] 10%|█         | 40678/400000 [00:05<00:41, 8557.00it/s] 10%|█         | 41589/400000 [00:05<00:41, 8713.33it/s] 11%|█         | 42462/400000 [00:05<00:41, 8606.45it/s] 11%|█         | 43325/400000 [00:05<00:42, 8423.83it/s] 11%|█         | 44170/400000 [00:05<00:42, 8343.57it/s] 11%|█▏        | 45049/400000 [00:05<00:41, 8472.44it/s] 11%|█▏        | 45958/400000 [00:05<00:40, 8647.73it/s] 12%|█▏        | 46825/400000 [00:05<00:40, 8649.34it/s] 12%|█▏        | 47716/400000 [00:05<00:40, 8723.32it/s] 12%|█▏        | 48599/400000 [00:06<00:40, 8754.84it/s] 12%|█▏        | 49496/400000 [00:06<00:39, 8817.51it/s] 13%|█▎        | 50429/400000 [00:06<00:38, 8964.91it/s] 13%|█▎        | 51343/400000 [00:06<00:38, 9016.60it/s] 13%|█▎        | 52300/400000 [00:06<00:37, 9175.00it/s] 13%|█▎        | 53219/400000 [00:06<00:37, 9131.21it/s] 14%|█▎        | 54135/400000 [00:06<00:37, 9138.99it/s] 14%|█▍        | 55084/400000 [00:06<00:37, 9240.48it/s] 14%|█▍        | 56009/400000 [00:06<00:37, 9121.88it/s] 14%|█▍        | 56936/400000 [00:06<00:37, 9163.82it/s] 14%|█▍        | 57854/400000 [00:07<00:37, 9028.45it/s] 15%|█▍        | 58758/400000 [00:07<00:38, 8963.60it/s] 15%|█▍        | 59664/400000 [00:07<00:37, 8990.12it/s] 15%|█▌        | 60574/400000 [00:07<00:37, 9022.33it/s] 15%|█▌        | 61477/400000 [00:07<00:38, 8733.46it/s] 16%|█▌        | 62375/400000 [00:07<00:38, 8803.92it/s] 16%|█▌        | 63289/400000 [00:07<00:37, 8899.68it/s] 16%|█▌        | 64201/400000 [00:07<00:37, 8963.25it/s] 16%|█▋        | 65099/400000 [00:07<00:37, 8891.71it/s] 17%|█▋        | 66004/400000 [00:07<00:37, 8936.42it/s] 17%|█▋        | 66899/400000 [00:08<00:37, 8896.41it/s] 17%|█▋        | 67791/400000 [00:08<00:37, 8900.55it/s] 17%|█▋        | 68683/400000 [00:08<00:37, 8906.19it/s] 17%|█▋        | 69581/400000 [00:08<00:37, 8926.49it/s] 18%|█▊        | 70474/400000 [00:08<00:37, 8905.89it/s] 18%|█▊        | 71365/400000 [00:08<00:37, 8870.92it/s] 18%|█▊        | 72277/400000 [00:08<00:36, 8942.44it/s] 18%|█▊        | 73205/400000 [00:08<00:36, 9038.50it/s] 19%|█▊        | 74121/400000 [00:08<00:35, 9071.70it/s] 19%|█▉        | 75029/400000 [00:08<00:36, 9008.35it/s] 19%|█▉        | 75931/400000 [00:09<00:36, 8943.65it/s] 19%|█▉        | 76857/400000 [00:09<00:35, 9034.57it/s] 19%|█▉        | 77765/400000 [00:09<00:35, 9047.20it/s] 20%|█▉        | 78676/400000 [00:09<00:35, 9064.17it/s] 20%|█▉        | 79631/400000 [00:09<00:34, 9202.29it/s] 20%|██        | 80552/400000 [00:09<00:35, 8952.41it/s] 20%|██        | 81450/400000 [00:09<00:36, 8661.25it/s] 21%|██        | 82320/400000 [00:09<00:37, 8576.73it/s] 21%|██        | 83181/400000 [00:09<00:37, 8496.44it/s] 21%|██        | 84033/400000 [00:10<00:37, 8484.73it/s] 21%|██        | 84883/400000 [00:10<00:37, 8375.25it/s] 21%|██▏       | 85722/400000 [00:10<00:38, 8205.82it/s] 22%|██▏       | 86614/400000 [00:10<00:37, 8405.93it/s] 22%|██▏       | 87495/400000 [00:10<00:36, 8522.22it/s] 22%|██▏       | 88406/400000 [00:10<00:35, 8688.38it/s] 22%|██▏       | 89309/400000 [00:10<00:35, 8786.01it/s] 23%|██▎       | 90198/400000 [00:10<00:35, 8814.24it/s] 23%|██▎       | 91091/400000 [00:10<00:34, 8845.93it/s] 23%|██▎       | 92010/400000 [00:10<00:34, 8945.87it/s] 23%|██▎       | 92906/400000 [00:11<00:34, 8927.04it/s] 23%|██▎       | 93800/400000 [00:11<00:34, 8912.74it/s] 24%|██▎       | 94692/400000 [00:11<00:34, 8811.82it/s] 24%|██▍       | 95574/400000 [00:11<00:35, 8682.59it/s] 24%|██▍       | 96465/400000 [00:11<00:34, 8748.02it/s] 24%|██▍       | 97377/400000 [00:11<00:34, 8855.38it/s] 25%|██▍       | 98321/400000 [00:11<00:33, 9018.48it/s] 25%|██▍       | 99238/400000 [00:11<00:33, 9062.98it/s] 25%|██▌       | 100150/400000 [00:11<00:33, 9077.19it/s] 25%|██▌       | 101059/400000 [00:11<00:33, 8832.21it/s] 25%|██▌       | 101945/400000 [00:12<00:34, 8734.58it/s] 26%|██▌       | 102821/400000 [00:12<00:34, 8632.35it/s] 26%|██▌       | 103686/400000 [00:12<00:34, 8580.94it/s] 26%|██▌       | 104546/400000 [00:12<00:34, 8559.79it/s] 26%|██▋       | 105403/400000 [00:12<00:35, 8343.93it/s] 27%|██▋       | 106332/400000 [00:12<00:34, 8604.71it/s] 27%|██▋       | 107232/400000 [00:12<00:33, 8717.70it/s] 27%|██▋       | 108107/400000 [00:12<00:33, 8694.02it/s] 27%|██▋       | 108979/400000 [00:12<00:33, 8657.59it/s] 27%|██▋       | 109847/400000 [00:12<00:33, 8595.10it/s] 28%|██▊       | 110708/400000 [00:13<00:33, 8571.34it/s] 28%|██▊       | 111566/400000 [00:13<00:33, 8571.97it/s] 28%|██▊       | 112438/400000 [00:13<00:33, 8613.12it/s] 28%|██▊       | 113300/400000 [00:13<00:33, 8601.07it/s] 29%|██▊       | 114161/400000 [00:13<00:33, 8585.13it/s] 29%|██▉       | 115031/400000 [00:13<00:33, 8618.50it/s] 29%|██▉       | 115894/400000 [00:13<00:33, 8535.33it/s] 29%|██▉       | 116748/400000 [00:13<00:33, 8484.48it/s] 29%|██▉       | 117610/400000 [00:13<00:33, 8522.42it/s] 30%|██▉       | 118463/400000 [00:13<00:33, 8448.06it/s] 30%|██▉       | 119406/400000 [00:14<00:32, 8718.68it/s] 30%|███       | 120364/400000 [00:14<00:31, 8958.51it/s] 30%|███       | 121332/400000 [00:14<00:30, 9162.67it/s] 31%|███       | 122317/400000 [00:14<00:29, 9357.10it/s] 31%|███       | 123294/400000 [00:14<00:29, 9474.73it/s] 31%|███       | 124262/400000 [00:14<00:28, 9534.59it/s] 31%|███▏      | 125218/400000 [00:14<00:29, 9367.15it/s] 32%|███▏      | 126157/400000 [00:14<00:29, 9213.77it/s] 32%|███▏      | 127114/400000 [00:14<00:29, 9315.68it/s] 32%|███▏      | 128048/400000 [00:14<00:29, 9174.29it/s] 32%|███▏      | 129045/400000 [00:15<00:28, 9397.11it/s] 33%|███▎      | 130027/400000 [00:15<00:28, 9518.43it/s] 33%|███▎      | 130981/400000 [00:15<00:28, 9407.06it/s] 33%|███▎      | 131924/400000 [00:15<00:28, 9301.72it/s] 33%|███▎      | 132856/400000 [00:15<00:28, 9288.16it/s] 33%|███▎      | 133848/400000 [00:15<00:28, 9467.09it/s] 34%|███▎      | 134797/400000 [00:15<00:28, 9448.25it/s] 34%|███▍      | 135752/400000 [00:15<00:27, 9476.25it/s] 34%|███▍      | 136725/400000 [00:15<00:27, 9549.03it/s] 34%|███▍      | 137692/400000 [00:16<00:27, 9582.33it/s] 35%|███▍      | 138742/400000 [00:16<00:26, 9839.86it/s] 35%|███▍      | 139784/400000 [00:16<00:26, 10005.44it/s] 35%|███▌      | 140787/400000 [00:16<00:26, 9840.71it/s]  35%|███▌      | 141774/400000 [00:16<00:26, 9823.48it/s] 36%|███▌      | 142758/400000 [00:16<00:26, 9662.68it/s] 36%|███▌      | 143726/400000 [00:16<00:26, 9636.57it/s] 36%|███▌      | 144691/400000 [00:16<00:26, 9479.39it/s] 36%|███▋      | 145641/400000 [00:16<00:26, 9435.18it/s] 37%|███▋      | 146595/400000 [00:16<00:26, 9464.37it/s] 37%|███▋      | 147543/400000 [00:17<00:26, 9404.17it/s] 37%|███▋      | 148530/400000 [00:17<00:26, 9537.01it/s] 37%|███▋      | 149514/400000 [00:17<00:26, 9625.58it/s] 38%|███▊      | 150478/400000 [00:17<00:26, 9467.01it/s] 38%|███▊      | 151474/400000 [00:17<00:25, 9608.05it/s] 38%|███▊      | 152437/400000 [00:17<00:25, 9579.45it/s] 38%|███▊      | 153396/400000 [00:17<00:25, 9552.74it/s] 39%|███▊      | 154352/400000 [00:17<00:25, 9551.71it/s] 39%|███▉      | 155308/400000 [00:17<00:25, 9527.54it/s] 39%|███▉      | 156262/400000 [00:17<00:25, 9513.75it/s] 39%|███▉      | 157214/400000 [00:18<00:26, 9295.90it/s] 40%|███▉      | 158182/400000 [00:18<00:25, 9406.16it/s] 40%|███▉      | 159144/400000 [00:18<00:25, 9468.63it/s] 40%|████      | 160092/400000 [00:18<00:25, 9445.45it/s] 40%|████      | 161048/400000 [00:18<00:25, 9478.44it/s] 40%|████      | 161997/400000 [00:18<00:25, 9411.78it/s] 41%|████      | 162957/400000 [00:18<00:25, 9465.22it/s] 41%|████      | 163904/400000 [00:18<00:25, 9420.37it/s] 41%|████      | 164847/400000 [00:18<00:25, 9403.64it/s] 41%|████▏     | 165808/400000 [00:18<00:24, 9463.54it/s] 42%|████▏     | 166755/400000 [00:19<00:24, 9415.98it/s] 42%|████▏     | 167727/400000 [00:19<00:24, 9502.81it/s] 42%|████▏     | 168727/400000 [00:19<00:23, 9645.38it/s] 42%|████▏     | 169693/400000 [00:19<00:24, 9569.55it/s] 43%|████▎     | 170670/400000 [00:19<00:23, 9627.93it/s] 43%|████▎     | 171634/400000 [00:19<00:24, 9474.69it/s] 43%|████▎     | 172583/400000 [00:19<00:24, 9372.58it/s] 43%|████▎     | 173547/400000 [00:19<00:23, 9448.54it/s] 44%|████▎     | 174493/400000 [00:19<00:24, 9162.60it/s] 44%|████▍     | 175412/400000 [00:19<00:24, 9142.44it/s] 44%|████▍     | 176328/400000 [00:20<00:24, 8991.64it/s] 44%|████▍     | 177243/400000 [00:20<00:24, 9036.91it/s] 45%|████▍     | 178148/400000 [00:20<00:25, 8854.10it/s] 45%|████▍     | 179058/400000 [00:20<00:24, 8924.31it/s] 45%|████▌     | 180062/400000 [00:20<00:23, 9231.90it/s] 45%|████▌     | 181011/400000 [00:20<00:23, 9304.97it/s] 45%|████▌     | 181969/400000 [00:20<00:23, 9384.54it/s] 46%|████▌     | 182995/400000 [00:20<00:22, 9630.47it/s] 46%|████▌     | 183985/400000 [00:20<00:22, 9709.49it/s] 46%|████▌     | 184959/400000 [00:20<00:22, 9619.30it/s] 46%|████▋     | 185923/400000 [00:21<00:22, 9405.64it/s] 47%|████▋     | 186866/400000 [00:21<00:22, 9392.80it/s] 47%|████▋     | 187836/400000 [00:21<00:22, 9480.85it/s] 47%|████▋     | 188819/400000 [00:21<00:22, 9581.28it/s] 47%|████▋     | 189800/400000 [00:21<00:21, 9646.86it/s] 48%|████▊     | 190766/400000 [00:21<00:21, 9627.05it/s] 48%|████▊     | 191730/400000 [00:21<00:21, 9593.83it/s] 48%|████▊     | 192690/400000 [00:21<00:21, 9588.69it/s] 48%|████▊     | 193655/400000 [00:21<00:21, 9605.18it/s] 49%|████▊     | 194632/400000 [00:22<00:21, 9652.00it/s] 49%|████▉     | 195598/400000 [00:22<00:21, 9555.10it/s] 49%|████▉     | 196582/400000 [00:22<00:21, 9637.67it/s] 49%|████▉     | 197552/400000 [00:22<00:20, 9655.99it/s] 50%|████▉     | 198518/400000 [00:22<00:20, 9595.09it/s] 50%|████▉     | 199480/400000 [00:22<00:20, 9599.10it/s] 50%|█████     | 200448/400000 [00:22<00:20, 9621.26it/s] 50%|█████     | 201411/400000 [00:22<00:20, 9583.00it/s] 51%|█████     | 202370/400000 [00:22<00:20, 9578.89it/s] 51%|█████     | 203331/400000 [00:22<00:20, 9586.45it/s] 51%|█████     | 204290/400000 [00:23<00:20, 9559.91it/s] 51%|█████▏    | 205247/400000 [00:23<00:20, 9554.50it/s] 52%|█████▏    | 206211/400000 [00:23<00:20, 9578.43it/s] 52%|█████▏    | 207169/400000 [00:23<00:20, 9543.14it/s] 52%|█████▏    | 208124/400000 [00:23<00:20, 9280.19it/s] 52%|█████▏    | 209069/400000 [00:23<00:20, 9329.67it/s] 53%|█████▎    | 210004/400000 [00:23<00:20, 9143.97it/s] 53%|█████▎    | 210921/400000 [00:23<00:21, 8901.00it/s] 53%|█████▎    | 211931/400000 [00:23<00:20, 9228.72it/s] 53%|█████▎    | 212873/400000 [00:23<00:20, 9283.13it/s] 53%|█████▎    | 213863/400000 [00:24<00:19, 9458.66it/s] 54%|█████▎    | 214813/400000 [00:24<00:19, 9277.31it/s] 54%|█████▍    | 215779/400000 [00:24<00:19, 9387.98it/s] 54%|█████▍    | 216780/400000 [00:24<00:19, 9565.21it/s] 54%|█████▍    | 217784/400000 [00:24<00:18, 9699.84it/s] 55%|█████▍    | 218883/400000 [00:24<00:18, 10052.08it/s] 55%|█████▍    | 219926/400000 [00:24<00:17, 10162.54it/s] 55%|█████▌    | 220946/400000 [00:24<00:18, 9789.33it/s]  55%|█████▌    | 221931/400000 [00:24<00:18, 9588.28it/s] 56%|█████▌    | 222897/400000 [00:24<00:18, 9607.81it/s] 56%|█████▌    | 223862/400000 [00:25<00:18, 9552.03it/s] 56%|█████▌    | 224876/400000 [00:25<00:18, 9718.87it/s] 56%|█████▋    | 225970/400000 [00:25<00:17, 10053.84it/s] 57%|█████▋    | 227052/400000 [00:25<00:16, 10270.92it/s] 57%|█████▋    | 228159/400000 [00:25<00:16, 10496.92it/s] 57%|█████▋    | 229257/400000 [00:25<00:16, 10636.84it/s] 58%|█████▊    | 230325/400000 [00:25<00:16, 10557.97it/s] 58%|█████▊    | 231402/400000 [00:25<00:15, 10620.64it/s] 58%|█████▊    | 232532/400000 [00:25<00:15, 10814.77it/s] 58%|█████▊    | 233616/400000 [00:25<00:15, 10560.26it/s] 59%|█████▊    | 234675/400000 [00:26<00:16, 9967.19it/s]  59%|█████▉    | 235681/400000 [00:26<00:16, 9811.82it/s] 59%|█████▉    | 236669/400000 [00:26<00:16, 9623.83it/s] 59%|█████▉    | 237692/400000 [00:26<00:16, 9797.09it/s] 60%|█████▉    | 238677/400000 [00:26<00:16, 9687.22it/s] 60%|█████▉    | 239733/400000 [00:26<00:16, 9932.55it/s] 60%|██████    | 240808/400000 [00:26<00:15, 10162.10it/s] 60%|██████    | 241882/400000 [00:26<00:15, 10327.16it/s] 61%|██████    | 242972/400000 [00:26<00:14, 10491.15it/s] 61%|██████    | 244038/400000 [00:27<00:14, 10539.91it/s] 61%|██████▏   | 245095/400000 [00:27<00:14, 10529.39it/s] 62%|██████▏   | 246185/400000 [00:27<00:14, 10635.78it/s] 62%|██████▏   | 247250/400000 [00:27<00:14, 10618.78it/s] 62%|██████▏   | 248313/400000 [00:27<00:14, 10409.01it/s] 62%|██████▏   | 249356/400000 [00:27<00:15, 9940.71it/s]  63%|██████▎   | 250356/400000 [00:27<00:15, 9608.68it/s] 63%|██████▎   | 251341/400000 [00:27<00:15, 9677.33it/s] 63%|██████▎   | 252314/400000 [00:27<00:15, 9678.60it/s] 63%|██████▎   | 253286/400000 [00:27<00:15, 9475.25it/s] 64%|██████▎   | 254237/400000 [00:28<00:15, 9402.17it/s] 64%|██████▍   | 255180/400000 [00:28<00:15, 9258.60it/s] 64%|██████▍   | 256193/400000 [00:28<00:15, 9501.57it/s] 64%|██████▍   | 257147/400000 [00:28<00:15, 9443.31it/s] 65%|██████▍   | 258181/400000 [00:28<00:14, 9692.23it/s] 65%|██████▍   | 259154/400000 [00:28<00:14, 9640.69it/s] 65%|██████▌   | 260122/400000 [00:28<00:14, 9651.94it/s] 65%|██████▌   | 261142/400000 [00:28<00:14, 9807.76it/s] 66%|██████▌   | 262125/400000 [00:28<00:14, 9783.42it/s] 66%|██████▌   | 263105/400000 [00:28<00:14, 9727.06it/s] 66%|██████▌   | 264079/400000 [00:29<00:14, 9630.01it/s] 66%|██████▋   | 265043/400000 [00:29<00:14, 9575.07it/s] 67%|██████▋   | 266002/400000 [00:29<00:14, 9460.04it/s] 67%|██████▋   | 267008/400000 [00:29<00:13, 9630.85it/s] 67%|██████▋   | 268125/400000 [00:29<00:13, 10045.86it/s] 67%|██████▋   | 269220/400000 [00:29<00:12, 10298.62it/s] 68%|██████▊   | 270256/400000 [00:29<00:12, 10260.78it/s] 68%|██████▊   | 271352/400000 [00:29<00:12, 10460.31it/s] 68%|██████▊   | 272402/400000 [00:29<00:12, 10304.82it/s] 68%|██████▊   | 273450/400000 [00:30<00:12, 10354.22it/s] 69%|██████▊   | 274488/400000 [00:30<00:12, 10110.12it/s] 69%|██████▉   | 275502/400000 [00:30<00:12, 10084.64it/s] 69%|██████▉   | 276513/400000 [00:30<00:12, 9931.19it/s]  69%|██████▉   | 277509/400000 [00:30<00:12, 9539.36it/s] 70%|██████▉   | 278552/400000 [00:30<00:12, 9788.14it/s] 70%|██████▉   | 279545/400000 [00:30<00:12, 9829.76it/s] 70%|███████   | 280532/400000 [00:30<00:12, 9793.54it/s] 70%|███████   | 281514/400000 [00:30<00:12, 9785.91it/s] 71%|███████   | 282508/400000 [00:30<00:11, 9828.16it/s] 71%|███████   | 283494/400000 [00:31<00:11, 9837.40it/s] 71%|███████   | 284499/400000 [00:31<00:11, 9898.45it/s] 71%|███████▏  | 285490/400000 [00:31<00:11, 9801.92it/s] 72%|███████▏  | 286546/400000 [00:31<00:11, 10014.73it/s] 72%|███████▏  | 287579/400000 [00:31<00:11, 10104.52it/s] 72%|███████▏  | 288657/400000 [00:31<00:10, 10295.98it/s] 72%|███████▏  | 289745/400000 [00:31<00:10, 10462.85it/s] 73%|███████▎  | 290794/400000 [00:31<00:10, 10462.03it/s] 73%|███████▎  | 291892/400000 [00:31<00:10, 10610.94it/s] 73%|███████▎  | 292982/400000 [00:31<00:10, 10695.40it/s] 74%|███████▎  | 294101/400000 [00:32<00:09, 10836.85it/s] 74%|███████▍  | 295186/400000 [00:32<00:09, 10625.75it/s] 74%|███████▍  | 296251/400000 [00:32<00:09, 10615.08it/s] 74%|███████▍  | 297315/400000 [00:32<00:09, 10622.32it/s] 75%|███████▍  | 298379/400000 [00:32<00:09, 10414.40it/s] 75%|███████▍  | 299441/400000 [00:32<00:09, 10473.67it/s] 75%|███████▌  | 300528/400000 [00:32<00:09, 10588.48it/s] 75%|███████▌  | 301602/400000 [00:32<00:09, 10633.45it/s] 76%|███████▌  | 302677/400000 [00:32<00:09, 10665.57it/s] 76%|███████▌  | 303745/400000 [00:32<00:09, 10552.25it/s] 76%|███████▌  | 304903/400000 [00:33<00:08, 10839.59it/s] 77%|███████▋  | 306015/400000 [00:33<00:08, 10918.97it/s] 77%|███████▋  | 307109/400000 [00:33<00:08, 10746.06it/s] 77%|███████▋  | 308186/400000 [00:33<00:08, 10650.10it/s] 77%|███████▋  | 309253/400000 [00:33<00:08, 10551.12it/s] 78%|███████▊  | 310310/400000 [00:33<00:08, 10525.49it/s] 78%|███████▊  | 311364/400000 [00:33<00:08, 10478.93it/s] 78%|███████▊  | 312413/400000 [00:33<00:08, 10357.15it/s] 78%|███████▊  | 313450/400000 [00:33<00:08, 9687.35it/s]  79%|███████▊  | 314429/400000 [00:34<00:08, 9577.16it/s] 79%|███████▉  | 315394/400000 [00:34<00:08, 9411.24it/s] 79%|███████▉  | 316341/400000 [00:34<00:08, 9392.27it/s] 79%|███████▉  | 317285/400000 [00:34<00:08, 9264.27it/s] 80%|███████▉  | 318215/400000 [00:34<00:08, 9203.73it/s] 80%|███████▉  | 319138/400000 [00:34<00:09, 8965.23it/s] 80%|████████  | 320038/400000 [00:34<00:09, 8875.62it/s] 80%|████████  | 320928/400000 [00:34<00:09, 8682.89it/s] 80%|████████  | 321833/400000 [00:34<00:08, 8787.37it/s] 81%|████████  | 322728/400000 [00:34<00:08, 8833.69it/s] 81%|████████  | 323639/400000 [00:35<00:08, 8912.55it/s] 81%|████████  | 324553/400000 [00:35<00:08, 8977.61it/s] 81%|████████▏ | 325452/400000 [00:35<00:08, 8980.42it/s] 82%|████████▏ | 326357/400000 [00:35<00:08, 8999.20it/s] 82%|████████▏ | 327258/400000 [00:35<00:08, 8975.22it/s] 82%|████████▏ | 328174/400000 [00:35<00:07, 9028.06it/s] 82%|████████▏ | 329136/400000 [00:35<00:07, 9195.13it/s] 83%|████████▎ | 330102/400000 [00:35<00:07, 9328.59it/s] 83%|████████▎ | 331037/400000 [00:35<00:07, 9203.48it/s] 83%|████████▎ | 331976/400000 [00:35<00:07, 9256.38it/s] 83%|████████▎ | 332946/400000 [00:36<00:07, 9384.41it/s] 83%|████████▎ | 333950/400000 [00:36<00:06, 9570.94it/s] 84%|████████▎ | 334946/400000 [00:36<00:06, 9683.57it/s] 84%|████████▍ | 335916/400000 [00:36<00:06, 9492.66it/s] 84%|████████▍ | 336868/400000 [00:36<00:06, 9419.50it/s] 84%|████████▍ | 337824/400000 [00:36<00:06, 9458.78it/s] 85%|████████▍ | 338771/400000 [00:36<00:06, 8931.70it/s] 85%|████████▍ | 339793/400000 [00:36<00:06, 9281.58it/s] 85%|████████▌ | 340766/400000 [00:36<00:06, 9409.12it/s] 85%|████████▌ | 341779/400000 [00:36<00:06, 9613.00it/s] 86%|████████▌ | 342876/400000 [00:37<00:05, 9980.73it/s] 86%|████████▌ | 343922/400000 [00:37<00:05, 10119.24it/s] 86%|████████▌ | 344940/400000 [00:37<00:05, 10102.44it/s] 86%|████████▋ | 345955/400000 [00:37<00:05, 9955.42it/s]  87%|████████▋ | 346997/400000 [00:37<00:05, 10087.23it/s] 87%|████████▋ | 348033/400000 [00:37<00:05, 10166.27it/s] 87%|████████▋ | 349101/400000 [00:37<00:04, 10313.81it/s] 88%|████████▊ | 350190/400000 [00:37<00:04, 10477.98it/s] 88%|████████▊ | 351240/400000 [00:37<00:04, 10448.80it/s] 88%|████████▊ | 352287/400000 [00:37<00:04, 10334.45it/s] 88%|████████▊ | 353322/400000 [00:38<00:04, 10338.14it/s] 89%|████████▊ | 354373/400000 [00:38<00:04, 10388.95it/s] 89%|████████▉ | 355450/400000 [00:38<00:04, 10498.35it/s] 89%|████████▉ | 356501/400000 [00:38<00:04, 10145.95it/s] 89%|████████▉ | 357519/400000 [00:38<00:04, 10118.42it/s] 90%|████████▉ | 358557/400000 [00:38<00:04, 10194.09it/s] 90%|████████▉ | 359611/400000 [00:38<00:03, 10295.36it/s] 90%|█████████ | 360688/400000 [00:38<00:03, 10430.26it/s] 90%|█████████ | 361733/400000 [00:38<00:03, 10048.76it/s] 91%|█████████ | 362804/400000 [00:39<00:03, 10235.51it/s] 91%|█████████ | 363907/400000 [00:39<00:03, 10460.82it/s] 91%|█████████ | 364988/400000 [00:39<00:03, 10561.28it/s] 92%|█████████▏| 366073/400000 [00:39<00:03, 10644.77it/s] 92%|█████████▏| 367140/400000 [00:39<00:03, 10505.41it/s] 92%|█████████▏| 368216/400000 [00:39<00:03, 10578.64it/s] 92%|█████████▏| 369280/400000 [00:39<00:02, 10596.39it/s] 93%|█████████▎| 370341/400000 [00:39<00:02, 10477.11it/s] 93%|█████████▎| 371390/400000 [00:39<00:02, 10435.32it/s] 93%|█████████▎| 372435/400000 [00:39<00:02, 10429.48it/s] 93%|█████████▎| 373510/400000 [00:40<00:02, 10523.18it/s] 94%|█████████▎| 374563/400000 [00:40<00:02, 10503.57it/s] 94%|█████████▍| 375640/400000 [00:40<00:02, 10580.62it/s] 94%|█████████▍| 376719/400000 [00:40<00:02, 10642.02it/s] 94%|█████████▍| 377784/400000 [00:40<00:02, 10376.49it/s] 95%|█████████▍| 378824/400000 [00:40<00:02, 10369.10it/s] 95%|█████████▍| 379887/400000 [00:40<00:01, 10444.61it/s] 95%|█████████▌| 380933/400000 [00:40<00:01, 10428.88it/s] 96%|█████████▌| 382050/400000 [00:40<00:01, 10640.65it/s] 96%|█████████▌| 383116/400000 [00:40<00:01, 10419.71it/s] 96%|█████████▌| 384161/400000 [00:41<00:01, 10274.81it/s] 96%|█████████▋| 385241/400000 [00:41<00:01, 10424.84it/s] 97%|█████████▋| 386321/400000 [00:41<00:01, 10533.90it/s] 97%|█████████▋| 387376/400000 [00:41<00:01, 10505.17it/s] 97%|█████████▋| 388428/400000 [00:41<00:01, 10366.20it/s] 97%|█████████▋| 389490/400000 [00:41<00:01, 10438.50it/s] 98%|█████████▊| 390535/400000 [00:41<00:00, 10347.46it/s] 98%|█████████▊| 391571/400000 [00:41<00:00, 10057.35it/s] 98%|█████████▊| 392580/400000 [00:41<00:00, 9907.56it/s]  98%|█████████▊| 393589/400000 [00:41<00:00, 9961.23it/s] 99%|█████████▊| 394712/400000 [00:42<00:00, 10309.81it/s] 99%|█████████▉| 395748/400000 [00:42<00:00, 10089.09it/s] 99%|█████████▉| 396803/400000 [00:42<00:00, 10215.30it/s] 99%|█████████▉| 397875/400000 [00:42<00:00, 10361.25it/s]100%|█████████▉| 398969/400000 [00:42<00:00, 10527.15it/s]100%|█████████▉| 399999/400000 [00:42<00:00, 9394.79it/s] Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f9abb865b00>, <torchtext.data.dataset.TabularDataset object at 0x7f9abb865c50>, <torchtext.vocab.Vocab object at 0x7f9abb865b70>), {}) 

  




 #################################################################################################### 

  /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/transformer_sentence.json 
 

  #####  Load JSON data_pars 

  {
  "data_info":{

  },
  "preprocessors":[
    {
      "name":"DataReader",
      "uri":"mlmodels.model_tch.util_transformer:TransformerDataReader",
      "args":{
        "train":{
          "uri":"sentence_transformers.readers:NLIDataReader",
          "dataset":"dataset\/text\/AllNLI\/train"
        },
        "test":{
          "uri":"sentence_transformers.readers:STSBenchmarkDataReader",
          "dataset":"dataset\/text\/stsbenchmark\/val"
        }
      }
    }
  ]
} 

  
 #####  Load DataLoader  

  
 #####  compute DataLoader  

  URL:  mlmodels.model_tch.util_transformer:TransformerDataReader {'train': {'uri': 'sentence_transformers.readers:NLIDataReader', 'dataset': 'dataset/text/AllNLI/train'}, 'test': {'uri': 'sentence_transformers.readers:STSBenchmarkDataReader', 'dataset': 'dataset/text/stsbenchmark/val'}} 

  
###### load_callable_from_uri LOADED <class 'mlmodels.model_tch.util_transformer.TransformerDataReader'> 
cls_name : TransformerDataReader

  
 Object Creation 

  
 Object Compute 
Error /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/transformer_sentence.json Module ['sentence_transformers.readers', 'STSBenchmarkDataReader'] notfound, module 'sentence_transformers.readers' has no attribute 'STSBenchmarkDataReader', tuple index out of range

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/util.py", line 672, in load_function_uri
    return  getattr(importlib.import_module(package), name)
AttributeError: module 'sentence_transformers.readers' has no attribute 'STSBenchmarkDataReader'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/util.py", line 683, in load_function_uri
    package_name = str(Path(package).parts[-2]) + "." + str(model_name)
IndexError: tuple index out of range

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 452, in test_json_list
    loader.compute()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 275, in compute
    obj_preprocessor.compute(input_tmp)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/util_transformer.py", line 275, in compute
    test_func = load_function(self.test_reader)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/util.py", line 688, in load_function_uri
    raise NameError(f"Module {pkg} notfound, {e1}, {e2}")
NameError: Module ['sentence_transformers.readers', 'STSBenchmarkDataReader'] notfound, module 'sentence_transformers.readers' has no attribute 'STSBenchmarkDataReader', tuple index out of range





 ********************************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/preprocess/generic.py --do test  

  #### Test unit Dataloader/Dataset   #################################### 

  


 #################### tf_dataset_download 

  tf_dataset_download mlmodels/preprocess/generic:tf_dataset_download {'train_samples': 500, 'test_samples': 500} 

  MNIST 

  Dataset Name is :  mnist 
WARNING:absl:Dataset mnist is hosted on GCS. It will automatically be downloaded to your
local data directory. If you'd instead prefer to read directly from our public
GCS bucket (recommended if you're running on GCP), you can instead set
data_dir=gs://tfds-data/datasets.

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  8.54 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  8.54 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  8.54 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  8.54 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  7.33 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  7.33 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.03 file/s]2020-07-21 00:17:59.198338: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-21 00:17:59.202003: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095190000 Hz
2020-07-21 00:17:59.202157: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x560cae3d23c0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-21 00:17:59.202170: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
[1mDownloading and preparing dataset mnist/3.0.1 (download: 11.06 MiB, generated: 21.00 MiB, total: 32.06 MiB) to /home/runner/tensorflow_datasets/mnist/3.0.1...[0m

[1mDataset mnist downloaded and prepared to /home/runner/tensorflow_datasets/mnist/3.0.1. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/ ['mnist_dataset_small.npy', 'cifar10', 'test', 'train', 'mnist2', 'fashion-mnist_small.npy'] 

  


 #################### get_dataset_torch 

  get_dataset_torch mlmodels/preprocess/generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s]  0%|          | 49152/9912422 [00:00<00:33, 296731.60it/s]  2%|▏         | 212992/9912422 [00:00<00:25, 383347.94it/s]  9%|▉         | 876544/9912422 [00:00<00:17, 529829.02it/s] 31%|███       | 3055616/9912422 [00:00<00:09, 746810.79it/s] 59%|█████▉    | 5865472/9912422 [00:00<00:03, 1051028.65it/s] 90%|█████████ | 8953856/9912422 [00:01<00:00, 1474313.11it/s]9920512it [00:01, 8903078.89it/s]                             
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 138872.11it/s]           
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:05, 291910.70it/s] 13%|█▎        | 212992/1648877 [00:00<00:03, 378083.75it/s] 53%|█████▎    | 876544/1648877 [00:00<00:01, 524137.77it/s]1654784it [00:00, 2525793.97it/s]                           
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 47950.36it/s]            Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
Extracting dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz to dataset/vision/MNIST/raw/train-labels-idx1-ubyte.gz
Extracting dataset/vision/MNIST/raw/train-labels-idx1-ubyte.gz to dataset/vision/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/t10k-images-idx3-ubyte.gz
Extracting dataset/vision/MNIST/raw/t10k-images-idx3-ubyte.gz to dataset/vision/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to dataset/vision/MNIST/raw/t10k-labels-idx1-ubyte.gz
Extracting dataset/vision/MNIST/raw/t10k-labels-idx1-ubyte.gz to dataset/vision/MNIST/raw
Processing...
Done!

  


 #################### PandasDataset 

  PandasDataset mlmodels/preprocess/generic:pandasDataset {'colX': ['colX'], 'coly': ['coly'], 'encoding': 'ISO-8859-1', 'read_csv_parm': {'usecols': [0, 1], 'names': ['coly', 'colX'], 'encoding': 'ISO-8859-1'}} 

  


 #################### NumpyDataset 

  NumpyDataset mlmodels/preprocess/generic:NumpyDataset {'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'numpy_loader_args': {}} 

Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/train/mnist.npz
