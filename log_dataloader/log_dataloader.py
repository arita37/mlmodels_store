
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f40f2dc7a60> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f40f2dc7a60> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f415da32510> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f415da32510> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f417cc81ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f417cc81ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f410ad65950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f410ad65950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f410ad65950> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 35%|███▍      | 3440640/9912422 [00:00<00:00, 33254654.53it/s]9920512it [00:00, 30662015.74it/s]                             
0it [00:00, ?it/s]32768it [00:00, 571398.09it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 161058.13it/s]1654784it [00:00, 11330388.07it/s]                         
0it [00:00, ?it/s]8192it [00:00, 196205.70it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f41092af5f8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f41092d6898>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f410ad65598> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f410ad65598> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f410ad65598> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:43,  1.55 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:43,  1.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:43,  1.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:42,  1.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:41,  1.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:41,  1.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:40,  1.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:39,  1.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<01:10,  2.19 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<01:10,  2.19 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<01:09,  2.19 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<01:09,  2.19 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<01:08,  2.19 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<01:08,  2.19 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<01:07,  2.19 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<01:07,  2.19 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<01:06,  2.19 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<01:06,  2.19 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<01:06,  2.19 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  11%|█         | 18/162 [00:00<00:46,  3.10 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:46,  3.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:46,  3.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:45,  3.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:45,  3.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:45,  3.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:44,  3.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:44,  3.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:44,  3.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:43,  3.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:43,  3.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  17%|█▋        | 28/162 [00:00<00:30,  4.37 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:30,  4.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:30,  4.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:30,  4.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:29,  4.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:01<00:29,  4.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:01<00:29,  4.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:29,  4.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:29,  4.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:28,  4.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:28,  4.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  23%|██▎       | 38/162 [00:01<00:20,  6.12 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:20,  6.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:20,  6.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:19,  6.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:19,  6.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:19,  6.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:19,  6.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:19,  6.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:19,  6.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:18,  6.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  29%|██▉       | 47/162 [00:01<00:13,  8.49 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:13,  8.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:13,  8.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:13,  8.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:13,  8.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:13,  8.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:12,  8.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:12,  8.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:12,  8.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:12,  8.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:12,  8.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  35%|███▌      | 57/162 [00:01<00:08, 11.68 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:08, 11.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:08, 11.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:08, 11.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:08, 11.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:08, 11.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:08, 11.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:08, 11.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:08, 11.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:08, 11.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:08, 11.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  41%|████▏     | 67/162 [00:01<00:05, 15.85 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:05, 15.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:05, 15.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:05, 15.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:05, 15.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:05, 15.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:05, 15.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:05, 15.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:05, 15.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:05, 15.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:05, 15.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  48%|████▊     | 77/162 [00:01<00:04, 21.08 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:04, 21.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:03, 21.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:03, 21.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:03, 21.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:03, 21.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:03, 21.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:03, 21.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:03, 21.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:03, 21.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 27.36 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 27.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 27.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 27.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:02, 27.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:02, 27.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:02, 27.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:02, 27.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:02, 27.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:02, 27.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:02, 27.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 34.65 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 34.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 34.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 34.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 34.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 34.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 34.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 34.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 34.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 34.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 34.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 42.86 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 42.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 42.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 42.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 42.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:01, 42.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:01, 42.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:01, 42.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:01, 42.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:01, 42.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:01, 42.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 51.32 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 51.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 51.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 51.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 51.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 51.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 51.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 51.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 51.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 51.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 51.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 58.72 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 58.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 58.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 58.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 58.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 58.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 58.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 58.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 58.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 58.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 58.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 65.31 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 65.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 65.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 65.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 65.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 65.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 65.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 65.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 65.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 65.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 65.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 71.11 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 71.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 71.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 71.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 71.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 71.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 71.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 71.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 71.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 71.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 71.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 74.97 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 74.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 74.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 74.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 74.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 74.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 74.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 74.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.43s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.43s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 74.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.43s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 74.97 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.50s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.43s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 74.97 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.50s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.50s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 35.99 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.50s/ url]
0 examples [00:00, ? examples/s]2020-07-19 06:09:00.956886: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-19 06:09:00.971768: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095190000 Hz
2020-07-19 06:09:00.972003: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55770d963c60 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-19 06:09:00.972038: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  2.21 examples/s]107 examples [00:00,  3.15 examples/s]208 examples [00:00,  4.50 examples/s]314 examples [00:00,  6.42 examples/s]423 examples [00:00,  9.14 examples/s]525 examples [00:00, 13.01 examples/s]635 examples [00:01, 18.50 examples/s]744 examples [00:01, 26.23 examples/s]849 examples [00:01, 37.08 examples/s]954 examples [00:01, 52.18 examples/s]1059 examples [00:01, 72.98 examples/s]1165 examples [00:01, 101.26 examples/s]1276 examples [00:01, 139.19 examples/s]1388 examples [00:01, 188.73 examples/s]1499 examples [00:01, 251.24 examples/s]1613 examples [00:01, 327.80 examples/s]1725 examples [00:02, 415.98 examples/s]1838 examples [00:02, 513.04 examples/s]1955 examples [00:02, 616.24 examples/s]2068 examples [00:02, 707.16 examples/s]2180 examples [00:02, 790.15 examples/s]2291 examples [00:02, 864.57 examples/s]2402 examples [00:02, 922.74 examples/s]2516 examples [00:02, 978.58 examples/s]2629 examples [00:02, 1018.95 examples/s]2743 examples [00:02, 1050.60 examples/s]2856 examples [00:03, 1036.27 examples/s]2965 examples [00:03, 976.29 examples/s] 3067 examples [00:03, 980.35 examples/s]3169 examples [00:03, 934.16 examples/s]3265 examples [00:03, 930.70 examples/s]3360 examples [00:03, 889.12 examples/s]3451 examples [00:03, 891.92 examples/s]3542 examples [00:03, 860.10 examples/s]3630 examples [00:03, 840.66 examples/s]3719 examples [00:04, 853.20 examples/s]3808 examples [00:04, 863.15 examples/s]3916 examples [00:04, 916.74 examples/s]4027 examples [00:04, 965.59 examples/s]4141 examples [00:04, 1011.40 examples/s]4251 examples [00:04, 1036.33 examples/s]4359 examples [00:04, 1047.81 examples/s]4473 examples [00:04, 1072.34 examples/s]4584 examples [00:04, 1083.03 examples/s]4700 examples [00:04, 1103.33 examples/s]4814 examples [00:05, 1112.88 examples/s]4931 examples [00:05, 1127.09 examples/s]5046 examples [00:05, 1131.07 examples/s]5160 examples [00:05, 1080.36 examples/s]5269 examples [00:05, 1082.11 examples/s]5381 examples [00:05, 1092.32 examples/s]5493 examples [00:05, 1084.70 examples/s]5604 examples [00:05, 1067.41 examples/s]5713 examples [00:05, 1073.89 examples/s]5824 examples [00:06, 1083.27 examples/s]5934 examples [00:06, 1087.80 examples/s]6043 examples [00:06, 1055.35 examples/s]6149 examples [00:06, 1056.14 examples/s]6255 examples [00:06, 1054.23 examples/s]6362 examples [00:06, 1057.72 examples/s]6468 examples [00:06, 1055.83 examples/s]6576 examples [00:06, 1062.87 examples/s]6687 examples [00:06, 1074.16 examples/s]6795 examples [00:06, 1074.48 examples/s]6905 examples [00:07, 1081.21 examples/s]7014 examples [00:07, 1078.05 examples/s]7127 examples [00:07, 1092.72 examples/s]7240 examples [00:07, 1103.32 examples/s]7351 examples [00:07, 1100.64 examples/s]7462 examples [00:07, 1073.96 examples/s]7570 examples [00:07, 1058.64 examples/s]7681 examples [00:07, 1071.86 examples/s]7793 examples [00:07, 1084.47 examples/s]7902 examples [00:07, 1072.83 examples/s]8010 examples [00:08, 1066.51 examples/s]8121 examples [00:08, 1077.84 examples/s]8231 examples [00:08, 1082.22 examples/s]8340 examples [00:08, 1082.97 examples/s]8450 examples [00:08, 1085.42 examples/s]8562 examples [00:08, 1093.20 examples/s]8673 examples [00:08, 1097.70 examples/s]8783 examples [00:08, 1098.14 examples/s]8894 examples [00:08, 1099.32 examples/s]9004 examples [00:08, 1063.77 examples/s]9112 examples [00:09, 1067.89 examples/s]9219 examples [00:09, 1059.15 examples/s]9328 examples [00:09, 1066.27 examples/s]9435 examples [00:09, 1056.13 examples/s]9541 examples [00:09, 1050.13 examples/s]9647 examples [00:09, 1039.94 examples/s]9752 examples [00:09, 988.95 examples/s] 9861 examples [00:09, 1016.62 examples/s]9969 examples [00:09, 1030.10 examples/s]10073 examples [00:10, 982.20 examples/s]10180 examples [00:10, 1004.79 examples/s]10285 examples [00:10, 1016.87 examples/s]10392 examples [00:10, 1032.09 examples/s]10496 examples [00:10, 1031.39 examples/s]10602 examples [00:10, 1037.35 examples/s]10709 examples [00:10, 1045.34 examples/s]10814 examples [00:10, 1042.30 examples/s]10923 examples [00:10, 1056.13 examples/s]11034 examples [00:10, 1071.70 examples/s]11142 examples [00:11, 1070.25 examples/s]11254 examples [00:11, 1082.06 examples/s]11363 examples [00:11, 1073.83 examples/s]11471 examples [00:11, 1075.53 examples/s]11581 examples [00:11, 1082.31 examples/s]11690 examples [00:11, 1080.40 examples/s]11803 examples [00:11, 1093.05 examples/s]11913 examples [00:11, 1092.97 examples/s]12023 examples [00:11, 1091.59 examples/s]12133 examples [00:11, 1081.02 examples/s]12242 examples [00:12, 1081.47 examples/s]12354 examples [00:12, 1092.10 examples/s]12464 examples [00:12, 1092.30 examples/s]12576 examples [00:12, 1098.40 examples/s]12686 examples [00:12, 1098.78 examples/s]12796 examples [00:12, 1084.43 examples/s]12905 examples [00:12, 1076.98 examples/s]13013 examples [00:12, 1062.29 examples/s]13120 examples [00:12, 1056.25 examples/s]13231 examples [00:12, 1069.95 examples/s]13340 examples [00:13, 1075.87 examples/s]13453 examples [00:13, 1090.91 examples/s]13564 examples [00:13, 1096.28 examples/s]13677 examples [00:13, 1103.74 examples/s]13791 examples [00:13, 1112.64 examples/s]13903 examples [00:13, 1095.10 examples/s]14013 examples [00:13, 1061.59 examples/s]14120 examples [00:13, 1049.80 examples/s]14228 examples [00:13, 1056.27 examples/s]14334 examples [00:13, 1014.34 examples/s]14444 examples [00:14, 1037.10 examples/s]14557 examples [00:14, 1061.83 examples/s]14670 examples [00:14, 1081.12 examples/s]14784 examples [00:14, 1095.97 examples/s]14894 examples [00:14, 1091.18 examples/s]15004 examples [00:14, 1092.09 examples/s]15116 examples [00:14, 1099.49 examples/s]15232 examples [00:14, 1114.45 examples/s]15345 examples [00:14, 1117.56 examples/s]15459 examples [00:15, 1122.57 examples/s]15572 examples [00:15, 1113.23 examples/s]15684 examples [00:15, 1114.80 examples/s]15796 examples [00:15, 1111.22 examples/s]15909 examples [00:15, 1113.72 examples/s]16021 examples [00:15, 1115.51 examples/s]16133 examples [00:15, 1106.94 examples/s]16246 examples [00:15, 1112.61 examples/s]16359 examples [00:15, 1115.62 examples/s]16471 examples [00:15, 1113.24 examples/s]16583 examples [00:16, 1089.84 examples/s]16693 examples [00:16, 1079.35 examples/s]16804 examples [00:16, 1087.66 examples/s]16913 examples [00:16, 1084.80 examples/s]17022 examples [00:16, 1084.94 examples/s]17131 examples [00:16, 1078.17 examples/s]17239 examples [00:16, 1062.63 examples/s]17347 examples [00:16, 1065.49 examples/s]17458 examples [00:16, 1075.83 examples/s]17566 examples [00:16, 1075.36 examples/s]17674 examples [00:17, 1075.40 examples/s]17782 examples [00:17, 1059.16 examples/s]17888 examples [00:17, 1052.14 examples/s]17997 examples [00:17, 1060.34 examples/s]18106 examples [00:17, 1067.85 examples/s]18213 examples [00:17, 1056.13 examples/s]18319 examples [00:17, 1017.09 examples/s]18431 examples [00:17, 1045.37 examples/s]18545 examples [00:17, 1070.36 examples/s]18655 examples [00:17, 1076.86 examples/s]18766 examples [00:18, 1085.66 examples/s]18875 examples [00:18, 1074.47 examples/s]18983 examples [00:18, 1034.54 examples/s]19094 examples [00:18, 1054.67 examples/s]19206 examples [00:18, 1072.15 examples/s]19316 examples [00:18, 1079.96 examples/s]19425 examples [00:18, 1072.48 examples/s]19533 examples [00:18, 1063.56 examples/s]19640 examples [00:18, 1047.80 examples/s]19746 examples [00:18, 1051.42 examples/s]19858 examples [00:19, 1070.34 examples/s]19968 examples [00:19, 1077.66 examples/s]20076 examples [00:19, 1028.97 examples/s]20186 examples [00:19, 1047.21 examples/s]20296 examples [00:19, 1061.83 examples/s]20406 examples [00:19, 1071.85 examples/s]20515 examples [00:19, 1076.23 examples/s]20627 examples [00:19, 1087.97 examples/s]20738 examples [00:19, 1094.29 examples/s]20849 examples [00:20, 1098.72 examples/s]20959 examples [00:20, 1096.56 examples/s]21069 examples [00:20, 1097.29 examples/s]21179 examples [00:20, 1097.24 examples/s]21289 examples [00:20, 1073.05 examples/s]21400 examples [00:20, 1083.34 examples/s]21510 examples [00:20, 1087.86 examples/s]21620 examples [00:20, 1089.33 examples/s]21731 examples [00:20, 1093.78 examples/s]21842 examples [00:20, 1097.33 examples/s]21954 examples [00:21, 1102.28 examples/s]22065 examples [00:21, 1101.03 examples/s]22176 examples [00:21, 1099.47 examples/s]22286 examples [00:21, 1098.56 examples/s]22398 examples [00:21, 1102.43 examples/s]22509 examples [00:21, 1104.03 examples/s]22620 examples [00:21, 1101.33 examples/s]22731 examples [00:21, 1101.38 examples/s]22843 examples [00:21, 1106.46 examples/s]22954 examples [00:21, 1106.10 examples/s]23065 examples [00:22, 1090.12 examples/s]23175 examples [00:22, 1089.98 examples/s]23287 examples [00:22, 1096.14 examples/s]23397 examples [00:22, 1091.79 examples/s]23507 examples [00:22, 1075.96 examples/s]23615 examples [00:22, 1024.79 examples/s]23722 examples [00:22, 1035.43 examples/s]23834 examples [00:22, 1058.17 examples/s]23947 examples [00:22, 1078.19 examples/s]24060 examples [00:22, 1092.65 examples/s]24171 examples [00:23, 1096.42 examples/s]24284 examples [00:23, 1105.68 examples/s]24397 examples [00:23, 1110.16 examples/s]24510 examples [00:23, 1113.37 examples/s]24622 examples [00:23, 1113.54 examples/s]24734 examples [00:23, 1111.07 examples/s]24846 examples [00:23, 1109.28 examples/s]24959 examples [00:23, 1113.53 examples/s]25073 examples [00:23, 1119.28 examples/s]25186 examples [00:23, 1121.29 examples/s]25299 examples [00:24, 1120.83 examples/s]25412 examples [00:24, 1122.04 examples/s]25525 examples [00:24, 1122.22 examples/s]25639 examples [00:24, 1126.83 examples/s]25753 examples [00:24, 1129.78 examples/s]25866 examples [00:24, 1059.18 examples/s]25973 examples [00:24, 1041.98 examples/s]26080 examples [00:24, 1049.78 examples/s]26187 examples [00:24, 1055.08 examples/s]26293 examples [00:25, 1042.52 examples/s]26399 examples [00:25, 1047.32 examples/s]26504 examples [00:25, 1045.54 examples/s]26609 examples [00:25, 1026.70 examples/s]26712 examples [00:25, 1027.68 examples/s]26815 examples [00:25, 1017.88 examples/s]26917 examples [00:25, 1007.80 examples/s]27018 examples [00:25, 980.47 examples/s] 27126 examples [00:25, 1007.92 examples/s]27233 examples [00:25, 1024.90 examples/s]27338 examples [00:26, 1030.11 examples/s]27444 examples [00:26, 1036.93 examples/s]27553 examples [00:26, 1050.29 examples/s]27661 examples [00:26, 1057.62 examples/s]27772 examples [00:26, 1070.99 examples/s]27880 examples [00:26, 1061.11 examples/s]27987 examples [00:26, 1046.19 examples/s]28092 examples [00:26, 1040.13 examples/s]28197 examples [00:26, 1042.41 examples/s]28307 examples [00:26, 1058.04 examples/s]28417 examples [00:27, 1068.50 examples/s]28524 examples [00:27, 1054.29 examples/s]28630 examples [00:27, 1055.83 examples/s]28736 examples [00:27, 1046.71 examples/s]28843 examples [00:27, 1051.50 examples/s]28949 examples [00:27, 1032.59 examples/s]29054 examples [00:27, 1035.64 examples/s]29158 examples [00:27, 1036.52 examples/s]29272 examples [00:27, 1065.04 examples/s]29383 examples [00:27, 1077.30 examples/s]29491 examples [00:28, 1073.70 examples/s]29601 examples [00:28, 1081.25 examples/s]29712 examples [00:28, 1088.29 examples/s]29822 examples [00:28, 1088.94 examples/s]29933 examples [00:28, 1095.11 examples/s]30043 examples [00:28, 1013.71 examples/s]30148 examples [00:28, 1023.68 examples/s]30252 examples [00:28, 1007.25 examples/s]30365 examples [00:28, 1039.58 examples/s]30470 examples [00:29, 1009.85 examples/s]30579 examples [00:29, 1030.06 examples/s]30689 examples [00:29, 1048.18 examples/s]30800 examples [00:29, 1063.15 examples/s]30913 examples [00:29, 1079.74 examples/s]31022 examples [00:29, 1065.53 examples/s]31131 examples [00:29, 1069.94 examples/s]31241 examples [00:29, 1076.63 examples/s]31351 examples [00:29, 1083.19 examples/s]31463 examples [00:29, 1092.71 examples/s]31573 examples [00:30, 1092.97 examples/s]31683 examples [00:30, 1070.12 examples/s]31791 examples [00:30, 1047.56 examples/s]31896 examples [00:30, 1040.58 examples/s]32001 examples [00:30, 1041.03 examples/s]32106 examples [00:30, 1015.45 examples/s]32208 examples [00:30, 1007.53 examples/s]32309 examples [00:30, 1008.25 examples/s]32412 examples [00:30, 1012.52 examples/s]32516 examples [00:30, 1018.67 examples/s]32621 examples [00:31, 1024.92 examples/s]32724 examples [00:31, 1009.19 examples/s]32826 examples [00:31, 1006.75 examples/s]32936 examples [00:31, 1030.71 examples/s]33047 examples [00:31, 1050.97 examples/s]33156 examples [00:31, 1061.35 examples/s]33265 examples [00:31, 1069.75 examples/s]33373 examples [00:31, 1064.92 examples/s]33482 examples [00:31, 1069.78 examples/s]33593 examples [00:31, 1078.64 examples/s]33704 examples [00:32, 1087.86 examples/s]33813 examples [00:32, 1087.06 examples/s]33924 examples [00:32, 1092.63 examples/s]34034 examples [00:32, 1091.58 examples/s]34144 examples [00:32, 1086.28 examples/s]34253 examples [00:32, 1048.68 examples/s]34360 examples [00:32, 1052.21 examples/s]34466 examples [00:32, 1051.07 examples/s]34574 examples [00:32, 1057.10 examples/s]34681 examples [00:32, 1058.95 examples/s]34790 examples [00:33, 1065.77 examples/s]34897 examples [00:33, 1065.52 examples/s]35007 examples [00:33, 1074.07 examples/s]35115 examples [00:33, 1005.79 examples/s]35217 examples [00:33, 1006.63 examples/s]35320 examples [00:33, 1013.49 examples/s]35427 examples [00:33, 1027.82 examples/s]35537 examples [00:33, 1046.98 examples/s]35648 examples [00:33, 1062.37 examples/s]35759 examples [00:34, 1075.47 examples/s]35867 examples [00:34, 1065.90 examples/s]35977 examples [00:34, 1075.33 examples/s]36088 examples [00:34, 1082.73 examples/s]36197 examples [00:34, 1036.13 examples/s]36302 examples [00:34, 1007.67 examples/s]36405 examples [00:34, 1011.84 examples/s]36511 examples [00:34, 1023.88 examples/s]36621 examples [00:34, 1043.84 examples/s]36731 examples [00:34, 1057.31 examples/s]36840 examples [00:35, 1065.50 examples/s]36950 examples [00:35, 1074.22 examples/s]37061 examples [00:35, 1083.22 examples/s]37170 examples [00:35, 1084.94 examples/s]37281 examples [00:35, 1090.20 examples/s]37391 examples [00:35, 1058.01 examples/s]37500 examples [00:35, 1065.50 examples/s]37607 examples [00:35, 1051.77 examples/s]37713 examples [00:35, 1045.20 examples/s]37818 examples [00:35, 1041.74 examples/s]37925 examples [00:36, 1048.15 examples/s]38032 examples [00:36, 1052.71 examples/s]38140 examples [00:36, 1059.13 examples/s]38247 examples [00:36, 1061.86 examples/s]38356 examples [00:36, 1067.32 examples/s]38463 examples [00:36, 1061.10 examples/s]38573 examples [00:36, 1071.90 examples/s]38683 examples [00:36, 1080.03 examples/s]38793 examples [00:36, 1084.94 examples/s]38904 examples [00:36, 1090.48 examples/s]39014 examples [00:37, 1081.98 examples/s]39127 examples [00:37, 1094.39 examples/s]39241 examples [00:37, 1106.97 examples/s]39352 examples [00:37, 1093.75 examples/s]39463 examples [00:37, 1096.47 examples/s]39573 examples [00:37, 1082.06 examples/s]39682 examples [00:37, 1056.42 examples/s]39792 examples [00:37, 1067.65 examples/s]39903 examples [00:37, 1077.88 examples/s]40011 examples [00:38, 1038.59 examples/s]40122 examples [00:38, 1058.49 examples/s]40233 examples [00:38, 1071.44 examples/s]40346 examples [00:38, 1085.50 examples/s]40455 examples [00:38, 1085.89 examples/s]40564 examples [00:38, 1066.15 examples/s]40671 examples [00:38, 1060.78 examples/s]40778 examples [00:38, 1040.14 examples/s]40892 examples [00:38, 1068.19 examples/s]41000 examples [00:38, 1071.70 examples/s]41108 examples [00:39, 1059.45 examples/s]41218 examples [00:39, 1069.74 examples/s]41326 examples [00:39, 1071.83 examples/s]41434 examples [00:39, 1067.91 examples/s]41541 examples [00:39, 1064.10 examples/s]41649 examples [00:39, 1066.06 examples/s]41756 examples [00:39, 1042.96 examples/s]41862 examples [00:39, 1047.26 examples/s]41967 examples [00:39, 1027.48 examples/s]42070 examples [00:39, 992.61 examples/s] 42180 examples [00:40, 1020.11 examples/s]42283 examples [00:40, 1020.61 examples/s]42387 examples [00:40, 1026.19 examples/s]42490 examples [00:40, 1012.53 examples/s]42596 examples [00:40, 1024.64 examples/s]42702 examples [00:40, 1032.93 examples/s]42812 examples [00:40, 1050.98 examples/s]42918 examples [00:40, 1051.36 examples/s]43030 examples [00:40, 1070.23 examples/s]43138 examples [00:40, 1072.70 examples/s]43248 examples [00:41, 1078.29 examples/s]43357 examples [00:41, 1079.15 examples/s]43466 examples [00:41, 1080.77 examples/s]43577 examples [00:41, 1088.95 examples/s]43686 examples [00:41, 1080.96 examples/s]43795 examples [00:41, 1080.31 examples/s]43904 examples [00:41, 1076.30 examples/s]44012 examples [00:41, 1073.63 examples/s]44120 examples [00:41, 1062.16 examples/s]44227 examples [00:42, 1063.34 examples/s]44338 examples [00:42, 1074.49 examples/s]44452 examples [00:42, 1092.76 examples/s]44564 examples [00:42, 1100.05 examples/s]44675 examples [00:42, 1099.08 examples/s]44785 examples [00:42, 1088.27 examples/s]44894 examples [00:42, 1082.34 examples/s]45003 examples [00:42, 1078.72 examples/s]45111 examples [00:42, 1057.73 examples/s]45219 examples [00:42, 1063.32 examples/s]45329 examples [00:43, 1071.46 examples/s]45438 examples [00:43, 1075.48 examples/s]45546 examples [00:43, 1070.53 examples/s]45654 examples [00:43, 1072.99 examples/s]45762 examples [00:43, 1074.88 examples/s]45870 examples [00:43, 1067.86 examples/s]45977 examples [00:43, 1053.06 examples/s]46083 examples [00:43, 1019.62 examples/s]46192 examples [00:43, 1038.74 examples/s]46302 examples [00:43, 1055.94 examples/s]46412 examples [00:44, 1066.09 examples/s]46521 examples [00:44, 1072.36 examples/s]46629 examples [00:44, 1041.92 examples/s]46741 examples [00:44, 1064.00 examples/s]46848 examples [00:44, 1059.98 examples/s]46955 examples [00:44, 1052.88 examples/s]47067 examples [00:44, 1070.87 examples/s]47180 examples [00:44, 1085.45 examples/s]47293 examples [00:44, 1096.46 examples/s]47403 examples [00:44, 1094.11 examples/s]47513 examples [00:45, 1093.42 examples/s]47623 examples [00:45, 1092.78 examples/s]47733 examples [00:45, 1089.21 examples/s]47842 examples [00:45, 1080.87 examples/s]47953 examples [00:45, 1087.26 examples/s]48063 examples [00:45, 1088.62 examples/s]48172 examples [00:45, 1081.49 examples/s]48281 examples [00:45, 1081.74 examples/s]48393 examples [00:45, 1091.10 examples/s]48503 examples [00:45, 1091.41 examples/s]48613 examples [00:46, 1086.60 examples/s]48722 examples [00:46, 1081.41 examples/s]48831 examples [00:46, 1075.49 examples/s]48939 examples [00:46, 1045.00 examples/s]49051 examples [00:46, 1063.70 examples/s]49158 examples [00:46, 1064.40 examples/s]49269 examples [00:46, 1076.41 examples/s]49381 examples [00:46, 1088.07 examples/s]49495 examples [00:46, 1101.09 examples/s]49607 examples [00:46, 1105.57 examples/s]49718 examples [00:47, 1080.69 examples/s]49832 examples [00:47, 1096.86 examples/s]49947 examples [00:47, 1110.90 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 16%|█▌        | 8059/50000 [00:00<00:00, 80585.47 examples/s] 43%|████▎     | 21564/50000 [00:00<00:00, 91676.93 examples/s] 71%|███████   | 35280/50000 [00:00<00:00, 101802.77 examples/s] 98%|█████████▊| 49226/50000 [00:00<00:00, 110775.46 examples/s]                                                                0 examples [00:00, ? examples/s]89 examples [00:00, 885.96 examples/s]201 examples [00:00, 944.74 examples/s]312 examples [00:00, 988.00 examples/s]412 examples [00:00, 989.02 examples/s]522 examples [00:00, 1018.05 examples/s]630 examples [00:00, 1035.12 examples/s]724 examples [00:00, 1001.68 examples/s]836 examples [00:00, 1032.55 examples/s]953 examples [00:00, 1068.76 examples/s]1069 examples [00:01, 1094.47 examples/s]1187 examples [00:01, 1116.90 examples/s]1304 examples [00:01, 1130.23 examples/s]1417 examples [00:01, 1094.24 examples/s]1527 examples [00:01, 1079.50 examples/s]1639 examples [00:01, 1088.68 examples/s]1752 examples [00:01, 1097.95 examples/s]1862 examples [00:01, 1068.98 examples/s]1970 examples [00:01, 1071.61 examples/s]2081 examples [00:01, 1082.41 examples/s]2192 examples [00:02, 1089.46 examples/s]2305 examples [00:02, 1100.38 examples/s]2416 examples [00:02, 1088.76 examples/s]2525 examples [00:02, 1084.65 examples/s]2634 examples [00:02, 1072.40 examples/s]2742 examples [00:02, 1064.54 examples/s]2858 examples [00:02, 1089.13 examples/s]2972 examples [00:02, 1101.36 examples/s]3083 examples [00:02, 1095.18 examples/s]3196 examples [00:02, 1103.91 examples/s]3310 examples [00:03, 1113.14 examples/s]3426 examples [00:03, 1124.32 examples/s]3540 examples [00:03, 1128.13 examples/s]3653 examples [00:03, 1122.21 examples/s]3766 examples [00:03, 1124.14 examples/s]3883 examples [00:03, 1136.67 examples/s]4000 examples [00:03, 1146.30 examples/s]4115 examples [00:03, 1143.62 examples/s]4230 examples [00:03, 1112.72 examples/s]4342 examples [00:03, 1096.45 examples/s]4452 examples [00:04, 1095.11 examples/s]4562 examples [00:04, 1084.05 examples/s]4673 examples [00:04, 1089.49 examples/s]4783 examples [00:04, 1083.66 examples/s]4892 examples [00:04, 1072.34 examples/s]5006 examples [00:04, 1089.84 examples/s]5120 examples [00:04, 1103.29 examples/s]5233 examples [00:04, 1110.08 examples/s]5345 examples [00:04, 1107.08 examples/s]5456 examples [00:04, 1071.16 examples/s]5568 examples [00:05, 1084.42 examples/s]5680 examples [00:05, 1094.65 examples/s]5790 examples [00:05, 1088.89 examples/s]5900 examples [00:05, 1086.88 examples/s]6009 examples [00:05, 1075.97 examples/s]6117 examples [00:05, 1073.22 examples/s]6225 examples [00:05, 1071.00 examples/s]6334 examples [00:05, 1076.31 examples/s]6442 examples [00:05, 895.62 examples/s] 6537 examples [00:06, 815.11 examples/s]6639 examples [00:06, 865.95 examples/s]6752 examples [00:06, 929.15 examples/s]6861 examples [00:06, 970.83 examples/s]6973 examples [00:06, 1009.39 examples/s]7085 examples [00:06, 1039.34 examples/s]7196 examples [00:06, 1059.20 examples/s]7304 examples [00:06, 1032.28 examples/s]7414 examples [00:06, 1049.70 examples/s]7527 examples [00:07, 1070.39 examples/s]7639 examples [00:07, 1084.22 examples/s]7749 examples [00:07, 1078.88 examples/s]7858 examples [00:07, 1051.43 examples/s]7964 examples [00:07, 1021.38 examples/s]8071 examples [00:07, 1034.19 examples/s]8177 examples [00:07, 1041.77 examples/s]8284 examples [00:07, 1047.79 examples/s]8391 examples [00:07, 1052.68 examples/s]8498 examples [00:07, 1055.21 examples/s]8604 examples [00:08, 1054.61 examples/s]8711 examples [00:08, 1057.27 examples/s]8819 examples [00:08, 1063.56 examples/s]8927 examples [00:08, 1066.65 examples/s]9034 examples [00:08, 1061.58 examples/s]9142 examples [00:08, 1066.46 examples/s]9249 examples [00:08, 1064.37 examples/s]9356 examples [00:08, 1048.76 examples/s]9461 examples [00:08, 1048.62 examples/s]9566 examples [00:08, 1007.72 examples/s]9674 examples [00:09, 1025.95 examples/s]9780 examples [00:09, 1033.24 examples/s]9884 examples [00:09, 1030.75 examples/s]9990 examples [00:09, 1039.22 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteUETSY9/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteUETSY9/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f410ad65950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f410ad65950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f410ad65950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f4108e40c18>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f417628cf98>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f410ad65950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f410ad65950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f410ad65950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f41092af5f8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f41092d64a8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f4093092ea0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f4093092ea0> 

  function with postional parmater data_info <function split_train_valid at 0x7f4093092ea0> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f409308b048> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f409308b048> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f409308b048> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.1
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.1) (2.3.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.0.3)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.7.1)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (45.2.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.2)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.4.1)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (4.48.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.19.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.1.3)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.24.0)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (7.4.1)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.7.0)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.10)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2020.6.20)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.1-py3-none-any.whl size=12047105 sha256=2bd936442dfa234c15231758ed1003d008b32c162e9a7645d45982c3a6dc5e7b
  Stored in directory: /tmp/pip-ephem-wheel-cache-9ps5tqrn/wheels/10/6f/a6/ddd8204ceecdedddea923f8514e13afb0c1f0f556d2c9c3da0
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<20:39:42, 11.6kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<14:41:49, 16.3kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<10:20:30, 23.2kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<7:14:51, 33.0kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.64M/862M [00:01<5:03:39, 47.1kB/s].vector_cache/glove.6B.zip:   1%|          | 9.38M/862M [00:01<3:31:14, 67.3kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.1M/862M [00:01<2:26:58, 96.1kB/s].vector_cache/glove.6B.zip:   2%|▏         | 20.8M/862M [00:01<1:42:17, 137kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 26.6M/862M [00:01<1:11:12, 196kB/s].vector_cache/glove.6B.zip:   4%|▍         | 32.5M/862M [00:01<49:35, 279kB/s]  .vector_cache/glove.6B.zip:   4%|▍         | 38.2M/862M [00:02<34:32, 398kB/s].vector_cache/glove.6B.zip:   5%|▍         | 41.2M/862M [00:02<24:14, 564kB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.8M/862M [00:02<16:56, 802kB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.5M/862M [00:02<12:17, 1.10MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.7M/862M [00:04<10:28, 1.28MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.8M/862M [00:05<11:41, 1.15MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.3M/862M [00:05<09:07, 1.47MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.2M/862M [00:05<06:35, 2.03MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.9M/862M [00:06<08:30, 1.57MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.2M/862M [00:07<07:31, 1.77MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.5M/862M [00:07<05:36, 2.38MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.0M/862M [00:08<06:39, 1.99MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.2M/862M [00:09<07:29, 1.77MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.0M/862M [00:09<05:51, 2.26MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.5M/862M [00:09<04:14, 3.12MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.1M/862M [00:10<13:47, 959kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 69.5M/862M [00:11<11:01, 1.20MB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.1M/862M [00:11<08:02, 1.64MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.2M/862M [00:12<08:42, 1.51MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.6M/862M [00:12<07:25, 1.77MB/s].vector_cache/glove.6B.zip:   9%|▊         | 75.2M/862M [00:13<05:31, 2.38MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.4M/862M [00:14<06:55, 1.89MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.6M/862M [00:14<07:30, 1.74MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.3M/862M [00:15<05:49, 2.25MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.3M/862M [00:15<04:15, 3.06MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.5M/862M [00:16<08:18, 1.57MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.9M/862M [00:16<07:07, 1.82MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.4M/862M [00:17<05:18, 2.44MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.6M/862M [00:18<06:45, 1.92MB/s].vector_cache/glove.6B.zip:  10%|▉         | 86.0M/862M [00:18<06:02, 2.14MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.5M/862M [00:19<04:32, 2.84MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.7M/862M [00:20<06:11, 2.08MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.9M/862M [00:20<07:04, 1.82MB/s].vector_cache/glove.6B.zip:  11%|█         | 90.7M/862M [00:20<05:36, 2.29MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.7M/862M [00:21<04:04, 3.14MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.8M/862M [00:22<1:34:15, 136kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.2M/862M [00:22<1:07:17, 190kB/s].vector_cache/glove.6B.zip:  11%|█         | 95.7M/862M [00:22<47:17, 270kB/s]  .vector_cache/glove.6B.zip:  11%|█▏        | 97.9M/862M [00:24<35:59, 354kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.1M/862M [00:24<27:47, 458kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.9M/862M [00:24<20:00, 636kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:24<14:05, 899kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<24:48, 511kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<18:40, 678kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:26<13:22, 945kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<12:17, 1.02MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<09:53, 1.27MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:28<07:14, 1.74MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<08:01, 1.56MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<06:40, 1.87MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<04:55, 2.54MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:30<03:36, 3.45MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<49:34, 251kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<37:12, 335kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<26:35, 468kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:32<18:43, 663kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<19:30, 635kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<14:54, 830kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<10:44, 1.15MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<10:23, 1.19MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<08:32, 1.44MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:36<06:14, 1.97MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<07:14, 1.69MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<07:27, 1.64MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<05:50, 2.09MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<06:03, 2.01MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<05:30, 2.21MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<04:07, 2.95MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<05:43, 2.12MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<06:27, 1.87MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<05:08, 2.35MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:42<03:42, 3.24MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<11:39:27, 17.2kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<8:10:34, 24.6kB/s] .vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<5:42:58, 35.0kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<4:02:10, 49.5kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<2:51:55, 69.7kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<2:00:50, 99.0kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:46<1:24:25, 141kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<1:08:59, 173kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<49:39, 240kB/s]  .vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<34:59, 340kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<26:55, 440kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<21:36, 548kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<15:41, 754kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:50<11:08, 1.06MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<11:51, 993kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<09:28, 1.24MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<06:58, 1.69MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:52<05:01, 2.33MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<22:27, 521kB/s] .vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<18:26, 634kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<13:29, 867kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:54<09:33, 1.22MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<12:06, 961kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<09:52, 1.18MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:56<07:13, 1.60MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<07:28, 1.55MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:58<07:55, 1.46MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<06:06, 1.89MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:58<04:25, 2.60MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<07:53, 1.46MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<06:51, 1.68MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<05:06, 2.25MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<05:57, 1.92MB/s].vector_cache/glove.6B.zip:  20%|██        | 177M/862M [01:02<06:49, 1.67MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<05:26, 2.10MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:02<03:57, 2.87MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<12:12, 930kB/s] .vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<09:52, 1.15MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<07:10, 1.58MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<07:24, 1.53MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<07:48, 1.44MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<06:07, 1.84MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:06<04:25, 2.53MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<12:24, 904kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<09:59, 1.12MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:08<07:18, 1.53MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<07:26, 1.50MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<07:47, 1.43MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:10<06:07, 1.82MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:10<04:25, 2.50MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<12:16, 903kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<09:52, 1.12MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:12<07:10, 1.54MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<07:20, 1.50MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<07:41, 1.43MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<05:55, 1.85MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:14<04:17, 2.55MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<07:16, 1.50MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<06:21, 1.72MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<04:42, 2.31MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<05:35, 1.94MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<06:25, 1.69MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:18<05:04, 2.14MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:18<03:40, 2.94MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<07:52, 1.37MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<06:45, 1.60MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:20<04:59, 2.16MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<05:45, 1.86MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:22<05:18, 2.02MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 220M/862M [01:22<04:01, 2.66MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<05:02, 2.11MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:24<06:00, 1.78MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<04:44, 2.25MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:24<03:25, 3.10MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 226M/862M [01:25<10:09, 1.04MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<08:20, 1.27MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:26<06:07, 1.72MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<06:30, 1.62MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<06:59, 1.50MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<05:30, 1.91MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:28<03:59, 2.62MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<11:28, 911kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<09:14, 1.13MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<06:42, 1.55MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<06:53, 1.51MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<07:13, 1.44MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<05:39, 1.83MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:32<04:04, 2.53MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<11:32, 894kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<09:16, 1.11MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<06:44, 1.53MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<06:52, 1.49MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<07:11, 1.42MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:36<05:31, 1.85MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:36<03:59, 2.55MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<07:15, 1.40MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<06:15, 1.62MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<04:37, 2.20MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<05:22, 1.88MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<06:06, 1.66MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<04:48, 2.10MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:40<03:28, 2.89MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<07:21, 1.36MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<06:20, 1.58MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<04:41, 2.14MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<05:22, 1.85MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<06:04, 1.64MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<04:49, 2.06MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:44<03:29, 2.84MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<10:38, 931kB/s] .vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<08:33, 1.16MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<06:15, 1.58MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<06:29, 1.51MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<06:49, 1.44MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<05:20, 1.84MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:48<03:51, 2.53MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<10:58, 889kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<08:47, 1.11MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<06:25, 1.51MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<06:31, 1.48MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<06:49, 1.42MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:52<05:20, 1.81MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:52<03:51, 2.49MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<10:41, 901kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<08:36, 1.12MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:53<06:17, 1.52MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<06:23, 1.49MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<05:34, 1.71MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:55<04:08, 2.30MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<04:53, 1.94MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<05:37, 1.69MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:57<04:23, 2.16MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<03:12, 2.94MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<05:45, 1.63MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:59<05:07, 1.83MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [01:59<03:48, 2.46MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<04:38, 2.01MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:01<04:19, 2.16MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:01<03:15, 2.86MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:03<04:14, 2.19MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:03<05:07, 1.81MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:03<04:01, 2.30MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:03<02:56, 3.14MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<06:00, 1.53MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<05:16, 1.74MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:05<03:55, 2.34MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<04:40, 1.96MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<05:23, 1.70MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:07<04:12, 2.17MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:07<03:04, 2.96MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<05:36, 1.62MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<04:58, 1.82MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:09<03:42, 2.44MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<04:29, 2.01MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<05:13, 1.72MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<04:10, 2.15MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:11<03:01, 2.96MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<09:30, 940kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<07:41, 1.16MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:13<05:37, 1.58MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<05:48, 1.52MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<06:01, 1.47MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:15<04:41, 1.88MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:15<03:23, 2.59MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<16:05, 546kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<12:17, 715kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<08:50, 992kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<07:59, 1.09MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<06:36, 1.32MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:19<04:50, 1.79MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<05:12, 1.66MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<05:39, 1.53MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<04:28, 1.93MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:21<03:13, 2.66MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<09:20, 918kB/s] .vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<07:33, 1.13MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:23<05:31, 1.55MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<05:38, 1.51MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<05:55, 1.44MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<04:38, 1.83MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:25<03:20, 2.53MB/s].vector_cache/glove.6B.zip:  41%|████      | 356M/862M [02:27<09:18, 906kB/s] .vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<07:29, 1.13MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:27<05:26, 1.55MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<05:34, 1.50MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<05:51, 1.43MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<04:31, 1.85MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:29<03:15, 2.56MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<06:41, 1.24MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<05:38, 1.47MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:31<04:08, 2.00MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<04:37, 1.78MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<04:11, 1.97MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:33<03:08, 2.62MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<03:54, 2.09MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<04:37, 1.76MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<03:37, 2.24MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:35<02:39, 3.06MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<04:48, 1.68MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<04:18, 1.88MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:37<03:14, 2.49MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<03:56, 2.03MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<04:37, 1.74MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<03:38, 2.20MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:39<02:39, 2.99MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<04:43, 1.68MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<04:14, 1.88MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<03:11, 2.49MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<03:53, 2.03MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<03:38, 2.17MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:43<02:46, 2.83MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<03:34, 2.19MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<04:19, 1.81MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<03:28, 2.25MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:45<02:30, 3.08MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<07:36, 1.02MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<06:12, 1.25MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:47<04:33, 1.70MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<04:47, 1.60MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<05:13, 1.47MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<04:06, 1.87MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:49<02:57, 2.58MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<08:12, 927kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<06:39, 1.14MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<04:51, 1.56MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<04:58, 1.52MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<04:21, 1.73MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<03:15, 2.31MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<03:51, 1.94MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<03:33, 2.10MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<02:40, 2.79MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<03:25, 2.16MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:57<04:07, 1.80MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<03:18, 2.23MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:57<02:24, 3.05MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<07:47, 942kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<06:16, 1.17MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<04:33, 1.60MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<04:43, 1.54MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:01<04:59, 1.45MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<03:54, 1.85MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:01<02:49, 2.55MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<08:02, 895kB/s] .vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<06:24, 1.12MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<04:39, 1.54MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<04:49, 1.48MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<05:02, 1.41MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<03:56, 1.80MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:05<02:49, 2.49MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<07:13, 975kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<05:53, 1.20MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:07<04:18, 1.63MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<04:28, 1.56MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<04:45, 1.47MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:09<03:44, 1.87MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:09<02:41, 2.57MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<07:38, 906kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<06:05, 1.13MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<04:24, 1.56MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<04:35, 1.49MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<04:49, 1.42MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<03:42, 1.85MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:13<02:40, 2.54MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<04:31, 1.50MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<03:56, 1.72MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<02:55, 2.30MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<03:27, 1.94MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<03:58, 1.69MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<03:07, 2.15MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:17<02:14, 2.96MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<05:44, 1.16MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<04:47, 1.38MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:19<03:32, 1.87MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<03:50, 1.71MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<04:12, 1.56MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<03:16, 2.00MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:21<02:21, 2.76MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<05:08, 1.27MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<04:21, 1.49MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<03:12, 2.02MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<03:35, 1.79MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<03:15, 1.97MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<02:26, 2.63MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<03:02, 2.10MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<03:40, 1.73MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<02:52, 2.21MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:27<02:06, 3.00MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:29<03:34, 1.76MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:29<03:14, 1.93MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<02:24, 2.59MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<03:00, 2.07MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<03:33, 1.75MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<02:50, 2.18MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:31<02:03, 3.00MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<06:25, 958kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<05:09, 1.19MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<03:45, 1.63MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<03:58, 1.53MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<04:11, 1.45MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<03:16, 1.85MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:35<02:21, 2.55MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:37<06:33, 916kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<05:18, 1.13MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<03:50, 1.56MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:37<02:44, 2.17MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<1:13:31, 80.8kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<52:45, 113kB/s]   .vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<37:10, 159kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:39<25:53, 227kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<26:55, 218kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<19:30, 301kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<13:46, 424kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<10:47, 538kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<08:51, 655kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<06:27, 897kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:43<04:34, 1.26MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<05:41, 1.01MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<04:36, 1.25MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<03:21, 1.70MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<03:35, 1.57MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:47<03:49, 1.48MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<02:57, 1.91MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:47<02:08, 2.63MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<03:49, 1.46MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<03:19, 1.68MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:49<02:29, 2.24MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<02:53, 1.91MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<03:18, 1.67MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<02:34, 2.14MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:51<01:53, 2.90MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<03:01, 1.81MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<02:44, 1.99MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<02:03, 2.63MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:54<02:34, 2.09MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:55<03:03, 1.77MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<02:26, 2.20MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:55<01:45, 3.02MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<05:12, 1.02MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:57<04:15, 1.25MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<03:07, 1.69MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<03:17, 1.60MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:59<03:30, 1.49MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<02:42, 1.93MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [03:59<01:57, 2.66MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<03:41, 1.40MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<03:08, 1.65MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<02:18, 2.23MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:02<02:44, 1.86MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:03<03:06, 1.65MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:03<02:27, 2.07MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:03<01:46, 2.85MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<05:21, 940kB/s] .vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<04:20, 1.16MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<03:08, 1.59MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<03:14, 1.54MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<03:25, 1.45MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<02:41, 1.84MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:07<01:55, 2.55MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<05:21, 916kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<04:19, 1.13MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<03:07, 1.56MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<03:11, 1.51MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<03:25, 1.41MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<02:37, 1.84MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:11<01:54, 2.52MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<02:52, 1.65MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<02:31, 1.88MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<01:52, 2.53MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<02:20, 2.00MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<02:43, 1.72MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<02:10, 2.15MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:15<01:34, 2.93MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<04:56, 935kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<03:59, 1.15MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<02:54, 1.58MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<02:58, 1.53MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<03:09, 1.44MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<02:25, 1.88MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:19<01:44, 2.58MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<02:54, 1.54MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<02:31, 1.78MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<01:52, 2.37MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<02:16, 1.93MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<02:37, 1.68MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<02:05, 2.11MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:23<01:30, 2.88MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<04:39, 931kB/s] .vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<03:44, 1.16MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<02:42, 1.59MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<02:49, 1.51MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<02:58, 1.44MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<02:17, 1.86MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:27<01:39, 2.56MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<02:46, 1.52MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<02:25, 1.73MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:28<01:47, 2.33MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<02:07, 1.95MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<01:50, 2.23MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:30<01:23, 2.94MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<01:52, 2.16MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<02:12, 1.83MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:32<01:44, 2.32MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:33<01:15, 3.17MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<06:20, 630kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:34<04:53, 815kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:34<03:30, 1.13MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<03:15, 1.20MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<02:41, 1.45MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:36<01:58, 1.98MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<02:13, 1.73MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<02:00, 1.92MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:38<01:29, 2.56MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<01:50, 2.06MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<02:09, 1.75MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:40<01:44, 2.18MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:41<01:15, 2.97MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<03:57, 939kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<03:08, 1.19MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:42<02:17, 1.61MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<02:21, 1.55MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<02:29, 1.46MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<01:57, 1.86MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 647M/862M [04:44<01:23, 2.57MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<03:40, 973kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<02:57, 1.21MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:46<02:08, 1.66MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:48<02:15, 1.55MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<02:23, 1.46MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<01:52, 1.86MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:48<01:20, 2.56MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<03:47, 906kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<03:02, 1.13MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:50<02:12, 1.54MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<02:15, 1.50MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<01:57, 1.72MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<01:27, 2.30MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<01:42, 1.94MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<01:57, 1.68MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:54<01:32, 2.14MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:54<01:06, 2.95MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<02:44, 1.18MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [04:56<02:18, 1.40MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:56<01:41, 1.89MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<01:50, 1.72MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<01:39, 1.91MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:58<01:13, 2.56MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<01:30, 2.06MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 677M/862M [05:00<01:24, 2.19MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:00<01:04, 2.87MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<01:22, 2.19MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<01:40, 1.81MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<01:18, 2.29MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:02<00:57, 3.12MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<01:49, 1.62MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<01:37, 1.82MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:04<01:12, 2.42MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<01:26, 2.00MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<01:42, 1.69MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:06<01:21, 2.11MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:06<00:58, 2.91MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<02:46, 1.01MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<02:16, 1.24MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:08<01:39, 1.68MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<01:43, 1.59MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<01:50, 1.49MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<01:26, 1.89MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:10<01:01, 2.61MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<02:42, 987kB/s] .vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<02:12, 1.21MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:12<01:36, 1.65MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<01:39, 1.58MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<01:45, 1.48MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<01:22, 1.88MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:14<00:59, 2.58MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<02:47, 907kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<02:15, 1.13MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 711M/862M [05:16<01:37, 1.55MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:38, 1.51MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:43, 1.44MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<01:19, 1.86MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:18<00:56, 2.57MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:39, 1.45MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:24, 1.69MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:20<01:02, 2.28MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:22<01:13, 1.90MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:22<01:24, 1.66MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<01:05, 2.13MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:22<00:47, 2.91MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<01:25, 1.59MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<01:15, 1.80MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:24<00:56, 2.38MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<01:06, 1.99MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<01:16, 1.71MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<01:01, 2.14MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:26<00:43, 2.94MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<02:17, 928kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<01:50, 1.15MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<01:19, 1.58MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:20, 1.52MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:26, 1.42MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<01:07, 1.82MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:30<00:47, 2.50MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<02:12, 900kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:46, 1.12MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:32<01:16, 1.53MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<01:17, 1.49MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<01:20, 1.42MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<01:02, 1.83MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:34<00:44, 2.51MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<01:10, 1.57MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<01:02, 1.78MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<00:46, 2.36MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<00:54, 1.97MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<01:02, 1.70MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<00:48, 2.18MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:38<00:34, 2.98MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:08, 1.49MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<00:59, 1.71MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<00:43, 2.30MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<00:50, 1.94MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<00:58, 1.69MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<00:45, 2.15MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:42<00:31, 2.97MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<01:20, 1.17MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<01:07, 1.40MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:44<00:48, 1.90MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<00:52, 1.73MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<00:57, 1.56MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<00:44, 2.01MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:46<00:31, 2.76MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<00:54, 1.56MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:48, 1.77MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<00:35, 2.37MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:41, 1.98MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:48, 1.67MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:38, 2.09MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:50<00:27, 2.87MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<01:23, 927kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<01:07, 1.15MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:48, 1.58MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<00:47, 1.53MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<00:41, 1.74MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:54<00:30, 2.34MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:35, 1.96MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:32, 2.11MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:56<00:24, 2.77MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:30, 2.15MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:49, 1.30MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:42, 1.52MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<00:30, 2.06MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<00:32, 1.84MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:37, 1.63MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:28, 2.10MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:00<00:20, 2.82MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<00:28, 1.97MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<00:32, 1.73MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:02<00:25, 2.17MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:02<00:17, 2.97MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<01:11, 736kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<01:00, 858kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:44, 1.15MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:04<00:30, 1.61MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<01:40, 482kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<01:20, 598kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [06:06<00:57, 819kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:06<00:39, 1.15MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:50, 867kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:44, 989kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<00:32, 1.33MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:08<00:22, 1.85MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:35, 1.11MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:33, 1.20MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:24, 1.58MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:10<00:16, 2.19MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<02:05, 286kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<01:34, 379kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<01:06, 527kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:12<00:42, 745kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<02:10, 243kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<01:38, 322kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<01:08, 449kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:14<00:44, 636kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:15<00:59, 461kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:47, 574kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:34, 785kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:16<00:21, 1.11MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<00:35, 657kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:29, 796kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:20, 1.08MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:18<00:12, 1.51MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<09:20, 34.5kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<06:32, 48.8kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<04:26, 69.5kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:20<02:53, 99.0kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<01:51, 137kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<01:19, 189kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:52, 266kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:22<00:30, 379kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:29, 379kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:22, 488kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:15, 672kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:24<00:08, 950kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:10, 689kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:08, 812kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:05, 1.10MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:26<00:02, 1.54MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:02, 1.23MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:27<00:01, 1.34MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:00, 1.76MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 665/400000 [00:00<01:00, 6649.20it/s]  0%|          | 1569/400000 [00:00<00:55, 7220.75it/s]  1%|          | 2476/400000 [00:00<00:51, 7690.85it/s]  1%|          | 3342/400000 [00:00<00:49, 7956.89it/s]  1%|          | 4201/400000 [00:00<00:48, 8135.44it/s]  1%|▏         | 5056/400000 [00:00<00:47, 8255.24it/s]  1%|▏         | 5946/400000 [00:00<00:46, 8437.57it/s]  2%|▏         | 6862/400000 [00:00<00:45, 8641.31it/s]  2%|▏         | 7732/400000 [00:00<00:45, 8656.97it/s]  2%|▏         | 8656/400000 [00:01<00:44, 8823.37it/s]  2%|▏         | 9554/400000 [00:01<00:44, 8868.90it/s]  3%|▎         | 10435/400000 [00:01<00:44, 8850.77it/s]  3%|▎         | 11349/400000 [00:01<00:43, 8934.85it/s]  3%|▎         | 12238/400000 [00:01<00:43, 8883.66it/s]  3%|▎         | 13145/400000 [00:01<00:43, 8937.01it/s]  4%|▎         | 14037/400000 [00:01<00:43, 8811.82it/s]  4%|▎         | 14917/400000 [00:01<00:44, 8740.78it/s]  4%|▍         | 15791/400000 [00:01<00:44, 8555.07it/s]  4%|▍         | 16647/400000 [00:01<00:45, 8511.62it/s]  4%|▍         | 17539/400000 [00:02<00:44, 8628.92it/s]  5%|▍         | 18403/400000 [00:02<00:44, 8579.13it/s]  5%|▍         | 19319/400000 [00:02<00:43, 8742.96it/s]  5%|▌         | 20195/400000 [00:02<00:43, 8740.66it/s]  5%|▌         | 21070/400000 [00:02<00:43, 8692.03it/s]  5%|▌         | 21966/400000 [00:02<00:43, 8768.92it/s]  6%|▌         | 22844/400000 [00:02<00:43, 8662.33it/s]  6%|▌         | 23711/400000 [00:02<00:43, 8650.83it/s]  6%|▌         | 24588/400000 [00:02<00:43, 8685.69it/s]  6%|▋         | 25457/400000 [00:02<00:43, 8628.90it/s]  7%|▋         | 26321/400000 [00:03<00:43, 8538.27it/s]  7%|▋         | 27202/400000 [00:03<00:43, 8615.05it/s]  7%|▋         | 28098/400000 [00:03<00:42, 8713.21it/s]  7%|▋         | 29024/400000 [00:03<00:41, 8868.86it/s]  7%|▋         | 29912/400000 [00:03<00:41, 8847.40it/s]  8%|▊         | 30816/400000 [00:03<00:41, 8901.86it/s]  8%|▊         | 31707/400000 [00:03<00:41, 8882.21it/s]  8%|▊         | 32603/400000 [00:03<00:41, 8904.23it/s]  8%|▊         | 33494/400000 [00:03<00:41, 8877.86it/s]  9%|▊         | 34383/400000 [00:03<00:41, 8754.16it/s]  9%|▉         | 35259/400000 [00:04<00:41, 8746.00it/s]  9%|▉         | 36134/400000 [00:04<00:41, 8697.94it/s]  9%|▉         | 37009/400000 [00:04<00:41, 8711.62it/s]  9%|▉         | 37881/400000 [00:04<00:41, 8688.49it/s] 10%|▉         | 38751/400000 [00:04<00:41, 8665.55it/s] 10%|▉         | 39629/400000 [00:04<00:41, 8699.23it/s] 10%|█         | 40500/400000 [00:04<00:41, 8684.57it/s] 10%|█         | 41370/400000 [00:04<00:41, 8687.71it/s] 11%|█         | 42239/400000 [00:04<00:41, 8609.98it/s] 11%|█         | 43127/400000 [00:04<00:41, 8689.11it/s] 11%|█         | 43997/400000 [00:05<00:41, 8626.72it/s] 11%|█         | 44862/400000 [00:05<00:41, 8633.33it/s] 11%|█▏        | 45735/400000 [00:05<00:40, 8661.75it/s] 12%|█▏        | 46607/400000 [00:05<00:40, 8677.06it/s] 12%|█▏        | 47475/400000 [00:05<00:40, 8648.75it/s] 12%|█▏        | 48340/400000 [00:05<00:40, 8634.28it/s] 12%|█▏        | 49204/400000 [00:05<00:40, 8635.79it/s] 13%|█▎        | 50106/400000 [00:05<00:40, 8745.59it/s] 13%|█▎        | 50982/400000 [00:05<00:39, 8748.46it/s] 13%|█▎        | 51869/400000 [00:05<00:39, 8783.70it/s] 13%|█▎        | 52748/400000 [00:06<00:40, 8605.31it/s] 13%|█▎        | 53610/400000 [00:06<00:40, 8533.95it/s] 14%|█▎        | 54527/400000 [00:06<00:39, 8714.23it/s] 14%|█▍        | 55416/400000 [00:06<00:39, 8764.20it/s] 14%|█▍        | 56294/400000 [00:06<00:39, 8719.04it/s] 14%|█▍        | 57167/400000 [00:06<00:39, 8668.27it/s] 15%|█▍        | 58044/400000 [00:06<00:39, 8696.89it/s] 15%|█▍        | 58930/400000 [00:06<00:39, 8742.53it/s] 15%|█▍        | 59805/400000 [00:06<00:39, 8714.40it/s] 15%|█▌        | 60677/400000 [00:06<00:39, 8577.28it/s] 15%|█▌        | 61559/400000 [00:07<00:39, 8648.61it/s] 16%|█▌        | 62445/400000 [00:07<00:38, 8708.86it/s] 16%|█▌        | 63367/400000 [00:07<00:38, 8855.48it/s] 16%|█▌        | 64268/400000 [00:07<00:37, 8901.23it/s] 16%|█▋        | 65159/400000 [00:07<00:38, 8702.13it/s] 17%|█▋        | 66031/400000 [00:07<00:38, 8571.71it/s] 17%|█▋        | 66890/400000 [00:07<00:38, 8566.84it/s] 17%|█▋        | 67748/400000 [00:07<00:39, 8467.94it/s] 17%|█▋        | 68618/400000 [00:07<00:38, 8534.30it/s] 17%|█▋        | 69475/400000 [00:07<00:38, 8543.16it/s] 18%|█▊        | 70333/400000 [00:08<00:38, 8553.26it/s] 18%|█▊        | 71209/400000 [00:08<00:38, 8613.86it/s] 18%|█▊        | 72113/400000 [00:08<00:37, 8735.62it/s] 18%|█▊        | 73008/400000 [00:08<00:37, 8796.83it/s] 18%|█▊        | 73896/400000 [00:08<00:36, 8819.50it/s] 19%|█▊        | 74779/400000 [00:08<00:36, 8814.11it/s] 19%|█▉        | 75664/400000 [00:08<00:36, 8823.47it/s] 19%|█▉        | 76547/400000 [00:08<00:37, 8708.86it/s] 19%|█▉        | 77419/400000 [00:08<00:37, 8559.46it/s] 20%|█▉        | 78276/400000 [00:09<00:38, 8376.48it/s] 20%|█▉        | 79135/400000 [00:09<00:38, 8437.04it/s] 20%|██        | 80028/400000 [00:09<00:37, 8577.32it/s] 20%|██        | 80900/400000 [00:09<00:37, 8618.06it/s] 20%|██        | 81774/400000 [00:09<00:36, 8651.77it/s] 21%|██        | 82666/400000 [00:09<00:36, 8728.93it/s] 21%|██        | 83545/400000 [00:09<00:36, 8746.50it/s] 21%|██        | 84441/400000 [00:09<00:35, 8808.99it/s] 21%|██▏       | 85323/400000 [00:09<00:35, 8798.54it/s] 22%|██▏       | 86204/400000 [00:09<00:36, 8656.69it/s] 22%|██▏       | 87071/400000 [00:10<00:36, 8538.94it/s] 22%|██▏       | 87926/400000 [00:10<00:36, 8532.78it/s] 22%|██▏       | 88885/400000 [00:10<00:35, 8824.15it/s] 22%|██▏       | 89831/400000 [00:10<00:34, 9004.69it/s] 23%|██▎       | 90751/400000 [00:10<00:34, 9060.09it/s] 23%|██▎       | 91660/400000 [00:10<00:34, 9068.18it/s] 23%|██▎       | 92569/400000 [00:10<00:34, 8997.82it/s] 23%|██▎       | 93526/400000 [00:10<00:33, 9161.10it/s] 24%|██▎       | 94444/400000 [00:10<00:33, 9140.45it/s] 24%|██▍       | 95360/400000 [00:10<00:33, 9142.32it/s] 24%|██▍       | 96275/400000 [00:11<00:33, 9034.09it/s] 24%|██▍       | 97180/400000 [00:11<00:34, 8780.72it/s] 25%|██▍       | 98061/400000 [00:11<00:34, 8688.24it/s] 25%|██▍       | 98949/400000 [00:11<00:34, 8744.41it/s] 25%|██▍       | 99859/400000 [00:11<00:33, 8847.15it/s] 25%|██▌       | 100762/400000 [00:11<00:33, 8899.30it/s] 25%|██▌       | 101659/400000 [00:11<00:33, 8917.89it/s] 26%|██▌       | 102552/400000 [00:11<00:34, 8571.96it/s] 26%|██▌       | 103425/400000 [00:11<00:34, 8617.38it/s] 26%|██▌       | 104308/400000 [00:11<00:34, 8677.49it/s] 26%|██▋       | 105180/400000 [00:12<00:33, 8687.89it/s] 27%|██▋       | 106051/400000 [00:12<00:34, 8611.70it/s] 27%|██▋       | 106914/400000 [00:12<00:34, 8587.53it/s] 27%|██▋       | 107774/400000 [00:12<00:34, 8516.83it/s] 27%|██▋       | 108627/400000 [00:12<00:34, 8397.53it/s] 27%|██▋       | 109468/400000 [00:12<00:34, 8371.04it/s] 28%|██▊       | 110306/400000 [00:12<00:35, 8270.98it/s] 28%|██▊       | 111171/400000 [00:12<00:34, 8379.81it/s] 28%|██▊       | 112010/400000 [00:12<00:34, 8273.99it/s] 28%|██▊       | 112841/400000 [00:12<00:34, 8283.17it/s] 28%|██▊       | 113676/400000 [00:13<00:34, 8301.50it/s] 29%|██▊       | 114567/400000 [00:13<00:33, 8473.23it/s] 29%|██▉       | 115478/400000 [00:13<00:32, 8653.38it/s] 29%|██▉       | 116367/400000 [00:13<00:32, 8721.12it/s] 29%|██▉       | 117257/400000 [00:13<00:32, 8771.83it/s] 30%|██▉       | 118136/400000 [00:13<00:32, 8764.90it/s] 30%|██▉       | 119017/400000 [00:13<00:32, 8778.28it/s] 30%|██▉       | 119896/400000 [00:13<00:31, 8762.26it/s] 30%|███       | 120773/400000 [00:13<00:32, 8657.46it/s] 30%|███       | 121674/400000 [00:13<00:31, 8759.26it/s] 31%|███       | 122554/400000 [00:14<00:31, 8770.86it/s] 31%|███       | 123432/400000 [00:14<00:31, 8668.30it/s] 31%|███       | 124300/400000 [00:14<00:32, 8597.26it/s] 31%|███▏      | 125184/400000 [00:14<00:31, 8666.48it/s] 32%|███▏      | 126074/400000 [00:14<00:31, 8732.63it/s] 32%|███▏      | 126948/400000 [00:14<00:31, 8657.39it/s] 32%|███▏      | 127815/400000 [00:14<00:31, 8620.27it/s] 32%|███▏      | 128691/400000 [00:14<00:31, 8660.14it/s] 32%|███▏      | 129558/400000 [00:14<00:31, 8510.02it/s] 33%|███▎      | 130448/400000 [00:15<00:31, 8622.97it/s] 33%|███▎      | 131322/400000 [00:15<00:31, 8654.96it/s] 33%|███▎      | 132202/400000 [00:15<00:30, 8695.31it/s] 33%|███▎      | 133114/400000 [00:15<00:30, 8816.97it/s] 34%|███▎      | 134005/400000 [00:15<00:30, 8842.88it/s] 34%|███▎      | 134909/400000 [00:15<00:29, 8899.89it/s] 34%|███▍      | 135815/400000 [00:15<00:29, 8945.56it/s] 34%|███▍      | 136710/400000 [00:15<00:29, 8928.56it/s] 34%|███▍      | 137604/400000 [00:15<00:29, 8850.40it/s] 35%|███▍      | 138490/400000 [00:15<00:29, 8809.34it/s] 35%|███▍      | 139372/400000 [00:16<00:29, 8759.23it/s] 35%|███▌      | 140249/400000 [00:16<00:30, 8526.92it/s] 35%|███▌      | 141104/400000 [00:16<00:30, 8458.72it/s] 35%|███▌      | 141974/400000 [00:16<00:30, 8528.86it/s] 36%|███▌      | 142828/400000 [00:16<00:30, 8424.51it/s] 36%|███▌      | 143722/400000 [00:16<00:29, 8570.91it/s] 36%|███▌      | 144627/400000 [00:16<00:29, 8707.16it/s] 36%|███▋      | 145538/400000 [00:16<00:28, 8822.94it/s] 37%|███▋      | 146438/400000 [00:16<00:28, 8873.92it/s] 37%|███▋      | 147332/400000 [00:16<00:28, 8892.01it/s] 37%|███▋      | 148222/400000 [00:17<00:28, 8841.91it/s] 37%|███▋      | 149107/400000 [00:17<00:28, 8785.72it/s] 37%|███▋      | 149987/400000 [00:17<00:28, 8765.16it/s] 38%|███▊      | 150864/400000 [00:17<00:29, 8391.68it/s] 38%|███▊      | 151716/400000 [00:17<00:29, 8427.08it/s] 38%|███▊      | 152587/400000 [00:17<00:29, 8509.32it/s] 38%|███▊      | 153464/400000 [00:17<00:28, 8585.05it/s] 39%|███▊      | 154342/400000 [00:17<00:28, 8640.35it/s] 39%|███▉      | 155264/400000 [00:17<00:27, 8804.99it/s] 39%|███▉      | 156147/400000 [00:17<00:28, 8705.84it/s] 39%|███▉      | 157019/400000 [00:18<00:28, 8629.14it/s] 39%|███▉      | 157898/400000 [00:18<00:27, 8674.65it/s] 40%|███▉      | 158767/400000 [00:18<00:29, 8258.48it/s] 40%|███▉      | 159598/400000 [00:18<00:29, 8229.89it/s] 40%|████      | 160425/400000 [00:18<00:29, 8135.12it/s] 40%|████      | 161319/400000 [00:18<00:28, 8358.85it/s] 41%|████      | 162207/400000 [00:18<00:27, 8507.02it/s] 41%|████      | 163113/400000 [00:18<00:27, 8664.31it/s] 41%|████      | 163983/400000 [00:18<00:27, 8638.17it/s] 41%|████      | 164849/400000 [00:18<00:28, 8263.20it/s] 41%|████▏     | 165716/400000 [00:19<00:27, 8378.68it/s] 42%|████▏     | 166558/400000 [00:19<00:27, 8365.19it/s] 42%|████▏     | 167419/400000 [00:19<00:27, 8436.94it/s] 42%|████▏     | 168265/400000 [00:19<00:27, 8345.66it/s] 42%|████▏     | 169102/400000 [00:19<00:27, 8307.92it/s] 42%|████▏     | 169934/400000 [00:19<00:27, 8246.12it/s] 43%|████▎     | 170760/400000 [00:19<00:27, 8219.53it/s] 43%|████▎     | 171583/400000 [00:19<00:27, 8197.03it/s] 43%|████▎     | 172404/400000 [00:19<00:28, 8059.52it/s] 43%|████▎     | 173211/400000 [00:20<00:28, 8011.15it/s] 44%|████▎     | 174080/400000 [00:20<00:27, 8201.90it/s] 44%|████▎     | 174911/400000 [00:20<00:27, 8233.95it/s] 44%|████▍     | 175747/400000 [00:20<00:27, 8270.68it/s] 44%|████▍     | 176588/400000 [00:20<00:26, 8309.82it/s] 44%|████▍     | 177434/400000 [00:20<00:26, 8352.32it/s] 45%|████▍     | 178286/400000 [00:20<00:26, 8400.19it/s] 45%|████▍     | 179127/400000 [00:20<00:26, 8400.20it/s] 45%|████▍     | 179968/400000 [00:20<00:26, 8367.20it/s] 45%|████▌     | 180805/400000 [00:20<00:26, 8182.21it/s] 45%|████▌     | 181674/400000 [00:21<00:26, 8326.61it/s] 46%|████▌     | 182511/400000 [00:21<00:26, 8337.18it/s] 46%|████▌     | 183346/400000 [00:21<00:26, 8167.35it/s] 46%|████▌     | 184165/400000 [00:21<00:26, 8049.55it/s] 46%|████▋     | 185040/400000 [00:21<00:26, 8246.82it/s] 46%|████▋     | 185914/400000 [00:21<00:25, 8386.72it/s] 47%|████▋     | 186766/400000 [00:21<00:25, 8424.11it/s] 47%|████▋     | 187641/400000 [00:21<00:24, 8517.64it/s] 47%|████▋     | 188548/400000 [00:21<00:24, 8675.06it/s] 47%|████▋     | 189418/400000 [00:21<00:24, 8538.21it/s] 48%|████▊     | 190310/400000 [00:22<00:24, 8647.30it/s] 48%|████▊     | 191177/400000 [00:22<00:24, 8633.05it/s] 48%|████▊     | 192042/400000 [00:22<00:24, 8468.40it/s] 48%|████▊     | 192891/400000 [00:22<00:24, 8381.75it/s] 48%|████▊     | 193731/400000 [00:22<00:25, 7967.08it/s] 49%|████▊     | 194533/400000 [00:22<00:26, 7849.09it/s] 49%|████▉     | 195329/400000 [00:22<00:25, 7881.07it/s] 49%|████▉     | 196143/400000 [00:22<00:25, 7956.50it/s] 49%|████▉     | 196941/400000 [00:22<00:25, 7855.93it/s] 49%|████▉     | 197816/400000 [00:22<00:24, 8102.11it/s] 50%|████▉     | 198645/400000 [00:23<00:24, 8155.08it/s] 50%|████▉     | 199482/400000 [00:23<00:24, 8217.67it/s] 50%|█████     | 200306/400000 [00:23<00:24, 8057.15it/s] 50%|█████     | 201114/400000 [00:23<00:24, 7959.35it/s] 50%|█████     | 201912/400000 [00:23<00:25, 7855.35it/s] 51%|█████     | 202699/400000 [00:23<00:25, 7835.92it/s] 51%|█████     | 203553/400000 [00:23<00:24, 8033.27it/s] 51%|█████     | 204423/400000 [00:23<00:23, 8221.16it/s] 51%|█████▏    | 205248/400000 [00:23<00:24, 8075.97it/s] 52%|█████▏    | 206058/400000 [00:24<00:24, 8016.25it/s] 52%|█████▏    | 206874/400000 [00:24<00:23, 8056.12it/s] 52%|█████▏    | 207722/400000 [00:24<00:23, 8177.80it/s] 52%|█████▏    | 208542/400000 [00:24<00:23, 8174.96it/s] 52%|█████▏    | 209382/400000 [00:24<00:23, 8240.35it/s] 53%|█████▎    | 210223/400000 [00:24<00:22, 8288.67it/s] 53%|█████▎    | 211072/400000 [00:24<00:22, 8346.25it/s] 53%|█████▎    | 211908/400000 [00:24<00:22, 8311.95it/s] 53%|█████▎    | 212753/400000 [00:24<00:22, 8350.32it/s] 53%|█████▎    | 213616/400000 [00:24<00:22, 8429.79it/s] 54%|█████▎    | 214488/400000 [00:25<00:21, 8514.51it/s] 54%|█████▍    | 215365/400000 [00:25<00:21, 8589.53it/s] 54%|█████▍    | 216244/400000 [00:25<00:21, 8647.53it/s] 54%|█████▍    | 217147/400000 [00:25<00:20, 8755.70it/s] 55%|█████▍    | 218033/400000 [00:25<00:20, 8785.44it/s] 55%|█████▍    | 218932/400000 [00:25<00:20, 8845.20it/s] 55%|█████▍    | 219825/400000 [00:25<00:20, 8869.71it/s] 55%|█████▌    | 220713/400000 [00:25<00:20, 8631.27it/s] 55%|█████▌    | 221578/400000 [00:25<00:20, 8550.15it/s] 56%|█████▌    | 222435/400000 [00:25<00:21, 8418.93it/s] 56%|█████▌    | 223279/400000 [00:26<00:21, 8353.94it/s] 56%|█████▌    | 224123/400000 [00:26<00:20, 8377.05it/s] 56%|█████▌    | 224962/400000 [00:26<00:20, 8377.31it/s] 56%|█████▋    | 225842/400000 [00:26<00:20, 8498.84it/s] 57%|█████▋    | 226763/400000 [00:26<00:19, 8697.79it/s] 57%|█████▋    | 227662/400000 [00:26<00:19, 8783.24it/s] 57%|█████▋    | 228565/400000 [00:26<00:19, 8853.46it/s] 57%|█████▋    | 229458/400000 [00:26<00:19, 8875.63it/s] 58%|█████▊    | 230347/400000 [00:26<00:19, 8865.88it/s] 58%|█████▊    | 231235/400000 [00:26<00:19, 8785.20it/s] 58%|█████▊    | 232119/400000 [00:27<00:19, 8799.23it/s] 58%|█████▊    | 233000/400000 [00:27<00:19, 8627.41it/s] 58%|█████▊    | 233894/400000 [00:27<00:19, 8716.56it/s] 59%|█████▊    | 234797/400000 [00:27<00:18, 8808.05it/s] 59%|█████▉    | 235702/400000 [00:27<00:18, 8878.49it/s] 59%|█████▉    | 236615/400000 [00:27<00:18, 8950.22it/s] 59%|█████▉    | 237511/400000 [00:27<00:18, 8700.34it/s] 60%|█████▉    | 238384/400000 [00:27<00:18, 8568.08it/s] 60%|█████▉    | 239243/400000 [00:27<00:18, 8544.44it/s] 60%|██████    | 240099/400000 [00:27<00:18, 8486.65it/s] 60%|██████    | 240949/400000 [00:28<00:19, 8336.63it/s] 60%|██████    | 241784/400000 [00:28<00:19, 8246.68it/s] 61%|██████    | 242651/400000 [00:28<00:18, 8324.19it/s] 61%|██████    | 243485/400000 [00:28<00:18, 8297.79it/s] 61%|██████    | 244320/400000 [00:28<00:18, 8309.16it/s] 61%|██████▏   | 245152/400000 [00:28<00:18, 8302.63it/s] 61%|██████▏   | 245983/400000 [00:28<00:18, 8258.12it/s] 62%|██████▏   | 246810/400000 [00:28<00:18, 8241.01it/s] 62%|██████▏   | 247646/400000 [00:28<00:18, 8275.03it/s] 62%|██████▏   | 248519/400000 [00:28<00:18, 8406.43it/s] 62%|██████▏   | 249361/400000 [00:29<00:18, 7986.15it/s] 63%|██████▎   | 250171/400000 [00:29<00:18, 8018.51it/s] 63%|██████▎   | 251034/400000 [00:29<00:18, 8190.81it/s] 63%|██████▎   | 251905/400000 [00:29<00:17, 8339.14it/s] 63%|██████▎   | 252778/400000 [00:29<00:17, 8451.73it/s] 63%|██████▎   | 253626/400000 [00:29<00:17, 8383.79it/s] 64%|██████▎   | 254467/400000 [00:29<00:17, 8135.94it/s] 64%|██████▍   | 255284/400000 [00:29<00:17, 8075.17it/s] 64%|██████▍   | 256138/400000 [00:29<00:17, 8207.68it/s] 64%|██████▍   | 257017/400000 [00:30<00:17, 8373.60it/s] 64%|██████▍   | 257920/400000 [00:30<00:16, 8558.59it/s] 65%|██████▍   | 258817/400000 [00:30<00:16, 8676.81it/s] 65%|██████▍   | 259709/400000 [00:30<00:16, 8745.51it/s] 65%|██████▌   | 260624/400000 [00:30<00:15, 8862.60it/s] 65%|██████▌   | 261546/400000 [00:30<00:15, 8964.74it/s] 66%|██████▌   | 262449/400000 [00:30<00:15, 8981.23it/s] 66%|██████▌   | 263349/400000 [00:30<00:15, 8920.24it/s] 66%|██████▌   | 264242/400000 [00:30<00:15, 8807.93it/s] 66%|██████▋   | 265124/400000 [00:30<00:15, 8580.13it/s] 66%|██████▋   | 265984/400000 [00:31<00:15, 8557.46it/s] 67%|██████▋   | 266842/400000 [00:31<00:15, 8499.61it/s] 67%|██████▋   | 267715/400000 [00:31<00:15, 8565.82it/s] 67%|██████▋   | 268573/400000 [00:31<00:15, 8465.30it/s] 67%|██████▋   | 269470/400000 [00:31<00:15, 8610.26it/s] 68%|██████▊   | 270397/400000 [00:31<00:14, 8798.06it/s] 68%|██████▊   | 271314/400000 [00:31<00:14, 8906.08it/s] 68%|██████▊   | 272212/400000 [00:31<00:14, 8927.80it/s] 68%|██████▊   | 273106/400000 [00:31<00:14, 8923.89it/s] 68%|██████▊   | 274000/400000 [00:31<00:14, 8802.43it/s] 69%|██████▊   | 274882/400000 [00:32<00:14, 8444.13it/s] 69%|██████▉   | 275749/400000 [00:32<00:14, 8508.98it/s] 69%|██████▉   | 276676/400000 [00:32<00:14, 8722.68it/s] 69%|██████▉   | 277552/400000 [00:32<00:14, 8730.26it/s] 70%|██████▉   | 278428/400000 [00:32<00:14, 8652.53it/s] 70%|██████▉   | 279303/400000 [00:32<00:13, 8679.41it/s] 70%|███████   | 280179/400000 [00:32<00:13, 8701.21it/s] 70%|███████   | 281051/400000 [00:32<00:13, 8609.61it/s] 70%|███████   | 281913/400000 [00:32<00:13, 8540.05it/s] 71%|███████   | 282768/400000 [00:32<00:14, 8329.27it/s] 71%|███████   | 283642/400000 [00:33<00:13, 8448.15it/s] 71%|███████   | 284512/400000 [00:33<00:13, 8520.78it/s] 71%|███████▏  | 285373/400000 [00:33<00:13, 8545.38it/s] 72%|███████▏  | 286242/400000 [00:33<00:13, 8587.29it/s] 72%|███████▏  | 287123/400000 [00:33<00:13, 8650.71it/s] 72%|███████▏  | 288035/400000 [00:33<00:12, 8786.20it/s] 72%|███████▏  | 288950/400000 [00:33<00:12, 8891.64it/s] 72%|███████▏  | 289841/400000 [00:33<00:12, 8872.48it/s] 73%|███████▎  | 290729/400000 [00:33<00:12, 8857.90it/s] 73%|███████▎  | 291621/400000 [00:33<00:12, 8876.29it/s] 73%|███████▎  | 292509/400000 [00:34<00:12, 8870.94it/s] 73%|███████▎  | 293397/400000 [00:34<00:12, 8805.73it/s] 74%|███████▎  | 294278/400000 [00:34<00:12, 8756.86it/s] 74%|███████▍  | 295154/400000 [00:34<00:11, 8747.14it/s] 74%|███████▍  | 296051/400000 [00:34<00:11, 8812.09it/s] 74%|███████▍  | 296972/400000 [00:34<00:11, 8926.47it/s] 74%|███████▍  | 297901/400000 [00:34<00:11, 9031.96it/s] 75%|███████▍  | 298818/400000 [00:34<00:11, 9070.56it/s] 75%|███████▍  | 299728/400000 [00:34<00:11, 9078.75it/s] 75%|███████▌  | 300637/400000 [00:34<00:10, 9058.51it/s] 75%|███████▌  | 301544/400000 [00:35<00:10, 9057.49it/s] 76%|███████▌  | 302450/400000 [00:35<00:10, 9028.28it/s] 76%|███████▌  | 303353/400000 [00:35<00:10, 8970.90it/s] 76%|███████▌  | 304251/400000 [00:35<00:10, 8922.28it/s] 76%|███████▋  | 305144/400000 [00:35<00:11, 8521.50it/s] 77%|███████▋  | 306014/400000 [00:35<00:10, 8571.46it/s] 77%|███████▋  | 306888/400000 [00:35<00:10, 8621.16it/s] 77%|███████▋  | 307774/400000 [00:35<00:10, 8688.64it/s] 77%|███████▋  | 308648/400000 [00:35<00:10, 8703.37it/s] 77%|███████▋  | 309520/400000 [00:36<00:10, 8633.15it/s] 78%|███████▊  | 310385/400000 [00:36<00:10, 8294.07it/s] 78%|███████▊  | 311264/400000 [00:36<00:10, 8435.54it/s] 78%|███████▊  | 312166/400000 [00:36<00:10, 8601.95it/s] 78%|███████▊  | 313045/400000 [00:36<00:10, 8656.23it/s] 78%|███████▊  | 313913/400000 [00:36<00:10, 8502.23it/s] 79%|███████▊  | 314766/400000 [00:36<00:10, 8383.17it/s] 79%|███████▉  | 315607/400000 [00:36<00:10, 8348.34it/s] 79%|███████▉  | 316445/400000 [00:36<00:09, 8357.06it/s] 79%|███████▉  | 317282/400000 [00:36<00:09, 8335.24it/s] 80%|███████▉  | 318126/400000 [00:37<00:09, 8364.54it/s] 80%|███████▉  | 318965/400000 [00:37<00:09, 8371.76it/s] 80%|███████▉  | 319805/400000 [00:37<00:09, 8380.12it/s] 80%|████████  | 320644/400000 [00:37<00:09, 8376.88it/s] 80%|████████  | 321482/400000 [00:37<00:09, 8252.84it/s] 81%|████████  | 322372/400000 [00:37<00:09, 8433.40it/s] 81%|████████  | 323240/400000 [00:37<00:09, 8505.47it/s] 81%|████████  | 324092/400000 [00:37<00:09, 8421.08it/s] 81%|████████  | 324936/400000 [00:37<00:09, 8198.46it/s] 81%|████████▏ | 325768/400000 [00:37<00:09, 8233.08it/s] 82%|████████▏ | 326606/400000 [00:38<00:08, 8275.54it/s] 82%|████████▏ | 327475/400000 [00:38<00:08, 8393.60it/s] 82%|████████▏ | 328316/400000 [00:38<00:08, 8394.87it/s] 82%|████████▏ | 329172/400000 [00:38<00:08, 8441.90it/s] 83%|████████▎ | 330025/400000 [00:38<00:08, 8467.87it/s] 83%|████████▎ | 330882/400000 [00:38<00:08, 8498.14it/s] 83%|████████▎ | 331737/400000 [00:38<00:08, 8511.55it/s] 83%|████████▎ | 332589/400000 [00:38<00:07, 8448.66it/s] 83%|████████▎ | 333435/400000 [00:38<00:08, 8303.46it/s] 84%|████████▎ | 334267/400000 [00:38<00:07, 8256.77it/s] 84%|████████▍ | 335099/400000 [00:39<00:07, 8274.42it/s] 84%|████████▍ | 335986/400000 [00:39<00:07, 8442.86it/s] 84%|████████▍ | 336843/400000 [00:39<00:07, 8480.13it/s] 84%|████████▍ | 337720/400000 [00:39<00:07, 8563.88it/s] 85%|████████▍ | 338578/400000 [00:39<00:07, 8504.90it/s] 85%|████████▍ | 339430/400000 [00:39<00:07, 8500.72it/s] 85%|████████▌ | 340343/400000 [00:39<00:06, 8677.59it/s] 85%|████████▌ | 341259/400000 [00:39<00:06, 8815.65it/s] 86%|████████▌ | 342142/400000 [00:39<00:06, 8800.58it/s] 86%|████████▌ | 343024/400000 [00:39<00:06, 8768.00it/s] 86%|████████▌ | 343902/400000 [00:40<00:06, 8457.39it/s] 86%|████████▌ | 344751/400000 [00:40<00:06, 8382.17it/s] 86%|████████▋ | 345592/400000 [00:40<00:06, 8349.67it/s] 87%|████████▋ | 346452/400000 [00:40<00:06, 8421.62it/s] 87%|████████▋ | 347296/400000 [00:40<00:06, 8415.84it/s] 87%|████████▋ | 348139/400000 [00:40<00:06, 8229.60it/s] 87%|████████▋ | 348988/400000 [00:40<00:06, 8305.39it/s] 87%|████████▋ | 349858/400000 [00:40<00:05, 8418.02it/s] 88%|████████▊ | 350713/400000 [00:40<00:05, 8455.78it/s] 88%|████████▊ | 351560/400000 [00:41<00:05, 8263.54it/s] 88%|████████▊ | 352414/400000 [00:41<00:05, 8342.35it/s] 88%|████████▊ | 353261/400000 [00:41<00:05, 8378.04it/s] 89%|████████▊ | 354138/400000 [00:41<00:05, 8491.42it/s] 89%|████████▉ | 355004/400000 [00:41<00:05, 8541.03it/s] 89%|████████▉ | 355879/400000 [00:41<00:05, 8600.93it/s] 89%|████████▉ | 356748/400000 [00:41<00:05, 8626.45it/s] 89%|████████▉ | 357622/400000 [00:41<00:04, 8659.85it/s] 90%|████████▉ | 358498/400000 [00:41<00:04, 8689.49it/s] 90%|████████▉ | 359376/400000 [00:41<00:04, 8714.22it/s] 90%|█████████ | 360291/400000 [00:42<00:04, 8840.29it/s] 90%|█████████ | 361190/400000 [00:42<00:04, 8884.07it/s] 91%|█████████ | 362083/400000 [00:42<00:04, 8896.92it/s] 91%|█████████ | 362977/400000 [00:42<00:04, 8909.62it/s] 91%|█████████ | 363869/400000 [00:42<00:04, 8857.27it/s] 91%|█████████ | 364767/400000 [00:42<00:03, 8890.86it/s] 91%|█████████▏| 365657/400000 [00:42<00:03, 8767.80it/s] 92%|█████████▏| 366561/400000 [00:42<00:03, 8845.57it/s] 92%|█████████▏| 367451/400000 [00:42<00:03, 8861.46it/s] 92%|█████████▏| 368350/400000 [00:42<00:03, 8897.12it/s] 92%|█████████▏| 369241/400000 [00:43<00:03, 8758.40it/s] 93%|█████████▎| 370143/400000 [00:43<00:03, 8834.76it/s] 93%|█████████▎| 371050/400000 [00:43<00:03, 8903.59it/s] 93%|█████████▎| 371943/400000 [00:43<00:03, 8909.18it/s] 93%|█████████▎| 372841/400000 [00:43<00:03, 8929.95it/s] 93%|█████████▎| 373735/400000 [00:43<00:02, 8869.16it/s] 94%|█████████▎| 374623/400000 [00:43<00:02, 8792.42it/s] 94%|█████████▍| 375503/400000 [00:43<00:02, 8554.05it/s] 94%|█████████▍| 376361/400000 [00:43<00:02, 8427.37it/s] 94%|█████████▍| 377206/400000 [00:43<00:02, 8390.83it/s] 95%|█████████▍| 378047/400000 [00:44<00:02, 8353.88it/s] 95%|█████████▍| 378885/400000 [00:44<00:02, 8360.18it/s] 95%|█████████▍| 379747/400000 [00:44<00:02, 8435.02it/s] 95%|█████████▌| 380592/400000 [00:44<00:02, 8368.86it/s] 95%|█████████▌| 381430/400000 [00:44<00:02, 8335.02it/s] 96%|█████████▌| 382339/400000 [00:44<00:02, 8547.83it/s] 96%|█████████▌| 383196/400000 [00:44<00:02, 8377.75it/s] 96%|█████████▌| 384036/400000 [00:44<00:01, 8325.75it/s] 96%|█████████▌| 384870/400000 [00:44<00:01, 8184.55it/s] 96%|█████████▋| 385690/400000 [00:44<00:01, 8145.41it/s] 97%|█████████▋| 386506/400000 [00:45<00:01, 8136.69it/s] 97%|█████████▋| 387321/400000 [00:45<00:01, 8073.53it/s] 97%|█████████▋| 388215/400000 [00:45<00:01, 8314.16it/s] 97%|█████████▋| 389088/400000 [00:45<00:01, 8433.63it/s] 97%|█████████▋| 389940/400000 [00:45<00:01, 8457.57it/s] 98%|█████████▊| 390806/400000 [00:45<00:01, 8517.12it/s] 98%|█████████▊| 391659/400000 [00:45<00:00, 8496.26it/s] 98%|█████████▊| 392531/400000 [00:45<00:00, 8561.13it/s] 98%|█████████▊| 393388/400000 [00:45<00:00, 8528.83it/s] 99%|█████████▊| 394242/400000 [00:45<00:00, 8487.11it/s] 99%|█████████▉| 395112/400000 [00:46<00:00, 8547.35it/s] 99%|█████████▉| 395970/400000 [00:46<00:00, 8555.86it/s] 99%|█████████▉| 396826/400000 [00:46<00:00, 8278.92it/s] 99%|█████████▉| 397657/400000 [00:46<00:00, 8001.09it/s]100%|█████████▉| 398461/400000 [00:46<00:00, 6981.88it/s]100%|█████████▉| 399185/400000 [00:46<00:00, 6829.50it/s]100%|█████████▉| 399999/400000 [00:46<00:00, 8556.03it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f40930edcc0>, <torchtext.data.dataset.TabularDataset object at 0x7f40930ede10>, <torchtext.vocab.Vocab object at 0x7f40930edd30>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  6.90 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  6.90 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  6.90 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  8.27 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  8.27 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.84 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.84 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.85 file/s]2020-07-19 06:18:09.341430: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-19 06:18:09.347972: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095190000 Hz
2020-07-19 06:18:09.348254: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x555d58ad1260 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-19 06:18:09.348307: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:06, 148862.35it/s] 82%|████████▏ | 8151040/9912422 [00:00<00:08, 212478.20it/s]9920512it [00:00, 43821904.36it/s]                           
0it [00:00, ?it/s]32768it [00:00, 598726.88it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 462449.12it/s]1654784it [00:00, 11544631.60it/s]                         
0it [00:00, ?it/s]8192it [00:00, 203327.70it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
