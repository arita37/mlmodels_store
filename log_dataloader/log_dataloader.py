
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '6ca6da91408244e26c157e9e6467cc18ede43e71', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/6ca6da91408244e26c157e9e6467cc18ede43e71

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f3ee3cb0510> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f3ee3cb0510> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f3f4e95d400> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f3f4e95d400> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f3f6db79ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f3f6db79ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f3efbc9a0d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f3efbc9a0d0> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f3efbc9a0d0> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 29%|██▉       | 2850816/9912422 [00:00<00:00, 28277212.46it/s]9920512it [00:00, 32258888.26it/s]                             
0it [00:00, ?it/s]32768it [00:00, 479571.49it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 162663.89it/s]1654784it [00:00, 11657972.43it/s]                         
0it [00:00, ?it/s]8192it [00:00, 203469.78it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f3ef9e66550>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f3efa172908>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f3efbc91d08> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f3efbc91d08> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f3efbc91d08> , (data_info, **args) 

  CIFAR10 

  Dataset Name is :  cifar10 

Dl Completed...: 0 url [00:00, ? url/s]
Dl Size...: 0 MiB [00:00, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...: 0 MiB [00:00, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/urllib3/connectionpool.py:988: InsecureRequestWarning: Unverified HTTPS request is being made to host 'www.cs.toronto.edu'. Adding certificate verification is strongly advised. See: https://urllib3.readthedocs.io/en/latest/advanced-usage.html#ssl-warnings
  InsecureRequestWarning,
Dl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   0%|          | 0/162 [00:00<?, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   1%|          | 1/162 [00:00<01:47,  1.50 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:47,  1.50 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:46,  1.50 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:46,  1.50 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:45,  1.50 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:44,  1.50 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:44,  1.50 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:43,  1.50 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<01:12,  2.12 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<01:12,  2.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<01:12,  2.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<01:11,  2.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<01:11,  2.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<01:10,  2.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<01:10,  2.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<01:09,  2.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<01:09,  2.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<01:08,  2.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<01:08,  2.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  11%|█         | 18/162 [00:00<00:48,  2.99 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:48,  2.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:47,  2.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:47,  2.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:47,  2.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:46,  2.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:46,  2.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:46,  2.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:45,  2.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  16%|█▌        | 26/162 [00:00<00:32,  4.20 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:32,  4.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:01<00:32,  4.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:01<00:31,  4.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:01<00:31,  4.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:01<00:31,  4.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:01<00:31,  4.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:01<00:30,  4.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:01<00:30,  4.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:30,  4.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  22%|██▏       | 35/162 [00:01<00:21,  5.87 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:21,  5.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:21,  5.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:21,  5.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:21,  5.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:20,  5.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:20,  5.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:20,  5.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:20,  5.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  27%|██▋       | 43/162 [00:01<00:14,  8.10 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:14,  8.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:14,  8.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:14,  8.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:14,  8.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:14,  8.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:14,  8.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:13,  8.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:13,  8.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  31%|███▏      | 51/162 [00:01<00:10, 11.07 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:10, 11.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:09, 11.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:09, 11.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:09, 11.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:09, 11.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:09, 11.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:09, 11.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:09, 11.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  36%|███▋      | 59/162 [00:01<00:06, 14.85 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:06, 14.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:06, 14.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:06, 14.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:06, 14.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:06, 14.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:06, 14.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:06, 14.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:06, 14.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  41%|████▏     | 67/162 [00:01<00:04, 19.51 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:04, 19.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:04, 19.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:04, 19.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:04, 19.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:04, 19.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:04, 19.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:04, 19.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:04, 19.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  46%|████▋     | 75/162 [00:01<00:03, 25.15 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:03, 25.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:03, 25.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:03, 25.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:03, 25.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:03, 25.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:03, 25.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:03, 25.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:03, 25.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 31.56 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 31.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 31.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 31.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 31.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 31.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 31.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:02, 31.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:02, 31.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 38.36 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 38.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 38.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 38.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 38.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 38.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 38.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 38.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 38.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 38.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 45.54 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 45.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 45.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 45.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:02<00:01, 45.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:02<00:01, 45.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:02<00:01, 45.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:02<00:01, 45.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:02<00:01, 45.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  67%|██████▋   | 108/162 [00:02<00:01, 52.23 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:02<00:01, 52.23 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:02<00:01, 52.23 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:02<00:00, 52.23 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:00, 52.23 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:00, 52.23 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:00, 52.23 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 52.23 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 52.23 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 52.23 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 58.48 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 58.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 58.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 58.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 58.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 58.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 58.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 58.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 58.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 57.40 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 57.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 57.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 57.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 57.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 57.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 57.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 57.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 57.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 57.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 62.95 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 62.95 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 62.95 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 62.95 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 62.95 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 62.95 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 62.95 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 62.95 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 62.95 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 62.95 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 67.91 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 67.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 67.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 67.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 67.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 67.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 67.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 67.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 67.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 67.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 71.20 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 71.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 71.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 71.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 71.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 71.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 71.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 71.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 71.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 71.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 73.29 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 73.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 73.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.78s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.78s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 73.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.78s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 73.29 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.83s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.78s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 73.29 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.83s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.83s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 33.55 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.83s/ url]
0 examples [00:00, ? examples/s]2020-08-14 06:06:36.756053: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-08-14 06:06:36.768091: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095080000 Hz
2020-08-14 06:06:36.768295: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55ec896a5790 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-14 06:06:36.768317: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  2.79 examples/s]110 examples [00:00,  3.98 examples/s]221 examples [00:00,  5.68 examples/s]329 examples [00:00,  8.09 examples/s]439 examples [00:00, 11.53 examples/s]553 examples [00:00, 16.40 examples/s]665 examples [00:00, 23.28 examples/s]769 examples [00:01, 32.93 examples/s]882 examples [00:01, 46.47 examples/s]997 examples [00:01, 65.25 examples/s]1112 examples [00:01, 90.99 examples/s]1222 examples [00:01, 125.50 examples/s]1332 examples [00:01, 170.88 examples/s]1445 examples [00:01, 229.15 examples/s]1556 examples [00:01, 299.23 examples/s]1665 examples [00:01, 380.47 examples/s]1773 examples [00:01, 467.36 examples/s]1878 examples [00:02, 555.06 examples/s]1989 examples [00:02, 651.87 examples/s]2101 examples [00:02, 744.81 examples/s]2216 examples [00:02, 831.95 examples/s]2327 examples [00:02, 897.89 examples/s]2441 examples [00:02, 958.14 examples/s]2558 examples [00:02, 1011.85 examples/s]2672 examples [00:02, 1046.47 examples/s]2785 examples [00:02, 1054.30 examples/s]2897 examples [00:03, 1036.96 examples/s]3005 examples [00:03, 1049.49 examples/s]3117 examples [00:03, 1068.85 examples/s]3226 examples [00:03, 1071.77 examples/s]3345 examples [00:03, 1102.80 examples/s]3458 examples [00:03, 1110.45 examples/s]3570 examples [00:03, 1108.15 examples/s]3684 examples [00:03, 1117.20 examples/s]3798 examples [00:03, 1122.36 examples/s]3913 examples [00:03, 1129.53 examples/s]4027 examples [00:04, 1126.25 examples/s]4140 examples [00:04, 1123.09 examples/s]4253 examples [00:04, 1121.32 examples/s]4366 examples [00:04, 1115.38 examples/s]4478 examples [00:04, 1103.32 examples/s]4589 examples [00:04, 1089.73 examples/s]4699 examples [00:04, 1079.61 examples/s]4808 examples [00:04, 1080.09 examples/s]4917 examples [00:04, 1079.16 examples/s]5031 examples [00:04, 1096.31 examples/s]5142 examples [00:05, 1099.62 examples/s]5257 examples [00:05, 1113.29 examples/s]5369 examples [00:05, 1106.79 examples/s]5482 examples [00:05, 1113.48 examples/s]5594 examples [00:05, 1112.58 examples/s]5706 examples [00:05, 1099.48 examples/s]5817 examples [00:05, 1072.67 examples/s]5926 examples [00:05, 1075.63 examples/s]6039 examples [00:05, 1090.59 examples/s]6149 examples [00:05, 1081.99 examples/s]6258 examples [00:06, 1080.96 examples/s]6368 examples [00:06, 1084.89 examples/s]6479 examples [00:06, 1089.65 examples/s]6592 examples [00:06, 1099.67 examples/s]6703 examples [00:06, 1099.63 examples/s]6814 examples [00:06, 1099.83 examples/s]6925 examples [00:06, 1093.57 examples/s]7037 examples [00:06, 1101.04 examples/s]7148 examples [00:06, 1058.88 examples/s]7259 examples [00:06, 1073.21 examples/s]7367 examples [00:07, 1031.58 examples/s]7471 examples [00:07, 1021.83 examples/s]7584 examples [00:07, 1050.80 examples/s]7693 examples [00:07, 1060.26 examples/s]7803 examples [00:07, 1071.12 examples/s]7911 examples [00:07, 1069.70 examples/s]8019 examples [00:07, 1036.27 examples/s]8123 examples [00:07, 1003.75 examples/s]8227 examples [00:07, 1014.04 examples/s]8329 examples [00:08, 1005.21 examples/s]8433 examples [00:08, 1015.36 examples/s]8535 examples [00:08, 1008.56 examples/s]8644 examples [00:08, 1029.61 examples/s]8756 examples [00:08, 1054.95 examples/s]8865 examples [00:08, 1063.29 examples/s]8976 examples [00:08, 1075.16 examples/s]9084 examples [00:08, 1071.91 examples/s]9193 examples [00:08, 1076.11 examples/s]9305 examples [00:08, 1086.66 examples/s]9414 examples [00:09, 1081.94 examples/s]9523 examples [00:09, 1081.88 examples/s]9634 examples [00:09, 1088.43 examples/s]9751 examples [00:09, 1109.15 examples/s]9865 examples [00:09, 1116.61 examples/s]9980 examples [00:09, 1125.58 examples/s]10093 examples [00:09, 1051.09 examples/s]10204 examples [00:09, 1067.61 examples/s]10313 examples [00:09, 1071.58 examples/s]10422 examples [00:09, 1076.77 examples/s]10531 examples [00:10, 1055.21 examples/s]10642 examples [00:10, 1069.33 examples/s]10752 examples [00:10, 1076.04 examples/s]10865 examples [00:10, 1091.66 examples/s]10979 examples [00:10, 1103.64 examples/s]11090 examples [00:10, 1099.34 examples/s]11201 examples [00:10, 1075.15 examples/s]11309 examples [00:10, 1065.11 examples/s]11416 examples [00:10, 1049.37 examples/s]11528 examples [00:10, 1069.35 examples/s]11636 examples [00:11, 1061.04 examples/s]11743 examples [00:11, 1058.62 examples/s]11857 examples [00:11, 1081.21 examples/s]11969 examples [00:11, 1091.08 examples/s]12081 examples [00:11, 1098.90 examples/s]12192 examples [00:11, 1094.93 examples/s]12304 examples [00:11, 1100.77 examples/s]12415 examples [00:11, 1102.04 examples/s]12527 examples [00:11, 1107.00 examples/s]12641 examples [00:12, 1115.78 examples/s]12753 examples [00:12, 1116.68 examples/s]12865 examples [00:12, 1074.40 examples/s]12979 examples [00:12, 1091.22 examples/s]13089 examples [00:12, 1092.41 examples/s]13203 examples [00:12, 1104.72 examples/s]13317 examples [00:12, 1112.84 examples/s]13429 examples [00:12, 1111.83 examples/s]13543 examples [00:12, 1118.52 examples/s]13655 examples [00:12, 1118.94 examples/s]13768 examples [00:13, 1120.60 examples/s]13881 examples [00:13, 1107.01 examples/s]13994 examples [00:13, 1110.95 examples/s]14106 examples [00:13, 1111.05 examples/s]14218 examples [00:13, 1108.68 examples/s]14329 examples [00:13, 1107.13 examples/s]14442 examples [00:13, 1111.73 examples/s]14554 examples [00:13, 1102.91 examples/s]14667 examples [00:13, 1110.05 examples/s]14779 examples [00:13, 1111.25 examples/s]14891 examples [00:14, 1112.87 examples/s]15003 examples [00:14, 1090.81 examples/s]15113 examples [00:14, 1092.04 examples/s]15227 examples [00:14, 1105.80 examples/s]15341 examples [00:14, 1112.99 examples/s]15453 examples [00:14, 1113.95 examples/s]15565 examples [00:14, 1109.99 examples/s]15677 examples [00:14, 1101.40 examples/s]15788 examples [00:14, 1094.38 examples/s]15898 examples [00:14, 1090.68 examples/s]16008 examples [00:15, 1091.39 examples/s]16118 examples [00:15, 1077.35 examples/s]16230 examples [00:15, 1087.64 examples/s]16342 examples [00:15, 1095.01 examples/s]16452 examples [00:15, 1091.05 examples/s]16565 examples [00:15, 1101.70 examples/s]16678 examples [00:15, 1107.30 examples/s]16789 examples [00:15, 1106.46 examples/s]16902 examples [00:15, 1112.11 examples/s]17017 examples [00:15, 1122.25 examples/s]17132 examples [00:16, 1129.98 examples/s]17246 examples [00:16, 1130.40 examples/s]17360 examples [00:16, 1124.70 examples/s]17473 examples [00:16, 1124.22 examples/s]17586 examples [00:16, 1125.87 examples/s]17701 examples [00:16, 1130.36 examples/s]17815 examples [00:16, 1130.96 examples/s]17929 examples [00:16, 1116.48 examples/s]18041 examples [00:16, 1115.31 examples/s]18156 examples [00:16, 1122.73 examples/s]18271 examples [00:17, 1130.67 examples/s]18385 examples [00:17, 1071.12 examples/s]18498 examples [00:17, 1085.41 examples/s]18613 examples [00:17, 1101.98 examples/s]18727 examples [00:17, 1111.04 examples/s]18841 examples [00:17, 1117.89 examples/s]18956 examples [00:17, 1124.35 examples/s]19069 examples [00:17, 1119.38 examples/s]19183 examples [00:17, 1122.60 examples/s]19297 examples [00:18, 1125.59 examples/s]19411 examples [00:18, 1129.01 examples/s]19527 examples [00:18, 1137.73 examples/s]19641 examples [00:18, 1122.80 examples/s]19754 examples [00:18, 1120.27 examples/s]19869 examples [00:18, 1128.11 examples/s]19987 examples [00:18, 1142.12 examples/s]20102 examples [00:18, 1100.53 examples/s]20217 examples [00:18, 1114.39 examples/s]20329 examples [00:18, 1116.01 examples/s]20444 examples [00:19, 1123.21 examples/s]20557 examples [00:19, 1110.10 examples/s]20670 examples [00:19, 1115.88 examples/s]20782 examples [00:19, 1115.87 examples/s]20895 examples [00:19, 1118.85 examples/s]21010 examples [00:19, 1125.39 examples/s]21124 examples [00:19, 1129.08 examples/s]21241 examples [00:19, 1139.32 examples/s]21355 examples [00:19, 1133.07 examples/s]21470 examples [00:19, 1137.40 examples/s]21587 examples [00:20, 1146.07 examples/s]21704 examples [00:20, 1150.80 examples/s]21820 examples [00:20, 1130.60 examples/s]21934 examples [00:20, 1124.59 examples/s]22049 examples [00:20, 1129.99 examples/s]22165 examples [00:20, 1135.98 examples/s]22279 examples [00:20, 1132.45 examples/s]22395 examples [00:20, 1138.69 examples/s]22509 examples [00:20, 1128.98 examples/s]22626 examples [00:20, 1138.31 examples/s]22741 examples [00:21, 1140.31 examples/s]22856 examples [00:21, 1124.55 examples/s]22969 examples [00:21, 1125.29 examples/s]23082 examples [00:21, 1121.04 examples/s]23198 examples [00:21, 1131.23 examples/s]23320 examples [00:21, 1154.82 examples/s]23437 examples [00:21, 1157.05 examples/s]23556 examples [00:21, 1165.54 examples/s]23673 examples [00:21, 1150.59 examples/s]23789 examples [00:21, 1138.30 examples/s]23903 examples [00:22, 1134.22 examples/s]24019 examples [00:22, 1139.03 examples/s]24134 examples [00:22, 1140.36 examples/s]24249 examples [00:22, 1132.71 examples/s]24363 examples [00:22, 1128.62 examples/s]24477 examples [00:22, 1130.45 examples/s]24591 examples [00:22, 1126.38 examples/s]24708 examples [00:22, 1136.27 examples/s]24822 examples [00:22, 1137.24 examples/s]24936 examples [00:22, 1138.01 examples/s]25052 examples [00:23, 1143.42 examples/s]25167 examples [00:23, 1140.39 examples/s]25282 examples [00:23, 1128.80 examples/s]25395 examples [00:23, 1126.77 examples/s]25508 examples [00:23, 1126.44 examples/s]25623 examples [00:23, 1132.52 examples/s]25739 examples [00:23, 1139.43 examples/s]25853 examples [00:23, 1117.56 examples/s]25965 examples [00:23, 1113.69 examples/s]26077 examples [00:24, 1109.90 examples/s]26189 examples [00:24, 1107.37 examples/s]26300 examples [00:24, 1107.85 examples/s]26411 examples [00:24, 1107.24 examples/s]26522 examples [00:24, 1105.66 examples/s]26633 examples [00:24, 1105.66 examples/s]26745 examples [00:24, 1107.67 examples/s]26856 examples [00:24, 1106.48 examples/s]26967 examples [00:24, 1107.01 examples/s]27078 examples [00:24, 1096.39 examples/s]27189 examples [00:25, 1098.67 examples/s]27301 examples [00:25, 1103.78 examples/s]27412 examples [00:25, 1091.57 examples/s]27524 examples [00:25, 1098.77 examples/s]27634 examples [00:25, 1094.26 examples/s]27745 examples [00:25, 1098.26 examples/s]27856 examples [00:25, 1099.42 examples/s]27968 examples [00:25, 1105.08 examples/s]28079 examples [00:25, 1103.47 examples/s]28190 examples [00:25, 1097.36 examples/s]28302 examples [00:26, 1102.07 examples/s]28413 examples [00:26, 1102.95 examples/s]28524 examples [00:26, 1097.51 examples/s]28634 examples [00:26, 1086.72 examples/s]28743 examples [00:26, 1073.78 examples/s]28853 examples [00:26, 1078.77 examples/s]28964 examples [00:26, 1087.47 examples/s]29076 examples [00:26, 1095.80 examples/s]29189 examples [00:26, 1104.39 examples/s]29300 examples [00:26, 1103.16 examples/s]29413 examples [00:27, 1108.43 examples/s]29524 examples [00:27, 1103.21 examples/s]29635 examples [00:27, 1104.35 examples/s]29747 examples [00:27, 1108.66 examples/s]29858 examples [00:27, 1103.95 examples/s]29969 examples [00:27, 1103.33 examples/s]30080 examples [00:27, 1053.03 examples/s]30192 examples [00:27, 1071.08 examples/s]30304 examples [00:27, 1083.78 examples/s]30413 examples [00:27, 1078.06 examples/s]30525 examples [00:28, 1088.56 examples/s]30638 examples [00:28, 1098.40 examples/s]30751 examples [00:28, 1106.27 examples/s]30863 examples [00:28, 1109.85 examples/s]30975 examples [00:28, 1109.69 examples/s]31087 examples [00:28, 1112.60 examples/s]31199 examples [00:28, 1112.95 examples/s]31311 examples [00:28, 1110.84 examples/s]31423 examples [00:28, 1112.31 examples/s]31535 examples [00:28, 1102.47 examples/s]31646 examples [00:29, 1102.96 examples/s]31757 examples [00:29, 1087.21 examples/s]31866 examples [00:29, 1084.49 examples/s]31977 examples [00:29, 1082.52 examples/s]32086 examples [00:29, 1071.18 examples/s]32196 examples [00:29, 1079.59 examples/s]32308 examples [00:29, 1090.02 examples/s]32419 examples [00:29, 1094.17 examples/s]32530 examples [00:29, 1098.58 examples/s]32640 examples [00:29, 1087.14 examples/s]32749 examples [00:30, 1085.97 examples/s]32861 examples [00:30, 1093.56 examples/s]32972 examples [00:30, 1097.40 examples/s]33083 examples [00:30, 1100.81 examples/s]33194 examples [00:30, 1096.25 examples/s]33306 examples [00:30, 1102.33 examples/s]33417 examples [00:30, 1087.69 examples/s]33526 examples [00:30, 1087.02 examples/s]33636 examples [00:30, 1088.56 examples/s]33745 examples [00:31, 1082.69 examples/s]33854 examples [00:31, 1076.98 examples/s]33962 examples [00:31, 1052.81 examples/s]34074 examples [00:31, 1071.71 examples/s]34185 examples [00:31, 1082.57 examples/s]34297 examples [00:31, 1091.25 examples/s]34411 examples [00:31, 1103.99 examples/s]34522 examples [00:31, 1102.06 examples/s]34635 examples [00:31, 1108.18 examples/s]34746 examples [00:31, 1068.88 examples/s]34855 examples [00:32, 1073.69 examples/s]34966 examples [00:32, 1053.76 examples/s]35077 examples [00:32, 1065.38 examples/s]35185 examples [00:32, 1068.00 examples/s]35299 examples [00:32, 1086.19 examples/s]35408 examples [00:32, 1086.51 examples/s]35520 examples [00:32, 1095.63 examples/s]35632 examples [00:32, 1101.64 examples/s]35744 examples [00:32, 1105.14 examples/s]35855 examples [00:32, 1098.91 examples/s]35966 examples [00:33, 1101.64 examples/s]36077 examples [00:33, 1086.88 examples/s]36186 examples [00:33, 1081.30 examples/s]36301 examples [00:33, 1098.06 examples/s]36413 examples [00:33, 1102.69 examples/s]36524 examples [00:33, 1099.99 examples/s]36635 examples [00:33, 1095.55 examples/s]36747 examples [00:33, 1100.19 examples/s]36859 examples [00:33, 1105.69 examples/s]36971 examples [00:33, 1107.56 examples/s]37082 examples [00:34, 1088.79 examples/s]37191 examples [00:34, 1079.12 examples/s]37303 examples [00:34, 1089.13 examples/s]37412 examples [00:34, 1076.96 examples/s]37522 examples [00:34, 1081.16 examples/s]37631 examples [00:34, 1076.44 examples/s]37739 examples [00:34, 1074.74 examples/s]37847 examples [00:34, 1069.16 examples/s]37954 examples [00:34, 1055.69 examples/s]38064 examples [00:34, 1067.46 examples/s]38172 examples [00:35, 1071.16 examples/s]38280 examples [00:35, 1057.56 examples/s]38389 examples [00:35, 1065.50 examples/s]38500 examples [00:35, 1076.88 examples/s]38610 examples [00:35, 1083.22 examples/s]38719 examples [00:35, 1078.37 examples/s]38830 examples [00:35, 1085.75 examples/s]38940 examples [00:35, 1089.71 examples/s]39050 examples [00:35, 1090.22 examples/s]39161 examples [00:35, 1094.37 examples/s]39271 examples [00:36, 1092.27 examples/s]39381 examples [00:36, 1073.76 examples/s]39489 examples [00:36, 1067.57 examples/s]39596 examples [00:36, 1059.83 examples/s]39705 examples [00:36, 1068.65 examples/s]39812 examples [00:36, 1062.32 examples/s]39922 examples [00:36, 1071.85 examples/s]40030 examples [00:36, 1024.41 examples/s]40133 examples [00:36, 1022.16 examples/s]40242 examples [00:37, 1040.01 examples/s]40350 examples [00:37, 1049.07 examples/s]40456 examples [00:37, 1042.82 examples/s]40567 examples [00:37, 1059.60 examples/s]40675 examples [00:37, 1064.12 examples/s]40785 examples [00:37, 1072.63 examples/s]40896 examples [00:37, 1081.25 examples/s]41007 examples [00:37, 1088.27 examples/s]41118 examples [00:37, 1093.03 examples/s]41230 examples [00:37, 1098.05 examples/s]41343 examples [00:38, 1107.43 examples/s]41456 examples [00:38, 1111.25 examples/s]41568 examples [00:38, 1101.38 examples/s]41679 examples [00:38, 1088.13 examples/s]41793 examples [00:38, 1100.57 examples/s]41906 examples [00:38, 1108.15 examples/s]42017 examples [00:38, 1107.70 examples/s]42128 examples [00:38, 1107.66 examples/s]42243 examples [00:38, 1118.54 examples/s]42359 examples [00:38, 1128.32 examples/s]42474 examples [00:39, 1131.90 examples/s]42588 examples [00:39, 1119.06 examples/s]42700 examples [00:39, 1102.35 examples/s]42811 examples [00:39, 1097.47 examples/s]42922 examples [00:39, 1099.40 examples/s]43034 examples [00:39, 1102.73 examples/s]43145 examples [00:39, 1098.88 examples/s]43256 examples [00:39, 1101.59 examples/s]43369 examples [00:39, 1108.76 examples/s]43484 examples [00:39, 1117.88 examples/s]43599 examples [00:40, 1127.22 examples/s]43712 examples [00:40, 1113.01 examples/s]43825 examples [00:40, 1116.87 examples/s]43938 examples [00:40, 1120.68 examples/s]44051 examples [00:40, 1123.06 examples/s]44164 examples [00:40, 1118.41 examples/s]44276 examples [00:40, 1113.03 examples/s]44388 examples [00:40, 1114.01 examples/s]44500 examples [00:40, 1114.23 examples/s]44614 examples [00:40, 1120.70 examples/s]44727 examples [00:41, 1113.36 examples/s]44839 examples [00:41, 1097.15 examples/s]44952 examples [00:41, 1106.45 examples/s]45068 examples [00:41, 1120.48 examples/s]45181 examples [00:41, 1121.81 examples/s]45296 examples [00:41, 1127.55 examples/s]45409 examples [00:41, 1125.75 examples/s]45522 examples [00:41, 1120.11 examples/s]45635 examples [00:41, 1112.13 examples/s]45747 examples [00:41, 1098.08 examples/s]45857 examples [00:42, 1079.12 examples/s]45967 examples [00:42, 1082.84 examples/s]46077 examples [00:42, 1087.04 examples/s]46189 examples [00:42, 1096.04 examples/s]46302 examples [00:42, 1105.70 examples/s]46413 examples [00:42, 1105.82 examples/s]46525 examples [00:42, 1108.09 examples/s]46636 examples [00:42, 1108.24 examples/s]46749 examples [00:42, 1113.32 examples/s]46861 examples [00:43, 1113.44 examples/s]46973 examples [00:43, 1110.18 examples/s]47085 examples [00:43, 1075.38 examples/s]47194 examples [00:43, 1077.50 examples/s]47306 examples [00:43, 1088.69 examples/s]47418 examples [00:43, 1096.01 examples/s]47528 examples [00:43, 1096.84 examples/s]47642 examples [00:43, 1109.24 examples/s]47755 examples [00:43, 1114.76 examples/s]47869 examples [00:43, 1120.65 examples/s]47983 examples [00:44, 1124.77 examples/s]48096 examples [00:44, 1113.88 examples/s]48209 examples [00:44, 1118.44 examples/s]48321 examples [00:44, 1095.15 examples/s]48432 examples [00:44, 1099.34 examples/s]48546 examples [00:44, 1110.76 examples/s]48659 examples [00:44, 1115.54 examples/s]48771 examples [00:44, 1112.14 examples/s]48883 examples [00:44, 1106.59 examples/s]48994 examples [00:44, 1099.96 examples/s]49105 examples [00:45, 1097.63 examples/s]49215 examples [00:45, 1093.78 examples/s]49325 examples [00:45, 1035.70 examples/s]49435 examples [00:45, 1051.41 examples/s]49545 examples [00:45, 1064.58 examples/s]49655 examples [00:45, 1073.73 examples/s]49763 examples [00:45, 1070.99 examples/s]49871 examples [00:45, 1072.46 examples/s]49979 examples [00:45, 1066.76 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 13%|█▎        | 6287/50000 [00:00<00:00, 62866.01 examples/s] 39%|███▊      | 19366/50000 [00:00<00:00, 74467.33 examples/s] 65%|██████▍   | 32263/50000 [00:00<00:00, 85278.60 examples/s] 91%|█████████ | 45316/50000 [00:00<00:00, 95175.34 examples/s]                                                               0 examples [00:00, ? examples/s]93 examples [00:00, 923.72 examples/s]206 examples [00:00, 975.84 examples/s]319 examples [00:00, 1016.62 examples/s]432 examples [00:00, 1046.63 examples/s]544 examples [00:00, 1067.35 examples/s]655 examples [00:00, 1077.30 examples/s]768 examples [00:00, 1089.98 examples/s]882 examples [00:00, 1102.63 examples/s]991 examples [00:00, 1096.77 examples/s]1103 examples [00:01, 1101.35 examples/s]1219 examples [00:01, 1118.01 examples/s]1335 examples [00:01, 1127.40 examples/s]1450 examples [00:01, 1132.96 examples/s]1563 examples [00:01, 1129.57 examples/s]1676 examples [00:01, 1126.51 examples/s]1789 examples [00:01, 1103.77 examples/s]1905 examples [00:01, 1118.75 examples/s]2020 examples [00:01, 1125.50 examples/s]2136 examples [00:01, 1134.28 examples/s]2250 examples [00:02, 1128.00 examples/s]2367 examples [00:02, 1138.57 examples/s]2482 examples [00:02, 1140.73 examples/s]2597 examples [00:02, 1143.35 examples/s]2712 examples [00:02, 1137.83 examples/s]2826 examples [00:02, 1125.87 examples/s]2939 examples [00:02, 1124.70 examples/s]3052 examples [00:02, 1122.07 examples/s]3165 examples [00:02, 1122.61 examples/s]3278 examples [00:02, 1088.13 examples/s]3392 examples [00:03, 1102.76 examples/s]3510 examples [00:03, 1124.43 examples/s]3628 examples [00:03, 1140.51 examples/s]3745 examples [00:03, 1147.03 examples/s]3862 examples [00:03, 1151.26 examples/s]3978 examples [00:03, 1141.91 examples/s]4094 examples [00:03, 1147.13 examples/s]4209 examples [00:03, 1140.56 examples/s]4324 examples [00:03, 1136.15 examples/s]4439 examples [00:03, 1138.12 examples/s]4553 examples [00:04, 1111.90 examples/s]4674 examples [00:04, 1138.99 examples/s]4790 examples [00:04, 1143.91 examples/s]4908 examples [00:04, 1154.22 examples/s]5026 examples [00:04, 1160.51 examples/s]5143 examples [00:04, 1145.66 examples/s]5260 examples [00:04, 1152.75 examples/s]5379 examples [00:04, 1161.43 examples/s]5496 examples [00:04, 1163.41 examples/s]5613 examples [00:04, 1160.87 examples/s]5730 examples [00:05, 1151.75 examples/s]5846 examples [00:05, 1136.29 examples/s]5964 examples [00:05, 1146.97 examples/s]6083 examples [00:05, 1159.00 examples/s]6201 examples [00:05, 1165.21 examples/s]6318 examples [00:05, 1147.26 examples/s]6433 examples [00:05, 1135.28 examples/s]6547 examples [00:05, 1136.18 examples/s]6661 examples [00:05, 1131.23 examples/s]6775 examples [00:05, 1123.37 examples/s]6891 examples [00:06, 1131.83 examples/s]7005 examples [00:06, 1126.85 examples/s]7119 examples [00:06, 1128.84 examples/s]7235 examples [00:06, 1137.16 examples/s]7352 examples [00:06, 1146.10 examples/s]7467 examples [00:06, 1141.37 examples/s]7582 examples [00:06, 1131.88 examples/s]7699 examples [00:06, 1142.22 examples/s]7814 examples [00:06, 1121.23 examples/s]7927 examples [00:07, 1119.54 examples/s]8040 examples [00:07, 1109.95 examples/s]8152 examples [00:07, 1108.17 examples/s]8264 examples [00:07, 1109.22 examples/s]8378 examples [00:07, 1116.24 examples/s]8490 examples [00:07, 1110.95 examples/s]8602 examples [00:07, 1102.68 examples/s]8713 examples [00:07, 1099.40 examples/s]8824 examples [00:07, 1101.68 examples/s]8935 examples [00:07, 1101.21 examples/s]9046 examples [00:08, 1102.97 examples/s]9157 examples [00:08, 1092.81 examples/s]9268 examples [00:08, 1095.91 examples/s]9378 examples [00:08, 1092.67 examples/s]9488 examples [00:08, 1076.67 examples/s]9597 examples [00:08, 1078.45 examples/s]9705 examples [00:08, 1059.89 examples/s]9815 examples [00:08, 1070.66 examples/s]9925 examples [00:08, 1077.93 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteUPW2LJ/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteUPW2LJ/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f3efbc9a0d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f3efbc9a0d0> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f3efbc9a0d0> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f3f479e95c0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f3ee736ce80>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f3efbc9a0d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f3efbc9a0d0> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f3efbc9a0d0> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f3efa172da0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f3efa172ef0>), {}) 

  




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
Error /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor/textcnn.json Module ['mlmodels.model_tch.textcnn', 'split_train_valid'] notfound, libtorch_cpu.so: cannot open shared object file: No such file or directory, tuple index out of range

  




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
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/textcnn.py", line 24, in <module>
    import torchtext
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/torchtext/__init__.py", line 42, in <module>
    _init_extension()
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/torchtext/__init__.py", line 38, in _init_extension
    torch.ops.load_library(ext_specs.origin)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/torch/_ops.py", line 106, in load_library
    ctypes.CDLL(path)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/ctypes/__init__.py", line 348, in __init__
    self._handle = _dlopen(self._name, mode)
OSError: libtorch_cpu.so: cannot open shared object file: No such file or directory

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/util.py", line 683, in load_function_uri
    package_name = str(Path(package).parts[-2]) + "." + str(model_name)
IndexError: tuple index out of range

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 452, in test_json_list
    loader.compute()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 248, in compute
    preprocessor_func = load_function(uri)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/util.py", line 688, in load_function_uri
    raise NameError(f"Module {pkg} notfound, {e1}, {e2}")
NameError: Module ['mlmodels.model_tch.textcnn', 'split_train_valid'] notfound, libtorch_cpu.so: cannot open shared object file: No such file or directory, tuple index out of range
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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 11.83 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 15.70 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 15.70 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  7.93 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  7.93 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.44 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.44 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.56 file/s]2020-08-14 06:07:39.077907: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-08-14 06:07:39.081980: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095080000 Hz
2020-08-14 06:07:39.082147: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x564e6f1c14a0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-14 06:07:39.082163: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
[1mDownloading and preparing dataset mnist/3.0.1 (download: 11.06 MiB, generated: 21.00 MiB, total: 32.06 MiB) to /home/runner/tensorflow_datasets/mnist/3.0.1...[0m

[1mDataset mnist downloaded and prepared to /home/runner/tensorflow_datasets/mnist/3.0.1. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/ ['mnist2', 'train', 'test', 'cifar10', 'fashion-mnist_small.npy', 'mnist_dataset_small.npy'] 

  


 #################### get_dataset_torch 

  get_dataset_torch mlmodels/preprocess/generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:02, 159001.09it/s] 77%|███████▋  | 7634944/9912422 [00:00<00:10, 226935.73it/s]9920512it [00:00, 43832060.11it/s]                           
0it [00:00, ?it/s]32768it [00:00, 484923.48it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:11, 137538.53it/s]1654784it [00:00, 10363907.13it/s]                         
0it [00:00, ?it/s]8192it [00:00, 190490.58it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
