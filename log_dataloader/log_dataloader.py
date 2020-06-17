
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/db74a4d005299c75e83c5708d4c48e255c70b17e', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/adata2/', 'repo': 'arita37/mlmodels', 'branch': 'adata2', 'sha': 'db74a4d005299c75e83c5708d4c48e255c70b17e', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/adata2/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/db74a4d005299c75e83c5708d4c48e255c70b17e

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/db74a4d005299c75e83c5708d4c48e255c70b17e

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/db74a4d005299c75e83c5708d4c48e255c70b17e

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f5d29089f28> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f5d29089f28> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f5d946410d0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f5d946410d0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f5dae36dea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f5dae36dea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f5d41ebc950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f5d41ebc950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f5d41ebc950> , (data_info, **args) 

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
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 685, in parser_f
    return _read(filepath_or_buffer, kwds)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 457, in _read
    parser = TextFileReader(fp_or_buf, **kwds)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 895, in __init__
    self._make_engine(self.engine)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1135, in _make_engine
    self._engine = CParserWrapper(self.f, **self.options)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1917, in __init__
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
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 685, in parser_f
    return _read(filepath_or_buffer, kwds)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 457, in _read
    parser = TextFileReader(fp_or_buf, **kwds)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 895, in __init__
    self._make_engine(self.engine)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1135, in _make_engine
    self._engine = CParserWrapper(self.f, **self.options)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1917, in __init__
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
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/numpy/lib/npyio.py", line 428, in load
    fid = open(os_fspath(file), "rb")
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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 35%|███▍      | 3457024/9912422 [00:00<00:00, 34438489.89it/s]9920512it [00:00, 34579551.24it/s]                             
0it [00:00, ?it/s]32768it [00:00, 511450.24it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 153888.39it/s]1654784it [00:00, 10974150.22it/s]                         
0it [00:00, ?it/s]8192it [00:00, 185656.06it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5d3f50a748>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5d3f5039e8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f5d41ebc598> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f5d41ebc598> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f5d41ebc598> , (data_info, **args) 

  CIFAR10 

  Dataset Name is :  cifar10 

Dl Completed...: 0 url [00:00, ? url/s]
Dl Size...: 0 MiB [00:00, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...: 0 MiB [00:00, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/urllib3/connectionpool.py:986: InsecureRequestWarning: Unverified HTTPS request is being made to host 'www.cs.toronto.edu'. Adding certificate verification is strongly advised. See: https://urllib3.readthedocs.io/en/latest/advanced-usage.html#ssl-warnings
  InsecureRequestWarning,
Dl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   0%|          | 0/162 [00:00<?, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   1%|          | 1/162 [00:00<01:22,  1.96 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:22,  1.96 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:21,  1.96 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:21,  1.96 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:20,  1.96 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:20,  1.96 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:19,  1.96 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:19,  1.96 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<00:55,  2.77 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:55,  2.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:55,  2.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:54,  2.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:54,  2.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:54,  2.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:53,  2.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:53,  2.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:53,  2.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:52,  2.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  10%|█         | 17/162 [00:00<00:37,  3.90 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:37,  3.90 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:36,  3.90 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:36,  3.90 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:36,  3.90 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:36,  3.90 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:35,  3.90 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:35,  3.90 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:35,  3.90 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:35,  3.90 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:34,  3.90 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  17%|█▋        | 27/162 [00:00<00:24,  5.48 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:24,  5.48 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:24,  5.48 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:24,  5.48 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:24,  5.48 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:23,  5.48 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:23,  5.48 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:23,  5.48 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:23,  5.48 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:23,  5.48 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:22,  5.48 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  23%|██▎       | 37/162 [00:00<00:16,  7.64 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:16,  7.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:16,  7.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:16,  7.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:15,  7.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:15,  7.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:15,  7.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:15,  7.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:15,  7.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:15,  7.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:15,  7.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  29%|██▉       | 47/162 [00:01<00:10, 10.53 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:10, 10.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:10, 10.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:10, 10.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:10, 10.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:10, 10.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:10, 10.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:10, 10.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:10, 10.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:10, 10.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:10, 10.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  35%|███▌      | 57/162 [00:01<00:07, 14.33 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:07, 14.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:07, 14.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:07, 14.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:07, 14.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:07, 14.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:06, 14.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:06, 14.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:06, 14.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:06, 14.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  41%|████      | 66/162 [00:01<00:05, 19.14 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:05, 19.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:04, 19.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:04, 19.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:04, 19.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:04, 19.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:04, 19.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:04, 19.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:04, 19.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:04, 19.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  46%|████▋     | 75/162 [00:01<00:03, 25.04 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:03, 25.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:03, 25.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:03, 25.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:03, 25.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:03, 25.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:03, 25.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:03, 25.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:03, 25.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:03, 25.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:03, 25.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 32.12 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 32.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 32.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 32.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 32.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:02, 32.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:02, 32.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:02, 32.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:02, 32.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:02, 32.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:02, 32.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 40.04 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 40.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 40.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 40.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 40.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 40.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 40.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 40.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 40.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 40.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 40.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 48.40 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 48.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 48.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 48.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 48.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 48.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:01, 48.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:01, 48.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:01, 48.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:01, 48.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 48.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 56.60 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 56.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 56.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 56.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 56.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 56.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 56.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 56.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 56.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 56.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 56.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 64.33 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 64.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 64.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 64.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 64.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 64.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:01<00:00, 64.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 64.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:01<00:00, 64.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:01<00:00, 64.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:01<00:00, 64.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  83%|████████▎ | 135/162 [00:01<00:00, 70.83 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:01<00:00, 70.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:01<00:00, 70.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 70.83 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 70.83 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 70.83 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 70.83 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 70.83 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 70.83 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 70.83 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 70.83 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 77.13 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 77.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 77.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 77.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 77.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 77.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 77.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 77.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 77.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 77.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 77.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 80.58 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 80.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 80.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 80.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 80.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 80.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 80.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 80.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 80.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.28s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.28s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 80.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.28s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 80.58 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.45s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.28s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 80.58 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.45s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.45s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 36.41 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.45s/ url]
0 examples [00:00, ? examples/s]2020-06-17 14:57:56.353949: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-06-17 14:57:56.369875: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-06-17 14:57:56.370050: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x564f169c7550 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-17 14:57:56.370065: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
14 examples [00:00, 139.37 examples/s]122 examples [00:00, 188.58 examples/s]225 examples [00:00, 249.69 examples/s]330 examples [00:00, 323.45 examples/s]433 examples [00:00, 407.21 examples/s]539 examples [00:00, 499.25 examples/s]643 examples [00:00, 591.09 examples/s]736 examples [00:00, 653.02 examples/s]832 examples [00:00, 721.87 examples/s]935 examples [00:01, 791.33 examples/s]1038 examples [00:01, 849.80 examples/s]1141 examples [00:01, 895.70 examples/s]1244 examples [00:01, 930.91 examples/s]1346 examples [00:01, 954.36 examples/s]1448 examples [00:01, 970.87 examples/s]1552 examples [00:01, 989.88 examples/s]1657 examples [00:01, 1007.10 examples/s]1762 examples [00:01, 1018.52 examples/s]1866 examples [00:01, 1004.43 examples/s]1973 examples [00:02, 1022.60 examples/s]2081 examples [00:02, 1037.79 examples/s]2188 examples [00:02, 1046.44 examples/s]2295 examples [00:02, 1051.46 examples/s]2401 examples [00:02, 1050.14 examples/s]2507 examples [00:02, 1033.65 examples/s]2614 examples [00:02, 1042.61 examples/s]2721 examples [00:02, 1048.35 examples/s]2826 examples [00:02, 1040.78 examples/s]2931 examples [00:02, 1040.36 examples/s]3036 examples [00:03, 1030.71 examples/s]3140 examples [00:03, 1030.46 examples/s]3244 examples [00:03, 1029.42 examples/s]3347 examples [00:03, 1008.30 examples/s]3453 examples [00:03, 1022.29 examples/s]3557 examples [00:03, 1026.42 examples/s]3661 examples [00:03, 1029.89 examples/s]3765 examples [00:03, 1031.53 examples/s]3869 examples [00:03, 1004.09 examples/s]3975 examples [00:03, 1018.65 examples/s]4080 examples [00:04, 1021.77 examples/s]4183 examples [00:04, 1007.98 examples/s]4284 examples [00:04, 1000.13 examples/s]4385 examples [00:04, 1001.89 examples/s]4486 examples [00:04, 1001.20 examples/s]4587 examples [00:04, 1002.38 examples/s]4689 examples [00:04, 1005.52 examples/s]4790 examples [00:04, 1004.76 examples/s]4893 examples [00:04, 1009.62 examples/s]4994 examples [00:04, 1004.29 examples/s]5099 examples [00:05, 1016.86 examples/s]5204 examples [00:05, 1025.77 examples/s]5309 examples [00:05, 1032.49 examples/s]5416 examples [00:05, 1040.90 examples/s]5521 examples [00:05, 1042.25 examples/s]5626 examples [00:05, 1029.77 examples/s]5730 examples [00:05, 1032.13 examples/s]5834 examples [00:05, 1027.01 examples/s]5940 examples [00:05, 1033.97 examples/s]6045 examples [00:05, 1036.74 examples/s]6149 examples [00:06, 1032.72 examples/s]6253 examples [00:06, 1020.99 examples/s]6356 examples [00:06, 997.51 examples/s] 6463 examples [00:06, 1016.27 examples/s]6569 examples [00:06, 1028.77 examples/s]6673 examples [00:06, 1027.25 examples/s]6776 examples [00:06, 982.70 examples/s] 6877 examples [00:06, 989.22 examples/s]6977 examples [00:06, 970.45 examples/s]7084 examples [00:07, 997.86 examples/s]7190 examples [00:07, 1013.40 examples/s]7296 examples [00:07, 1025.17 examples/s]7403 examples [00:07, 1035.42 examples/s]7508 examples [00:07, 1036.91 examples/s]7612 examples [00:07, 994.94 examples/s] 7717 examples [00:07, 1008.91 examples/s]7819 examples [00:07, 1010.21 examples/s]7921 examples [00:07, 1012.78 examples/s]8025 examples [00:07, 1018.61 examples/s]8129 examples [00:08, 1023.63 examples/s]8235 examples [00:08, 1033.38 examples/s]8339 examples [00:08, 1035.10 examples/s]8443 examples [00:08, 1035.87 examples/s]8549 examples [00:08, 1041.60 examples/s]8654 examples [00:08, 1041.84 examples/s]8759 examples [00:08, 1042.09 examples/s]8864 examples [00:08, 1041.20 examples/s]8969 examples [00:08, 1035.80 examples/s]9073 examples [00:08, 1019.07 examples/s]9177 examples [00:09, 1025.17 examples/s]9280 examples [00:09, 1026.38 examples/s]9385 examples [00:09, 1030.77 examples/s]9489 examples [00:09, 1009.97 examples/s]9592 examples [00:09, 1015.37 examples/s]9694 examples [00:09, 1009.62 examples/s]9796 examples [00:09, 1009.43 examples/s]9898 examples [00:09, 1011.63 examples/s]10000 examples [00:09, 1007.00 examples/s]10101 examples [00:10, 917.49 examples/s] 10198 examples [00:10, 930.07 examples/s]10300 examples [00:10, 953.44 examples/s]10402 examples [00:10, 969.81 examples/s]10505 examples [00:10, 984.67 examples/s]10608 examples [00:10, 996.87 examples/s]10709 examples [00:10, 989.79 examples/s]10810 examples [00:10, 994.58 examples/s]10911 examples [00:10, 997.33 examples/s]11013 examples [00:10, 1003.44 examples/s]11115 examples [00:11, 1007.07 examples/s]11217 examples [00:11, 1008.65 examples/s]11318 examples [00:11, 1006.57 examples/s]11420 examples [00:11, 1010.39 examples/s]11522 examples [00:11, 1011.15 examples/s]11626 examples [00:11, 1018.01 examples/s]11730 examples [00:11, 1023.37 examples/s]11835 examples [00:11, 1029.39 examples/s]11938 examples [00:11, 1010.50 examples/s]12040 examples [00:11, 1011.73 examples/s]12142 examples [00:12, 988.94 examples/s] 12244 examples [00:12, 997.36 examples/s]12344 examples [00:12, 996.06 examples/s]12444 examples [00:12, 994.49 examples/s]12546 examples [00:12, 999.50 examples/s]12649 examples [00:12, 1005.93 examples/s]12750 examples [00:12, 999.99 examples/s] 12851 examples [00:12, 999.35 examples/s]12952 examples [00:12, 1001.80 examples/s]13058 examples [00:12, 1018.40 examples/s]13160 examples [00:13, 977.08 examples/s] 13263 examples [00:13, 992.17 examples/s]13363 examples [00:13, 980.63 examples/s]13470 examples [00:13, 1003.60 examples/s]13577 examples [00:13, 1020.10 examples/s]13684 examples [00:13, 1033.83 examples/s]13788 examples [00:13, 1022.03 examples/s]13891 examples [00:13, 1011.11 examples/s]13993 examples [00:13, 1011.27 examples/s]14096 examples [00:13, 1016.36 examples/s]14198 examples [00:14, 1016.26 examples/s]14300 examples [00:14, 1013.58 examples/s]14402 examples [00:14, 1012.99 examples/s]14504 examples [00:14, 1012.28 examples/s]14608 examples [00:14, 1017.75 examples/s]14710 examples [00:14, 958.88 examples/s] 14807 examples [00:14, 955.29 examples/s]14908 examples [00:14, 970.56 examples/s]15010 examples [00:14, 983.32 examples/s]15113 examples [00:15, 995.07 examples/s]15217 examples [00:15, 1008.00 examples/s]15319 examples [00:15, 998.03 examples/s] 15419 examples [00:15, 990.17 examples/s]15521 examples [00:15, 997.49 examples/s]15623 examples [00:15, 1002.54 examples/s]15725 examples [00:15, 1005.76 examples/s]15826 examples [00:15, 984.58 examples/s] 15925 examples [00:15, 978.56 examples/s]16023 examples [00:15, 974.91 examples/s]16125 examples [00:16, 987.28 examples/s]16224 examples [00:16, 955.92 examples/s]16324 examples [00:16, 968.64 examples/s]16430 examples [00:16, 992.41 examples/s]16533 examples [00:16, 1002.72 examples/s]16634 examples [00:16, 1000.89 examples/s]16735 examples [00:16, 1003.42 examples/s]16836 examples [00:16, 986.33 examples/s] 16935 examples [00:16, 981.75 examples/s]17042 examples [00:16, 1004.24 examples/s]17148 examples [00:17, 1019.69 examples/s]17254 examples [00:17, 1031.23 examples/s]17358 examples [00:17, 1031.71 examples/s]17464 examples [00:17, 1039.15 examples/s]17569 examples [00:17, 1039.82 examples/s]17675 examples [00:17, 1043.64 examples/s]17782 examples [00:17, 1048.67 examples/s]17887 examples [00:17, 984.21 examples/s] 17987 examples [00:17, 956.77 examples/s]18091 examples [00:17, 978.36 examples/s]18194 examples [00:18, 990.54 examples/s]18294 examples [00:18, 975.93 examples/s]18393 examples [00:18, 977.67 examples/s]18496 examples [00:18, 990.16 examples/s]18601 examples [00:18, 1006.87 examples/s]18702 examples [00:18, 1006.34 examples/s]18806 examples [00:18, 1015.86 examples/s]18908 examples [00:18, 1009.89 examples/s]19013 examples [00:18, 1019.46 examples/s]19118 examples [00:19, 1026.16 examples/s]19221 examples [00:19, 1014.08 examples/s]19323 examples [00:19, 998.83 examples/s] 19423 examples [00:19, 990.83 examples/s]19523 examples [00:19, 987.20 examples/s]19622 examples [00:19, 971.98 examples/s]19727 examples [00:19, 993.19 examples/s]19827 examples [00:19, 985.82 examples/s]19931 examples [00:19, 999.80 examples/s]20032 examples [00:19, 957.48 examples/s]20136 examples [00:20, 980.39 examples/s]20235 examples [00:20, 978.35 examples/s]20334 examples [00:20, 948.93 examples/s]20437 examples [00:20, 970.20 examples/s]20539 examples [00:20, 983.89 examples/s]20644 examples [00:20, 1001.64 examples/s]20746 examples [00:20, 1006.52 examples/s]20850 examples [00:20, 1014.65 examples/s]20954 examples [00:20, 1020.01 examples/s]21057 examples [00:20, 1017.82 examples/s]21161 examples [00:21, 1022.60 examples/s]21264 examples [00:21, 1014.13 examples/s]21366 examples [00:21, 1015.84 examples/s]21470 examples [00:21, 1022.00 examples/s]21573 examples [00:21, 1020.23 examples/s]21679 examples [00:21, 1030.16 examples/s]21785 examples [00:21, 1037.35 examples/s]21889 examples [00:21, 1032.93 examples/s]21993 examples [00:21, 992.42 examples/s] 22093 examples [00:21, 989.98 examples/s]22194 examples [00:22, 994.37 examples/s]22294 examples [00:22, 975.91 examples/s]22396 examples [00:22, 986.80 examples/s]22499 examples [00:22, 998.61 examples/s]22602 examples [00:22, 1007.64 examples/s]22706 examples [00:22, 1016.20 examples/s]22811 examples [00:22, 1023.31 examples/s]22914 examples [00:22, 1011.98 examples/s]23016 examples [00:22, 985.86 examples/s] 23115 examples [00:23, 987.10 examples/s]23219 examples [00:23, 1001.48 examples/s]23325 examples [00:23, 1017.01 examples/s]23428 examples [00:23, 1019.89 examples/s]23531 examples [00:23, 1010.41 examples/s]23633 examples [00:23, 993.17 examples/s] 23733 examples [00:23, 979.69 examples/s]23835 examples [00:23, 989.98 examples/s]23935 examples [00:23, 988.33 examples/s]24041 examples [00:23, 1007.85 examples/s]24147 examples [00:24, 1021.93 examples/s]24253 examples [00:24, 1031.88 examples/s]24357 examples [00:24, 1025.99 examples/s]24460 examples [00:24, 1000.79 examples/s]24566 examples [00:24, 1017.08 examples/s]24668 examples [00:24, 1005.46 examples/s]24772 examples [00:24, 1012.93 examples/s]24874 examples [00:24, 1004.53 examples/s]24977 examples [00:24, 1010.10 examples/s]25084 examples [00:24, 1024.93 examples/s]25189 examples [00:25, 1029.83 examples/s]25294 examples [00:25, 1033.91 examples/s]25398 examples [00:25, 994.74 examples/s] 25498 examples [00:25, 979.12 examples/s]25601 examples [00:25, 991.57 examples/s]25706 examples [00:25, 1005.95 examples/s]25811 examples [00:25, 1017.33 examples/s]25916 examples [00:25, 1025.47 examples/s]26021 examples [00:25, 1032.44 examples/s]26125 examples [00:25, 1029.33 examples/s]26229 examples [00:26, 1020.33 examples/s]26332 examples [00:26, 1021.36 examples/s]26435 examples [00:26, 1023.34 examples/s]26538 examples [00:26, 1020.44 examples/s]26641 examples [00:26, 1022.13 examples/s]26744 examples [00:26, 1021.00 examples/s]26847 examples [00:26, 996.49 examples/s] 26947 examples [00:26, 982.28 examples/s]27047 examples [00:26, 987.51 examples/s]27153 examples [00:26, 1006.64 examples/s]27257 examples [00:27, 1015.12 examples/s]27364 examples [00:27, 1029.73 examples/s]27468 examples [00:27, 1027.60 examples/s]27571 examples [00:27, 1024.42 examples/s]27674 examples [00:27, 1009.42 examples/s]27776 examples [00:27, 982.54 examples/s] 27876 examples [00:27, 985.90 examples/s]27980 examples [00:27, 999.20 examples/s]28081 examples [00:27, 1000.90 examples/s]28183 examples [00:28, 1006.50 examples/s]28287 examples [00:28, 1014.06 examples/s]28389 examples [00:28, 1006.66 examples/s]28490 examples [00:28, 969.35 examples/s] 28596 examples [00:28, 994.16 examples/s]28702 examples [00:28, 1010.40 examples/s]28808 examples [00:28, 1023.26 examples/s]28911 examples [00:28, 1019.09 examples/s]29019 examples [00:28, 1034.35 examples/s]29125 examples [00:28, 1039.64 examples/s]29232 examples [00:29, 1047.35 examples/s]29339 examples [00:29, 1052.45 examples/s]29446 examples [00:29, 1055.12 examples/s]29552 examples [00:29, 1053.39 examples/s]29658 examples [00:29, 1037.14 examples/s]29762 examples [00:29, 1036.64 examples/s]29868 examples [00:29, 1042.25 examples/s]29973 examples [00:29, 1035.78 examples/s]30077 examples [00:29, 987.80 examples/s] 30181 examples [00:29, 1000.96 examples/s]30287 examples [00:30, 1016.73 examples/s]30393 examples [00:30, 1027.75 examples/s]30497 examples [00:30, 1013.08 examples/s]30599 examples [00:30, 1009.30 examples/s]30701 examples [00:30, 997.30 examples/s] 30808 examples [00:30, 1017.68 examples/s]30912 examples [00:30, 1023.04 examples/s]31018 examples [00:30, 1033.06 examples/s]31124 examples [00:30, 1038.18 examples/s]31228 examples [00:30, 1030.94 examples/s]31336 examples [00:31, 1043.24 examples/s]31443 examples [00:31, 1050.51 examples/s]31550 examples [00:31, 1053.56 examples/s]31656 examples [00:31, 1024.70 examples/s]31759 examples [00:31, 1017.98 examples/s]31867 examples [00:31, 1033.34 examples/s]31974 examples [00:31, 1041.70 examples/s]32081 examples [00:31, 1047.21 examples/s]32188 examples [00:31, 1052.42 examples/s]32294 examples [00:32, 1049.12 examples/s]32401 examples [00:32, 1053.09 examples/s]32508 examples [00:32, 1057.93 examples/s]32616 examples [00:32, 1062.08 examples/s]32723 examples [00:32, 1062.04 examples/s]32830 examples [00:32, 1063.76 examples/s]32937 examples [00:32, 1064.88 examples/s]33044 examples [00:32, 1042.90 examples/s]33149 examples [00:32, 1024.54 examples/s]33252 examples [00:32, 1024.70 examples/s]33356 examples [00:33, 1027.06 examples/s]33459 examples [00:33, 1017.09 examples/s]33563 examples [00:33, 1021.14 examples/s]33666 examples [00:33, 987.94 examples/s] 33768 examples [00:33, 994.94 examples/s]33873 examples [00:33, 1008.16 examples/s]33980 examples [00:33, 1023.37 examples/s]34083 examples [00:33, 1018.74 examples/s]34186 examples [00:33, 1013.19 examples/s]34293 examples [00:33, 1026.89 examples/s]34400 examples [00:34, 1038.33 examples/s]34506 examples [00:34, 1043.94 examples/s]34611 examples [00:34, 1039.47 examples/s]34718 examples [00:34, 1045.97 examples/s]34823 examples [00:34, 1013.21 examples/s]34928 examples [00:34, 1023.50 examples/s]35035 examples [00:34, 1035.64 examples/s]35139 examples [00:34, 1011.72 examples/s]35246 examples [00:34, 1025.80 examples/s]35351 examples [00:34, 1032.35 examples/s]35456 examples [00:35, 1036.09 examples/s]35563 examples [00:35, 1045.27 examples/s]35668 examples [00:35, 1046.37 examples/s]35776 examples [00:35, 1054.11 examples/s]35882 examples [00:35, 1042.04 examples/s]35987 examples [00:35, 1042.46 examples/s]36092 examples [00:35, 1044.52 examples/s]36197 examples [00:35, 1043.79 examples/s]36304 examples [00:35, 1050.23 examples/s]36410 examples [00:35, 1052.92 examples/s]36516 examples [00:36, 1043.82 examples/s]36621 examples [00:36, 1039.99 examples/s]36726 examples [00:36, 1027.45 examples/s]36829 examples [00:36, 1025.33 examples/s]36932 examples [00:36, 1015.65 examples/s]37034 examples [00:36, 1005.27 examples/s]37136 examples [00:36, 1009.00 examples/s]37240 examples [00:36, 1015.98 examples/s]37345 examples [00:36, 1024.04 examples/s]37450 examples [00:37, 1029.90 examples/s]37555 examples [00:37, 1035.30 examples/s]37662 examples [00:37, 1044.27 examples/s]37768 examples [00:37, 1046.87 examples/s]37874 examples [00:37, 1049.74 examples/s]37979 examples [00:37, 1016.65 examples/s]38081 examples [00:37, 1012.39 examples/s]38183 examples [00:37, 1013.90 examples/s]38285 examples [00:37, 1004.97 examples/s]38386 examples [00:37, 999.34 examples/s] 38491 examples [00:38, 1013.93 examples/s]38595 examples [00:38, 1021.50 examples/s]38700 examples [00:38, 1029.72 examples/s]38804 examples [00:38, 1026.56 examples/s]38909 examples [00:38, 1030.96 examples/s]39015 examples [00:38, 1037.82 examples/s]39119 examples [00:38, 1036.90 examples/s]39224 examples [00:38, 1039.04 examples/s]39328 examples [00:38, 1039.26 examples/s]39432 examples [00:38, 1037.87 examples/s]39538 examples [00:39, 1042.66 examples/s]39643 examples [00:39, 1041.96 examples/s]39748 examples [00:39, 1043.50 examples/s]39853 examples [00:39, 1037.13 examples/s]39960 examples [00:39, 1044.66 examples/s]40065 examples [00:39, 997.74 examples/s] 40169 examples [00:39, 1009.30 examples/s]40274 examples [00:39, 1020.14 examples/s]40377 examples [00:39, 1022.65 examples/s]40482 examples [00:39, 1028.29 examples/s]40587 examples [00:40, 1032.01 examples/s]40692 examples [00:40, 1037.15 examples/s]40796 examples [00:40, 1036.68 examples/s]40903 examples [00:40, 1043.90 examples/s]41008 examples [00:40, 1027.87 examples/s]41111 examples [00:40, 1004.15 examples/s]41212 examples [00:40, 996.59 examples/s] 41314 examples [00:40, 1003.29 examples/s]41417 examples [00:40, 1011.01 examples/s]41519 examples [00:40, 1012.06 examples/s]41624 examples [00:41, 1020.45 examples/s]41727 examples [00:41, 1010.85 examples/s]41829 examples [00:41, 1011.96 examples/s]41931 examples [00:41, 1008.08 examples/s]42034 examples [00:41, 1013.14 examples/s]42136 examples [00:41, 945.84 examples/s] 42239 examples [00:41, 969.45 examples/s]42343 examples [00:41, 987.11 examples/s]42447 examples [00:41, 1000.48 examples/s]42549 examples [00:42, 1003.97 examples/s]42654 examples [00:42, 1017.31 examples/s]42759 examples [00:42, 1026.24 examples/s]42862 examples [00:42, 1017.24 examples/s]42964 examples [00:42, 1007.85 examples/s]43066 examples [00:42, 1009.69 examples/s]43168 examples [00:42, 1004.23 examples/s]43274 examples [00:42, 1018.64 examples/s]43381 examples [00:42, 1031.11 examples/s]43485 examples [00:42, 1024.74 examples/s]43588 examples [00:43, 998.61 examples/s] 43692 examples [00:43, 1009.21 examples/s]43794 examples [00:43, 1006.55 examples/s]43895 examples [00:43, 984.59 examples/s] 43995 examples [00:43, 988.41 examples/s]44094 examples [00:43, 964.93 examples/s]44191 examples [00:43, 950.62 examples/s]44289 examples [00:43, 957.25 examples/s]44390 examples [00:43, 971.69 examples/s]44494 examples [00:43, 990.17 examples/s]44596 examples [00:44, 996.33 examples/s]44699 examples [00:44, 1005.53 examples/s]44801 examples [00:44, 1008.89 examples/s]44903 examples [00:44, 1010.33 examples/s]45007 examples [00:44, 1016.20 examples/s]45110 examples [00:44, 1019.10 examples/s]45212 examples [00:44, 1019.35 examples/s]45314 examples [00:44, 1017.77 examples/s]45416 examples [00:44, 1016.68 examples/s]45518 examples [00:44, 1017.31 examples/s]45620 examples [00:45, 1016.41 examples/s]45724 examples [00:45, 1021.63 examples/s]45827 examples [00:45, 1019.46 examples/s]45929 examples [00:45, 996.97 examples/s] 46033 examples [00:45, 1007.97 examples/s]46134 examples [00:45, 990.93 examples/s] 46237 examples [00:45, 1002.33 examples/s]46341 examples [00:45, 1011.62 examples/s]46445 examples [00:45, 1018.55 examples/s]46552 examples [00:45, 1031.99 examples/s]46656 examples [00:46, 1025.57 examples/s]46759 examples [00:46, 1014.50 examples/s]46861 examples [00:46, 1009.00 examples/s]46964 examples [00:46, 1012.45 examples/s]47068 examples [00:46, 1017.76 examples/s]47170 examples [00:46, 1005.22 examples/s]47271 examples [00:46, 998.43 examples/s] 47375 examples [00:46, 1009.21 examples/s]47480 examples [00:46, 1019.66 examples/s]47587 examples [00:47, 1032.35 examples/s]47694 examples [00:47, 1041.22 examples/s]47799 examples [00:47, 1042.38 examples/s]47904 examples [00:47, 1038.67 examples/s]48011 examples [00:47, 1047.54 examples/s]48116 examples [00:47, 1046.99 examples/s]48223 examples [00:47, 1052.99 examples/s]48329 examples [00:47, 1036.74 examples/s]48433 examples [00:47, 1018.27 examples/s]48536 examples [00:47, 1018.70 examples/s]48639 examples [00:48, 1021.95 examples/s]48744 examples [00:48, 1027.95 examples/s]48847 examples [00:48, 1024.61 examples/s]48950 examples [00:48, 1006.20 examples/s]49055 examples [00:48, 1017.85 examples/s]49160 examples [00:48, 1024.00 examples/s]49265 examples [00:48, 1031.57 examples/s]49369 examples [00:48, 1019.25 examples/s]49472 examples [00:48, 1020.25 examples/s]49575 examples [00:48, 1021.84 examples/s]49678 examples [00:49, 1019.25 examples/s]49780 examples [00:49, 1015.96 examples/s]49882 examples [00:49, 1015.76 examples/s]49985 examples [00:49, 1018.40 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 13%|█▎        | 6312/50000 [00:00<00:00, 63094.79 examples/s] 36%|███▌      | 18113/50000 [00:00<00:00, 73332.04 examples/s] 60%|██████    | 30081/50000 [00:00<00:00, 82971.35 examples/s] 84%|████████▎ | 41796/50000 [00:00<00:00, 90929.40 examples/s]                                                               0 examples [00:00, ? examples/s]88 examples [00:00, 875.67 examples/s]196 examples [00:00, 926.95 examples/s]303 examples [00:00, 964.38 examples/s]410 examples [00:00, 992.32 examples/s]518 examples [00:00, 1014.44 examples/s]626 examples [00:00, 1032.22 examples/s]733 examples [00:00, 1041.74 examples/s]841 examples [00:00, 1050.41 examples/s]947 examples [00:00, 1053.09 examples/s]1055 examples [00:01, 1059.01 examples/s]1161 examples [00:01, 1057.24 examples/s]1266 examples [00:01, 1053.84 examples/s]1374 examples [00:01, 1061.45 examples/s]1480 examples [00:01, 1060.88 examples/s]1586 examples [00:01, 1028.09 examples/s]1695 examples [00:01, 1043.31 examples/s]1802 examples [00:01, 1050.42 examples/s]1909 examples [00:01, 1055.06 examples/s]2016 examples [00:01, 1056.81 examples/s]2123 examples [00:02, 1059.59 examples/s]2229 examples [00:02, 1044.17 examples/s]2337 examples [00:02, 1054.03 examples/s]2445 examples [00:02, 1060.37 examples/s]2552 examples [00:02, 1060.05 examples/s]2659 examples [00:02, 1058.91 examples/s]2767 examples [00:02, 1064.19 examples/s]2874 examples [00:02, 1063.62 examples/s]2981 examples [00:02, 1047.10 examples/s]3086 examples [00:02, 1006.22 examples/s]3192 examples [00:03, 1020.67 examples/s]3297 examples [00:03, 1027.69 examples/s]3403 examples [00:03, 1033.16 examples/s]3510 examples [00:03, 1041.17 examples/s]3616 examples [00:03, 1045.14 examples/s]3723 examples [00:03, 1051.10 examples/s]3831 examples [00:03, 1058.67 examples/s]3939 examples [00:03, 1063.11 examples/s]4046 examples [00:03, 1061.85 examples/s]4153 examples [00:03, 1031.35 examples/s]4257 examples [00:04, 1023.35 examples/s]4360 examples [00:04, 1013.35 examples/s]4464 examples [00:04, 1020.96 examples/s]4573 examples [00:04, 1038.29 examples/s]4677 examples [00:04, 1014.67 examples/s]4785 examples [00:04, 1032.43 examples/s]4892 examples [00:04, 1043.12 examples/s]4998 examples [00:04, 1045.70 examples/s]5103 examples [00:04, 987.08 examples/s] 5203 examples [00:05, 945.84 examples/s]5299 examples [00:05, 942.30 examples/s]5401 examples [00:05, 962.95 examples/s]5507 examples [00:05, 989.68 examples/s]5612 examples [00:05, 1006.39 examples/s]5714 examples [00:05, 1003.60 examples/s]5822 examples [00:05, 1023.29 examples/s]5928 examples [00:05, 1032.41 examples/s]6032 examples [00:05, 1033.65 examples/s]6138 examples [00:05, 1041.30 examples/s]6243 examples [00:06, 987.32 examples/s] 6347 examples [00:06, 1000.23 examples/s]6452 examples [00:06, 1014.07 examples/s]6559 examples [00:06, 1028.47 examples/s]6667 examples [00:06, 1040.95 examples/s]6773 examples [00:06, 1045.62 examples/s]6882 examples [00:06, 1056.46 examples/s]6991 examples [00:06, 1064.73 examples/s]7099 examples [00:06, 1068.99 examples/s]7207 examples [00:06, 1070.38 examples/s]7315 examples [00:07, 1068.17 examples/s]7425 examples [00:07, 1075.45 examples/s]7534 examples [00:07, 1077.43 examples/s]7643 examples [00:07, 1079.88 examples/s]7752 examples [00:07, 1079.86 examples/s]7861 examples [00:07, 1041.12 examples/s]7969 examples [00:07, 1049.78 examples/s]8077 examples [00:07, 1057.47 examples/s]8184 examples [00:07, 1059.69 examples/s]8291 examples [00:07, 1051.66 examples/s]8397 examples [00:08, 1051.92 examples/s]8503 examples [00:08, 1049.27 examples/s]8612 examples [00:08, 1060.54 examples/s]8721 examples [00:08, 1064.94 examples/s]8828 examples [00:08, 1054.38 examples/s]8934 examples [00:08, 1051.88 examples/s]9041 examples [00:08, 1057.13 examples/s]9147 examples [00:08, 1057.40 examples/s]9253 examples [00:08, 1047.74 examples/s]9358 examples [00:09, 970.75 examples/s] 9457 examples [00:09, 956.95 examples/s]9560 examples [00:09, 975.46 examples/s]9666 examples [00:09, 999.16 examples/s]9767 examples [00:09, 1000.82 examples/s]9871 examples [00:09, 1010.88 examples/s]9973 examples [00:09, 960.18 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteGLF9A0/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteGLF9A0/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f5d41ebc950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f5d41ebc950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f5d41ebc950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5cdc0f7828>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5cdc429908>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f5d41ebc950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f5d41ebc950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f5d41ebc950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5d41eaaac8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5d3f5036d8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f5cb99706a8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f5cb99706a8> 

  function with postional parmater data_info <function split_train_valid at 0x7f5cb99706a8> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f5cb99707b8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f5cb99707b8> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f5cb99707b8> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.0
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.0/en_core_web_sm-2.3.0.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.0) (2.3.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (45.2.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.0.3)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.6.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.4.1)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.23.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.1.3)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (7.4.1)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.18.5)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (4.46.1)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.25.9)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2020.4.5.2)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.4)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.6.1)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.0-py3-none-any.whl size=12048606 sha256=b1139c13d0ccfb4d40e9e93adcb20793bf2593551e97c72eccb3f563be1e5ed9
  Stored in directory: /tmp/pip-ephem-wheel-cache-daguxkhh/wheels/4a/db/07/94eee4f3a60150464a04160bd0dfe9c8752ab981fe92f16aea
Successfully built en-core-web-sm
Installing collected packages: en-core-web-sm
Successfully installed en-core-web-sm-2.3.0
[38;5;2m✔ Download and installation successful[0m
You can now load the model via spacy.load('en_core_web_sm')
[38;5;2m✔ Linking successful[0m
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/en_core_web_sm
-->
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/data/en
You can now load the model via spacy.load('en')
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<22:34:37, 10.6kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<16:03:26, 14.9kB/s].vector_cache/glove.6B.zip:   0%|          | 147k/862M [00:01<11:18:47, 21.2kB/s] .vector_cache/glove.6B.zip:   0%|          | 352k/862M [00:01<7:57:09, 30.1kB/s] .vector_cache/glove.6B.zip:   0%|          | 852k/862M [00:01<5:34:40, 42.9kB/s].vector_cache/glove.6B.zip:   0%|          | 1.79M/862M [00:01<3:54:28, 61.2kB/s].vector_cache/glove.6B.zip:   0%|          | 3.08M/862M [00:01<2:44:13, 87.2kB/s].vector_cache/glove.6B.zip:   0%|          | 4.31M/862M [00:01<1:55:09, 124kB/s] .vector_cache/glove.6B.zip:   1%|          | 5.65M/862M [00:01<1:20:48, 177kB/s].vector_cache/glove.6B.zip:   1%|          | 7.00M/862M [00:01<56:47, 251kB/s]  .vector_cache/glove.6B.zip:   1%|          | 8.54M/862M [00:01<39:57, 356kB/s].vector_cache/glove.6B.zip:   1%|▏         | 11.8M/862M [00:01<27:59, 506kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.8M/862M [00:02<19:36, 719kB/s].vector_cache/glove.6B.zip:   2%|▏         | 19.7M/862M [00:02<13:46, 1.02MB/s].vector_cache/glove.6B.zip:   3%|▎         | 23.6M/862M [00:02<09:42, 1.44MB/s].vector_cache/glove.6B.zip:   3%|▎         | 27.4M/862M [00:02<06:52, 2.02MB/s].vector_cache/glove.6B.zip:   4%|▎         | 30.3M/862M [00:04<07:14, 1.91MB/s].vector_cache/glove.6B.zip:   4%|▎         | 30.8M/862M [00:04<06:01, 2.30MB/s].vector_cache/glove.6B.zip:   4%|▎         | 32.3M/862M [00:04<04:30, 3.07MB/s].vector_cache/glove.6B.zip:   4%|▍         | 34.4M/862M [00:10<15:24, 895kB/s] .vector_cache/glove.6B.zip:   4%|▍         | 34.4M/862M [00:10<24:49, 556kB/s].vector_cache/glove.6B.zip:   4%|▍         | 34.6M/862M [00:10<20:41, 667kB/s].vector_cache/glove.6B.zip:   4%|▍         | 35.1M/862M [00:10<15:17, 901kB/s].vector_cache/glove.6B.zip:   4%|▍         | 36.1M/862M [00:10<11:05, 1.24MB/s].vector_cache/glove.6B.zip:   4%|▍         | 37.8M/862M [00:11<08:00, 1.72MB/s].vector_cache/glove.6B.zip:   4%|▍         | 38.5M/862M [00:12<14:47, 928kB/s] .vector_cache/glove.6B.zip:   4%|▍         | 38.8M/862M [00:12<12:22, 1.11MB/s].vector_cache/glove.6B.zip:   5%|▍         | 39.9M/862M [00:12<09:09, 1.50MB/s].vector_cache/glove.6B.zip:   5%|▍         | 42.6M/862M [00:14<08:30, 1.60MB/s].vector_cache/glove.6B.zip:   5%|▍         | 43.0M/862M [00:14<07:05, 1.93MB/s].vector_cache/glove.6B.zip:   5%|▌         | 43.9M/862M [00:14<05:27, 2.50MB/s].vector_cache/glove.6B.zip:   5%|▌         | 44.7M/862M [00:14<04:18, 3.16MB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.2M/862M [00:14<03:16, 4.14MB/s].vector_cache/glove.6B.zip:   6%|▌         | 48.6M/862M [00:14<02:28, 5.50MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.0M/862M [00:14<01:53, 7.16MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.5M/862M [00:15<02:52, 4.70MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.0M/862M [00:15<02:13, 6.07MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.6M/862M [00:17<06:16, 2.14MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.7M/862M [00:17<11:10, 1.20MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.0M/862M [00:17<09:27, 1.42MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.3M/862M [00:17<07:02, 1.90MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.6M/862M [00:17<05:05, 2.62MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.7M/862M [00:19<37:37, 355kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 60.9M/862M [00:19<29:59, 445kB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.5M/862M [00:19<21:52, 610kB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.9M/862M [00:19<15:28, 860kB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.9M/862M [00:21<16:57, 784kB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.1M/862M [00:21<15:20, 866kB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.7M/862M [00:21<11:34, 1.15MB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.4M/862M [00:21<08:19, 1.59MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.1M/862M [00:23<10:25, 1.27MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.1M/862M [00:23<18:11, 726kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 69.3M/862M [00:23<15:59, 826kB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.9M/862M [00:24<11:51, 1.11MB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.1M/862M [00:24<08:37, 1.53MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.2M/862M [00:25<08:49, 1.49MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.4M/862M [00:25<09:37, 1.37MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.0M/862M [00:25<07:34, 1.74MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.4M/862M [00:26<05:28, 2.39MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.4M/862M [00:27<10:17, 1.27MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.5M/862M [00:27<10:37, 1.23MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.2M/862M [00:27<08:16, 1.58MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.6M/862M [00:28<05:57, 2.18MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.5M/862M [00:29<10:37, 1.23MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.7M/862M [00:29<10:46, 1.21MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.3M/862M [00:29<08:21, 1.55MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.5M/862M [00:30<06:10, 2.10MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.7M/862M [00:42<25:54, 500kB/s] .vector_cache/glove.6B.zip:  10%|▉         | 85.7M/862M [00:42<28:44, 450kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.9M/862M [00:42<23:09, 559kB/s].vector_cache/glove.6B.zip:  10%|█         | 86.6M/862M [00:42<16:58, 762kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.1M/862M [00:42<12:03, 1.07MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.9M/862M [00:48<37:43, 341kB/s] .vector_cache/glove.6B.zip:  10%|█         | 89.9M/862M [00:48<39:07, 329kB/s].vector_cache/glove.6B.zip:  10%|█         | 90.1M/862M [00:48<30:27, 423kB/s].vector_cache/glove.6B.zip:  11%|█         | 90.8M/862M [00:49<22:04, 582kB/s].vector_cache/glove.6B.zip:  11%|█         | 91.9M/862M [00:49<15:46, 814kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.5M/862M [00:49<11:15, 1.14MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.0M/862M [00:53<38:15, 335kB/s] .vector_cache/glove.6B.zip:  11%|█         | 94.1M/862M [00:53<40:00, 320kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.1M/862M [00:53<32:15, 397kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.3M/862M [00:53<25:07, 509kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.5M/862M [00:53<19:30, 656kB/s].vector_cache/glove.6B.zip:  11%|█         | 95.1M/862M [00:53<14:15, 897kB/s].vector_cache/glove.6B.zip:  11%|█         | 96.3M/862M [00:53<10:18, 1.24MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.6M/862M [00:53<07:28, 1.70MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.1M/862M [00:54<14:44, 864kB/s] .vector_cache/glove.6B.zip:  11%|█▏        | 98.3M/862M [00:55<12:45, 998kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.5M/862M [00:55<11:12, 1.14MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.6M/862M [00:55<10:09, 1.25MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.8M/862M [00:55<09:19, 1.36MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 99.1M/862M [00:55<07:58, 1.60MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.6M/862M [00:55<06:17, 2.02MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 100M/862M [00:55<05:00, 2.54MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:55<03:49, 3.32MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:56<08:23, 1.51MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:57<06:45, 1.87MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:57<05:00, 2.52MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:58<06:27, 1.95MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:59<11:31, 1.09MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:59<09:39, 1.30MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 108M/862M [00:59<07:13, 1.74MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:59<05:17, 2.37MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [01:00<07:45, 1.61MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [01:01<08:44, 1.43MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [01:01<06:55, 1.81MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [01:01<05:01, 2.49MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [01:02<08:02, 1.55MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [01:02<08:43, 1.43MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [01:03<06:48, 1.83MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [01:03<05:04, 2.45MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [01:04<06:17, 1.97MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [01:04<07:31, 1.65MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [01:05<05:58, 2.07MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [01:05<04:21, 2.83MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [01:06<07:21, 1.68MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [01:06<08:02, 1.53MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [01:07<06:13, 1.97MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [01:07<04:45, 2.59MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [01:08<05:51, 2.09MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [01:08<06:59, 1.75MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [01:09<05:37, 2.18MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 129M/862M [01:09<04:10, 2.92MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [01:10<06:06, 2.00MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [01:10<07:03, 1.72MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [01:11<05:37, 2.16MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [01:11<04:16, 2.85MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [01:11<03:08, 3.86MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [01:15<5:10:06, 39.1kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [01:15<3:47:04, 53.3kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [01:15<2:41:23, 75.0kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [01:15<1:54:32, 106kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [01:15<1:20:29, 150kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [01:15<56:22, 214kB/s]  .vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [01:17<45:40, 264kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [01:17<33:52, 355kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 140M/862M [01:17<24:55, 483kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [01:17<18:17, 658kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [01:17<13:29, 891kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [01:17<09:47, 1.23MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [01:17<07:05, 1.69MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [01:19<31:26, 381kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [01:19<23:40, 506kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [01:19<16:58, 704kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [01:19<12:03, 989kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [01:21<13:39, 872kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [01:21<12:55, 921kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [01:21<10:19, 1.15MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [01:21<07:39, 1.55MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [01:21<05:30, 2.15MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [01:24<16:50, 702kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [01:24<24:54, 475kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [01:24<23:52, 496kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [01:24<23:18, 508kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [01:24<19:48, 597kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [01:24<14:48, 799kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [01:24<10:44, 1.10MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [01:25<07:42, 1.53MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [01:26<13:08, 896kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [01:27<15:04, 780kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [01:27<12:01, 978kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [01:27<08:44, 1.34MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [01:27<06:17, 1.86MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [01:28<11:27, 1.02MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [01:28<10:10, 1.15MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [01:29<08:30, 1.37MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [01:29<06:55, 1.69MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [01:29<05:01, 2.32MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [01:29<03:49, 3.04MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [01:30<22:20, 521kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [01:30<23:05, 504kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [01:31<17:47, 653kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [01:31<13:25, 866kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [01:31<09:42, 1.19MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [01:31<07:11, 1.61MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [01:31<05:19, 2.17MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [01:32<15:56, 725kB/s] .vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [01:32<14:33, 794kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [01:33<10:57, 1.05MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [01:33<08:05, 1.43MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [01:33<05:55, 1.94MB/s].vector_cache/glove.6B.zip:  20%|██        | 172M/862M [01:33<04:28, 2.57MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:34<20:06, 571kB/s] .vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:34<20:18, 566kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:35<15:43, 730kB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:35<11:24, 1.01MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:35<08:18, 1.38MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:35<06:06, 1.87MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:36<11:00, 1.04MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:36<11:36, 984kB/s] .vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:36<10:27, 1.09MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:37<08:42, 1.31MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:37<07:41, 1.48MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:37<06:40, 1.71MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:37<06:01, 1.89MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:37<04:35, 2.48MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:37<03:31, 3.23MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:38<06:52, 1.65MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:38<07:20, 1.55MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:38<06:36, 1.72MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:39<05:22, 2.11MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:39<04:25, 2.56MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:39<03:38, 3.11MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:39<03:01, 3.73MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:39<02:23, 4.71MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:40<19:39, 574kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 185M/862M [01:40<16:26, 686kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:41<12:09, 927kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:41<08:49, 1.28MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:41<06:26, 1.74MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:42<09:56, 1.13MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:42<12:00, 934kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:43<10:12, 1.10MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:43<08:12, 1.36MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:43<06:37, 1.69MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:43<05:23, 2.08MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:43<04:20, 2.58MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:43<03:33, 3.14MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:43<03:00, 3.71MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:50<30:10, 369kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:50<32:09, 347kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:50<25:06, 444kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:50<18:09, 613kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:50<13:02, 852kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:50<09:27, 1.17MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:54<15:39, 707kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:54<22:03, 502kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:54<18:12, 608kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:54<13:18, 831kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:54<09:37, 1.15MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:55<06:58, 1.58MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:55<05:24, 2.04MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:55<04:02, 2.72MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:55<03:07, 3.51MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:55<02:30, 4.37MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:59<18:59, 576kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:59<24:42, 443kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:59<20:14, 540kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:59<15:20, 712kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:59<11:07, 982kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:59<08:02, 1.36MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [02:00<05:49, 1.87MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [02:01<2:58:04, 61.0kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [02:01<2:08:12, 84.8kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [02:01<1:32:06, 118kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [02:01<1:05:39, 165kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [02:01<46:51, 232kB/s]  .vector_cache/glove.6B.zip:  25%|██▍       | 211M/862M [02:01<33:23, 325kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [02:02<23:52, 454kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [02:02<17:03, 634kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [02:02<12:41, 851kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [02:05<23:41, 456kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [02:05<27:10, 397kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [02:05<21:34, 500kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [02:05<16:09, 668kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [02:05<11:46, 915kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [02:05<08:58, 1.20MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [02:05<07:41, 1.40MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [02:06<06:32, 1.64MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [02:06<05:16, 2.04MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [02:06<04:10, 2.57MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [02:12<26:59, 398kB/s] .vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [02:12<29:37, 362kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [02:12<23:18, 460kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [02:12<16:54, 634kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [02:12<12:13, 875kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [02:13<08:54, 1.20MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [02:13<06:40, 1.60MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [02:16<18:02, 591kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [02:16<23:20, 457kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [02:16<18:50, 566kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [02:16<13:43, 775kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [02:17<09:59, 1.07MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [02:17<07:20, 1.44MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [02:17<05:34, 1.90MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [02:17<04:19, 2.45MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [02:17<03:19, 3.18MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [02:17<02:44, 3.86MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [02:17<02:27, 4.30MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [02:17<02:13, 4.75MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [02:17<01:58, 5.31MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [02:19<2:01:34, 86.6kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [02:19<1:33:02, 113kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [02:19<1:07:03, 157kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [02:19<47:19, 222kB/s]  .vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [02:19<33:31, 313kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [02:20<23:42, 442kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [02:20<17:00, 615kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [02:21<20:34, 508kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [02:21<18:10, 575kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [02:21<13:32, 772kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [02:21<09:59, 1.04MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [02:21<07:19, 1.42MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [02:21<05:30, 1.89MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [02:22<04:13, 2.47MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [02:23<14:03, 739kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [02:23<13:28, 771kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [02:23<10:11, 1.02MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [02:23<07:41, 1.35MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [02:23<05:44, 1.80MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [02:23<04:35, 2.26MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [02:24<03:37, 2.86MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [02:24<03:22, 3.05MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [02:24<03:12, 3.21MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [02:25<35:04, 294kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [02:25<27:35, 374kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [02:25<21:01, 491kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [02:25<16:24, 628kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [02:25<13:06, 787kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [02:25<10:58, 939kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [02:25<09:20, 1.10MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [02:26<07:28, 1.38MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [02:26<05:44, 1.79MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [02:26<04:31, 2.27MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [02:26<03:34, 2.87MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [02:27<06:17, 1.63MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [02:27<06:09, 1.66MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [02:27<04:59, 2.05MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [02:27<03:59, 2.56MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [02:27<03:12, 3.18MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [02:27<02:39, 3.84MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [02:27<02:14, 4.55MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [02:29<09:27, 1.08MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [02:29<10:07, 1.01MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [02:29<08:01, 1.27MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [02:29<06:09, 1.65MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [02:29<04:43, 2.15MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [02:29<03:41, 2.75MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [02:29<03:02, 3.33MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [02:30<02:32, 3.99MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [02:33<59:45, 169kB/s] .vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [02:33<51:52, 195kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [02:34<38:25, 263kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [02:34<27:21, 369kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [02:34<19:49, 509kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [02:34<14:15, 707kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [02:34<10:21, 972kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 259M/862M [02:34<07:48, 1.29MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [02:34<06:10, 1.63MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [02:35<14:40, 684kB/s] .vector_cache/glove.6B.zip:  30%|███       | 259M/862M [02:35<12:59, 773kB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [02:36<11:38, 863kB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [02:36<10:41, 939kB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [02:36<08:50, 1.14MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [02:36<07:22, 1.36MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [02:36<06:38, 1.51MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [02:36<06:00, 1.67MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [02:36<04:39, 2.15MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [02:36<03:41, 2.71MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [02:36<02:55, 3.42MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [02:37<06:39, 1.50MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [02:37<06:19, 1.58MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [02:38<04:50, 2.05MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [02:38<03:46, 2.63MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [02:38<02:54, 3.41MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [02:38<02:22, 4.17MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [02:38<01:57, 5.05MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [02:38<01:39, 5.96MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [02:38<01:27, 6.80MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [02:38<01:17, 7.58MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [02:41<2:26:42, 67.1kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [02:41<1:52:27, 87.5kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [02:41<1:20:43, 122kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [02:41<57:43, 170kB/s]  .vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [02:42<40:46, 241kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [02:42<28:51, 340kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [02:42<20:29, 478kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [02:42<14:37, 669kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [02:43<24:44, 395kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [02:43<20:13, 483kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [02:43<14:48, 660kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [02:43<10:51, 899kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [02:43<07:52, 1.24MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [02:44<05:46, 1.68MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [02:45<09:05, 1.07MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [02:45<12:24, 783kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [02:45<10:52, 893kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [02:45<08:42, 1.11MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [02:45<06:59, 1.39MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [02:46<05:37, 1.72MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [02:46<04:44, 2.04MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [02:46<04:16, 2.27MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [02:46<04:00, 2.42MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [02:46<03:51, 2.51MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [02:46<03:48, 2.53MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [02:46<03:28, 2.77MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [02:46<03:20, 2.89MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [02:46<03:06, 3.10MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [02:47<08:13, 1.17MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [02:47<06:25, 1.50MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [02:47<05:10, 1.86MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [02:47<03:47, 2.53MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [02:47<03:05, 3.10MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [02:48<03:06, 3.09MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [02:48<02:55, 3.28MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [02:52<1:07:07, 143kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [02:53<56:34, 169kB/s]  .vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [02:53<41:49, 229kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [02:53<29:48, 320kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [02:53<21:12, 450kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [02:53<15:08, 629kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [02:53<10:56, 870kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [02:53<08:20, 1.14MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [02:56<30:00, 317kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [02:56<28:50, 329kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [02:56<22:24, 424kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [02:56<16:13, 584kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [02:56<11:37, 814kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [02:57<08:26, 1.12MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:57<06:11, 1.52MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [03:00<34:27, 274kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [03:00<33:30, 281kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [03:00<25:39, 367kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [03:00<18:27, 511kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [03:00<13:16, 708kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [03:01<09:31, 986kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [03:01<06:56, 1.35MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [03:01<05:05, 1.83MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [03:01<05:11, 1.80MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [03:01<04:22, 2.13MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [03:01<03:48, 2.45MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [03:01<03:23, 2.75MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [03:02<03:06, 2.99MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [03:02<02:54, 3.20MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [03:02<02:39, 3.50MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [03:02<02:20, 3.97MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [03:02<03:15, 2.85MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [03:02<02:51, 3.25MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [03:02<03:58, 2.34MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [03:02<04:57, 1.87MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [03:02<05:06, 1.82MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [03:03<1:26:31, 107kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [03:03<1:02:07, 149kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [03:03<44:00, 210kB/s]  .vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [03:03<31:12, 296kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [03:04<22:16, 415kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [03:04<16:01, 576kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [03:04<11:37, 793kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [03:04<08:34, 1.07MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [03:08<53:04, 173kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [03:08<46:52, 196kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [03:08<36:33, 252kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [03:08<29:08, 316kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [03:08<24:13, 380kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [03:08<20:42, 444kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [03:08<17:24, 528kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [03:08<14:14, 646kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [03:09<11:45, 781kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [03:09<09:53, 928kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [03:09<09:01, 1.02MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [03:09<08:06, 1.13MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [03:09<07:19, 1.25MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [03:09<06:55, 1.33MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [03:09<06:39, 1.38MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [03:09<06:22, 1.44MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [03:09<05:46, 1.59MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [03:09<04:41, 1.95MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [03:10<03:47, 2.42MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [03:10<03:06, 2.94MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [03:10<02:36, 3.50MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [03:10<02:28, 3.70MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [03:10<02:19, 3.92MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [03:10<02:05, 4.34MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [03:10<01:55, 4.72MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [03:11<04:45, 1.91MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [03:11<04:21, 2.08MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [03:11<03:22, 2.68MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [03:11<02:55, 3.10MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [03:11<02:23, 3.78MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [03:12<02:06, 4.28MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [03:12<01:54, 4.72MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [03:13<10:04, 894kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [03:13<11:06, 811kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [03:13<08:47, 1.02MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [03:13<06:34, 1.37MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [03:14<04:58, 1.80MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [03:14<03:51, 2.32MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [03:14<03:00, 2.98MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [03:15<1:39:31, 89.9kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [03:15<1:13:40, 121kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [03:15<52:32, 170kB/s]  .vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [03:15<37:09, 240kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [03:15<26:18, 339kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [03:16<18:55, 471kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [03:16<13:39, 651kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [03:16<09:55, 895kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [03:20<1:22:12, 108kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [03:21<1:05:04, 136kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [03:21<47:39, 186kB/s]  .vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [03:21<33:58, 261kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [03:21<24:09, 367kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [03:21<17:16, 512kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [03:21<12:27, 709kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [03:21<09:08, 964kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [03:21<06:51, 1.29MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [03:22<19:56, 442kB/s] .vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [03:22<16:42, 527kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [03:23<13:16, 664kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [03:23<10:56, 804kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [03:23<09:43, 905kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [03:23<08:25, 1.04MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [03:23<07:48, 1.13MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [03:23<07:31, 1.17MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [03:23<07:51, 1.12MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [03:23<07:31, 1.17MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [03:23<06:38, 1.32MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [03:23<06:18, 1.39MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [03:24<05:47, 1.51MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [03:24<05:11, 1.69MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [03:24<04:49, 1.82MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [03:24<04:41, 1.87MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [03:24<04:38, 1.89MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [03:24<03:55, 2.23MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [03:24<03:26, 2.54MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [03:26<08:52, 984kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [03:26<14:51, 588kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [03:26<12:27, 701kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [03:26<09:17, 940kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [03:26<06:52, 1.27MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [03:27<05:07, 1.70MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [03:27<03:58, 2.19MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [03:27<03:15, 2.66MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [03:27<03:02, 2.85MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [03:35<1:44:28, 83.0kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [03:35<1:21:39, 106kB/s] .vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [03:35<58:56, 147kB/s]  .vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [03:36<42:18, 205kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [03:36<29:54, 289kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [03:36<21:20, 405kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [03:36<15:16, 565kB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [03:36<11:02, 780kB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [03:36<07:59, 1.08MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [03:36<05:56, 1.45MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [03:36<04:40, 1.83MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [03:36<14:53, 576kB/s] .vector_cache/glove.6B.zip:  40%|████      | 348M/862M [03:37<10:48, 793kB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [03:37<07:56, 1.08MB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [03:37<07:25, 1.15MB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [03:37<55:37, 154kB/s] .vector_cache/glove.6B.zip:  41%|████      | 350M/862M [03:37<39:12, 218kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [03:37<27:53, 306kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [03:38<19:59, 426kB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [03:38<14:32, 585kB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [03:38<10:33, 805kB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [03:38<07:46, 1.09MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [03:41<22:29, 377kB/s] .vector_cache/glove.6B.zip:  41%|████      | 354M/862M [03:41<24:09, 351kB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [03:41<18:56, 447kB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [03:41<13:45, 615kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [03:42<09:52, 855kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [03:42<07:09, 1.18MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 358M/862M [03:44<12:41, 663kB/s] .vector_cache/glove.6B.zip:  41%|████▏     | 358M/862M [03:44<15:54, 528kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [03:44<13:14, 635kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [03:44<09:43, 864kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [03:44<07:13, 1.16MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [03:44<05:22, 1.56MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [03:44<03:59, 2.09MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [03:46<06:28, 1.29MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [03:46<06:40, 1.25MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [03:46<05:54, 1.41MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [03:46<04:27, 1.87MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [03:46<03:31, 2.36MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [03:46<02:40, 3.09MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [03:46<02:08, 3.86MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [03:48<14:39, 564kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [03:48<18:40, 443kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [03:49<15:03, 549kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [03:49<11:47, 701kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [03:49<09:20, 884kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [03:49<08:19, 992kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [03:49<07:35, 1.09MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [03:49<06:23, 1.29MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [03:49<05:22, 1.54MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [03:49<04:46, 1.73MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [03:49<04:22, 1.88MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [03:49<03:59, 2.06MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [03:50<03:45, 2.19MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [03:50<03:38, 2.26MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [03:50<03:15, 2.52MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [03:50<02:34, 3.19MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [03:52<29:05, 282kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [03:52<28:27, 288kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [03:52<21:50, 375kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [03:52<15:41, 522kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [03:53<11:15, 726kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [03:53<08:05, 1.01MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [03:53<05:56, 1.37MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [03:56<24:51, 327kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [03:56<26:49, 303kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [03:56<20:53, 389kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [03:56<15:00, 541kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [03:56<10:56, 741kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [03:56<07:56, 1.02MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [03:57<05:53, 1.37MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [03:57<04:41, 1.72MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [03:57<03:47, 2.12MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [03:58<18:30, 436kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [03:58<14:33, 554kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [03:58<11:07, 724kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [03:58<08:44, 921kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [03:58<06:59, 1.15MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [03:58<05:30, 1.46MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [03:58<04:06, 1.95MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [03:59<05:41, 1.41MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [03:59<04:14, 1.88MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [03:59<03:08, 2.55MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [04:06<10:52, 731kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [04:06<14:16, 557kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [04:06<11:53, 668kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [04:06<08:47, 904kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [04:06<06:14, 1.27MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [04:06<04:29, 1.76MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [04:07<04:51, 1.62MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [04:07<03:34, 2.19MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [04:07<02:42, 2.88MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [04:07<02:05, 3.74MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [04:09<08:20, 934kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [04:09<08:30, 915kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [04:09<06:35, 1.18MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [04:09<04:45, 1.63MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [04:11<05:33, 1.39MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [04:11<05:53, 1.31MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [04:11<04:39, 1.66MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [04:11<03:23, 2.26MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [04:13<05:32, 1.38MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [04:13<07:04, 1.08MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [04:13<07:09, 1.07MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [04:13<06:12, 1.23MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [04:13<04:45, 1.61MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [04:13<03:35, 2.12MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [04:13<02:35, 2.92MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [04:15<6:13:31, 20.3kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [04:15<4:22:46, 28.8kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [04:15<3:04:23, 41.1kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [04:15<2:09:00, 58.6kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [04:15<1:29:55, 83.6kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [04:17<1:40:43, 74.6kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [04:18<1:16:44, 97.9kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [04:18<55:09, 136kB/s]   .vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [04:18<38:57, 192kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [04:18<27:12, 274kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [04:19<35:03, 212kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [04:20<25:59, 286kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [04:20<18:49, 395kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [04:20<13:22, 555kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [04:20<09:29, 780kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [04:21<09:41, 761kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [04:21<08:44, 844kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [04:22<07:37, 966kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [04:22<05:57, 1.24MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [04:22<04:24, 1.67MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [04:22<03:10, 2.30MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [04:23<07:42, 948kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [04:23<07:29, 974kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [04:24<06:06, 1.19MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [04:24<04:36, 1.58MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [04:24<03:21, 2.16MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [04:25<04:32, 1.59MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [04:25<04:21, 1.66MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [04:26<03:29, 2.07MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [04:26<02:44, 2.64MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [04:26<02:13, 3.23MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [04:26<01:55, 3.72MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [04:26<01:32, 4.67MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [04:27<17:34, 408kB/s] .vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [04:27<13:15, 540kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [04:28<09:49, 729kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [04:28<07:09, 999kB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [04:28<05:10, 1.38MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [04:29<05:41, 1.25MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [04:29<05:48, 1.22MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [04:30<04:45, 1.49MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [04:30<03:32, 2.00MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [04:30<02:32, 2.76MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [04:33<3:34:13, 32.8kB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [04:33<2:36:53, 44.8kB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [04:33<1:51:27, 63.1kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [04:33<1:18:14, 89.7kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [04:34<54:57, 127kB/s]   .vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [04:38<42:21, 164kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [04:38<36:36, 190kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [04:38<27:23, 254kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [04:39<19:31, 356kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [04:39<13:58, 497kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [04:39<10:04, 688kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [04:39<07:17, 948kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [04:39<09:27, 731kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [04:39<07:01, 984kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [04:39<05:10, 1.33MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [04:39<03:53, 1.77MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [04:39<02:59, 2.30MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [04:39<02:20, 2.92MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [04:40<01:54, 3.57MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [04:42<50:29, 135kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [04:42<42:17, 162kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [04:42<31:13, 219kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [04:42<22:10, 308kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [04:42<15:35, 436kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [04:42<11:04, 612kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [04:44<22:47, 297kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [04:44<17:27, 387kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [04:44<12:30, 540kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [04:44<08:56, 753kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [04:44<06:22, 1.05MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [04:46<08:17, 808kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [04:46<08:32, 784kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [04:46<07:37, 878kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [04:46<06:17, 1.07MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [04:46<04:52, 1.37MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 461M/862M [04:46<03:57, 1.69MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [04:46<03:12, 2.08MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [04:46<02:33, 2.60MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [04:46<02:04, 3.20MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [04:47<01:42, 3.87MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [04:48<07:11, 921kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [04:48<05:59, 1.11MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [04:48<04:36, 1.44MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [04:48<03:38, 1.81MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [04:48<02:50, 2.32MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [04:48<02:17, 2.87MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [04:48<01:51, 3.53MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [04:48<01:33, 4.22MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [04:50<34:47, 189kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [04:50<25:54, 253kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [04:50<18:22, 356kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [04:50<13:02, 501kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [04:50<09:17, 701kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [04:50<06:39, 974kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [04:52<47:17, 137kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [04:52<37:12, 174kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [04:52<27:53, 233kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [04:52<21:17, 305kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [04:52<16:20, 397kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [04:52<15:06, 429kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [04:53<1:04:59, 99.7kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [04:53<45:48, 141kB/s]   .vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [04:53<32:06, 201kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [04:53<22:37, 284kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [04:53<16:06, 398kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [04:53<11:31, 555kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [04:54<12:19, 519kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [04:55<09:14, 691kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [04:55<06:37, 961kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [04:55<04:50, 1.31MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [04:55<03:37, 1.75MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [04:55<02:50, 2.23MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [05:01<38:00, 166kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [05:01<32:13, 196kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [05:01<24:04, 263kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [05:01<17:11, 367kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [05:02<12:09, 517kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [05:02<08:35, 729kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [05:09<46:17, 135kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [05:09<38:49, 161kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [05:09<28:51, 217kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [05:09<21:17, 294kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [05:09<15:16, 409kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [05:09<10:55, 571kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [05:09<07:46, 799kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [05:09<05:35, 1.11MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [05:12<27:12, 227kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [05:12<25:51, 239kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [05:12<19:35, 316kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [05:12<14:02, 440kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [05:12<09:57, 618kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [05:13<07:02, 870kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [05:14<42:06, 145kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [05:14<30:34, 200kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [05:14<21:46, 280kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [05:14<15:28, 394kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [05:14<10:58, 553kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [05:14<07:50, 773kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [05:16<09:10, 659kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [05:16<08:46, 689kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [05:16<06:38, 909kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [05:16<04:53, 1.23MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [05:16<03:31, 1.70MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [05:18<04:29, 1.33MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [05:18<05:37, 1.06MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [05:18<04:25, 1.35MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [05:18<03:23, 1.76MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [05:18<02:29, 2.39MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [05:18<01:52, 3.17MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [05:21<22:02, 268kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [05:21<21:46, 271kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [05:21<16:47, 352kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [05:21<12:25, 475kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [05:21<09:00, 655kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [05:21<06:23, 919kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [05:23<06:02, 967kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [05:23<05:18, 1.10MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [05:23<03:59, 1.46MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [05:23<02:55, 1.98MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [05:23<02:12, 2.63MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [05:25<04:20, 1.33MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [05:25<05:20, 1.08MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [05:25<04:16, 1.35MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [05:25<03:10, 1.81MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [05:25<02:21, 2.43MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [05:25<01:49, 3.13MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [05:27<05:45, 991kB/s] .vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [05:27<05:46, 986kB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [05:27<05:02, 1.13MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [05:27<03:59, 1.43MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [05:27<03:06, 1.82MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [05:27<02:24, 2.35MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [05:27<01:53, 2.99MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [05:28<01:28, 3.84MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [05:33<43:25, 130kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [05:33<36:16, 155kB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [05:34<26:43, 211kB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [05:34<19:00, 296kB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [05:34<13:26, 417kB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [05:34<09:28, 588kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [05:34<06:49, 816kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [05:34<04:56, 1.12MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [05:34<05:16, 1.05MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 530M/862M [05:35<03:55, 1.41MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [05:35<02:57, 1.87MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [05:35<02:17, 2.40MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [05:35<01:51, 2.95MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [05:35<01:36, 3.42MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [05:36<04:16, 1.28MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [05:37<06:01, 909kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [05:37<05:35, 979kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [05:37<04:28, 1.22MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [05:37<03:19, 1.64MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [05:37<02:25, 2.24MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [05:38<03:30, 1.54MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [05:39<04:57, 1.09MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [05:39<04:47, 1.13MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [05:39<04:02, 1.34MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [05:39<03:10, 1.69MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [05:39<02:23, 2.25MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [05:39<01:50, 2.92MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [05:39<01:29, 3.60MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [05:40<04:54, 1.09MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [05:41<04:38, 1.15MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [05:41<03:29, 1.52MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [05:41<02:33, 2.07MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [05:42<03:05, 1.71MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [05:43<03:54, 1.35MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [05:43<03:10, 1.66MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [05:43<02:19, 2.25MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [05:43<01:43, 3.02MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [05:46<10:07, 513kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [05:47<12:30, 415kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [05:47<09:54, 524kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [05:47<07:43, 671kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [05:47<05:38, 918kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [05:47<04:03, 1.27MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [05:48<04:01, 1.27MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [05:49<03:36, 1.42MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [05:49<02:49, 1.81MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [05:49<02:11, 2.33MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [05:49<01:41, 3.00MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [05:49<01:16, 3.97MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [05:53<1:19:13, 63.8kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [05:53<1:00:28, 83.6kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [05:53<43:29, 116kB/s]   .vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [05:53<30:37, 165kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [05:53<21:26, 234kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [05:53<15:06, 330kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [05:54<10:34, 468kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [05:54<07:25, 662kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [05:55<24:07, 204kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [05:56<19:34, 251kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [05:56<14:20, 343kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [05:56<10:22, 473kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [05:56<07:17, 669kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [05:56<05:11, 936kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [05:57<18:38, 260kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [05:58<15:11, 319kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [05:58<11:48, 410kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [05:58<08:47, 551kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [05:58<06:24, 754kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [05:58<04:36, 1.05MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [06:00<04:22, 1.09MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [06:00<04:18, 1.11MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [06:00<03:20, 1.43MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [06:00<02:25, 1.95MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [06:07<06:19, 746kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [06:07<09:11, 513kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [06:07<08:02, 586kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [06:07<06:33, 717kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [06:07<04:50, 972kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [06:07<03:30, 1.34MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [06:17<08:43, 533kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [06:17<10:41, 434kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [06:17<08:35, 540kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [06:17<06:14, 741kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [06:17<04:30, 1.02MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [06:18<03:09, 1.44MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [06:18<03:25, 1.33MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [06:18<02:33, 1.77MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [06:18<01:53, 2.39MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [06:21<03:19, 1.35MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [06:22<06:47, 660kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [06:22<05:48, 770kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [06:22<04:17, 1.04MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [06:22<03:07, 1.42MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [06:23<03:02, 1.45MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [06:23<03:18, 1.33MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [06:24<02:51, 1.54MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [06:24<02:23, 1.84MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [06:24<01:50, 2.38MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [06:24<01:27, 3.02MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [06:24<01:07, 3.85MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [06:30<12:07, 359kB/s] .vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [06:30<12:52, 338kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [06:31<10:05, 430kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [06:31<07:48, 556kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [06:31<06:09, 705kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [06:31<05:05, 850kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [06:31<04:15, 1.02MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [06:31<03:23, 1.27MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [06:31<02:31, 1.71MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [06:31<01:54, 2.24MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [06:31<01:27, 2.93MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [06:33<21:52, 195kB/s] .vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [06:33<17:41, 242kB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [06:33<12:59, 329kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [06:33<09:11, 463kB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [06:34<06:25, 655kB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [06:49<1:59:25, 35.2kB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [06:49<1:27:45, 47.9kB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [06:50<1:02:22, 67.4kB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [06:50<43:52, 95.6kB/s]  .vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [06:50<30:45, 136kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [06:50<21:31, 193kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 614M/862M [06:50<15:00, 275kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [06:50<11:15, 366kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [06:51<07:57, 514kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [06:51<05:37, 724kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [06:51<04:00, 1.01MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [06:53<10:35, 382kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [06:53<09:37, 420kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [06:53<07:17, 555kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [06:53<05:22, 751kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [06:53<03:47, 1.05MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [06:54<03:43, 1.07MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [06:55<03:38, 1.09MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [06:55<02:45, 1.43MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [06:55<02:01, 1.95MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [06:56<02:17, 1.71MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [06:57<02:35, 1.51MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [06:57<02:01, 1.92MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [06:57<01:31, 2.54MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [06:58<01:50, 2.08MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [06:59<02:05, 1.83MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [06:59<01:54, 2.01MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [06:59<01:35, 2.39MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 634M/862M [06:59<01:15, 3.01MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [06:59<01:10, 3.25MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [06:59<01:07, 3.38MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [06:59<01:03, 3.60MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [06:59<00:59, 3.82MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [06:59<00:53, 4.22MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [07:00<05:57, 632kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [07:01<04:29, 838kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [07:01<03:11, 1.17MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [07:01<02:18, 1.61MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [07:02<05:58, 620kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [07:02<04:51, 761kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [07:03<03:32, 1.04MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [07:03<02:35, 1.42MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [07:08<04:18, 843kB/s] .vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [07:08<06:36, 550kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [07:08<05:37, 644kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [07:08<04:22, 827kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [07:08<03:13, 1.12MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [07:08<02:16, 1.57MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [07:10<03:14, 1.10MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [07:10<02:53, 1.23MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [07:10<02:38, 1.34MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [07:10<02:32, 1.40MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [07:10<02:01, 1.75MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [07:10<01:31, 2.31MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [07:10<01:07, 3.11MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [07:12<03:34, 977kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [07:12<03:13, 1.08MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [07:12<02:37, 1.33MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [07:12<01:55, 1.80MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [07:12<01:24, 2.46MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [07:14<02:37, 1.30MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [07:14<02:43, 1.26MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [07:14<02:23, 1.43MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 657M/862M [07:14<01:55, 1.77MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [07:14<01:28, 2.30MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [07:14<01:09, 2.93MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [07:14<00:52, 3.83MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [07:16<04:06, 817kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [07:16<03:14, 1.04MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [07:16<02:19, 1.43MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [07:18<02:21, 1.39MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [07:18<03:06, 1.06MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [07:18<02:43, 1.21MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [07:18<02:15, 1.45MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [07:18<01:45, 1.85MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [07:18<01:22, 2.37MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [07:18<01:05, 2.99MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [07:18<00:49, 3.91MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [07:19<09:34, 336kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [07:20<07:08, 450kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [07:20<05:17, 607kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [07:20<03:51, 830kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [07:20<02:45, 1.15MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [07:20<01:58, 1.59MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [07:23<09:57, 316kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [07:23<10:09, 310kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [07:23<07:49, 402kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [07:23<05:39, 554kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [07:23<03:58, 781kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [07:26<04:48, 640kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [07:27<06:31, 472kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [07:27<05:16, 583kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [07:27<04:30, 683kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [07:27<03:47, 811kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [07:27<03:15, 944kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [07:27<02:33, 1.20MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [07:27<01:52, 1.62MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [07:27<01:21, 2.21MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [07:28<02:19, 1.30MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [07:29<01:47, 1.68MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [07:29<01:18, 2.28MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [07:29<00:58, 3.04MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [07:33<04:25, 666kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [07:33<06:13, 473kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [07:33<05:36, 525kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [07:33<04:56, 594kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [07:33<04:26, 660kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [07:33<03:39, 801kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [07:33<02:44, 1.06MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [07:33<02:01, 1.44MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [07:33<01:27, 1.99MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [07:34<02:20, 1.22MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [07:35<01:51, 1.54MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [07:35<01:26, 1.97MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [07:35<01:11, 2.41MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [07:35<01:02, 2.71MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [07:35<00:57, 2.94MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [07:35<00:52, 3.24MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [07:35<00:45, 3.71MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [07:35<00:39, 4.22MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [07:40<31:11, 89.9kB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [07:40<24:37, 114kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [07:40<17:53, 157kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [07:40<12:38, 221kB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [07:41<08:53, 312kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [07:41<06:12, 443kB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [07:41<04:39, 587kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [07:41<03:18, 819kB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [07:41<02:23, 1.13MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [07:41<01:43, 1.55MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [07:43<02:59, 891kB/s] .vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [07:43<03:01, 884kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [07:43<02:44, 970kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 703M/862M [07:43<02:14, 1.19MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [07:43<01:42, 1.55MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [07:43<01:16, 2.08MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [07:44<00:56, 2.79MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [07:45<02:21, 1.11MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [07:45<02:07, 1.23MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [07:45<01:56, 1.34MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [07:45<01:35, 1.62MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [07:45<01:13, 2.11MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [07:45<00:54, 2.83MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [07:52<04:27, 569kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [07:52<05:40, 446kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [07:52<04:29, 562kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [07:52<03:22, 749kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [07:52<02:24, 1.04MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [07:52<01:43, 1.44MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [07:53<01:49, 1.35MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [07:53<01:22, 1.77MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [07:54<01:01, 2.37MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [07:54<00:44, 3.21MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [07:57<05:29, 435kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [07:58<05:52, 407kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [07:58<04:41, 509kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [07:58<03:23, 701kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [07:58<02:24, 978kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [08:04<03:55, 593kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [08:05<05:01, 462kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [08:05<04:04, 570kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [08:05<02:57, 783kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [08:05<02:07, 1.08MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [08:08<02:26, 926kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [08:08<03:58, 568kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [08:08<03:18, 681kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [08:09<02:25, 925kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [08:09<01:41, 1.30MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [08:09<01:52, 1.16MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [08:09<01:25, 1.53MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [08:10<01:07, 1.92MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [08:10<00:56, 2.29MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [08:10<00:46, 2.78MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [08:10<00:35, 3.61MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [08:11<01:33, 1.36MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [08:12<01:31, 1.39MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [08:12<01:09, 1.80MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 737M/862M [08:12<00:51, 2.43MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [08:12<00:37, 3.26MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [08:13<04:34, 447kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [08:14<03:37, 564kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [08:14<02:37, 773kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [08:14<01:50, 1.09MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [08:15<02:05, 949kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 744M/862M [08:15<01:52, 1.06MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [08:16<01:24, 1.40MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [08:16<01:01, 1.90MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [08:17<01:08, 1.67MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [08:17<01:12, 1.59MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [08:18<00:55, 2.06MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [08:18<00:40, 2.76MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [08:22<01:54, 966kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [08:23<02:54, 635kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [08:23<02:27, 748kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [08:23<01:49, 1.01MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [08:23<01:19, 1.38MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [08:27<01:53, 934kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [08:27<03:06, 572kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [08:28<02:35, 685kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [08:28<01:56, 910kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [08:28<01:24, 1.25MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [08:28<00:58, 1.74MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [08:29<06:06, 279kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [08:29<04:42, 362kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [08:29<03:31, 482kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [08:30<02:33, 660kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [08:30<01:49, 920kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [08:30<01:16, 1.28MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [08:37<09:30, 172kB/s] .vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [08:37<08:17, 197kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [08:37<06:10, 264kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [08:37<04:22, 370kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [08:37<03:04, 521kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [08:41<02:47, 561kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [08:41<03:29, 449kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [08:41<02:48, 557kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [08:41<02:02, 759kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [08:41<01:25, 1.07MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [08:43<01:26, 1.04MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [08:43<01:11, 1.25MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [08:43<00:55, 1.60MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [08:43<00:42, 2.08MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [08:43<00:33, 2.60MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [08:43<00:28, 3.05MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [08:43<00:23, 3.73MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [08:43<00:21, 3.93MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [08:47<05:37, 255kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [08:47<05:20, 268kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [08:47<04:04, 351kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [08:47<02:54, 488kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [08:47<01:59, 690kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [08:54<04:31, 301kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [08:54<04:33, 299kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [08:54<03:29, 388kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [08:55<02:29, 539kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [08:55<01:44, 759kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [08:56<01:36, 804kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [08:57<01:37, 799kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [08:57<01:14, 1.03MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [08:57<00:55, 1.38MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [08:57<00:39, 1.91MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [09:00<01:28, 830kB/s] .vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [09:00<02:16, 537kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [09:00<01:53, 648kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [09:01<01:22, 879kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [09:01<00:58, 1.22MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [09:02<00:56, 1.23MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [09:02<00:59, 1.16MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [09:02<01:14, 925kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [09:02<01:20, 857kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [09:03<01:33, 739kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [09:03<01:31, 750kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [09:03<01:31, 753kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [09:03<01:21, 848kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [09:03<01:05, 1.05MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [09:03<00:52, 1.31MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [09:03<00:42, 1.60MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [09:03<00:37, 1.82MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [09:03<00:32, 2.08MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [09:03<00:27, 2.46MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [09:04<00:21, 3.09MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [09:04<00:17, 3.70MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [09:04<01:41, 644kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [09:04<01:13, 879kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [09:04<00:54, 1.18MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [09:04<00:40, 1.55MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [09:05<00:32, 1.95MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [09:05<00:26, 2.37MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [09:05<00:21, 2.83MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [09:05<00:17, 3.40MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [09:08<08:17, 123kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [09:08<06:48, 149kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [09:09<04:59, 203kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [09:09<03:34, 282kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [09:09<02:28, 399kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [09:14<02:15, 419kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [09:14<02:31, 376kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [09:14<01:58, 477kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [09:14<01:25, 654kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [09:14<00:57, 925kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [09:15<00:58, 896kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [09:15<00:42, 1.22MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [09:15<00:32, 1.57MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [09:15<00:27, 1.84MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [09:15<00:23, 2.16MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [09:16<00:19, 2.58MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [09:16<00:16, 2.96MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [09:16<00:14, 3.44MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [09:16<00:12, 3.82MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [09:21<13:39, 59.3kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [09:21<10:21, 78.1kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [09:21<07:25, 109kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [09:21<05:09, 154kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [09:21<03:33, 219kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [09:25<02:44, 271kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [09:25<02:39, 278kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [09:25<02:00, 366kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [09:25<01:28, 496kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [09:25<01:01, 696kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [09:26<00:42, 975kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [09:28<01:03, 640kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [09:28<01:18, 512kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [09:28<01:04, 620kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [09:28<00:49, 806kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [09:28<00:35, 1.10MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [09:28<00:23, 1.54MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [09:30<00:45, 788kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [09:30<00:44, 813kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [09:30<00:38, 938kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [09:31<00:30, 1.15MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [09:31<00:23, 1.50MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [09:31<00:17, 1.99MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [09:31<00:13, 2.50MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [09:31<00:11, 3.01MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [09:31<00:09, 3.41MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [09:43<03:53, 137kB/s] .vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [09:43<03:09, 169kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [09:43<02:19, 228kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [09:43<01:37, 320kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [09:44<01:03, 454kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [09:45<01:01, 451kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [09:45<00:47, 578kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [09:45<00:34, 783kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [09:45<00:24, 1.08MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [09:46<00:17, 1.44MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [09:46<00:12, 1.93MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [09:47<00:20, 1.18MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [09:47<00:16, 1.40MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [09:47<00:11, 1.90MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [09:47<00:07, 2.61MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [09:49<00:37, 526kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [09:49<00:29, 659kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [09:49<00:21, 893kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [09:49<00:14, 1.23MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [09:49<00:09, 1.68MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [09:50<00:07, 2.24MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [09:51<00:55, 280kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [09:51<00:44, 348kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [09:51<00:34, 451kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [09:51<00:24, 604kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [09:51<00:18, 811kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [09:52<00:12, 1.12MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 849M/862M [09:52<00:08, 1.52MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [09:53<00:10, 1.12MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [09:54<00:13, 829kB/s] .vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [09:54<00:13, 817kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [09:54<00:13, 810kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [09:54<00:12, 895kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [09:54<00:09, 1.13MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [09:54<00:06, 1.49MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [09:54<00:04, 1.99MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [09:54<00:03, 2.65MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [10:00<00:19, 382kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [10:00<00:20, 353kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [10:01<00:15, 450kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [10:01<00:10, 623kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [10:01<00:06, 869kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [10:02<00:02, 1.10MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [10:02<00:00, 1.53MB/s].vector_cache/glove.6B.zip: 862MB [10:02, 1.43MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<21:54:02,  5.07it/s]  0%|          | 790/400000 [00:00<15:18:15,  7.25it/s]  0%|          | 1605/400000 [00:00<10:41:43, 10.35it/s]  1%|          | 2417/400000 [00:00<7:28:31, 14.77it/s]   1%|          | 3231/400000 [00:00<5:13:34, 21.09it/s]  1%|          | 4036/400000 [00:00<3:39:18, 30.09it/s]  1%|          | 4854/400000 [00:00<2:33:26, 42.92it/s]  1%|▏         | 5659/400000 [00:00<1:47:25, 61.18it/s]  2%|▏         | 6475/400000 [00:00<1:15:17, 87.12it/s]  2%|▏         | 7281/400000 [00:01<52:50, 123.88it/s]   2%|▏         | 8079/400000 [00:01<37:09, 175.80it/s]  2%|▏         | 8889/400000 [00:01<26:11, 248.82it/s]  2%|▏         | 9704/400000 [00:01<18:32, 350.86it/s]  3%|▎         | 10508/400000 [00:01<13:11, 492.02it/s]  3%|▎         | 11308/400000 [00:01<09:27, 684.67it/s]  3%|▎         | 12106/400000 [00:01<06:51, 943.31it/s]  3%|▎         | 12925/400000 [00:01<05:01, 1284.16it/s]  3%|▎         | 13729/400000 [00:01<03:45, 1712.92it/s]  4%|▎         | 14536/400000 [00:02<02:51, 2242.78it/s]  4%|▍         | 15340/400000 [00:02<02:14, 2861.78it/s]  4%|▍         | 16155/400000 [00:02<01:48, 3553.41it/s]  4%|▍         | 16960/400000 [00:02<01:29, 4266.94it/s]  4%|▍         | 17776/400000 [00:02<01:16, 4978.92it/s]  5%|▍         | 18590/400000 [00:02<01:07, 5635.38it/s]  5%|▍         | 19400/400000 [00:02<01:03, 6005.72it/s]  5%|▌         | 20212/400000 [00:02<00:58, 6514.01it/s]  5%|▌         | 21026/400000 [00:02<00:54, 6928.42it/s]  5%|▌         | 21838/400000 [00:02<00:52, 7246.66it/s]  6%|▌         | 22644/400000 [00:03<00:50, 7470.77it/s]  6%|▌         | 23455/400000 [00:03<00:49, 7649.41it/s]  6%|▌         | 24269/400000 [00:03<00:48, 7789.14it/s]  6%|▋         | 25086/400000 [00:03<00:47, 7897.79it/s]  6%|▋         | 25897/400000 [00:03<00:47, 7891.03it/s]  7%|▋         | 26710/400000 [00:03<00:46, 7960.75it/s]  7%|▋         | 27519/400000 [00:03<00:46, 7996.87it/s]  7%|▋         | 28334/400000 [00:03<00:46, 8041.57it/s]  7%|▋         | 29154/400000 [00:03<00:45, 8087.95it/s]  7%|▋         | 29969/400000 [00:03<00:45, 8104.51it/s]  8%|▊         | 30782/400000 [00:04<00:46, 8022.44it/s]  8%|▊         | 31601/400000 [00:04<00:45, 8071.34it/s]  8%|▊         | 32410/400000 [00:04<00:46, 7866.83it/s]  8%|▊         | 33229/400000 [00:04<00:46, 7959.09it/s]  9%|▊         | 34044/400000 [00:04<00:45, 8013.64it/s]  9%|▊         | 34847/400000 [00:04<00:45, 8007.63it/s]  9%|▉         | 35654/400000 [00:04<00:45, 8024.10it/s]  9%|▉         | 36458/400000 [00:04<00:45, 8023.17it/s]  9%|▉         | 37272/400000 [00:04<00:45, 8055.47it/s] 10%|▉         | 38079/400000 [00:04<00:44, 8059.72it/s] 10%|▉         | 38894/400000 [00:05<00:44, 8084.30it/s] 10%|▉         | 39703/400000 [00:05<00:44, 8063.68it/s] 10%|█         | 40517/400000 [00:05<00:44, 8083.68it/s] 10%|█         | 41326/400000 [00:05<00:44, 7988.44it/s] 11%|█         | 42137/400000 [00:05<00:44, 8021.87it/s] 11%|█         | 42945/400000 [00:05<00:44, 8037.65it/s] 11%|█         | 43750/400000 [00:05<00:44, 8038.38it/s] 11%|█         | 44554/400000 [00:05<00:44, 7942.75it/s] 11%|█▏        | 45349/400000 [00:05<00:44, 7892.79it/s] 12%|█▏        | 46142/400000 [00:05<00:44, 7903.88it/s] 12%|█▏        | 46946/400000 [00:06<00:44, 7942.95it/s] 12%|█▏        | 47765/400000 [00:06<00:43, 8014.31it/s] 12%|█▏        | 48585/400000 [00:06<00:43, 8068.26it/s] 12%|█▏        | 49395/400000 [00:06<00:43, 8075.08it/s] 13%|█▎        | 50203/400000 [00:06<00:43, 8062.16it/s] 13%|█▎        | 51016/400000 [00:06<00:43, 8081.85it/s] 13%|█▎        | 51825/400000 [00:06<00:43, 7958.32it/s] 13%|█▎        | 52634/400000 [00:06<00:43, 7996.45it/s] 13%|█▎        | 53445/400000 [00:06<00:43, 8028.15it/s] 14%|█▎        | 54257/400000 [00:06<00:42, 8053.02it/s] 14%|█▍        | 55064/400000 [00:07<00:42, 8055.70it/s] 14%|█▍        | 55870/400000 [00:07<00:43, 7995.64it/s] 14%|█▍        | 56677/400000 [00:07<00:42, 8015.92it/s] 14%|█▍        | 57488/400000 [00:07<00:42, 8043.36it/s] 15%|█▍        | 58305/400000 [00:07<00:42, 8080.09it/s] 15%|█▍        | 59114/400000 [00:07<00:42, 8012.02it/s] 15%|█▍        | 59930/400000 [00:07<00:42, 8054.12it/s] 15%|█▌        | 60736/400000 [00:07<00:43, 7759.09it/s] 15%|█▌        | 61538/400000 [00:07<00:43, 7834.33it/s] 16%|█▌        | 62358/400000 [00:07<00:42, 7940.11it/s] 16%|█▌        | 63154/400000 [00:08<00:42, 7919.35it/s] 16%|█▌        | 63951/400000 [00:08<00:42, 7931.90it/s] 16%|█▌        | 64746/400000 [00:08<00:42, 7877.76it/s] 16%|█▋        | 65559/400000 [00:08<00:42, 7950.13it/s] 17%|█▋        | 66375/400000 [00:08<00:41, 7989.13it/s] 17%|█▋        | 67179/400000 [00:08<00:41, 8002.41it/s] 17%|█▋        | 67988/400000 [00:08<00:41, 8026.78it/s] 17%|█▋        | 68796/400000 [00:08<00:41, 8041.14it/s] 17%|█▋        | 69603/400000 [00:08<00:41, 8049.66it/s] 18%|█▊        | 70416/400000 [00:08<00:40, 8070.03it/s] 18%|█▊        | 71224/400000 [00:09<00:41, 7910.51it/s] 18%|█▊        | 72037/400000 [00:09<00:41, 7973.93it/s] 18%|█▊        | 72854/400000 [00:09<00:40, 8031.20it/s] 18%|█▊        | 73674/400000 [00:09<00:40, 8079.79it/s] 19%|█▊        | 74484/400000 [00:09<00:40, 8084.66it/s] 19%|█▉        | 75293/400000 [00:09<00:40, 8073.25it/s] 19%|█▉        | 76101/400000 [00:09<00:40, 8069.38it/s] 19%|█▉        | 76920/400000 [00:09<00:39, 8102.85it/s] 19%|█▉        | 77731/400000 [00:09<00:39, 8066.36it/s] 20%|█▉        | 78538/400000 [00:09<00:39, 8054.04it/s] 20%|█▉        | 79344/400000 [00:10<00:40, 7977.20it/s] 20%|██        | 80160/400000 [00:10<00:39, 8030.37it/s] 20%|██        | 80977/400000 [00:10<00:39, 8070.41it/s] 20%|██        | 81797/400000 [00:10<00:39, 8107.58it/s] 21%|██        | 82608/400000 [00:10<00:39, 8088.08it/s] 21%|██        | 83417/400000 [00:10<00:39, 8025.36it/s] 21%|██        | 84220/400000 [00:10<00:39, 8011.47it/s] 21%|██▏       | 85022/400000 [00:10<00:39, 7983.58it/s] 21%|██▏       | 85821/400000 [00:10<00:39, 7919.69it/s] 22%|██▏       | 86614/400000 [00:11<00:39, 7884.65it/s] 22%|██▏       | 87422/400000 [00:11<00:39, 7939.82it/s] 22%|██▏       | 88242/400000 [00:11<00:38, 8014.42it/s] 22%|██▏       | 89065/400000 [00:11<00:38, 8075.85it/s] 22%|██▏       | 89885/400000 [00:11<00:38, 8112.45it/s] 23%|██▎       | 90697/400000 [00:11<00:38, 8024.49it/s] 23%|██▎       | 91500/400000 [00:11<00:38, 7993.02it/s] 23%|██▎       | 92300/400000 [00:11<00:39, 7825.05it/s] 23%|██▎       | 93117/400000 [00:11<00:38, 7925.43it/s] 23%|██▎       | 93911/400000 [00:11<00:38, 7915.33it/s] 24%|██▎       | 94715/400000 [00:12<00:38, 7950.04it/s] 24%|██▍       | 95518/400000 [00:12<00:38, 7972.71it/s] 24%|██▍       | 96316/400000 [00:12<00:38, 7966.00it/s] 24%|██▍       | 97113/400000 [00:12<00:38, 7911.37it/s] 24%|██▍       | 97905/400000 [00:12<00:38, 7856.83it/s] 25%|██▍       | 98714/400000 [00:12<00:38, 7923.38it/s] 25%|██▍       | 99521/400000 [00:12<00:37, 7966.77it/s] 25%|██▌       | 100318/400000 [00:12<00:37, 7951.11it/s] 25%|██▌       | 101121/400000 [00:12<00:37, 7972.90it/s] 25%|██▌       | 101919/400000 [00:12<00:37, 7968.98it/s] 26%|██▌       | 102717/400000 [00:13<00:37, 7869.89it/s] 26%|██▌       | 103536/400000 [00:13<00:37, 7961.05it/s] 26%|██▌       | 104356/400000 [00:13<00:36, 8028.76it/s] 26%|██▋       | 105160/400000 [00:13<00:37, 7818.95it/s] 26%|██▋       | 105970/400000 [00:13<00:37, 7898.79it/s] 27%|██▋       | 106762/400000 [00:13<00:37, 7854.67it/s] 27%|██▋       | 107568/400000 [00:13<00:36, 7913.67it/s] 27%|██▋       | 108380/400000 [00:13<00:36, 7973.70it/s] 27%|██▋       | 109206/400000 [00:13<00:36, 8055.22it/s] 28%|██▊       | 110031/400000 [00:13<00:35, 8110.35it/s] 28%|██▊       | 110844/400000 [00:14<00:35, 8114.23it/s] 28%|██▊       | 111666/400000 [00:14<00:35, 8145.63it/s] 28%|██▊       | 112490/400000 [00:14<00:35, 8172.86it/s] 28%|██▊       | 113308/400000 [00:14<00:35, 8157.98it/s] 29%|██▊       | 114134/400000 [00:14<00:34, 8187.73it/s] 29%|██▊       | 114958/400000 [00:14<00:34, 8201.24it/s] 29%|██▉       | 115779/400000 [00:14<00:34, 8172.61it/s] 29%|██▉       | 116597/400000 [00:14<00:35, 8081.16it/s] 29%|██▉       | 117416/400000 [00:14<00:34, 8112.58it/s] 30%|██▉       | 118228/400000 [00:14<00:34, 8071.80it/s] 30%|██▉       | 119036/400000 [00:15<00:35, 8017.24it/s] 30%|██▉       | 119846/400000 [00:15<00:34, 8040.53it/s] 30%|███       | 120651/400000 [00:15<00:34, 8039.51it/s] 30%|███       | 121456/400000 [00:15<00:34, 7991.47it/s] 31%|███       | 122278/400000 [00:15<00:34, 8056.50it/s] 31%|███       | 123084/400000 [00:15<00:34, 8052.97it/s] 31%|███       | 123906/400000 [00:15<00:34, 8099.52it/s] 31%|███       | 124729/400000 [00:15<00:33, 8138.16it/s] 31%|███▏      | 125552/400000 [00:15<00:33, 8164.90it/s] 32%|███▏      | 126369/400000 [00:15<00:33, 8151.50it/s] 32%|███▏      | 127185/400000 [00:16<00:33, 8113.79it/s] 32%|███▏      | 128004/400000 [00:16<00:33, 8136.14it/s] 32%|███▏      | 128818/400000 [00:16<00:33, 7993.97it/s] 32%|███▏      | 129635/400000 [00:16<00:33, 8044.74it/s] 33%|███▎      | 130440/400000 [00:16<00:33, 7993.04it/s] 33%|███▎      | 131240/400000 [00:16<00:33, 7940.16it/s] 33%|███▎      | 132035/400000 [00:16<00:33, 7888.60it/s] 33%|███▎      | 132825/400000 [00:16<00:34, 7682.05it/s] 33%|███▎      | 133598/400000 [00:16<00:34, 7696.26it/s] 34%|███▎      | 134369/400000 [00:16<00:34, 7665.32it/s] 34%|███▍      | 135168/400000 [00:17<00:34, 7758.08it/s] 34%|███▍      | 135989/400000 [00:17<00:33, 7887.49it/s] 34%|███▍      | 136789/400000 [00:17<00:33, 7918.91it/s] 34%|███▍      | 137605/400000 [00:17<00:32, 7988.38it/s] 35%|███▍      | 138430/400000 [00:17<00:32, 8063.84it/s] 35%|███▍      | 139238/400000 [00:17<00:32, 8033.58it/s] 35%|███▌      | 140058/400000 [00:17<00:32, 8081.48it/s] 35%|███▌      | 140868/400000 [00:17<00:32, 8085.75it/s] 35%|███▌      | 141686/400000 [00:17<00:31, 8112.85it/s] 36%|███▌      | 142501/400000 [00:17<00:31, 8122.90it/s] 36%|███▌      | 143314/400000 [00:18<00:31, 8095.67it/s] 36%|███▌      | 144124/400000 [00:18<00:31, 8075.75it/s] 36%|███▌      | 144932/400000 [00:18<00:31, 7986.39it/s] 36%|███▋      | 145751/400000 [00:18<00:31, 8046.23it/s] 37%|███▋      | 146563/400000 [00:18<00:31, 8065.46it/s] 37%|███▋      | 147370/400000 [00:18<00:31, 8026.66it/s] 37%|███▋      | 148184/400000 [00:18<00:31, 8058.24it/s] 37%|███▋      | 148999/400000 [00:18<00:31, 8083.38it/s] 37%|███▋      | 149821/400000 [00:18<00:30, 8123.50it/s] 38%|███▊      | 150641/400000 [00:18<00:30, 8145.52it/s] 38%|███▊      | 151456/400000 [00:19<00:30, 8113.29it/s] 38%|███▊      | 152283/400000 [00:19<00:30, 8157.08it/s] 38%|███▊      | 153104/400000 [00:19<00:30, 8171.03it/s] 38%|███▊      | 153926/400000 [00:19<00:30, 8184.43it/s] 39%|███▊      | 154745/400000 [00:19<00:30, 8145.13it/s] 39%|███▉      | 155560/400000 [00:19<00:30, 8105.02it/s] 39%|███▉      | 156371/400000 [00:19<00:30, 8097.40it/s] 39%|███▉      | 157181/400000 [00:19<00:30, 8052.30it/s] 39%|███▉      | 157995/400000 [00:19<00:29, 8077.18it/s] 40%|███▉      | 158804/400000 [00:19<00:29, 8078.91it/s] 40%|███▉      | 159622/400000 [00:20<00:29, 8106.81it/s] 40%|████      | 160443/400000 [00:20<00:29, 8134.70it/s] 40%|████      | 161257/400000 [00:20<00:29, 8038.20it/s] 41%|████      | 162077/400000 [00:20<00:29, 8085.53it/s] 41%|████      | 162889/400000 [00:20<00:29, 8094.93it/s] 41%|████      | 163712/400000 [00:20<00:29, 8133.46it/s] 41%|████      | 164536/400000 [00:20<00:28, 8164.14it/s] 41%|████▏     | 165360/400000 [00:20<00:28, 8184.64it/s] 42%|████▏     | 166179/400000 [00:20<00:28, 8082.75it/s] 42%|████▏     | 166999/400000 [00:21<00:28, 8117.03it/s] 42%|████▏     | 167811/400000 [00:21<00:28, 8060.75it/s] 42%|████▏     | 168618/400000 [00:21<00:28, 8056.33it/s] 42%|████▏     | 169424/400000 [00:21<00:30, 7680.72it/s] 43%|████▎     | 170216/400000 [00:21<00:29, 7748.66it/s] 43%|████▎     | 171029/400000 [00:21<00:29, 7858.65it/s] 43%|████▎     | 171823/400000 [00:21<00:28, 7882.07it/s] 43%|████▎     | 172618/400000 [00:21<00:28, 7898.91it/s] 43%|████▎     | 173436/400000 [00:21<00:28, 7980.88it/s] 44%|████▎     | 174257/400000 [00:21<00:28, 8047.56it/s] 44%|████▍     | 175067/400000 [00:22<00:27, 8062.09it/s] 44%|████▍     | 175874/400000 [00:22<00:27, 8051.68it/s] 44%|████▍     | 176697/400000 [00:22<00:27, 8104.01it/s] 44%|████▍     | 177508/400000 [00:22<00:27, 8081.71it/s] 45%|████▍     | 178317/400000 [00:22<00:27, 8007.70it/s] 45%|████▍     | 179119/400000 [00:22<00:28, 7812.54it/s] 45%|████▍     | 179924/400000 [00:22<00:27, 7881.66it/s] 45%|████▌     | 180733/400000 [00:22<00:27, 7942.02it/s] 45%|████▌     | 181532/400000 [00:22<00:27, 7955.88it/s] 46%|████▌     | 182344/400000 [00:22<00:27, 8001.88it/s] 46%|████▌     | 183159/400000 [00:23<00:26, 8044.59it/s] 46%|████▌     | 183966/400000 [00:23<00:26, 8048.67it/s] 46%|████▌     | 184772/400000 [00:23<00:27, 7890.94it/s] 46%|████▋     | 185562/400000 [00:23<00:28, 7591.40it/s] 47%|████▋     | 186334/400000 [00:23<00:28, 7627.15it/s] 47%|████▋     | 187147/400000 [00:23<00:27, 7769.66it/s] 47%|████▋     | 187964/400000 [00:23<00:26, 7883.82it/s] 47%|████▋     | 188784/400000 [00:23<00:26, 7973.85it/s] 47%|████▋     | 189595/400000 [00:23<00:26, 8012.07it/s] 48%|████▊     | 190415/400000 [00:23<00:25, 8067.00it/s] 48%|████▊     | 191234/400000 [00:24<00:25, 8102.13it/s] 48%|████▊     | 192048/400000 [00:24<00:25, 8112.31it/s] 48%|████▊     | 192866/400000 [00:24<00:25, 8130.01it/s] 48%|████▊     | 193680/400000 [00:24<00:25, 8127.70it/s] 49%|████▊     | 194497/400000 [00:24<00:25, 8136.55it/s] 49%|████▉     | 195314/400000 [00:24<00:25, 8145.49it/s] 49%|████▉     | 196129/400000 [00:24<00:25, 8136.72it/s] 49%|████▉     | 196943/400000 [00:24<00:25, 8120.60it/s] 49%|████▉     | 197756/400000 [00:24<00:25, 7959.30it/s] 50%|████▉     | 198553/400000 [00:24<00:25, 7830.44it/s] 50%|████▉     | 199338/400000 [00:25<00:25, 7781.82it/s] 50%|█████     | 200117/400000 [00:25<00:25, 7752.01it/s] 50%|█████     | 200893/400000 [00:25<00:25, 7741.58it/s] 50%|█████     | 201668/400000 [00:25<00:25, 7716.68it/s] 51%|█████     | 202440/400000 [00:25<00:25, 7702.70it/s] 51%|█████     | 203211/400000 [00:25<00:25, 7673.84it/s] 51%|█████     | 203981/400000 [00:25<00:25, 7680.67it/s] 51%|█████     | 204752/400000 [00:25<00:25, 7689.39it/s] 51%|█████▏    | 205522/400000 [00:25<00:25, 7642.86it/s] 52%|█████▏    | 206329/400000 [00:25<00:24, 7763.99it/s] 52%|█████▏    | 207149/400000 [00:26<00:24, 7888.45it/s] 52%|█████▏    | 207949/400000 [00:26<00:24, 7918.05it/s] 52%|█████▏    | 208742/400000 [00:26<00:24, 7817.41it/s] 52%|█████▏    | 209525/400000 [00:26<00:24, 7753.58it/s] 53%|█████▎    | 210301/400000 [00:26<00:24, 7747.91it/s] 53%|█████▎    | 211077/400000 [00:26<00:24, 7699.71it/s] 53%|█████▎    | 211848/400000 [00:26<00:24, 7612.60it/s] 53%|█████▎    | 212624/400000 [00:26<00:24, 7654.84it/s] 53%|█████▎    | 213391/400000 [00:26<00:24, 7657.55it/s] 54%|█████▎    | 214159/400000 [00:26<00:24, 7663.60it/s] 54%|█████▎    | 214933/400000 [00:27<00:24, 7685.60it/s] 54%|█████▍    | 215723/400000 [00:27<00:23, 7746.63it/s] 54%|█████▍    | 216534/400000 [00:27<00:23, 7850.36it/s] 54%|█████▍    | 217332/400000 [00:27<00:23, 7888.13it/s] 55%|█████▍    | 218134/400000 [00:27<00:22, 7925.39it/s] 55%|█████▍    | 218927/400000 [00:27<00:23, 7809.55it/s] 55%|█████▍    | 219709/400000 [00:27<00:23, 7757.22it/s] 55%|█████▌    | 220486/400000 [00:27<00:23, 7708.85it/s] 55%|█████▌    | 221258/400000 [00:27<00:23, 7701.25it/s] 56%|█████▌    | 222052/400000 [00:28<00:22, 7769.77it/s] 56%|█████▌    | 222863/400000 [00:28<00:22, 7868.64it/s] 56%|█████▌    | 223678/400000 [00:28<00:22, 7949.59it/s] 56%|█████▌    | 224476/400000 [00:28<00:22, 7957.57it/s] 56%|█████▋    | 225273/400000 [00:28<00:22, 7892.28it/s] 57%|█████▋    | 226063/400000 [00:28<00:22, 7600.01it/s] 57%|█████▋    | 226842/400000 [00:28<00:22, 7654.32it/s] 57%|█████▋    | 227634/400000 [00:28<00:22, 7731.22it/s] 57%|█████▋    | 228409/400000 [00:28<00:22, 7665.51it/s] 57%|█████▋    | 229177/400000 [00:28<00:22, 7584.07it/s] 57%|█████▋    | 229999/400000 [00:29<00:21, 7761.73it/s] 58%|█████▊    | 230777/400000 [00:29<00:21, 7753.56it/s] 58%|█████▊    | 231554/400000 [00:29<00:21, 7737.50it/s] 58%|█████▊    | 232329/400000 [00:29<00:21, 7676.00it/s] 58%|█████▊    | 233098/400000 [00:29<00:21, 7650.41it/s] 58%|█████▊    | 233870/400000 [00:29<00:21, 7670.39it/s] 59%|█████▊    | 234646/400000 [00:29<00:21, 7696.69it/s] 59%|█████▉    | 235416/400000 [00:29<00:21, 7683.94it/s] 59%|█████▉    | 236185/400000 [00:29<00:21, 7683.32it/s] 59%|█████▉    | 236964/400000 [00:29<00:21, 7713.55it/s] 59%|█████▉    | 237754/400000 [00:30<00:20, 7766.91it/s] 60%|█████▉    | 238572/400000 [00:30<00:20, 7885.94it/s] 60%|█████▉    | 239383/400000 [00:30<00:20, 7949.14it/s] 60%|██████    | 240196/400000 [00:30<00:19, 8001.36it/s] 60%|██████    | 241006/400000 [00:30<00:19, 8029.34it/s] 60%|██████    | 241810/400000 [00:30<00:19, 7999.07it/s] 61%|██████    | 242611/400000 [00:30<00:19, 7906.03it/s] 61%|██████    | 243403/400000 [00:30<00:19, 7863.04it/s] 61%|██████    | 244190/400000 [00:30<00:19, 7834.48it/s] 61%|██████    | 244974/400000 [00:30<00:19, 7753.93it/s] 61%|██████▏   | 245750/400000 [00:31<00:19, 7745.60it/s] 62%|██████▏   | 246527/400000 [00:31<00:19, 7750.75it/s] 62%|██████▏   | 247303/400000 [00:31<00:19, 7752.59it/s] 62%|██████▏   | 248080/400000 [00:31<00:19, 7755.94it/s] 62%|██████▏   | 248856/400000 [00:31<00:19, 7742.58it/s] 62%|██████▏   | 249631/400000 [00:31<00:19, 7640.00it/s] 63%|██████▎   | 250396/400000 [00:31<00:19, 7562.85it/s] 63%|██████▎   | 251156/400000 [00:31<00:19, 7571.94it/s] 63%|██████▎   | 251914/400000 [00:31<00:19, 7502.40it/s] 63%|██████▎   | 252665/400000 [00:31<00:19, 7499.61it/s] 63%|██████▎   | 253474/400000 [00:32<00:19, 7666.87it/s] 64%|██████▎   | 254297/400000 [00:32<00:18, 7825.76it/s] 64%|██████▍   | 255082/400000 [00:32<00:18, 7823.65it/s] 64%|██████▍   | 255866/400000 [00:32<00:18, 7766.71it/s] 64%|██████▍   | 256644/400000 [00:32<00:18, 7744.15it/s] 64%|██████▍   | 257420/400000 [00:32<00:18, 7737.81it/s] 65%|██████▍   | 258198/400000 [00:32<00:18, 7749.07it/s] 65%|██████▍   | 258974/400000 [00:32<00:18, 7746.77it/s] 65%|██████▍   | 259751/400000 [00:32<00:18, 7752.85it/s] 65%|██████▌   | 260527/400000 [00:32<00:18, 7719.54it/s] 65%|██████▌   | 261300/400000 [00:33<00:17, 7711.69it/s] 66%|██████▌   | 262084/400000 [00:33<00:17, 7748.70it/s] 66%|██████▌   | 262879/400000 [00:33<00:17, 7805.77it/s] 66%|██████▌   | 263685/400000 [00:33<00:17, 7880.14it/s] 66%|██████▌   | 264495/400000 [00:33<00:17, 7942.20it/s] 66%|██████▋   | 265290/400000 [00:33<00:16, 7926.08it/s] 67%|██████▋   | 266083/400000 [00:33<00:17, 7857.86it/s] 67%|██████▋   | 266870/400000 [00:33<00:17, 7598.16it/s] 67%|██████▋   | 267635/400000 [00:33<00:17, 7612.64it/s] 67%|██████▋   | 268406/400000 [00:33<00:17, 7638.63it/s] 67%|██████▋   | 269171/400000 [00:34<00:17, 7516.87it/s] 67%|██████▋   | 269973/400000 [00:34<00:16, 7659.47it/s] 68%|██████▊   | 270789/400000 [00:34<00:16, 7801.00it/s] 68%|██████▊   | 271600/400000 [00:34<00:16, 7888.79it/s] 68%|██████▊   | 272412/400000 [00:34<00:16, 7954.29it/s] 68%|██████▊   | 273209/400000 [00:34<00:16, 7853.06it/s] 68%|██████▊   | 273996/400000 [00:34<00:16, 7829.41it/s] 69%|██████▊   | 274780/400000 [00:34<00:16, 7802.40it/s] 69%|██████▉   | 275561/400000 [00:34<00:16, 7768.92it/s] 69%|██████▉   | 276339/400000 [00:34<00:15, 7740.10it/s] 69%|██████▉   | 277114/400000 [00:35<00:15, 7695.73it/s] 69%|██████▉   | 277884/400000 [00:35<00:15, 7661.75it/s] 70%|██████▉   | 278661/400000 [00:35<00:15, 7693.43it/s] 70%|██████▉   | 279439/400000 [00:35<00:15, 7718.07it/s] 70%|███████   | 280244/400000 [00:35<00:15, 7813.22it/s] 70%|███████   | 281058/400000 [00:35<00:15, 7905.75it/s] 70%|███████   | 281850/400000 [00:35<00:15, 7856.50it/s] 71%|███████   | 282637/400000 [00:35<00:15, 7812.01it/s] 71%|███████   | 283419/400000 [00:35<00:15, 7578.01it/s] 71%|███████   | 284179/400000 [00:36<00:15, 7558.03it/s] 71%|███████   | 284947/400000 [00:36<00:15, 7591.85it/s] 71%|███████▏  | 285708/400000 [00:36<00:15, 7597.08it/s] 72%|███████▏  | 286521/400000 [00:36<00:14, 7747.50it/s] 72%|███████▏  | 287336/400000 [00:36<00:14, 7863.13it/s] 72%|███████▏  | 288143/400000 [00:36<00:14, 7922.10it/s] 72%|███████▏  | 288946/400000 [00:36<00:13, 7951.86it/s] 72%|███████▏  | 289742/400000 [00:36<00:14, 7758.91it/s] 73%|███████▎  | 290520/400000 [00:36<00:14, 7616.75it/s] 73%|███████▎  | 291335/400000 [00:36<00:13, 7769.02it/s] 73%|███████▎  | 292145/400000 [00:37<00:13, 7863.88it/s] 73%|███████▎  | 292959/400000 [00:37<00:13, 7943.20it/s] 73%|███████▎  | 293775/400000 [00:37<00:13, 8005.65it/s] 74%|███████▎  | 294589/400000 [00:37<00:13, 8044.57it/s] 74%|███████▍  | 295405/400000 [00:37<00:12, 8078.47it/s] 74%|███████▍  | 296214/400000 [00:37<00:12, 8062.88it/s] 74%|███████▍  | 297021/400000 [00:37<00:12, 7983.44it/s] 74%|███████▍  | 297832/400000 [00:37<00:12, 8020.91it/s] 75%|███████▍  | 298635/400000 [00:37<00:12, 8009.11it/s] 75%|███████▍  | 299452/400000 [00:37<00:12, 8054.79it/s] 75%|███████▌  | 300269/400000 [00:38<00:12, 8087.61it/s] 75%|███████▌  | 301078/400000 [00:38<00:12, 8038.57it/s] 75%|███████▌  | 301891/400000 [00:38<00:12, 8064.03it/s] 76%|███████▌  | 302702/400000 [00:38<00:12, 8074.83it/s] 76%|███████▌  | 303510/400000 [00:38<00:12, 7968.84it/s] 76%|███████▌  | 304308/400000 [00:38<00:12, 7894.44it/s] 76%|███████▋  | 305098/400000 [00:38<00:12, 7826.63it/s] 76%|███████▋  | 305882/400000 [00:38<00:12, 7798.61it/s] 77%|███████▋  | 306663/400000 [00:38<00:11, 7783.92it/s] 77%|███████▋  | 307442/400000 [00:38<00:11, 7780.79it/s] 77%|███████▋  | 308221/400000 [00:39<00:11, 7735.92it/s] 77%|███████▋  | 308995/400000 [00:39<00:11, 7716.91it/s] 77%|███████▋  | 309802/400000 [00:39<00:11, 7819.07it/s] 78%|███████▊  | 310601/400000 [00:39<00:11, 7869.32it/s] 78%|███████▊  | 311389/400000 [00:39<00:11, 7822.94it/s] 78%|███████▊  | 312172/400000 [00:39<00:11, 7775.04it/s] 78%|███████▊  | 312950/400000 [00:39<00:11, 7760.76it/s] 78%|███████▊  | 313727/400000 [00:39<00:11, 7752.16it/s] 79%|███████▊  | 314503/400000 [00:39<00:11, 7714.19it/s] 79%|███████▉  | 315278/400000 [00:39<00:10, 7722.52it/s] 79%|███████▉  | 316051/400000 [00:40<00:10, 7703.89it/s] 79%|███████▉  | 316822/400000 [00:40<00:10, 7676.02it/s] 79%|███████▉  | 317595/400000 [00:40<00:10, 7689.90it/s] 80%|███████▉  | 318368/400000 [00:40<00:10, 7699.75it/s] 80%|███████▉  | 319143/400000 [00:40<00:10, 7711.85it/s] 80%|███████▉  | 319915/400000 [00:40<00:10, 7665.32it/s] 80%|████████  | 320713/400000 [00:40<00:10, 7755.05it/s] 80%|████████  | 321511/400000 [00:40<00:10, 7821.17it/s] 81%|████████  | 322331/400000 [00:40<00:09, 7929.02it/s] 81%|████████  | 323155/400000 [00:40<00:09, 8017.91it/s] 81%|████████  | 323980/400000 [00:41<00:09, 8083.35it/s] 81%|████████  | 324789/400000 [00:41<00:09, 8035.19it/s] 81%|████████▏ | 325600/400000 [00:41<00:09, 8056.72it/s] 82%|████████▏ | 326414/400000 [00:41<00:09, 8078.84it/s] 82%|████████▏ | 327223/400000 [00:41<00:09, 8039.27it/s] 82%|████████▏ | 328041/400000 [00:41<00:08, 8078.09it/s] 82%|████████▏ | 328850/400000 [00:41<00:08, 8016.30it/s] 82%|████████▏ | 329652/400000 [00:41<00:08, 7858.84it/s] 83%|████████▎ | 330439/400000 [00:41<00:08, 7849.13it/s] 83%|████████▎ | 331248/400000 [00:41<00:08, 7918.42it/s] 83%|████████▎ | 332057/400000 [00:42<00:08, 7966.86it/s] 83%|████████▎ | 332863/400000 [00:42<00:08, 7994.55it/s] 83%|████████▎ | 333682/400000 [00:42<00:08, 8050.88it/s] 84%|████████▎ | 334499/400000 [00:42<00:08, 8082.85it/s] 84%|████████▍ | 335316/400000 [00:42<00:07, 8107.79it/s] 84%|████████▍ | 336127/400000 [00:42<00:08, 7970.21it/s] 84%|████████▍ | 336925/400000 [00:42<00:08, 7873.67it/s] 84%|████████▍ | 337714/400000 [00:42<00:07, 7822.40it/s] 85%|████████▍ | 338497/400000 [00:42<00:07, 7804.14it/s] 85%|████████▍ | 339278/400000 [00:42<00:07, 7771.91it/s] 85%|████████▌ | 340056/400000 [00:43<00:07, 7769.77it/s] 85%|████████▌ | 340834/400000 [00:43<00:07, 7687.24it/s] 85%|████████▌ | 341615/400000 [00:43<00:07, 7721.16it/s] 86%|████████▌ | 342388/400000 [00:43<00:07, 7694.63it/s] 86%|████████▌ | 343158/400000 [00:43<00:07, 7669.12it/s] 86%|████████▌ | 343926/400000 [00:43<00:07, 7573.85it/s] 86%|████████▌ | 344698/400000 [00:43<00:07, 7614.44it/s] 86%|████████▋ | 345495/400000 [00:43<00:07, 7715.70it/s] 87%|████████▋ | 346310/400000 [00:43<00:06, 7840.00it/s] 87%|████████▋ | 347127/400000 [00:44<00:06, 7935.87it/s] 87%|████████▋ | 347939/400000 [00:44<00:06, 7987.77it/s] 87%|████████▋ | 348739/400000 [00:44<00:06, 7881.98it/s] 87%|████████▋ | 349529/400000 [00:44<00:06, 7755.52it/s] 88%|████████▊ | 350306/400000 [00:44<00:06, 7726.15it/s] 88%|████████▊ | 351086/400000 [00:44<00:06, 7745.60it/s] 88%|████████▊ | 351862/400000 [00:44<00:06, 7695.79it/s] 88%|████████▊ | 352644/400000 [00:44<00:06, 7732.52it/s] 88%|████████▊ | 353459/400000 [00:44<00:05, 7851.87it/s] 89%|████████▊ | 354268/400000 [00:44<00:05, 7921.44it/s] 89%|████████▉ | 355061/400000 [00:45<00:05, 7848.61it/s] 89%|████████▉ | 355847/400000 [00:45<00:05, 7794.73it/s] 89%|████████▉ | 356627/400000 [00:45<00:05, 7776.77it/s] 89%|████████▉ | 357406/400000 [00:45<00:05, 7721.07it/s] 90%|████████▉ | 358179/400000 [00:45<00:05, 7716.34it/s] 90%|████████▉ | 358954/400000 [00:45<00:05, 7725.13it/s] 90%|████████▉ | 359727/400000 [00:45<00:05, 7726.42it/s] 90%|█████████ | 360519/400000 [00:45<00:05, 7781.21it/s] 90%|█████████ | 361341/400000 [00:45<00:04, 7906.47it/s] 91%|█████████ | 362133/400000 [00:45<00:04, 7704.28it/s] 91%|█████████ | 362908/400000 [00:46<00:04, 7715.09it/s] 91%|█████████ | 363681/400000 [00:46<00:04, 7623.92it/s] 91%|█████████ | 364461/400000 [00:46<00:04, 7675.49it/s] 91%|█████████▏| 365267/400000 [00:46<00:04, 7786.29it/s] 92%|█████████▏| 366081/400000 [00:46<00:04, 7888.33it/s] 92%|█████████▏| 366885/400000 [00:46<00:04, 7931.47it/s] 92%|█████████▏| 367679/400000 [00:46<00:04, 7873.40it/s] 92%|█████████▏| 368467/400000 [00:46<00:04, 7833.13it/s] 92%|█████████▏| 369251/400000 [00:46<00:03, 7805.31it/s] 93%|█████████▎| 370032/400000 [00:46<00:03, 7772.65it/s] 93%|█████████▎| 370827/400000 [00:47<00:03, 7823.06it/s] 93%|█████████▎| 371634/400000 [00:47<00:03, 7894.48it/s] 93%|█████████▎| 372424/400000 [00:47<00:03, 7866.89it/s] 93%|█████████▎| 373211/400000 [00:47<00:03, 7803.98it/s] 94%|█████████▎| 374030/400000 [00:47<00:03, 7915.00it/s] 94%|█████████▎| 374845/400000 [00:47<00:03, 7982.20it/s] 94%|█████████▍| 375648/400000 [00:47<00:03, 7995.08it/s] 94%|█████████▍| 376448/400000 [00:47<00:02, 7904.46it/s] 94%|█████████▍| 377239/400000 [00:47<00:02, 7863.10it/s] 95%|█████████▍| 378026/400000 [00:47<00:02, 7821.44it/s] 95%|█████████▍| 378809/400000 [00:48<00:02, 7605.39it/s] 95%|█████████▍| 379574/400000 [00:48<00:02, 7617.72it/s] 95%|█████████▌| 380337/400000 [00:48<00:02, 7453.03it/s] 95%|█████████▌| 381148/400000 [00:48<00:02, 7635.84it/s] 95%|█████████▌| 381960/400000 [00:48<00:02, 7773.12it/s] 96%|█████████▌| 382773/400000 [00:48<00:02, 7875.51it/s] 96%|█████████▌| 383575/400000 [00:48<00:02, 7918.16it/s] 96%|█████████▌| 384386/400000 [00:48<00:01, 7973.69it/s] 96%|█████████▋| 385198/400000 [00:48<00:01, 8015.71it/s] 97%|█████████▋| 386015/400000 [00:48<00:01, 8061.02it/s] 97%|█████████▋| 386838/400000 [00:49<00:01, 8109.43it/s] 97%|█████████▋| 387650/400000 [00:49<00:01, 8101.91it/s] 97%|█████████▋| 388461/400000 [00:49<00:01, 8049.56it/s] 97%|█████████▋| 389267/400000 [00:49<00:01, 7959.80it/s] 98%|█████████▊| 390064/400000 [00:49<00:01, 7859.43it/s] 98%|█████████▊| 390851/400000 [00:49<00:01, 7755.76it/s] 98%|█████████▊| 391657/400000 [00:49<00:01, 7842.86it/s] 98%|█████████▊| 392464/400000 [00:49<00:00, 7908.41it/s] 98%|█████████▊| 393270/400000 [00:49<00:00, 7951.81it/s] 99%|█████████▊| 394066/400000 [00:50<00:00, 7915.06it/s] 99%|█████████▊| 394858/400000 [00:50<00:00, 7807.86it/s] 99%|█████████▉| 395640/400000 [00:50<00:00, 7776.99it/s] 99%|█████████▉| 396419/400000 [00:50<00:00, 7695.65it/s] 99%|█████████▉| 397237/400000 [00:50<00:00, 7833.56it/s]100%|█████████▉| 398059/400000 [00:50<00:00, 7943.71it/s]100%|█████████▉| 398883/400000 [00:50<00:00, 8027.96it/s]100%|█████████▉| 399687/400000 [00:50<00:00, 8018.92it/s]100%|█████████▉| 399999/400000 [00:50<00:00, 7881.70it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f5cb9a200f0>, <torchtext.data.dataset.TabularDataset object at 0x7f5cb9a20208>, <torchtext.vocab.Vocab object at 0x7f5dae0efba8>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 29.44 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 46.61 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 24.31 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 24.31 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.05 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.05 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.89 file/s]2020-06-17 15:10:43.422759: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-06-17 15:10:43.427256: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-06-17 15:10:43.427602: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55cd30f4d950 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-17 15:10:43.427629: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
[1mDownloading and preparing dataset mnist/3.0.1 (download: 11.06 MiB, generated: 21.00 MiB, total: 32.06 MiB) to /home/runner/tensorflow_datasets/mnist/3.0.1...[0m

[1mDataset mnist downloaded and prepared to /home/runner/tensorflow_datasets/mnist/3.0.1. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/ ['train', 'cifar10', 'mnist_dataset_small.npy', 'test', 'fashion-mnist_small.npy', 'mnist2'] 

  


 #################### get_dataset_torch 

  get_dataset_torch mlmodels/preprocess/generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:14, 133290.16it/s] 69%|██████▊   | 6807552/9912422 [00:00<00:16, 190254.25it/s]9920512it [00:00, 38572734.62it/s]                           
0it [00:00, ?it/s]32768it [00:00, 569754.19it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 453475.17it/s]1654784it [00:00, 11885375.63it/s]                         
0it [00:00, ?it/s]8192it [00:00, 189001.62it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
