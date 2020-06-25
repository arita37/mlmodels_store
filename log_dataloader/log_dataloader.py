
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/077ac9573b3255f0836baba55f19fb6dbaa40c9d', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '077ac9573b3255f0836baba55f19fb6dbaa40c9d', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/077ac9573b3255f0836baba55f19fb6dbaa40c9d

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/077ac9573b3255f0836baba55f19fb6dbaa40c9d

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/077ac9573b3255f0836baba55f19fb6dbaa40c9d

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7efcc0079ea0> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7efcc0079ea0> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7efd2b3391e0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7efd2b3391e0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7efd49681e18> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7efd49681e18> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7efcd8662488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7efcd8662488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7efcd8662488> , (data_info, **args) 

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
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/numpy/lib/npyio.py", line 416, in load
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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:09, 142227.47it/s] 34%|███▍      | 3391488/9912422 [00:00<00:32, 202815.60it/s] 98%|█████████▊| 9715712/9912422 [00:00<00:00, 289336.85it/s]9920512it [00:00, 31325674.75it/s]                           
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 254526.48it/s]           
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 473207.28it/s]1654784it [00:00, 11869399.82it/s]                         
0it [00:00, ?it/s]8192it [00:00, 185651.04it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7efcbf78d3c8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7efcbf775668>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7efcd86620d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7efcd86620d0> 

  function with postional parmater data_info <function tf_dataset_download at 0x7efcd86620d0> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:02,  2.58 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:02,  2.58 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:01,  2.58 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   2%|▏         | 3/162 [00:00<00:46,  3.39 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<00:46,  3.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<00:46,  3.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   3%|▎         | 5/162 [00:00<00:36,  4.36 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<00:36,  4.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<00:35,  4.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▍         | 7/162 [00:00<00:28,  5.42 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:28,  5.42 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:28,  5.42 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   6%|▌         | 9/162 [00:01<00:23,  6.53 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:01<00:23,  6.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:01<00:23,  6.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   7%|▋         | 11/162 [00:01<00:19,  7.60 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:01<00:19,  7.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:01<00:19,  7.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   8%|▊         | 13/162 [00:01<00:17,  8.62 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:01<00:17,  8.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:01<00:17,  8.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   9%|▉         | 15/162 [00:01<00:15,  9.46 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:01<00:15,  9.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:01<00:15,  9.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  10%|█         | 17/162 [00:01<00:14, 10.18 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:01<00:14, 10.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:01<00:14, 10.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  12%|█▏        | 19/162 [00:01<00:13, 10.59 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:01<00:13, 10.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:01<00:13, 10.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  13%|█▎        | 21/162 [00:02<00:13, 10.75 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:02<00:13, 10.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:02<00:13, 10.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  14%|█▍        | 23/162 [00:02<00:12, 10.87 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:02<00:12, 10.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:02<00:12, 10.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  15%|█▌        | 25/162 [00:02<00:12, 10.94 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:02<00:12, 10.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:02<00:12, 10.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  17%|█▋        | 27/162 [00:02<00:12, 11.02 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:02<00:12, 11.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:02<00:12, 11.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  18%|█▊        | 29/162 [00:02<00:12, 11.05 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:02<00:12, 11.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:02<00:11, 11.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  19%|█▉        | 31/162 [00:02<00:11, 11.10 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:02<00:11, 11.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:02<00:11, 11.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  20%|██        | 33/162 [00:03<00:11, 11.12 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:03<00:11, 11.12 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:03<00:11, 11.12 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  22%|██▏       | 35/162 [00:03<00:11, 11.21 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:03<00:11, 11.21 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:03<00:11, 11.21 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  23%|██▎       | 37/162 [00:03<00:11, 11.17 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:03<00:11, 11.17 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:03<00:11, 11.17 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  24%|██▍       | 39/162 [00:03<00:11, 11.18 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:03<00:11, 11.18 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:03<00:10, 11.18 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  25%|██▌       | 41/162 [00:03<00:10, 11.13 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:03<00:10, 11.13 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:03<00:10, 11.13 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  27%|██▋       | 43/162 [00:03<00:10, 11.08 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:03<00:10, 11.08 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:04<00:10, 11.08 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  28%|██▊       | 45/162 [00:04<00:10, 11.02 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:04<00:10, 11.02 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:04<00:10, 11.02 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  29%|██▉       | 47/162 [00:04<00:10, 11.03 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:04<00:10, 11.03 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:04<00:10, 11.03 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  30%|███       | 49/162 [00:04<00:10, 10.95 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:04<00:10, 10.95 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:04<00:10, 10.95 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  31%|███▏      | 51/162 [00:04<00:10, 10.88 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:04<00:10, 10.88 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:04<00:10, 10.88 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  33%|███▎      | 53/162 [00:04<00:10, 10.82 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:04<00:10, 10.82 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:05<00:09, 10.82 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  34%|███▍      | 55/162 [00:05<00:10, 10.68 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:05<00:10, 10.68 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:05<00:09, 10.68 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  35%|███▌      | 57/162 [00:05<00:10, 10.42 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:05<00:10, 10.42 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:05<00:09, 10.42 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  36%|███▋      | 59/162 [00:05<00:09, 10.32 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:05<00:09, 10.32 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:05<00:09, 10.32 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  38%|███▊      | 61/162 [00:05<00:09, 10.32 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:05<00:09, 10.32 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:05<00:09, 10.32 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  39%|███▉      | 63/162 [00:05<00:09, 10.31 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:05<00:09, 10.31 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:05<00:09, 10.31 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  40%|████      | 65/162 [00:06<00:09, 10.24 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:06<00:09, 10.24 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:06<00:09, 10.24 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  41%|████▏     | 67/162 [00:06<00:09, 10.23 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:06<00:09, 10.23 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:06<00:09, 10.23 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  43%|████▎     | 69/162 [00:06<00:09, 10.21 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:06<00:09, 10.21 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:06<00:09, 10.21 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  44%|████▍     | 71/162 [00:06<00:08, 10.19 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:06<00:08, 10.19 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:06<00:08, 10.19 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  45%|████▌     | 73/162 [00:06<00:08, 10.17 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:06<00:08, 10.17 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:06<00:08, 10.17 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  46%|████▋     | 75/162 [00:07<00:08, 10.16 MiB/s][ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:07<00:08, 10.16 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:07<00:08, 10.16 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[A
Dl Size...:  48%|████▊     | 77/162 [00:07<00:08, 10.16 MiB/s][ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:07<00:08, 10.16 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:07<00:08, 10.16 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[A
Dl Size...:  49%|████▉     | 79/162 [00:07<00:08, 10.15 MiB/s][ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:07<00:08, 10.15 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:07<00:08, 10.15 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[A
Dl Size...:  50%|█████     | 81/162 [00:07<00:07, 10.14 MiB/s][ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:07<00:07, 10.14 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:07<00:07, 10.14 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[A
Dl Size...:  51%|█████     | 83/162 [00:07<00:07, 10.15 MiB/s][ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:07<00:07, 10.15 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:07<00:07, 10.15 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[A
Dl Size...:  52%|█████▏    | 85/162 [00:08<00:07, 10.14 MiB/s][ADl Completed...:   0%|          | 0/1 [00:08<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:08<00:07, 10.14 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:08<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:08<00:07, 10.14 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[A
Dl Size...:  54%|█████▎    | 87/162 [00:08<00:07, 10.10 MiB/s][ADl Completed...:   0%|          | 0/1 [00:08<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:08<00:07, 10.10 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:08<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:08<00:07, 10.10 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[A
Dl Size...:  55%|█████▍    | 89/162 [00:08<00:07, 10.13 MiB/s][ADl Completed...:   0%|          | 0/1 [00:08<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:08<00:07, 10.13 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:08<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:08<00:07, 10.13 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[A
Dl Size...:  56%|█████▌    | 91/162 [00:08<00:07, 10.13 MiB/s][ADl Completed...:   0%|          | 0/1 [00:08<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:08<00:07, 10.13 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:08<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:08<00:06, 10.13 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[A
Dl Size...:  57%|█████▋    | 93/162 [00:08<00:06, 10.11 MiB/s][ADl Completed...:   0%|          | 0/1 [00:08<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:08<00:06, 10.11 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:08<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:08<00:06, 10.11 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[A
Dl Size...:  59%|█████▊    | 95/162 [00:09<00:06, 10.10 MiB/s][ADl Completed...:   0%|          | 0/1 [00:09<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:09<00:06, 10.10 MiB/s][A

Extraction completed...: 0 file [00:09, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:09<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:09<00:06, 10.10 MiB/s][A

Extraction completed...: 0 file [00:09, ? file/s][A[A
Dl Size...:  60%|█████▉    | 97/162 [00:09<00:06, 10.13 MiB/s][ADl Completed...:   0%|          | 0/1 [00:09<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:09<00:06, 10.13 MiB/s][A

Extraction completed...: 0 file [00:09, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:09<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:09<00:06, 10.13 MiB/s][A

Extraction completed...: 0 file [00:09, ? file/s][A[A
Dl Size...:  61%|██████    | 99/162 [00:09<00:06, 10.12 MiB/s][ADl Completed...:   0%|          | 0/1 [00:09<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:09<00:06, 10.12 MiB/s][A

Extraction completed...: 0 file [00:09, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:09<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:09<00:06, 10.12 MiB/s][A

Extraction completed...: 0 file [00:09, ? file/s][A[A
Dl Size...:  62%|██████▏   | 101/162 [00:09<00:06, 10.14 MiB/s][ADl Completed...:   0%|          | 0/1 [00:09<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:09<00:06, 10.14 MiB/s][A

Extraction completed...: 0 file [00:09, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:09<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:09<00:05, 10.14 MiB/s][A

Extraction completed...: 0 file [00:09, ? file/s][A[A
Dl Size...:  64%|██████▎   | 103/162 [00:09<00:05, 10.15 MiB/s][ADl Completed...:   0%|          | 0/1 [00:09<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:09<00:05, 10.15 MiB/s][A

Extraction completed...: 0 file [00:09, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:09<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:09<00:05, 10.15 MiB/s][A

Extraction completed...: 0 file [00:09, ? file/s][A[A
Dl Size...:  65%|██████▍   | 105/162 [00:10<00:05, 10.13 MiB/s][ADl Completed...:   0%|          | 0/1 [00:10<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:10<00:05, 10.13 MiB/s][A

Extraction completed...: 0 file [00:10, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:10<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:10<00:05, 10.13 MiB/s][A

Extraction completed...: 0 file [00:10, ? file/s][A[A
Dl Size...:  66%|██████▌   | 107/162 [00:10<00:05, 10.13 MiB/s][ADl Completed...:   0%|          | 0/1 [00:10<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:10<00:05, 10.13 MiB/s][A

Extraction completed...: 0 file [00:10, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:10<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:10<00:05, 10.13 MiB/s][A

Extraction completed...: 0 file [00:10, ? file/s][A[A
Dl Size...:  67%|██████▋   | 109/162 [00:10<00:05, 10.13 MiB/s][ADl Completed...:   0%|          | 0/1 [00:10<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:10<00:05, 10.13 MiB/s][A

Extraction completed...: 0 file [00:10, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:10<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:10<00:05, 10.13 MiB/s][A

Extraction completed...: 0 file [00:10, ? file/s][A[A
Dl Size...:  69%|██████▊   | 111/162 [00:10<00:05, 10.11 MiB/s][ADl Completed...:   0%|          | 0/1 [00:10<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:10<00:05, 10.11 MiB/s][A

Extraction completed...: 0 file [00:10, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:10<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:10<00:04, 10.11 MiB/s][A

Extraction completed...: 0 file [00:10, ? file/s][A[A
Dl Size...:  70%|██████▉   | 113/162 [00:10<00:04, 10.12 MiB/s][ADl Completed...:   0%|          | 0/1 [00:10<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:10<00:04, 10.12 MiB/s][A

Extraction completed...: 0 file [00:10, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:10<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:10<00:04, 10.12 MiB/s][A

Extraction completed...: 0 file [00:10, ? file/s][A[A
Dl Size...:  71%|███████   | 115/162 [00:11<00:04, 10.09 MiB/s][ADl Completed...:   0%|          | 0/1 [00:11<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:11<00:04, 10.09 MiB/s][A

Extraction completed...: 0 file [00:11, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:11<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:11<00:04, 10.09 MiB/s][A

Extraction completed...: 0 file [00:11, ? file/s][A[A
Dl Size...:  72%|███████▏  | 117/162 [00:11<00:04, 10.09 MiB/s][ADl Completed...:   0%|          | 0/1 [00:11<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:11<00:04, 10.09 MiB/s][A

Extraction completed...: 0 file [00:11, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:11<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:11<00:04, 10.09 MiB/s][A

Extraction completed...: 0 file [00:11, ? file/s][A[A
Dl Size...:  73%|███████▎  | 119/162 [00:11<00:04, 10.08 MiB/s][ADl Completed...:   0%|          | 0/1 [00:11<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:11<00:04, 10.08 MiB/s][A

Extraction completed...: 0 file [00:11, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:11<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:11<00:04, 10.08 MiB/s][A

Extraction completed...: 0 file [00:11, ? file/s][A[A
Dl Size...:  75%|███████▍  | 121/162 [00:11<00:04, 10.04 MiB/s][ADl Completed...:   0%|          | 0/1 [00:11<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:11<00:04, 10.04 MiB/s][A

Extraction completed...: 0 file [00:11, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:11<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:11<00:03, 10.04 MiB/s][A

Extraction completed...: 0 file [00:11, ? file/s][A[A
Dl Size...:  76%|███████▌  | 123/162 [00:11<00:03, 10.05 MiB/s][ADl Completed...:   0%|          | 0/1 [00:11<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:11<00:03, 10.05 MiB/s][A

Extraction completed...: 0 file [00:11, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:11<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:11<00:03, 10.05 MiB/s][A

Extraction completed...: 0 file [00:11, ? file/s][A[A
Dl Size...:  77%|███████▋  | 125/162 [00:12<00:03, 10.07 MiB/s][ADl Completed...:   0%|          | 0/1 [00:12<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:12<00:03, 10.07 MiB/s][A

Extraction completed...: 0 file [00:12, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:12<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:12<00:03, 10.07 MiB/s][A

Extraction completed...: 0 file [00:12, ? file/s][A[A
Dl Size...:  78%|███████▊  | 127/162 [00:12<00:03, 10.03 MiB/s][ADl Completed...:   0%|          | 0/1 [00:12<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:12<00:03, 10.03 MiB/s][A

Extraction completed...: 0 file [00:12, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:12<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:12<00:03, 10.03 MiB/s][A

Extraction completed...: 0 file [00:12, ? file/s][A[A
Dl Size...:  80%|███████▉  | 129/162 [00:12<00:03, 10.01 MiB/s][ADl Completed...:   0%|          | 0/1 [00:12<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:12<00:03, 10.01 MiB/s][A

Extraction completed...: 0 file [00:12, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:12<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:12<00:03, 10.01 MiB/s][A

Extraction completed...: 0 file [00:12, ? file/s][A[A
Dl Size...:  81%|████████  | 131/162 [00:12<00:03, 10.02 MiB/s][ADl Completed...:   0%|          | 0/1 [00:12<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:12<00:03, 10.02 MiB/s][A

Extraction completed...: 0 file [00:12, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:12<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:12<00:02, 10.02 MiB/s][A

Extraction completed...: 0 file [00:12, ? file/s][A[A
Dl Size...:  82%|████████▏ | 133/162 [00:12<00:02, 10.05 MiB/s][ADl Completed...:   0%|          | 0/1 [00:12<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:12<00:02, 10.05 MiB/s][A

Extraction completed...: 0 file [00:12, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:12<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:12<00:02, 10.05 MiB/s][A

Extraction completed...: 0 file [00:12, ? file/s][A[A
Dl Size...:  83%|████████▎ | 135/162 [00:13<00:02, 10.09 MiB/s][ADl Completed...:   0%|          | 0/1 [00:13<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:13<00:02, 10.09 MiB/s][A

Extraction completed...: 0 file [00:13, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:13<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:13<00:02, 10.09 MiB/s][A

Extraction completed...: 0 file [00:13, ? file/s][A[A
Dl Size...:  85%|████████▍ | 137/162 [00:13<00:02,  9.99 MiB/s][ADl Completed...:   0%|          | 0/1 [00:13<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:13<00:02,  9.99 MiB/s][A

Extraction completed...: 0 file [00:13, ? file/s][A[A
Dl Size...:  85%|████████▌ | 138/162 [00:13<00:02,  9.83 MiB/s][ADl Completed...:   0%|          | 0/1 [00:13<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:13<00:02,  9.83 MiB/s][A

Extraction completed...: 0 file [00:13, ? file/s][A[A
Dl Size...:  86%|████████▌ | 139/162 [00:13<00:02,  9.52 MiB/s][ADl Completed...:   0%|          | 0/1 [00:13<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:13<00:02,  9.52 MiB/s][A

Extraction completed...: 0 file [00:13, ? file/s][A[A
Dl Size...:  86%|████████▋ | 140/162 [00:13<00:02,  9.47 MiB/s][ADl Completed...:   0%|          | 0/1 [00:13<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:13<00:02,  9.47 MiB/s][A

Extraction completed...: 0 file [00:13, ? file/s][A[A
Dl Size...:  87%|████████▋ | 141/162 [00:13<00:02,  9.42 MiB/s][ADl Completed...:   0%|          | 0/1 [00:13<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:13<00:02,  9.42 MiB/s][A

Extraction completed...: 0 file [00:13, ? file/s][A[A
Dl Size...:  88%|████████▊ | 142/162 [00:13<00:02,  9.36 MiB/s][ADl Completed...:   0%|          | 0/1 [00:13<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:13<00:02,  9.36 MiB/s][A

Extraction completed...: 0 file [00:13, ? file/s][A[A
Dl Size...:  88%|████████▊ | 143/162 [00:13<00:02,  9.28 MiB/s][ADl Completed...:   0%|          | 0/1 [00:13<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:13<00:02,  9.28 MiB/s][A

Extraction completed...: 0 file [00:13, ? file/s][A[A
Dl Size...:  89%|████████▉ | 144/162 [00:13<00:01,  9.30 MiB/s][ADl Completed...:   0%|          | 0/1 [00:13<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:13<00:01,  9.30 MiB/s][A

Extraction completed...: 0 file [00:13, ? file/s][A[A
Dl Size...:  90%|████████▉ | 145/162 [00:14<00:01,  9.33 MiB/s][ADl Completed...:   0%|          | 0/1 [00:14<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:14<00:01,  9.33 MiB/s][A

Extraction completed...: 0 file [00:14, ? file/s][A[A
Dl Size...:  90%|█████████ | 146/162 [00:14<00:01,  9.32 MiB/s][ADl Completed...:   0%|          | 0/1 [00:14<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:14<00:01,  9.32 MiB/s][A

Extraction completed...: 0 file [00:14, ? file/s][A[A
Dl Size...:  91%|█████████ | 147/162 [00:14<00:01,  9.36 MiB/s][ADl Completed...:   0%|          | 0/1 [00:14<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:14<00:01,  9.36 MiB/s][A

Extraction completed...: 0 file [00:14, ? file/s][A[A
Dl Size...:  91%|█████████▏| 148/162 [00:14<00:01,  9.17 MiB/s][ADl Completed...:   0%|          | 0/1 [00:14<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:14<00:01,  9.17 MiB/s][A

Extraction completed...: 0 file [00:14, ? file/s][A[A
Dl Size...:  92%|█████████▏| 149/162 [00:14<00:01,  9.34 MiB/s][ADl Completed...:   0%|          | 0/1 [00:14<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:14<00:01,  9.34 MiB/s][A

Extraction completed...: 0 file [00:14, ? file/s][A[A
Dl Size...:  93%|█████████▎| 150/162 [00:14<00:01,  9.38 MiB/s][ADl Completed...:   0%|          | 0/1 [00:14<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:14<00:01,  9.38 MiB/s][A

Extraction completed...: 0 file [00:14, ? file/s][A[A
Dl Size...:  93%|█████████▎| 151/162 [00:14<00:01,  9.29 MiB/s][ADl Completed...:   0%|          | 0/1 [00:14<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:14<00:01,  9.29 MiB/s][A

Extraction completed...: 0 file [00:14, ? file/s][A[A
Dl Size...:  94%|█████████▍| 152/162 [00:14<00:01,  9.36 MiB/s][ADl Completed...:   0%|          | 0/1 [00:14<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:14<00:01,  9.36 MiB/s][A

Extraction completed...: 0 file [00:14, ? file/s][A[A
Dl Size...:  94%|█████████▍| 153/162 [00:14<00:00,  9.34 MiB/s][ADl Completed...:   0%|          | 0/1 [00:14<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:14<00:00,  9.34 MiB/s][A

Extraction completed...: 0 file [00:14, ? file/s][A[A
Dl Size...:  95%|█████████▌| 154/162 [00:15<00:00,  9.26 MiB/s][ADl Completed...:   0%|          | 0/1 [00:15<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:15<00:00,  9.26 MiB/s][A

Extraction completed...: 0 file [00:15, ? file/s][A[A
Dl Size...:  96%|█████████▌| 155/162 [00:15<00:00,  9.27 MiB/s][ADl Completed...:   0%|          | 0/1 [00:15<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:15<00:00,  9.27 MiB/s][A

Extraction completed...: 0 file [00:15, ? file/s][A[A
Dl Size...:  96%|█████████▋| 156/162 [00:15<00:00,  9.27 MiB/s][ADl Completed...:   0%|          | 0/1 [00:15<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:15<00:00,  9.27 MiB/s][A

Extraction completed...: 0 file [00:15, ? file/s][A[A
Dl Size...:  97%|█████████▋| 157/162 [00:15<00:00,  9.23 MiB/s][ADl Completed...:   0%|          | 0/1 [00:15<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:15<00:00,  9.23 MiB/s][A

Extraction completed...: 0 file [00:15, ? file/s][A[A
Dl Size...:  98%|█████████▊| 158/162 [00:15<00:00,  9.26 MiB/s][ADl Completed...:   0%|          | 0/1 [00:15<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:15<00:00,  9.26 MiB/s][A

Extraction completed...: 0 file [00:15, ? file/s][A[A
Dl Size...:  98%|█████████▊| 159/162 [00:15<00:00,  9.15 MiB/s][ADl Completed...:   0%|          | 0/1 [00:15<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:15<00:00,  9.15 MiB/s][A

Extraction completed...: 0 file [00:15, ? file/s][A[A
Dl Size...:  99%|█████████▉| 160/162 [00:15<00:00,  9.17 MiB/s][ADl Completed...:   0%|          | 0/1 [00:15<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:15<00:00,  9.17 MiB/s][A

Extraction completed...: 0 file [00:15, ? file/s][A[A
Dl Size...:  99%|█████████▉| 161/162 [00:15<00:00,  9.25 MiB/s][ADl Completed...:   0%|          | 0/1 [00:15<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:15<00:00,  9.25 MiB/s][A

Extraction completed...: 0 file [00:15, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:15<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:15<00:00,  9.25 MiB/s][A

Extraction completed...: 0 file [00:15, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:15<00:00, 15.89s/ url]Dl Completed...: 100%|██████████| 1/1 [00:15<00:00, 15.89s/ url]
Dl Size...: 100%|██████████| 162/162 [00:15<00:00,  9.25 MiB/s][A

Extraction completed...: 0 file [00:15, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:15<00:00, 15.89s/ url]
Dl Size...: 100%|██████████| 162/162 [00:15<00:00,  9.25 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:15<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:17<00:00, 17.44s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:17<00:00, 15.89s/ url]
Dl Size...: 100%|██████████| 162/162 [00:17<00:00,  9.25 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:17<00:00, 17.44s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:17<00:00, 17.44s/ file]
Dl Size...: 100%|██████████| 162/162 [00:17<00:00,  9.29 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:17<00:00, 17.44s/ url]
0 examples [00:00, ? examples/s]2020-06-25 00:08:01.096102: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-06-25 00:08:01.107637: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-06-25 00:08:01.108373: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55f4ec31c1f0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-25 00:08:01.108402: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
47 examples [00:00, 469.59 examples/s]194 examples [00:00, 589.77 examples/s]332 examples [00:00, 711.90 examples/s]465 examples [00:00, 827.00 examples/s]611 examples [00:00, 949.61 examples/s]741 examples [00:00, 1031.96 examples/s]882 examples [00:00, 1122.14 examples/s]1006 examples [00:00, 1142.66 examples/s]1129 examples [00:00, 1131.39 examples/s]1254 examples [00:01, 1163.64 examples/s]1382 examples [00:01, 1195.84 examples/s]1520 examples [00:01, 1245.33 examples/s]1656 examples [00:01, 1276.87 examples/s]1798 examples [00:01, 1315.62 examples/s]1934 examples [00:01, 1327.59 examples/s]2073 examples [00:01, 1344.14 examples/s]2218 examples [00:01, 1372.07 examples/s]2357 examples [00:01, 1345.95 examples/s]2500 examples [00:01, 1368.92 examples/s]2638 examples [00:02, 1371.47 examples/s]2781 examples [00:02, 1388.16 examples/s]2921 examples [00:02, 1377.37 examples/s]3059 examples [00:02, 1376.28 examples/s]3197 examples [00:02, 1327.25 examples/s]3345 examples [00:02, 1367.75 examples/s]3494 examples [00:02, 1400.35 examples/s]3640 examples [00:02, 1415.86 examples/s]3783 examples [00:02, 1403.91 examples/s]3925 examples [00:02, 1406.83 examples/s]4066 examples [00:03, 1384.39 examples/s]4213 examples [00:03, 1408.02 examples/s]4359 examples [00:03, 1422.11 examples/s]4502 examples [00:03, 1411.79 examples/s]4646 examples [00:03, 1418.74 examples/s]4792 examples [00:03, 1429.45 examples/s]4939 examples [00:03, 1438.55 examples/s]5085 examples [00:03, 1444.72 examples/s]5230 examples [00:03, 1431.93 examples/s]5378 examples [00:03, 1444.15 examples/s]5525 examples [00:04, 1451.71 examples/s]5671 examples [00:04, 1453.02 examples/s]5817 examples [00:04, 1453.34 examples/s]5963 examples [00:04, 1408.95 examples/s]6105 examples [00:04, 1408.65 examples/s]6250 examples [00:04, 1419.64 examples/s]6393 examples [00:04, 1414.80 examples/s]6542 examples [00:04, 1435.08 examples/s]6686 examples [00:04, 1426.76 examples/s]6832 examples [00:04, 1435.91 examples/s]6980 examples [00:05, 1448.31 examples/s]7125 examples [00:05, 1406.94 examples/s]7268 examples [00:05, 1411.74 examples/s]7410 examples [00:05, 1408.74 examples/s]7552 examples [00:05, 1349.70 examples/s]7700 examples [00:05, 1385.48 examples/s]7844 examples [00:05, 1401.10 examples/s]7992 examples [00:05, 1420.89 examples/s]8135 examples [00:05, 1418.61 examples/s]8279 examples [00:06, 1423.70 examples/s]8422 examples [00:06, 1415.32 examples/s]8569 examples [00:06, 1429.05 examples/s]8716 examples [00:06, 1440.30 examples/s]8861 examples [00:06, 1425.82 examples/s]9006 examples [00:06, 1431.17 examples/s]9150 examples [00:06, 1430.93 examples/s]9294 examples [00:06, 1368.08 examples/s]9432 examples [00:06, 1360.52 examples/s]9571 examples [00:06, 1368.39 examples/s]9713 examples [00:07, 1383.33 examples/s]9859 examples [00:07, 1403.53 examples/s]10000 examples [00:07, 1402.16 examples/s]10141 examples [00:07, 1354.85 examples/s]10277 examples [00:07, 1351.94 examples/s]10413 examples [00:07, 1349.52 examples/s]10558 examples [00:07, 1376.76 examples/s]10702 examples [00:07, 1392.35 examples/s]10842 examples [00:07, 1391.18 examples/s]10982 examples [00:07, 1374.14 examples/s]11120 examples [00:08, 1375.34 examples/s]11266 examples [00:08, 1398.52 examples/s]11413 examples [00:08, 1417.65 examples/s]11555 examples [00:08, 1400.90 examples/s]11696 examples [00:08, 1358.44 examples/s]11833 examples [00:08, 1343.45 examples/s]11973 examples [00:08, 1358.26 examples/s]12120 examples [00:08, 1387.82 examples/s]12264 examples [00:08, 1399.76 examples/s]12405 examples [00:08, 1393.48 examples/s]12545 examples [00:09, 1395.36 examples/s]12687 examples [00:09, 1402.33 examples/s]12832 examples [00:09, 1415.68 examples/s]12975 examples [00:09, 1418.50 examples/s]13117 examples [00:09, 1409.39 examples/s]13259 examples [00:09, 1407.52 examples/s]13403 examples [00:09, 1414.50 examples/s]13549 examples [00:09, 1425.56 examples/s]13692 examples [00:09, 1411.18 examples/s]13834 examples [00:10, 1387.65 examples/s]13973 examples [00:10, 1386.67 examples/s]14113 examples [00:10, 1388.57 examples/s]14254 examples [00:10, 1392.25 examples/s]14394 examples [00:10, 1387.18 examples/s]14535 examples [00:10, 1390.06 examples/s]14675 examples [00:10, 1390.82 examples/s]14821 examples [00:10, 1408.74 examples/s]14962 examples [00:10, 1407.91 examples/s]15103 examples [00:10, 1384.59 examples/s]15247 examples [00:11, 1398.49 examples/s]15389 examples [00:11, 1404.21 examples/s]15530 examples [00:11, 1390.79 examples/s]15675 examples [00:11, 1405.27 examples/s]15816 examples [00:11, 1403.32 examples/s]15957 examples [00:11, 1375.13 examples/s]16095 examples [00:11, 1328.51 examples/s]16230 examples [00:11, 1333.83 examples/s]16364 examples [00:11, 1335.61 examples/s]16498 examples [00:11, 1320.83 examples/s]16640 examples [00:12, 1347.37 examples/s]16782 examples [00:12, 1367.40 examples/s]16928 examples [00:12, 1391.77 examples/s]17077 examples [00:12, 1417.51 examples/s]17220 examples [00:12, 1418.73 examples/s]17365 examples [00:12, 1425.53 examples/s]17510 examples [00:12, 1428.34 examples/s]17659 examples [00:12, 1443.42 examples/s]17804 examples [00:12, 1424.97 examples/s]17947 examples [00:12, 1412.20 examples/s]18089 examples [00:13, 1349.88 examples/s]18225 examples [00:13, 1327.68 examples/s]18361 examples [00:13, 1335.01 examples/s]18500 examples [00:13, 1350.18 examples/s]18636 examples [00:13, 1353.03 examples/s]18776 examples [00:13, 1364.39 examples/s]18916 examples [00:13, 1374.34 examples/s]19061 examples [00:13, 1395.32 examples/s]19201 examples [00:13, 1395.92 examples/s]19341 examples [00:13, 1387.39 examples/s]19485 examples [00:14, 1401.18 examples/s]19630 examples [00:14, 1415.22 examples/s]19772 examples [00:14, 1369.85 examples/s]19910 examples [00:14, 1355.48 examples/s]20046 examples [00:14, 1309.95 examples/s]20178 examples [00:14, 1278.85 examples/s]20307 examples [00:14, 1251.31 examples/s]20439 examples [00:14, 1268.71 examples/s]20580 examples [00:14, 1305.97 examples/s]20712 examples [00:15, 1310.06 examples/s]20844 examples [00:15, 1279.49 examples/s]20973 examples [00:15, 1255.91 examples/s]21099 examples [00:15, 1249.09 examples/s]21225 examples [00:15, 1248.85 examples/s]21351 examples [00:15, 1243.20 examples/s]21478 examples [00:15, 1249.01 examples/s]21620 examples [00:15, 1293.77 examples/s]21765 examples [00:15, 1335.50 examples/s]21911 examples [00:15, 1368.40 examples/s]22049 examples [00:16, 1363.70 examples/s]22186 examples [00:16, 1351.40 examples/s]22322 examples [00:16, 1334.19 examples/s]22456 examples [00:16, 1331.54 examples/s]22593 examples [00:16, 1342.14 examples/s]22728 examples [00:16, 1339.67 examples/s]22867 examples [00:16, 1353.55 examples/s]23003 examples [00:16, 1337.71 examples/s]23137 examples [00:16, 1334.76 examples/s]23271 examples [00:16, 1328.55 examples/s]23404 examples [00:17, 1315.43 examples/s]23546 examples [00:17, 1343.05 examples/s]23691 examples [00:17, 1372.16 examples/s]23835 examples [00:17, 1390.72 examples/s]23977 examples [00:17, 1397.36 examples/s]24119 examples [00:17, 1403.81 examples/s]24260 examples [00:17, 1399.76 examples/s]24401 examples [00:17, 1374.80 examples/s]24539 examples [00:17, 1352.22 examples/s]24675 examples [00:17, 1347.84 examples/s]24810 examples [00:18, 1333.43 examples/s]24944 examples [00:18, 1322.84 examples/s]25077 examples [00:18, 1284.69 examples/s]25216 examples [00:18, 1312.37 examples/s]25355 examples [00:18, 1332.48 examples/s]25489 examples [00:18, 1331.14 examples/s]25623 examples [00:18, 1320.56 examples/s]25762 examples [00:18, 1339.20 examples/s]25899 examples [00:18, 1346.11 examples/s]26034 examples [00:19, 1347.16 examples/s]26169 examples [00:19, 1340.12 examples/s]26311 examples [00:19, 1361.87 examples/s]26454 examples [00:19, 1379.79 examples/s]26593 examples [00:19, 1377.73 examples/s]26738 examples [00:19, 1397.42 examples/s]26878 examples [00:19, 1362.70 examples/s]27020 examples [00:19, 1377.71 examples/s]27162 examples [00:19, 1389.35 examples/s]27305 examples [00:19, 1398.87 examples/s]27446 examples [00:20, 1400.51 examples/s]27587 examples [00:20, 1386.26 examples/s]27728 examples [00:20, 1390.76 examples/s]27868 examples [00:20, 1375.00 examples/s]28016 examples [00:20, 1403.18 examples/s]28162 examples [00:20, 1418.82 examples/s]28305 examples [00:20, 1413.56 examples/s]28450 examples [00:20, 1422.10 examples/s]28595 examples [00:20, 1429.96 examples/s]28740 examples [00:20, 1432.84 examples/s]28884 examples [00:21, 1402.69 examples/s]29025 examples [00:21, 1383.78 examples/s]29164 examples [00:21, 1375.57 examples/s]29302 examples [00:21, 1326.41 examples/s]29447 examples [00:21, 1357.13 examples/s]29590 examples [00:21, 1377.69 examples/s]29729 examples [00:21, 1372.82 examples/s]29868 examples [00:21, 1377.90 examples/s]30007 examples [00:21, 1322.82 examples/s]30147 examples [00:21, 1342.31 examples/s]30289 examples [00:22, 1363.49 examples/s]30426 examples [00:22, 1330.20 examples/s]30562 examples [00:22, 1338.42 examples/s]30704 examples [00:22, 1359.49 examples/s]30850 examples [00:22, 1386.29 examples/s]30993 examples [00:22, 1396.02 examples/s]31133 examples [00:22, 1383.10 examples/s]31272 examples [00:22, 1367.66 examples/s]31409 examples [00:22, 1351.70 examples/s]31546 examples [00:23, 1354.99 examples/s]31683 examples [00:23, 1358.60 examples/s]31821 examples [00:23, 1363.79 examples/s]31966 examples [00:23, 1387.78 examples/s]32107 examples [00:23, 1393.09 examples/s]32252 examples [00:23, 1409.05 examples/s]32396 examples [00:23, 1417.36 examples/s]32538 examples [00:23, 1407.59 examples/s]32681 examples [00:23, 1413.54 examples/s]32825 examples [00:23, 1419.84 examples/s]32968 examples [00:24, 1418.11 examples/s]33112 examples [00:24, 1422.46 examples/s]33255 examples [00:24, 1382.80 examples/s]33394 examples [00:24, 1375.37 examples/s]33532 examples [00:24, 1310.44 examples/s]33673 examples [00:24, 1336.65 examples/s]33811 examples [00:24, 1348.67 examples/s]33950 examples [00:24, 1358.87 examples/s]34095 examples [00:24, 1384.06 examples/s]34239 examples [00:24, 1398.50 examples/s]34382 examples [00:25, 1406.88 examples/s]34529 examples [00:25, 1423.05 examples/s]34672 examples [00:25, 1390.64 examples/s]34812 examples [00:25, 1392.32 examples/s]34952 examples [00:25, 1392.63 examples/s]35099 examples [00:25, 1413.27 examples/s]35242 examples [00:25, 1416.24 examples/s]35384 examples [00:25, 1394.80 examples/s]35530 examples [00:25, 1410.88 examples/s]35672 examples [00:25, 1342.99 examples/s]35808 examples [00:26, 1316.08 examples/s]35947 examples [00:26, 1336.80 examples/s]36090 examples [00:26, 1362.89 examples/s]36234 examples [00:26, 1382.37 examples/s]36380 examples [00:26, 1402.41 examples/s]36524 examples [00:26, 1412.25 examples/s]36666 examples [00:26, 1398.50 examples/s]36807 examples [00:26, 1390.76 examples/s]36951 examples [00:26, 1402.22 examples/s]37092 examples [00:27, 1400.28 examples/s]37237 examples [00:27, 1412.58 examples/s]37379 examples [00:27, 1407.29 examples/s]37520 examples [00:27, 1375.21 examples/s]37658 examples [00:27, 1345.41 examples/s]37793 examples [00:27, 1304.80 examples/s]37935 examples [00:27, 1336.98 examples/s]38070 examples [00:27, 1328.76 examples/s]38218 examples [00:27, 1369.46 examples/s]38358 examples [00:27, 1375.88 examples/s]38497 examples [00:28, 1366.37 examples/s]38643 examples [00:28, 1391.76 examples/s]38785 examples [00:28, 1399.84 examples/s]38931 examples [00:28, 1416.41 examples/s]39076 examples [00:28, 1425.42 examples/s]39224 examples [00:28, 1441.08 examples/s]39369 examples [00:28, 1437.59 examples/s]39513 examples [00:28, 1402.62 examples/s]39660 examples [00:28, 1420.25 examples/s]39803 examples [00:28, 1417.97 examples/s]39945 examples [00:29, 1413.81 examples/s]40087 examples [00:29, 1361.11 examples/s]40224 examples [00:29, 1362.78 examples/s]40365 examples [00:29, 1374.58 examples/s]40507 examples [00:29, 1385.79 examples/s]40650 examples [00:29, 1397.58 examples/s]40792 examples [00:29, 1403.53 examples/s]40933 examples [00:29, 1390.50 examples/s]41077 examples [00:29, 1403.31 examples/s]41218 examples [00:29, 1387.04 examples/s]41358 examples [00:30, 1390.87 examples/s]41501 examples [00:30, 1400.55 examples/s]41642 examples [00:30, 1396.77 examples/s]41783 examples [00:30, 1398.92 examples/s]41923 examples [00:30, 1374.40 examples/s]42064 examples [00:30, 1383.02 examples/s]42212 examples [00:30, 1409.50 examples/s]42354 examples [00:30, 1403.44 examples/s]42495 examples [00:30, 1367.39 examples/s]42633 examples [00:31, 1370.87 examples/s]42777 examples [00:31, 1390.14 examples/s]42917 examples [00:31, 1368.92 examples/s]43055 examples [00:31, 1344.05 examples/s]43191 examples [00:31, 1346.20 examples/s]43331 examples [00:31, 1361.86 examples/s]43476 examples [00:31, 1384.70 examples/s]43622 examples [00:31, 1405.96 examples/s]43763 examples [00:31, 1377.56 examples/s]43905 examples [00:31, 1388.99 examples/s]44045 examples [00:32, 1380.06 examples/s]44190 examples [00:32, 1397.95 examples/s]44336 examples [00:32, 1415.60 examples/s]44478 examples [00:32, 1390.43 examples/s]44618 examples [00:32, 1375.04 examples/s]44756 examples [00:32, 1336.17 examples/s]44890 examples [00:32, 1335.23 examples/s]45024 examples [00:32, 1326.47 examples/s]45160 examples [00:32, 1333.58 examples/s]45297 examples [00:32, 1341.61 examples/s]45433 examples [00:33, 1345.44 examples/s]45568 examples [00:33, 1342.92 examples/s]45713 examples [00:33, 1371.31 examples/s]45853 examples [00:33, 1377.13 examples/s]45998 examples [00:33, 1397.67 examples/s]46144 examples [00:33, 1414.13 examples/s]46290 examples [00:33, 1425.67 examples/s]46435 examples [00:33, 1431.43 examples/s]46579 examples [00:33, 1411.46 examples/s]46721 examples [00:33, 1389.51 examples/s]46861 examples [00:34, 1376.29 examples/s]47001 examples [00:34, 1382.90 examples/s]47140 examples [00:34, 1333.61 examples/s]47277 examples [00:34, 1342.66 examples/s]47422 examples [00:34, 1371.52 examples/s]47568 examples [00:34, 1394.98 examples/s]47708 examples [00:34, 1371.31 examples/s]47846 examples [00:34, 1360.20 examples/s]47983 examples [00:34, 1346.82 examples/s]48125 examples [00:34, 1366.03 examples/s]48264 examples [00:35, 1371.56 examples/s]48403 examples [00:35, 1376.53 examples/s]48541 examples [00:35, 1364.30 examples/s]48680 examples [00:35, 1370.64 examples/s]48822 examples [00:35, 1383.15 examples/s]48968 examples [00:35, 1402.99 examples/s]49109 examples [00:35, 1382.50 examples/s]49253 examples [00:35, 1397.35 examples/s]49393 examples [00:35, 1388.19 examples/s]49533 examples [00:36, 1388.57 examples/s]49672 examples [00:36, 1384.34 examples/s]49811 examples [00:36, 1380.90 examples/s]49955 examples [00:36, 1397.45 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 21%|██▏       | 10738/50000 [00:00<00:00, 107377.29 examples/s] 56%|█████▌    | 28047/50000 [00:00<00:00, 121178.71 examples/s] 91%|█████████ | 45501/50000 [00:00<00:00, 133415.12 examples/s]                                                                0 examples [00:00, ? examples/s]103 examples [00:00, 1025.03 examples/s]231 examples [00:00, 1090.01 examples/s]376 examples [00:00, 1175.82 examples/s]517 examples [00:00, 1236.21 examples/s]660 examples [00:00, 1286.81 examples/s]785 examples [00:00, 1274.68 examples/s]914 examples [00:00, 1278.46 examples/s]1058 examples [00:00, 1321.45 examples/s]1199 examples [00:00, 1344.01 examples/s]1335 examples [00:01, 1345.81 examples/s]1477 examples [00:01, 1364.90 examples/s]1616 examples [00:01, 1370.20 examples/s]1752 examples [00:01, 1366.29 examples/s]1888 examples [00:01, 1356.81 examples/s]2026 examples [00:01, 1363.05 examples/s]2174 examples [00:01, 1393.57 examples/s]2315 examples [00:01, 1397.61 examples/s]2457 examples [00:01, 1401.63 examples/s]2602 examples [00:01, 1415.06 examples/s]2748 examples [00:02, 1427.11 examples/s]2891 examples [00:02, 1403.60 examples/s]3032 examples [00:02, 1340.08 examples/s]3179 examples [00:02, 1375.33 examples/s]3325 examples [00:02, 1398.91 examples/s]3466 examples [00:02, 1401.28 examples/s]3612 examples [00:02, 1416.05 examples/s]3754 examples [00:02, 1407.59 examples/s]3900 examples [00:02, 1421.57 examples/s]4049 examples [00:02, 1439.22 examples/s]4194 examples [00:03, 1441.40 examples/s]4339 examples [00:03, 1428.16 examples/s]4482 examples [00:03, 1411.14 examples/s]4624 examples [00:03, 1411.83 examples/s]4771 examples [00:03, 1428.62 examples/s]4918 examples [00:03, 1438.91 examples/s]5064 examples [00:03, 1443.23 examples/s]5209 examples [00:03, 1316.23 examples/s]5350 examples [00:03, 1341.91 examples/s]5497 examples [00:03, 1376.58 examples/s]5647 examples [00:04, 1410.15 examples/s]5792 examples [00:04, 1419.86 examples/s]5935 examples [00:04, 1409.66 examples/s]6079 examples [00:04, 1416.31 examples/s]6225 examples [00:04, 1428.02 examples/s]6369 examples [00:04, 1377.92 examples/s]6508 examples [00:04, 1351.47 examples/s]6644 examples [00:04, 1335.66 examples/s]6791 examples [00:04, 1372.89 examples/s]6935 examples [00:05, 1391.62 examples/s]7077 examples [00:05, 1398.30 examples/s]7218 examples [00:05, 1391.63 examples/s]7358 examples [00:05, 1381.73 examples/s]7497 examples [00:05, 1362.80 examples/s]7641 examples [00:05, 1383.46 examples/s]7783 examples [00:05, 1393.97 examples/s]7925 examples [00:05, 1399.98 examples/s]8066 examples [00:05, 1392.39 examples/s]8211 examples [00:05, 1407.44 examples/s]8354 examples [00:06, 1413.54 examples/s]8496 examples [00:06, 1392.70 examples/s]8636 examples [00:06, 1380.12 examples/s]8775 examples [00:06, 1372.57 examples/s]8913 examples [00:06, 1360.43 examples/s]9050 examples [00:06, 1357.57 examples/s]9196 examples [00:06, 1384.64 examples/s]9335 examples [00:06, 1346.29 examples/s]9471 examples [00:06, 1348.55 examples/s]9607 examples [00:06, 1347.58 examples/s]9746 examples [00:07, 1357.84 examples/s]9882 examples [00:07, 1357.91 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete5PCPRY/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete5PCPRY/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7efcd8662488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7efcd8662488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7efcd8662488> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7efcbf80d5f8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7efc5dab8080>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7efcd8662488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7efcd8662488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7efcd8662488> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7efc5dab8a58>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7efcbf775160>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7efc504911e0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7efc504911e0> 

  function with postional parmater data_info <function split_train_valid at 0x7efc504911e0> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7efc504912f0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7efc504912f0> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7efc504912f0> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.0
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.0/en_core_web_sm-2.3.0.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.0) (2.3.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (45.2.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.24.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.0.3)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.7.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.1.3)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.4.1)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.19.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.2)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (7.4.1)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (4.46.1)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2020.6.20)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.9)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.6.1)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.0-py3-none-any.whl size=12048606 sha256=fcdcb53a110ae114a676ce86d70a1d04ebff6da6dd50cf2c76eb63c413fb52df
  Stored in directory: /tmp/pip-ephem-wheel-cache-nxa29ldp/wheels/4a/db/07/94eee4f3a60150464a04160bd0dfe9c8752ab981fe92f16aea
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<20:04:58, 11.9kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<14:17:27, 16.8kB/s].vector_cache/glove.6B.zip:   0%|          | 188k/862M [00:00<10:04:13, 23.8kB/s] .vector_cache/glove.6B.zip:   0%|          | 754k/862M [00:01<7:03:41, 33.9kB/s] .vector_cache/glove.6B.zip:   0%|          | 2.97M/862M [00:01<4:56:03, 48.4kB/s].vector_cache/glove.6B.zip:   1%|          | 7.48M/862M [00:01<3:26:14, 69.1kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.1M/862M [00:01<2:23:41, 98.6kB/s].vector_cache/glove.6B.zip:   2%|▏         | 17.0M/862M [00:01<1:40:05, 141kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 21.4M/862M [00:01<1:09:48, 201kB/s].vector_cache/glove.6B.zip:   3%|▎         | 26.5M/862M [00:01<48:39, 286kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 29.9M/862M [00:01<34:02, 407kB/s].vector_cache/glove.6B.zip:   4%|▍         | 34.8M/862M [00:01<23:46, 580kB/s].vector_cache/glove.6B.zip:   4%|▍         | 38.7M/862M [00:02<16:40, 823kB/s].vector_cache/glove.6B.zip:   5%|▌         | 43.5M/862M [00:02<11:41, 1.17MB/s].vector_cache/glove.6B.zip:   6%|▌         | 47.6M/862M [00:02<08:14, 1.65MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.9M/862M [00:02<06:06, 2.21MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.0M/862M [00:04<06:10, 2.17MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.2M/862M [00:04<07:57, 1.69MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.8M/862M [00:04<06:29, 2.07MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.1M/862M [00:05<04:43, 2.83MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.2M/862M [00:06<08:52, 1.51MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.5M/862M [00:06<07:48, 1.71MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.9M/862M [00:06<05:47, 2.30MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.4M/862M [00:08<06:50, 1.95MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.6M/862M [00:08<07:39, 1.74MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.3M/862M [00:08<06:02, 2.20MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.1M/862M [00:08<04:21, 3.04MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.5M/862M [00:10<19:08, 691kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 68.8M/862M [00:10<14:51, 890kB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.1M/862M [00:10<10:41, 1.24MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.1M/862M [00:10<07:39, 1.72MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.7M/862M [00:12<17:59, 731kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 72.8M/862M [00:12<15:31, 847kB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.5M/862M [00:12<11:28, 1.15MB/s].vector_cache/glove.6B.zip:   9%|▉         | 75.5M/862M [00:12<08:12, 1.60MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.8M/862M [00:14<10:47, 1.21MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.2M/862M [00:14<08:59, 1.45MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.6M/862M [00:14<06:36, 1.98MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.1M/862M [00:16<07:24, 1.76MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.2M/862M [00:16<08:07, 1.60MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.0M/862M [00:16<06:24, 2.03MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.8M/862M [00:16<04:38, 2.79MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.2M/862M [00:18<17:02, 760kB/s] .vector_cache/glove.6B.zip:  10%|▉         | 85.6M/862M [00:18<13:21, 969kB/s].vector_cache/glove.6B.zip:  10%|█         | 87.0M/862M [00:18<09:41, 1.33MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.4M/862M [00:20<09:34, 1.34MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.6M/862M [00:20<09:33, 1.35MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.3M/862M [00:20<07:24, 1.74MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.1M/862M [00:20<05:19, 2.41MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.6M/862M [00:22<17:26, 734kB/s] .vector_cache/glove.6B.zip:  11%|█         | 93.9M/862M [00:22<13:38, 939kB/s].vector_cache/glove.6B.zip:  11%|█         | 95.4M/862M [00:22<09:53, 1.29MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.7M/862M [00:24<09:42, 1.31MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.9M/862M [00:24<09:36, 1.33MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.6M/862M [00:24<07:19, 1.74MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:24<05:17, 2.40MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<09:56, 1.28MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<08:21, 1.52MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:26<06:11, 2.04MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<07:04, 1.78MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<07:44, 1.63MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<06:07, 2.05MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:28<04:26, 2.83MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<17:28, 717kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<13:36, 920kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<09:51, 1.27MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<09:43, 1.28MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<09:33, 1.30MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<07:16, 1.71MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:32<05:15, 2.36MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 119M/862M [00:34<09:11, 1.35MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<07:48, 1.59MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<05:45, 2.15MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<06:40, 1.84MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<07:24, 1.66MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<05:46, 2.13MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:36<04:11, 2.93MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<08:28, 1.44MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<07:17, 1.68MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:38<05:23, 2.27MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<06:23, 1.91MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<07:11, 1.69MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<05:42, 2.13MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:40<04:07, 2.93MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<15:50, 765kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<12:26, 973kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<09:01, 1.34MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<08:55, 1.35MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<08:54, 1.35MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 140M/862M [00:44<06:54, 1.74MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:44<04:58, 2.41MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<17:00, 704kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<13:13, 905kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<09:31, 1.25MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<09:15, 1.29MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<09:08, 1.30MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<06:57, 1.71MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:48<05:00, 2.37MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<09:17, 1.27MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<07:51, 1.51MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:50<05:45, 2.05MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:50<04:10, 2.82MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<5:56:43, 33.0kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<4:12:15, 46.6kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<2:57:02, 66.4kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:52<2:03:35, 94.7kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<1:37:23, 120kB/s] .vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<1:09:26, 168kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:54<48:47, 239kB/s]  .vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<36:34, 318kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<26:53, 432kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:56<19:04, 608kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<15:50, 730kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<13:40, 845kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<10:06, 1.14MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:58<07:12, 1.60MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<10:47, 1.06MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<08:49, 1.30MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:00<06:28, 1.77MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<07:01, 1.63MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<07:28, 1.53MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<05:51, 1.95MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:02<04:14, 2.68MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<16:42, 679kB/s] .vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<12:57, 876kB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:04<09:21, 1.21MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<08:59, 1.25MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 185M/862M [01:06<08:49, 1.28MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<06:47, 1.66MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:06<04:52, 2.30MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<13:52, 808kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<10:56, 1.02MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:08<07:57, 1.41MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:10<07:59, 1.39MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:10<06:49, 1.63MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:10<05:03, 2.19MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<06:00, 1.84MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<06:39, 1.66MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:12<05:16, 2.10MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:12<03:48, 2.89MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<14:27, 761kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<11:19, 971kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:14<08:10, 1.34MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<08:04, 1.36MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<08:04, 1.35MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<06:12, 1.76MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:16<04:26, 2.45MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<12:24, 876kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:18<09:51, 1.10MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:18<07:08, 1.52MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<07:22, 1.46MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<07:33, 1.43MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<05:47, 1.86MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:20<04:11, 2.57MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<07:26, 1.44MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<06:21, 1.69MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:22<04:43, 2.26MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<05:40, 1.88MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<06:22, 1.67MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:24<04:56, 2.15MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:24<03:36, 2.93MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<06:21, 1.66MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<05:37, 1.88MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:26<04:10, 2.53MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<05:12, 2.02MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<05:54, 1.78MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:28<04:41, 2.24MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:28<03:22, 3.09MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<12:41, 823kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<10:02, 1.04MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:30<07:18, 1.43MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:32<07:21, 1.41MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<07:30, 1.38MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<05:44, 1.81MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:32<04:12, 2.46MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<05:57, 1.73MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<05:18, 1.94MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<03:57, 2.60MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:36<05:01, 2.04MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:36<05:42, 1.79MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<04:32, 2.26MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:36<03:17, 3.09MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<32:51, 310kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<24:08, 421kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:38<17:06, 593kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:38<12:03, 838kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<5:11:19, 32.5kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<3:40:05, 45.9kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<2:34:27, 65.3kB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:40<1:47:44, 93.2kB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:42<1:26:05, 117kB/s] .vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<1:01:17, 164kB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<43:03, 232kB/s]  .vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:44<32:16, 309kB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<24:42, 403kB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<17:48, 559kB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:44<12:31, 791kB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<25:22, 390kB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<18:50, 525kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<13:24, 736kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<11:29, 855kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<10:12, 962kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<07:41, 1.27MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:48<05:28, 1.78MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<14:14, 685kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<11:04, 880kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<08:00, 1.21MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<07:41, 1.26MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:52<07:32, 1.28MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<05:43, 1.69MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:52<04:08, 2.32MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<06:16, 1.53MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<05:27, 1.76MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<04:02, 2.37MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<04:53, 1.95MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<04:26, 2.15MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:56<03:21, 2.84MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<04:31, 2.09MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<04:13, 2.24MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<03:12, 2.95MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:59<04:20, 2.16MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<05:02, 1.86MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<04:01, 2.33MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 302M/862M [02:00<02:55, 3.20MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 302M/862M [02:01<21:57, 426kB/s] .vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:02<16:25, 568kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<11:43, 794kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:03<10:12, 909kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:04<09:11, 1.01MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<06:56, 1.33MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:04<04:57, 1.86MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<14:38, 629kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<11:15, 816kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<08:05, 1.13MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<07:37, 1.20MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<07:21, 1.24MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<05:38, 1.61MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:08<04:02, 2.24MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<12:33, 722kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:09<09:47, 925kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<07:05, 1.27MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<06:53, 1.30MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<06:49, 1.32MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 323M/862M [02:12<05:11, 1.73MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<03:48, 2.36MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<05:11, 1.72MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<04:39, 1.92MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<03:27, 2.57MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<04:20, 2.04MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<05:00, 1.77MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:16<03:55, 2.25MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:16<02:52, 3.06MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<04:56, 1.78MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<04:23, 2.00MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<03:17, 2.66MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<04:18, 2.03MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<03:56, 2.21MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<02:57, 2.94MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<04:00, 2.16MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<04:45, 1.82MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<03:48, 2.27MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:22<02:46, 3.10MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<12:18, 697kB/s] .vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<09:31, 900kB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<06:52, 1.24MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<06:42, 1.27MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<06:34, 1.29MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<05:04, 1.67MB/s].vector_cache/glove.6B.zip:  41%|████      | 356M/862M [02:26<03:37, 2.33MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<11:35, 727kB/s] .vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<09:02, 932kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 358M/862M [02:27<06:31, 1.29MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<06:22, 1.31MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<05:23, 1.55MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:29<03:57, 2.10MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<04:34, 1.81MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<04:57, 1.67MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<03:51, 2.15MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:32<02:46, 2.96MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<08:14, 998kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<06:39, 1.24MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:33<04:51, 1.69MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<05:12, 1.57MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<05:26, 1.50MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:35<04:12, 1.94MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:36<03:03, 2.65MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<04:55, 1.64MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<04:20, 1.86MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:37<03:13, 2.50MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<03:59, 2.01MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<04:35, 1.75MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<03:39, 2.19MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:40<02:39, 3.00MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<11:28, 693kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<08:54, 892kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:41<06:24, 1.24MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<06:10, 1.27MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<06:04, 1.30MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<04:40, 1.68MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:44<03:21, 2.32MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<11:49, 661kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<09:08, 853kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:45<06:35, 1.18MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<06:16, 1.23MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<06:06, 1.27MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 399M/862M [02:47<04:42, 1.64MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:48<03:21, 2.28MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<10:43, 715kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<08:19, 922kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:49<06:00, 1.27MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<05:54, 1.29MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<04:55, 1.54MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:51<03:37, 2.08MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<04:16, 1.77MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<04:35, 1.64MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<03:33, 2.12MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:53<02:34, 2.90MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<05:13, 1.43MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<04:28, 1.67MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<03:20, 2.23MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:57<03:55, 1.88MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<04:20, 1.70MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<03:25, 2.15MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:57<02:28, 2.95MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<24:57, 294kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<18:16, 401kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<12:55, 564kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:01<10:35, 685kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<08:13, 881kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<06:13, 1.16MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:01<04:24, 1.63MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<08:23, 854kB/s] .vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:04<09:03, 791kB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:04<07:07, 1.00MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<05:10, 1.38MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<05:05, 1.40MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:06<04:19, 1.64MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<03:12, 2.20MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<03:48, 1.85MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:08<04:14, 1.66MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<03:20, 2.10MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:08<02:25, 2.87MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:09<09:57, 699kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<07:44, 899kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<05:34, 1.25MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<05:22, 1.28MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<05:18, 1.30MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<04:05, 1.68MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:12<02:56, 2.33MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<10:19, 661kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<07:56, 858kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<05:42, 1.19MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<05:30, 1.23MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<05:22, 1.26MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<04:07, 1.63MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:16<02:57, 2.26MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<09:24, 711kB/s] .vector_cache/glove.6B.zip:  54%|█████▎    | 461M/862M [03:17<07:17, 917kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<05:13, 1.27MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<05:09, 1.28MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<05:04, 1.30MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<03:54, 1.69MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:20<02:48, 2.34MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<08:56, 732kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:21<07:00, 934kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<05:03, 1.29MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<04:55, 1.32MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<04:53, 1.33MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:24<03:42, 1.74MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<02:43, 2.36MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<03:34, 1.79MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<03:13, 1.98MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<02:24, 2.65MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<03:03, 2.07MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<03:28, 1.82MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<02:43, 2.32MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:28<01:59, 3.17MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<04:26, 1.41MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<03:48, 1.64MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:29<02:49, 2.21MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<03:19, 1.87MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<03:00, 2.06MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:31<02:16, 2.72MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<02:55, 2.10MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<02:41, 2.27MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:33<02:00, 3.03MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<02:46, 2.18MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<03:17, 1.84MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<02:38, 2.29MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:36<01:54, 3.14MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<07:43, 776kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<06:04, 987kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:37<04:22, 1.36MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<04:19, 1.37MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<03:39, 1.62MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:39<02:42, 2.18MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<03:13, 1.82MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<03:32, 1.65MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:41<02:48, 2.08MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:41<02:01, 2.86MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<07:34, 763kB/s] .vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<05:56, 972kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:43<04:17, 1.34MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<04:13, 1.35MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<04:13, 1.35MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<03:14, 1.76MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:45<02:19, 2.42MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<07:26, 759kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<06:27, 873kB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<04:46, 1.18MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<04:13, 1.32MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:49<03:34, 1.56MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:49<02:38, 2.10MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<03:02, 1.81MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<03:20, 1.64MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:51<02:39, 2.07MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:51<01:54, 2.86MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<06:30, 836kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<05:07, 1.06MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:53<03:43, 1.45MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<03:46, 1.42MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<03:49, 1.40MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<02:58, 1.80MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:55<02:08, 2.48MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<07:55, 669kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<06:07, 864kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:57<04:23, 1.20MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<04:11, 1.25MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<04:06, 1.27MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<03:06, 1.68MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [03:59<02:15, 2.30MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<03:08, 1.64MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<02:46, 1.86MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:01<02:03, 2.50MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<02:32, 2.00MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<02:55, 1.74MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:03<02:19, 2.18MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 560M/862M [04:03<01:40, 3.01MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<06:33, 765kB/s] .vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<05:08, 975kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:05<03:41, 1.35MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<03:39, 1.36MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<03:39, 1.35MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<02:49, 1.74MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:07<02:01, 2.42MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<06:38, 735kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<05:12, 937kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:09<03:45, 1.29MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 573M/862M [04:11<03:39, 1.32MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<03:37, 1.33MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<02:47, 1.72MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:11<01:59, 2.38MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<06:28, 733kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<05:02, 938kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:13<03:38, 1.30MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:15<03:32, 1.32MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:15<03:30, 1.33MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<02:42, 1.72MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:15<01:56, 2.38MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<06:14, 737kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<04:52, 942kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:17<03:30, 1.30MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<03:23, 1.34MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 591M/862M [04:19<03:22, 1.34MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<02:34, 1.75MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:19<01:50, 2.43MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<04:19, 1.03MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<03:31, 1.26MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:21<02:33, 1.73MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<02:44, 1.60MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<02:50, 1.54MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:23<02:13, 1.97MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:23<01:35, 2.72MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<14:06, 306kB/s] .vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<10:20, 417kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:25<07:19, 587kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<06:00, 708kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<04:40, 910kB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:27<03:21, 1.26MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<03:14, 1.29MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<03:11, 1.31MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<02:27, 1.70MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:29<01:45, 2.35MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<05:36, 733kB/s] .vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<04:22, 938kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:31<03:09, 1.29MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<03:04, 1.32MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<03:00, 1.35MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<02:16, 1.77MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:33<01:37, 2.45MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<03:54, 1.02MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<03:10, 1.25MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:35<02:18, 1.71MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<02:27, 1.59MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<02:35, 1.51MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<02:01, 1.92MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:37<01:27, 2.64MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<05:40, 677kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<04:22, 877kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:39<03:07, 1.22MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<03:01, 1.25MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<02:56, 1.28MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<02:16, 1.65MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:41<01:36, 2.30MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<05:04, 728kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<03:57, 933kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:43<02:50, 1.29MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<02:45, 1.32MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<02:19, 1.55MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<01:42, 2.11MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<01:57, 1.82MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<02:09, 1.65MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<01:40, 2.12MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:47<01:12, 2.90MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<02:15, 1.55MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<01:57, 1.78MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:49<01:27, 2.37MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<01:45, 1.95MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<01:59, 1.71MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<01:33, 2.19MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:51<01:07, 2.98MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<02:00, 1.67MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<01:46, 1.89MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<01:19, 2.51MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<01:37, 2.01MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<01:30, 2.18MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<01:08, 2.86MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<01:29, 2.16MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<01:23, 2.32MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<01:02, 3.07MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<01:26, 2.18MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<01:21, 2.31MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<01:01, 3.06MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<01:22, 2.25MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<01:38, 1.87MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:01<01:17, 2.37MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:01<00:56, 3.22MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<01:43, 1.75MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<01:32, 1.95MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:03<01:08, 2.61MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<01:25, 2.05MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<01:19, 2.21MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:05<00:59, 2.94MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<01:18, 2.19MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<01:31, 1.88MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<01:12, 2.35MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:07<00:52, 3.22MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:09<09:01, 310kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<06:36, 422kB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<04:39, 594kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<03:48, 717kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<02:56, 924kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:11<02:06, 1.28MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<02:04, 1.28MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<02:02, 1.30MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<01:34, 1.68MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:13<01:06, 2.33MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<03:54, 661kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<03:01, 855kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:15<02:09, 1.18MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<02:02, 1.24MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<01:58, 1.27MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<01:30, 1.66MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:17<01:04, 2.30MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<03:21, 728kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<02:37, 931kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<01:52, 1.29MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<01:48, 1.31MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:21<01:31, 1.55MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<01:06, 2.11MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<01:16, 1.82MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<01:23, 1.65MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<01:06, 2.08MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:23<00:46, 2.87MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<02:56, 760kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<02:18, 969kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:25<01:38, 1.34MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<01:36, 1.35MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<01:34, 1.37MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:27<01:12, 1.77MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:27<00:51, 2.46MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<05:05, 413kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:29<03:47, 554kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<02:40, 774kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<02:16, 891kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:31<02:02, 993kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<01:31, 1.33MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:31<01:04, 1.85MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<01:43, 1.14MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<01:25, 1.38MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:33<01:01, 1.88MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:34<01:07, 1.69MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<01:12, 1.57MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<00:56, 1.99MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:35<00:39, 2.75MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<02:24, 756kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<01:53, 964kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 754M/862M [05:37<01:20, 1.33MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<01:18, 1.35MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<01:17, 1.35MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<01:00, 1.74MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:39<00:42, 2.40MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<02:32, 663kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:41<01:57, 860kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:41<01:23, 1.19MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 765M/862M [05:42<01:18, 1.23MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 765M/862M [05:43<01:15, 1.28MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<00:57, 1.66MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:43<00:40, 2.31MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<03:46, 410kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:44<02:48, 550kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<01:58, 770kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<01:40, 883kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<01:29, 984kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<01:06, 1.32MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:47<00:46, 1.84MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<01:12, 1.16MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<01:00, 1.39MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<00:43, 1.88MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<00:47, 1.70MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<00:41, 1.91MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<00:30, 2.56MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:37, 2.03MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:43, 1.77MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<00:33, 2.25MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:53<00:23, 3.09MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<01:02, 1.15MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:54<00:51, 1.38MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<00:37, 1.88MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:40, 1.70MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:56<00:42, 1.58MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:57<00:33, 2.00MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:57<00:23, 2.75MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<01:33, 681kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<01:12, 878kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<00:50, 1.22MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<00:47, 1.26MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<00:46, 1.29MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:00<00:34, 1.69MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:01<00:23, 2.35MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<01:16, 719kB/s] .vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:58, 936kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:41, 1.28MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:38, 1.31MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:38, 1.32MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:04<00:28, 1.73MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:05<00:20, 2.39MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:30, 1.52MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:28, 1.63MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:06<00:21, 2.14MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:08<00:21, 1.96MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:08<00:24, 1.72MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:08<00:19, 2.16MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:09<00:13, 2.96MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:10<00:56, 689kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:10<00:43, 887kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:10<00:29, 1.23MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:27, 1.27MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:26, 1.31MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:12<00:19, 1.70MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:13<00:12, 2.35MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:14<01:40, 303kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:14<01:12, 413kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:14<00:49, 581kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:37, 704kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:31, 823kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:16<00:22, 1.10MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:17<00:14, 1.55MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:33, 656kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 841M/862M [06:18<00:25, 849kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:18<00:17, 1.18MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:14, 1.23MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:20<00:14, 1.26MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:20<00:10, 1.63MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:20<00:06, 2.27MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:22<00:17, 794kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:22<00:13, 1.01MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:22<00:08, 1.38MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:06, 1.38MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:06, 1.37MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:24<00:04, 1.77MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:24<00:02, 2.45MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:06, 820kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:04, 1.03MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:26<00:02, 1.43MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:00, 1.41MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:00, 1.39MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:28<00:00, 1.79MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:01<116:34:32,  1.05s/it]  0%|          | 1057/400000 [00:01<81:23:26,  1.36it/s]  1%|          | 2264/400000 [00:01<56:48:13,  1.94it/s]  1%|          | 3397/400000 [00:01<39:39:08,  2.78it/s]  1%|          | 4628/400000 [00:01<27:40:23,  3.97it/s]  1%|▏         | 5836/400000 [00:01<19:18:53,  5.67it/s]  2%|▏         | 6871/400000 [00:01<13:29:16,  8.10it/s]  2%|▏         | 8113/400000 [00:01<9:24:51, 11.56it/s]   2%|▏         | 9265/400000 [00:01<6:34:24, 16.51it/s]  3%|▎         | 10364/400000 [00:01<4:35:29, 23.57it/s]  3%|▎         | 11531/400000 [00:02<3:12:25, 33.65it/s]  3%|▎         | 12667/400000 [00:02<2:14:28, 48.00it/s]  3%|▎         | 13826/400000 [00:02<1:34:01, 68.46it/s]  4%|▎         | 14960/400000 [00:02<1:05:47, 97.54it/s]  4%|▍         | 16148/400000 [00:02<46:04, 138.85it/s]   4%|▍         | 17385/400000 [00:02<32:18, 197.41it/s]  5%|▍         | 18613/400000 [00:02<22:41, 280.08it/s]  5%|▍         | 19837/400000 [00:02<15:59, 396.22it/s]  5%|▌         | 21036/400000 [00:02<11:19, 557.77it/s]  6%|▌         | 22258/400000 [00:02<08:03, 781.52it/s]  6%|▌         | 23453/400000 [00:03<05:46, 1085.37it/s]  6%|▌         | 24692/400000 [00:03<04:11, 1494.39it/s]  6%|▋         | 25926/400000 [00:03<03:04, 2029.50it/s]  7%|▋         | 27138/400000 [00:03<02:18, 2694.18it/s]  7%|▋         | 28340/400000 [00:03<01:45, 3511.30it/s]  7%|▋         | 29566/400000 [00:03<01:22, 4467.58it/s]  8%|▊         | 30780/400000 [00:03<01:06, 5512.28it/s]  8%|▊         | 31987/400000 [00:03<00:56, 6543.25it/s]  8%|▊         | 33180/400000 [00:03<00:49, 7457.66it/s]  9%|▊         | 34347/400000 [00:03<00:43, 8337.87it/s]  9%|▉         | 35578/400000 [00:04<00:39, 9230.37it/s]  9%|▉         | 36779/400000 [00:04<00:36, 9917.37it/s] 10%|▉         | 38002/400000 [00:04<00:34, 10512.99it/s] 10%|▉         | 39201/400000 [00:04<00:33, 10815.07it/s] 10%|█         | 40440/400000 [00:04<00:31, 11243.29it/s] 10%|█         | 41651/400000 [00:04<00:31, 11487.75it/s] 11%|█         | 42856/400000 [00:04<00:30, 11593.72it/s] 11%|█         | 44055/400000 [00:04<00:30, 11627.85it/s] 11%|█▏        | 45246/400000 [00:04<00:30, 11590.84it/s] 12%|█▏        | 46478/400000 [00:04<00:29, 11800.24it/s] 12%|█▏        | 47683/400000 [00:05<00:29, 11873.82it/s] 12%|█▏        | 48881/400000 [00:05<00:29, 11875.05it/s] 13%|█▎        | 50095/400000 [00:05<00:29, 11952.20it/s] 13%|█▎        | 51296/400000 [00:05<00:29, 11876.36it/s] 13%|█▎        | 52497/400000 [00:05<00:29, 11916.13it/s] 13%|█▎        | 53692/400000 [00:05<00:29, 11581.66it/s] 14%|█▎        | 54855/400000 [00:05<00:30, 11356.93it/s] 14%|█▍        | 56072/400000 [00:05<00:29, 11587.28it/s] 14%|█▍        | 57235/400000 [00:05<00:29, 11553.30it/s] 15%|█▍        | 58460/400000 [00:06<00:29, 11751.29it/s] 15%|█▍        | 59638/400000 [00:06<00:29, 11669.15it/s] 15%|█▌        | 60827/400000 [00:06<00:28, 11731.27it/s] 16%|█▌        | 62002/400000 [00:06<00:28, 11701.00it/s] 16%|█▌        | 63174/400000 [00:06<00:28, 11665.67it/s] 16%|█▌        | 64381/400000 [00:06<00:28, 11784.06it/s] 16%|█▋        | 65614/400000 [00:06<00:28, 11940.59it/s] 17%|█▋        | 66810/400000 [00:06<00:28, 11879.18it/s] 17%|█▋        | 68032/400000 [00:06<00:27, 11978.16it/s] 17%|█▋        | 69231/400000 [00:06<00:27, 11865.95it/s] 18%|█▊        | 70458/400000 [00:07<00:27, 11982.00it/s] 18%|█▊        | 71658/400000 [00:07<00:27, 11974.22it/s] 18%|█▊        | 72856/400000 [00:07<00:28, 11666.99it/s] 19%|█▊        | 74041/400000 [00:07<00:27, 11720.31it/s] 19%|█▉        | 75215/400000 [00:07<00:28, 11562.13it/s] 19%|█▉        | 76424/400000 [00:07<00:27, 11715.15it/s] 19%|█▉        | 77654/400000 [00:07<00:27, 11883.74it/s] 20%|█▉        | 78845/400000 [00:07<00:27, 11881.11it/s] 20%|██        | 80035/400000 [00:07<00:26, 11851.01it/s] 20%|██        | 81221/400000 [00:07<00:27, 11724.53it/s] 21%|██        | 82459/400000 [00:08<00:26, 11913.56it/s] 21%|██        | 83695/400000 [00:08<00:26, 12042.19it/s] 21%|██        | 84906/400000 [00:08<00:26, 12059.71it/s] 22%|██▏       | 86129/400000 [00:08<00:25, 12109.58it/s] 22%|██▏       | 87341/400000 [00:08<00:26, 11959.05it/s] 22%|██▏       | 88547/400000 [00:08<00:25, 11987.80it/s] 22%|██▏       | 89750/400000 [00:08<00:25, 11998.77it/s] 23%|██▎       | 90951/400000 [00:08<00:25, 11963.09it/s] 23%|██▎       | 92164/400000 [00:08<00:25, 12009.59it/s] 23%|██▎       | 93366/400000 [00:08<00:26, 11653.29it/s] 24%|██▎       | 94581/400000 [00:09<00:25, 11797.36it/s] 24%|██▍       | 95804/400000 [00:09<00:25, 11923.46it/s] 24%|██▍       | 97017/400000 [00:09<00:25, 11983.53it/s] 25%|██▍       | 98256/400000 [00:09<00:24, 12099.67it/s] 25%|██▍       | 99468/400000 [00:09<00:25, 11967.82it/s] 25%|██▌       | 100702/400000 [00:09<00:24, 12076.29it/s] 25%|██▌       | 101928/400000 [00:09<00:24, 12127.59it/s] 26%|██▌       | 103142/400000 [00:09<00:24, 12042.67it/s] 26%|██▌       | 104347/400000 [00:09<00:24, 12032.61it/s] 26%|██▋       | 105551/400000 [00:09<00:24, 11862.89it/s] 27%|██▋       | 106778/400000 [00:10<00:24, 11980.48it/s] 27%|██▋       | 108012/400000 [00:10<00:24, 12085.53it/s] 27%|██▋       | 109246/400000 [00:10<00:23, 12159.53it/s] 28%|██▊       | 110463/400000 [00:10<00:23, 12133.36it/s] 28%|██▊       | 111677/400000 [00:10<00:23, 12024.85it/s] 28%|██▊       | 112881/400000 [00:10<00:24, 11615.22it/s] 29%|██▊       | 114087/400000 [00:10<00:24, 11743.87it/s] 29%|██▉       | 115270/400000 [00:10<00:24, 11767.71it/s] 29%|██▉       | 116477/400000 [00:10<00:23, 11855.73it/s] 29%|██▉       | 117665/400000 [00:10<00:24, 11591.21it/s] 30%|██▉       | 118865/400000 [00:11<00:24, 11709.43it/s] 30%|███       | 120092/400000 [00:11<00:23, 11870.00it/s] 30%|███       | 121329/400000 [00:11<00:23, 12015.59it/s] 31%|███       | 122559/400000 [00:11<00:22, 12097.47it/s] 31%|███       | 123771/400000 [00:11<00:23, 11975.40it/s] 31%|███       | 124970/400000 [00:11<00:22, 11968.80it/s] 32%|███▏      | 126198/400000 [00:11<00:22, 12058.95it/s] 32%|███▏      | 127427/400000 [00:11<00:22, 12125.49it/s] 32%|███▏      | 128644/400000 [00:11<00:22, 12137.31it/s] 32%|███▏      | 129859/400000 [00:12<00:23, 11527.91it/s] 33%|███▎      | 131046/400000 [00:12<00:23, 11627.36it/s] 33%|███▎      | 132214/400000 [00:12<00:23, 11529.73it/s] 33%|███▎      | 133394/400000 [00:12<00:22, 11606.86it/s] 34%|███▎      | 134561/400000 [00:12<00:22, 11621.90it/s] 34%|███▍      | 135726/400000 [00:12<00:22, 11583.46it/s] 34%|███▍      | 136934/400000 [00:12<00:22, 11727.16it/s] 35%|███▍      | 138158/400000 [00:12<00:22, 11874.57it/s] 35%|███▍      | 139362/400000 [00:12<00:21, 11922.78it/s] 35%|███▌      | 140556/400000 [00:12<00:22, 11626.26it/s] 35%|███▌      | 141744/400000 [00:13<00:22, 11700.15it/s] 36%|███▌      | 142960/400000 [00:13<00:21, 11833.63it/s] 36%|███▌      | 144162/400000 [00:13<00:21, 11886.63it/s] 36%|███▋      | 145389/400000 [00:13<00:21, 11999.07it/s] 37%|███▋      | 146590/400000 [00:13<00:21, 11766.27it/s] 37%|███▋      | 147781/400000 [00:13<00:21, 11807.63it/s] 37%|███▋      | 148990/400000 [00:13<00:21, 11888.78it/s] 38%|███▊      | 150220/400000 [00:13<00:20, 12008.16it/s] 38%|███▊      | 151422/400000 [00:13<00:21, 11655.12it/s] 38%|███▊      | 152591/400000 [00:13<00:21, 11602.78it/s] 38%|███▊      | 153781/400000 [00:14<00:21, 11687.78it/s] 39%|███▉      | 155007/400000 [00:14<00:20, 11851.77it/s] 39%|███▉      | 156236/400000 [00:14<00:20, 11978.96it/s] 39%|███▉      | 157459/400000 [00:14<00:20, 12051.49it/s] 40%|███▉      | 158666/400000 [00:14<00:20, 12029.67it/s] 40%|███▉      | 159870/400000 [00:14<00:20, 11996.26it/s] 40%|████      | 161101/400000 [00:14<00:19, 12086.69it/s] 41%|████      | 162311/400000 [00:14<00:19, 12048.46it/s] 41%|████      | 163531/400000 [00:14<00:19, 12092.45it/s] 41%|████      | 164741/400000 [00:14<00:19, 11961.85it/s] 41%|████▏     | 165938/400000 [00:15<00:19, 11865.00it/s] 42%|████▏     | 167159/400000 [00:15<00:19, 11964.14it/s] 42%|████▏     | 168386/400000 [00:15<00:19, 12053.80it/s] 42%|████▏     | 169618/400000 [00:15<00:18, 12130.35it/s] 43%|████▎     | 170832/400000 [00:15<00:19, 11805.47it/s] 43%|████▎     | 172033/400000 [00:15<00:19, 11864.66it/s] 43%|████▎     | 173266/400000 [00:15<00:18, 11999.85it/s] 44%|████▎     | 174468/400000 [00:15<00:18, 12005.17it/s] 44%|████▍     | 175694/400000 [00:15<00:18, 12080.02it/s] 44%|████▍     | 176903/400000 [00:15<00:18, 11990.81it/s] 45%|████▍     | 178103/400000 [00:16<00:18, 11750.07it/s] 45%|████▍     | 179290/400000 [00:16<00:18, 11783.15it/s] 45%|████▌     | 180511/400000 [00:16<00:18, 11906.86it/s] 45%|████▌     | 181753/400000 [00:16<00:18, 12055.84it/s] 46%|████▌     | 182960/400000 [00:16<00:18, 11881.69it/s] 46%|████▌     | 184179/400000 [00:16<00:18, 11969.75it/s] 46%|████▋     | 185395/400000 [00:16<00:17, 12026.07it/s] 47%|████▋     | 186599/400000 [00:16<00:17, 12028.62it/s] 47%|████▋     | 187811/400000 [00:16<00:17, 12052.65it/s] 47%|████▋     | 189017/400000 [00:16<00:18, 11518.16it/s] 48%|████▊     | 190175/400000 [00:17<00:18, 11426.94it/s] 48%|████▊     | 191393/400000 [00:17<00:17, 11640.17it/s] 48%|████▊     | 192572/400000 [00:17<00:17, 11684.10it/s] 48%|████▊     | 193792/400000 [00:17<00:17, 11832.49it/s] 49%|████▊     | 194978/400000 [00:17<00:17, 11689.00it/s] 49%|████▉     | 196150/400000 [00:17<00:17, 11695.69it/s] 49%|████▉     | 197375/400000 [00:17<00:17, 11855.17it/s] 50%|████▉     | 198607/400000 [00:17<00:16, 11990.89it/s] 50%|████▉     | 199808/400000 [00:17<00:16, 11986.58it/s] 50%|█████     | 201008/400000 [00:18<00:16, 11889.40it/s] 51%|█████     | 202240/400000 [00:18<00:16, 12013.22it/s] 51%|█████     | 203459/400000 [00:18<00:16, 12064.07it/s] 51%|█████     | 204685/400000 [00:18<00:16, 12121.20it/s] 51%|█████▏    | 205911/400000 [00:18<00:15, 12162.03it/s] 52%|█████▏    | 207128/400000 [00:18<00:16, 11952.90it/s] 52%|█████▏    | 208325/400000 [00:18<00:16, 11952.01it/s] 52%|█████▏    | 209521/400000 [00:18<00:16, 11784.23it/s] 53%|█████▎    | 210709/400000 [00:18<00:16, 11812.59it/s] 53%|█████▎    | 211933/400000 [00:18<00:15, 11935.31it/s] 53%|█████▎    | 213128/400000 [00:19<00:15, 11790.37it/s] 54%|█████▎    | 214309/400000 [00:19<00:15, 11767.81it/s] 54%|█████▍    | 215508/400000 [00:19<00:15, 11831.40it/s] 54%|█████▍    | 216749/400000 [00:19<00:15, 11999.14it/s] 54%|█████▍    | 217950/400000 [00:19<00:15, 11958.04it/s] 55%|█████▍    | 219147/400000 [00:19<00:15, 11937.47it/s] 55%|█████▌    | 220357/400000 [00:19<00:14, 11983.26it/s] 55%|█████▌    | 221556/400000 [00:19<00:14, 11964.20it/s] 56%|█████▌    | 222795/400000 [00:19<00:14, 12088.77it/s] 56%|█████▌    | 224030/400000 [00:19<00:14, 12164.97it/s] 56%|█████▋    | 225247/400000 [00:20<00:14, 12065.11it/s] 57%|█████▋    | 226455/400000 [00:20<00:14, 11941.66it/s] 57%|█████▋    | 227665/400000 [00:20<00:14, 11987.52it/s] 57%|█████▋    | 228865/400000 [00:20<00:14, 11823.14it/s] 58%|█████▊    | 230095/400000 [00:20<00:14, 11961.78it/s] 58%|█████▊    | 231293/400000 [00:20<00:14, 11958.60it/s] 58%|█████▊    | 232490/400000 [00:20<00:14, 11901.20it/s] 58%|█████▊    | 233681/400000 [00:20<00:13, 11889.26it/s] 59%|█████▊    | 234902/400000 [00:20<00:13, 11983.19it/s] 59%|█████▉    | 236127/400000 [00:20<00:13, 12060.73it/s] 59%|█████▉    | 237334/400000 [00:21<00:13, 12038.90it/s] 60%|█████▉    | 238539/400000 [00:21<00:13, 12020.94it/s] 60%|█████▉    | 239742/400000 [00:21<00:13, 12022.80it/s] 60%|██████    | 240945/400000 [00:21<00:13, 11973.57it/s] 61%|██████    | 242181/400000 [00:21<00:13, 12086.12it/s] 61%|██████    | 243390/400000 [00:21<00:13, 11974.19it/s] 61%|██████    | 244588/400000 [00:21<00:12, 11962.71it/s] 61%|██████▏   | 245798/400000 [00:21<00:12, 12001.56it/s] 62%|██████▏   | 247015/400000 [00:21<00:12, 12049.10it/s] 62%|██████▏   | 248221/400000 [00:21<00:12, 11863.06it/s] 62%|██████▏   | 249409/400000 [00:22<00:12, 11674.37it/s] 63%|██████▎   | 250578/400000 [00:22<00:12, 11565.35it/s] 63%|██████▎   | 251750/400000 [00:22<00:12, 11608.48it/s] 63%|██████▎   | 252964/400000 [00:22<00:12, 11761.42it/s] 64%|██████▎   | 254200/400000 [00:22<00:12, 11933.66it/s] 64%|██████▍   | 255402/400000 [00:22<00:12, 11957.50it/s] 64%|██████▍   | 256599/400000 [00:22<00:12, 11518.72it/s] 64%|██████▍   | 257783/400000 [00:22<00:12, 11611.55it/s] 65%|██████▍   | 258987/400000 [00:22<00:12, 11735.50it/s] 65%|██████▌   | 260186/400000 [00:22<00:11, 11808.61it/s] 65%|██████▌   | 261402/400000 [00:23<00:11, 11909.22it/s] 66%|██████▌   | 262595/400000 [00:23<00:11, 11902.66it/s] 66%|██████▌   | 263839/400000 [00:23<00:11, 12057.15it/s] 66%|██████▋   | 265074/400000 [00:23<00:11, 12140.76it/s] 67%|██████▋   | 266300/400000 [00:23<00:10, 12174.01it/s] 67%|██████▋   | 267519/400000 [00:23<00:11, 11805.24it/s] 67%|██████▋   | 268703/400000 [00:23<00:11, 11742.07it/s] 67%|██████▋   | 269923/400000 [00:23<00:10, 11875.63it/s] 68%|██████▊   | 271148/400000 [00:23<00:10, 11984.13it/s] 68%|██████▊   | 272355/400000 [00:23<00:10, 12008.34it/s] 68%|██████▊   | 273557/400000 [00:24<00:10, 11968.07it/s] 69%|██████▊   | 274755/400000 [00:24<00:10, 11923.29it/s] 69%|██████▉   | 275982/400000 [00:24<00:10, 12024.37it/s] 69%|██████▉   | 277203/400000 [00:24<00:10, 12078.93it/s] 70%|██████▉   | 278422/400000 [00:24<00:10, 12110.48it/s] 70%|██████▉   | 279652/400000 [00:24<00:09, 12165.57it/s] 70%|███████   | 280869/400000 [00:24<00:09, 11968.44it/s] 71%|███████   | 282067/400000 [00:24<00:09, 11903.19it/s] 71%|███████   | 283259/400000 [00:24<00:09, 11830.68it/s] 71%|███████   | 284487/400000 [00:24<00:09, 11961.14it/s] 71%|███████▏  | 285684/400000 [00:25<00:09, 11836.86it/s] 72%|███████▏  | 286869/400000 [00:25<00:10, 10903.10it/s] 72%|███████▏  | 287975/400000 [00:25<00:10, 10297.94it/s] 72%|███████▏  | 289023/400000 [00:25<00:10, 10094.07it/s] 73%|███████▎  | 290046/400000 [00:25<00:11, 9814.72it/s]  73%|███████▎  | 291039/400000 [00:25<00:11, 9788.59it/s] 73%|███████▎  | 292092/400000 [00:25<00:10, 9997.67it/s] 73%|███████▎  | 293231/400000 [00:25<00:10, 10375.44it/s] 74%|███████▎  | 294441/400000 [00:25<00:09, 10835.93it/s] 74%|███████▍  | 295658/400000 [00:26<00:09, 11202.43it/s] 74%|███████▍  | 296790/400000 [00:26<00:09, 11228.67it/s] 74%|███████▍  | 297927/400000 [00:26<00:09, 11269.11it/s] 75%|███████▍  | 299068/400000 [00:26<00:08, 11309.31it/s] 75%|███████▌  | 300280/400000 [00:26<00:08, 11538.34it/s] 75%|███████▌  | 301509/400000 [00:26<00:08, 11751.99it/s] 76%|███████▌  | 302688/400000 [00:26<00:08, 11658.35it/s] 76%|███████▌  | 303858/400000 [00:26<00:08, 11667.74it/s] 76%|███████▋  | 305027/400000 [00:26<00:08, 11637.20it/s] 77%|███████▋  | 306244/400000 [00:26<00:07, 11790.82it/s] 77%|███████▋  | 307425/400000 [00:27<00:08, 11247.31it/s] 77%|███████▋  | 308561/400000 [00:27<00:08, 11279.14it/s] 77%|███████▋  | 309788/400000 [00:27<00:07, 11559.12it/s] 78%|███████▊  | 311009/400000 [00:27<00:07, 11744.95it/s] 78%|███████▊  | 312221/400000 [00:27<00:07, 11854.77it/s] 78%|███████▊  | 313410/400000 [00:27<00:07, 11659.83it/s] 79%|███████▊  | 314579/400000 [00:27<00:07, 11486.92it/s] 79%|███████▉  | 315731/400000 [00:27<00:07, 11336.65it/s] 79%|███████▉  | 316867/400000 [00:27<00:07, 11229.93it/s] 80%|███████▉  | 318069/400000 [00:28<00:07, 11455.74it/s] 80%|███████▉  | 319301/400000 [00:28<00:06, 11700.63it/s] 80%|████████  | 320475/400000 [00:28<00:06, 11435.52it/s] 80%|████████  | 321665/400000 [00:28<00:06, 11569.65it/s] 81%|████████  | 322836/400000 [00:28<00:06, 11607.61it/s] 81%|████████  | 324008/400000 [00:28<00:06, 11638.81it/s] 81%|████████▏ | 325245/400000 [00:28<00:06, 11848.39it/s] 82%|████████▏ | 326432/400000 [00:28<00:06, 11726.79it/s] 82%|████████▏ | 327648/400000 [00:28<00:06, 11852.24it/s] 82%|████████▏ | 328835/400000 [00:28<00:06, 11705.93it/s] 83%|████████▎ | 330008/400000 [00:29<00:06, 11611.73it/s] 83%|████████▎ | 331236/400000 [00:29<00:05, 11803.94it/s] 83%|████████▎ | 332418/400000 [00:29<00:05, 11674.98it/s] 83%|████████▎ | 333619/400000 [00:29<00:05, 11772.27it/s] 84%|████████▎ | 334848/400000 [00:29<00:05, 11922.09it/s] 84%|████████▍ | 336054/400000 [00:29<00:05, 11962.38it/s] 84%|████████▍ | 337252/400000 [00:29<00:05, 11804.41it/s] 85%|████████▍ | 338434/400000 [00:29<00:05, 11718.40it/s] 85%|████████▍ | 339661/400000 [00:29<00:05, 11876.78it/s] 85%|████████▌ | 340874/400000 [00:29<00:04, 11951.21it/s] 86%|████████▌ | 342071/400000 [00:30<00:04, 11803.99it/s] 86%|████████▌ | 343253/400000 [00:30<00:04, 11620.65it/s] 86%|████████▌ | 344417/400000 [00:30<00:04, 11613.21it/s] 86%|████████▋ | 345626/400000 [00:30<00:04, 11749.67it/s] 87%|████████▋ | 346808/400000 [00:30<00:04, 11769.81it/s] 87%|████████▋ | 347986/400000 [00:30<00:04, 11757.80it/s] 87%|████████▋ | 349216/400000 [00:30<00:04, 11914.09it/s] 88%|████████▊ | 350409/400000 [00:30<00:04, 11848.85it/s] 88%|████████▊ | 351654/400000 [00:30<00:04, 12021.70it/s] 88%|████████▊ | 352858/400000 [00:30<00:03, 12016.09it/s] 89%|████████▊ | 354061/400000 [00:31<00:03, 11957.71it/s] 89%|████████▉ | 355280/400000 [00:31<00:03, 12025.00it/s] 89%|████████▉ | 356484/400000 [00:31<00:03, 11884.66it/s] 89%|████████▉ | 357674/400000 [00:31<00:03, 11775.24it/s] 90%|████████▉ | 358887/400000 [00:31<00:03, 11877.71it/s] 90%|█████████ | 360109/400000 [00:31<00:03, 11977.15it/s] 90%|█████████ | 361308/400000 [00:31<00:03, 11701.80it/s] 91%|█████████ | 362481/400000 [00:31<00:03, 11600.50it/s] 91%|█████████ | 363712/400000 [00:31<00:03, 11801.84it/s] 91%|█████████ | 364895/400000 [00:31<00:02, 11727.90it/s] 92%|█████████▏| 366081/400000 [00:32<00:02, 11765.91it/s] 92%|█████████▏| 367279/400000 [00:32<00:02, 11828.44it/s] 92%|█████████▏| 368463/400000 [00:32<00:02, 11789.55it/s] 92%|█████████▏| 369691/400000 [00:32<00:02, 11931.33it/s] 93%|█████████▎| 370918/400000 [00:32<00:02, 12030.84it/s] 93%|█████████▎| 372154/400000 [00:32<00:02, 12125.54it/s] 93%|█████████▎| 373368/400000 [00:32<00:02, 12106.14it/s] 94%|█████████▎| 374580/400000 [00:32<00:02, 11875.67it/s] 94%|█████████▍| 375786/400000 [00:32<00:02, 11930.01it/s] 94%|█████████▍| 376980/400000 [00:32<00:01, 11753.45it/s] 95%|█████████▍| 378166/400000 [00:33<00:01, 11782.65it/s] 95%|█████████▍| 379354/400000 [00:33<00:01, 11810.09it/s] 95%|█████████▌| 380536/400000 [00:33<00:01, 11433.69it/s] 95%|█████████▌| 381773/400000 [00:33<00:01, 11699.30it/s] 96%|█████████▌| 382999/400000 [00:33<00:01, 11860.85it/s] 96%|█████████▌| 384204/400000 [00:33<00:01, 11914.54it/s] 96%|█████████▋| 385398/400000 [00:33<00:01, 11899.35it/s] 97%|█████████▋| 386590/400000 [00:33<00:01, 11815.43it/s] 97%|█████████▋| 387789/400000 [00:33<00:01, 11866.13it/s] 97%|█████████▋| 389023/400000 [00:34<00:00, 12004.27it/s] 98%|█████████▊| 390264/400000 [00:34<00:00, 12120.56it/s] 98%|█████████▊| 391478/400000 [00:34<00:00, 12117.06it/s] 98%|█████████▊| 392691/400000 [00:34<00:00, 11928.96it/s] 98%|█████████▊| 393886/400000 [00:34<00:00, 11783.02it/s] 99%|█████████▉| 395111/400000 [00:34<00:00, 11916.71it/s] 99%|█████████▉| 396327/400000 [00:34<00:00, 11988.43it/s] 99%|█████████▉| 397555/400000 [00:34<00:00, 12074.30it/s]100%|█████████▉| 398764/400000 [00:34<00:00, 11589.85it/s]100%|█████████▉| 399928/400000 [00:34<00:00, 11540.12it/s]100%|█████████▉| 399999/400000 [00:34<00:00, 11447.29it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7efc504a7710>, <torchtext.data.dataset.TabularDataset object at 0x7efc504a7860>, <torchtext.vocab.Vocab object at 0x7efc504a7780>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 13.13 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 23.17 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 14.64 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 14.64 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  7.82 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  7.82 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  8.45 file/s]2020-06-25 00:16:38.007648: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-06-25 00:16:38.010752: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-06-25 00:16:38.010913: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x559c15d7ad90 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-25 00:16:38.010927: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 37%|███▋      | 3645440/9912422 [00:00<00:00, 36427858.40it/s]9920512it [00:00, 33596262.61it/s]                             
0it [00:00, ?it/s]32768it [00:00, 629529.83it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 154347.40it/s]1654784it [00:00, 11282665.62it/s]                         
0it [00:00, ?it/s]8192it [00:00, 168805.77it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
