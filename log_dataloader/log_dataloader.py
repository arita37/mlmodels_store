
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/d5c43e974a89bacfc0e66fbd41ec7beedf38a669', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': 'd5c43e974a89bacfc0e66fbd41ec7beedf38a669', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/d5c43e974a89bacfc0e66fbd41ec7beedf38a669

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/d5c43e974a89bacfc0e66fbd41ec7beedf38a669

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/d5c43e974a89bacfc0e66fbd41ec7beedf38a669

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f308590a6a8> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f308590a6a8> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f30f0633488> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f30f0633488> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f310f852ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f310f852ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f309d969488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f309d969488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f309d969488> , (data_info, **args) 

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
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 685, in parser_f
    return _read(filepath_or_buffer, kwds)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 457, in _read
    parser = TextFileReader(fp_or_buf, **kwds)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 895, in __init__
    self._make_engine(self.engine)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1135, in _make_engine
    self._engine = CParserWrapper(self.f, **self.options)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1917, in __init__
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
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 685, in parser_f
    return _read(filepath_or_buffer, kwds)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 457, in _read
    parser = TextFileReader(fp_or_buf, **kwds)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 895, in __init__
    self._make_engine(self.engine)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1135, in _make_engine
    self._engine = CParserWrapper(self.f, **self.options)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1917, in __init__
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
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/numpy/lib/npyio.py", line 416, in load
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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 18%|█▊        | 1753088/9912422 [00:00<00:00, 17451053.67it/s] 76%|███████▌  | 7520256/9912422 [00:00<00:00, 21974551.67it/s]9920512it [00:00, 27513600.31it/s]                             
0it [00:00, ?it/s]32768it [00:00, 659948.78it/s]
0it [00:00, ?it/s]  6%|▋         | 106496/1648877 [00:00<00:01, 1032775.88it/s]1654784it [00:00, 12588039.16it/s]                           
0it [00:00, ?it/s]8192it [00:00, 158138.67it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f308583df28>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f307c385ba8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f309d969158> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f309d969158> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f309d969158> , (data_info, **args) 

  CIFAR10 

  Dataset Name is :  cifar10 

Dl Completed...: 0 url [00:00, ? url/s]
Dl Size...: 0 MiB [00:00, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...: 0 MiB [00:00, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/urllib3/connectionpool.py:988: InsecureRequestWarning: Unverified HTTPS request is being made to host 'www.cs.toronto.edu'. Adding certificate verification is strongly advised. See: https://urllib3.readthedocs.io/en/latest/advanced-usage.html#ssl-warnings
  InsecureRequestWarning,
Dl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   0%|          | 0/162 [00:00<?, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   1%|          | 1/162 [00:00<01:27,  1.83 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:27,  1.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:27,  1.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:26,  1.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:26,  1.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:25,  1.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:25,  1.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:24,  1.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<00:59,  2.59 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:59,  2.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:59,  2.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:58,  2.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:58,  2.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:57,  2.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:57,  2.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:57,  2.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:56,  2.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:56,  2.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:55,  2.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  11%|█         | 18/162 [00:00<00:39,  3.66 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:39,  3.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:39,  3.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:38,  3.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:38,  3.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:38,  3.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:37,  3.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:37,  3.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:37,  3.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:37,  3.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:36,  3.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  17%|█▋        | 28/162 [00:00<00:26,  5.14 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:26,  5.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:25,  5.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:25,  5.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:25,  5.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:25,  5.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:25,  5.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:24,  5.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:24,  5.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:24,  5.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  23%|██▎       | 37/162 [00:00<00:17,  7.16 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:17,  7.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:17,  7.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:17,  7.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:17,  7.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:16,  7.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:16,  7.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:16,  7.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:16,  7.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:16,  7.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:16,  7.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  29%|██▉       | 47/162 [00:01<00:11,  9.93 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:11,  9.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:11,  9.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:11,  9.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:11,  9.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:11,  9.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:11,  9.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:10,  9.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:10,  9.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  34%|███▍      | 55/162 [00:01<00:08, 13.36 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:08, 13.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:07, 13.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:07, 13.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:07, 13.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:07, 13.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:07, 13.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:07, 13.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:07, 13.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  39%|███▉      | 63/162 [00:01<00:05, 17.72 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:05, 17.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:05, 17.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:05, 17.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:05, 17.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:05, 17.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:05, 17.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:05, 17.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:05, 17.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  44%|████▍     | 71/162 [00:01<00:03, 23.05 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:03, 23.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:03, 23.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:03, 23.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:03, 23.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:03, 23.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:03, 23.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:03, 23.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:03, 23.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 29.12 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 29.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 29.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 29.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 29.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 29.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 29.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 29.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 29.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 35.81 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 35.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 35.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:02, 35.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:02, 35.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 35.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 35.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 35.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 35.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 35.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 42.89 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 42.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 42.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 42.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 42.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 42.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 42.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 42.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 42.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 49.61 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 49.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 49.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 49.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 49.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 49.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 49.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:01, 49.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:01, 49.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:01, 49.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 56.05 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 56.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 56.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 56.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 56.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 56.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 56.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 56.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 56.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 56.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 61.82 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 61.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 61.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 61.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 61.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 61.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 61.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 61.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 61.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 61.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 66.41 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 66.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 66.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 66.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 66.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 66.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 66.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 66.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 66.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 66.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 70.48 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 70.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 70.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 70.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 70.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 70.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 70.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 70.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 70.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 70.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 73.59 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 73.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 73.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 73.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 73.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 73.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 73.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 73.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 73.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 73.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 76.49 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 76.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 76.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 76.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 76.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 76.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.53s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.53s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 76.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.53s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 76.49 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.75s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.53s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 76.49 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.75s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.75s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 34.07 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.76s/ url]
0 examples [00:00, ? examples/s]2020-09-03 06:07:46.993582: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-09-03 06:07:47.010786: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-09-03 06:07:47.011414: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55bcf4b2eab0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-09-03 06:07:47.011447: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  3.05 examples/s]106 examples [00:00,  4.35 examples/s]212 examples [00:00,  6.21 examples/s]313 examples [00:00,  8.84 examples/s]417 examples [00:00, 12.58 examples/s]523 examples [00:00, 17.89 examples/s]628 examples [00:00, 25.37 examples/s]734 examples [00:01, 35.87 examples/s]836 examples [00:01, 50.47 examples/s]936 examples [00:01, 70.58 examples/s]1040 examples [00:01, 97.96 examples/s]1146 examples [00:01, 134.58 examples/s]1253 examples [00:01, 182.40 examples/s]1360 examples [00:01, 242.82 examples/s]1466 examples [00:01, 315.66 examples/s]1572 examples [00:01, 399.72 examples/s]1679 examples [00:01, 491.94 examples/s]1785 examples [00:02, 585.75 examples/s]1891 examples [00:02, 675.90 examples/s]1997 examples [00:02, 756.95 examples/s]2103 examples [00:02, 824.87 examples/s]2209 examples [00:02, 881.94 examples/s]2315 examples [00:02, 924.68 examples/s]2420 examples [00:02, 958.24 examples/s]2525 examples [00:02, 980.41 examples/s]2630 examples [00:02, 997.64 examples/s]2735 examples [00:02, 1001.14 examples/s]2839 examples [00:03, 1001.18 examples/s]2943 examples [00:03, 1011.69 examples/s]3046 examples [00:03, 1005.17 examples/s]3151 examples [00:03, 1017.83 examples/s]3256 examples [00:03, 1024.83 examples/s]3360 examples [00:03, 1027.94 examples/s]3466 examples [00:03, 1034.96 examples/s]3570 examples [00:03, 1033.05 examples/s]3674 examples [00:03, 1024.10 examples/s]3780 examples [00:03, 1033.51 examples/s]3884 examples [00:04, 1028.71 examples/s]3988 examples [00:04, 1030.53 examples/s]4093 examples [00:04, 1036.29 examples/s]4197 examples [00:04, 1036.97 examples/s]4304 examples [00:04, 1044.67 examples/s]4411 examples [00:04, 1051.83 examples/s]4518 examples [00:04, 1055.72 examples/s]4624 examples [00:04, 1051.39 examples/s]4731 examples [00:04, 1055.65 examples/s]4837 examples [00:04, 1045.26 examples/s]4944 examples [00:05, 1050.44 examples/s]5050 examples [00:05, 1024.85 examples/s]5153 examples [00:05, 1021.24 examples/s]5256 examples [00:05, 1013.51 examples/s]5359 examples [00:05, 1016.85 examples/s]5461 examples [00:05, 1017.20 examples/s]5566 examples [00:05, 1026.44 examples/s]5669 examples [00:05, 1019.59 examples/s]5774 examples [00:05, 1028.28 examples/s]5877 examples [00:05, 1025.41 examples/s]5981 examples [00:06, 1029.57 examples/s]6086 examples [00:06, 1033.61 examples/s]6190 examples [00:06, 1033.74 examples/s]6296 examples [00:06, 1040.19 examples/s]6401 examples [00:06, 1031.58 examples/s]6505 examples [00:06, 1032.10 examples/s]6611 examples [00:06, 1040.11 examples/s]6716 examples [00:06, 1032.47 examples/s]6823 examples [00:06, 1041.13 examples/s]6928 examples [00:07, 1033.28 examples/s]7032 examples [00:07, 1029.99 examples/s]7136 examples [00:07, 1025.77 examples/s]7241 examples [00:07, 1032.35 examples/s]7347 examples [00:07, 1038.67 examples/s]7454 examples [00:07, 1046.60 examples/s]7559 examples [00:07, 1044.50 examples/s]7665 examples [00:07, 1048.00 examples/s]7770 examples [00:07, 1048.47 examples/s]7877 examples [00:07, 1052.03 examples/s]7983 examples [00:08, 1052.37 examples/s]8089 examples [00:08, 1047.21 examples/s]8195 examples [00:08, 1049.33 examples/s]8300 examples [00:08, 1047.79 examples/s]8405 examples [00:08, 1035.20 examples/s]8509 examples [00:08, 1027.12 examples/s]8615 examples [00:08, 1035.40 examples/s]8720 examples [00:08, 1037.63 examples/s]8824 examples [00:08, 1035.64 examples/s]8932 examples [00:08, 1046.19 examples/s]9037 examples [00:09, 1030.14 examples/s]9141 examples [00:09, 1027.36 examples/s]9246 examples [00:09, 1033.51 examples/s]9350 examples [00:09, 1021.18 examples/s]9454 examples [00:09, 1025.64 examples/s]9559 examples [00:09, 1032.52 examples/s]9663 examples [00:09, 1029.16 examples/s]9767 examples [00:09, 1030.28 examples/s]9871 examples [00:09, 1023.74 examples/s]9974 examples [00:09, 1025.60 examples/s]10077 examples [00:10, 972.81 examples/s]10180 examples [00:10, 987.10 examples/s]10283 examples [00:10, 998.27 examples/s]10384 examples [00:10, 1000.16 examples/s]10489 examples [00:10, 1012.60 examples/s]10596 examples [00:10, 1027.01 examples/s]10699 examples [00:10, 1025.41 examples/s]10805 examples [00:10, 1034.01 examples/s]10909 examples [00:10, 1034.49 examples/s]11013 examples [00:10, 1010.71 examples/s]11116 examples [00:11, 1015.75 examples/s]11222 examples [00:11, 1027.78 examples/s]11329 examples [00:11, 1037.47 examples/s]11435 examples [00:11, 1041.48 examples/s]11540 examples [00:11, 1041.45 examples/s]11645 examples [00:11, 1038.60 examples/s]11749 examples [00:11, 1034.98 examples/s]11855 examples [00:11, 1040.31 examples/s]11960 examples [00:11, 1038.37 examples/s]12064 examples [00:11, 1036.81 examples/s]12168 examples [00:12, 1033.95 examples/s]12272 examples [00:12, 1028.67 examples/s]12375 examples [00:12, 1020.72 examples/s]12479 examples [00:12, 1024.29 examples/s]12583 examples [00:12, 1026.80 examples/s]12686 examples [00:12, 1021.81 examples/s]12791 examples [00:12, 1028.88 examples/s]12898 examples [00:12, 1039.63 examples/s]13005 examples [00:12, 1047.58 examples/s]13110 examples [00:13, 1023.29 examples/s]13216 examples [00:13, 1032.03 examples/s]13321 examples [00:13, 1036.99 examples/s]13427 examples [00:13, 1041.68 examples/s]13532 examples [00:13, 1038.01 examples/s]13636 examples [00:13, 1033.31 examples/s]13741 examples [00:13, 1036.06 examples/s]13846 examples [00:13, 1037.82 examples/s]13952 examples [00:13, 1042.74 examples/s]14058 examples [00:13, 1045.81 examples/s]14163 examples [00:14, 1035.23 examples/s]14269 examples [00:14, 1040.00 examples/s]14374 examples [00:14, 1042.16 examples/s]14479 examples [00:14, 1038.36 examples/s]14585 examples [00:14, 1042.41 examples/s]14690 examples [00:14, 1040.07 examples/s]14795 examples [00:14, 1042.35 examples/s]14900 examples [00:14, 1043.30 examples/s]15006 examples [00:14, 1046.46 examples/s]15111 examples [00:14, 1046.20 examples/s]15216 examples [00:15, 1045.13 examples/s]15323 examples [00:15, 1050.10 examples/s]15429 examples [00:15, 1047.34 examples/s]15534 examples [00:15, 1047.11 examples/s]15639 examples [00:15, 1034.96 examples/s]15743 examples [00:15, 1035.77 examples/s]15847 examples [00:15, 1011.53 examples/s]15949 examples [00:15, 1013.24 examples/s]16054 examples [00:15, 1021.79 examples/s]16160 examples [00:15, 1030.52 examples/s]16264 examples [00:16, 1029.82 examples/s]16370 examples [00:16, 1036.06 examples/s]16474 examples [00:16, 1033.22 examples/s]16578 examples [00:16, 1035.24 examples/s]16682 examples [00:16, 1014.04 examples/s]16784 examples [00:16, 1015.57 examples/s]16890 examples [00:16, 1026.43 examples/s]16996 examples [00:16, 1035.92 examples/s]17103 examples [00:16, 1043.02 examples/s]17210 examples [00:16, 1048.94 examples/s]17315 examples [00:17, 1046.38 examples/s]17420 examples [00:17, 1020.54 examples/s]17526 examples [00:17, 1028.35 examples/s]17629 examples [00:17, 1024.16 examples/s]17735 examples [00:17, 1033.78 examples/s]17839 examples [00:17, 1031.80 examples/s]17943 examples [00:17, 1033.38 examples/s]18049 examples [00:17, 1040.52 examples/s]18154 examples [00:17, 1039.60 examples/s]18259 examples [00:17, 1041.26 examples/s]18364 examples [00:18, 1041.07 examples/s]18469 examples [00:18, 1042.98 examples/s]18575 examples [00:18, 1048.03 examples/s]18680 examples [00:18, 1044.26 examples/s]18785 examples [00:18, 1043.93 examples/s]18890 examples [00:18, 1035.77 examples/s]18994 examples [00:18, 1036.55 examples/s]19099 examples [00:18, 1038.10 examples/s]19204 examples [00:18, 1041.18 examples/s]19309 examples [00:18, 1039.14 examples/s]19413 examples [00:19, 1035.00 examples/s]19519 examples [00:19, 1040.71 examples/s]19625 examples [00:19, 1045.31 examples/s]19730 examples [00:19, 977.76 examples/s] 19836 examples [00:19, 999.51 examples/s]19940 examples [00:19, 1009.33 examples/s]20042 examples [00:19, 970.54 examples/s] 20149 examples [00:19, 995.93 examples/s]20253 examples [00:19, 1007.88 examples/s]20358 examples [00:20, 1019.60 examples/s]20462 examples [00:20, 1025.61 examples/s]20565 examples [00:20, 1004.42 examples/s]20670 examples [00:20, 1015.00 examples/s]20772 examples [00:20, 1014.55 examples/s]20874 examples [00:20, 985.37 examples/s] 20973 examples [00:20, 974.28 examples/s]21071 examples [00:20, 954.19 examples/s]21175 examples [00:20, 977.28 examples/s]21278 examples [00:20, 992.34 examples/s]21378 examples [00:21, 976.95 examples/s]21476 examples [00:21, 959.17 examples/s]21582 examples [00:21, 984.85 examples/s]21686 examples [00:21, 999.96 examples/s]21790 examples [00:21, 1009.17 examples/s]21892 examples [00:21, 1006.54 examples/s]21993 examples [00:21, 1006.91 examples/s]22097 examples [00:21, 1014.99 examples/s]22203 examples [00:21, 1026.09 examples/s]22308 examples [00:21, 1031.43 examples/s]22414 examples [00:22, 1037.61 examples/s]22520 examples [00:22, 1043.63 examples/s]22627 examples [00:22, 1050.94 examples/s]22734 examples [00:22, 1054.22 examples/s]22840 examples [00:22, 1043.15 examples/s]22945 examples [00:22, 1033.84 examples/s]23049 examples [00:22, 1031.92 examples/s]23154 examples [00:22, 1035.77 examples/s]23258 examples [00:22, 1032.94 examples/s]23363 examples [00:22, 1037.49 examples/s]23467 examples [00:23, 1037.05 examples/s]23571 examples [00:23, 1037.60 examples/s]23676 examples [00:23, 1040.29 examples/s]23782 examples [00:23, 1043.55 examples/s]23887 examples [00:23, 983.77 examples/s] 23988 examples [00:23, 989.17 examples/s]24088 examples [00:23, 984.50 examples/s]24195 examples [00:23, 1005.65 examples/s]24302 examples [00:23, 1023.54 examples/s]24408 examples [00:24, 1033.35 examples/s]24512 examples [00:24, 1030.06 examples/s]24617 examples [00:24, 1035.66 examples/s]24721 examples [00:24, 1031.93 examples/s]24826 examples [00:24, 1034.58 examples/s]24932 examples [00:24, 1040.99 examples/s]25037 examples [00:24, 1041.63 examples/s]25142 examples [00:24, 1040.02 examples/s]25249 examples [00:24, 1046.59 examples/s]25356 examples [00:24, 1052.69 examples/s]25463 examples [00:25, 1056.06 examples/s]25569 examples [00:25, 1051.37 examples/s]25675 examples [00:25, 1052.27 examples/s]25781 examples [00:25, 1051.61 examples/s]25887 examples [00:25, 1047.68 examples/s]25992 examples [00:25, 1043.28 examples/s]26097 examples [00:25, 1040.18 examples/s]26202 examples [00:25, 1037.73 examples/s]26308 examples [00:25, 1043.22 examples/s]26413 examples [00:25, 1044.44 examples/s]26518 examples [00:26, 1032.93 examples/s]26622 examples [00:26, 1028.93 examples/s]26726 examples [00:26, 1030.32 examples/s]26830 examples [00:26, 999.22 examples/s] 26937 examples [00:26, 1017.99 examples/s]27043 examples [00:26, 1029.65 examples/s]27148 examples [00:26, 1033.01 examples/s]27253 examples [00:26, 1036.70 examples/s]27360 examples [00:26, 1043.80 examples/s]27467 examples [00:26, 1050.69 examples/s]27573 examples [00:27, 1042.13 examples/s]27678 examples [00:27, 1018.44 examples/s]27781 examples [00:27, 959.16 examples/s] 27885 examples [00:27, 981.58 examples/s]27990 examples [00:27, 999.45 examples/s]28096 examples [00:27, 1014.82 examples/s]28200 examples [00:27, 1020.42 examples/s]28303 examples [00:27, 1006.67 examples/s]28408 examples [00:27, 1016.70 examples/s]28514 examples [00:28, 1026.87 examples/s]28619 examples [00:28, 1030.68 examples/s]28723 examples [00:28, 1014.71 examples/s]28829 examples [00:28, 1027.27 examples/s]28932 examples [00:28, 1015.84 examples/s]29037 examples [00:28, 1025.34 examples/s]29142 examples [00:28, 1031.33 examples/s]29246 examples [00:28, 1033.55 examples/s]29350 examples [00:28, 1030.93 examples/s]29454 examples [00:28, 1026.82 examples/s]29558 examples [00:29, 1029.50 examples/s]29663 examples [00:29, 1033.57 examples/s]29767 examples [00:29, 1034.92 examples/s]29872 examples [00:29, 1039.17 examples/s]29977 examples [00:29, 1040.04 examples/s]30082 examples [00:29, 987.50 examples/s] 30187 examples [00:29, 1003.54 examples/s]30290 examples [00:29, 1009.52 examples/s]30394 examples [00:29, 1018.43 examples/s]30499 examples [00:29, 1025.16 examples/s]30602 examples [00:30, 1022.44 examples/s]30706 examples [00:30, 1026.25 examples/s]30810 examples [00:30, 1029.79 examples/s]30916 examples [00:30, 1037.53 examples/s]31020 examples [00:30, 1037.76 examples/s]31124 examples [00:30, 1037.74 examples/s]31229 examples [00:30, 1040.32 examples/s]31334 examples [00:30, 1033.06 examples/s]31438 examples [00:30, 1021.62 examples/s]31545 examples [00:30, 1032.88 examples/s]31649 examples [00:31, 1032.32 examples/s]31754 examples [00:31, 1036.49 examples/s]31858 examples [00:31, 1036.26 examples/s]31963 examples [00:31, 1038.85 examples/s]32067 examples [00:31, 1033.07 examples/s]32171 examples [00:31, 1023.28 examples/s]32275 examples [00:31, 1028.21 examples/s]32380 examples [00:31, 1032.02 examples/s]32485 examples [00:31, 1034.74 examples/s]32589 examples [00:31, 1030.98 examples/s]32694 examples [00:32, 1034.78 examples/s]32798 examples [00:32, 1033.67 examples/s]32902 examples [00:32, 1033.32 examples/s]33006 examples [00:32, 1029.46 examples/s]33112 examples [00:32, 1035.58 examples/s]33216 examples [00:32, 1028.19 examples/s]33319 examples [00:32, 1024.85 examples/s]33423 examples [00:32, 1027.78 examples/s]33526 examples [00:32, 1023.80 examples/s]33631 examples [00:32, 1030.79 examples/s]33736 examples [00:33, 1034.06 examples/s]33840 examples [00:33, 1027.21 examples/s]33943 examples [00:33, 1028.03 examples/s]34046 examples [00:33, 1015.29 examples/s]34151 examples [00:33, 1023.34 examples/s]34254 examples [00:33, 1024.92 examples/s]34360 examples [00:33, 1033.70 examples/s]34464 examples [00:33, 1018.00 examples/s]34568 examples [00:33, 1023.53 examples/s]34671 examples [00:33, 1023.02 examples/s]34775 examples [00:34, 1025.40 examples/s]34880 examples [00:34, 1031.93 examples/s]34984 examples [00:34, 1032.20 examples/s]35088 examples [00:34, 1024.00 examples/s]35191 examples [00:34, 1005.69 examples/s]35295 examples [00:34, 1012.72 examples/s]35399 examples [00:34, 1020.74 examples/s]35503 examples [00:34, 1024.51 examples/s]35606 examples [00:34, 1019.57 examples/s]35711 examples [00:35, 1027.24 examples/s]35814 examples [00:35, 1026.54 examples/s]35919 examples [00:35, 1031.26 examples/s]36023 examples [00:35, 1015.07 examples/s]36125 examples [00:35, 1014.53 examples/s]36229 examples [00:35, 1021.59 examples/s]36332 examples [00:35, 1016.97 examples/s]36436 examples [00:35, 1023.74 examples/s]36541 examples [00:35, 1030.50 examples/s]36646 examples [00:35, 1034.65 examples/s]36750 examples [00:36, 1027.28 examples/s]36853 examples [00:36, 1022.70 examples/s]36956 examples [00:36, 1022.73 examples/s]37059 examples [00:36, 1012.00 examples/s]37163 examples [00:36, 1019.93 examples/s]37266 examples [00:36, 1018.37 examples/s]37370 examples [00:36, 1024.10 examples/s]37475 examples [00:36, 1029.01 examples/s]37578 examples [00:36, 1028.79 examples/s]37683 examples [00:36, 1033.54 examples/s]37789 examples [00:37, 1039.49 examples/s]37893 examples [00:37, 1037.16 examples/s]37998 examples [00:37, 1038.40 examples/s]38102 examples [00:37, 1030.57 examples/s]38206 examples [00:37, 1031.11 examples/s]38310 examples [00:37, 1033.67 examples/s]38414 examples [00:37, 1030.39 examples/s]38518 examples [00:37, 1032.89 examples/s]38622 examples [00:37, 1033.68 examples/s]38726 examples [00:37, 1033.59 examples/s]38831 examples [00:38, 1037.60 examples/s]38935 examples [00:38, 1034.73 examples/s]39039 examples [00:38, 1033.10 examples/s]39143 examples [00:38, 1025.87 examples/s]39247 examples [00:38, 1028.04 examples/s]39350 examples [00:38, 1028.38 examples/s]39453 examples [00:38, 1028.85 examples/s]39556 examples [00:38, 1022.64 examples/s]39661 examples [00:38, 1029.29 examples/s]39766 examples [00:38, 1033.03 examples/s]39871 examples [00:39, 1035.12 examples/s]39976 examples [00:39, 1037.06 examples/s]40080 examples [00:39, 986.40 examples/s] 40183 examples [00:39, 996.41 examples/s]40288 examples [00:39, 1010.23 examples/s]40393 examples [00:39, 1021.45 examples/s]40497 examples [00:39, 1026.38 examples/s]40601 examples [00:39, 1029.07 examples/s]40705 examples [00:39, 1008.14 examples/s]40809 examples [00:39, 1015.82 examples/s]40914 examples [00:40, 1025.18 examples/s]41019 examples [00:40, 1030.61 examples/s]41124 examples [00:40, 1034.98 examples/s]41228 examples [00:40, 1034.37 examples/s]41332 examples [00:40, 1033.20 examples/s]41436 examples [00:40, 1033.72 examples/s]41540 examples [00:40, 1010.78 examples/s]41645 examples [00:40, 1019.58 examples/s]41749 examples [00:40, 1023.89 examples/s]41852 examples [00:40, 1014.07 examples/s]41956 examples [00:41, 1020.58 examples/s]42059 examples [00:41, 1023.23 examples/s]42163 examples [00:41, 1027.88 examples/s]42266 examples [00:41, 1016.35 examples/s]42371 examples [00:41, 1023.82 examples/s]42474 examples [00:41, 1005.03 examples/s]42580 examples [00:41, 1018.41 examples/s]42685 examples [00:41, 1026.93 examples/s]42790 examples [00:41, 1033.24 examples/s]42895 examples [00:42, 1036.55 examples/s]43000 examples [00:42, 1040.06 examples/s]43106 examples [00:42, 1044.99 examples/s]43212 examples [00:42, 1046.54 examples/s]43317 examples [00:42, 1039.65 examples/s]43422 examples [00:42, 1041.02 examples/s]43528 examples [00:42, 1046.13 examples/s]43634 examples [00:42, 1049.06 examples/s]43739 examples [00:42, 1047.32 examples/s]43844 examples [00:42, 1041.55 examples/s]43950 examples [00:43, 1044.74 examples/s]44056 examples [00:43, 1049.24 examples/s]44162 examples [00:43, 1049.83 examples/s]44267 examples [00:43, 1040.41 examples/s]44372 examples [00:43, 1033.13 examples/s]44478 examples [00:43, 1039.66 examples/s]44585 examples [00:43, 1045.76 examples/s]44690 examples [00:43, 1046.58 examples/s]44796 examples [00:43, 1049.43 examples/s]44901 examples [00:43, 1035.23 examples/s]45006 examples [00:44, 1038.47 examples/s]45113 examples [00:44, 1045.18 examples/s]45219 examples [00:44, 1048.02 examples/s]45324 examples [00:44, 1012.83 examples/s]45426 examples [00:44, 1014.32 examples/s]45531 examples [00:44, 1022.16 examples/s]45636 examples [00:44, 1027.73 examples/s]45741 examples [00:44, 1032.99 examples/s]45846 examples [00:44, 1035.39 examples/s]45950 examples [00:44, 1023.96 examples/s]46053 examples [00:45, 1025.02 examples/s]46158 examples [00:45, 1029.76 examples/s]46263 examples [00:45, 1034.04 examples/s]46368 examples [00:45, 1038.61 examples/s]46472 examples [00:45, 1013.06 examples/s]46575 examples [00:45, 1017.26 examples/s]46677 examples [00:45, 974.36 examples/s] 46775 examples [00:45, 975.43 examples/s]46873 examples [00:45, 964.84 examples/s]46976 examples [00:45, 982.38 examples/s]47079 examples [00:46, 995.97 examples/s]47184 examples [00:46, 1011.35 examples/s]47286 examples [00:46, 1009.83 examples/s]47388 examples [00:46, 1007.68 examples/s]47489 examples [00:46, 996.40 examples/s] 47594 examples [00:46, 1010.76 examples/s]47698 examples [00:46, 1018.35 examples/s]47803 examples [00:46, 1024.78 examples/s]47907 examples [00:46, 1028.07 examples/s]48010 examples [00:46, 1028.46 examples/s]48113 examples [00:47, 1025.02 examples/s]48218 examples [00:47, 1031.90 examples/s]48322 examples [00:47, 1033.97 examples/s]48426 examples [00:47, 1018.76 examples/s]48528 examples [00:47, 1018.68 examples/s]48631 examples [00:47, 1020.07 examples/s]48734 examples [00:47, 1017.31 examples/s]48838 examples [00:47, 1022.39 examples/s]48943 examples [00:47, 1029.55 examples/s]49046 examples [00:48, 1028.16 examples/s]49150 examples [00:48, 1030.31 examples/s]49254 examples [00:48, 1031.80 examples/s]49359 examples [00:48, 1034.02 examples/s]49463 examples [00:48, 1016.62 examples/s]49565 examples [00:48, 1014.33 examples/s]49671 examples [00:48, 1025.43 examples/s]49777 examples [00:48, 1034.25 examples/s]49881 examples [00:48, 1035.61 examples/s]49985 examples [00:48, 1033.69 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 11%|█         | 5520/50000 [00:00<00:00, 55198.87 examples/s] 36%|███▋      | 18173/50000 [00:00<00:00, 66433.91 examples/s] 62%|██████▏   | 30824/50000 [00:00<00:00, 77469.27 examples/s] 87%|████████▋ | 43612/50000 [00:00<00:00, 87858.67 examples/s]                                                               0 examples [00:00, ? examples/s]87 examples [00:00, 865.82 examples/s]192 examples [00:00, 911.78 examples/s]298 examples [00:00, 951.68 examples/s]405 examples [00:00, 982.81 examples/s]511 examples [00:00, 1003.67 examples/s]617 examples [00:00, 1019.88 examples/s]723 examples [00:00, 1028.88 examples/s]829 examples [00:00, 1036.51 examples/s]936 examples [00:00, 1044.21 examples/s]1042 examples [00:01, 1047.19 examples/s]1149 examples [00:01, 1052.49 examples/s]1254 examples [00:01, 1049.58 examples/s]1360 examples [00:01, 1050.30 examples/s]1466 examples [00:01, 1052.73 examples/s]1571 examples [00:01, 1046.62 examples/s]1676 examples [00:01, 1032.90 examples/s]1782 examples [00:01, 1038.22 examples/s]1888 examples [00:01, 1043.59 examples/s]1994 examples [00:01, 1048.40 examples/s]2100 examples [00:02, 1049.37 examples/s]2206 examples [00:02, 1051.76 examples/s]2312 examples [00:02, 1053.53 examples/s]2418 examples [00:02, 1048.89 examples/s]2525 examples [00:02, 1053.76 examples/s]2631 examples [00:02, 1041.77 examples/s]2737 examples [00:02, 1045.01 examples/s]2842 examples [00:02, 1045.33 examples/s]2949 examples [00:02, 1051.08 examples/s]3055 examples [00:02, 1044.98 examples/s]3160 examples [00:03, 1026.97 examples/s]3266 examples [00:03, 1026.69 examples/s]3371 examples [00:03, 1032.66 examples/s]3475 examples [00:03, 1032.76 examples/s]3579 examples [00:03, 1026.64 examples/s]3685 examples [00:03, 1035.13 examples/s]3791 examples [00:03, 1041.79 examples/s]3896 examples [00:03, 1003.56 examples/s]4000 examples [00:03, 1013.81 examples/s]4104 examples [00:03, 1020.47 examples/s]4209 examples [00:04, 1027.75 examples/s]4312 examples [00:04, 1016.82 examples/s]4417 examples [00:04, 1024.75 examples/s]4523 examples [00:04, 1035.02 examples/s]4627 examples [00:04, 1020.81 examples/s]4733 examples [00:04, 1030.92 examples/s]4837 examples [00:04, 813.11 examples/s] 4926 examples [00:04, 832.46 examples/s]5028 examples [00:04, 880.37 examples/s]5133 examples [00:05, 925.13 examples/s]5238 examples [00:05, 958.06 examples/s]5342 examples [00:05, 980.15 examples/s]5446 examples [00:05, 996.15 examples/s]5549 examples [00:05, 1005.18 examples/s]5654 examples [00:05, 1017.88 examples/s]5757 examples [00:05, 1017.24 examples/s]5862 examples [00:05, 1024.98 examples/s]5965 examples [00:05, 1020.17 examples/s]6069 examples [00:05, 1025.59 examples/s]6172 examples [00:06, 1014.34 examples/s]6278 examples [00:06, 1025.09 examples/s]6382 examples [00:06, 1029.45 examples/s]6488 examples [00:06, 1036.55 examples/s]6592 examples [00:06, 1025.64 examples/s]6698 examples [00:06, 1034.06 examples/s]6805 examples [00:06, 1042.58 examples/s]6911 examples [00:06, 1044.95 examples/s]7016 examples [00:06, 1045.69 examples/s]7121 examples [00:06, 1043.33 examples/s]7226 examples [00:07, 1007.25 examples/s]7332 examples [00:07, 1020.98 examples/s]7436 examples [00:07, 1023.48 examples/s]7540 examples [00:07, 1027.21 examples/s]7645 examples [00:07, 1033.73 examples/s]7750 examples [00:07, 1035.71 examples/s]7854 examples [00:07, 1032.59 examples/s]7958 examples [00:07, 1032.36 examples/s]8062 examples [00:07, 1032.30 examples/s]8166 examples [00:07, 1027.61 examples/s]8269 examples [00:08, 1020.02 examples/s]8374 examples [00:08, 1028.47 examples/s]8477 examples [00:08, 1024.93 examples/s]8583 examples [00:08, 1032.43 examples/s]8687 examples [00:08, 1024.52 examples/s]8790 examples [00:08, 1018.27 examples/s]8893 examples [00:08, 1020.86 examples/s]8996 examples [00:08, 1022.84 examples/s]9102 examples [00:08, 1031.83 examples/s]9206 examples [00:09, 1033.49 examples/s]9310 examples [00:09, 1030.68 examples/s]9415 examples [00:09, 1034.61 examples/s]9519 examples [00:09, 1035.29 examples/s]9625 examples [00:09, 1041.24 examples/s]9730 examples [00:09, 1036.71 examples/s]9835 examples [00:09, 1038.05 examples/s]9939 examples [00:09, 1037.53 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteS71XJR/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteS71XJR/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f309d969488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f309d969488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f309d969488> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f309b9cc240>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f30289ce358>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f309d969488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f309d969488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f309d969488> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f30289ce208>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f307c3a61d0>), {}) 

  




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
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/textcnn.py", line 24, in <module>
    import torchtext
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/torchtext/__init__.py", line 42, in <module>
    _init_extension()
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/torchtext/__init__.py", line 38, in _init_extension
    torch.ops.load_library(ext_specs.origin)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/torch/_ops.py", line 106, in load_library
    ctypes.CDLL(path)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/ctypes/__init__.py", line 348, in __init__
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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  9.25 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  9.25 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  9.25 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  9.14 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  9.14 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.43 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.43 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.70 file/s]2020-09-03 06:08:53.496712: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-09-03 06:08:53.501044: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-09-03 06:08:53.501224: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5621910ca950 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-09-03 06:08:53.501239: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
[1mDownloading and preparing dataset mnist/3.0.1 (download: 11.06 MiB, generated: 21.00 MiB, total: 32.06 MiB) to /home/runner/tensorflow_datasets/mnist/3.0.1...[0m

[1mDataset mnist downloaded and prepared to /home/runner/tensorflow_datasets/mnist/3.0.1. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/ ['cifar10', 'mnist2', 'fashion-mnist_small.npy', 'mnist_dataset_small.npy', 'train', 'test'] 

  


 #################### get_dataset_torch 

  get_dataset_torch mlmodels/preprocess/generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

0it [00:00, ?it/s]  1%|          | 106496/9912422 [00:00<00:09, 1006822.04it/s] 88%|████████▊ | 8749056/9912422 [00:00<00:00, 1431074.67it/s]9920512it [00:00, 43721891.46it/s]                            
0it [00:00, ?it/s]32768it [00:00, 533907.31it/s]
0it [00:00, ?it/s]  4%|▍         | 65536/1648877 [00:00<00:02, 654959.31it/s]1654784it [00:00, 11995083.44it/s]                         
0it [00:00, ?it/s]8192it [00:00, 178427.26it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
