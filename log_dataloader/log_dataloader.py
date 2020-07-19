
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f3dc666f9d8> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f3dc666f9d8> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f3e312da510> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f3e312da510> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f3e50529ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f3e50529ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f3dde60b950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f3dde60b950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f3dde60b950> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:18, 125938.04it/s] 71%|███████   | 7028736/9912422 [00:00<00:16, 179773.07it/s]9920512it [00:00, 38634228.09it/s]                           
0it [00:00, ?it/s]32768it [00:00, 457746.86it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 152525.56it/s]1654784it [00:00, 11001181.09it/s]                         
0it [00:00, ?it/s]8192it [00:00, 144522.89it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f3ddcb59588>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f3ddc7b17f0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f3dde60b598> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f3dde60b598> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f3dde60b598> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:52,  1.44 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:52,  1.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:51,  1.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:50,  1.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   2%|▏         | 4/162 [00:00<01:18,  2.00 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:18,  2.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:18,  2.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:17,  2.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▍         | 7/162 [00:00<00:55,  2.77 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:55,  2.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:55,  2.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:55,  2.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   6%|▌         | 10/162 [00:01<00:39,  3.81 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:01<00:39,  3.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:01<00:39,  3.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:01<00:39,  3.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   8%|▊         | 13/162 [00:01<00:28,  5.15 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:01<00:28,  5.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:01<00:28,  5.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:01<00:28,  5.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:01<00:28,  5.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  10%|█         | 17/162 [00:01<00:21,  6.84 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:01<00:21,  6.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:01<00:21,  6.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:01<00:20,  6.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  12%|█▏        | 20/162 [00:01<00:16,  8.83 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:01<00:16,  8.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:01<00:15,  8.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:01<00:15,  8.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  14%|█▍        | 23/162 [00:01<00:12, 11.03 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:01<00:12, 11.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:01<00:12, 11.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:01<00:12, 11.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  16%|█▌        | 26/162 [00:01<00:10, 13.45 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:01<00:10, 13.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:01<00:10, 13.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:01<00:09, 13.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  18%|█▊        | 29/162 [00:01<00:08, 15.92 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:01<00:08, 15.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:01<00:08, 15.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:01<00:08, 15.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:01<00:08, 15.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  20%|██        | 33/162 [00:01<00:06, 19.41 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:01<00:06, 19.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:06, 19.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:06, 19.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:06, 19.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  23%|██▎       | 37/162 [00:01<00:05, 22.73 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:05, 22.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:05, 22.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:05, 22.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:05, 22.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:02<00:05, 22.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  26%|██▌       | 42/162 [00:02<00:04, 26.15 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:02<00:04, 26.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:02<00:04, 26.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:02<00:04, 26.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:02<00:04, 26.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  28%|██▊       | 46/162 [00:02<00:03, 29.09 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:02<00:03, 29.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:02<00:03, 29.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:02<00:03, 29.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:02<00:03, 29.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  31%|███       | 50/162 [00:02<00:03, 31.65 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:02<00:03, 31.65 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:02<00:03, 31.65 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:02<00:03, 31.65 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:02<00:03, 31.65 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  33%|███▎      | 54/162 [00:02<00:03, 33.60 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:02<00:03, 33.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:02<00:03, 33.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:02<00:03, 33.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:02<00:03, 33.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  36%|███▌      | 58/162 [00:02<00:02, 35.28 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:02<00:02, 35.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:02<00:02, 35.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:02<00:02, 35.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:02<00:02, 35.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  38%|███▊      | 62/162 [00:02<00:02, 36.52 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:02<00:02, 36.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:02<00:02, 36.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:02<00:02, 36.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:02<00:02, 36.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  41%|████      | 66/162 [00:02<00:02, 37.46 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:02<00:02, 37.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:02<00:02, 37.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:02<00:02, 37.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:02<00:02, 37.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  43%|████▎     | 70/162 [00:02<00:02, 38.16 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:02<00:02, 38.16 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:02<00:02, 38.16 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:02<00:02, 38.16 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:02<00:02, 38.16 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  46%|████▌     | 74/162 [00:02<00:02, 38.64 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:02<00:02, 38.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:02<00:02, 38.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:02<00:02, 38.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:02<00:02, 38.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  48%|████▊     | 78/162 [00:02<00:02, 38.97 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:02<00:02, 38.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:02<00:02, 38.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:02<00:02, 38.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:03<00:02, 38.97 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:03<00:02, 38.97 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  51%|█████     | 83/162 [00:03<00:02, 39.32 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:03<00:02, 39.32 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:03<00:01, 39.32 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:03<00:01, 39.32 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:03<00:01, 39.32 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:03<00:01, 39.32 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  54%|█████▍    | 88/162 [00:03<00:01, 39.73 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:03<00:01, 39.73 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:03<00:01, 39.73 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:03<00:01, 39.73 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:03<00:01, 39.73 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:03<00:01, 39.73 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  57%|█████▋    | 93/162 [00:03<00:01, 39.88 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:03<00:01, 39.88 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:03<00:01, 39.88 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:03<00:01, 39.88 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:03<00:01, 39.88 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:03<00:01, 39.88 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  60%|██████    | 98/162 [00:03<00:01, 39.92 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:03<00:01, 39.92 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:03<00:01, 39.92 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:03<00:01, 39.92 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:03<00:01, 39.92 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:03<00:01, 39.92 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  64%|██████▎   | 103/162 [00:03<00:01, 39.81 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:03<00:01, 39.81 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:03<00:01, 39.81 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:03<00:01, 39.81 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:03<00:01, 39.81 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:03<00:01, 39.81 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  67%|██████▋   | 108/162 [00:03<00:01, 40.30 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:03<00:01, 40.30 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:03<00:01, 40.30 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:03<00:01, 40.30 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:03<00:01, 40.30 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:03<00:01, 40.30 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  70%|██████▉   | 113/162 [00:03<00:01, 40.30 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:03<00:01, 40.30 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:03<00:01, 40.30 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:03<00:01, 40.30 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:03<00:01, 40.30 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:03<00:01, 40.30 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  73%|███████▎  | 118/162 [00:03<00:01, 40.37 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:03<00:01, 40.37 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:03<00:01, 40.37 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:03<00:01, 40.37 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:04<00:01, 40.37 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:04<00:00, 40.37 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  76%|███████▌  | 123/162 [00:04<00:00, 40.27 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:04<00:00, 40.27 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:04<00:00, 40.27 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:04<00:00, 40.27 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:04<00:00, 40.27 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:04<00:00, 40.27 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  79%|███████▉  | 128/162 [00:04<00:00, 40.30 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:04<00:00, 40.30 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:04<00:00, 40.30 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:04<00:00, 40.30 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:04<00:00, 40.30 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:04<00:00, 40.30 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  82%|████████▏ | 133/162 [00:04<00:00, 40.65 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:04<00:00, 40.65 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:04<00:00, 40.65 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:04<00:00, 40.65 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:04<00:00, 40.65 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:04<00:00, 40.65 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  85%|████████▌ | 138/162 [00:04<00:00, 40.26 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:04<00:00, 40.26 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:04<00:00, 40.26 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:04<00:00, 40.26 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:04<00:00, 40.26 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:04<00:00, 40.26 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  88%|████████▊ | 143/162 [00:04<00:00, 40.33 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:04<00:00, 40.33 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:04<00:00, 40.33 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:04<00:00, 40.33 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:04<00:00, 40.33 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:04<00:00, 40.33 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  91%|█████████▏| 148/162 [00:04<00:00, 40.91 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:04<00:00, 40.91 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:04<00:00, 40.91 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:04<00:00, 40.91 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:04<00:00, 40.91 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:04<00:00, 40.91 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  94%|█████████▍| 153/162 [00:04<00:00, 41.36 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:04<00:00, 41.36 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:04<00:00, 41.36 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:04<00:00, 41.36 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:04<00:00, 41.36 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:04<00:00, 41.36 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  98%|█████████▊| 158/162 [00:04<00:00, 41.65 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:04<00:00, 41.65 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:04<00:00, 41.65 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:04<00:00, 41.65 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:04<00:00, 41.65 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 41.65 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:05<00:00,  5.02s/ url]Dl Completed...: 100%|██████████| 1/1 [00:05<00:00,  5.02s/ url]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 41.65 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:05<00:00,  5.02s/ url]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 41.65 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:05<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:07<00:00,  7.10s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:07<00:00,  5.02s/ url]
Dl Size...: 100%|██████████| 162/162 [00:07<00:00, 41.65 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:07<00:00,  7.10s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:07<00:00,  7.10s/ file]
Dl Size...: 100%|██████████| 162/162 [00:07<00:00, 22.81 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:07<00:00,  7.10s/ url]
0 examples [00:00, ? examples/s]2020-07-19 00:09:19.990970: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-19 00:09:20.787368: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-07-19 00:09:20.788019: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55cbc7d1cd50 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-19 00:09:20.788046: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:03,  3.91s/ examples]111 examples [00:04,  2.74s/ examples]221 examples [00:04,  1.92s/ examples]330 examples [00:04,  1.34s/ examples]441 examples [00:04,  1.06 examples/s]553 examples [00:04,  1.52 examples/s]664 examples [00:04,  2.17 examples/s]775 examples [00:04,  3.10 examples/s]886 examples [00:04,  4.42 examples/s]999 examples [00:04,  6.30 examples/s]1112 examples [00:04,  8.98 examples/s]1224 examples [00:05, 12.79 examples/s]1340 examples [00:05, 18.18 examples/s]1452 examples [00:05, 25.78 examples/s]1562 examples [00:05, 36.46 examples/s]1678 examples [00:05, 51.39 examples/s]1793 examples [00:05, 72.04 examples/s]1910 examples [00:05, 100.23 examples/s]2024 examples [00:05, 137.97 examples/s]2138 examples [00:05, 187.36 examples/s]2253 examples [00:05, 250.15 examples/s]2370 examples [00:06, 327.27 examples/s]2485 examples [00:06, 415.70 examples/s]2600 examples [00:06, 510.17 examples/s]2713 examples [00:06, 603.21 examples/s]2824 examples [00:06, 694.03 examples/s]2938 examples [00:06, 785.57 examples/s]3049 examples [00:06, 860.12 examples/s]3160 examples [00:06, 920.43 examples/s]3271 examples [00:06, 963.35 examples/s]3384 examples [00:06, 1007.90 examples/s]3497 examples [00:07, 1041.26 examples/s]3610 examples [00:07, 1065.99 examples/s]3722 examples [00:07, 1068.47 examples/s]3835 examples [00:07, 1085.70 examples/s]3951 examples [00:07, 1105.11 examples/s]4064 examples [00:07, 1107.58 examples/s]4177 examples [00:07, 1112.80 examples/s]4290 examples [00:07, 1111.88 examples/s]4405 examples [00:07, 1123.05 examples/s]4523 examples [00:07, 1136.66 examples/s]4638 examples [00:08, 1132.16 examples/s]4752 examples [00:08, 1128.42 examples/s]4866 examples [00:08, 1113.56 examples/s]4983 examples [00:08, 1129.25 examples/s]5097 examples [00:08, 1127.09 examples/s]5212 examples [00:08, 1131.95 examples/s]5326 examples [00:08, 1130.56 examples/s]5440 examples [00:08, 1108.12 examples/s]5555 examples [00:08, 1118.62 examples/s]5669 examples [00:08, 1123.05 examples/s]5783 examples [00:09, 1125.41 examples/s]5897 examples [00:09, 1126.94 examples/s]6010 examples [00:09, 1123.72 examples/s]6126 examples [00:09, 1132.18 examples/s]6240 examples [00:09, 1133.38 examples/s]6354 examples [00:09, 1130.16 examples/s]6468 examples [00:09, 1121.61 examples/s]6581 examples [00:09, 1118.27 examples/s]6697 examples [00:09, 1127.64 examples/s]6813 examples [00:09, 1135.98 examples/s]6927 examples [00:10, 1128.66 examples/s]7040 examples [00:10, 1112.68 examples/s]7152 examples [00:10, 1091.11 examples/s]7263 examples [00:10, 1096.30 examples/s]7373 examples [00:10, 1080.70 examples/s]7491 examples [00:10, 1107.43 examples/s]7602 examples [00:10, 1104.30 examples/s]7717 examples [00:10, 1114.98 examples/s]7835 examples [00:10, 1131.92 examples/s]7950 examples [00:11, 1136.00 examples/s]8064 examples [00:11, 1133.70 examples/s]8178 examples [00:11, 1119.95 examples/s]8292 examples [00:11, 1125.61 examples/s]8410 examples [00:11, 1138.88 examples/s]8525 examples [00:11, 1139.68 examples/s]8640 examples [00:11, 1140.52 examples/s]8755 examples [00:11, 1129.43 examples/s]8868 examples [00:11, 1112.65 examples/s]8980 examples [00:11, 1096.64 examples/s]9095 examples [00:12, 1111.13 examples/s]9211 examples [00:12, 1124.21 examples/s]9324 examples [00:12, 1059.79 examples/s]9431 examples [00:12, 1059.26 examples/s]9545 examples [00:12, 1079.92 examples/s]9654 examples [00:12, 1050.10 examples/s]9766 examples [00:12, 1069.84 examples/s]9874 examples [00:12, 1065.16 examples/s]9981 examples [00:12, 1031.43 examples/s]10085 examples [00:13, 981.37 examples/s]10199 examples [00:13, 1022.44 examples/s]10310 examples [00:13, 1045.89 examples/s]10416 examples [00:13, 1044.82 examples/s]10522 examples [00:13, 1042.45 examples/s]10627 examples [00:13, 1023.96 examples/s]10736 examples [00:13, 1042.84 examples/s]10841 examples [00:13, 1000.03 examples/s]10947 examples [00:13, 1014.91 examples/s]11060 examples [00:13, 1045.92 examples/s]11174 examples [00:14, 1072.38 examples/s]11285 examples [00:14, 1082.96 examples/s]11394 examples [00:14, 1070.49 examples/s]11505 examples [00:14, 1079.17 examples/s]11619 examples [00:14, 1095.03 examples/s]11732 examples [00:14, 1103.31 examples/s]11843 examples [00:14, 1093.88 examples/s]11953 examples [00:14, 1020.42 examples/s]12057 examples [00:14, 1014.87 examples/s]12164 examples [00:14, 1030.78 examples/s]12271 examples [00:15, 1039.72 examples/s]12387 examples [00:15, 1071.71 examples/s]12501 examples [00:15, 1089.98 examples/s]12616 examples [00:15, 1105.08 examples/s]12730 examples [00:15, 1113.96 examples/s]12847 examples [00:15, 1128.07 examples/s]12963 examples [00:15, 1137.20 examples/s]13077 examples [00:15, 1136.03 examples/s]13191 examples [00:15, 1122.91 examples/s]13306 examples [00:15, 1130.02 examples/s]13424 examples [00:16, 1141.89 examples/s]13539 examples [00:16, 1137.68 examples/s]13653 examples [00:16, 1136.01 examples/s]13767 examples [00:16, 1131.01 examples/s]13881 examples [00:16, 1100.03 examples/s]13992 examples [00:16, 1091.56 examples/s]14106 examples [00:16, 1103.44 examples/s]14218 examples [00:16, 1104.98 examples/s]14329 examples [00:16, 1059.82 examples/s]14444 examples [00:17, 1083.54 examples/s]14557 examples [00:17, 1094.71 examples/s]14669 examples [00:17, 1099.79 examples/s]14780 examples [00:17, 1095.71 examples/s]14894 examples [00:17, 1107.76 examples/s]15010 examples [00:17, 1121.27 examples/s]15123 examples [00:17, 1106.94 examples/s]15240 examples [00:17, 1123.08 examples/s]15355 examples [00:17, 1128.90 examples/s]15469 examples [00:17, 1118.00 examples/s]15583 examples [00:18, 1123.73 examples/s]15697 examples [00:18, 1126.09 examples/s]15810 examples [00:18, 1106.42 examples/s]15923 examples [00:18, 1112.21 examples/s]16039 examples [00:18, 1124.99 examples/s]16153 examples [00:18, 1127.94 examples/s]16266 examples [00:18, 1123.65 examples/s]16383 examples [00:18, 1135.63 examples/s]16501 examples [00:18, 1147.60 examples/s]16616 examples [00:18, 1128.52 examples/s]16731 examples [00:19, 1132.90 examples/s]16846 examples [00:19, 1136.52 examples/s]16960 examples [00:19, 1122.52 examples/s]17074 examples [00:19, 1125.65 examples/s]17189 examples [00:19, 1130.37 examples/s]17303 examples [00:19, 1112.76 examples/s]17417 examples [00:19, 1119.77 examples/s]17530 examples [00:19, 1116.11 examples/s]17642 examples [00:19, 1104.92 examples/s]17756 examples [00:19, 1113.60 examples/s]17870 examples [00:20, 1120.87 examples/s]17983 examples [00:20, 1116.39 examples/s]18095 examples [00:20, 1111.56 examples/s]18207 examples [00:20, 1109.10 examples/s]18325 examples [00:20, 1127.04 examples/s]18443 examples [00:20, 1139.61 examples/s]18561 examples [00:20, 1150.41 examples/s]18677 examples [00:20, 1141.49 examples/s]18792 examples [00:20, 1104.98 examples/s]18903 examples [00:20, 1078.15 examples/s]19018 examples [00:21, 1096.14 examples/s]19130 examples [00:21, 1102.12 examples/s]19242 examples [00:21, 1105.05 examples/s]19353 examples [00:21, 1098.41 examples/s]19466 examples [00:21, 1107.64 examples/s]19577 examples [00:21, 1107.22 examples/s]19688 examples [00:21, 1100.91 examples/s]19802 examples [00:21, 1110.55 examples/s]19914 examples [00:21, 1089.10 examples/s]20024 examples [00:22, 1032.13 examples/s]20128 examples [00:22, 1029.39 examples/s]20232 examples [00:22, 1026.03 examples/s]20337 examples [00:22, 1032.92 examples/s]20441 examples [00:22, 1015.56 examples/s]20549 examples [00:22, 1032.63 examples/s]20653 examples [00:22, 1009.71 examples/s]20755 examples [00:22, 1010.90 examples/s]20863 examples [00:22, 1029.57 examples/s]20970 examples [00:22, 1038.48 examples/s]21076 examples [00:23, 1044.78 examples/s]21181 examples [00:23, 1031.97 examples/s]21286 examples [00:23, 1035.08 examples/s]21395 examples [00:23, 1049.30 examples/s]21501 examples [00:23, 1049.31 examples/s]21618 examples [00:23, 1081.66 examples/s]21730 examples [00:23, 1091.94 examples/s]21847 examples [00:23, 1113.36 examples/s]21962 examples [00:23, 1122.33 examples/s]22075 examples [00:23, 1109.63 examples/s]22188 examples [00:24, 1113.61 examples/s]22301 examples [00:24, 1116.06 examples/s]22413 examples [00:24, 1104.35 examples/s]22524 examples [00:24, 1105.81 examples/s]22636 examples [00:24, 1109.29 examples/s]22751 examples [00:24, 1119.34 examples/s]22863 examples [00:24, 1089.66 examples/s]22975 examples [00:24, 1096.04 examples/s]23087 examples [00:24, 1099.87 examples/s]23198 examples [00:24, 1074.82 examples/s]23310 examples [00:25, 1087.69 examples/s]23423 examples [00:25, 1099.15 examples/s]23534 examples [00:25, 1064.29 examples/s]23647 examples [00:25, 1081.58 examples/s]23756 examples [00:25, 1081.24 examples/s]23874 examples [00:25, 1108.83 examples/s]23993 examples [00:25, 1130.25 examples/s]24114 examples [00:25, 1150.77 examples/s]24230 examples [00:25, 1150.95 examples/s]24346 examples [00:26, 1129.58 examples/s]24460 examples [00:26, 1124.00 examples/s]24573 examples [00:26, 1100.39 examples/s]24686 examples [00:26, 1107.21 examples/s]24799 examples [00:26, 1112.11 examples/s]24911 examples [00:26, 1109.03 examples/s]25025 examples [00:26, 1116.99 examples/s]25137 examples [00:26, 1109.04 examples/s]25252 examples [00:26, 1118.11 examples/s]25364 examples [00:26, 1112.70 examples/s]25476 examples [00:27, 1088.78 examples/s]25591 examples [00:27, 1104.64 examples/s]25702 examples [00:27, 1103.54 examples/s]25813 examples [00:27, 1083.56 examples/s]25922 examples [00:27, 1084.01 examples/s]26032 examples [00:27, 1087.12 examples/s]26143 examples [00:27, 1093.15 examples/s]26261 examples [00:27, 1117.63 examples/s]26378 examples [00:27, 1130.41 examples/s]26492 examples [00:27, 1113.39 examples/s]26604 examples [00:28, 1109.40 examples/s]26716 examples [00:28, 1056.50 examples/s]26825 examples [00:28, 1064.46 examples/s]26932 examples [00:28, 1043.76 examples/s]27040 examples [00:28, 1052.72 examples/s]27150 examples [00:28, 1065.71 examples/s]27266 examples [00:28, 1092.15 examples/s]27378 examples [00:28, 1099.54 examples/s]27494 examples [00:28, 1114.44 examples/s]27606 examples [00:28, 1110.63 examples/s]27718 examples [00:29, 1107.96 examples/s]27829 examples [00:29, 1104.47 examples/s]27940 examples [00:29, 1104.39 examples/s]28051 examples [00:29, 1093.85 examples/s]28161 examples [00:29, 1076.48 examples/s]28269 examples [00:29, 1061.16 examples/s]28380 examples [00:29, 1074.36 examples/s]28490 examples [00:29, 1081.12 examples/s]28600 examples [00:29, 1083.55 examples/s]28713 examples [00:30, 1095.99 examples/s]28829 examples [00:30, 1114.27 examples/s]28941 examples [00:30, 1097.96 examples/s]29051 examples [00:30, 1096.75 examples/s]29164 examples [00:30, 1104.11 examples/s]29275 examples [00:30, 1101.38 examples/s]29386 examples [00:30, 1101.55 examples/s]29497 examples [00:30, 1076.23 examples/s]29605 examples [00:30, 1060.43 examples/s]29720 examples [00:30, 1083.47 examples/s]29829 examples [00:31, 1085.01 examples/s]29943 examples [00:31, 1098.32 examples/s]30053 examples [00:31, 1034.49 examples/s]30165 examples [00:31, 1057.22 examples/s]30276 examples [00:31, 1070.56 examples/s]30384 examples [00:31, 1051.46 examples/s]30490 examples [00:31, 1046.88 examples/s]30606 examples [00:31, 1077.33 examples/s]30720 examples [00:31, 1095.20 examples/s]30830 examples [00:31, 1089.37 examples/s]30942 examples [00:32, 1095.70 examples/s]31052 examples [00:32, 1079.07 examples/s]31161 examples [00:32, 1048.68 examples/s]31269 examples [00:32, 1057.66 examples/s]31380 examples [00:32, 1070.89 examples/s]31488 examples [00:32, 1071.47 examples/s]31602 examples [00:32, 1088.95 examples/s]31719 examples [00:32, 1111.17 examples/s]31831 examples [00:32, 1111.00 examples/s]31949 examples [00:32, 1129.09 examples/s]32063 examples [00:33, 1106.09 examples/s]32174 examples [00:33, 1100.20 examples/s]32285 examples [00:33, 1091.84 examples/s]32400 examples [00:33, 1106.00 examples/s]32511 examples [00:33, 1103.42 examples/s]32625 examples [00:33, 1112.65 examples/s]32737 examples [00:33, 1110.68 examples/s]32849 examples [00:33, 1103.84 examples/s]32966 examples [00:33, 1120.07 examples/s]33079 examples [00:34, 1118.28 examples/s]33191 examples [00:34, 1111.34 examples/s]33303 examples [00:34, 1108.64 examples/s]33415 examples [00:34, 1111.30 examples/s]33527 examples [00:34, 1097.66 examples/s]33638 examples [00:34, 1099.15 examples/s]33749 examples [00:34, 1101.39 examples/s]33863 examples [00:34, 1109.51 examples/s]33977 examples [00:34, 1116.72 examples/s]34091 examples [00:34, 1122.03 examples/s]34204 examples [00:35, 1124.27 examples/s]34320 examples [00:35, 1133.41 examples/s]34434 examples [00:35, 1113.11 examples/s]34551 examples [00:35, 1127.27 examples/s]34664 examples [00:35, 1124.63 examples/s]34777 examples [00:35, 1121.98 examples/s]34890 examples [00:35, 1112.78 examples/s]35002 examples [00:35, 1055.20 examples/s]35114 examples [00:35, 1072.53 examples/s]35226 examples [00:35, 1084.23 examples/s]35344 examples [00:36, 1109.39 examples/s]35456 examples [00:36, 1102.36 examples/s]35568 examples [00:36, 1106.71 examples/s]35683 examples [00:36, 1117.27 examples/s]35798 examples [00:36, 1125.73 examples/s]35911 examples [00:36, 1126.92 examples/s]36024 examples [00:36, 1125.21 examples/s]36137 examples [00:36, 1124.61 examples/s]36250 examples [00:36, 1124.75 examples/s]36364 examples [00:36, 1129.25 examples/s]36478 examples [00:37, 1131.14 examples/s]36592 examples [00:37, 1122.98 examples/s]36705 examples [00:37, 1112.66 examples/s]36817 examples [00:37, 1107.51 examples/s]36928 examples [00:37, 1094.36 examples/s]37038 examples [00:37, 1074.60 examples/s]37146 examples [00:37, 1067.80 examples/s]37260 examples [00:37, 1087.73 examples/s]37369 examples [00:37, 1070.43 examples/s]37479 examples [00:37, 1076.41 examples/s]37587 examples [00:38, 1069.79 examples/s]37695 examples [00:38, 1030.45 examples/s]37799 examples [00:38, 1031.44 examples/s]37908 examples [00:38, 1047.48 examples/s]38023 examples [00:38, 1075.43 examples/s]38137 examples [00:38, 1093.98 examples/s]38251 examples [00:38, 1104.79 examples/s]38362 examples [00:38, 1100.61 examples/s]38473 examples [00:38, 1103.06 examples/s]38590 examples [00:39, 1121.71 examples/s]38703 examples [00:39, 1115.23 examples/s]38815 examples [00:39, 1107.38 examples/s]38928 examples [00:39, 1111.26 examples/s]39043 examples [00:39, 1121.51 examples/s]39157 examples [00:39, 1125.29 examples/s]39270 examples [00:39, 1119.69 examples/s]39383 examples [00:39, 1120.60 examples/s]39498 examples [00:39, 1128.80 examples/s]39611 examples [00:39, 1099.41 examples/s]39723 examples [00:40, 1104.17 examples/s]39840 examples [00:40, 1122.26 examples/s]39953 examples [00:40, 1116.44 examples/s]40065 examples [00:40, 1060.58 examples/s]40178 examples [00:40, 1077.75 examples/s]40290 examples [00:40, 1087.20 examples/s]40404 examples [00:40, 1099.97 examples/s]40515 examples [00:40, 1102.07 examples/s]40629 examples [00:40, 1111.94 examples/s]40741 examples [00:40, 1113.06 examples/s]40856 examples [00:41, 1121.71 examples/s]40969 examples [00:41, 1117.12 examples/s]41081 examples [00:41, 1106.32 examples/s]41193 examples [00:41, 1109.61 examples/s]41305 examples [00:41, 1103.44 examples/s]41416 examples [00:41, 1100.56 examples/s]41527 examples [00:41, 1100.33 examples/s]41638 examples [00:41, 1101.92 examples/s]41749 examples [00:41, 1055.45 examples/s]41855 examples [00:41, 1049.01 examples/s]41961 examples [00:42, 1023.17 examples/s]42067 examples [00:42, 1033.30 examples/s]42176 examples [00:42, 1046.89 examples/s]42288 examples [00:42, 1066.58 examples/s]42404 examples [00:42, 1090.52 examples/s]42519 examples [00:42, 1106.82 examples/s]42635 examples [00:42, 1120.91 examples/s]42748 examples [00:42, 1123.49 examples/s]42861 examples [00:42, 1119.72 examples/s]42974 examples [00:43, 1105.00 examples/s]43085 examples [00:43, 1095.37 examples/s]43195 examples [00:43, 1077.71 examples/s]43303 examples [00:43, 1069.15 examples/s]43411 examples [00:43, 1054.56 examples/s]43520 examples [00:43, 1063.98 examples/s]43632 examples [00:43, 1078.70 examples/s]43740 examples [00:43, 990.08 examples/s] 43851 examples [00:43, 1021.33 examples/s]43965 examples [00:43, 1053.90 examples/s]44072 examples [00:44, 1049.00 examples/s]44178 examples [00:44, 1033.88 examples/s]44283 examples [00:44, 1014.34 examples/s]44394 examples [00:44, 1040.45 examples/s]44505 examples [00:44, 1059.97 examples/s]44618 examples [00:44, 1078.63 examples/s]44733 examples [00:44, 1096.40 examples/s]44843 examples [00:44, 1078.57 examples/s]44952 examples [00:44, 1057.26 examples/s]45059 examples [00:45, 1031.03 examples/s]45163 examples [00:45, 1033.52 examples/s]45268 examples [00:45, 1036.03 examples/s]45372 examples [00:45, 1014.56 examples/s]45482 examples [00:45, 1036.54 examples/s]45592 examples [00:45, 1052.11 examples/s]45700 examples [00:45, 1058.09 examples/s]45807 examples [00:45, 1061.22 examples/s]45914 examples [00:45, 1057.51 examples/s]46020 examples [00:45, 1052.75 examples/s]46126 examples [00:46, 1049.11 examples/s]46235 examples [00:46, 1060.14 examples/s]46347 examples [00:46, 1074.95 examples/s]46455 examples [00:46, 1052.83 examples/s]46561 examples [00:46, 1019.25 examples/s]46664 examples [00:46, 1006.71 examples/s]46768 examples [00:46, 1016.29 examples/s]46878 examples [00:46, 1038.86 examples/s]46989 examples [00:46, 1057.40 examples/s]47098 examples [00:46, 1065.61 examples/s]47212 examples [00:47, 1086.50 examples/s]47328 examples [00:47, 1106.23 examples/s]47439 examples [00:47, 1092.15 examples/s]47551 examples [00:47, 1099.48 examples/s]47663 examples [00:47, 1103.79 examples/s]47775 examples [00:47, 1106.56 examples/s]47892 examples [00:47, 1122.41 examples/s]48006 examples [00:47, 1126.76 examples/s]48120 examples [00:47, 1130.20 examples/s]48234 examples [00:47, 1114.56 examples/s]48346 examples [00:48, 1115.47 examples/s]48459 examples [00:48, 1119.51 examples/s]48571 examples [00:48, 1102.79 examples/s]48687 examples [00:48, 1117.22 examples/s]48799 examples [00:48, 1111.18 examples/s]48913 examples [00:48, 1117.10 examples/s]49025 examples [00:48, 1115.20 examples/s]49137 examples [00:48, 1113.25 examples/s]49249 examples [00:48, 1112.11 examples/s]49362 examples [00:48, 1115.57 examples/s]49474 examples [00:49, 1116.66 examples/s]49586 examples [00:49, 1080.71 examples/s]49695 examples [00:49, 1072.76 examples/s]49803 examples [00:49, 1061.33 examples/s]49911 examples [00:49, 1064.50 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 14%|█▍        | 6967/50000 [00:00<00:00, 69661.43 examples/s] 39%|███▉      | 19490/50000 [00:00<00:00, 80358.44 examples/s] 65%|██████▍   | 32363/50000 [00:00<00:00, 90567.86 examples/s] 91%|█████████▏| 45715/50000 [00:00<00:00, 100241.90 examples/s]                                                                0 examples [00:00, ? examples/s]90 examples [00:00, 892.33 examples/s]200 examples [00:00, 944.76 examples/s]310 examples [00:00, 985.28 examples/s]422 examples [00:00, 1020.31 examples/s]531 examples [00:00, 1038.58 examples/s]639 examples [00:00, 1049.42 examples/s]746 examples [00:00, 1052.70 examples/s]856 examples [00:00, 1064.10 examples/s]963 examples [00:00, 1064.69 examples/s]1073 examples [00:01, 1073.56 examples/s]1185 examples [00:01, 1086.40 examples/s]1293 examples [00:01, 1081.18 examples/s]1404 examples [00:01, 1087.58 examples/s]1521 examples [00:01, 1110.34 examples/s]1632 examples [00:01, 1104.49 examples/s]1744 examples [00:01, 1106.93 examples/s]1859 examples [00:01, 1117.70 examples/s]1972 examples [00:01, 1118.79 examples/s]2084 examples [00:01, 1109.00 examples/s]2200 examples [00:02, 1121.89 examples/s]2315 examples [00:02, 1128.78 examples/s]2428 examples [00:02, 1119.90 examples/s]2542 examples [00:02, 1125.39 examples/s]2655 examples [00:02, 1098.97 examples/s]2768 examples [00:02, 1105.43 examples/s]2879 examples [00:02, 1106.16 examples/s]2990 examples [00:02, 1101.44 examples/s]3101 examples [00:02, 1083.30 examples/s]3215 examples [00:02, 1097.02 examples/s]3331 examples [00:03, 1114.76 examples/s]3445 examples [00:03, 1121.67 examples/s]3558 examples [00:03, 1110.67 examples/s]3670 examples [00:03, 1111.45 examples/s]3786 examples [00:03, 1124.69 examples/s]3899 examples [00:03, 1116.71 examples/s]4017 examples [00:03, 1134.67 examples/s]4133 examples [00:03, 1140.40 examples/s]4248 examples [00:03, 1125.86 examples/s]4361 examples [00:03, 1125.04 examples/s]4475 examples [00:04, 1127.37 examples/s]4590 examples [00:04, 1132.23 examples/s]4704 examples [00:04, 1126.49 examples/s]4818 examples [00:04, 1129.17 examples/s]4931 examples [00:04, 1099.26 examples/s]5042 examples [00:04, 1083.96 examples/s]5155 examples [00:04, 1096.18 examples/s]5268 examples [00:04, 1103.99 examples/s]5379 examples [00:04, 1057.88 examples/s]5491 examples [00:04, 1073.36 examples/s]5602 examples [00:05, 1082.50 examples/s]5716 examples [00:05, 1099.07 examples/s]5829 examples [00:05, 1107.94 examples/s]5944 examples [00:05, 1118.94 examples/s]6058 examples [00:05, 1124.52 examples/s]6172 examples [00:05, 1127.46 examples/s]6287 examples [00:05, 1131.79 examples/s]6403 examples [00:05, 1137.36 examples/s]6517 examples [00:05, 1132.83 examples/s]6636 examples [00:05, 1148.25 examples/s]6751 examples [00:06, 1145.60 examples/s]6866 examples [00:06, 1138.42 examples/s]6980 examples [00:06, 1125.55 examples/s]7093 examples [00:06, 1117.91 examples/s]7207 examples [00:06, 1122.57 examples/s]7320 examples [00:06, 1069.30 examples/s]7435 examples [00:06, 1091.75 examples/s]7549 examples [00:06, 1104.28 examples/s]7664 examples [00:06, 1116.25 examples/s]7781 examples [00:07, 1131.53 examples/s]7898 examples [00:07, 1142.71 examples/s]8013 examples [00:07, 1119.89 examples/s]8127 examples [00:07, 1124.32 examples/s]8243 examples [00:07, 1134.30 examples/s]8357 examples [00:07, 1134.86 examples/s]8472 examples [00:07, 1136.72 examples/s]8586 examples [00:07, 1135.94 examples/s]8701 examples [00:07, 1137.34 examples/s]8815 examples [00:07, 1132.89 examples/s]8931 examples [00:08, 1138.88 examples/s]9050 examples [00:08, 1152.46 examples/s]9166 examples [00:08, 1141.76 examples/s]9281 examples [00:08, 1142.72 examples/s]9396 examples [00:08, 1142.04 examples/s]9511 examples [00:08, 1127.22 examples/s]9624 examples [00:08, 1110.45 examples/s]9739 examples [00:08, 1120.15 examples/s]9854 examples [00:08, 1126.50 examples/s]9967 examples [00:08, 1118.08 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete74TKX3/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete74TKX3/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f3dde60b950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f3dde60b950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f3dde60b950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f3dbda08ba8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f3d9c9c43c8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f3dde60b950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f3dde60b950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f3dde60b950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f3d9c9c4eb8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f3ddc7b1080>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f3d56fc9e18> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f3d56fc9e18> 

  function with postional parmater data_info <function split_train_valid at 0x7f3d56fc9e18> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f3d56fc9f28> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f3d56fc9f28> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f3d56fc9f28> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.1
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.1) (2.3.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (4.48.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (45.2.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (7.4.1)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.0.3)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.7.1)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.2)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.4.1)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.1.3)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.24.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.19.0)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.7.0)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2020.6.20)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.10)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.25.9)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.1-py3-none-any.whl size=12047105 sha256=f75bc6db699f54a28e9de9c66283cade0dbba0044f47da9b8039ad122a037bda
  Stored in directory: /tmp/pip-ephem-wheel-cache-q_guru7g/wheels/10/6f/a6/ddd8204ceecdedddea923f8514e13afb0c1f0f556d2c9c3da0
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<19:40:23, 12.2kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<14:00:04, 17.1kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<9:51:14, 24.3kB/s]  .vector_cache/glove.6B.zip:   0%|          | 893k/862M [00:01<6:54:15, 34.7kB/s].vector_cache/glove.6B.zip:   0%|          | 2.78M/862M [00:01<4:49:34, 49.5kB/s].vector_cache/glove.6B.zip:   1%|          | 6.68M/862M [00:01<3:21:54, 70.6kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.2M/862M [00:01<2:20:30, 101kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 15.7M/862M [00:01<1:38:04, 144kB/s].vector_cache/glove.6B.zip:   2%|▏         | 21.4M/862M [00:01<1:08:16, 205kB/s].vector_cache/glove.6B.zip:   3%|▎         | 24.4M/862M [00:01<47:45, 292kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 30.1M/862M [00:01<33:18, 416kB/s].vector_cache/glove.6B.zip:   4%|▍         | 35.2M/862M [00:01<23:15, 593kB/s].vector_cache/glove.6B.zip:   5%|▍         | 39.1M/862M [00:02<16:18, 841kB/s].vector_cache/glove.6B.zip:   5%|▌         | 44.6M/862M [00:02<11:24, 1.19MB/s].vector_cache/glove.6B.zip:   6%|▌         | 47.8M/862M [00:02<08:05, 1.68MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.2M/862M [00:02<06:08, 2.20MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.4M/862M [00:04<06:11, 2.17MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.7M/862M [00:04<06:12, 2.16MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.8M/862M [00:05<04:48, 2.79MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.5M/862M [00:06<05:53, 2.27MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.9M/862M [00:06<05:43, 2.33MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.3M/862M [00:07<04:23, 3.04MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.7M/862M [00:08<05:51, 2.27MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.9M/862M [00:08<06:55, 1.92MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.6M/862M [00:08<05:27, 2.43MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.0M/862M [00:09<03:58, 3.33MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.8M/862M [00:10<11:05, 1.19MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.2M/862M [00:10<09:08, 1.45MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.7M/862M [00:10<06:40, 1.98MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.9M/862M [00:12<07:45, 1.70MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.1M/862M [00:12<08:04, 1.63MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.9M/862M [00:12<06:19, 2.07MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.0M/862M [00:14<06:32, 2.00MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.4M/862M [00:14<05:53, 2.22MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.0M/862M [00:14<04:27, 2.93MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.1M/862M [00:16<06:10, 2.11MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.5M/862M [00:16<05:37, 2.31MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.1M/862M [00:16<04:12, 3.08MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.3M/862M [00:18<06:02, 2.15MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.7M/862M [00:18<05:31, 2.34MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.2M/862M [00:18<04:08, 3.12MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.4M/862M [00:20<05:58, 2.16MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.6M/862M [00:20<06:47, 1.90MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.4M/862M [00:20<05:23, 2.38MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.1M/862M [00:20<04:00, 3.21MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.5M/862M [00:22<07:06, 1.80MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.9M/862M [00:22<06:04, 2.11MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.4M/862M [00:22<04:34, 2.79MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.6M/862M [00:24<06:10, 2.07MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.8M/862M [00:24<06:53, 1.85MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.6M/862M [00:24<05:27, 2.33MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<05:52, 2.16MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<05:25, 2.34MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:26<04:06, 3.07MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<05:48, 2.17MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<05:18, 2.37MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:28<03:59, 3.15MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<05:46, 2.17MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<06:34, 1.91MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<05:08, 2.43MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:30<03:43, 3.34MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<14:29, 860kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<11:23, 1.09MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<08:14, 1.51MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<08:40, 1.43MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<08:34, 1.45MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<06:35, 1.88MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:34<04:45, 2.60MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<1:44:54, 118kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<1:14:40, 165kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<52:28, 234kB/s]  .vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<39:29, 311kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<28:51, 425kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<20:28, 597kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<17:10, 710kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<13:04, 932kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<09:26, 1.29MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<09:28, 1.28MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<07:51, 1.54MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<05:47, 2.09MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<06:54, 1.75MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<07:16, 1.66MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<05:42, 2.11MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<05:55, 2.02MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<06:40, 1.80MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<05:16, 2.27MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:46<03:49, 3.11MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<1:28:02, 135kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<1:02:50, 190kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<44:11, 269kB/s]  .vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:49<33:37, 352kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<24:43, 479kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<17:34, 673kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<15:03, 782kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<12:55, 912kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<09:37, 1.22MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:52<06:50, 1.71MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<11:21:13, 17.2kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:53<7:57:48, 24.5kB/s] .vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<5:33:58, 35.0kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<3:55:48, 49.4kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:55<2:47:22, 69.6kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<1:57:32, 98.9kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:56<1:22:14, 141kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<1:02:01, 187kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<44:34, 260kB/s]  .vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<31:22, 368kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<24:37, 467kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<19:34, 588kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<14:16, 805kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<11:48, 969kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<09:25, 1.21MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:01<06:52, 1.66MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<07:27, 1.52MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<06:23, 1.78MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:03<04:45, 2.39MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<05:58, 1.89MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<05:20, 2.11MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:05<03:58, 2.83MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<05:25, 2.07MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<04:56, 2.27MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:07<03:41, 3.04MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<05:14, 2.13MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<04:46, 2.33MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:09<03:34, 3.12MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<05:08, 2.15MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<05:50, 1.90MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<04:34, 2.42MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:11<03:18, 3.34MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<48:48, 226kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<35:15, 313kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:13<24:51, 442kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:15<19:56, 550kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<16:10, 678kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<11:51, 923kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<10:01, 1.09MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<09:14, 1.18MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<06:55, 1.57MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:17<05:00, 2.16MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<07:32, 1.44MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<06:21, 1.70MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:19<04:43, 2.29MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<05:48, 1.85MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<05:09, 2.08MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:21<03:52, 2.77MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<05:14, 2.04MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<04:44, 2.25MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:23<03:34, 2.98MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<05:01, 2.11MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<05:39, 1.88MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<04:29, 2.36MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<04:50, 2.18MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<04:28, 2.36MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<03:20, 3.14MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<04:48, 2.18MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<05:29, 1.91MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<04:17, 2.44MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:29<03:08, 3.32MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:31<07:03, 1.48MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<05:59, 1.74MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<04:25, 2.35MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<05:31, 1.87MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<04:54, 2.11MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:33<03:41, 2.80MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<05:00, 2.05MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<05:35, 1.84MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<04:21, 2.36MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:35<03:09, 3.23MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<08:34, 1.19MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<07:03, 1.44MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<05:11, 1.96MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:38<06:00, 1.69MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 254M/862M [01:39<05:13, 1.94MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<03:53, 2.59MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<05:07, 1.97MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<05:36, 1.79MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<04:25, 2.27MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<04:42, 2.12MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:43<04:19, 2.31MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<03:16, 3.04MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<04:35, 2.16MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<05:14, 1.90MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<04:04, 2.43MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:45<02:58, 3.32MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<08:21, 1.18MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:46<06:51, 1.44MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<05:02, 1.95MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:48<05:49, 1.68MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:48<05:04, 1.93MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<03:47, 2.58MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<04:57, 1.96MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<05:27, 1.78MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:50<04:13, 2.30MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<03:06, 3.11MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<05:28, 1.76MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<04:50, 2.00MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:52<03:35, 2.68MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<04:45, 2.02MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<05:22, 1.78MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:54<04:15, 2.25MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:55<03:04, 3.09MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:56<1:10:16, 135kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:56<50:09, 190kB/s]  .vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:56<35:15, 269kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<26:48, 353kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<20:40, 457kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<14:51, 635kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:58<10:28, 897kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<13:29, 695kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<10:23, 903kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:00<07:26, 1.26MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<07:23, 1.26MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<06:59, 1.33MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<05:21, 1.74MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<05:13, 1.77MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<05:36, 1.65MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<04:21, 2.12MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:04<03:08, 2.92MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<46:21, 198kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<33:22, 275kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:06<23:29, 389kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<18:30, 492kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<14:51, 613kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<10:47, 843kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:08<07:37, 1.19MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<12:46, 708kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<09:52, 914kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:10<07:07, 1.26MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<07:04, 1.27MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<06:45, 1.33MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<05:10, 1.73MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<05:03, 1.76MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<04:26, 2.00MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:14<03:17, 2.70MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<04:22, 2.02MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<04:52, 1.81MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<03:50, 2.29MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:16<02:46, 3.16MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<8:30:01, 17.2kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<5:57:39, 24.5kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:18<4:09:50, 35.0kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<2:56:12, 49.4kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 341M/862M [02:20<2:05:02, 69.5kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<1:27:50, 98.8kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<1:02:31, 138kB/s] .vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:22<44:37, 193kB/s]  .vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<31:21, 274kB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<23:51, 359kB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<18:25, 464kB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<13:15, 645kB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:24<09:20, 910kB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<11:27, 741kB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<08:52, 956kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:26<06:24, 1.32MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:27<06:26, 1.31MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<05:21, 1.57MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<03:55, 2.14MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<04:42, 1.78MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<04:59, 1.67MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<03:54, 2.13MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:30<02:48, 2.95MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<8:01:51, 17.2kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<5:37:53, 24.5kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<3:55:59, 35.0kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<2:46:24, 49.4kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<1:58:05, 69.6kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<1:22:56, 98.9kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:34<57:47, 141kB/s]   .vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<1:40:14, 81.3kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:35<1:10:56, 115kB/s] .vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:36<49:40, 163kB/s]  .vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:37<36:31, 221kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:37<26:20, 306kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<18:33, 434kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<14:50, 540kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<11:11, 715kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<08:00, 996kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<07:28, 1.06MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<06:01, 1.32MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:42<04:24, 1.80MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<04:56, 1.60MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<05:02, 1.56MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:43<03:55, 2.00MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:44<02:48, 2.77MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<7:34:18, 17.2kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<5:18:33, 24.5kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:45<3:42:27, 34.9kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<2:36:48, 49.3kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<1:51:16, 69.5kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:47<1:18:09, 98.8kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:48<54:27, 141kB/s]   .vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<1:33:13, 82.3kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<1:05:58, 116kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:49<46:12, 165kB/s]  .vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<33:58, 224kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<25:18, 300kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<18:00, 421kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:51<12:38, 597kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<14:20, 525kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<10:47, 698kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:53<07:41, 975kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<07:06, 1.05MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<06:28, 1.15MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<04:51, 1.53MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:55<03:27, 2.14MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<15:40, 472kB/s] .vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<11:43, 630kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 420M/862M [02:57<08:20, 883kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<07:31, 973kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<06:45, 1.08MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<05:05, 1.44MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:01<04:43, 1.54MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<04:02, 1.79MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:01<02:58, 2.43MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<03:45, 1.91MB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:03<03:22, 2.13MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:03<02:31, 2.83MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<03:27, 2.06MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<03:51, 1.84MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<03:03, 2.32MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<03:16, 2.15MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<03:01, 2.33MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:07<02:17, 3.07MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<03:13, 2.17MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<03:43, 1.88MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:09<02:57, 2.36MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:09<02:08, 3.23MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<51:22, 135kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<36:36, 189kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<25:57, 266kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:11<18:07, 378kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<18:26, 371kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<14:18, 478kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<10:18, 662kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:13<07:15, 936kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<11:30, 589kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<08:45, 773kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<06:14, 1.08MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<05:55, 1.13MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<05:31, 1.21MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<04:12, 1.59MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<03:59, 1.66MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<03:28, 1.91MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:19<02:35, 2.55MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<03:20, 1.96MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<03:00, 2.19MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:21<02:15, 2.89MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<03:05, 2.10MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<03:28, 1.87MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<02:42, 2.39MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:23<01:58, 3.26MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<04:46, 1.35MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<04:00, 1.60MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<02:56, 2.18MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<03:32, 1.80MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:27<03:45, 1.69MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<02:55, 2.17MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:27<02:06, 2.98MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<53:20, 118kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:28<37:57, 166kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<26:37, 235kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<19:59, 312kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<14:36, 426kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<10:20, 599kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<08:38, 713kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<06:39, 924kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<04:47, 1.28MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<04:47, 1.27MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<03:57, 1.54MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:34<02:53, 2.10MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<03:27, 1.74MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<03:01, 1.99MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:36<02:15, 2.65MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<02:59, 2.00MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<03:17, 1.81MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<02:33, 2.32MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:39<01:52, 3.16MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<04:02, 1.46MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<03:25, 1.72MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:40<02:32, 2.31MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:42<03:07, 1.86MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<02:46, 2.09MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:42<02:03, 2.81MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<02:47, 2.06MB/s].vector_cache/glove.6B.zip:  60%|██████    | 517M/862M [03:44<03:06, 1.85MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<02:25, 2.37MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:44<01:45, 3.24MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<04:29, 1.26MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<03:43, 1.52MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:46<02:44, 2.06MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<03:13, 1.74MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<03:23, 1.65MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<02:37, 2.13MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:48<01:52, 2.95MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<09:06, 609kB/s] .vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<06:55, 800kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:50<04:57, 1.11MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<04:44, 1.15MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<04:26, 1.23MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<03:22, 1.61MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:52<02:24, 2.24MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<40:16, 134kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<28:42, 188kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:54<20:07, 267kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<15:15, 350kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<11:44, 454kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<08:27, 629kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:56<05:56, 888kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<21:04, 250kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<15:16, 345kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:58<10:45, 487kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<08:41, 598kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<07:05, 733kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<05:10, 1.00MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:00<03:39, 1.40MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<05:35, 917kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<04:26, 1.15MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:02<03:12, 1.59MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<03:25, 1.48MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<03:24, 1.48MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<02:36, 1.94MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:04<01:52, 2.68MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<04:54, 1.02MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<03:56, 1.26MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<02:52, 1.73MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<03:09, 1.56MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<02:42, 1.81MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:08<01:59, 2.45MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<02:31, 1.92MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<02:45, 1.76MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<02:10, 2.23MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:10<01:33, 3.08MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:11<4:38:28, 17.2kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<3:15:10, 24.5kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<2:15:59, 35.0kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:13<1:35:35, 49.4kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<1:07:17, 70.0kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<46:57, 99.8kB/s]  .vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<33:42, 138kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<24:30, 190kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<17:20, 267kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<12:45, 359kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<09:22, 488kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<06:38, 685kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<05:40, 796kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<04:55, 917kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<03:40, 1.23MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:20<02:35, 1.72MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<30:24, 146kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:21<21:42, 205kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<15:13, 290kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:23<11:35, 378kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:23<09:01, 485kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<06:29, 672kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<04:34, 947kB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<04:57, 869kB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<03:54, 1.10MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<02:49, 1.51MB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:27<02:57, 1.43MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:27<02:55, 1.45MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:27<02:13, 1.90MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:28<01:35, 2.63MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<08:21, 499kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<06:15, 665kB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:29<04:27, 927kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<04:03, 1.01MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<03:40, 1.11MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:31<02:44, 1.49MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:31<01:57, 2.07MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<03:22, 1.20MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<02:46, 1.46MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:33<02:01, 1.97MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<02:20, 1.70MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:35<02:01, 1.95MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:35<01:29, 2.64MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<01:58, 1.98MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<02:10, 1.80MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<01:42, 2.27MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<01:48, 2.12MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<01:38, 2.32MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:39<01:13, 3.09MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<01:44, 2.16MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<01:35, 2.35MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:41<01:12, 3.10MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<01:42, 2.16MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<01:34, 2.35MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:43<01:11, 3.09MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<01:40, 2.16MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<01:32, 2.35MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:45<01:09, 3.10MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<01:39, 2.15MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<01:52, 1.90MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<01:27, 2.43MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:47<01:02, 3.35MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<06:37, 527kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<04:59, 698kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:49<03:32, 976kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<03:15, 1.05MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<02:58, 1.15MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<02:14, 1.52MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<02:05, 1.60MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<01:48, 1.86MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<01:20, 2.48MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<01:41, 1.93MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<01:51, 1.77MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<01:25, 2.28MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [04:55<01:02, 3.11MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<02:05, 1.54MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<01:46, 1.80MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<01:18, 2.44MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<01:38, 1.91MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<01:27, 2.15MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:59<01:05, 2.84MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<01:29, 2.07MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<01:39, 1.84MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:01<01:18, 2.32MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<01:23, 2.16MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<01:13, 2.44MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<00:55, 3.20MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:03<00:41, 4.28MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:04<15:05, 195kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<10:50, 271kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:05<07:34, 383kB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:06<05:54, 485kB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<04:25, 648kB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<03:07, 907kB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:08<02:49, 989kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<02:15, 1.24MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<01:38, 1.69MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<01:46, 1.53MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<01:47, 1.52MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<01:23, 1.96MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:11<00:58, 2.71MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<2:34:40, 17.2kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:12<1:48:17, 24.5kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<1:15:07, 35.0kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:14<52:28, 49.4kB/s]  .vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:14<37:12, 69.6kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<26:02, 99.0kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:15<17:53, 141kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<2:38:49, 15.9kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<1:51:06, 22.7kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:16<1:17:02, 32.4kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<53:43, 45.7kB/s]  .vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<38:03, 64.5kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:18<26:37, 91.7kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:19<18:16, 131kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<40:15, 59.3kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<28:21, 84.0kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:20<19:41, 120kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<14:06, 164kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<10:18, 225kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:22<07:17, 316kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<05:21, 420kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<03:58, 566kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:24<02:48, 791kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<02:26, 895kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<02:08, 1.02MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<01:34, 1.37MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:26<01:07, 1.91MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<01:56, 1.09MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<01:34, 1.34MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 737M/862M [05:28<01:07, 1.84MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:15, 1.62MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<01:17, 1.58MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<01:00, 2.03MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 744M/862M [05:32<01:00, 1.97MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<00:53, 2.19MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:32<00:39, 2.94MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<00:54, 2.11MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<00:49, 2.31MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:34<00:36, 3.09MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<00:51, 2.14MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<00:45, 2.44MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:36<00:33, 3.24MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:36<00:24, 4.35MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<07:27, 238kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<05:34, 318kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<03:56, 445kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:38<02:43, 631kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<03:07, 546kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<02:20, 723kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:40<01:39, 1.01MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<01:31, 1.08MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<01:11, 1.36MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:42<00:51, 1.85MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<00:57, 1.62MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<00:59, 1.57MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<00:45, 2.04MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:44<00:31, 2.83MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<02:28, 605kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<01:51, 803kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<01:18, 1.11MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<01:13, 1.16MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<01:08, 1.24MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:52, 1.63MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:48, 1.69MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:42, 1.93MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<00:30, 2.58MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:39, 1.98MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:35, 2.20MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 787M/862M [05:52<00:25, 2.91MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:54<00:35, 2.09MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<00:31, 2.30MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:54<00:23, 3.07MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:55<00:32, 2.14MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:35, 1.92MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:28, 2.41MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<00:29, 2.21MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:34, 1.90MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:26, 2.42MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:58<00:18, 3.31MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:51, 1.18MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:42, 1.43MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<00:30, 1.95MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:33, 1.68MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<00:35, 1.58MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:02<00:27, 2.06MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:02<00:19, 2.81MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:34, 1.53MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:03<00:28, 1.84MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:20, 2.47MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:25, 1.92MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:05<00:27, 1.77MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:21, 2.24MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<00:21, 2.10MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<00:19, 2.31MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:07<00:13, 3.08MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:18, 2.16MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:21, 1.90MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:09<00:16, 2.39MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:16, 2.19MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:19, 1.88MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:11<00:14, 2.36MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:12<00:09, 3.24MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<03:33, 151kB/s] .vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<02:30, 211kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:13<01:41, 300kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<01:12, 389kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:56, 498kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:15<00:39, 690kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:15<00:25, 974kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:33, 723kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<00:25, 933kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:17<00:17, 1.30MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:15, 1.29MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:12, 1.55MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:19<00:08, 2.11MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:09, 1.75MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:09, 1.66MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:06, 2.15MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:21<00:04, 2.97MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:14, 796kB/s] .vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:11, 1.02MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:23<00:06, 1.40MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:05, 1.36MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:05, 1.40MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:25<00:03, 1.81MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:01, 1.82MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:01, 1.67MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:27<00:01, 2.13MB/s].vector_cache/glove.6B.zip: 862MB [06:27, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 100/400000 [00:00<06:40, 999.63it/s]  0%|          | 995/400000 [00:00<04:52, 1362.77it/s]  0%|          | 1878/400000 [00:00<03:38, 1825.95it/s]  1%|          | 2802/400000 [00:00<02:45, 2404.82it/s]  1%|          | 3679/400000 [00:00<02:08, 3074.05it/s]  1%|          | 4552/400000 [00:00<01:43, 3815.51it/s]  1%|▏         | 5409/400000 [00:00<01:26, 4576.91it/s]  2%|▏         | 6276/400000 [00:00<01:13, 5331.49it/s]  2%|▏         | 7116/400000 [00:00<01:05, 5987.43it/s]  2%|▏         | 7989/400000 [00:01<00:59, 6610.39it/s]  2%|▏         | 8835/400000 [00:01<00:55, 7072.30it/s]  2%|▏         | 9671/400000 [00:01<00:52, 7372.60it/s]  3%|▎         | 10544/400000 [00:01<00:50, 7732.90it/s]  3%|▎         | 11426/400000 [00:01<00:48, 8027.64it/s]  3%|▎         | 12284/400000 [00:01<00:47, 8185.12it/s]  3%|▎         | 13161/400000 [00:01<00:46, 8350.45it/s]  4%|▎         | 14053/400000 [00:01<00:45, 8512.32it/s]  4%|▎         | 14924/400000 [00:01<00:45, 8420.70it/s]  4%|▍         | 15805/400000 [00:01<00:45, 8532.53it/s]  4%|▍         | 16670/400000 [00:02<00:44, 8566.04it/s]  4%|▍         | 17534/400000 [00:02<00:44, 8559.82it/s]  5%|▍         | 18395/400000 [00:02<00:44, 8565.44it/s]  5%|▍         | 19266/400000 [00:02<00:44, 8606.44it/s]  5%|▌         | 20130/400000 [00:02<00:44, 8592.94it/s]  5%|▌         | 21008/400000 [00:02<00:43, 8648.18it/s]  5%|▌         | 21909/400000 [00:02<00:43, 8751.68it/s]  6%|▌         | 22805/400000 [00:02<00:42, 8811.37it/s]  6%|▌         | 23709/400000 [00:02<00:42, 8875.79it/s]  6%|▌         | 24622/400000 [00:02<00:41, 8949.19it/s]  6%|▋         | 25518/400000 [00:03<00:42, 8899.39it/s]  7%|▋         | 26409/400000 [00:03<00:42, 8775.05it/s]  7%|▋         | 27288/400000 [00:03<00:43, 8614.96it/s]  7%|▋         | 28170/400000 [00:03<00:42, 8674.58it/s]  7%|▋         | 29039/400000 [00:03<00:42, 8665.27it/s]  7%|▋         | 29907/400000 [00:03<00:43, 8585.85it/s]  8%|▊         | 30778/400000 [00:03<00:42, 8621.55it/s]  8%|▊         | 31658/400000 [00:03<00:42, 8673.13it/s]  8%|▊         | 32526/400000 [00:03<00:42, 8647.06it/s]  8%|▊         | 33392/400000 [00:03<00:42, 8645.86it/s]  9%|▊         | 34257/400000 [00:04<00:42, 8620.47it/s]  9%|▉         | 35120/400000 [00:04<00:42, 8564.50it/s]  9%|▉         | 35982/400000 [00:04<00:42, 8579.33it/s]  9%|▉         | 36858/400000 [00:04<00:42, 8630.58it/s]  9%|▉         | 37722/400000 [00:04<00:42, 8511.66it/s] 10%|▉         | 38614/400000 [00:04<00:41, 8628.22it/s] 10%|▉         | 39504/400000 [00:04<00:41, 8705.18it/s] 10%|█         | 40394/400000 [00:04<00:41, 8760.31it/s] 10%|█         | 41271/400000 [00:04<00:40, 8756.88it/s] 11%|█         | 42173/400000 [00:04<00:40, 8833.44it/s] 11%|█         | 43074/400000 [00:05<00:40, 8885.52it/s] 11%|█         | 43963/400000 [00:05<00:40, 8774.71it/s] 11%|█         | 44853/400000 [00:05<00:40, 8811.30it/s] 11%|█▏        | 45761/400000 [00:05<00:39, 8889.16it/s] 12%|█▏        | 46673/400000 [00:05<00:39, 8956.72it/s] 12%|█▏        | 47571/400000 [00:05<00:39, 8962.28it/s] 12%|█▏        | 48468/400000 [00:05<00:39, 8826.43it/s] 12%|█▏        | 49361/400000 [00:05<00:39, 8856.57it/s] 13%|█▎        | 50251/400000 [00:05<00:39, 8866.98it/s] 13%|█▎        | 51139/400000 [00:05<00:39, 8866.95it/s] 13%|█▎        | 52026/400000 [00:06<00:39, 8778.97it/s] 13%|█▎        | 52905/400000 [00:06<00:40, 8646.75it/s] 13%|█▎        | 53771/400000 [00:06<00:40, 8630.31it/s] 14%|█▎        | 54651/400000 [00:06<00:39, 8679.93it/s] 14%|█▍        | 55520/400000 [00:06<00:39, 8638.14it/s] 14%|█▍        | 56396/400000 [00:06<00:39, 8672.45it/s] 14%|█▍        | 57264/400000 [00:06<00:40, 8475.53it/s] 15%|█▍        | 58147/400000 [00:06<00:39, 8578.84it/s] 15%|█▍        | 59041/400000 [00:06<00:39, 8681.36it/s] 15%|█▍        | 59918/400000 [00:06<00:39, 8705.39it/s] 15%|█▌        | 60838/400000 [00:07<00:38, 8846.31it/s] 15%|█▌        | 61724/400000 [00:07<00:38, 8805.25it/s] 16%|█▌        | 62607/400000 [00:07<00:38, 8810.33it/s] 16%|█▌        | 63498/400000 [00:07<00:38, 8839.61it/s] 16%|█▌        | 64383/400000 [00:07<00:38, 8766.04it/s] 16%|█▋        | 65261/400000 [00:07<00:38, 8761.79it/s] 17%|█▋        | 66138/400000 [00:07<00:38, 8659.43it/s] 17%|█▋        | 67014/400000 [00:07<00:38, 8687.09it/s] 17%|█▋        | 67913/400000 [00:07<00:37, 8775.22it/s] 17%|█▋        | 68823/400000 [00:07<00:37, 8869.02it/s] 17%|█▋        | 69723/400000 [00:08<00:37, 8906.08it/s] 18%|█▊        | 70615/400000 [00:08<00:37, 8808.33it/s] 18%|█▊        | 71497/400000 [00:08<00:38, 8531.79it/s] 18%|█▊        | 72353/400000 [00:08<00:38, 8421.48it/s] 18%|█▊        | 73256/400000 [00:08<00:38, 8594.83it/s] 19%|█▊        | 74158/400000 [00:08<00:37, 8715.49it/s] 19%|█▉        | 75032/400000 [00:08<00:37, 8686.02it/s] 19%|█▉        | 75938/400000 [00:08<00:36, 8792.70it/s] 19%|█▉        | 76832/400000 [00:08<00:36, 8833.35it/s] 19%|█▉        | 77725/400000 [00:08<00:36, 8859.30it/s] 20%|█▉        | 78630/400000 [00:09<00:36, 8914.44it/s] 20%|█▉        | 79523/400000 [00:09<00:36, 8772.07it/s] 20%|██        | 80429/400000 [00:09<00:36, 8856.03it/s] 20%|██        | 81316/400000 [00:09<00:36, 8840.29it/s] 21%|██        | 82211/400000 [00:09<00:35, 8870.38it/s] 21%|██        | 83131/400000 [00:09<00:35, 8964.07it/s] 21%|██        | 84028/400000 [00:09<00:35, 8816.67it/s] 21%|██        | 84911/400000 [00:09<00:36, 8688.61it/s] 21%|██▏       | 85848/400000 [00:09<00:35, 8882.22it/s] 22%|██▏       | 86790/400000 [00:10<00:34, 9036.66it/s] 22%|██▏       | 87696/400000 [00:10<00:34, 8941.48it/s] 22%|██▏       | 88592/400000 [00:10<00:35, 8776.80it/s] 22%|██▏       | 89502/400000 [00:10<00:35, 8869.54it/s] 23%|██▎       | 90439/400000 [00:10<00:34, 9011.47it/s] 23%|██▎       | 91373/400000 [00:10<00:33, 9105.92it/s] 23%|██▎       | 92285/400000 [00:10<00:33, 9104.30it/s] 23%|██▎       | 93197/400000 [00:10<00:34, 8970.29it/s] 24%|██▎       | 94096/400000 [00:10<00:34, 8899.15it/s] 24%|██▍       | 95019/400000 [00:10<00:33, 8993.17it/s] 24%|██▍       | 95920/400000 [00:11<00:33, 8972.16it/s] 24%|██▍       | 96818/400000 [00:11<00:34, 8865.47it/s] 24%|██▍       | 97706/400000 [00:11<00:34, 8660.11it/s] 25%|██▍       | 98574/400000 [00:11<00:34, 8661.09it/s] 25%|██▍       | 99456/400000 [00:11<00:34, 8707.04it/s] 25%|██▌       | 100359/400000 [00:11<00:34, 8798.98it/s] 25%|██▌       | 101267/400000 [00:11<00:33, 8880.88it/s] 26%|██▌       | 102156/400000 [00:11<00:33, 8825.65it/s] 26%|██▌       | 103040/400000 [00:11<00:33, 8823.92it/s] 26%|██▌       | 103923/400000 [00:11<00:33, 8744.31it/s] 26%|██▌       | 104798/400000 [00:12<00:33, 8707.54it/s] 26%|██▋       | 105670/400000 [00:12<00:33, 8708.85it/s] 27%|██▋       | 106575/400000 [00:12<00:33, 8805.93it/s] 27%|██▋       | 107457/400000 [00:12<00:33, 8807.32it/s] 27%|██▋       | 108339/400000 [00:12<00:33, 8682.02it/s] 27%|██▋       | 109215/400000 [00:12<00:33, 8702.67it/s] 28%|██▊       | 110091/400000 [00:12<00:33, 8717.53it/s] 28%|██▊       | 110964/400000 [00:12<00:33, 8657.19it/s] 28%|██▊       | 111831/400000 [00:12<00:33, 8643.83it/s] 28%|██▊       | 112732/400000 [00:12<00:32, 8748.08it/s] 28%|██▊       | 113608/400000 [00:13<00:32, 8721.88it/s] 29%|██▊       | 114484/400000 [00:13<00:32, 8730.93it/s] 29%|██▉       | 115358/400000 [00:13<00:32, 8666.35it/s] 29%|██▉       | 116259/400000 [00:13<00:32, 8766.09it/s] 29%|██▉       | 117137/400000 [00:13<00:33, 8566.80it/s] 30%|██▉       | 118007/400000 [00:13<00:32, 8605.52it/s] 30%|██▉       | 118908/400000 [00:13<00:32, 8720.54it/s] 30%|██▉       | 119782/400000 [00:13<00:32, 8608.37it/s] 30%|███       | 120665/400000 [00:13<00:32, 8672.73it/s] 30%|███       | 121564/400000 [00:13<00:31, 8763.97it/s] 31%|███       | 122442/400000 [00:14<00:31, 8709.86it/s] 31%|███       | 123314/400000 [00:14<00:31, 8664.81it/s] 31%|███       | 124181/400000 [00:14<00:32, 8498.22it/s] 31%|███▏      | 125032/400000 [00:14<00:32, 8498.84it/s] 31%|███▏      | 125894/400000 [00:14<00:32, 8534.02it/s] 32%|███▏      | 126797/400000 [00:14<00:31, 8675.49it/s] 32%|███▏      | 127672/400000 [00:14<00:31, 8695.89it/s] 32%|███▏      | 128543/400000 [00:14<00:31, 8626.23it/s] 32%|███▏      | 129463/400000 [00:14<00:30, 8790.33it/s] 33%|███▎      | 130408/400000 [00:14<00:30, 8975.86it/s] 33%|███▎      | 131308/400000 [00:15<00:30, 8858.46it/s] 33%|███▎      | 132218/400000 [00:15<00:29, 8928.76it/s] 33%|███▎      | 133113/400000 [00:15<00:29, 8902.39it/s] 34%|███▎      | 134035/400000 [00:15<00:29, 8993.43it/s] 34%|███▎      | 134962/400000 [00:15<00:29, 9074.30it/s] 34%|███▍      | 135871/400000 [00:15<00:29, 9035.91it/s] 34%|███▍      | 136776/400000 [00:15<00:29, 8947.91it/s] 34%|███▍      | 137672/400000 [00:15<00:29, 8841.96it/s] 35%|███▍      | 138557/400000 [00:15<00:29, 8822.53it/s] 35%|███▍      | 139440/400000 [00:16<00:29, 8759.24it/s] 35%|███▌      | 140317/400000 [00:16<00:29, 8729.82it/s] 35%|███▌      | 141200/400000 [00:16<00:29, 8757.70it/s] 36%|███▌      | 142106/400000 [00:16<00:29, 8845.56it/s] 36%|███▌      | 142995/400000 [00:16<00:29, 8857.57it/s] 36%|███▌      | 143882/400000 [00:16<00:29, 8756.59it/s] 36%|███▌      | 144773/400000 [00:16<00:28, 8801.21it/s] 36%|███▋      | 145654/400000 [00:16<00:29, 8724.19it/s] 37%|███▋      | 146527/400000 [00:16<00:29, 8652.77it/s] 37%|███▋      | 147399/400000 [00:16<00:29, 8671.54it/s] 37%|███▋      | 148267/400000 [00:17<00:29, 8663.99it/s] 37%|███▋      | 149134/400000 [00:17<00:29, 8622.71it/s] 38%|███▊      | 150010/400000 [00:17<00:28, 8657.73it/s] 38%|███▊      | 150876/400000 [00:17<00:28, 8591.57it/s] 38%|███▊      | 151737/400000 [00:17<00:28, 8595.92it/s] 38%|███▊      | 152605/400000 [00:17<00:28, 8619.96it/s] 38%|███▊      | 153468/400000 [00:17<00:28, 8588.26it/s] 39%|███▊      | 154333/400000 [00:17<00:28, 8605.56it/s] 39%|███▉      | 155194/400000 [00:17<00:28, 8592.95it/s] 39%|███▉      | 156061/400000 [00:17<00:28, 8615.36it/s] 39%|███▉      | 156939/400000 [00:18<00:28, 8663.38it/s] 39%|███▉      | 157806/400000 [00:18<00:28, 8635.55it/s] 40%|███▉      | 158670/400000 [00:18<00:27, 8619.68it/s] 40%|███▉      | 159533/400000 [00:18<00:27, 8594.47it/s] 40%|████      | 160419/400000 [00:18<00:27, 8671.03it/s] 40%|████      | 161311/400000 [00:18<00:27, 8742.93it/s] 41%|████      | 162196/400000 [00:18<00:27, 8772.34it/s] 41%|████      | 163074/400000 [00:18<00:27, 8735.34it/s] 41%|████      | 163948/400000 [00:18<00:27, 8496.47it/s] 41%|████      | 164845/400000 [00:18<00:27, 8631.10it/s] 41%|████▏     | 165720/400000 [00:19<00:27, 8663.82it/s] 42%|████▏     | 166588/400000 [00:19<00:26, 8650.76it/s] 42%|████▏     | 167496/400000 [00:19<00:26, 8772.47it/s] 42%|████▏     | 168375/400000 [00:19<00:26, 8718.87it/s] 42%|████▏     | 169249/400000 [00:19<00:26, 8723.33it/s] 43%|████▎     | 170174/400000 [00:19<00:25, 8873.18it/s] 43%|████▎     | 171123/400000 [00:19<00:25, 9049.21it/s] 43%|████▎     | 172030/400000 [00:19<00:25, 9049.37it/s] 43%|████▎     | 172937/400000 [00:19<00:25, 8919.33it/s] 43%|████▎     | 173831/400000 [00:19<00:25, 8835.99it/s] 44%|████▎     | 174720/400000 [00:20<00:25, 8850.82it/s] 44%|████▍     | 175606/400000 [00:20<00:25, 8796.44it/s] 44%|████▍     | 176520/400000 [00:20<00:25, 8894.38it/s] 44%|████▍     | 177411/400000 [00:20<00:25, 8841.04it/s] 45%|████▍     | 178312/400000 [00:20<00:24, 8889.91it/s] 45%|████▍     | 179203/400000 [00:20<00:24, 8895.06it/s] 45%|████▌     | 180140/400000 [00:20<00:24, 9030.85it/s] 45%|████▌     | 181075/400000 [00:20<00:23, 9122.05it/s] 45%|████▌     | 181988/400000 [00:20<00:24, 9021.90it/s] 46%|████▌     | 182914/400000 [00:20<00:23, 9089.57it/s] 46%|████▌     | 183829/400000 [00:21<00:23, 9106.44it/s] 46%|████▌     | 184741/400000 [00:21<00:24, 8949.65it/s] 46%|████▋     | 185637/400000 [00:21<00:24, 8930.31it/s] 47%|████▋     | 186531/400000 [00:21<00:24, 8755.66it/s] 47%|████▋     | 187408/400000 [00:21<00:24, 8647.91it/s] 47%|████▋     | 188284/400000 [00:21<00:24, 8680.37it/s] 47%|████▋     | 189186/400000 [00:21<00:24, 8779.05it/s] 48%|████▊     | 190075/400000 [00:21<00:23, 8810.82it/s] 48%|████▊     | 190957/400000 [00:21<00:23, 8745.75it/s] 48%|████▊     | 191834/400000 [00:21<00:23, 8751.16it/s] 48%|████▊     | 192730/400000 [00:22<00:23, 8812.68it/s] 48%|████▊     | 193612/400000 [00:22<00:23, 8737.65it/s] 49%|████▊     | 194487/400000 [00:22<00:23, 8724.79it/s] 49%|████▉     | 195360/400000 [00:22<00:23, 8591.72it/s] 49%|████▉     | 196269/400000 [00:22<00:23, 8733.81it/s] 49%|████▉     | 197148/400000 [00:22<00:23, 8748.71it/s] 50%|████▉     | 198041/400000 [00:22<00:22, 8799.54it/s] 50%|████▉     | 198968/400000 [00:22<00:22, 8935.51it/s] 50%|████▉     | 199863/400000 [00:22<00:22, 8788.79it/s] 50%|█████     | 200758/400000 [00:23<00:22, 8834.99it/s] 50%|█████     | 201663/400000 [00:23<00:22, 8896.05it/s] 51%|█████     | 202554/400000 [00:23<00:22, 8844.33it/s] 51%|█████     | 203456/400000 [00:23<00:22, 8896.10it/s] 51%|█████     | 204365/400000 [00:23<00:21, 8952.10it/s] 51%|█████▏    | 205300/400000 [00:23<00:21, 9066.09it/s] 52%|█████▏    | 206221/400000 [00:23<00:21, 9108.54it/s] 52%|█████▏    | 207133/400000 [00:23<00:21, 9105.81it/s] 52%|█████▏    | 208049/400000 [00:23<00:21, 9120.49it/s] 52%|█████▏    | 208962/400000 [00:23<00:21, 8848.05it/s] 52%|█████▏    | 209858/400000 [00:24<00:21, 8879.06it/s] 53%|█████▎    | 210748/400000 [00:24<00:21, 8851.74it/s] 53%|█████▎    | 211637/400000 [00:24<00:21, 8860.19it/s] 53%|█████▎    | 212579/400000 [00:24<00:20, 9018.77it/s] 53%|█████▎    | 213483/400000 [00:24<00:20, 8916.76it/s] 54%|█████▎    | 214376/400000 [00:24<00:20, 8870.65it/s] 54%|█████▍    | 215280/400000 [00:24<00:20, 8919.17it/s] 54%|█████▍    | 216190/400000 [00:24<00:20, 8970.08it/s] 54%|█████▍    | 217088/400000 [00:24<00:20, 8926.85it/s] 54%|█████▍    | 217982/400000 [00:24<00:20, 8869.47it/s] 55%|█████▍    | 218870/400000 [00:25<00:20, 8862.11it/s] 55%|█████▍    | 219757/400000 [00:25<00:20, 8782.05it/s] 55%|█████▌    | 220636/400000 [00:25<00:20, 8759.44it/s] 55%|█████▌    | 221518/400000 [00:25<00:20, 8775.08it/s] 56%|█████▌    | 222396/400000 [00:25<00:20, 8742.29it/s] 56%|█████▌    | 223323/400000 [00:25<00:19, 8892.96it/s] 56%|█████▌    | 224214/400000 [00:25<00:19, 8871.99it/s] 56%|█████▋    | 225105/400000 [00:25<00:19, 8881.56it/s] 56%|█████▋    | 225994/400000 [00:25<00:19, 8847.02it/s] 57%|█████▋    | 226879/400000 [00:25<00:19, 8717.59it/s] 57%|█████▋    | 227752/400000 [00:26<00:19, 8670.89it/s] 57%|█████▋    | 228620/400000 [00:26<00:19, 8640.36it/s] 57%|█████▋    | 229486/400000 [00:26<00:19, 8643.46it/s] 58%|█████▊    | 230363/400000 [00:26<00:19, 8679.62it/s] 58%|█████▊    | 231232/400000 [00:26<00:19, 8633.39it/s] 58%|█████▊    | 232121/400000 [00:26<00:19, 8706.19it/s] 58%|█████▊    | 233013/400000 [00:26<00:19, 8767.38it/s] 58%|█████▊    | 233901/400000 [00:26<00:18, 8798.05it/s] 59%|█████▊    | 234791/400000 [00:26<00:18, 8826.16it/s] 59%|█████▉    | 235674/400000 [00:26<00:18, 8783.57it/s] 59%|█████▉    | 236554/400000 [00:27<00:18, 8785.88it/s] 59%|█████▉    | 237433/400000 [00:27<00:18, 8728.15it/s] 60%|█████▉    | 238306/400000 [00:27<00:18, 8644.47it/s] 60%|█████▉    | 239176/400000 [00:27<00:18, 8657.93it/s] 60%|██████    | 240043/400000 [00:27<00:18, 8489.47it/s] 60%|██████    | 240906/400000 [00:27<00:18, 8529.72it/s] 60%|██████    | 241779/400000 [00:27<00:18, 8586.60it/s] 61%|██████    | 242659/400000 [00:27<00:18, 8646.99it/s] 61%|██████    | 243536/400000 [00:27<00:18, 8681.64it/s] 61%|██████    | 244405/400000 [00:27<00:18, 8629.88it/s] 61%|██████▏   | 245273/400000 [00:28<00:17, 8642.69it/s] 62%|██████▏   | 246150/400000 [00:28<00:17, 8678.37it/s] 62%|██████▏   | 247034/400000 [00:28<00:17, 8724.36it/s] 62%|██████▏   | 247969/400000 [00:28<00:17, 8900.58it/s] 62%|██████▏   | 248861/400000 [00:28<00:16, 8902.76it/s] 62%|██████▏   | 249752/400000 [00:28<00:16, 8851.48it/s] 63%|██████▎   | 250680/400000 [00:28<00:16, 8973.09it/s] 63%|██████▎   | 251646/400000 [00:28<00:16, 9166.96it/s] 63%|██████▎   | 252593/400000 [00:28<00:15, 9255.54it/s] 63%|██████▎   | 253520/400000 [00:28<00:15, 9237.72it/s] 64%|██████▎   | 254445/400000 [00:29<00:16, 9075.63it/s] 64%|██████▍   | 255354/400000 [00:29<00:16, 8806.54it/s] 64%|██████▍   | 256239/400000 [00:29<00:16, 8818.78it/s] 64%|██████▍   | 257141/400000 [00:29<00:16, 8877.59it/s] 65%|██████▍   | 258031/400000 [00:29<00:16, 8763.44it/s] 65%|██████▍   | 258967/400000 [00:29<00:15, 8932.23it/s] 65%|██████▍   | 259890/400000 [00:29<00:15, 9018.18it/s] 65%|██████▌   | 260813/400000 [00:29<00:15, 9078.29it/s] 65%|██████▌   | 261722/400000 [00:29<00:15, 9056.10it/s] 66%|██████▌   | 262629/400000 [00:29<00:15, 9026.95it/s] 66%|██████▌   | 263539/400000 [00:30<00:15, 9048.18it/s] 66%|██████▌   | 264445/400000 [00:30<00:15, 9004.16it/s] 66%|██████▋   | 265346/400000 [00:30<00:15, 8941.91it/s] 67%|██████▋   | 266241/400000 [00:30<00:15, 8899.59it/s] 67%|██████▋   | 267132/400000 [00:30<00:15, 8815.05it/s] 67%|██████▋   | 268054/400000 [00:30<00:14, 8931.24it/s] 67%|██████▋   | 269005/400000 [00:30<00:14, 9096.16it/s] 67%|██████▋   | 269928/400000 [00:30<00:14, 9133.88it/s] 68%|██████▊   | 270843/400000 [00:30<00:14, 9101.58it/s] 68%|██████▊   | 271754/400000 [00:31<00:14, 9072.11it/s] 68%|██████▊   | 272662/400000 [00:31<00:14, 9031.95it/s] 68%|██████▊   | 273580/400000 [00:31<00:13, 9073.68it/s] 69%|██████▊   | 274514/400000 [00:31<00:13, 9151.74it/s] 69%|██████▉   | 275436/400000 [00:31<00:13, 9169.75it/s] 69%|██████▉   | 276354/400000 [00:31<00:13, 9012.09it/s] 69%|██████▉   | 277261/400000 [00:31<00:13, 9029.32it/s] 70%|██████▉   | 278181/400000 [00:31<00:13, 9078.91it/s] 70%|██████▉   | 279124/400000 [00:31<00:13, 9179.47it/s] 70%|███████   | 280043/400000 [00:31<00:13, 9112.22it/s] 70%|███████   | 280955/400000 [00:32<00:13, 9004.15it/s] 70%|███████   | 281857/400000 [00:32<00:13, 8856.98it/s] 71%|███████   | 282744/400000 [00:32<00:13, 8768.63it/s] 71%|███████   | 283658/400000 [00:32<00:13, 8874.87it/s] 71%|███████   | 284547/400000 [00:32<00:13, 8765.98it/s] 71%|███████▏  | 285425/400000 [00:32<00:13, 8739.75it/s] 72%|███████▏  | 286373/400000 [00:32<00:12, 8948.61it/s] 72%|███████▏  | 287324/400000 [00:32<00:12, 9108.37it/s] 72%|███████▏  | 288266/400000 [00:32<00:12, 9198.57it/s] 72%|███████▏  | 289188/400000 [00:32<00:12, 9105.29it/s] 73%|███████▎  | 290108/400000 [00:33<00:12, 9131.85it/s] 73%|███████▎  | 291023/400000 [00:33<00:12, 8946.02it/s] 73%|███████▎  | 291928/400000 [00:33<00:12, 8974.03it/s] 73%|███████▎  | 292881/400000 [00:33<00:11, 9132.53it/s] 73%|███████▎  | 293796/400000 [00:33<00:11, 9044.51it/s] 74%|███████▎  | 294702/400000 [00:33<00:11, 8914.37it/s] 74%|███████▍  | 295595/400000 [00:33<00:11, 8861.68it/s] 74%|███████▍  | 296502/400000 [00:33<00:11, 8922.04it/s] 74%|███████▍  | 297445/400000 [00:33<00:11, 9068.35it/s] 75%|███████▍  | 298353/400000 [00:33<00:11, 9024.29it/s] 75%|███████▍  | 299257/400000 [00:34<00:11, 8746.28it/s] 75%|███████▌  | 300135/400000 [00:34<00:11, 8678.96it/s] 75%|███████▌  | 301005/400000 [00:34<00:11, 8450.62it/s] 75%|███████▌  | 301853/400000 [00:34<00:11, 8433.64it/s] 76%|███████▌  | 302725/400000 [00:34<00:11, 8516.57it/s] 76%|███████▌  | 303579/400000 [00:34<00:11, 8131.53it/s] 76%|███████▌  | 304448/400000 [00:34<00:11, 8289.26it/s] 76%|███████▋  | 305352/400000 [00:34<00:11, 8499.79it/s] 77%|███████▋  | 306207/400000 [00:34<00:11, 8499.19it/s] 77%|███████▋  | 307060/400000 [00:34<00:10, 8496.22it/s] 77%|███████▋  | 307912/400000 [00:35<00:10, 8484.33it/s] 77%|███████▋  | 308778/400000 [00:35<00:10, 8534.08it/s] 77%|███████▋  | 309643/400000 [00:35<00:10, 8567.51it/s] 78%|███████▊  | 310559/400000 [00:35<00:10, 8722.76it/s] 78%|███████▊  | 311433/400000 [00:35<00:10, 8273.59it/s] 78%|███████▊  | 312273/400000 [00:35<00:10, 8310.87it/s] 78%|███████▊  | 313206/400000 [00:35<00:10, 8592.20it/s] 79%|███████▊  | 314137/400000 [00:35<00:09, 8795.44it/s] 79%|███████▉  | 315022/400000 [00:35<00:09, 8554.26it/s] 79%|███████▉  | 315936/400000 [00:36<00:09, 8721.25it/s] 79%|███████▉  | 316813/400000 [00:36<00:09, 8645.33it/s] 79%|███████▉  | 317719/400000 [00:36<00:09, 8762.11it/s] 80%|███████▉  | 318642/400000 [00:36<00:09, 8895.23it/s] 80%|███████▉  | 319570/400000 [00:36<00:08, 9004.84it/s] 80%|████████  | 320473/400000 [00:36<00:08, 9008.76it/s] 80%|████████  | 321376/400000 [00:36<00:08, 8928.70it/s] 81%|████████  | 322306/400000 [00:36<00:08, 9035.32it/s] 81%|████████  | 323225/400000 [00:36<00:08, 9080.54it/s] 81%|████████  | 324151/400000 [00:36<00:08, 9131.55it/s] 81%|████████▏ | 325065/400000 [00:37<00:08, 9124.05it/s] 81%|████████▏ | 325978/400000 [00:37<00:08, 8949.24it/s] 82%|████████▏ | 326885/400000 [00:37<00:08, 8982.29it/s] 82%|████████▏ | 327794/400000 [00:37<00:08, 9013.34it/s] 82%|████████▏ | 328705/400000 [00:37<00:07, 9041.34it/s] 82%|████████▏ | 329642/400000 [00:37<00:07, 9135.13it/s] 83%|████████▎ | 330573/400000 [00:37<00:07, 9184.49it/s] 83%|████████▎ | 331544/400000 [00:37<00:07, 9335.98it/s] 83%|████████▎ | 332479/400000 [00:37<00:07, 9226.85it/s] 83%|████████▎ | 333418/400000 [00:37<00:07, 9272.44it/s] 84%|████████▎ | 334349/400000 [00:38<00:07, 9281.03it/s] 84%|████████▍ | 335278/400000 [00:38<00:07, 8959.15it/s] 84%|████████▍ | 336177/400000 [00:38<00:07, 8963.86it/s] 84%|████████▍ | 337076/400000 [00:38<00:07, 8937.77it/s] 84%|████████▍ | 337990/400000 [00:38<00:06, 8997.14it/s] 85%|████████▍ | 338899/400000 [00:38<00:06, 9023.41it/s] 85%|████████▍ | 339803/400000 [00:38<00:06, 8964.16it/s] 85%|████████▌ | 340720/400000 [00:38<00:06, 9024.91it/s] 85%|████████▌ | 341624/400000 [00:38<00:06, 8974.57it/s] 86%|████████▌ | 342522/400000 [00:38<00:06, 8887.93it/s] 86%|████████▌ | 343426/400000 [00:39<00:06, 8931.55it/s] 86%|████████▌ | 344320/400000 [00:39<00:06, 8807.90it/s] 86%|████████▋ | 345202/400000 [00:39<00:06, 8773.29it/s] 87%|████████▋ | 346080/400000 [00:39<00:06, 8649.95it/s] 87%|████████▋ | 346946/400000 [00:39<00:06, 8529.78it/s] 87%|████████▋ | 347800/400000 [00:39<00:06, 8286.10it/s] 87%|████████▋ | 348649/400000 [00:39<00:06, 8345.41it/s] 87%|████████▋ | 349533/400000 [00:39<00:05, 8485.00it/s] 88%|████████▊ | 350384/400000 [00:39<00:05, 8423.04it/s] 88%|████████▊ | 351296/400000 [00:39<00:05, 8620.44it/s] 88%|████████▊ | 352164/400000 [00:40<00:05, 8635.81it/s] 88%|████████▊ | 353030/400000 [00:40<00:05, 8554.24it/s] 88%|████████▊ | 353916/400000 [00:40<00:05, 8642.18it/s] 89%|████████▊ | 354792/400000 [00:40<00:05, 8675.21it/s] 89%|████████▉ | 355697/400000 [00:40<00:05, 8783.26it/s] 89%|████████▉ | 356592/400000 [00:40<00:04, 8831.82it/s] 89%|████████▉ | 357476/400000 [00:40<00:04, 8779.77it/s] 90%|████████▉ | 358355/400000 [00:40<00:04, 8768.98it/s] 90%|████████▉ | 359241/400000 [00:40<00:04, 8795.36it/s] 90%|█████████ | 360121/400000 [00:41<00:04, 8789.25it/s] 90%|█████████ | 361007/400000 [00:41<00:04, 8808.83it/s] 90%|█████████ | 361889/400000 [00:41<00:04, 8728.84it/s] 91%|█████████ | 362804/400000 [00:41<00:04, 8849.62it/s] 91%|█████████ | 363700/400000 [00:41<00:04, 8880.15it/s] 91%|█████████ | 364589/400000 [00:41<00:04, 8826.40it/s] 91%|█████████▏| 365481/400000 [00:41<00:03, 8851.82it/s] 92%|█████████▏| 366367/400000 [00:41<00:03, 8828.81it/s] 92%|█████████▏| 367328/400000 [00:41<00:03, 9048.34it/s] 92%|█████████▏| 368260/400000 [00:41<00:03, 9125.83it/s] 92%|█████████▏| 369222/400000 [00:42<00:03, 9268.18it/s] 93%|█████████▎| 370151/400000 [00:42<00:03, 9177.82it/s] 93%|█████████▎| 371070/400000 [00:42<00:03, 9150.91it/s] 93%|█████████▎| 372009/400000 [00:42<00:03, 9219.53it/s] 93%|█████████▎| 372932/400000 [00:42<00:03, 9022.22it/s] 93%|█████████▎| 373836/400000 [00:42<00:02, 8947.78it/s] 94%|█████████▎| 374745/400000 [00:42<00:02, 8989.82it/s] 94%|█████████▍| 375656/400000 [00:42<00:02, 9023.28it/s] 94%|█████████▍| 376601/400000 [00:42<00:02, 9144.64it/s] 94%|█████████▍| 377517/400000 [00:42<00:02, 9091.51it/s] 95%|█████████▍| 378444/400000 [00:43<00:02, 9142.16it/s] 95%|█████████▍| 379359/400000 [00:43<00:02, 9007.95it/s] 95%|█████████▌| 380261/400000 [00:43<00:02, 8955.03it/s] 95%|█████████▌| 381204/400000 [00:43<00:02, 9091.93it/s] 96%|█████████▌| 382130/400000 [00:43<00:01, 9141.00it/s] 96%|█████████▌| 383045/400000 [00:43<00:01, 8998.09it/s] 96%|█████████▌| 383946/400000 [00:43<00:01, 8908.46it/s] 96%|█████████▌| 384838/400000 [00:43<00:01, 8734.91it/s] 96%|█████████▋| 385761/400000 [00:43<00:01, 8876.43it/s] 97%|█████████▋| 386668/400000 [00:43<00:01, 8931.09it/s] 97%|█████████▋| 387565/400000 [00:44<00:01, 8939.98it/s] 97%|█████████▋| 388460/400000 [00:44<00:01, 8828.59it/s] 97%|█████████▋| 389344/400000 [00:44<00:01, 8733.51it/s] 98%|█████████▊| 390219/400000 [00:44<00:01, 8704.79it/s] 98%|█████████▊| 391111/400000 [00:44<00:01, 8766.52it/s] 98%|█████████▊| 391989/400000 [00:44<00:00, 8745.09it/s] 98%|█████████▊| 392864/400000 [00:44<00:00, 8717.57it/s] 98%|█████████▊| 393737/400000 [00:44<00:00, 8417.97it/s] 99%|█████████▊| 394582/400000 [00:44<00:00, 8321.57it/s] 99%|█████████▉| 395440/400000 [00:44<00:00, 8396.91it/s] 99%|█████████▉| 396334/400000 [00:45<00:00, 8551.92it/s] 99%|█████████▉| 397237/400000 [00:45<00:00, 8689.16it/s]100%|█████████▉| 398158/400000 [00:45<00:00, 8837.61it/s]100%|█████████▉| 399097/400000 [00:45<00:00, 8995.84it/s]100%|█████████▉| 399999/400000 [00:45<00:00, 8796.13it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f3d650f7be0>, <torchtext.data.dataset.TabularDataset object at 0x7f3d650f7d30>, <torchtext.vocab.Vocab object at 0x7f3d650f7c50>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 12.20 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 11.74 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 11.74 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  8.07 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  8.07 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.14 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.14 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.25 file/s]2020-07-19 00:18:26.480257: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-19 00:18:26.484772: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-07-19 00:18:26.484930: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x561fd3274230 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-19 00:18:26.484945: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 41%|████      | 4038656/9912422 [00:00<00:00, 40386502.23it/s]9920512it [00:00, 32471586.66it/s]                             
0it [00:00, ?it/s]32768it [00:00, 570679.20it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 461803.75it/s]1654784it [00:00, 11700027.90it/s]                         
0it [00:00, ?it/s]8192it [00:00, 186069.27it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
