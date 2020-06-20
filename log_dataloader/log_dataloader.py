
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/bd7dfc233939710e05244f8e6a394a7ce12a3485', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': 'bd7dfc233939710e05244f8e6a394a7ce12a3485', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/bd7dfc233939710e05244f8e6a394a7ce12a3485

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/bd7dfc233939710e05244f8e6a394a7ce12a3485

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/bd7dfc233939710e05244f8e6a394a7ce12a3485

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7fe5200b8e18> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7fe5200b8e18> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7fe58b66f048> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7fe58b66f048> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7fe5a539de18> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7fe5a539de18> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fe538eee8c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fe538eee8c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fe538eee8c8> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:01, 161844.07it/s] 75%|███████▌  | 7454720/9912422 [00:00<00:10, 230986.74it/s]9920512it [00:00, 45201538.19it/s]                           
0it [00:00, ?it/s]32768it [00:00, 685088.12it/s]
0it [00:00, ?it/s]  5%|▌         | 90112/1648877 [00:00<00:01, 891332.63it/s]1654784it [00:00, 11840181.70it/s]                         
0it [00:00, ?it/s]8192it [00:00, 234179.40it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe53645b518>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe5365238d0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7fe538eee510> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7fe538eee510> 

  function with postional parmater data_info <function tf_dataset_download at 0x7fe538eee510> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:36,  1.67 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:36,  1.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   1%|          | 2/162 [00:00<01:12,  2.21 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:12,  2.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:11,  2.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   2%|▏         | 4/162 [00:00<00:54,  2.92 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<00:54,  2.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<00:53,  2.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▎         | 6/162 [00:01<00:40,  3.87 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:01<00:40,  3.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:01<00:40,  3.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:01<00:30,  5.10 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:01<00:30,  5.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:01<00:30,  5.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:01<00:29,  5.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   7%|▋         | 11/162 [00:01<00:22,  6.69 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:01<00:22,  6.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:01<00:22,  6.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:01<00:22,  6.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   9%|▊         | 14/162 [00:01<00:16,  8.71 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:01<00:16,  8.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:01<00:16,  8.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:01<00:16,  8.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:01<00:16,  8.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  11%|█         | 18/162 [00:01<00:12, 11.25 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:01<00:12, 11.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:01<00:12, 11.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:01<00:12, 11.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:01<00:12, 11.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:01<00:12, 11.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  14%|█▍        | 23/162 [00:01<00:09, 14.51 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:01<00:09, 14.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:01<00:09, 14.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:01<00:09, 14.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:01<00:09, 14.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:01<00:09, 14.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:01<00:09, 14.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  18%|█▊        | 29/162 [00:01<00:07, 18.42 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:01<00:07, 18.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:01<00:07, 18.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:01<00:07, 18.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:01<00:07, 18.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:01<00:07, 18.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:06, 18.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  22%|██▏       | 35/162 [00:01<00:05, 22.90 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:05, 22.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:05, 22.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:05, 22.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:05, 22.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:05, 22.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:05, 22.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  25%|██▌       | 41/162 [00:01<00:04, 27.41 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:04, 27.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:04, 27.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:04, 27.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:04, 27.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:04, 27.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:02<00:04, 27.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  29%|██▉       | 47/162 [00:02<00:03, 32.07 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:02<00:03, 32.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:02<00:03, 32.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:02<00:03, 32.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:02<00:03, 32.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:02<00:03, 32.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:02<00:03, 32.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  33%|███▎      | 53/162 [00:02<00:03, 36.24 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:02<00:03, 36.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:02<00:02, 36.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:02<00:02, 36.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:02<00:02, 36.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:02<00:02, 36.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:02<00:02, 36.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  36%|███▋      | 59/162 [00:02<00:02, 39.98 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:02<00:02, 39.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:02<00:02, 39.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:02<00:02, 39.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:02<00:02, 39.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:02<00:02, 39.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:02<00:02, 39.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  40%|████      | 65/162 [00:02<00:02, 43.12 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:02<00:02, 43.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:02<00:02, 43.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:02<00:02, 43.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:02<00:02, 43.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:02<00:02, 43.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:02<00:02, 43.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  44%|████▍     | 71/162 [00:02<00:01, 45.66 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:02<00:01, 45.66 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:02<00:01, 45.66 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:02<00:01, 45.66 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:02<00:01, 45.66 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:02<00:01, 45.66 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:02<00:01, 45.66 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  48%|████▊     | 77/162 [00:02<00:01, 47.71 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:02<00:01, 47.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:02<00:01, 47.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:02<00:01, 47.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:02<00:01, 47.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:02<00:01, 47.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:02<00:01, 47.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  51%|█████     | 83/162 [00:02<00:01, 49.17 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:02<00:01, 49.17 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:02<00:01, 49.17 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:02<00:01, 49.17 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:02<00:01, 49.17 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:02<00:01, 49.17 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:02<00:01, 49.17 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  55%|█████▍    | 89/162 [00:02<00:01, 49.42 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:02<00:01, 49.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:02<00:01, 49.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:02<00:01, 49.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:02<00:01, 49.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:02<00:01, 49.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:02<00:01, 49.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  59%|█████▊    | 95/162 [00:02<00:01, 50.51 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:02<00:01, 50.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:02<00:01, 50.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:02<00:01, 50.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:02<00:01, 50.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:03<00:01, 50.51 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:03<00:01, 50.51 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  62%|██████▏   | 101/162 [00:03<00:01, 51.17 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:03<00:01, 51.17 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:03<00:01, 51.17 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:03<00:01, 51.17 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:03<00:01, 51.17 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:03<00:01, 51.17 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:03<00:01, 51.17 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  66%|██████▌   | 107/162 [00:03<00:01, 51.78 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:03<00:01, 51.78 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:03<00:01, 51.78 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:03<00:01, 51.78 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:03<00:01, 51.78 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:03<00:00, 51.78 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:03<00:00, 51.78 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  70%|██████▉   | 113/162 [00:03<00:00, 51.59 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:03<00:00, 51.59 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:03<00:00, 51.59 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:03<00:00, 51.59 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:03<00:00, 51.59 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:03<00:00, 51.59 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:03<00:00, 51.59 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  73%|███████▎  | 119/162 [00:03<00:00, 52.90 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:03<00:00, 52.90 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:03<00:00, 52.90 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:03<00:00, 52.90 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:03<00:00, 52.90 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:03<00:00, 52.90 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:03<00:00, 52.90 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  77%|███████▋  | 125/162 [00:03<00:00, 52.85 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:03<00:00, 52.85 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:03<00:00, 52.85 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:03<00:00, 52.85 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:03<00:00, 52.85 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:03<00:00, 52.85 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:03<00:00, 52.85 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  81%|████████  | 131/162 [00:03<00:00, 52.14 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:03<00:00, 52.14 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:03<00:00, 52.14 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:03<00:00, 52.14 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:03<00:00, 52.14 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:03<00:00, 52.14 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:03<00:00, 52.14 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  85%|████████▍ | 137/162 [00:03<00:00, 53.30 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:03<00:00, 53.30 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:03<00:00, 53.30 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:03<00:00, 53.30 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:03<00:00, 53.30 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:03<00:00, 53.30 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:03<00:00, 53.30 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  88%|████████▊ | 143/162 [00:03<00:00, 52.95 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:03<00:00, 52.95 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:03<00:00, 52.95 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:03<00:00, 52.95 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:03<00:00, 52.95 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:03<00:00, 52.95 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:03<00:00, 52.95 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  92%|█████████▏| 149/162 [00:03<00:00, 52.68 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:03<00:00, 52.68 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:03<00:00, 52.68 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:03<00:00, 52.68 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:04<00:00, 52.68 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:04<00:00, 52.68 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:04<00:00, 52.68 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  96%|█████████▌| 155/162 [00:04<00:00, 53.66 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:04<00:00, 53.66 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:04<00:00, 53.66 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:04<00:00, 53.66 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:04<00:00, 53.66 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:04<00:00, 53.66 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:04<00:00, 53.66 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  99%|█████████▉| 161/162 [00:04<00:00, 54.10 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:04<00:00, 54.10 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 54.10 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.19s/ url]Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.19s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 54.10 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.19s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 54.10 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:04<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:06<00:00,  6.30s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:06<00:00,  4.19s/ url]
Dl Size...: 100%|██████████| 162/162 [00:06<00:00, 54.10 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:06<00:00,  6.30s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:06<00:00,  6.31s/ file]
Dl Size...: 100%|██████████| 162/162 [00:06<00:00, 25.69 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:06<00:00,  6.31s/ url]
0 examples [00:00, ? examples/s]2020-06-20 00:10:46.432264: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-06-20 00:10:46.444679: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095125000 Hz
2020-06-20 00:10:46.444934: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55ab48b19f30 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-20 00:10:46.444955: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  2.31 examples/s]80 examples [00:00,  3.30 examples/s]186 examples [00:00,  4.70 examples/s]282 examples [00:00,  6.70 examples/s]375 examples [00:00,  9.55 examples/s]483 examples [00:00, 13.59 examples/s]586 examples [00:01, 19.30 examples/s]683 examples [00:01, 27.34 examples/s]789 examples [00:01, 38.63 examples/s]887 examples [00:01, 54.26 examples/s]984 examples [00:01, 75.70 examples/s]1089 examples [00:01, 104.88 examples/s]1197 examples [00:01, 143.83 examples/s]1304 examples [00:01, 194.21 examples/s]1407 examples [00:01, 256.29 examples/s]1514 examples [00:01, 331.99 examples/s]1618 examples [00:02, 406.22 examples/s]1720 examples [00:02, 495.51 examples/s]1825 examples [00:02, 588.27 examples/s]1926 examples [00:02, 656.52 examples/s]2029 examples [00:02, 736.24 examples/s]2133 examples [00:02, 806.51 examples/s]2234 examples [00:02, 836.29 examples/s]2339 examples [00:02, 888.95 examples/s]2439 examples [00:02, 903.30 examples/s]2541 examples [00:02, 932.49 examples/s]2640 examples [00:03, 942.51 examples/s]2743 examples [00:03, 966.33 examples/s]2843 examples [00:03, 957.95 examples/s]2941 examples [00:03, 961.70 examples/s]3045 examples [00:03, 983.43 examples/s]3155 examples [00:03, 1015.69 examples/s]3259 examples [00:03, 1020.78 examples/s]3362 examples [00:03, 992.94 examples/s] 3466 examples [00:03, 1006.07 examples/s]3571 examples [00:04, 1017.60 examples/s]3674 examples [00:04, 1000.02 examples/s]3775 examples [00:04, 987.82 examples/s] 3875 examples [00:04, 942.09 examples/s]3972 examples [00:04, 948.17 examples/s]4073 examples [00:04, 965.08 examples/s]4180 examples [00:04, 993.38 examples/s]4285 examples [00:04, 1007.40 examples/s]4387 examples [00:04, 986.24 examples/s] 4497 examples [00:04, 1017.46 examples/s]4606 examples [00:05, 1037.44 examples/s]4713 examples [00:05, 1046.21 examples/s]4818 examples [00:05, 992.56 examples/s] 4919 examples [00:05, 969.92 examples/s]5023 examples [00:05, 987.82 examples/s]5123 examples [00:05, 955.41 examples/s]5221 examples [00:05, 960.34 examples/s]5323 examples [00:05, 976.83 examples/s]5428 examples [00:05, 995.97 examples/s]5529 examples [00:06, 999.55 examples/s]5630 examples [00:06, 955.52 examples/s]5734 examples [00:06, 978.47 examples/s]5833 examples [00:06, 980.75 examples/s]5938 examples [00:06, 998.44 examples/s]6045 examples [00:06, 1018.67 examples/s]6153 examples [00:06, 1035.57 examples/s]6257 examples [00:06, 1000.01 examples/s]6358 examples [00:06, 998.25 examples/s] 6459 examples [00:06, 973.44 examples/s]6557 examples [00:07, 965.19 examples/s]6659 examples [00:07, 979.13 examples/s]6759 examples [00:07, 984.37 examples/s]6858 examples [00:07, 975.96 examples/s]6965 examples [00:07, 1001.10 examples/s]7066 examples [00:07, 997.42 examples/s] 7178 examples [00:07, 1028.87 examples/s]7283 examples [00:07, 1035.01 examples/s]7387 examples [00:07, 1016.21 examples/s]7491 examples [00:07, 1021.84 examples/s]7595 examples [00:08, 1024.70 examples/s]7698 examples [00:08, 1024.37 examples/s]7803 examples [00:08, 1030.23 examples/s]7907 examples [00:08, 1025.52 examples/s]8010 examples [00:08, 1026.59 examples/s]8116 examples [00:08, 1035.02 examples/s]8221 examples [00:08, 1036.83 examples/s]8327 examples [00:08, 1043.50 examples/s]8432 examples [00:08, 1039.62 examples/s]8540 examples [00:08, 1050.43 examples/s]8646 examples [00:09, 1032.43 examples/s]8750 examples [00:09, 1031.16 examples/s]8854 examples [00:09, 1031.87 examples/s]8958 examples [00:09, 1006.70 examples/s]9059 examples [00:09, 999.29 examples/s] 9160 examples [00:09, 1000.46 examples/s]9264 examples [00:09, 1011.71 examples/s]9367 examples [00:09, 1015.02 examples/s]9469 examples [00:09, 988.71 examples/s] 9569 examples [00:10, 974.74 examples/s]9672 examples [00:10, 989.26 examples/s]9773 examples [00:10, 995.34 examples/s]9873 examples [00:10, 982.83 examples/s]9981 examples [00:10, 1007.65 examples/s]10083 examples [00:10, 945.96 examples/s]10191 examples [00:10, 981.40 examples/s]10292 examples [00:10, 988.52 examples/s]10392 examples [00:10, 967.70 examples/s]10490 examples [00:10, 946.42 examples/s]10597 examples [00:11, 978.46 examples/s]10703 examples [00:11, 1000.63 examples/s]10811 examples [00:11, 1021.01 examples/s]10917 examples [00:11, 1032.15 examples/s]11027 examples [00:11, 1049.02 examples/s]11136 examples [00:11, 1059.71 examples/s]11243 examples [00:11, 1058.65 examples/s]11351 examples [00:11, 1063.90 examples/s]11459 examples [00:11, 1068.09 examples/s]11566 examples [00:11, 1067.24 examples/s]11676 examples [00:12, 1075.20 examples/s]11784 examples [00:12, 1038.51 examples/s]11889 examples [00:12, 996.86 examples/s] 11990 examples [00:12, 999.20 examples/s]12099 examples [00:12, 1022.81 examples/s]12206 examples [00:12, 1034.83 examples/s]12314 examples [00:12, 1046.76 examples/s]12419 examples [00:12, 1000.25 examples/s]12520 examples [00:12, 978.54 examples/s] 12619 examples [00:13, 964.50 examples/s]12718 examples [00:13, 971.68 examples/s]12823 examples [00:13, 991.10 examples/s]12923 examples [00:13, 976.08 examples/s]13023 examples [00:13, 981.43 examples/s]13129 examples [00:13, 1003.15 examples/s]13236 examples [00:13, 1022.30 examples/s]13343 examples [00:13, 1035.47 examples/s]13450 examples [00:13, 1044.67 examples/s]13555 examples [00:13, 999.18 examples/s] 13656 examples [00:14, 961.55 examples/s]13761 examples [00:14, 985.53 examples/s]13863 examples [00:14, 995.53 examples/s]13964 examples [00:14, 994.96 examples/s]14064 examples [00:14, 991.26 examples/s]14164 examples [00:14, 988.12 examples/s]14271 examples [00:14, 1008.91 examples/s]14379 examples [00:14, 1027.63 examples/s]14483 examples [00:14, 1011.48 examples/s]14585 examples [00:14, 993.09 examples/s] 14685 examples [00:15, 982.39 examples/s]14789 examples [00:15, 997.27 examples/s]14889 examples [00:15, 991.91 examples/s]14990 examples [00:15, 996.68 examples/s]15095 examples [00:15, 1011.97 examples/s]15197 examples [00:15, 966.14 examples/s] 15308 examples [00:15, 1002.73 examples/s]15416 examples [00:15, 1023.03 examples/s]15521 examples [00:15, 1029.14 examples/s]15626 examples [00:16, 1032.84 examples/s]15733 examples [00:16, 1043.56 examples/s]15841 examples [00:16, 1053.76 examples/s]15947 examples [00:16, 1043.21 examples/s]16052 examples [00:16, 1004.13 examples/s]16153 examples [00:16, 965.87 examples/s] 16255 examples [00:16, 980.71 examples/s]16357 examples [00:16, 991.47 examples/s]16457 examples [00:16, 941.34 examples/s]16552 examples [00:16, 905.46 examples/s]16649 examples [00:17, 922.11 examples/s]16754 examples [00:17, 944.43 examples/s]16860 examples [00:17, 974.20 examples/s]16969 examples [00:17, 1004.41 examples/s]17076 examples [00:17, 1021.03 examples/s]17182 examples [00:17, 1031.22 examples/s]17289 examples [00:17, 1042.33 examples/s]17400 examples [00:17, 1059.25 examples/s]17507 examples [00:17, 1061.63 examples/s]17614 examples [00:17, 1032.92 examples/s]17718 examples [00:18, 1010.25 examples/s]17820 examples [00:18, 1010.32 examples/s]17927 examples [00:18, 1027.21 examples/s]18030 examples [00:18, 1005.26 examples/s]18133 examples [00:18, 1010.05 examples/s]18238 examples [00:18, 1021.49 examples/s]18349 examples [00:18, 1045.14 examples/s]18457 examples [00:18, 1052.71 examples/s]18563 examples [00:18, 1043.92 examples/s]18671 examples [00:19, 1052.51 examples/s]18777 examples [00:19, 1019.18 examples/s]18882 examples [00:19, 1026.39 examples/s]18985 examples [00:19, 1009.98 examples/s]19087 examples [00:19, 984.45 examples/s] 19186 examples [00:19, 951.14 examples/s]19282 examples [00:19, 947.49 examples/s]19378 examples [00:19, 947.52 examples/s]19474 examples [00:19, 949.71 examples/s]19570 examples [00:19, 945.76 examples/s]19667 examples [00:20, 951.27 examples/s]19763 examples [00:20, 832.67 examples/s]19858 examples [00:20, 862.55 examples/s]19953 examples [00:20, 886.97 examples/s]20044 examples [00:20, 850.01 examples/s]20148 examples [00:20, 899.15 examples/s]20259 examples [00:20, 952.63 examples/s]20357 examples [00:20, 958.97 examples/s]20455 examples [00:20, 961.66 examples/s]20558 examples [00:21, 980.49 examples/s]20661 examples [00:21, 992.95 examples/s]20761 examples [00:21, 993.38 examples/s]20870 examples [00:21, 1019.42 examples/s]20973 examples [00:21, 1003.77 examples/s]21079 examples [00:21, 1015.49 examples/s]21181 examples [00:21, 976.42 examples/s] 21287 examples [00:21, 999.75 examples/s]21388 examples [00:21, 995.49 examples/s]21488 examples [00:21, 984.80 examples/s]21587 examples [00:22, 980.87 examples/s]21686 examples [00:22, 961.37 examples/s]21790 examples [00:22, 981.49 examples/s]21889 examples [00:22, 970.75 examples/s]21993 examples [00:22, 990.24 examples/s]22095 examples [00:22, 997.43 examples/s]22195 examples [00:22, 945.97 examples/s]22300 examples [00:22, 974.45 examples/s]22399 examples [00:22, 972.87 examples/s]22505 examples [00:22, 997.29 examples/s]22606 examples [00:23, 1000.77 examples/s]22707 examples [00:23, 1002.84 examples/s]22808 examples [00:23, 999.54 examples/s] 22911 examples [00:23, 1006.98 examples/s]23020 examples [00:23, 1029.06 examples/s]23126 examples [00:23, 1036.27 examples/s]23236 examples [00:23, 1053.02 examples/s]23348 examples [00:23, 1070.09 examples/s]23460 examples [00:23, 1083.82 examples/s]23569 examples [00:24, 1070.84 examples/s]23677 examples [00:24, 1068.76 examples/s]23784 examples [00:24, 1062.26 examples/s]23891 examples [00:24, 1052.09 examples/s]23997 examples [00:24, 1042.08 examples/s]24102 examples [00:24, 1020.94 examples/s]24208 examples [00:24, 1031.61 examples/s]24312 examples [00:24, 1013.80 examples/s]24423 examples [00:24, 1038.61 examples/s]24531 examples [00:24, 1050.24 examples/s]24637 examples [00:25, 1052.51 examples/s]24743 examples [00:25, 1039.54 examples/s]24850 examples [00:25, 1045.93 examples/s]24955 examples [00:25, 1007.74 examples/s]25057 examples [00:25, 1004.47 examples/s]25158 examples [00:25, 1003.75 examples/s]25267 examples [00:25, 1027.85 examples/s]25371 examples [00:25, 991.65 examples/s] 25471 examples [00:25, 990.63 examples/s]25576 examples [00:25, 1005.63 examples/s]25677 examples [00:26, 1004.83 examples/s]25782 examples [00:26, 1017.89 examples/s]25886 examples [00:26, 1023.97 examples/s]25995 examples [00:26, 1040.91 examples/s]26106 examples [00:26, 1059.74 examples/s]26213 examples [00:26, 1062.59 examples/s]26322 examples [00:26, 1069.28 examples/s]26430 examples [00:26, 1066.07 examples/s]26537 examples [00:26, 1049.80 examples/s]26646 examples [00:26, 1059.43 examples/s]26753 examples [00:27, 1053.63 examples/s]26859 examples [00:27, 1051.49 examples/s]26967 examples [00:27, 1057.82 examples/s]27074 examples [00:27, 1061.34 examples/s]27181 examples [00:27, 1040.97 examples/s]27286 examples [00:27, 1019.19 examples/s]27389 examples [00:27, 1012.79 examples/s]27497 examples [00:27, 1031.82 examples/s]27602 examples [00:27, 1036.35 examples/s]27707 examples [00:28, 1039.77 examples/s]27814 examples [00:28, 1044.86 examples/s]27919 examples [00:28, 989.82 examples/s] 28024 examples [00:28, 1006.00 examples/s]28126 examples [00:28, 990.30 examples/s] 28234 examples [00:28, 1013.92 examples/s]28336 examples [00:28, 1007.53 examples/s]28438 examples [00:28, 993.52 examples/s] 28547 examples [00:28, 1018.78 examples/s]28650 examples [00:28, 1015.86 examples/s]28752 examples [00:29, 999.20 examples/s] 28853 examples [00:29, 995.76 examples/s]28957 examples [00:29, 1007.33 examples/s]29058 examples [00:29, 961.22 examples/s] 29164 examples [00:29, 986.67 examples/s]29264 examples [00:29, 980.99 examples/s]29370 examples [00:29, 1002.89 examples/s]29476 examples [00:29, 1019.29 examples/s]29582 examples [00:29, 1030.05 examples/s]29686 examples [00:29, 990.28 examples/s] 29786 examples [00:30, 978.44 examples/s]29885 examples [00:30, 979.29 examples/s]29994 examples [00:30, 1007.66 examples/s]30096 examples [00:30, 970.88 examples/s] 30205 examples [00:30, 1002.09 examples/s]30306 examples [00:30, 981.60 examples/s] 30406 examples [00:30, 984.59 examples/s]30505 examples [00:30, 975.76 examples/s]30603 examples [00:30, 975.34 examples/s]30707 examples [00:31, 993.20 examples/s]30810 examples [00:31, 1002.31 examples/s]30911 examples [00:31, 991.05 examples/s] 31016 examples [00:31, 1006.89 examples/s]31117 examples [00:31, 995.23 examples/s] 31222 examples [00:31, 1008.52 examples/s]31323 examples [00:31, 980.00 examples/s] 31422 examples [00:31, 970.67 examples/s]31520 examples [00:31, 962.30 examples/s]31628 examples [00:31, 993.10 examples/s]31728 examples [00:32, 983.41 examples/s]31827 examples [00:32, 932.86 examples/s]31931 examples [00:32, 961.99 examples/s]32037 examples [00:32, 982.27 examples/s]32136 examples [00:32, 972.93 examples/s]32244 examples [00:32, 1001.22 examples/s]32346 examples [00:32, 1004.93 examples/s]32448 examples [00:32, 1008.47 examples/s]32550 examples [00:32, 976.87 examples/s] 32659 examples [00:33, 1007.11 examples/s]32761 examples [00:33, 985.65 examples/s] 32865 examples [00:33, 999.03 examples/s]32975 examples [00:33, 1027.21 examples/s]33084 examples [00:33, 1044.82 examples/s]33189 examples [00:33, 1039.99 examples/s]33296 examples [00:33, 1047.15 examples/s]33401 examples [00:33, 1035.49 examples/s]33508 examples [00:33, 1045.58 examples/s]33613 examples [00:33, 1032.37 examples/s]33717 examples [00:34, 945.11 examples/s] 33814 examples [00:34, 947.35 examples/s]33910 examples [00:34, 926.72 examples/s]34005 examples [00:34, 932.23 examples/s]34108 examples [00:34, 958.74 examples/s]34208 examples [00:34, 968.42 examples/s]34311 examples [00:34, 985.32 examples/s]34413 examples [00:34, 993.63 examples/s]34517 examples [00:34, 1005.23 examples/s]34624 examples [00:34, 1023.21 examples/s]34729 examples [00:35, 1029.82 examples/s]34835 examples [00:35, 1037.87 examples/s]34940 examples [00:35, 1038.95 examples/s]35048 examples [00:35, 1048.65 examples/s]35153 examples [00:35, 1040.20 examples/s]35258 examples [00:35, 1028.09 examples/s]35361 examples [00:35, 1017.93 examples/s]35463 examples [00:35, 1014.56 examples/s]35570 examples [00:35, 1027.98 examples/s]35677 examples [00:35, 1037.68 examples/s]35781 examples [00:36, 988.70 examples/s] 35889 examples [00:36, 1013.19 examples/s]35991 examples [00:36, 1012.84 examples/s]36093 examples [00:36, 990.23 examples/s] 36198 examples [00:36, 1005.07 examples/s]36305 examples [00:36, 1021.88 examples/s]36408 examples [00:36, 979.35 examples/s] 36516 examples [00:36, 1005.93 examples/s]36618 examples [00:36, 999.19 examples/s] 36721 examples [00:37, 1006.55 examples/s]36823 examples [00:37, 1008.53 examples/s]36925 examples [00:37, 980.20 examples/s] 37029 examples [00:37, 995.17 examples/s]37129 examples [00:37, 992.30 examples/s]37235 examples [00:37, 1011.20 examples/s]37338 examples [00:37, 1015.92 examples/s]37443 examples [00:37, 1025.60 examples/s]37549 examples [00:37, 1034.71 examples/s]37661 examples [00:37, 1057.79 examples/s]37767 examples [00:38, 1045.69 examples/s]37874 examples [00:38, 1052.82 examples/s]37982 examples [00:38, 1058.48 examples/s]38088 examples [00:38, 1054.58 examples/s]38194 examples [00:38, 1048.88 examples/s]38299 examples [00:38, 1044.43 examples/s]38407 examples [00:38, 1053.82 examples/s]38516 examples [00:38, 1063.70 examples/s]38623 examples [00:38, 1006.93 examples/s]38725 examples [00:38, 990.27 examples/s] 38825 examples [00:39, 958.17 examples/s]38931 examples [00:39, 986.43 examples/s]39036 examples [00:39, 1004.65 examples/s]39140 examples [00:39, 1014.67 examples/s]39245 examples [00:39, 1024.23 examples/s]39355 examples [00:39, 1045.22 examples/s]39460 examples [00:39, 1027.81 examples/s]39564 examples [00:39, 1010.56 examples/s]39670 examples [00:39, 1024.87 examples/s]39773 examples [00:40, 1025.04 examples/s]39878 examples [00:40, 1032.04 examples/s]39982 examples [00:40, 1031.01 examples/s]40086 examples [00:40, 950.51 examples/s] 40191 examples [00:40, 975.93 examples/s]40297 examples [00:40, 999.28 examples/s]40404 examples [00:40, 1018.43 examples/s]40507 examples [00:40, 970.01 examples/s] 40614 examples [00:40, 995.43 examples/s]40715 examples [00:40, 997.07 examples/s]40816 examples [00:41, 973.24 examples/s]40926 examples [00:41, 1005.84 examples/s]41035 examples [00:41, 1028.71 examples/s]41143 examples [00:41, 1042.54 examples/s]41252 examples [00:41, 1056.28 examples/s]41361 examples [00:41, 1065.81 examples/s]41469 examples [00:41, 1067.32 examples/s]41579 examples [00:41, 1076.29 examples/s]41687 examples [00:41, 1054.91 examples/s]41797 examples [00:41, 1065.45 examples/s]41904 examples [00:42, 1028.99 examples/s]42012 examples [00:42, 1041.82 examples/s]42117 examples [00:42, 1040.14 examples/s]42224 examples [00:42, 1048.71 examples/s]42334 examples [00:42, 1060.83 examples/s]42441 examples [00:42, 1051.12 examples/s]42547 examples [00:42, 1053.40 examples/s]42653 examples [00:42, 1033.85 examples/s]42757 examples [00:42, 1025.49 examples/s]42860 examples [00:43, 1001.57 examples/s]42963 examples [00:43, 1008.85 examples/s]43065 examples [00:43, 994.28 examples/s] 43166 examples [00:43, 996.73 examples/s]43272 examples [00:43, 1012.91 examples/s]43382 examples [00:43, 1035.22 examples/s]43490 examples [00:43, 1047.49 examples/s]43599 examples [00:43, 1058.11 examples/s]43705 examples [00:43, 1051.99 examples/s]43811 examples [00:43, 991.64 examples/s] 43911 examples [00:44, 950.95 examples/s]44017 examples [00:44, 980.81 examples/s]44123 examples [00:44, 1001.88 examples/s]44224 examples [00:44, 966.94 examples/s] 44329 examples [00:44, 989.12 examples/s]44432 examples [00:44, 999.03 examples/s]44533 examples [00:44, 979.17 examples/s]44632 examples [00:44, 957.13 examples/s]44732 examples [00:44, 968.21 examples/s]44832 examples [00:45, 976.96 examples/s]44930 examples [00:45, 950.20 examples/s]45026 examples [00:45, 918.88 examples/s]45124 examples [00:45, 932.74 examples/s]45221 examples [00:45, 940.85 examples/s]45326 examples [00:45, 969.97 examples/s]45424 examples [00:45, 924.86 examples/s]45518 examples [00:45, 919.53 examples/s]45619 examples [00:45, 942.75 examples/s]45715 examples [00:45, 946.31 examples/s]45810 examples [00:46, 880.48 examples/s]45900 examples [00:46, 871.80 examples/s]45989 examples [00:46, 807.34 examples/s]46091 examples [00:46, 859.62 examples/s]46193 examples [00:46, 900.07 examples/s]46299 examples [00:46, 940.73 examples/s]46397 examples [00:46, 952.08 examples/s]46498 examples [00:46, 967.65 examples/s]46601 examples [00:46, 983.72 examples/s]46701 examples [00:47, 954.21 examples/s]46802 examples [00:47, 968.71 examples/s]46909 examples [00:47, 996.33 examples/s]47017 examples [00:47, 1019.40 examples/s]47120 examples [00:47, 1007.06 examples/s]47222 examples [00:47, 1002.81 examples/s]47328 examples [00:47, 1017.43 examples/s]47431 examples [00:47, 1018.61 examples/s]47538 examples [00:47, 1033.43 examples/s]47644 examples [00:47, 1040.70 examples/s]47750 examples [00:48, 1046.09 examples/s]47855 examples [00:48, 1041.60 examples/s]47960 examples [00:48, 1034.30 examples/s]48067 examples [00:48, 1043.37 examples/s]48175 examples [00:48, 1053.51 examples/s]48281 examples [00:48, 1004.51 examples/s]48386 examples [00:48, 1014.87 examples/s]48488 examples [00:48, 1013.33 examples/s]48593 examples [00:48, 1023.14 examples/s]48696 examples [00:48, 1003.38 examples/s]48797 examples [00:49, 994.95 examples/s] 48897 examples [00:49, 957.28 examples/s]48997 examples [00:49, 968.97 examples/s]49095 examples [00:49, 958.70 examples/s]49203 examples [00:49, 991.59 examples/s]49309 examples [00:49, 1010.06 examples/s]49411 examples [00:49, 983.18 examples/s] 49513 examples [00:49, 993.77 examples/s]49615 examples [00:49, 1000.82 examples/s]49716 examples [00:50, 984.92 examples/s] 49815 examples [00:50, 944.41 examples/s]49910 examples [00:50, 942.22 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 14%|█▍        | 7052/50000 [00:00<00:00, 70516.87 examples/s] 41%|████      | 20389/50000 [00:00<00:00, 82127.72 examples/s] 67%|██████▋   | 33708/50000 [00:00<00:00, 92800.55 examples/s] 94%|█████████▍| 46956/50000 [00:00<00:00, 101962.16 examples/s]                                                                0 examples [00:00, ? examples/s]67 examples [00:00, 669.24 examples/s]174 examples [00:00, 753.17 examples/s]278 examples [00:00, 819.57 examples/s]366 examples [00:00, 836.04 examples/s]466 examples [00:00, 879.12 examples/s]572 examples [00:00, 924.32 examples/s]679 examples [00:00, 961.99 examples/s]784 examples [00:00, 986.04 examples/s]894 examples [00:00, 1015.22 examples/s]1000 examples [00:01, 1026.66 examples/s]1106 examples [00:01, 1035.28 examples/s]1211 examples [00:01, 1037.57 examples/s]1315 examples [00:01, 1028.92 examples/s]1421 examples [00:01, 1036.35 examples/s]1531 examples [00:01, 1054.22 examples/s]1638 examples [00:01, 1058.24 examples/s]1749 examples [00:01, 1071.52 examples/s]1857 examples [00:01, 1049.95 examples/s]1963 examples [00:01, 1022.58 examples/s]2071 examples [00:02, 1035.78 examples/s]2175 examples [00:02, 1031.08 examples/s]2284 examples [00:02, 1046.70 examples/s]2392 examples [00:02, 1055.83 examples/s]2498 examples [00:02, 1034.31 examples/s]2602 examples [00:02, 1035.28 examples/s]2706 examples [00:02, 1034.23 examples/s]2810 examples [00:02, 1028.90 examples/s]2913 examples [00:02, 1009.68 examples/s]3017 examples [00:02, 1016.39 examples/s]3126 examples [00:03, 1036.13 examples/s]3237 examples [00:03, 1056.91 examples/s]3343 examples [00:03, 1057.03 examples/s]3449 examples [00:03, 1022.47 examples/s]3552 examples [00:03, 1007.53 examples/s]3663 examples [00:03, 1034.21 examples/s]3775 examples [00:03, 1058.26 examples/s]3882 examples [00:03, 1060.57 examples/s]3989 examples [00:03, 1061.68 examples/s]4099 examples [00:03, 1072.37 examples/s]4207 examples [00:04, 1059.19 examples/s]4314 examples [00:04, 1050.08 examples/s]4420 examples [00:04, 1032.61 examples/s]4524 examples [00:04, 997.08 examples/s] 4625 examples [00:04, 970.43 examples/s]4733 examples [00:04, 1000.71 examples/s]4838 examples [00:04, 1013.35 examples/s]4942 examples [00:04, 1019.25 examples/s]5045 examples [00:04, 1004.80 examples/s]5146 examples [00:05, 1004.63 examples/s]5248 examples [00:05, 1007.30 examples/s]5357 examples [00:05, 1029.76 examples/s]5463 examples [00:05, 1036.46 examples/s]5567 examples [00:05, 1020.95 examples/s]5673 examples [00:05, 1030.59 examples/s]5777 examples [00:05, 1015.17 examples/s]5882 examples [00:05, 1024.72 examples/s]5993 examples [00:05, 1046.68 examples/s]6104 examples [00:05, 1064.01 examples/s]6211 examples [00:06, 1062.50 examples/s]6318 examples [00:06, 1064.02 examples/s]6425 examples [00:06, 1057.45 examples/s]6531 examples [00:06, 998.95 examples/s] 6635 examples [00:06, 1010.16 examples/s]6737 examples [00:06, 1010.49 examples/s]6839 examples [00:06, 1004.20 examples/s]6949 examples [00:06, 1029.85 examples/s]7060 examples [00:06, 1050.77 examples/s]7170 examples [00:06, 1061.89 examples/s]7278 examples [00:07, 1066.59 examples/s]7387 examples [00:07, 1072.13 examples/s]7495 examples [00:07, 1014.62 examples/s]7598 examples [00:07, 966.71 examples/s] 7696 examples [00:07, 969.52 examples/s]7805 examples [00:07, 1000.39 examples/s]7915 examples [00:07, 1027.17 examples/s]8020 examples [00:07, 1033.40 examples/s]8129 examples [00:07, 1049.62 examples/s]8238 examples [00:08, 1058.94 examples/s]8350 examples [00:08, 1074.25 examples/s]8460 examples [00:08, 1081.17 examples/s]8570 examples [00:08, 1085.26 examples/s]8679 examples [00:08, 1072.36 examples/s]8787 examples [00:08, 1072.86 examples/s]8895 examples [00:08, 1067.97 examples/s]9006 examples [00:08, 1078.52 examples/s]9114 examples [00:08, 1047.95 examples/s]9220 examples [00:08, 1041.57 examples/s]9329 examples [00:09, 1053.42 examples/s]9439 examples [00:09, 1066.89 examples/s]9546 examples [00:09, 1061.88 examples/s]9654 examples [00:09, 1064.65 examples/s]9761 examples [00:09, 1053.23 examples/s]9868 examples [00:09, 1057.32 examples/s]9974 examples [00:09, 1045.53 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteI8N4YZ/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteI8N4YZ/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fe538eee8c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fe538eee8c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fe538eee8c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe4c23b7860>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe4c2345c18>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fe538eee8c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fe538eee8c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fe538eee8c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe4c345ecf8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe536523d68>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7fe4c002e378> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7fe4c002e378> 

  function with postional parmater data_info <function split_train_valid at 0x7fe4c002e378> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7fe4c002e488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7fe4c002e488> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7fe4c002e488> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.0
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.0/en_core_web_sm-2.3.0.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.0) (2.3.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.6.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (4.46.1)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (45.2.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.2)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.18.5)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.1.3)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.4.1)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.0.3)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (7.4.1)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.24.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.6.1)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2020.4.5.2)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.4)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.0-py3-none-any.whl size=12048606 sha256=c0cd63f96c7701b60525782626ff801ec59acb190ca0af1cd03ce76d5bed20cd
  Stored in directory: /tmp/pip-ephem-wheel-cache-hd5fovi_/wheels/4a/db/07/94eee4f3a60150464a04160bd0dfe9c8752ab981fe92f16aea
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<18:29:44, 12.9kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<13:10:55, 18.2kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<9:16:59, 25.8kB/s]  .vector_cache/glove.6B.zip:   0%|          | 737k/862M [00:01<6:30:29, 36.8kB/s].vector_cache/glove.6B.zip:   0%|          | 1.81M/862M [00:01<4:33:25, 52.4kB/s].vector_cache/glove.6B.zip:   1%|          | 4.69M/862M [00:01<3:10:54, 74.9kB/s].vector_cache/glove.6B.zip:   1%|          | 9.18M/862M [00:01<2:13:01, 107kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 13.5M/862M [00:01<1:32:45, 152kB/s].vector_cache/glove.6B.zip:   2%|▏         | 18.0M/862M [00:01<1:04:40, 218kB/s].vector_cache/glove.6B.zip:   3%|▎         | 22.1M/862M [00:01<45:09, 310kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 26.6M/862M [00:01<31:32, 442kB/s].vector_cache/glove.6B.zip:   4%|▎         | 30.6M/862M [00:01<22:04, 628kB/s].vector_cache/glove.6B.zip:   4%|▍         | 34.6M/862M [00:01<15:29, 891kB/s].vector_cache/glove.6B.zip:   5%|▍         | 38.9M/862M [00:02<10:52, 1.26MB/s].vector_cache/glove.6B.zip:   5%|▌         | 43.2M/862M [00:02<07:40, 1.78MB/s].vector_cache/glove.6B.zip:   5%|▌         | 47.0M/862M [00:02<05:27, 2.49MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.3M/862M [00:02<03:53, 3.47MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.1M/862M [00:02<04:43, 2.85MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.2M/862M [00:02<03:23, 3.96MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.3M/862M [00:04<2:14:57, 99.5kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.6M/862M [00:04<1:36:20, 139kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 57.2M/862M [00:04<1:08:00, 197kB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.6M/862M [00:05<47:48, 280kB/s]  .vector_cache/glove.6B.zip:   7%|▋         | 60.4M/862M [00:06<37:06, 360kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.7M/862M [00:06<27:42, 482kB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.7M/862M [00:06<19:48, 673kB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.7M/862M [00:07<14:02, 948kB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.6M/862M [00:08<17:09, 774kB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.9M/862M [00:08<13:33, 980kB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.3M/862M [00:08<09:51, 1.35MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.7M/862M [00:10<09:42, 1.36MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.1M/862M [00:10<08:07, 1.63MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.7M/862M [00:10<06:00, 2.19MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.9M/862M [00:12<07:20, 1.79MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.0M/862M [00:12<07:54, 1.66MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.8M/862M [00:12<06:12, 2.12MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.5M/862M [00:13<04:55, 2.66MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.0M/862M [00:14<05:59, 2.19MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.1M/862M [00:14<09:09, 1.43MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.5M/862M [00:14<07:36, 1.72MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.2M/862M [00:15<05:34, 2.34MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.7M/862M [00:15<04:09, 3.13MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.1M/862M [00:16<17:46, 733kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 81.2M/862M [00:16<16:43, 778kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.7M/862M [00:16<12:39, 1.03MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.5M/862M [00:16<09:20, 1.39MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.8M/862M [00:17<06:41, 1.94MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.3M/862M [00:18<18:36, 696kB/s] .vector_cache/glove.6B.zip:  10%|▉         | 85.4M/862M [00:18<17:22, 745kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.9M/862M [00:18<13:12, 980kB/s].vector_cache/glove.6B.zip:  10%|█         | 87.0M/862M [00:18<09:35, 1.35MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.7M/862M [00:19<06:55, 1.86MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.4M/862M [00:20<13:06, 983kB/s] .vector_cache/glove.6B.zip:  10%|█         | 89.5M/862M [00:20<13:23, 961kB/s].vector_cache/glove.6B.zip:  10%|█         | 90.0M/862M [00:20<10:16, 1.25MB/s].vector_cache/glove.6B.zip:  11%|█         | 90.8M/862M [00:20<07:38, 1.68MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.1M/862M [00:21<05:30, 2.33MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.6M/862M [00:22<16:29, 777kB/s] .vector_cache/glove.6B.zip:  11%|█         | 93.7M/862M [00:22<15:28, 827kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.2M/862M [00:22<11:49, 1.08MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.4M/862M [00:22<08:34, 1.49MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.5M/862M [00:23<06:10, 2.06MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.7M/862M [00:24<28:33, 446kB/s] .vector_cache/glove.6B.zip:  11%|█▏        | 97.9M/862M [00:24<23:55, 532kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.4M/862M [00:24<17:33, 725kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.3M/862M [00:24<12:42, 1.00MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:24<09:04, 1.40MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<15:13, 832kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<14:10, 894kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<10:40, 1.19MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:26<07:47, 1.62MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:26<05:36, 2.25MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<2:25:09, 86.8kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<1:45:06, 120kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<1:14:18, 169kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 108M/862M [00:28<52:25, 240kB/s]  .vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:28<36:50, 341kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<31:22, 400kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<25:23, 494kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<18:39, 671kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<13:19, 938kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:30<09:29, 1.31MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<2:23:52, 86.6kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<1:44:10, 120kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<1:13:41, 169kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<51:57, 239kB/s]  .vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:32<36:24, 341kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 119M/862M [00:34<53:33, 231kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<58:35, 212kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<41:20, 299kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:34<29:06, 424kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<24:07, 511kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<18:31, 665kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<13:21, 920kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:36<09:31, 1.29MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<14:46, 830kB/s] .vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<13:52, 883kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<10:35, 1.16MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:38<07:38, 1.60MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<08:24, 1.45MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<09:21, 1.30MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<08:11, 1.49MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<06:14, 1.95MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 134M/862M [00:40<04:36, 2.64MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<06:51, 1.77MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<07:56, 1.53MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<06:10, 1.96MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<04:48, 2.51MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:42<03:38, 3.31MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:42<02:52, 4.20MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<20:49, 578kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<18:37, 647kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<14:01, 858kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 140M/862M [00:44<10:20, 1.16MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 142M/862M [00:44<07:25, 1.62MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<09:55, 1.21MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<10:07, 1.18MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<07:44, 1.54MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<05:48, 2.06MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:46<04:14, 2.81MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<13:22, 890kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<12:37, 943kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<09:29, 1.25MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<06:58, 1.70MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<07:20, 1.61MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<07:53, 1.50MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<07:05, 1.67MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<05:23, 2.19MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:50<03:59, 2.96MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<07:06, 1.65MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<08:04, 1.46MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<06:17, 1.87MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:52<04:37, 2.54MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<06:19, 1.85MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<08:01, 1.46MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<07:52, 1.48MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<06:18, 1.85MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 162M/862M [00:54<04:49, 2.42MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:54<03:47, 3.08MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:54<03:00, 3.87MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<08:15, 1.41MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<07:50, 1.48MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<05:58, 1.94MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:56<04:18, 2.68MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:58<15:57, 725kB/s] .vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<14:22, 805kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<10:43, 1.08MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<07:53, 1.46MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<07:42, 1.49MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<08:22, 1.37MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<06:37, 1.73MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:00<04:50, 2.36MB/s].vector_cache/glove.6B.zip:  20%|██        | 177M/862M [01:02<07:33, 1.51MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<08:15, 1.38MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<06:25, 1.78MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:02<04:46, 2.39MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<05:51, 1.94MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<07:49, 1.45MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<06:22, 1.78MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:04<04:41, 2.41MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:04<03:26, 3.27MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<5:12:55, 36.1kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<3:41:58, 50.8kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<2:35:58, 72.3kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:06<1:49:02, 103kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<1:20:00, 140kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<59:05, 190kB/s]  .vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<42:02, 267kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:08<29:39, 377kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:08<20:51, 534kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<45:38, 244kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<34:32, 323kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:10<25:38, 435kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:10<18:14, 610kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:10<12:56, 857kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<13:39, 811kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<12:57, 854kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<10:24, 1.06MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<07:52, 1.40MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:12<05:43, 1.93MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<06:58, 1.58MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<07:54, 1.39MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<06:12, 1.77MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:14<04:36, 2.38MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<05:37, 1.94MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<06:48, 1.61MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<05:31, 1.98MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:16<04:02, 2.69MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<07:19, 1.49MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<07:50, 1.39MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:18<06:02, 1.80MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:18<04:37, 2.35MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:18<03:25, 3.16MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<07:47, 1.39MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<11:19, 954kB/s] .vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<09:23, 1.15MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:20<06:55, 1.55MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:20<05:02, 2.13MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:20<03:54, 2.74MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:22<36:19, 295kB/s] .vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:22<28:12, 380kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<20:23, 525kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:22<14:30, 737kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:24<13:04, 816kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<14:25, 739kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<11:11, 951kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:24<08:10, 1.30MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:24<05:59, 1.77MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<07:38, 1.38MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<10:38, 996kB/s] .vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<08:36, 1.23MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:26<06:37, 1.60MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:26<04:49, 2.19MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:26<03:42, 2.85MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<09:10, 1.15MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<11:43, 897kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<09:26, 1.11MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:28<06:55, 1.51MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:28<05:04, 2.06MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<10:44, 973kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<12:53, 811kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<10:37, 984kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<07:58, 1.31MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:30<05:51, 1.78MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:30<04:18, 2.41MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:30<03:30, 2.97MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<8:25:22, 20.6kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:32<5:57:12, 29.1kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<4:10:41, 41.4kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<2:55:25, 59.0kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:32<2:02:49, 84.1kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<1:31:26, 113kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:34<1:07:13, 153kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<47:49, 215kB/s]  .vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<33:40, 305kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:34<23:52, 430kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<20:30, 500kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:36<20:52, 491kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:36<16:10, 633kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:36<12:04, 848kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<09:01, 1.13MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:36<06:31, 1.56MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:36<04:57, 2.05MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<09:14, 1.10MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<11:06, 917kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<08:51, 1.15MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<06:31, 1.55MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:38<05:00, 2.03MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:38<03:50, 2.63MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:38<03:07, 3.24MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<14:18, 706kB/s] .vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<14:12, 711kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<10:45, 939kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<07:51, 1.28MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<05:56, 1.70MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 259M/862M [01:40<04:31, 2.23MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:40<03:31, 2.86MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<13:45, 730kB/s] .vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<13:35, 739kB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:42<10:13, 981kB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<07:45, 1.29MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<05:42, 1.75MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<04:27, 2.24MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:42<03:25, 2.92MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<11:44, 849kB/s] .vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<12:05, 824kB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:44<09:23, 1.06MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<06:56, 1.43MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:44<05:11, 1.91MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:44<04:01, 2.46MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<15:49, 626kB/s] .vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<15:58, 620kB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<12:19, 803kB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<09:14, 1.07MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 269M/862M [01:46<06:56, 1.42MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<05:22, 1.84MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:46<04:12, 2.34MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:46<03:25, 2.88MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:46<02:49, 3.48MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<35:27, 277kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<29:13, 336kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<21:37, 454kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<15:34, 630kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<11:18, 866kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:48<08:16, 1.18MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:48<06:12, 1.58MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<17:54, 545kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<17:25, 561kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<13:12, 738kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<09:45, 999kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<07:12, 1.35MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<05:31, 1.76MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<04:12, 2.30MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:50<03:24, 2.84MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [01:51<14:38, 662kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [01:51<14:35, 665kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<11:06, 872kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<08:06, 1.19MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<06:13, 1.55MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<04:54, 1.97MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<03:52, 2.48MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:52<03:09, 3.06MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<09:39, 996kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<11:14, 857kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<08:58, 1.07MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<06:42, 1.43MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<05:04, 1.89MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:54<03:55, 2.44MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:55<07:29, 1.28MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:55<09:10, 1.04MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<07:28, 1.28MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:55<05:39, 1.69MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:56<04:21, 2.19MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:56<03:25, 2.78MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:56<02:47, 3.41MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:56<02:19, 4.07MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<1:45:27, 90.0kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<1:18:03, 122kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<55:37, 170kB/s]  .vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:57<39:14, 241kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<27:49, 340kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<19:50, 476kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<14:14, 662kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<17:18, 544kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<16:02, 587kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<12:05, 779kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:59<09:05, 1.03MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [01:59<06:44, 1.39MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<05:06, 1.83MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<03:58, 2.36MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:00<03:08, 2.98MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<17:03, 548kB/s] .vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<19:44, 474kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:02<16:11, 577kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:02<12:36, 741kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:02<09:35, 973kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:02<07:16, 1.28MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<05:32, 1.68MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<04:15, 2.18MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<03:28, 2.67MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:02<02:56, 3.16MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<07:14, 1.28MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:03<06:03, 1.53MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:03<04:37, 2.00MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:03<03:34, 2.58MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:03<02:47, 3.31MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:04<02:26, 3.77MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<14:59, 615kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<16:12, 569kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<12:49, 718kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<09:22, 980kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:05<07:12, 1.27MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<05:28, 1.68MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<04:20, 2.11MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:06<03:28, 2.64MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:06<02:53, 3.17MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<12:45, 717kB/s] .vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<10:52, 841kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<08:16, 1.10MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:07<06:18, 1.44MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:07<04:51, 1.87MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:07<03:51, 2.36MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<03:07, 2.91MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<02:36, 3.49MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<10:58, 826kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<11:44, 772kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<09:17, 977kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:09<06:59, 1.30MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:09<05:13, 1.73MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<04:02, 2.23MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<03:09, 2.86MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<13:03, 690kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<13:11, 682kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<10:04, 893kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<07:37, 1.18MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:11<05:42, 1.57MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:11<04:17, 2.09MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<03:30, 2.55MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:12<02:48, 3.18MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<1:11:55, 124kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<50:38, 176kB/s]  .vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<37:07, 240kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:13<26:07, 341kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<18:42, 475kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<13:28, 659kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<14:36, 607kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<12:07, 731kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<09:00, 983kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:15<06:38, 1.33MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:15<05:01, 1.75MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:15<03:49, 2.30MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:16<03:04, 2.87MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:17<53:20, 165kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:17<41:16, 213kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<29:50, 295kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<21:16, 413kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<15:10, 577kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:17<10:57, 799kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:18<08:00, 1.09MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<16:59, 514kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<15:48, 552kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<11:59, 727kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<08:45, 993kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:19<06:25, 1.35MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:19<04:46, 1.82MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:19<03:45, 2.30MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<18:27, 469kB/s] .vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<16:48, 515kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<12:43, 680kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<09:14, 934kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<06:50, 1.26MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:21<05:06, 1.69MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:21<03:58, 2.17MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<08:07, 1.06MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<09:14, 929kB/s] .vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<07:23, 1.16MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<05:32, 1.54MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:23<04:11, 2.04MB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:23<03:17, 2.59MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:23<02:38, 3.23MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<10:12, 835kB/s] .vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<10:43, 795kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<08:22, 1.02MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<06:15, 1.36MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:25<04:42, 1.80MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:25<03:39, 2.32MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:25<02:51, 2.96MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<06:42, 1.26MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<08:11, 1.03MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<06:39, 1.27MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<05:01, 1.68MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:27<03:50, 2.19MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:27<02:59, 2.81MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<07:01, 1.19MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<08:36, 975kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<07:10, 1.17MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<05:33, 1.51MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<04:26, 1.88MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<03:27, 2.42MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:29<02:45, 3.03MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:29<02:13, 3.75MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<06:52, 1.21MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<08:14, 1.01MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<06:27, 1.29MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<04:51, 1.71MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:31<03:41, 2.24MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:31<02:49, 2.93MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<08:42, 947kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<09:22, 880kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<07:41, 1.07MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<05:55, 1.39MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<04:31, 1.82MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<03:35, 2.29MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:33<02:50, 2.89MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:33<02:17, 3.58MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:35<07:08, 1.14MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:35<07:54, 1.03MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<06:20, 1.29MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<04:50, 1.69MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<03:52, 2.10MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:35<02:54, 2.80MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:35<02:26, 3.33MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<08:59, 902kB/s] .vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<10:19, 786kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<08:15, 981kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<06:11, 1.31MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<04:44, 1.71MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:37<03:37, 2.22MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:37<02:53, 2.79MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:37<02:21, 3.42MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<10:02, 801kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<10:30, 765kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<08:26, 951kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<06:21, 1.26MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<04:53, 1.64MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<03:44, 2.14MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:39<02:55, 2.74MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:39<02:20, 3.41MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:40<09:14, 863kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<09:44, 818kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<07:36, 1.05MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<05:49, 1.37MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<04:16, 1.85MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:41<03:25, 2.32MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:41<02:47, 2.84MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:42<09:59, 791kB/s] .vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:43<11:26, 690kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:43<09:07, 865kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<06:50, 1.15MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<05:11, 1.52MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<04:08, 1.90MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<03:16, 2.40MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:43<02:40, 2.94MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:43<02:14, 3.49MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<09:33, 820kB/s] .vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:45<08:42, 899kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<06:51, 1.14MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<05:17, 1.48MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<04:03, 1.92MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<03:13, 2.42MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:45<02:35, 3.01MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:45<02:11, 3.56MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<06:20, 1.22MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:47<08:41, 893kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<07:03, 1.10MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<05:17, 1.46MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<04:04, 1.90MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 399M/862M [02:47<03:11, 2.43MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:47<02:35, 2.97MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:47<02:07, 3.63MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<09:27, 814kB/s] .vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<10:22, 742kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:49<08:09, 943kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<06:02, 1.27MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<04:36, 1.67MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:49<03:31, 2.17MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:49<02:49, 2.70MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:49<02:17, 3.34MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<24:58, 305kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:50<21:00, 363kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<15:34, 489kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<11:15, 676kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<08:12, 926kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<06:00, 1.26MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:51<04:36, 1.64MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<08:12, 921kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<09:37, 785kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<07:52, 958kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<06:14, 1.21MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<04:59, 1.51MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<04:07, 1.83MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<03:32, 2.13MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<02:47, 2.69MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:53<02:20, 3.21MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:53<01:58, 3.79MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<05:57, 1.26MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<05:56, 1.26MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<04:33, 1.64MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<03:29, 2.14MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<02:48, 2.66MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<02:18, 3.23MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<02:00, 3.69MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:55<01:48, 4.09MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<15:02, 493kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<13:58, 531kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<10:33, 703kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<07:53, 938kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<05:46, 1.28MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<04:28, 1.65MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:57<03:28, 2.12MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 420M/862M [02:57<02:43, 2.70MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<06:45, 1.09MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<08:20, 881kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<06:41, 1.10MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<05:01, 1.46MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<03:48, 1.92MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<03:02, 2.41MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<02:26, 2.98MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [02:59<02:03, 3.55MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<14:55, 488kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<13:49, 527kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<10:41, 681kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<07:54, 919kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:01<05:54, 1.23MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<04:28, 1.62MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<03:24, 2.12MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:01<02:42, 2.67MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<06:08, 1.17MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<07:33, 955kB/s] .vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<06:06, 1.18MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<04:39, 1.55MB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:03<03:32, 2.03MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<02:48, 2.56MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:03<02:13, 3.23MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:03<01:52, 3.81MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<25:01, 285kB/s] .vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<20:44, 345kB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<15:17, 467kB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<10:59, 648kB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<07:57, 893kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:05<05:49, 1.22MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<08:25, 840kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<08:48, 803kB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<06:53, 1.03MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:06<05:02, 1.40MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<03:55, 1.80MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<03:00, 2.34MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:07<02:21, 2.97MB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:08<06:32, 1.07MB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:08<07:14, 967kB/s] .vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:08<05:43, 1.22MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:08<04:15, 1.64MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:09<03:11, 2.19MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<02:29, 2.78MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<05:16, 1.31MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<06:10, 1.12MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<04:56, 1.40MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<03:45, 1.84MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<02:55, 2.36MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<02:17, 3.00MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:11<01:49, 3.76MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<07:32, 911kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<07:30, 914kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<05:47, 1.18MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<04:20, 1.58MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:12<03:13, 2.12MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<02:27, 2.78MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<05:27, 1.25MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<05:22, 1.26MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<04:48, 1.42MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<03:37, 1.87MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:14<02:42, 2.49MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<04:11, 1.60MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<06:37, 1.02MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<05:30, 1.22MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:16<04:05, 1.64MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 461M/862M [03:16<02:59, 2.24MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:17<02:17, 2.92MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<34:54, 191kB/s] .vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<25:59, 256kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<18:34, 358kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:18<13:08, 504kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:18<09:20, 708kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<09:42, 679kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<09:48, 672kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<07:35, 867kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<05:28, 1.20MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:20<04:02, 1.62MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:21<02:56, 2.21MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<19:40, 332kB/s] .vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<16:27, 396kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<12:05, 539kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<08:36, 755kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:22<06:10, 1.05MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<06:32, 988kB/s] .vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<07:01, 919kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<05:25, 1.19MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<04:00, 1.61MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:24<02:54, 2.20MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<04:54, 1.30MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<05:44, 1.11MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<04:33, 1.40MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:26<03:19, 1.91MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<03:57, 1.60MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<05:34, 1.13MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<04:38, 1.36MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:28<03:25, 1.83MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:28<02:30, 2.49MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:30<07:33, 827kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:30<08:09, 766kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<06:24, 974kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<04:36, 1.35MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:30<03:24, 1.82MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<04:30, 1.37MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<05:56, 1.04MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<04:43, 1.31MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<03:31, 1.75MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:32<02:34, 2.38MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [03:34<04:01, 1.52MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [03:34<05:17, 1.15MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<04:16, 1.43MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<03:09, 1.92MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:34<02:19, 2.60MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<03:59, 1.52MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<05:17, 1.14MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<04:17, 1.40MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<03:10, 1.90MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:36<02:19, 2.57MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<04:14, 1.41MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<05:14, 1.14MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<04:15, 1.40MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<03:14, 1.83MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<02:23, 2.48MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:38<01:47, 3.30MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<16:45, 352kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<13:48, 427kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<10:12, 577kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<07:14, 810kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:40<05:12, 1.12MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<06:17, 928kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<06:13, 938kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<05:55, 985kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:42<04:29, 1.29MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<03:17, 1.77MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:42<02:23, 2.42MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<07:23, 779kB/s] .vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<07:17, 791kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<05:34, 1.03MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<04:00, 1.43MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:44<02:55, 1.95MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:46<05:02, 1.13MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<05:32, 1.03MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<04:17, 1.32MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<03:15, 1.74MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:46<02:22, 2.37MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<03:32, 1.59MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<04:33, 1.23MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<03:40, 1.53MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:48<02:41, 2.08MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:48<01:59, 2.80MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<04:38, 1.20MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<05:05, 1.09MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<04:01, 1.38MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<02:56, 1.88MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:50<02:10, 2.54MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<04:34, 1.20MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<05:03, 1.08MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<04:00, 1.37MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<02:53, 1.89MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:52<02:09, 2.52MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<05:24, 1.00MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<05:36, 967kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<04:18, 1.26MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:54<03:05, 1.74MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:54<02:17, 2.34MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<05:05, 1.05MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<05:19, 1.00MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<04:07, 1.29MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<03:04, 1.73MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:56<02:15, 2.35MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:58<03:31, 1.50MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:58<04:11, 1.26MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<03:23, 1.56MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:58<02:29, 2.11MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 550M/862M [03:59<03:03, 1.70MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<03:51, 1.35MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<03:04, 1.69MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<02:21, 2.19MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:00<01:44, 2.95MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:01<03:02, 1.69MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<03:56, 1.30MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<03:08, 1.64MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:02<02:17, 2.23MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:02<01:42, 2.97MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:03<04:06, 1.24MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<04:42, 1.08MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<03:42, 1.36MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<02:42, 1.86MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:04<01:59, 2.53MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<04:41, 1.07MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<04:57, 1.01MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<03:51, 1.29MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<02:47, 1.78MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:06<02:03, 2.41MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<04:27, 1.11MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:08<04:46, 1.03MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<03:45, 1.31MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<02:43, 1.79MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<03:09, 1.54MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:10<03:48, 1.28MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<03:04, 1.58MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:10<02:14, 2.16MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<02:46, 1.72MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:11<03:33, 1.35MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<03:20, 1.43MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<02:36, 1.84MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<01:54, 2.48MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:13<02:30, 1.88MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:13<03:19, 1.42MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<02:44, 1.73MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<01:59, 2.36MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:14<01:30, 3.11MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<04:24, 1.05MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<04:38, 1.00MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<03:34, 1.30MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<02:39, 1.75MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:16<01:55, 2.40MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<03:45, 1.22MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<04:10, 1.10MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<03:15, 1.40MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<02:22, 1.92MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 591M/862M [04:18<01:44, 2.61MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<05:11, 871kB/s] .vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<05:06, 884kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<03:56, 1.15MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<02:50, 1.57MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:20<02:03, 2.17MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<2:01:31, 36.6kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<1:26:32, 51.4kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<1:00:47, 73.0kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<42:31, 104kB/s]   .vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:22<29:36, 148kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:23<24:16, 180kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:23<18:21, 239kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<13:08, 332kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<09:14, 469kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 604M/862M [04:25<07:33, 571kB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<06:38, 649kB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<05:23, 798kB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<04:00, 1.07MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<02:55, 1.46MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:26<02:07, 2.01MB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:27<03:41, 1.15MB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:27<04:22, 970kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:27<03:28, 1.22MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<02:31, 1.67MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:28<01:48, 2.30MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<13:51, 301kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<10:54, 382kB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<07:52, 528kB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<05:37, 737kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:30<03:57, 1.04MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<31:30, 130kB/s] .vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<23:07, 177kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<16:48, 244kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:31<11:59, 341kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<08:25, 483kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<06:55, 583kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<05:57, 677kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:33<04:26, 906kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<03:10, 1.26MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:34<02:16, 1.75MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<16:13, 244kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<12:26, 318kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:35<09:12, 430kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:36<06:54, 571kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:36<05:15, 751kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<04:16, 922kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<03:23, 1.16MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<02:58, 1.32MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<02:28, 1.59MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<02:21, 1.67MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<02:00, 1.95MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<01:59, 1.96MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<01:45, 2.23MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<01:48, 2.16MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<01:37, 2.40MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<01:43, 2.27MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<02:37, 1.48MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<02:27, 1.58MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<02:10, 1.78MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<02:26, 1.59MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:38<02:18, 1.68MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<02:12, 1.75MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<02:09, 1.79MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<02:03, 1.88MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<02:01, 1.91MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<01:57, 1.98MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<01:57, 1.98MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<01:52, 2.05MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<01:53, 2.04MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:39<01:50, 2.09MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:39<01:50, 2.09MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<01:46, 2.16MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<01:48, 2.13MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<01:46, 2.16MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<01:46, 2.16MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<01:44, 2.20MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<01:45, 2.18MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<01:43, 2.20MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<01:44, 2.20MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 634M/862M [04:40<01:42, 2.24MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<01:45, 2.17MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<01:42, 2.23MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<01:43, 2.21MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<01:40, 2.27MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<01:41, 2.24MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<01:39, 2.29MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<01:40, 2.27MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<01:37, 2.34MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:40<01:39, 2.29MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<01:36, 2.34MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<01:39, 2.27MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<01:34, 2.38MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<03:31, 1.07MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<02:56, 1.28MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<02:25, 1.55MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<02:17, 1.63MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:41<01:57, 1.91MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:41<01:56, 1.92MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<01:43, 2.17MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<01:48, 2.07MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<01:43, 2.17MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<01:36, 2.31MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<01:35, 2.35MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<01:35, 2.34MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<01:29, 2.47MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<01:31, 2.44MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<01:27, 2.53MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<01:28, 2.51MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<03:42, 998kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<03:10, 1.16MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<02:43, 1.35MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<02:18, 1.60MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<02:12, 1.66MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:43<01:59, 1.84MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:44<01:50, 1.99MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:44<01:44, 2.10MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<01:36, 2.29MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<01:30, 2.41MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<01:26, 2.54MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<01:24, 2.59MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<01:24, 2.58MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<01:19, 2.73MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<01:20, 2.72MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:44<01:16, 2.84MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<06:20, 572kB/s] .vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<04:59, 725kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<03:51, 939kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<03:05, 1.17MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<02:34, 1.40MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:46<02:12, 1.63MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:46<01:52, 1.91MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<01:41, 2.13MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<01:31, 2.35MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<01:24, 2.55MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<01:20, 2.66MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<01:16, 2.78MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<01:14, 2.87MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:46<01:13, 2.90MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<03:38, 975kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<03:02, 1.17MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<02:28, 1.44MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<02:02, 1.73MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<01:43, 2.05MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:48<01:30, 2.34MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:48<01:20, 2.62MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<01:13, 2.87MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<01:08, 3.08MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:48<01:03, 3.30MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<04:21, 800kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<04:01, 864kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<03:16, 1.06MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<02:40, 1.30MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<02:18, 1.51MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<02:01, 1.71MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:50<01:43, 2.02MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<01:24, 2.44MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<01:13, 2.81MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<01:03, 3.26MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<00:58, 3.55MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:50<00:52, 3.88MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<03:55, 871kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<03:15, 1.05MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<02:28, 1.37MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:51<01:54, 1.77MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:51<01:30, 2.24MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:51<01:13, 2.74MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<01:02, 3.25MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<00:52, 3.82MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<06:13, 538kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<06:00, 558kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<04:32, 736kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<03:23, 985kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<02:29, 1.33MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<01:55, 1.73MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:53<01:26, 2.29MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:54<01:10, 2.79MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<03:56, 832kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<04:07, 794kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<03:13, 1.01MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<02:22, 1.37MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<01:46, 1.82MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [04:55<01:21, 2.39MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:55<01:03, 3.06MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<05:19, 603kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<04:51, 661kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<03:38, 881kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<02:39, 1.20MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:57<02:00, 1.58MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:57<01:30, 2.09MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:57<01:09, 2.72MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<07:51, 400kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<06:25, 489kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<04:42, 665kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<03:25, 913kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:59<02:28, 1.26MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [04:59<01:47, 1.72MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<03:41, 832kB/s] .vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<04:06, 747kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<03:16, 935kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:01<02:22, 1.28MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:01<01:43, 1.76MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<02:12, 1.36MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<02:53, 1.04MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<02:21, 1.27MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:03<01:43, 1.72MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:03<01:15, 2.34MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<04:28, 657kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<04:21, 673kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<03:18, 886kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<02:27, 1.19MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:05<01:45, 1.64MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:05<01:16, 2.24MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<08:48, 326kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<07:10, 400kB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<05:16, 542kB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<03:43, 761kB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:07<02:38, 1.06MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:09<03:40, 762kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:09<03:31, 792kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<02:41, 1.04MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:09<01:55, 1.43MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:11<02:01, 1.35MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<02:15, 1.20MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<01:46, 1.53MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<01:17, 2.08MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:11<00:56, 2.82MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 703M/862M [05:13<03:02, 874kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<03:26, 772kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<02:44, 970kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<01:58, 1.33MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:13<01:25, 1.83MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:13<01:07, 2.30MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<07:50, 330kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<06:23, 405kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<04:40, 552kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<03:20, 769kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:15<02:23, 1.06MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:15<01:44, 1.45MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<03:05, 816kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<02:59, 842kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<02:17, 1.10MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<01:40, 1.49MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:17<01:13, 2.03MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:17<00:56, 2.61MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<09:23, 261kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<07:19, 334kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<05:17, 462kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<03:45, 645kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:19<02:40, 900kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:19<01:56, 1.23MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<03:07, 763kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<03:34, 666kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<02:48, 846kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:21<02:08, 1.11MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<01:36, 1.47MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:21<01:09, 2.00MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:21<00:53, 2.61MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<21:49, 106kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:23<16:18, 142kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<11:38, 198kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<08:10, 280kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:23<05:44, 395kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:23<04:07, 549kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:23<02:55, 766kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<28:06, 79.9kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<20:30, 109kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<14:30, 154kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<10:09, 218kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:25<07:07, 309kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:25<05:04, 431kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<04:33, 477kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<04:02, 539kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<03:02, 715kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:27<02:13, 972kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:27<01:38, 1.31MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<01:12, 1.77MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:27<00:55, 2.30MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<02:01, 1.04MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:29<02:09, 979kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:29<01:41, 1.24MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<01:15, 1.66MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<01:00, 2.04MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:29<00:45, 2.71MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:29<00:38, 3.17MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<02:25, 841kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<02:42, 754kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:31<02:09, 942kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<01:40, 1.20MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<01:14, 1.62MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<00:59, 2.02MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:31<00:48, 2.49MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:31<00:42, 2.80MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:31<00:38, 3.04MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<02:13, 883kB/s] .vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<02:12, 892kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<01:42, 1.15MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<01:21, 1.44MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:33<01:02, 1.85MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:33<00:53, 2.17MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:33<00:47, 2.46MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:33<00:44, 2.59MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:33<00:44, 2.56MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:33<00:39, 2.87MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<00:37, 3.02MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<09:06, 208kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<06:46, 280kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<04:52, 387kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<03:32, 530kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<02:36, 716kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<01:57, 950kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:35<01:29, 1.24MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:35<01:11, 1.55MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:35<00:57, 1.92MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:35<00:49, 2.24MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<02:56, 623kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<02:45, 662kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<02:07, 858kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<01:35, 1.14MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<01:17, 1.39MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<01:02, 1.74MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<00:51, 2.11MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:37<00:43, 2.48MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:37<00:38, 2.81MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:37<00:34, 3.13MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:37<00:31, 3.38MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<01:58, 890kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<01:43, 1.02MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<01:20, 1.31MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<01:03, 1.66MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<00:52, 2.00MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<00:42, 2.43MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<00:38, 2.70MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<00:33, 3.11MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:39<00:31, 3.26MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:39<00:28, 3.57MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<01:41, 1.00MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<01:49, 928kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<01:27, 1.16MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<01:07, 1.48MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<00:53, 1.88MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<00:46, 2.14MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:41<00:39, 2.51MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:41<00:35, 2.78MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:41<00:35, 2.78MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:41<00:32, 3.05MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:41<00:29, 3.26MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<02:17, 708kB/s] .vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<01:54, 848kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 765M/862M [05:42<01:27, 1.11MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<01:09, 1.39MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<00:55, 1.75MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<00:45, 2.10MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<00:38, 2.50MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<00:33, 2.81MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:43<00:29, 3.16MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:43<00:27, 3.37MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:43<00:26, 3.58MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<02:47, 557kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<02:34, 604kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<02:01, 768kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<01:31, 1.01MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<01:11, 1.29MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<00:56, 1.64MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<00:45, 2.00MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<00:38, 2.36MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 772M/862M [05:45<00:32, 2.76MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:45<00:29, 3.08MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:45<00:26, 3.42MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<01:20, 1.11MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<01:11, 1.24MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<00:56, 1.56MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<00:44, 1.96MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<00:36, 2.39MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<00:31, 2.77MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:47<00:27, 3.16MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:47<00:24, 3.49MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:47<00:22, 3.78MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<01:23, 1.01MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<01:26, 978kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<01:09, 1.22MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<00:53, 1.57MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<00:42, 1.97MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<00:34, 2.42MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<00:28, 2.92MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:49<00:24, 3.28MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<01:03, 1.27MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<01:08, 1.18MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<01:01, 1.31MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<00:49, 1.62MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<00:40, 1.97MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<00:32, 2.42MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<00:27, 2.83MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<00:23, 3.34MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<00:20, 3.80MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:51<00:18, 4.26MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<01:30, 850kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<01:23, 923kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<01:02, 1.21MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 787M/862M [05:52<00:47, 1.58MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:37, 2.01MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:28, 2.59MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:53<00:23, 3.09MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<01:21, 896kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<01:32, 783kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<01:13, 988kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:54<00:54, 1.32MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<00:39, 1.77MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<00:31, 2.24MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:55<00:23, 2.89MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<01:31, 754kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<01:32, 741kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<01:11, 959kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:53, 1.27MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:56<00:39, 1.70MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<00:29, 2.22MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<00:22, 2.85MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:53, 1.21MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<01:00, 1.07MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:47, 1.36MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<00:34, 1.81MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:58<00:26, 2.39MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:58<00:19, 3.07MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:43, 1.38MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:49, 1.22MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:38, 1.54MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:00<00:28, 2.07MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:00<00:21, 2.70MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:17, 3.22MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<00:54, 1.04MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<01:00, 921kB/s] .vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:02<00:46, 1.19MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:02<00:34, 1.60MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 808M/862M [06:02<00:25, 2.11MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:02<00:19, 2.72MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:03<00:15, 3.32MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<02:28, 351kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<02:03, 420kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<01:30, 569kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<01:05, 782kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:04<00:46, 1.08MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:04<00:33, 1.46MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:05<00:24, 1.96MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<01:46, 451kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<01:31, 525kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<01:06, 712kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:53, 880kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:06<00:37, 1.21MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:06<00:27, 1.62MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<00:21, 2.03MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:37, 1.18MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:47, 916kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<00:38, 1.12MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:08<00:28, 1.49MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:08<00:21, 1.95MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:08<00:16, 2.50MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:13, 2.99MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:37, 1.05MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:43, 899kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:34, 1.13MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 823M/862M [06:10<00:26, 1.47MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:10<00:19, 1.95MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:10<00:15, 2.40MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:10<00:12, 2.99MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:10, 3.57MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:34, 1.02MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:40, 874kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:34, 1.03MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:28, 1.23MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:22, 1.56MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:18, 1.83MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:16, 2.09MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:14, 2.35MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:13, 2.52MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:11, 2.84MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<00:09, 3.39MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:13<00:08, 3.94MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:37, 828kB/s] .vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:29, 1.04MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:14<00:21, 1.41MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:14<00:15, 1.85MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:14<00:12, 2.36MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:14<00:09, 3.01MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:33, 801kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:40, 671kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:32, 830kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:23, 1.12MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:16<00:17, 1.47MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:16<00:13, 1.89MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:16<00:10, 2.33MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:08, 2.77MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<00:07, 3.27MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:31, 745kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:27, 824kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:20, 1.08MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 841M/862M [06:18<00:15, 1.43MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:18<00:11, 1.86MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:18<00:08, 2.32MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:18<00:06, 2.86MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:05, 3.24MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:39, 478kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:38, 496kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:28, 645kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:20, 877kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:20<00:14, 1.18MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:20<00:10, 1.56MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:20<00:07, 2.02MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:20<00:06, 2.51MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:13, 1.07MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:13, 1.10MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:10, 1.32MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:08, 1.68MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:22<00:06, 2.15MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 849M/862M [06:22<00:04, 2.66MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:22<00:03, 3.24MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:22<00:03, 3.76MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:22<00:02, 4.31MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 851M/862M [06:24<00:41, 255kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:35, 302kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:25, 408kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:16, 566kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:11, 778kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:24<00:07, 1.05MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:24<00:05, 1.39MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:24<00:03, 1.82MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:07, 912kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:06, 969kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:04, 1.26MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:03, 1.65MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:26<00:01, 2.12MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:26<00:01, 2.68MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:26<00:00, 3.21MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:52, 46.6kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:38, 62.5kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:29, 81.8kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:21, 106kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:16, 135kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:13, 166kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:10, 205kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:08, 247kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:29<00:07, 265kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:29<00:06, 309kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:29<00:05, 348kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:29<00:05, 376kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:29<00:04, 401kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:29<00:04, 436kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:29<00:03, 464kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:29<00:03, 506kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:30<00:02, 533kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:30<00:02, 553kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:30<00:02, 584kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:30<00:02, 609kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:30<00:01, 659kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:30<00:01, 616kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:30<00:01, 659kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:30<00:01, 641kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:30<00:01, 683kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:31<00:01, 660kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:31<00:01, 695kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:31<00:00, 699kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:31<00:00, 744kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:31<00:00, 726kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:31<00:00, 784kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:31<00:00, 768kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:31<00:00, 782kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:31<00:00, 827kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:32<00:00, 846kB/s].vector_cache/glove.6B.zip: 862MB [06:32, 2.20MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 206/400000 [00:00<03:14, 2059.99it/s]  0%|          | 1045/400000 [00:00<02:29, 2662.47it/s]  0%|          | 1903/400000 [00:00<01:58, 3356.96it/s]  1%|          | 2705/400000 [00:00<01:37, 4065.50it/s]  1%|          | 3560/400000 [00:00<01:22, 4824.37it/s]  1%|          | 4431/400000 [00:00<01:11, 5569.30it/s]  1%|▏         | 5296/400000 [00:00<01:03, 6235.08it/s]  2%|▏         | 6170/400000 [00:00<00:57, 6821.46it/s]  2%|▏         | 7050/400000 [00:00<00:53, 7313.17it/s]  2%|▏         | 7871/400000 [00:01<00:51, 7546.06it/s]  2%|▏         | 8728/400000 [00:01<00:49, 7825.50it/s]  2%|▏         | 9604/400000 [00:01<00:48, 8083.91it/s]  3%|▎         | 10469/400000 [00:01<00:47, 8243.76it/s]  3%|▎         | 11334/400000 [00:01<00:46, 8360.47it/s]  3%|▎         | 12196/400000 [00:01<00:45, 8434.07it/s]  3%|▎         | 13053/400000 [00:01<00:45, 8432.99it/s]  3%|▎         | 13944/400000 [00:01<00:45, 8569.86it/s]  4%|▎         | 14808/400000 [00:01<00:44, 8574.06it/s]  4%|▍         | 15684/400000 [00:01<00:44, 8628.27it/s]  4%|▍         | 16551/400000 [00:02<00:44, 8636.58it/s]  4%|▍         | 17418/400000 [00:02<00:44, 8622.08it/s]  5%|▍         | 18296/400000 [00:02<00:44, 8667.27it/s]  5%|▍         | 19167/400000 [00:02<00:43, 8677.36it/s]  5%|▌         | 20036/400000 [00:02<00:43, 8642.35it/s]  5%|▌         | 20901/400000 [00:02<00:43, 8624.28it/s]  5%|▌         | 21764/400000 [00:02<00:44, 8584.65it/s]  6%|▌         | 22623/400000 [00:02<00:44, 8572.02it/s]  6%|▌         | 23481/400000 [00:02<00:44, 8492.11it/s]  6%|▌         | 24347/400000 [00:02<00:43, 8541.62it/s]  6%|▋         | 25202/400000 [00:03<00:44, 8509.74it/s]  7%|▋         | 26089/400000 [00:03<00:43, 8614.18it/s]  7%|▋         | 26951/400000 [00:03<00:44, 8398.20it/s]  7%|▋         | 27827/400000 [00:03<00:43, 8502.62it/s]  7%|▋         | 28700/400000 [00:03<00:43, 8569.58it/s]  7%|▋         | 29583/400000 [00:03<00:42, 8644.48it/s]  8%|▊         | 30454/400000 [00:03<00:42, 8662.09it/s]  8%|▊         | 31344/400000 [00:03<00:42, 8731.44it/s]  8%|▊         | 32225/400000 [00:03<00:42, 8754.13it/s]  8%|▊         | 33101/400000 [00:03<00:42, 8731.44it/s]  8%|▊         | 33976/400000 [00:04<00:41, 8735.91it/s]  9%|▊         | 34850/400000 [00:04<00:42, 8581.75it/s]  9%|▉         | 35728/400000 [00:04<00:42, 8639.52it/s]  9%|▉         | 36593/400000 [00:04<00:42, 8620.46it/s]  9%|▉         | 37456/400000 [00:04<00:42, 8617.31it/s] 10%|▉         | 38319/400000 [00:04<00:42, 8591.91it/s] 10%|▉         | 39179/400000 [00:04<00:42, 8399.80it/s] 10%|█         | 40064/400000 [00:04<00:42, 8529.06it/s] 10%|█         | 40944/400000 [00:04<00:41, 8607.72it/s] 10%|█         | 41820/400000 [00:04<00:41, 8651.71it/s] 11%|█         | 42704/400000 [00:05<00:41, 8706.38it/s] 11%|█         | 43576/400000 [00:05<00:41, 8516.66it/s] 11%|█         | 44443/400000 [00:05<00:41, 8559.21it/s] 11%|█▏        | 45328/400000 [00:05<00:41, 8642.71it/s] 12%|█▏        | 46194/400000 [00:05<00:41, 8513.11it/s] 12%|█▏        | 47047/400000 [00:05<00:41, 8494.00it/s] 12%|█▏        | 47936/400000 [00:05<00:40, 8608.25it/s] 12%|█▏        | 48798/400000 [00:05<00:41, 8536.90it/s] 12%|█▏        | 49653/400000 [00:05<00:41, 8494.98it/s] 13%|█▎        | 50509/400000 [00:05<00:41, 8511.21it/s] 13%|█▎        | 51374/400000 [00:06<00:40, 8548.29it/s] 13%|█▎        | 52261/400000 [00:06<00:40, 8641.36it/s] 13%|█▎        | 53140/400000 [00:06<00:39, 8685.11it/s] 14%|█▎        | 54020/400000 [00:06<00:39, 8717.22it/s] 14%|█▎        | 54893/400000 [00:06<00:39, 8715.49it/s] 14%|█▍        | 55765/400000 [00:06<00:39, 8656.64it/s] 14%|█▍        | 56639/400000 [00:06<00:39, 8680.61it/s] 14%|█▍        | 57513/400000 [00:06<00:39, 8697.94it/s] 15%|█▍        | 58383/400000 [00:06<00:39, 8662.05it/s] 15%|█▍        | 59264/400000 [00:06<00:39, 8704.84it/s] 15%|█▌        | 60135/400000 [00:07<00:39, 8595.03it/s] 15%|█▌        | 61000/400000 [00:07<00:39, 8609.17it/s] 15%|█▌        | 61862/400000 [00:07<00:40, 8307.70it/s] 16%|█▌        | 62740/400000 [00:07<00:39, 8442.06it/s] 16%|█▌        | 63614/400000 [00:07<00:39, 8525.03it/s] 16%|█▌        | 64469/400000 [00:07<00:39, 8488.58it/s] 16%|█▋        | 65347/400000 [00:07<00:39, 8571.36it/s] 17%|█▋        | 66244/400000 [00:07<00:38, 8686.38it/s] 17%|█▋        | 67114/400000 [00:07<00:38, 8547.63it/s] 17%|█▋        | 67986/400000 [00:07<00:38, 8596.85it/s] 17%|█▋        | 68865/400000 [00:08<00:38, 8652.95it/s] 17%|█▋        | 69738/400000 [00:08<00:38, 8674.89it/s] 18%|█▊        | 70607/400000 [00:08<00:38, 8666.05it/s] 18%|█▊        | 71474/400000 [00:08<00:37, 8659.12it/s] 18%|█▊        | 72350/400000 [00:08<00:37, 8687.14it/s] 18%|█▊        | 73221/400000 [00:08<00:37, 8692.06it/s] 19%|█▊        | 74097/400000 [00:08<00:37, 8709.52it/s] 19%|█▊        | 74969/400000 [00:08<00:37, 8693.75it/s] 19%|█▉        | 75839/400000 [00:08<00:37, 8637.82it/s] 19%|█▉        | 76703/400000 [00:08<00:37, 8619.53it/s] 19%|█▉        | 77566/400000 [00:09<00:37, 8614.31it/s] 20%|█▉        | 78440/400000 [00:09<00:37, 8649.44it/s] 20%|█▉        | 79315/400000 [00:09<00:36, 8678.10it/s] 20%|██        | 80191/400000 [00:09<00:36, 8702.37it/s] 20%|██        | 81075/400000 [00:09<00:36, 8742.53it/s] 20%|██        | 81950/400000 [00:09<00:36, 8724.44it/s] 21%|██        | 82842/400000 [00:09<00:36, 8781.25it/s] 21%|██        | 83721/400000 [00:09<00:36, 8614.18it/s] 21%|██        | 84599/400000 [00:09<00:36, 8661.05it/s] 21%|██▏       | 85490/400000 [00:10<00:36, 8731.40it/s] 22%|██▏       | 86367/400000 [00:10<00:35, 8741.11it/s] 22%|██▏       | 87242/400000 [00:10<00:36, 8675.58it/s] 22%|██▏       | 88118/400000 [00:10<00:35, 8698.49it/s] 22%|██▏       | 88989/400000 [00:10<00:35, 8660.12it/s] 22%|██▏       | 89862/400000 [00:10<00:35, 8680.71it/s] 23%|██▎       | 90742/400000 [00:10<00:35, 8714.65it/s] 23%|██▎       | 91614/400000 [00:10<00:35, 8711.17it/s] 23%|██▎       | 92500/400000 [00:10<00:35, 8754.31it/s] 23%|██▎       | 93384/400000 [00:10<00:34, 8777.22it/s] 24%|██▎       | 94262/400000 [00:11<00:34, 8740.06it/s] 24%|██▍       | 95150/400000 [00:11<00:34, 8778.97it/s] 24%|██▍       | 96041/400000 [00:11<00:34, 8814.45it/s] 24%|██▍       | 96923/400000 [00:11<00:34, 8693.88it/s] 24%|██▍       | 97803/400000 [00:11<00:34, 8724.51it/s] 25%|██▍       | 98691/400000 [00:11<00:34, 8768.93it/s] 25%|██▍       | 99577/400000 [00:11<00:34, 8794.86it/s] 25%|██▌       | 100482/400000 [00:11<00:33, 8867.62it/s] 25%|██▌       | 101373/400000 [00:11<00:33, 8877.16it/s] 26%|██▌       | 102261/400000 [00:11<00:33, 8850.32it/s] 26%|██▌       | 103150/400000 [00:12<00:33, 8859.78it/s] 26%|██▌       | 104037/400000 [00:12<00:33, 8838.57it/s] 26%|██▌       | 104921/400000 [00:12<00:33, 8759.65it/s] 26%|██▋       | 105798/400000 [00:12<00:33, 8755.65it/s] 27%|██▋       | 106674/400000 [00:12<00:33, 8697.27it/s] 27%|██▋       | 107544/400000 [00:12<00:34, 8558.94it/s] 27%|██▋       | 108420/400000 [00:12<00:33, 8616.84it/s] 27%|██▋       | 109283/400000 [00:12<00:34, 8421.81it/s] 28%|██▊       | 110160/400000 [00:12<00:34, 8521.09it/s] 28%|██▊       | 111027/400000 [00:12<00:33, 8564.86it/s] 28%|██▊       | 111907/400000 [00:13<00:33, 8633.75it/s] 28%|██▊       | 112772/400000 [00:13<00:33, 8578.47it/s] 28%|██▊       | 113631/400000 [00:13<00:34, 8344.10it/s] 29%|██▊       | 114511/400000 [00:13<00:33, 8473.06it/s] 29%|██▉       | 115389/400000 [00:13<00:33, 8560.78it/s] 29%|██▉       | 116272/400000 [00:13<00:32, 8639.11it/s] 29%|██▉       | 117156/400000 [00:13<00:32, 8697.21it/s] 30%|██▉       | 118055/400000 [00:13<00:32, 8780.51it/s] 30%|██▉       | 118947/400000 [00:13<00:31, 8819.02it/s] 30%|██▉       | 119835/400000 [00:13<00:31, 8836.78it/s] 30%|███       | 120720/400000 [00:14<00:31, 8833.70it/s] 30%|███       | 121606/400000 [00:14<00:31, 8839.01it/s] 31%|███       | 122491/400000 [00:14<00:32, 8664.87it/s] 31%|███       | 123363/400000 [00:14<00:31, 8679.52it/s] 31%|███       | 124251/400000 [00:14<00:31, 8737.10it/s] 31%|███▏      | 125126/400000 [00:14<00:31, 8640.81it/s] 32%|███▏      | 126007/400000 [00:14<00:31, 8690.12it/s] 32%|███▏      | 126877/400000 [00:14<00:31, 8612.20it/s] 32%|███▏      | 127739/400000 [00:14<00:31, 8603.22it/s] 32%|███▏      | 128625/400000 [00:14<00:31, 8677.95it/s] 32%|███▏      | 129509/400000 [00:15<00:31, 8723.11it/s] 33%|███▎      | 130382/400000 [00:15<00:31, 8535.07it/s] 33%|███▎      | 131245/400000 [00:15<00:31, 8562.94it/s] 33%|███▎      | 132103/400000 [00:15<00:31, 8565.83it/s] 33%|███▎      | 132984/400000 [00:15<00:30, 8637.08it/s] 33%|███▎      | 133868/400000 [00:15<00:30, 8696.31it/s] 34%|███▎      | 134746/400000 [00:15<00:30, 8719.49it/s] 34%|███▍      | 135619/400000 [00:15<00:30, 8675.23it/s] 34%|███▍      | 136487/400000 [00:15<00:30, 8568.58it/s] 34%|███▍      | 137345/400000 [00:15<00:31, 8315.84it/s] 35%|███▍      | 138179/400000 [00:16<00:32, 8168.63it/s] 35%|███▍      | 139021/400000 [00:16<00:31, 8241.53it/s] 35%|███▍      | 139890/400000 [00:16<00:31, 8369.30it/s] 35%|███▌      | 140753/400000 [00:16<00:30, 8444.81it/s] 35%|███▌      | 141629/400000 [00:16<00:30, 8534.90it/s] 36%|███▌      | 142513/400000 [00:16<00:29, 8622.29it/s] 36%|███▌      | 143380/400000 [00:16<00:29, 8634.05it/s] 36%|███▌      | 144254/400000 [00:16<00:29, 8663.29it/s] 36%|███▋      | 145121/400000 [00:16<00:29, 8658.24it/s] 36%|███▋      | 145988/400000 [00:16<00:29, 8649.13it/s] 37%|███▋      | 146854/400000 [00:17<00:29, 8628.12it/s] 37%|███▋      | 147718/400000 [00:17<00:29, 8607.03it/s] 37%|███▋      | 148579/400000 [00:17<00:29, 8556.07it/s] 37%|███▋      | 149460/400000 [00:17<00:29, 8628.46it/s] 38%|███▊      | 150334/400000 [00:17<00:28, 8660.88it/s] 38%|███▊      | 151201/400000 [00:17<00:29, 8568.66it/s] 38%|███▊      | 152064/400000 [00:17<00:28, 8582.49it/s] 38%|███▊      | 152930/400000 [00:17<00:28, 8604.25it/s] 38%|███▊      | 153794/400000 [00:17<00:28, 8613.91it/s] 39%|███▊      | 154672/400000 [00:18<00:28, 8660.98it/s] 39%|███▉      | 155548/400000 [00:18<00:28, 8687.61it/s] 39%|███▉      | 156417/400000 [00:18<00:28, 8543.62it/s] 39%|███▉      | 157272/400000 [00:18<00:28, 8419.67it/s] 40%|███▉      | 158131/400000 [00:18<00:28, 8469.43it/s] 40%|███▉      | 158979/400000 [00:18<00:28, 8416.91it/s] 40%|███▉      | 159860/400000 [00:18<00:28, 8528.94it/s] 40%|████      | 160742/400000 [00:18<00:27, 8612.48it/s] 40%|████      | 161626/400000 [00:18<00:27, 8678.77it/s] 41%|████      | 162496/400000 [00:18<00:27, 8684.46it/s] 41%|████      | 163376/400000 [00:19<00:27, 8716.56it/s] 41%|████      | 164259/400000 [00:19<00:26, 8749.09it/s] 41%|████▏     | 165135/400000 [00:19<00:27, 8684.78it/s] 42%|████▏     | 166004/400000 [00:19<00:27, 8548.57it/s] 42%|████▏     | 166876/400000 [00:19<00:27, 8597.02it/s] 42%|████▏     | 167761/400000 [00:19<00:26, 8670.28it/s] 42%|████▏     | 168655/400000 [00:19<00:26, 8748.07it/s] 42%|████▏     | 169550/400000 [00:19<00:26, 8806.51it/s] 43%|████▎     | 170434/400000 [00:19<00:26, 8813.34it/s] 43%|████▎     | 171317/400000 [00:19<00:25, 8815.87it/s] 43%|████▎     | 172199/400000 [00:20<00:25, 8811.61it/s] 43%|████▎     | 173081/400000 [00:20<00:26, 8710.17it/s] 43%|████▎     | 173960/400000 [00:20<00:25, 8732.71it/s] 44%|████▎     | 174842/400000 [00:20<00:25, 8756.13it/s] 44%|████▍     | 175718/400000 [00:20<00:25, 8747.55it/s] 44%|████▍     | 176603/400000 [00:20<00:25, 8777.51it/s] 44%|████▍     | 177481/400000 [00:20<00:25, 8734.78it/s] 45%|████▍     | 178361/400000 [00:20<00:25, 8753.89it/s] 45%|████▍     | 179237/400000 [00:20<00:25, 8687.20it/s] 45%|████▌     | 180112/400000 [00:20<00:25, 8704.03it/s] 45%|████▌     | 180983/400000 [00:21<00:25, 8644.99it/s] 45%|████▌     | 181848/400000 [00:21<00:25, 8572.14it/s] 46%|████▌     | 182706/400000 [00:21<00:25, 8539.57it/s] 46%|████▌     | 183586/400000 [00:21<00:25, 8616.06it/s] 46%|████▌     | 184468/400000 [00:21<00:24, 8676.10it/s] 46%|████▋     | 185350/400000 [00:21<00:24, 8716.57it/s] 47%|████▋     | 186245/400000 [00:21<00:24, 8784.99it/s] 47%|████▋     | 187151/400000 [00:21<00:24, 8862.82it/s] 47%|████▋     | 188038/400000 [00:21<00:23, 8863.33it/s] 47%|████▋     | 188948/400000 [00:21<00:23, 8932.12it/s] 47%|████▋     | 189842/400000 [00:22<00:23, 8904.03it/s] 48%|████▊     | 190733/400000 [00:22<00:23, 8813.76it/s] 48%|████▊     | 191615/400000 [00:22<00:23, 8809.16it/s] 48%|████▊     | 192497/400000 [00:22<00:23, 8693.04it/s] 48%|████▊     | 193367/400000 [00:22<00:23, 8678.20it/s] 49%|████▊     | 194272/400000 [00:22<00:23, 8783.88it/s] 49%|████▉     | 195167/400000 [00:22<00:23, 8830.95it/s] 49%|████▉     | 196052/400000 [00:22<00:23, 8836.00it/s] 49%|████▉     | 196936/400000 [00:22<00:23, 8823.01it/s] 49%|████▉     | 197831/400000 [00:22<00:22, 8859.59it/s] 50%|████▉     | 198742/400000 [00:23<00:22, 8930.41it/s] 50%|████▉     | 199651/400000 [00:23<00:22, 8976.71it/s] 50%|█████     | 200549/400000 [00:23<00:22, 8926.80it/s] 50%|█████     | 201442/400000 [00:23<00:22, 8877.63it/s] 51%|█████     | 202330/400000 [00:23<00:22, 8876.02it/s] 51%|█████     | 203232/400000 [00:23<00:22, 8917.83it/s] 51%|█████     | 204124/400000 [00:23<00:22, 8895.81it/s] 51%|█████▏    | 205014/400000 [00:23<00:22, 8861.73it/s] 51%|█████▏    | 205901/400000 [00:23<00:22, 8770.33it/s] 52%|█████▏    | 206779/400000 [00:23<00:22, 8679.45it/s] 52%|█████▏    | 207653/400000 [00:24<00:22, 8697.45it/s] 52%|█████▏    | 208524/400000 [00:24<00:22, 8432.98it/s] 52%|█████▏    | 209372/400000 [00:24<00:22, 8444.72it/s] 53%|█████▎    | 210218/400000 [00:24<00:22, 8429.60it/s] 53%|█████▎    | 211062/400000 [00:24<00:22, 8357.57it/s] 53%|█████▎    | 211950/400000 [00:24<00:22, 8506.53it/s] 53%|█████▎    | 212851/400000 [00:24<00:21, 8649.22it/s] 53%|█████▎    | 213726/400000 [00:24<00:21, 8677.84it/s] 54%|█████▎    | 214599/400000 [00:24<00:21, 8691.93it/s] 54%|█████▍    | 215488/400000 [00:24<00:21, 8750.11it/s] 54%|█████▍    | 216367/400000 [00:25<00:20, 8750.76it/s] 54%|█████▍    | 217243/400000 [00:25<00:21, 8550.78it/s] 55%|█████▍    | 218115/400000 [00:25<00:21, 8598.81it/s] 55%|█████▍    | 218981/400000 [00:25<00:21, 8615.72it/s] 55%|█████▍    | 219856/400000 [00:25<00:20, 8654.25it/s] 55%|█████▌    | 220723/400000 [00:25<00:20, 8657.02it/s] 55%|█████▌    | 221604/400000 [00:25<00:20, 8700.20it/s] 56%|█████▌    | 222478/400000 [00:25<00:20, 8711.04it/s] 56%|█████▌    | 223350/400000 [00:25<00:20, 8703.33it/s] 56%|█████▌    | 224234/400000 [00:25<00:20, 8742.40it/s] 56%|█████▋    | 225111/400000 [00:26<00:19, 8750.17it/s] 56%|█████▋    | 225993/400000 [00:26<00:19, 8768.37it/s] 57%|█████▋    | 226876/400000 [00:26<00:19, 8786.22it/s] 57%|█████▋    | 227760/400000 [00:26<00:19, 8800.10it/s] 57%|█████▋    | 228657/400000 [00:26<00:19, 8849.15it/s] 57%|█████▋    | 229558/400000 [00:26<00:19, 8895.28it/s] 58%|█████▊    | 230458/400000 [00:26<00:18, 8925.29it/s] 58%|█████▊    | 231354/400000 [00:26<00:18, 8935.46it/s] 58%|█████▊    | 232248/400000 [00:26<00:18, 8919.96it/s] 58%|█████▊    | 233141/400000 [00:26<00:18, 8819.98it/s] 59%|█████▊    | 234024/400000 [00:27<00:18, 8776.81it/s] 59%|█████▊    | 234902/400000 [00:27<00:18, 8774.42it/s] 59%|█████▉    | 235780/400000 [00:27<00:19, 8603.61it/s] 59%|█████▉    | 236642/400000 [00:27<00:19, 8348.76it/s] 59%|█████▉    | 237517/400000 [00:27<00:19, 8463.06it/s] 60%|█████▉    | 238366/400000 [00:27<00:19, 8318.20it/s] 60%|█████▉    | 239225/400000 [00:27<00:19, 8396.77it/s] 60%|██████    | 240115/400000 [00:27<00:18, 8539.20it/s] 60%|██████    | 240987/400000 [00:27<00:18, 8592.29it/s] 60%|██████    | 241865/400000 [00:28<00:18, 8645.29it/s] 61%|██████    | 242733/400000 [00:28<00:18, 8654.02it/s] 61%|██████    | 243600/400000 [00:28<00:18, 8535.87it/s] 61%|██████    | 244477/400000 [00:28<00:18, 8603.14it/s] 61%|██████▏   | 245344/400000 [00:28<00:17, 8622.99it/s] 62%|██████▏   | 246248/400000 [00:28<00:17, 8741.97it/s] 62%|██████▏   | 247131/400000 [00:28<00:17, 8765.80it/s] 62%|██████▏   | 248022/400000 [00:28<00:17, 8808.02it/s] 62%|██████▏   | 248912/400000 [00:28<00:17, 8833.59it/s] 62%|██████▏   | 249803/400000 [00:28<00:16, 8855.50it/s] 63%|██████▎   | 250702/400000 [00:29<00:16, 8892.63it/s] 63%|██████▎   | 251592/400000 [00:29<00:17, 8675.45it/s] 63%|██████▎   | 252461/400000 [00:29<00:17, 8429.54it/s] 63%|██████▎   | 253336/400000 [00:29<00:17, 8521.40it/s] 64%|██████▎   | 254202/400000 [00:29<00:17, 8561.66it/s] 64%|██████▍   | 255072/400000 [00:29<00:16, 8601.60it/s] 64%|██████▍   | 255937/400000 [00:29<00:16, 8615.42it/s] 64%|██████▍   | 256802/400000 [00:29<00:16, 8624.04it/s] 64%|██████▍   | 257673/400000 [00:29<00:16, 8648.81it/s] 65%|██████▍   | 258539/400000 [00:29<00:16, 8649.33it/s] 65%|██████▍   | 259405/400000 [00:30<00:16, 8643.64it/s] 65%|██████▌   | 260270/400000 [00:30<00:16, 8455.08it/s] 65%|██████▌   | 261117/400000 [00:30<00:16, 8322.39it/s] 65%|██████▌   | 261976/400000 [00:30<00:16, 8399.40it/s] 66%|██████▌   | 262829/400000 [00:30<00:16, 8436.34it/s] 66%|██████▌   | 263695/400000 [00:30<00:16, 8501.61it/s] 66%|██████▌   | 264562/400000 [00:30<00:15, 8550.70it/s] 66%|██████▋   | 265428/400000 [00:30<00:15, 8581.26it/s] 67%|██████▋   | 266289/400000 [00:30<00:15, 8587.58it/s] 67%|██████▋   | 267149/400000 [00:30<00:15, 8587.22it/s] 67%|██████▋   | 268009/400000 [00:31<00:15, 8589.70it/s] 67%|██████▋   | 268869/400000 [00:31<00:15, 8213.97it/s] 67%|██████▋   | 269725/400000 [00:31<00:15, 8313.74it/s] 68%|██████▊   | 270592/400000 [00:31<00:15, 8417.23it/s] 68%|██████▊   | 271449/400000 [00:31<00:15, 8459.14it/s] 68%|██████▊   | 272301/400000 [00:31<00:15, 8475.29it/s] 68%|██████▊   | 273165/400000 [00:31<00:14, 8522.77it/s] 69%|██████▊   | 274036/400000 [00:31<00:14, 8575.42it/s] 69%|██████▊   | 274900/400000 [00:31<00:14, 8594.55it/s] 69%|██████▉   | 275769/400000 [00:31<00:14, 8621.16it/s] 69%|██████▉   | 276643/400000 [00:32<00:14, 8654.71it/s] 69%|██████▉   | 277509/400000 [00:32<00:14, 8555.45it/s] 70%|██████▉   | 278382/400000 [00:32<00:14, 8606.19it/s] 70%|██████▉   | 279265/400000 [00:32<00:13, 8670.42it/s] 70%|███████   | 280136/400000 [00:32<00:13, 8681.60it/s] 70%|███████   | 281005/400000 [00:32<00:13, 8674.75it/s] 70%|███████   | 281894/400000 [00:32<00:13, 8735.75it/s] 71%|███████   | 282768/400000 [00:32<00:13, 8699.59it/s] 71%|███████   | 283658/400000 [00:32<00:13, 8757.52it/s] 71%|███████   | 284534/400000 [00:32<00:13, 8749.25it/s] 71%|███████▏  | 285415/400000 [00:33<00:13, 8767.21it/s] 72%|███████▏  | 286292/400000 [00:33<00:12, 8758.41it/s] 72%|███████▏  | 287168/400000 [00:33<00:12, 8683.84it/s] 72%|███████▏  | 288056/400000 [00:33<00:12, 8739.10it/s] 72%|███████▏  | 288931/400000 [00:33<00:12, 8722.59it/s] 72%|███████▏  | 289804/400000 [00:33<00:12, 8719.53it/s] 73%|███████▎  | 290677/400000 [00:33<00:12, 8700.75it/s] 73%|███████▎  | 291548/400000 [00:33<00:12, 8671.88it/s] 73%|███████▎  | 292416/400000 [00:33<00:12, 8588.65it/s] 73%|███████▎  | 293276/400000 [00:33<00:12, 8580.40it/s] 74%|███████▎  | 294135/400000 [00:34<00:12, 8480.60it/s] 74%|███████▎  | 294986/400000 [00:34<00:12, 8488.16it/s] 74%|███████▍  | 295862/400000 [00:34<00:12, 8565.12it/s] 74%|███████▍  | 296719/400000 [00:34<00:12, 8452.34it/s] 74%|███████▍  | 297590/400000 [00:34<00:12, 8527.24it/s] 75%|███████▍  | 298444/400000 [00:34<00:11, 8520.82it/s] 75%|███████▍  | 299297/400000 [00:34<00:11, 8487.43it/s] 75%|███████▌  | 300151/400000 [00:34<00:11, 8501.25it/s] 75%|███████▌  | 301013/400000 [00:34<00:11, 8536.31it/s] 75%|███████▌  | 301873/400000 [00:34<00:11, 8552.50it/s] 76%|███████▌  | 302747/400000 [00:35<00:11, 8605.28it/s] 76%|███████▌  | 303608/400000 [00:35<00:11, 8604.31it/s] 76%|███████▌  | 304492/400000 [00:35<00:11, 8670.79it/s] 76%|███████▋  | 305371/400000 [00:35<00:10, 8705.59it/s] 77%|███████▋  | 306282/400000 [00:35<00:10, 8820.30it/s] 77%|███████▋  | 307165/400000 [00:35<00:10, 8787.72it/s] 77%|███████▋  | 308048/400000 [00:35<00:10, 8798.70it/s] 77%|███████▋  | 308930/400000 [00:35<00:10, 8803.50it/s] 77%|███████▋  | 309819/400000 [00:35<00:10, 8826.51it/s] 78%|███████▊  | 310702/400000 [00:36<00:10, 8827.34it/s] 78%|███████▊  | 311585/400000 [00:36<00:10, 8710.18it/s] 78%|███████▊  | 312457/400000 [00:36<00:10, 8712.51it/s] 78%|███████▊  | 313329/400000 [00:36<00:10, 8634.03it/s] 79%|███████▊  | 314193/400000 [00:36<00:10, 8458.55it/s] 79%|███████▉  | 315070/400000 [00:36<00:09, 8548.70it/s] 79%|███████▉  | 315942/400000 [00:36<00:09, 8599.27it/s] 79%|███████▉  | 316825/400000 [00:36<00:09, 8665.66it/s] 79%|███████▉  | 317709/400000 [00:36<00:09, 8716.56it/s] 80%|███████▉  | 318608/400000 [00:36<00:09, 8796.44it/s] 80%|███████▉  | 319490/400000 [00:37<00:09, 8802.49it/s] 80%|████████  | 320371/400000 [00:37<00:09, 8775.10it/s] 80%|████████  | 321253/400000 [00:37<00:08, 8786.79it/s] 81%|████████  | 322138/400000 [00:37<00:08, 8804.67it/s] 81%|████████  | 323027/400000 [00:37<00:08, 8828.28it/s] 81%|████████  | 323912/400000 [00:37<00:08, 8834.18it/s] 81%|████████  | 324801/400000 [00:37<00:08, 8848.61it/s] 81%|████████▏ | 325686/400000 [00:37<00:08, 8823.60it/s] 82%|████████▏ | 326569/400000 [00:37<00:08, 8818.59it/s] 82%|████████▏ | 327452/400000 [00:37<00:08, 8818.68it/s] 82%|████████▏ | 328334/400000 [00:38<00:08, 8813.06it/s] 82%|████████▏ | 329216/400000 [00:38<00:08, 8786.38it/s] 83%|████████▎ | 330114/400000 [00:38<00:07, 8841.44it/s] 83%|████████▎ | 331010/400000 [00:38<00:07, 8875.41it/s] 83%|████████▎ | 331901/400000 [00:38<00:07, 8884.42it/s] 83%|████████▎ | 332790/400000 [00:38<00:07, 8839.94it/s] 83%|████████▎ | 333675/400000 [00:38<00:07, 8774.33it/s] 84%|████████▎ | 334558/400000 [00:38<00:07, 8788.64it/s] 84%|████████▍ | 335437/400000 [00:38<00:07, 8753.93it/s] 84%|████████▍ | 336313/400000 [00:38<00:07, 8742.90it/s] 84%|████████▍ | 337214/400000 [00:39<00:07, 8819.79it/s] 85%|████████▍ | 338097/400000 [00:39<00:07, 8628.83it/s] 85%|████████▍ | 338975/400000 [00:39<00:07, 8671.58it/s] 85%|████████▍ | 339863/400000 [00:39<00:06, 8732.09it/s] 85%|████████▌ | 340752/400000 [00:39<00:06, 8777.89it/s] 85%|████████▌ | 341639/400000 [00:39<00:06, 8803.52it/s] 86%|████████▌ | 342520/400000 [00:39<00:06, 8790.58it/s] 86%|████████▌ | 343413/400000 [00:39<00:06, 8831.22it/s] 86%|████████▌ | 344297/400000 [00:39<00:06, 8570.06it/s] 86%|████████▋ | 345199/400000 [00:39<00:06, 8697.70it/s] 87%|████████▋ | 346088/400000 [00:40<00:06, 8754.11it/s] 87%|████████▋ | 346965/400000 [00:40<00:06, 8710.04it/s] 87%|████████▋ | 347837/400000 [00:40<00:06, 8664.82it/s] 87%|████████▋ | 348715/400000 [00:40<00:05, 8697.78it/s] 87%|████████▋ | 349586/400000 [00:40<00:05, 8511.08it/s] 88%|████████▊ | 350439/400000 [00:40<00:05, 8444.06it/s] 88%|████████▊ | 351320/400000 [00:40<00:05, 8549.92it/s] 88%|████████▊ | 352209/400000 [00:40<00:05, 8648.21it/s] 88%|████████▊ | 353088/400000 [00:40<00:05, 8690.11it/s] 88%|████████▊ | 353958/400000 [00:40<00:05, 8678.28it/s] 89%|████████▊ | 354841/400000 [00:41<00:05, 8720.86it/s] 89%|████████▉ | 355714/400000 [00:41<00:05, 8673.80it/s] 89%|████████▉ | 356588/400000 [00:41<00:04, 8691.50it/s] 89%|████████▉ | 357458/400000 [00:41<00:04, 8667.01it/s] 90%|████████▉ | 358325/400000 [00:41<00:04, 8533.88it/s] 90%|████████▉ | 359188/400000 [00:41<00:04, 8561.39it/s] 90%|█████████ | 360045/400000 [00:41<00:04, 8530.60it/s] 90%|█████████ | 360923/400000 [00:41<00:04, 8601.23it/s] 90%|█████████ | 361794/400000 [00:41<00:04, 8633.02it/s] 91%|█████████ | 362658/400000 [00:41<00:04, 8559.97it/s] 91%|█████████ | 363533/400000 [00:42<00:04, 8615.39it/s] 91%|█████████ | 364396/400000 [00:42<00:04, 8619.67it/s] 91%|█████████▏| 365274/400000 [00:42<00:04, 8665.28it/s] 92%|█████████▏| 366152/400000 [00:42<00:03, 8698.30it/s] 92%|█████████▏| 367041/400000 [00:42<00:03, 8752.87it/s] 92%|█████████▏| 367919/400000 [00:42<00:03, 8758.63it/s] 92%|█████████▏| 368796/400000 [00:42<00:03, 8689.61it/s] 92%|█████████▏| 369689/400000 [00:42<00:03, 8757.93it/s] 93%|█████████▎| 370566/400000 [00:42<00:03, 8758.77it/s] 93%|█████████▎| 371443/400000 [00:42<00:03, 8750.88it/s] 93%|█████████▎| 372323/400000 [00:43<00:03, 8763.61it/s] 93%|█████████▎| 373200/400000 [00:43<00:03, 8654.13it/s] 94%|█████████▎| 374066/400000 [00:43<00:02, 8654.65it/s] 94%|█████████▎| 374932/400000 [00:43<00:02, 8645.31it/s] 94%|█████████▍| 375797/400000 [00:43<00:02, 8615.72it/s] 94%|█████████▍| 376659/400000 [00:43<00:02, 8611.83it/s] 94%|█████████▍| 377521/400000 [00:43<00:02, 8564.65it/s] 95%|█████████▍| 378402/400000 [00:43<00:02, 8635.55it/s] 95%|█████████▍| 379289/400000 [00:43<00:02, 8704.29it/s] 95%|█████████▌| 380164/400000 [00:43<00:02, 8716.33it/s] 95%|█████████▌| 381053/400000 [00:44<00:02, 8766.42it/s] 95%|█████████▌| 381930/400000 [00:44<00:02, 8739.62it/s] 96%|█████████▌| 382810/400000 [00:44<00:01, 8755.03it/s] 96%|█████████▌| 383686/400000 [00:44<00:01, 8744.29it/s] 96%|█████████▌| 384578/400000 [00:44<00:01, 8794.35it/s] 96%|█████████▋| 385459/400000 [00:44<00:01, 8798.53it/s] 97%|█████████▋| 386339/400000 [00:44<00:01, 8562.74it/s] 97%|█████████▋| 387222/400000 [00:44<00:01, 8639.98it/s] 97%|█████████▋| 388100/400000 [00:44<00:01, 8681.21it/s] 97%|█████████▋| 388969/400000 [00:44<00:01, 8634.26it/s] 97%|█████████▋| 389834/400000 [00:45<00:01, 8568.08it/s] 98%|█████████▊| 390692/400000 [00:45<00:01, 8517.91it/s] 98%|█████████▊| 391575/400000 [00:45<00:00, 8608.14it/s] 98%|█████████▊| 392455/400000 [00:45<00:00, 8664.74it/s] 98%|█████████▊| 393338/400000 [00:45<00:00, 8712.41it/s] 99%|█████████▊| 394234/400000 [00:45<00:00, 8785.04it/s] 99%|█████████▉| 395113/400000 [00:45<00:00, 8757.87it/s] 99%|█████████▉| 396004/400000 [00:45<00:00, 8802.92it/s] 99%|█████████▉| 396885/400000 [00:45<00:00, 8771.97it/s] 99%|█████████▉| 397767/400000 [00:46<00:00, 8785.18it/s]100%|█████████▉| 398648/400000 [00:46<00:00, 8789.70it/s]100%|█████████▉| 399535/400000 [00:46<00:00, 8811.90it/s]100%|█████████▉| 399999/400000 [00:46<00:00, 8647.16it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7fe4c0061c88>, <torchtext.data.dataset.TabularDataset object at 0x7fe4c0061dd8>, <torchtext.vocab.Vocab object at 0x7fe4c0061cf8>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  6.27 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  6.27 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  6.27 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  7.51 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  7.51 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.05 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.05 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.90 file/s]2020-06-20 00:19:59.193119: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-06-20 00:19:59.197577: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095125000 Hz
2020-06-20 00:19:59.197803: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5587fd93e5f0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-20 00:19:59.197840: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 49152/9912422 [00:00<00:21, 456751.47it/s] 90%|█████████ | 8937472/9912422 [00:00<00:01, 651053.08it/s]9920512it [00:00, 45945065.49it/s]                           
0it [00:00, ?it/s]32768it [00:00, 713760.95it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:11, 141123.69it/s]1654784it [00:00, 10475846.21it/s]                         
0it [00:00, ?it/s]8192it [00:00, 187424.25it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
