
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/525b5a33511bef1b32994ba341cc4ff09d4ffc92', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '525b5a33511bef1b32994ba341cc4ff09d4ffc92', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/525b5a33511bef1b32994ba341cc4ff09d4ffc92

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/525b5a33511bef1b32994ba341cc4ff09d4ffc92

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/525b5a33511bef1b32994ba341cc4ff09d4ffc92

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f4f50afab70> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f4f50afab70> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f4fbb8620d0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f4fbb8620d0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f4fd558ee18> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f4fd558ee18> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f4f690dd950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f4f690dd950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f4f690dd950> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:13, 135236.49it/s] 77%|███████▋  | 7667712/9912422 [00:00<00:11, 193048.72it/s]9920512it [00:00, 41095294.46it/s]                           
0it [00:00, ?it/s]32768it [00:00, 570880.68it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:11, 140168.88it/s]1654784it [00:00, 10985353.45it/s]                         
0it [00:00, ?it/s]8192it [00:00, 225212.29it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f4f6630a2b0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f4f5016aeb8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f4f690dd598> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f4f690dd598> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f4f690dd598> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<00:59,  2.70 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<00:59,  2.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<00:59,  2.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<00:58,  2.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<00:58,  2.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<00:58,  2.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▎         | 6/162 [00:00<00:41,  3.77 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<00:41,  3.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:41,  3.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:40,  3.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:40,  3.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:40,  3.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:40,  3.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   7%|▋         | 12/162 [00:00<00:28,  5.23 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:28,  5.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:28,  5.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:28,  5.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:28,  5.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:27,  5.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:27,  5.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  11%|█         | 18/162 [00:00<00:20,  7.18 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:20,  7.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:19,  7.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:19,  7.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:19,  7.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:19,  7.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:19,  7.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  15%|█▍        | 24/162 [00:00<00:14,  9.71 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:14,  9.71 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:14,  9.71 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:14,  9.71 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:13,  9.71 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:13,  9.71 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:13,  9.71 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  19%|█▊        | 30/162 [00:00<00:10, 12.89 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:10, 12.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:10, 12.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:10, 12.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:10, 12.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:09, 12.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:09, 12.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  22%|██▏       | 36/162 [00:01<00:07, 16.66 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:07, 16.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:07, 16.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:07, 16.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:07, 16.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:07, 16.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:07, 16.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  26%|██▌       | 42/162 [00:01<00:05, 21.07 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:05, 21.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:05, 21.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:05, 21.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:05, 21.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:05, 21.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:05, 21.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  30%|██▉       | 48/162 [00:01<00:04, 25.51 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:04, 25.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:04, 25.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:04, 25.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:04, 25.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:04, 25.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:04, 25.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  33%|███▎      | 54/162 [00:01<00:03, 30.24 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:03, 30.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:03, 30.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:03, 30.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:03, 30.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:03, 30.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:03, 30.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  37%|███▋      | 60/162 [00:01<00:03, 33.76 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:03, 33.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:02, 33.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:02, 33.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:02, 33.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:02, 33.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  40%|████      | 65/162 [00:01<00:02, 37.15 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:02, 37.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:02, 37.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:02, 37.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:02, 37.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:02, 37.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:02, 37.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 41.82 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 41.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 41.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 41.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 41.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 41.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 41.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 41.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  48%|████▊     | 78/162 [00:01<00:01, 47.10 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:01, 47.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:01, 47.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:01, 47.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:01, 47.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:01, 47.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:01, 47.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:01, 47.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 47.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 52.14 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 52.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 52.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 52.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 52.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 52.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 52.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:02<00:01, 52.14 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  57%|█████▋    | 93/162 [00:02<00:01, 56.10 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:02<00:01, 56.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:02<00:01, 56.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:02<00:01, 56.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:02<00:01, 56.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:02<00:01, 56.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:02<00:01, 56.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:02<00:01, 56.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  62%|██████▏   | 100/162 [00:02<00:01, 59.57 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:02<00:01, 59.57 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:02<00:01, 59.57 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:02<00:01, 59.57 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:02<00:00, 59.57 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:02<00:00, 59.57 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:02<00:00, 59.57 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:02<00:00, 59.57 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  66%|██████▌   | 107/162 [00:02<00:00, 62.08 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:02<00:00, 62.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:02<00:00, 62.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:02<00:00, 62.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:02<00:00, 62.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:00, 62.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:00, 62.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:00, 62.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 63.08 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 63.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 63.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 63.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 63.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 63.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 63.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 63.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 64.96 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 64.96 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 64.96 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 64.96 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 64.96 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 64.96 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 64.96 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 64.96 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 64.96 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 65.95 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 65.95 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 65.95 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 65.95 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 65.95 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 65.95 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 65.95 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 65.95 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 65.95 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 67.06 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 67.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 67.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 67.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 67.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 67.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 67.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 67.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 67.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 67.97 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 67.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 67.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 67.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 67.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 67.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 67.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 67.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 68.24 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 68.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 68.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 68.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 68.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 68.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 68.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 68.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 67.29 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 67.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:03<00:00, 67.29 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:03<00:00, 67.29 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 67.29 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.03s/ url]Dl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.03s/ url]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 67.29 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.03s/ url]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 67.29 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:03<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.07s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:05<00:00,  3.03s/ url]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 67.29 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.07s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.07s/ file]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 31.95 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:05<00:00,  5.07s/ url]
0 examples [00:00, ? examples/s]2020-06-16 00:08:31.625905: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-16 00:08:31.691394: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-06-16 00:08:31.691617: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55cbf0d732d0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-16 00:08:31.691640: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  1.78 examples/s]106 examples [00:00,  2.54 examples/s]204 examples [00:00,  3.63 examples/s]305 examples [00:00,  5.18 examples/s]404 examples [00:00,  7.38 examples/s]509 examples [00:01, 10.52 examples/s]610 examples [00:01, 14.96 examples/s]708 examples [00:01, 21.23 examples/s]810 examples [00:01, 30.05 examples/s]916 examples [00:01, 42.41 examples/s]1022 examples [00:01, 59.56 examples/s]1123 examples [00:01, 82.99 examples/s]1224 examples [00:01, 114.33 examples/s]1324 examples [00:01, 155.55 examples/s]1423 examples [00:01, 207.76 examples/s]1522 examples [00:02, 272.15 examples/s]1621 examples [00:02, 347.51 examples/s]1723 examples [00:02, 432.77 examples/s]1833 examples [00:02, 528.44 examples/s]1940 examples [00:02, 622.68 examples/s]2049 examples [00:02, 713.74 examples/s]2154 examples [00:02, 784.01 examples/s]2258 examples [00:02, 840.34 examples/s]2362 examples [00:02, 887.93 examples/s]2469 examples [00:02, 935.34 examples/s]2576 examples [00:03, 971.39 examples/s]2683 examples [00:03, 998.08 examples/s]2789 examples [00:03, 999.44 examples/s]2896 examples [00:03, 1019.08 examples/s]3001 examples [00:03, 1011.94 examples/s]3105 examples [00:03, 994.28 examples/s] 3206 examples [00:03, 993.10 examples/s]3307 examples [00:03, 986.81 examples/s]3407 examples [00:03, 970.66 examples/s]3511 examples [00:04, 989.98 examples/s]3618 examples [00:04, 1010.88 examples/s]3726 examples [00:04, 1029.97 examples/s]3830 examples [00:04, 1027.45 examples/s]3939 examples [00:04, 1042.73 examples/s]4046 examples [00:04, 1048.05 examples/s]4151 examples [00:04, 1048.29 examples/s]4260 examples [00:04, 1058.84 examples/s]4366 examples [00:04, 1030.96 examples/s]4472 examples [00:04, 1038.72 examples/s]4577 examples [00:05, 1017.65 examples/s]4685 examples [00:05, 1032.91 examples/s]4789 examples [00:05, 961.36 examples/s] 4887 examples [00:05, 961.79 examples/s]4991 examples [00:05, 982.25 examples/s]5098 examples [00:05, 1005.62 examples/s]5202 examples [00:05, 1014.57 examples/s]5304 examples [00:05, 1008.73 examples/s]5406 examples [00:05, 1009.00 examples/s]5508 examples [00:05, 1010.19 examples/s]5615 examples [00:06, 1025.92 examples/s]5718 examples [00:06, 1000.49 examples/s]5819 examples [00:06, 981.46 examples/s] 5918 examples [00:06, 974.00 examples/s]6020 examples [00:06, 987.08 examples/s]6122 examples [00:06, 996.22 examples/s]6224 examples [00:06, 1002.62 examples/s]6326 examples [00:06, 1006.33 examples/s]6427 examples [00:06, 986.34 examples/s] 6526 examples [00:07, 948.85 examples/s]6623 examples [00:07, 954.39 examples/s]6726 examples [00:07, 974.90 examples/s]6827 examples [00:07, 983.21 examples/s]6929 examples [00:07, 992.27 examples/s]7034 examples [00:07, 1007.11 examples/s]7137 examples [00:07, 1012.53 examples/s]7243 examples [00:07, 1026.13 examples/s]7346 examples [00:07, 1009.74 examples/s]7448 examples [00:07, 1012.38 examples/s]7556 examples [00:08, 1030.34 examples/s]7660 examples [00:08, 997.78 examples/s] 7761 examples [00:08, 988.99 examples/s]7861 examples [00:08, 954.65 examples/s]7957 examples [00:08, 955.97 examples/s]8060 examples [00:08, 975.12 examples/s]8165 examples [00:08, 993.49 examples/s]8265 examples [00:08, 981.95 examples/s]8364 examples [00:08, 976.40 examples/s]8462 examples [00:08, 976.12 examples/s]8573 examples [00:09, 1012.16 examples/s]8677 examples [00:09, 1018.05 examples/s]8782 examples [00:09, 1026.52 examples/s]8885 examples [00:09, 1020.80 examples/s]8988 examples [00:09, 1021.90 examples/s]9094 examples [00:09, 1030.42 examples/s]9200 examples [00:09, 1036.90 examples/s]9307 examples [00:09, 1046.40 examples/s]9412 examples [00:09, 1019.54 examples/s]9515 examples [00:09, 1014.02 examples/s]9618 examples [00:10, 1018.03 examples/s]9720 examples [00:10, 1011.22 examples/s]9822 examples [00:10, 1007.65 examples/s]9923 examples [00:10, 947.92 examples/s] 10019 examples [00:10, 897.15 examples/s]10123 examples [00:10, 935.24 examples/s]10229 examples [00:10, 968.48 examples/s]10327 examples [00:10, 946.90 examples/s]10425 examples [00:10, 955.78 examples/s]10533 examples [00:11, 987.73 examples/s]10641 examples [00:11, 1011.89 examples/s]10743 examples [00:11, 1009.01 examples/s]10845 examples [00:11, 989.97 examples/s] 10945 examples [00:11, 963.01 examples/s]11046 examples [00:11, 976.60 examples/s]11150 examples [00:11, 992.48 examples/s]11259 examples [00:11, 1017.92 examples/s]11365 examples [00:11, 1027.68 examples/s]11469 examples [00:11, 1001.62 examples/s]11570 examples [00:12, 991.64 examples/s] 11673 examples [00:12, 1002.77 examples/s]11774 examples [00:12, 996.01 examples/s] 11877 examples [00:12, 1003.71 examples/s]11978 examples [00:12, 1000.75 examples/s]12083 examples [00:12, 1012.77 examples/s]12187 examples [00:12, 1019.62 examples/s]12290 examples [00:12, 1012.51 examples/s]12395 examples [00:12, 1022.11 examples/s]12498 examples [00:12, 1022.05 examples/s]12607 examples [00:13, 1040.84 examples/s]12715 examples [00:13, 1051.54 examples/s]12823 examples [00:13, 1059.49 examples/s]12930 examples [00:13, 1059.05 examples/s]13036 examples [00:13, 1047.35 examples/s]13141 examples [00:13, 1028.85 examples/s]13245 examples [00:13, 1015.70 examples/s]13347 examples [00:13, 993.04 examples/s] 13447 examples [00:13, 985.49 examples/s]13546 examples [00:14, 984.22 examples/s]13650 examples [00:14, 998.14 examples/s]13756 examples [00:14, 1014.20 examples/s]13861 examples [00:14, 1022.22 examples/s]13964 examples [00:14, 1014.51 examples/s]14066 examples [00:14, 993.71 examples/s] 14170 examples [00:14, 1006.70 examples/s]14276 examples [00:14, 1019.44 examples/s]14379 examples [00:14, 994.02 examples/s] 14479 examples [00:14, 986.98 examples/s]14578 examples [00:15, 958.90 examples/s]14683 examples [00:15, 983.27 examples/s]14782 examples [00:15, 980.01 examples/s]14884 examples [00:15, 990.89 examples/s]14984 examples [00:15, 990.49 examples/s]15091 examples [00:15, 1010.89 examples/s]15195 examples [00:15, 1017.13 examples/s]15302 examples [00:15, 1030.12 examples/s]15411 examples [00:15, 1045.00 examples/s]15516 examples [00:15, 1042.13 examples/s]15623 examples [00:16, 1047.87 examples/s]15728 examples [00:16, 1041.36 examples/s]15834 examples [00:16, 1046.16 examples/s]15945 examples [00:16, 1063.18 examples/s]16052 examples [00:16, 1048.72 examples/s]16157 examples [00:16, 1045.78 examples/s]16264 examples [00:16, 1052.35 examples/s]16370 examples [00:16, 1051.46 examples/s]16476 examples [00:16, 1046.67 examples/s]16581 examples [00:16, 1002.22 examples/s]16685 examples [00:17, 1012.64 examples/s]16792 examples [00:17, 1028.06 examples/s]16900 examples [00:17, 1041.14 examples/s]17007 examples [00:17, 1048.94 examples/s]17113 examples [00:17, 1043.81 examples/s]17218 examples [00:17, 1026.38 examples/s]17322 examples [00:17, 1028.79 examples/s]17425 examples [00:17, 1023.56 examples/s]17528 examples [00:17, 1017.20 examples/s]17630 examples [00:18, 1007.24 examples/s]17731 examples [00:18, 1005.98 examples/s]17836 examples [00:18, 1018.29 examples/s]17945 examples [00:18, 1036.77 examples/s]18049 examples [00:18, 1022.50 examples/s]18152 examples [00:18, 1017.00 examples/s]18254 examples [00:18, 1004.75 examples/s]18356 examples [00:18, 1007.60 examples/s]18463 examples [00:18, 1023.29 examples/s]18569 examples [00:18, 1033.62 examples/s]18673 examples [00:19, 1027.01 examples/s]18776 examples [00:19, 1020.95 examples/s]18882 examples [00:19, 1031.02 examples/s]18987 examples [00:19, 1034.52 examples/s]19091 examples [00:19, 1022.39 examples/s]19194 examples [00:19, 980.51 examples/s] 19301 examples [00:19, 1005.35 examples/s]19410 examples [00:19, 1027.78 examples/s]19514 examples [00:19, 1017.12 examples/s]19617 examples [00:19, 1010.44 examples/s]19719 examples [00:20, 1002.26 examples/s]19826 examples [00:20, 1020.86 examples/s]19929 examples [00:20, 1010.48 examples/s]20031 examples [00:20, 940.79 examples/s] 20137 examples [00:20, 973.42 examples/s]20236 examples [00:20, 964.48 examples/s]20335 examples [00:20, 971.01 examples/s]20434 examples [00:20, 975.04 examples/s]20532 examples [00:20, 953.82 examples/s]20638 examples [00:21, 981.96 examples/s]20737 examples [00:21, 983.51 examples/s]20840 examples [00:21, 994.14 examples/s]20941 examples [00:21, 996.62 examples/s]21042 examples [00:21, 999.34 examples/s]21147 examples [00:21, 1013.38 examples/s]21249 examples [00:21, 1002.63 examples/s]21352 examples [00:21, 1009.46 examples/s]21454 examples [00:21, 1004.00 examples/s]21555 examples [00:21, 1002.61 examples/s]21656 examples [00:22, 987.75 examples/s] 21762 examples [00:22, 1006.12 examples/s]21863 examples [00:22, 1001.84 examples/s]21971 examples [00:22, 1023.11 examples/s]22075 examples [00:22, 1023.08 examples/s]22178 examples [00:22, 1016.95 examples/s]22282 examples [00:22, 1021.46 examples/s]22386 examples [00:22, 1025.96 examples/s]22489 examples [00:22, 1023.87 examples/s]22592 examples [00:22, 1013.66 examples/s]22695 examples [00:23, 1017.02 examples/s]22801 examples [00:23, 1026.58 examples/s]22909 examples [00:23, 1041.92 examples/s]23014 examples [00:23, 1041.88 examples/s]23122 examples [00:23, 1052.44 examples/s]23228 examples [00:23, 1041.78 examples/s]23333 examples [00:23, 1019.73 examples/s]23436 examples [00:23, 1004.79 examples/s]23544 examples [00:23, 1026.03 examples/s]23651 examples [00:23, 1036.56 examples/s]23755 examples [00:24, 965.38 examples/s] 23867 examples [00:24, 1005.93 examples/s]23969 examples [00:24, 990.72 examples/s] 24079 examples [00:24, 1018.16 examples/s]24190 examples [00:24, 1042.26 examples/s]24295 examples [00:24, 1014.45 examples/s]24399 examples [00:24, 1020.19 examples/s]24503 examples [00:24, 1016.48 examples/s]24608 examples [00:24, 1025.37 examples/s]24711 examples [00:25, 1005.76 examples/s]24812 examples [00:25, 980.54 examples/s] 24918 examples [00:25, 1001.06 examples/s]25019 examples [00:25, 988.19 examples/s] 25120 examples [00:25, 993.56 examples/s]25225 examples [00:25, 1009.24 examples/s]25329 examples [00:25, 1016.20 examples/s]25431 examples [00:25, 1003.13 examples/s]25533 examples [00:25, 1004.40 examples/s]25634 examples [00:25, 992.86 examples/s] 25734 examples [00:26, 991.70 examples/s]25837 examples [00:26, 1000.01 examples/s]25942 examples [00:26, 1012.93 examples/s]26045 examples [00:26, 1014.71 examples/s]26149 examples [00:26, 1021.14 examples/s]26252 examples [00:26, 995.51 examples/s] 26352 examples [00:26, 974.57 examples/s]26450 examples [00:26, 962.27 examples/s]26553 examples [00:26, 978.66 examples/s]26652 examples [00:26, 976.43 examples/s]26750 examples [00:27, 965.90 examples/s]26852 examples [00:27, 981.42 examples/s]26957 examples [00:27, 1000.98 examples/s]27060 examples [00:27, 1007.38 examples/s]27161 examples [00:27, 1005.57 examples/s]27262 examples [00:27, 994.38 examples/s] 27366 examples [00:27, 1006.25 examples/s]27474 examples [00:27, 1026.88 examples/s]27582 examples [00:27, 1038.74 examples/s]27690 examples [00:27, 1050.15 examples/s]27796 examples [00:28, 1037.83 examples/s]27907 examples [00:28, 1055.68 examples/s]28015 examples [00:28, 1060.34 examples/s]28122 examples [00:28, 1057.77 examples/s]28228 examples [00:28, 1039.86 examples/s]28333 examples [00:28, 1010.78 examples/s]28435 examples [00:28, 985.19 examples/s] 28538 examples [00:28, 995.42 examples/s]28638 examples [00:28, 993.25 examples/s]28738 examples [00:29, 980.82 examples/s]28837 examples [00:29, 975.34 examples/s]28939 examples [00:29, 987.07 examples/s]29039 examples [00:29, 988.25 examples/s]29138 examples [00:29, 985.05 examples/s]29244 examples [00:29, 1005.50 examples/s]29348 examples [00:29, 1011.84 examples/s]29451 examples [00:29, 1015.63 examples/s]29554 examples [00:29, 1019.80 examples/s]29662 examples [00:29, 1036.59 examples/s]29767 examples [00:30, 1039.59 examples/s]29872 examples [00:30, 1018.43 examples/s]29974 examples [00:30, 1006.79 examples/s]30075 examples [00:30, 944.03 examples/s] 30182 examples [00:30, 976.54 examples/s]30283 examples [00:30, 985.99 examples/s]30384 examples [00:30, 991.50 examples/s]30489 examples [00:30, 1006.39 examples/s]30595 examples [00:30, 1020.74 examples/s]30698 examples [00:30, 995.50 examples/s] 30804 examples [00:31, 1012.42 examples/s]30906 examples [00:31, 1001.54 examples/s]31007 examples [00:31, 986.21 examples/s] 31115 examples [00:31, 1011.13 examples/s]31221 examples [00:31, 1024.91 examples/s]31329 examples [00:31, 1040.42 examples/s]31434 examples [00:31, 1024.54 examples/s]31541 examples [00:31, 1035.32 examples/s]31645 examples [00:31, 1014.46 examples/s]31747 examples [00:32, 1011.37 examples/s]31851 examples [00:32, 1018.26 examples/s]31953 examples [00:32, 1016.23 examples/s]32055 examples [00:32, 1013.53 examples/s]32158 examples [00:32, 1018.34 examples/s]32261 examples [00:32, 1021.46 examples/s]32366 examples [00:32, 1027.82 examples/s]32469 examples [00:32, 1002.34 examples/s]32570 examples [00:32, 1003.66 examples/s]32671 examples [00:32, 1000.60 examples/s]32772 examples [00:33, 1003.17 examples/s]32876 examples [00:33, 1012.56 examples/s]32978 examples [00:33, 1007.87 examples/s]33082 examples [00:33, 1016.23 examples/s]33184 examples [00:33, 1015.32 examples/s]33287 examples [00:33, 1018.95 examples/s]33389 examples [00:33, 1004.46 examples/s]33490 examples [00:33, 987.15 examples/s] 33594 examples [00:33, 999.97 examples/s]33701 examples [00:33, 1017.86 examples/s]33808 examples [00:34, 1031.54 examples/s]33919 examples [00:34, 1053.80 examples/s]34025 examples [00:34, 1048.91 examples/s]34132 examples [00:34, 1053.07 examples/s]34238 examples [00:34, 1053.41 examples/s]34344 examples [00:34, 1050.92 examples/s]34450 examples [00:34, 1052.79 examples/s]34556 examples [00:34, 1051.17 examples/s]34662 examples [00:34, 1012.25 examples/s]34764 examples [00:34, 1006.06 examples/s]34865 examples [00:35, 939.09 examples/s] 34960 examples [00:35, 938.48 examples/s]35055 examples [00:35, 897.95 examples/s]35146 examples [00:35, 867.20 examples/s]35243 examples [00:35, 894.78 examples/s]35335 examples [00:35, 901.90 examples/s]35434 examples [00:35, 925.78 examples/s]35528 examples [00:35, 927.42 examples/s]35622 examples [00:35, 903.50 examples/s]35727 examples [00:36, 941.93 examples/s]35827 examples [00:36, 956.60 examples/s]35934 examples [00:36, 987.75 examples/s]36034 examples [00:36, 974.71 examples/s]36132 examples [00:36, 957.81 examples/s]36240 examples [00:36, 989.79 examples/s]36348 examples [00:36, 1012.30 examples/s]36455 examples [00:36, 1026.64 examples/s]36559 examples [00:36, 1004.43 examples/s]36661 examples [00:36, 1006.73 examples/s]36763 examples [00:37, 1008.01 examples/s]36867 examples [00:37, 1017.21 examples/s]36976 examples [00:37, 1035.43 examples/s]37080 examples [00:37, 1026.21 examples/s]37187 examples [00:37, 1037.84 examples/s]37295 examples [00:37, 1048.51 examples/s]37400 examples [00:37, 1033.92 examples/s]37508 examples [00:37, 1046.32 examples/s]37613 examples [00:37, 1023.98 examples/s]37721 examples [00:37, 1038.69 examples/s]37830 examples [00:38, 1051.07 examples/s]37938 examples [00:38, 1059.36 examples/s]38046 examples [00:38, 1062.73 examples/s]38153 examples [00:38, 1044.23 examples/s]38258 examples [00:38, 1044.16 examples/s]38363 examples [00:38, 1044.66 examples/s]38468 examples [00:38, 1039.14 examples/s]38572 examples [00:38, 1021.98 examples/s]38675 examples [00:38, 1007.45 examples/s]38776 examples [00:39, 995.20 examples/s] 38876 examples [00:39, 991.73 examples/s]38981 examples [00:39, 1008.31 examples/s]39087 examples [00:39, 1021.86 examples/s]39190 examples [00:39, 1017.74 examples/s]39296 examples [00:39, 1028.59 examples/s]39403 examples [00:39, 1038.47 examples/s]39509 examples [00:39, 1043.53 examples/s]39614 examples [00:39, 1027.76 examples/s]39717 examples [00:39, 1016.27 examples/s]39822 examples [00:40, 1023.39 examples/s]39925 examples [00:40, 996.90 examples/s] 40025 examples [00:40, 957.04 examples/s]40124 examples [00:40, 966.13 examples/s]40221 examples [00:40, 965.91 examples/s]40326 examples [00:40, 987.29 examples/s]40426 examples [00:40, 980.17 examples/s]40525 examples [00:40, 973.47 examples/s]40623 examples [00:40, 973.99 examples/s]40721 examples [00:40, 957.85 examples/s]40824 examples [00:41, 977.98 examples/s]40932 examples [00:41, 1005.22 examples/s]41033 examples [00:41, 1005.65 examples/s]41138 examples [00:41, 1016.49 examples/s]41240 examples [00:41, 1001.69 examples/s]41341 examples [00:41, 991.82 examples/s] 41446 examples [00:41, 1006.81 examples/s]41553 examples [00:41, 1022.92 examples/s]41660 examples [00:41, 1035.12 examples/s]41764 examples [00:41, 968.16 examples/s] 41866 examples [00:42, 980.36 examples/s]41967 examples [00:42, 988.50 examples/s]42068 examples [00:42, 993.78 examples/s]42168 examples [00:42, 985.34 examples/s]42267 examples [00:42, 979.52 examples/s]42367 examples [00:42, 983.80 examples/s]42466 examples [00:42, 983.00 examples/s]42565 examples [00:42, 975.21 examples/s]42663 examples [00:42, 975.59 examples/s]42766 examples [00:43, 989.28 examples/s]42870 examples [00:43, 1003.44 examples/s]42971 examples [00:43, 974.70 examples/s] 43072 examples [00:43, 984.17 examples/s]43171 examples [00:43, 982.62 examples/s]43272 examples [00:43, 990.01 examples/s]43373 examples [00:43, 995.92 examples/s]43473 examples [00:43, 978.68 examples/s]43571 examples [00:43, 972.69 examples/s]43669 examples [00:43, 955.20 examples/s]43767 examples [00:44, 960.53 examples/s]43869 examples [00:44, 974.82 examples/s]43972 examples [00:44, 988.99 examples/s]44078 examples [00:44, 1007.29 examples/s]44179 examples [00:44, 1002.83 examples/s]44287 examples [00:44, 1023.44 examples/s]44395 examples [00:44, 1037.79 examples/s]44501 examples [00:44, 1042.77 examples/s]44607 examples [00:44, 1047.81 examples/s]44712 examples [00:44, 1029.19 examples/s]44816 examples [00:45, 998.82 examples/s] 44923 examples [00:45, 1018.40 examples/s]45029 examples [00:45, 1028.21 examples/s]45133 examples [00:45, 1025.57 examples/s]45236 examples [00:45, 987.43 examples/s] 45340 examples [00:45, 1000.99 examples/s]45448 examples [00:45, 1022.55 examples/s]45555 examples [00:45, 1035.20 examples/s]45660 examples [00:45, 1038.43 examples/s]45765 examples [00:45, 1029.51 examples/s]45873 examples [00:46, 1043.03 examples/s]45980 examples [00:46, 1049.08 examples/s]46086 examples [00:46, 1049.40 examples/s]46192 examples [00:46, 1028.53 examples/s]46296 examples [00:46, 1014.89 examples/s]46401 examples [00:46, 1023.24 examples/s]46504 examples [00:46, 1015.54 examples/s]46606 examples [00:46, 1013.38 examples/s]46708 examples [00:46, 1011.75 examples/s]46810 examples [00:47, 966.33 examples/s] 46909 examples [00:47, 972.70 examples/s]47011 examples [00:47, 984.31 examples/s]47115 examples [00:47, 998.03 examples/s]47221 examples [00:47, 1012.88 examples/s]47324 examples [00:47, 1016.99 examples/s]47426 examples [00:47, 1017.29 examples/s]47532 examples [00:47, 1029.14 examples/s]47636 examples [00:47, 1031.19 examples/s]47740 examples [00:47, 1032.27 examples/s]47844 examples [00:48, 1017.85 examples/s]47946 examples [00:48, 996.88 examples/s] 48051 examples [00:48, 1011.53 examples/s]48154 examples [00:48, 1015.53 examples/s]48256 examples [00:48, 1004.73 examples/s]48357 examples [00:48, 990.31 examples/s] 48459 examples [00:48, 997.47 examples/s]48559 examples [00:48, 977.58 examples/s]48657 examples [00:48, 965.53 examples/s]48758 examples [00:48, 976.60 examples/s]48856 examples [00:49, 970.87 examples/s]48967 examples [00:49, 1008.40 examples/s]49075 examples [00:49, 1027.28 examples/s]49179 examples [00:49, 1025.71 examples/s]49284 examples [00:49, 1030.90 examples/s]49388 examples [00:49, 1029.98 examples/s]49498 examples [00:49, 1048.96 examples/s]49606 examples [00:49, 1056.93 examples/s]49712 examples [00:49, 1044.50 examples/s]49817 examples [00:49, 1015.19 examples/s]49919 examples [00:50, 1014.80 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 18%|█▊        | 9072/50000 [00:00<00:00, 90717.06 examples/s] 45%|████▍     | 22286/50000 [00:00<00:00, 100133.90 examples/s] 69%|██████▉   | 34692/50000 [00:00<00:00, 106282.71 examples/s] 95%|█████████▌| 47619/50000 [00:00<00:00, 112271.30 examples/s]                                                                0 examples [00:00, ? examples/s]87 examples [00:00, 867.27 examples/s]195 examples [00:00, 921.33 examples/s]293 examples [00:00, 936.71 examples/s]387 examples [00:00, 935.33 examples/s]488 examples [00:00, 954.26 examples/s]589 examples [00:00, 968.48 examples/s]701 examples [00:00, 1008.39 examples/s]806 examples [00:00, 1018.21 examples/s]913 examples [00:00, 1031.04 examples/s]1013 examples [00:01, 1011.64 examples/s]1118 examples [00:01, 1022.74 examples/s]1223 examples [00:01, 1030.58 examples/s]1326 examples [00:01, 1018.28 examples/s]1428 examples [00:01, 813.74 examples/s] 1531 examples [00:01, 868.41 examples/s]1638 examples [00:01, 918.48 examples/s]1745 examples [00:01, 958.48 examples/s]1852 examples [00:01, 987.74 examples/s]1954 examples [00:01, 993.49 examples/s]2058 examples [00:02, 1006.28 examples/s]2161 examples [00:02, 966.80 examples/s] 2259 examples [00:02, 926.43 examples/s]2353 examples [00:02, 916.46 examples/s]2446 examples [00:02, 904.99 examples/s]2552 examples [00:02, 946.23 examples/s]2657 examples [00:02, 973.00 examples/s]2763 examples [00:02, 997.10 examples/s]2871 examples [00:02, 1018.54 examples/s]2974 examples [00:03, 1011.40 examples/s]3076 examples [00:03, 1006.24 examples/s]3177 examples [00:03, 1002.10 examples/s]3282 examples [00:03, 1013.08 examples/s]3385 examples [00:03, 1017.02 examples/s]3488 examples [00:03, 1018.53 examples/s]3590 examples [00:03, 1003.03 examples/s]3700 examples [00:03, 1028.62 examples/s]3808 examples [00:03, 1041.44 examples/s]3914 examples [00:03, 1046.35 examples/s]4020 examples [00:04, 1048.32 examples/s]4128 examples [00:04, 1057.22 examples/s]4234 examples [00:04, 1054.97 examples/s]4340 examples [00:04, 1027.18 examples/s]4443 examples [00:04, 1025.66 examples/s]4546 examples [00:04, 1016.31 examples/s]4657 examples [00:04, 1041.14 examples/s]4762 examples [00:04, 1036.42 examples/s]4866 examples [00:04, 1029.45 examples/s]4974 examples [00:04, 1043.27 examples/s]5079 examples [00:05, 1020.35 examples/s]5185 examples [00:05, 1029.67 examples/s]5289 examples [00:05, 1009.99 examples/s]5391 examples [00:05, 1001.03 examples/s]5492 examples [00:05, 941.08 examples/s] 5587 examples [00:05, 934.78 examples/s]5690 examples [00:05, 960.75 examples/s]5790 examples [00:05, 972.19 examples/s]5893 examples [00:05, 986.17 examples/s]5992 examples [00:06, 982.77 examples/s]6091 examples [00:06, 976.38 examples/s]6196 examples [00:06, 995.11 examples/s]6296 examples [00:06, 977.94 examples/s]6395 examples [00:06, 960.71 examples/s]6492 examples [00:06, 946.33 examples/s]6589 examples [00:06, 952.69 examples/s]6688 examples [00:06, 962.65 examples/s]6794 examples [00:06, 988.52 examples/s]6900 examples [00:06, 1006.97 examples/s]7007 examples [00:07, 1022.73 examples/s]7110 examples [00:07, 991.17 examples/s] 7210 examples [00:07, 955.94 examples/s]7308 examples [00:07, 962.77 examples/s]7413 examples [00:07, 984.93 examples/s]7512 examples [00:07, 978.22 examples/s]7611 examples [00:07, 968.77 examples/s]7710 examples [00:07, 971.86 examples/s]7811 examples [00:07, 981.16 examples/s]7910 examples [00:07, 972.25 examples/s]8013 examples [00:08, 987.17 examples/s]8112 examples [00:08, 987.45 examples/s]8214 examples [00:08, 992.59 examples/s]8315 examples [00:08, 996.90 examples/s]8420 examples [00:08, 1010.21 examples/s]8527 examples [00:08, 1026.82 examples/s]8630 examples [00:08, 1016.11 examples/s]8733 examples [00:08, 1019.70 examples/s]8838 examples [00:08, 1027.51 examples/s]8943 examples [00:09, 1031.46 examples/s]9047 examples [00:09, 1030.02 examples/s]9151 examples [00:09, 989.91 examples/s] 9253 examples [00:09, 997.51 examples/s]9354 examples [00:09, 954.36 examples/s]9458 examples [00:09, 977.25 examples/s]9564 examples [00:09, 999.35 examples/s]9665 examples [00:09, 980.15 examples/s]9764 examples [00:09, 970.17 examples/s]9863 examples [00:09, 973.72 examples/s]9962 examples [00:10, 978.15 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteVC89B6/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteVC89B6/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f4f690dd950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f4f690dd950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f4f690dd950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f4eee540cc0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f4eee557080>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f4f690dd950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f4f690dd950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f4f690dd950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f4f4f49d0f0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f4f4f49d048>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f4ee0b17400> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f4ee0b17400> 

  function with postional parmater data_info <function split_train_valid at 0x7f4ee0b17400> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f4ee0b17510> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f4ee0b17510> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f4ee0b17510> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.1)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.5)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.2)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.1)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=030eb7c9815e480b1212b501dcc62ab046d446359f6d24486d72867700a8c9c1
  Stored in directory: /tmp/pip-ephem-wheel-cache-_jxkz8ko/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
Successfully built en-core-web-sm
Installing collected packages: en-core-web-sm
Successfully installed en-core-web-sm-2.2.5
[38;5;2m✔ Download and installation successful[0m
You can now load the model via spacy.load('en_core_web_sm')
[38;5;2m✔ Linking successful[0m
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/en_core_web_sm
-->
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/data/en
You can now load the model via spacy.load('en')
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:01<49:58:17, 4.79kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:01<35:12:40, 6.80kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<24:41:56, 9.69kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:02<17:17:22, 13.8kB/s].vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:02<12:04:03, 19.8kB/s].vector_cache/glove.6B.zip:   1%|          | 9.14M/862M [00:02<8:23:40, 28.2kB/s] .vector_cache/glove.6B.zip:   1%|▏         | 12.2M/862M [00:02<5:51:27, 40.3kB/s].vector_cache/glove.6B.zip:   2%|▏         | 16.8M/862M [00:02<4:04:47, 57.6kB/s].vector_cache/glove.6B.zip:   2%|▏         | 20.7M/862M [00:02<2:50:39, 82.2kB/s].vector_cache/glove.6B.zip:   3%|▎         | 25.2M/862M [00:02<1:58:55, 117kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 29.5M/862M [00:02<1:22:54, 167kB/s].vector_cache/glove.6B.zip:   4%|▍         | 33.9M/862M [00:02<57:50, 239kB/s]  .vector_cache/glove.6B.zip:   4%|▍         | 38.0M/862M [00:03<40:22, 340kB/s].vector_cache/glove.6B.zip:   5%|▍         | 42.4M/862M [00:03<28:12, 484kB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.6M/862M [00:03<19:44, 688kB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.0M/862M [00:03<13:50, 977kB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.5M/862M [00:03<11:19, 1.19MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.6M/862M [00:05<09:48, 1.37MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.8M/862M [00:06<09:03, 1.48MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.8M/862M [00:06<06:52, 1.95MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.7M/862M [00:07<07:09, 1.87MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.1M/862M [00:08<06:22, 2.09MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.7M/862M [00:08<04:44, 2.82MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.8M/862M [00:09<06:29, 2.05MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.2M/862M [00:10<05:54, 2.25MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.8M/862M [00:10<04:24, 3.01MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.9M/862M [00:11<06:11, 2.14MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.1M/862M [00:11<07:00, 1.89MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.9M/862M [00:12<05:30, 2.39MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.0M/862M [00:12<03:58, 3.31MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.1M/862M [00:13<3:16:22, 67.0kB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.4M/862M [00:13<2:18:45, 94.7kB/s].vector_cache/glove.6B.zip:   9%|▊         | 75.0M/862M [00:14<1:37:14, 135kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 77.2M/862M [00:15<1:10:57, 184kB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.6M/862M [00:15<50:58, 257kB/s]  .vector_cache/glove.6B.zip:   9%|▉         | 79.1M/862M [00:16<35:56, 363kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.3M/862M [00:17<28:11, 462kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.5M/862M [00:17<22:23, 581kB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.3M/862M [00:17<16:13, 801kB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.8M/862M [00:18<11:28, 1.13MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.4M/862M [00:19<17:57, 721kB/s] .vector_cache/glove.6B.zip:  10%|▉         | 85.8M/862M [00:19<13:55, 929kB/s].vector_cache/glove.6B.zip:  10%|█         | 87.3M/862M [00:19<10:00, 1.29MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.5M/862M [00:21<10:00, 1.29MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.7M/862M [00:21<09:38, 1.34MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.5M/862M [00:21<07:23, 1.74MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.6M/862M [00:23<07:14, 1.77MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.0M/862M [00:23<06:23, 2.00MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.6M/862M [00:23<04:47, 2.67MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.7M/862M [00:25<06:18, 2.02MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.9M/862M [00:25<07:00, 1.82MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.7M/862M [00:25<05:32, 2.30MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:27<05:55, 2.14MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:27<05:14, 2.42MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:27<04:01, 3.14MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:27<02:58, 4.23MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:29<1:17:14, 163kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:29<56:37, 223kB/s]  .vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:29<40:13, 313kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:31<30:04, 417kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:31<22:20, 561kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:31<15:52, 787kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:33<14:01, 889kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:33<12:20, 1.01MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:33<09:11, 1.35MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:33<06:33, 1.89MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:35<12:54, 960kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:35<10:05, 1.23MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:35<07:26, 1.66MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:35<05:19, 2.31MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:37<53:25, 231kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:37<39:53, 309kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:37<28:24, 433kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:37<20:00, 614kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:39<19:47, 619kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:39<15:05, 812kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:39<10:48, 1.13MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:41<10:24, 1.17MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:41<08:31, 1.43MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:41<06:13, 1.95MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:43<07:14, 1.68MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:43<07:31, 1.61MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:43<05:52, 2.06MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:45<06:03, 1.99MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:45<05:28, 2.20MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:45<04:06, 2.93MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:47<05:40, 2.11MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:47<05:11, 2.31MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:47<03:53, 3.07MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:49<05:31, 2.16MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:49<05:04, 2.35MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:49<03:50, 3.09MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:51<05:29, 2.16MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:51<06:15, 1.90MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:51<04:58, 2.38MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:53<05:22, 2.19MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:53<04:59, 2.36MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:53<03:44, 3.14MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:54<05:19, 2.20MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:55<04:55, 2.38MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:55<03:41, 3.17MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<05:18, 2.19MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:57<04:55, 2.37MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:57<03:43, 3.11MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<05:20, 2.16MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:59<04:55, 2.35MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:59<03:41, 3.13MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<05:19, 2.16MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:01<04:53, 2.35MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:01<03:40, 3.12MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<05:17, 2.16MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<06:01, 1.90MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:03<04:44, 2.41MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<03:26, 3.30MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<11:49, 962kB/s] .vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<09:25, 1.21MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:05<06:52, 1.65MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:06<07:27, 1.51MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:06<07:30, 1.50MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:07<05:49, 1.94MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:08<05:53, 1.91MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<05:15, 2.14MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<03:57, 2.83MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:10<05:22, 2.08MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<06:07, 1.82MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<04:47, 2.32MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<03:28, 3.20MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:12<13:19, 832kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<10:28, 1.06MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<07:36, 1.45MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:14<07:52, 1.40MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:14<07:44, 1.42MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<05:58, 1.84MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:16<05:56, 1.84MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:16<05:17, 2.07MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<03:57, 2.76MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:18<05:19, 2.05MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:18<04:49, 2.26MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:18<03:38, 2.98MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:20<05:06, 2.11MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:20<04:39, 2.32MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<03:31, 3.06MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:22<05:00, 2.15MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:22<05:41, 1.89MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:22<04:31, 2.37MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:24<04:53, 2.19MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:24<04:31, 2.36MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<03:23, 3.14MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:26<04:51, 2.18MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:26<05:33, 1.91MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:26<04:25, 2.39MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:28<04:47, 2.20MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:28<04:27, 2.36MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<03:20, 3.14MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:30<04:47, 2.19MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:30<05:28, 1.91MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:30<04:18, 2.43MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<03:11, 3.28MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:32<05:51, 1.78MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:32<05:10, 2.01MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<03:50, 2.70MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:34<05:06, 2.02MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:34<05:40, 1.82MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:34<04:24, 2.34MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<03:14, 3.18MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:36<06:20, 1.62MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:36<05:18, 1.94MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:36<04:02, 2.54MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:36<02:55, 3.49MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:38<40:40, 251kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:38<30:34, 334kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:38<21:50, 467kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<15:26, 658kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:40<13:53, 730kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 254M/862M [01:40<10:46, 941kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<07:46, 1.30MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:42<07:46, 1.29MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 259M/862M [01:42<06:29, 1.55MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:42<04:47, 2.10MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:44<05:42, 1.75MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:44<05:01, 1.99MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:44<03:45, 2.65MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:45<04:58, 1.99MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:46<04:30, 2.20MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:46<03:21, 2.94MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<04:39, 2.11MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:48<04:17, 2.30MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<03:14, 3.03MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<04:34, 2.14MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:50<04:13, 2.32MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<03:09, 3.08MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<04:31, 2.15MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:52<04:08, 2.34MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:52<03:08, 3.08MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<04:28, 2.16MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<04:06, 2.35MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:54<03:06, 3.09MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:55<04:26, 2.16MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:55<04:04, 2.35MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:56<03:03, 3.13MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:57<04:23, 2.17MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:57<05:00, 1.90MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:58<03:55, 2.42MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<02:52, 3.29MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:59<06:16, 1.51MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<05:22, 1.76MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<03:59, 2.36MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:01<04:59, 1.88MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<04:26, 2.11MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<03:20, 2.80MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:03<04:32, 2.05MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:03<05:06, 1.82MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:03<04:02, 2.30MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<02:57, 3.13MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:05<06:19, 1.46MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:05<05:13, 1.77MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<03:50, 2.40MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:05<02:48, 3.26MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:07<25:04, 366kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:07<19:24, 472kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<14:02, 653kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:09<11:14, 810kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:09<08:48, 1.03MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<06:21, 1.43MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:11<06:33, 1.38MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:11<06:24, 1.41MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<04:56, 1.83MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:13<04:53, 1.83MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:13<04:20, 2.07MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<03:15, 2.74MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:15<04:21, 2.04MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:15<04:46, 1.86MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:15<03:47, 2.34MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:17<04:04, 2.17MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:17<03:45, 2.35MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:17<02:51, 3.09MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:19<04:02, 2.16MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:19<03:43, 2.35MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<02:49, 3.10MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:21<04:01, 2.16MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:21<04:35, 1.89MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:21<03:36, 2.41MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<02:37, 3.30MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:23<07:40, 1.12MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:23<06:14, 1.38MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<04:34, 1.88MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:25<05:11, 1.65MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:25<05:24, 1.58MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:25<04:10, 2.05MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<03:01, 2.81MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:27<06:25, 1.32MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:27<05:22, 1.58MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<03:55, 2.15MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:29<04:43, 1.78MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:29<05:02, 1.67MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:29<03:56, 2.13MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:31<04:06, 2.04MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:31<03:43, 2.24MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<02:48, 2.95MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:33<03:54, 2.12MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:33<03:33, 2.32MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<02:41, 3.06MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:35<03:49, 2.15MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:35<04:20, 1.89MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:35<03:23, 2.42MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<02:28, 3.29MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:36<05:20, 1.53MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:37<04:33, 1.79MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:37<03:23, 2.40MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<04:15, 1.90MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:39<03:48, 2.12MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:39<02:49, 2.84MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<03:51, 2.08MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:41<04:20, 1.84MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:41<03:23, 2.35MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<02:27, 3.24MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<10:22, 765kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:43<08:04, 982kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:43<05:50, 1.35MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:45<06:09, 1.28MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:45<04:26, 1.77MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:46<04:56, 1.58MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:46<04:05, 1.90MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:47<03:09, 2.46MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<02:17, 3.39MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:48<30:52, 250kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:48<23:11, 333kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:49<16:31, 467kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:49<11:40, 659kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:50<10:34, 725kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<08:11, 935kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<05:54, 1.29MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:52<05:53, 1.29MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<05:40, 1.34MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<04:21, 1.74MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<03:05, 2.43MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:54<57:05, 132kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:54<40:41, 185kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<28:33, 262kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:56<21:39, 345kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:56<16:37, 448kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<11:57, 623kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<08:27, 876kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:58<08:24, 878kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:58<06:38, 1.11MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<04:48, 1.53MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:00<05:04, 1.45MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:00<05:02, 1.45MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<03:54, 1.87MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:02<03:53, 1.87MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:02<03:20, 2.16MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<02:30, 2.89MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:02<01:50, 3.91MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:04<30:12, 238kB/s] .vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:04<21:51, 329kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<15:25, 464kB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:06<12:25, 573kB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:06<09:19, 763kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<06:38, 1.07MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:06<04:43, 1.49MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:08<36:08, 195kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:08<26:44, 264kB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:08<18:59, 370kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:08<13:18, 526kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:10<14:47, 472kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:10<11:03, 631kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<07:53, 881kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:12<07:07, 971kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:12<05:39, 1.22MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<04:07, 1.67MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:14<04:29, 1.52MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:14<04:31, 1.51MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 453M/862M [03:14<03:28, 1.97MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<02:32, 2.68MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:16<04:11, 1.62MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:16<03:38, 1.86MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<02:42, 2.49MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:18<03:26, 1.95MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:18<02:58, 2.25MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 461M/862M [03:18<02:15, 2.96MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:18<01:39, 4.02MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:20<27:52, 238kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:20<20:47, 319kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:20<14:50, 446kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<10:24, 632kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:22<12:56, 507kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:22<09:44, 674kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:22<06:57, 939kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:24<06:21, 1.02MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:24<05:46, 1.12MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:24<04:20, 1.50MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<03:05, 2.08MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:26<06:40, 964kB/s] .vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:26<05:19, 1.21MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:26<03:52, 1.65MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:28<04:11, 1.52MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:28<03:35, 1.77MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:28<02:38, 2.40MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:29<03:20, 1.89MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:30<02:51, 2.20MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:30<02:14, 2.79MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<01:37, 3.82MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<24:38, 253kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:32<17:51, 348kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<12:36, 491kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<10:14, 602kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:34<08:24, 732kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:34<06:09, 998kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<04:22, 1.40MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<05:53, 1.03MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<04:44, 1.28MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:36<03:26, 1.76MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:37<03:48, 1.58MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:37<03:53, 1.55MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:38<03:01, 1.99MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<02:10, 2.75MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:39<5:44:07, 17.3kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:39<4:01:14, 24.6kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:40<2:48:16, 35.2kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:41<1:58:26, 49.7kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<1:23:25, 70.5kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:42<58:16, 100kB/s]   .vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:43<41:55, 139kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:43<30:28, 191kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<21:34, 269kB/s].vector_cache/glove.6B.zip:  60%|██████    | 517M/862M [03:45<15:54, 361kB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<11:43, 490kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<08:17, 689kB/s].vector_cache/glove.6B.zip:  60%|██████    | 522M/862M [03:47<07:06, 799kB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:47<06:08, 925kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<04:34, 1.24MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:47<03:13, 1.74MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:49<19:28, 288kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:49<14:11, 395kB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:49<10:01, 557kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:51<08:16, 669kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:51<06:55, 800kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<05:06, 1.08MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:53<04:26, 1.23MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:53<03:39, 1.49MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<02:40, 2.03MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:55<03:07, 1.73MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:55<02:43, 1.98MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<02:00, 2.66MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:57<02:40, 1.99MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:57<02:20, 2.28MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<01:45, 3.00MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:59<02:28, 2.12MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:59<02:16, 2.32MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<01:42, 3.05MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:01<02:25, 2.14MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:01<02:13, 2.33MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<01:40, 3.07MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:03<02:22, 2.15MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:03<02:42, 1.89MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:03<02:09, 2.37MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:05<02:18, 2.19MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:05<02:07, 2.37MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 560M/862M [04:05<01:36, 3.12MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:07<02:17, 2.18MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:07<02:34, 1.93MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:07<02:01, 2.46MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<01:28, 3.35MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:09<03:14, 1.52MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:09<02:46, 1.77MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<02:02, 2.40MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:11<02:33, 1.90MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:11<02:46, 1.75MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:11<02:09, 2.25MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<01:32, 3.10MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:13<04:57, 966kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:13<03:56, 1.21MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<02:51, 1.66MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:15<03:05, 1.52MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:15<03:07, 1.51MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:15<02:23, 1.97MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<01:42, 2.73MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:17<11:20, 410kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:17<08:24, 552kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<05:58, 773kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:19<05:13, 878kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:19<04:34, 1.00MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:19<03:24, 1.34MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<02:24, 1.87MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:20<06:03, 745kB/s] .vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:21<04:42, 958kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:21<03:22, 1.33MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<03:23, 1.31MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:23<03:16, 1.36MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:23<02:30, 1.76MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<02:26, 1.79MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:25<02:09, 2.03MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<01:36, 2.70MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<02:07, 2.03MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<02:21, 1.82MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:27<01:50, 2.33MB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:27<01:19, 3.22MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:28<08:43, 485kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:28<06:32, 647kB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:29<04:38, 905kB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:30<04:12, 992kB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:30<03:21, 1.24MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:31<02:26, 1.69MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:32<02:40, 1.54MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<02:16, 1.80MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<01:40, 2.43MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:34<02:06, 1.91MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<01:53, 2.14MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<01:24, 2.83MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:36<01:55, 2.07MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:36<01:44, 2.27MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<01:17, 3.03MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:38<01:50, 2.12MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:38<02:04, 1.88MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:38<01:38, 2.36MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<01:10, 3.25MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:40<3:41:57, 17.2kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:40<2:35:30, 24.6kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<1:48:12, 35.1kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:42<1:15:53, 49.5kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:42<53:50, 69.7kB/s]  .vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<37:43, 99.2kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<26:11, 141kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:44<20:23, 181kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:44<14:34, 253kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<10:13, 358kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:46<07:56, 456kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:46<06:17, 575kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:46<04:34, 789kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:48<03:43, 952kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:48<02:58, 1.19MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:48<02:08, 1.65MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:50<02:18, 1.51MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:50<02:18, 1.50MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:50<01:46, 1.96MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<01:16, 2.70MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:52<02:42, 1.26MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:52<02:14, 1.52MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<01:38, 2.07MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:54<01:55, 1.74MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:54<01:40, 1.99MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<01:15, 2.65MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:56<01:38, 1.99MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:56<01:28, 2.21MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:56<01:05, 2.96MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:58<01:31, 2.10MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:58<01:23, 2.30MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<01:02, 3.07MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:00<01:27, 2.14MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:00<01:20, 2.34MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<00:59, 3.11MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:02<01:25, 2.16MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:02<01:37, 1.89MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:02<01:15, 2.42MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<00:54, 3.30MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:04<02:10, 1.39MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:04<01:49, 1.64MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<01:19, 2.24MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:06<01:36, 1.82MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:06<01:25, 2.06MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:06<01:03, 2.75MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:08<01:24, 2.03MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:08<01:34, 1.83MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:08<01:13, 2.33MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:08<00:52, 3.22MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:10<04:21, 642kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:10<03:19, 838kB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:10<02:22, 1.17MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:11<02:16, 1.20MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:12<02:08, 1.27MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:12<01:37, 1.66MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 703M/862M [05:13<01:33, 1.71MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:14<01:21, 1.95MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<01:00, 2.60MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<01:18, 1.98MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:16<01:26, 1.80MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:16<01:08, 2.27MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<00:48, 3.11MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<18:36, 136kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:18<13:15, 190kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:18<09:14, 269kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<06:56, 353kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<05:21, 458kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:20<03:51, 633kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<02:39, 895kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<19:09, 125kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<13:37, 175kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:22<09:28, 248kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:23<07:04, 327kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:23<05:25, 427kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:24<03:53, 592kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:25<03:01, 743kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<02:20, 957kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<01:40, 1.32MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:27<01:39, 1.31MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<01:37, 1.34MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<01:14, 1.74MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<00:52, 2.42MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:29<07:47, 271kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:29<05:39, 372kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 737M/862M [05:29<03:57, 524kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:31<03:12, 637kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:31<02:26, 833kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<01:44, 1.16MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:33<01:39, 1.19MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:33<01:33, 1.26MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<01:10, 1.68MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:33<00:49, 2.31MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:35<01:24, 1.35MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:35<01:10, 1.62MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<00:51, 2.18MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:37<01:01, 1.80MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:37<01:05, 1.69MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<00:50, 2.18MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:37<00:35, 2.98MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:39<01:16, 1.38MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:39<01:04, 1.64MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<00:47, 2.21MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:41<00:56, 1.81MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:41<01:00, 1.70MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:41<00:46, 2.16MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:41<00:32, 2.98MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:43<12:11, 134kB/s] .vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:43<08:40, 187kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<06:00, 266kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:45<04:28, 349kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:45<03:16, 475kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<02:17, 667kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:47<01:55, 777kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:47<01:29, 998kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<01:03, 1.38MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:49<01:03, 1.34MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:49<00:53, 1.60MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<00:38, 2.17MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:51<00:45, 1.79MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:51<00:48, 1.68MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:51<00:37, 2.14MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:51<00:26, 2.95MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:53<1:14:49, 17.2kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:53<52:15, 24.5kB/s]  .vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<35:52, 35.0kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:55<24:40, 49.5kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:55<17:28, 69.7kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:55<12:09, 99.1kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<08:16, 141kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:57<06:15, 184kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:57<04:28, 256kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:57<03:05, 363kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:59<02:20, 461kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:59<01:51, 582kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:59<01:20, 797kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:01<01:03, 961kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:01<00:50, 1.20MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:01<00:35, 1.66MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<00:37, 1.52MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:03<00:37, 1.51MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:03<00:28, 1.97MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:19, 2.72MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:49, 1.07MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:05<00:39, 1.32MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:05<00:28, 1.81MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:30, 1.60MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:07<00:30, 1.56MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [06:07<00:23, 2.01MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:22, 1.96MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:09<00:20, 2.18MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:09<00:14, 2.88MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:19, 2.09MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:16, 2.38MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:11<00:12, 3.18MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:08, 4.27MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<02:22, 254kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<01:41, 351kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:13<01:09, 497kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<00:45, 703kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:14<02:40, 199kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:14<01:54, 277kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:15<01:16, 391kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:16<00:56, 494kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:41, 666kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:28, 932kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:18, 1.30MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:18<02:51, 139kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<02:03, 191kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<01:24, 270kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:20<00:54, 363kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:39, 496kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:26, 694kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:20<00:16, 980kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:50, 309kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:36, 422kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:22<00:22, 596kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:24<00:16, 707kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:24<00:13, 838kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:09, 1.14MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:24<00:05, 1.59MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:26<00:07, 942kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:26<00:05, 1.18MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:03, 1.62MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:28<00:02, 1.50MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:28<00:02, 1.49MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:01, 1.94MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<30:09:51,  3.68it/s]  0%|          | 828/400000 [00:00<21:04:31,  5.26it/s]  0%|          | 1609/400000 [00:00<14:43:41,  7.51it/s]  1%|          | 2575/400000 [00:00<10:17:17, 10.73it/s]  1%|          | 3496/400000 [00:00<7:11:18, 15.32it/s]   1%|          | 4414/400000 [00:00<5:01:26, 21.87it/s]  1%|▏         | 5347/400000 [00:00<3:30:43, 31.21it/s]  2%|▏         | 6176/400000 [00:00<2:27:25, 44.52it/s]  2%|▏         | 7082/400000 [00:01<1:43:10, 63.47it/s]  2%|▏         | 8000/400000 [00:01<1:12:16, 90.40it/s]  2%|▏         | 8912/400000 [00:01<50:41, 128.59it/s]   2%|▏         | 9831/400000 [00:01<35:36, 182.61it/s]  3%|▎         | 10722/400000 [00:01<25:05, 258.58it/s]  3%|▎         | 11670/400000 [00:01<17:43, 365.13it/s]  3%|▎         | 12576/400000 [00:01<12:36, 512.23it/s]  3%|▎         | 13526/400000 [00:01<09:00, 715.22it/s]  4%|▎         | 14458/400000 [00:01<06:29, 989.18it/s]  4%|▍         | 15373/400000 [00:01<04:45, 1346.18it/s]  4%|▍         | 16270/400000 [00:02<03:32, 1804.82it/s]  4%|▍         | 17209/400000 [00:02<02:40, 2382.07it/s]  5%|▍         | 18146/400000 [00:02<02:04, 3068.52it/s]  5%|▍         | 19096/400000 [00:02<01:38, 3850.14it/s]  5%|▌         | 20022/400000 [00:02<01:22, 4616.98it/s]  5%|▌         | 20929/400000 [00:02<01:11, 5294.44it/s]  5%|▌         | 21806/400000 [00:02<01:03, 5949.49it/s]  6%|▌         | 22711/400000 [00:02<00:56, 6630.78it/s]  6%|▌         | 23588/400000 [00:02<00:53, 7096.78it/s]  6%|▌         | 24501/400000 [00:03<00:49, 7604.42it/s]  6%|▋         | 25419/400000 [00:03<00:46, 8014.53it/s]  7%|▋         | 26372/400000 [00:03<00:44, 8414.03it/s]  7%|▋         | 27341/400000 [00:03<00:42, 8758.05it/s]  7%|▋         | 28289/400000 [00:03<00:41, 8961.04it/s]  7%|▋         | 29223/400000 [00:03<00:41, 8917.62it/s]  8%|▊         | 30156/400000 [00:03<00:40, 9036.47it/s]  8%|▊         | 31086/400000 [00:03<00:40, 9112.35it/s]  8%|▊         | 32011/400000 [00:03<00:40, 9091.95it/s]  8%|▊         | 32930/400000 [00:03<00:40, 8968.38it/s]  8%|▊         | 33834/400000 [00:04<00:41, 8883.50it/s]  9%|▊         | 34732/400000 [00:04<00:40, 8910.75it/s]  9%|▉         | 35655/400000 [00:04<00:40, 9001.98it/s]  9%|▉         | 36558/400000 [00:04<00:40, 8986.28it/s]  9%|▉         | 37473/400000 [00:04<00:40, 9030.55it/s] 10%|▉         | 38406/400000 [00:04<00:39, 9118.09it/s] 10%|▉         | 39394/400000 [00:04<00:38, 9332.82it/s] 10%|█         | 40365/400000 [00:04<00:38, 9440.82it/s] 10%|█         | 41311/400000 [00:04<00:38, 9364.37it/s] 11%|█         | 42249/400000 [00:04<00:40, 8939.88it/s] 11%|█         | 43177/400000 [00:05<00:39, 9038.43it/s] 11%|█         | 44151/400000 [00:05<00:38, 9235.55it/s] 11%|█▏        | 45117/400000 [00:05<00:37, 9357.34it/s] 12%|█▏        | 46094/400000 [00:05<00:37, 9476.42it/s] 12%|█▏        | 47045/400000 [00:05<00:39, 9021.41it/s] 12%|█▏        | 47954/400000 [00:05<00:39, 8858.79it/s] 12%|█▏        | 48898/400000 [00:05<00:38, 9023.59it/s] 12%|█▏        | 49857/400000 [00:05<00:38, 9185.21it/s] 13%|█▎        | 50811/400000 [00:05<00:37, 9286.40it/s] 13%|█▎        | 51743/400000 [00:06<00:39, 8757.68it/s] 13%|█▎        | 52627/400000 [00:06<00:39, 8719.53it/s] 13%|█▎        | 53603/400000 [00:06<00:38, 9007.29it/s] 14%|█▎        | 54588/400000 [00:06<00:37, 9244.08it/s] 14%|█▍        | 55554/400000 [00:06<00:36, 9363.70it/s] 14%|█▍        | 56495/400000 [00:06<00:36, 9355.35it/s] 14%|█▍        | 57462/400000 [00:06<00:36, 9447.27it/s] 15%|█▍        | 58453/400000 [00:06<00:35, 9579.34it/s] 15%|█▍        | 59425/400000 [00:06<00:35, 9618.71it/s] 15%|█▌        | 60389/400000 [00:06<00:35, 9529.40it/s] 15%|█▌        | 61344/400000 [00:07<00:36, 9219.68it/s] 16%|█▌        | 62291/400000 [00:07<00:36, 9293.04it/s] 16%|█▌        | 63239/400000 [00:07<00:36, 9347.27it/s] 16%|█▌        | 64178/400000 [00:07<00:35, 9358.73it/s] 16%|█▋        | 65127/400000 [00:07<00:35, 9396.27it/s] 17%|█▋        | 66068/400000 [00:07<00:35, 9356.78it/s] 17%|█▋        | 67005/400000 [00:07<00:36, 9115.75it/s] 17%|█▋        | 67919/400000 [00:07<00:36, 8993.99it/s] 17%|█▋        | 68822/400000 [00:07<00:36, 9003.69it/s] 17%|█▋        | 69765/400000 [00:07<00:36, 9126.86it/s] 18%|█▊        | 70679/400000 [00:08<00:36, 9112.54it/s] 18%|█▊        | 71630/400000 [00:08<00:35, 9227.04it/s] 18%|█▊        | 72576/400000 [00:08<00:35, 9294.94it/s] 18%|█▊        | 73528/400000 [00:08<00:34, 9361.08it/s] 19%|█▊        | 74487/400000 [00:08<00:34, 9427.66it/s] 19%|█▉        | 75431/400000 [00:08<00:35, 9221.09it/s] 19%|█▉        | 76403/400000 [00:08<00:34, 9362.92it/s] 19%|█▉        | 77341/400000 [00:08<00:34, 9281.85it/s] 20%|█▉        | 78276/400000 [00:08<00:34, 9300.66it/s] 20%|█▉        | 79238/400000 [00:08<00:34, 9394.16it/s] 20%|██        | 80179/400000 [00:09<00:34, 9356.38it/s] 20%|██        | 81116/400000 [00:09<00:34, 9213.95it/s] 21%|██        | 82052/400000 [00:09<00:34, 9255.78it/s] 21%|██        | 82979/400000 [00:09<00:34, 9099.79it/s] 21%|██        | 83891/400000 [00:09<00:35, 9009.76it/s] 21%|██        | 84793/400000 [00:09<00:35, 8992.92it/s] 21%|██▏       | 85693/400000 [00:09<00:35, 8932.75it/s] 22%|██▏       | 86587/400000 [00:09<00:36, 8531.62it/s] 22%|██▏       | 87497/400000 [00:09<00:35, 8686.78it/s] 22%|██▏       | 88393/400000 [00:09<00:35, 8764.90it/s] 22%|██▏       | 89275/400000 [00:10<00:35, 8780.30it/s] 23%|██▎       | 90224/400000 [00:10<00:34, 8981.79it/s] 23%|██▎       | 91189/400000 [00:10<00:33, 9171.66it/s] 23%|██▎       | 92159/400000 [00:10<00:33, 9322.67it/s] 23%|██▎       | 93119/400000 [00:10<00:32, 9402.43it/s] 24%|██▎       | 94062/400000 [00:10<00:32, 9368.59it/s] 24%|██▍       | 95001/400000 [00:10<00:33, 9118.79it/s] 24%|██▍       | 95916/400000 [00:10<00:33, 9031.21it/s] 24%|██▍       | 96857/400000 [00:10<00:33, 9137.95it/s] 24%|██▍       | 97773/400000 [00:11<00:33, 8994.68it/s] 25%|██▍       | 98675/400000 [00:11<00:35, 8560.02it/s] 25%|██▍       | 99613/400000 [00:11<00:34, 8788.05it/s] 25%|██▌       | 100577/400000 [00:11<00:33, 9027.11it/s] 25%|██▌       | 101489/400000 [00:11<00:32, 9054.49it/s] 26%|██▌       | 102451/400000 [00:11<00:32, 9216.32it/s] 26%|██▌       | 103376/400000 [00:11<00:33, 8809.17it/s] 26%|██▌       | 104292/400000 [00:11<00:33, 8909.51it/s] 26%|██▋       | 105225/400000 [00:11<00:32, 9029.29it/s] 27%|██▋       | 106151/400000 [00:11<00:32, 9094.77it/s] 27%|██▋       | 107108/400000 [00:12<00:31, 9231.56it/s] 27%|██▋       | 108034/400000 [00:12<00:32, 9071.13it/s] 27%|██▋       | 108973/400000 [00:12<00:31, 9162.16it/s] 27%|██▋       | 109929/400000 [00:12<00:31, 9276.67it/s] 28%|██▊       | 110859/400000 [00:12<00:31, 9145.43it/s] 28%|██▊       | 111776/400000 [00:12<00:31, 9111.51it/s] 28%|██▊       | 112689/400000 [00:12<00:31, 9018.97it/s] 28%|██▊       | 113592/400000 [00:12<00:31, 8951.42it/s] 29%|██▊       | 114488/400000 [00:12<00:32, 8864.26it/s] 29%|██▉       | 115384/400000 [00:12<00:32, 8892.17it/s] 29%|██▉       | 116300/400000 [00:13<00:31, 8968.50it/s] 29%|██▉       | 117198/400000 [00:13<00:32, 8697.06it/s] 30%|██▉       | 118070/400000 [00:13<00:32, 8699.68it/s] 30%|██▉       | 118980/400000 [00:13<00:31, 8814.75it/s] 30%|██▉       | 119924/400000 [00:13<00:31, 8992.98it/s] 30%|███       | 120870/400000 [00:13<00:30, 9125.94it/s] 30%|███       | 121785/400000 [00:13<00:30, 9103.40it/s] 31%|███       | 122754/400000 [00:13<00:29, 9269.71it/s] 31%|███       | 123731/400000 [00:13<00:29, 9412.68it/s] 31%|███       | 124690/400000 [00:13<00:29, 9461.76it/s] 31%|███▏      | 125667/400000 [00:14<00:28, 9551.27it/s] 32%|███▏      | 126624/400000 [00:14<00:29, 9361.64it/s] 32%|███▏      | 127580/400000 [00:14<00:28, 9417.36it/s] 32%|███▏      | 128526/400000 [00:14<00:28, 9428.14it/s] 32%|███▏      | 129470/400000 [00:14<00:29, 9259.38it/s] 33%|███▎      | 130431/400000 [00:14<00:28, 9356.26it/s] 33%|███▎      | 131368/400000 [00:14<00:29, 9179.72it/s] 33%|███▎      | 132291/400000 [00:14<00:29, 9192.35it/s] 33%|███▎      | 133212/400000 [00:14<00:29, 9150.77it/s] 34%|███▎      | 134140/400000 [00:14<00:28, 9188.28it/s] 34%|███▍      | 135123/400000 [00:15<00:28, 9370.35it/s] 34%|███▍      | 136062/400000 [00:15<00:28, 9150.83it/s] 34%|███▍      | 137034/400000 [00:15<00:28, 9312.30it/s] 34%|███▍      | 137968/400000 [00:15<00:28, 9300.73it/s] 35%|███▍      | 138900/400000 [00:15<00:28, 9283.30it/s] 35%|███▍      | 139845/400000 [00:15<00:27, 9331.72it/s] 35%|███▌      | 140779/400000 [00:15<00:28, 9237.89it/s] 35%|███▌      | 141722/400000 [00:15<00:27, 9293.90it/s] 36%|███▌      | 142653/400000 [00:15<00:28, 9149.38it/s] 36%|███▌      | 143569/400000 [00:16<00:28, 9038.86it/s] 36%|███▌      | 144474/400000 [00:16<00:28, 8988.99it/s] 36%|███▋      | 145374/400000 [00:16<00:28, 8856.36it/s] 37%|███▋      | 146301/400000 [00:16<00:28, 8974.37it/s] 37%|███▋      | 147266/400000 [00:16<00:27, 9165.81it/s] 37%|███▋      | 148259/400000 [00:16<00:26, 9382.13it/s] 37%|███▋      | 149243/400000 [00:16<00:26, 9514.12it/s] 38%|███▊      | 150197/400000 [00:16<00:26, 9377.52it/s] 38%|███▊      | 151137/400000 [00:16<00:26, 9235.47it/s] 38%|███▊      | 152063/400000 [00:16<00:27, 9045.22it/s] 38%|███▊      | 153029/400000 [00:17<00:26, 9218.91it/s] 38%|███▊      | 153963/400000 [00:17<00:26, 9253.20it/s] 39%|███▊      | 154890/400000 [00:17<00:26, 9203.81it/s] 39%|███▉      | 155868/400000 [00:17<00:26, 9367.31it/s] 39%|███▉      | 156863/400000 [00:17<00:25, 9530.72it/s] 39%|███▉      | 157818/400000 [00:17<00:25, 9512.18it/s] 40%|███▉      | 158771/400000 [00:17<00:26, 9166.05it/s] 40%|███▉      | 159692/400000 [00:17<00:26, 8948.00it/s] 40%|████      | 160591/400000 [00:17<00:27, 8790.28it/s] 40%|████      | 161547/400000 [00:17<00:26, 9006.54it/s] 41%|████      | 162500/400000 [00:18<00:25, 9155.87it/s] 41%|████      | 163419/400000 [00:18<00:26, 9027.88it/s] 41%|████      | 164325/400000 [00:18<00:26, 8995.85it/s] 41%|████▏     | 165289/400000 [00:18<00:25, 9177.93it/s] 42%|████▏     | 166242/400000 [00:18<00:25, 9280.40it/s] 42%|████▏     | 167172/400000 [00:18<00:25, 9280.78it/s] 42%|████▏     | 168102/400000 [00:18<00:25, 9215.46it/s] 42%|████▏     | 169025/400000 [00:18<00:25, 9145.80it/s] 42%|████▏     | 169982/400000 [00:18<00:24, 9268.63it/s] 43%|████▎     | 170946/400000 [00:18<00:24, 9375.40it/s] 43%|████▎     | 171901/400000 [00:19<00:24, 9424.89it/s] 43%|████▎     | 172845/400000 [00:19<00:24, 9377.18it/s] 43%|████▎     | 173784/400000 [00:19<00:24, 9328.62it/s] 44%|████▎     | 174718/400000 [00:19<00:24, 9270.23it/s] 44%|████▍     | 175646/400000 [00:19<00:24, 9129.88it/s] 44%|████▍     | 176560/400000 [00:19<00:24, 9128.93it/s] 44%|████▍     | 177474/400000 [00:19<00:24, 9074.02it/s] 45%|████▍     | 178410/400000 [00:19<00:24, 9156.82it/s] 45%|████▍     | 179349/400000 [00:19<00:23, 9224.61it/s] 45%|████▌     | 180286/400000 [00:19<00:23, 9266.77it/s] 45%|████▌     | 181214/400000 [00:20<00:23, 9242.68it/s] 46%|████▌     | 182139/400000 [00:20<00:24, 9034.32it/s] 46%|████▌     | 183044/400000 [00:20<00:24, 9002.69it/s] 46%|████▌     | 183965/400000 [00:20<00:23, 9062.76it/s] 46%|████▌     | 184928/400000 [00:20<00:23, 9225.57it/s] 46%|████▋     | 185923/400000 [00:20<00:22, 9430.49it/s] 47%|████▋     | 186883/400000 [00:20<00:22, 9480.61it/s] 47%|████▋     | 187833/400000 [00:20<00:22, 9415.48it/s] 47%|████▋     | 188776/400000 [00:20<00:22, 9325.42it/s] 47%|████▋     | 189710/400000 [00:21<00:22, 9219.05it/s] 48%|████▊     | 190633/400000 [00:21<00:23, 8939.27it/s] 48%|████▊     | 191530/400000 [00:21<00:23, 8757.77it/s] 48%|████▊     | 192457/400000 [00:21<00:23, 8904.55it/s] 48%|████▊     | 193427/400000 [00:21<00:22, 9127.57it/s] 49%|████▊     | 194414/400000 [00:21<00:22, 9336.58it/s] 49%|████▉     | 195351/400000 [00:21<00:21, 9313.78it/s] 49%|████▉     | 196292/400000 [00:21<00:21, 9340.20it/s] 49%|████▉     | 197228/400000 [00:21<00:22, 9160.71it/s] 50%|████▉     | 198147/400000 [00:21<00:22, 9046.19it/s] 50%|████▉     | 199054/400000 [00:22<00:22, 8919.47it/s] 50%|█████     | 200004/400000 [00:22<00:22, 9083.96it/s] 50%|█████     | 200930/400000 [00:22<00:21, 9132.24it/s] 50%|█████     | 201853/400000 [00:22<00:21, 9159.91it/s] 51%|█████     | 202847/400000 [00:22<00:21, 9378.93it/s] 51%|█████     | 203826/400000 [00:22<00:20, 9497.09it/s] 51%|█████     | 204778/400000 [00:22<00:20, 9444.36it/s] 51%|█████▏    | 205724/400000 [00:22<00:21, 9175.75it/s] 52%|█████▏    | 206645/400000 [00:22<00:21, 8984.36it/s] 52%|█████▏    | 207611/400000 [00:22<00:20, 9176.01it/s] 52%|█████▏    | 208581/400000 [00:23<00:20, 9324.73it/s] 52%|█████▏    | 209553/400000 [00:23<00:20, 9437.42it/s] 53%|█████▎    | 210499/400000 [00:23<00:20, 9299.17it/s] 53%|█████▎    | 211475/400000 [00:23<00:19, 9430.34it/s] 53%|█████▎    | 212457/400000 [00:23<00:19, 9542.01it/s] 53%|█████▎    | 213413/400000 [00:23<00:19, 9510.11it/s] 54%|█████▎    | 214366/400000 [00:23<00:20, 9153.44it/s] 54%|█████▍    | 215286/400000 [00:23<00:20, 8890.46it/s] 54%|█████▍    | 216252/400000 [00:23<00:20, 9105.75it/s] 54%|█████▍    | 217213/400000 [00:24<00:19, 9250.86it/s] 55%|█████▍    | 218142/400000 [00:24<00:19, 9215.91it/s] 55%|█████▍    | 219067/400000 [00:24<00:19, 9129.82it/s] 55%|█████▍    | 219982/400000 [00:24<00:19, 9090.06it/s] 55%|█████▌    | 220893/400000 [00:24<00:20, 8905.98it/s] 55%|█████▌    | 221786/400000 [00:24<00:20, 8783.88it/s] 56%|█████▌    | 222711/400000 [00:24<00:19, 8918.53it/s] 56%|█████▌    | 223620/400000 [00:24<00:19, 8966.35it/s] 56%|█████▌    | 224518/400000 [00:24<00:19, 8958.81it/s] 56%|█████▋    | 225419/400000 [00:24<00:19, 8972.88it/s] 57%|█████▋    | 226325/400000 [00:25<00:19, 8997.06it/s] 57%|█████▋    | 227274/400000 [00:25<00:18, 9138.30it/s] 57%|█████▋    | 228228/400000 [00:25<00:18, 9254.95it/s] 57%|█████▋    | 229155/400000 [00:25<00:18, 9115.71it/s] 58%|█████▊    | 230101/400000 [00:25<00:18, 9215.77it/s] 58%|█████▊    | 231041/400000 [00:25<00:18, 9269.10it/s] 58%|█████▊    | 231995/400000 [00:25<00:17, 9347.70it/s] 58%|█████▊    | 232942/400000 [00:25<00:17, 9383.72it/s] 58%|█████▊    | 233881/400000 [00:25<00:17, 9331.39it/s] 59%|█████▊    | 234840/400000 [00:25<00:17, 9405.81it/s] 59%|█████▉    | 235805/400000 [00:26<00:17, 9476.15it/s] 59%|█████▉    | 236754/400000 [00:26<00:17, 9235.10it/s] 59%|█████▉    | 237680/400000 [00:26<00:17, 9120.33it/s] 60%|█████▉    | 238594/400000 [00:26<00:17, 9013.87it/s] 60%|█████▉    | 239541/400000 [00:26<00:17, 9143.68it/s] 60%|██████    | 240515/400000 [00:26<00:17, 9313.20it/s] 60%|██████    | 241484/400000 [00:26<00:16, 9422.78it/s] 61%|██████    | 242428/400000 [00:26<00:16, 9397.12it/s] 61%|██████    | 243369/400000 [00:26<00:17, 9158.11it/s] 61%|██████    | 244332/400000 [00:26<00:16, 9294.13it/s] 61%|██████▏   | 245294/400000 [00:27<00:16, 9386.62it/s] 62%|██████▏   | 246264/400000 [00:27<00:16, 9476.09it/s] 62%|██████▏   | 247213/400000 [00:27<00:16, 9467.48it/s] 62%|██████▏   | 248161/400000 [00:27<00:16, 9414.76it/s] 62%|██████▏   | 249104/400000 [00:27<00:16, 9354.43it/s] 63%|██████▎   | 250097/400000 [00:27<00:15, 9518.74it/s] 63%|██████▎   | 251067/400000 [00:27<00:15, 9569.77it/s] 63%|██████▎   | 252025/400000 [00:27<00:16, 9193.66it/s] 63%|██████▎   | 252949/400000 [00:27<00:16, 9047.26it/s] 63%|██████▎   | 253859/400000 [00:27<00:16, 9060.81it/s] 64%|██████▎   | 254809/400000 [00:28<00:15, 9188.12it/s] 64%|██████▍   | 255777/400000 [00:28<00:15, 9329.40it/s] 64%|██████▍   | 256732/400000 [00:28<00:15, 9393.71it/s] 64%|██████▍   | 257673/400000 [00:28<00:15, 9159.78it/s] 65%|██████▍   | 258606/400000 [00:28<00:15, 9210.04it/s] 65%|██████▍   | 259571/400000 [00:28<00:15, 9336.63it/s] 65%|██████▌   | 260517/400000 [00:28<00:14, 9371.63it/s] 65%|██████▌   | 261463/400000 [00:28<00:14, 9395.71it/s] 66%|██████▌   | 262404/400000 [00:28<00:14, 9197.76it/s] 66%|██████▌   | 263396/400000 [00:29<00:14, 9402.98it/s] 66%|██████▌   | 264378/400000 [00:29<00:14, 9523.37it/s] 66%|██████▋   | 265333/400000 [00:29<00:14, 9332.22it/s] 67%|██████▋   | 266272/400000 [00:29<00:14, 9348.12it/s] 67%|██████▋   | 267209/400000 [00:29<00:14, 9062.52it/s] 67%|██████▋   | 268119/400000 [00:29<00:14, 9067.05it/s] 67%|██████▋   | 269073/400000 [00:29<00:14, 9202.09it/s] 68%|██████▊   | 270060/400000 [00:29<00:13, 9391.73it/s] 68%|██████▊   | 271028/400000 [00:29<00:13, 9473.66it/s] 68%|██████▊   | 271978/400000 [00:29<00:13, 9429.51it/s] 68%|██████▊   | 272956/400000 [00:30<00:13, 9531.26it/s] 68%|██████▊   | 273938/400000 [00:30<00:13, 9613.55it/s] 69%|██████▊   | 274901/400000 [00:30<00:13, 9594.37it/s] 69%|██████▉   | 275875/400000 [00:30<00:12, 9637.44it/s] 69%|██████▉   | 276840/400000 [00:30<00:12, 9495.83it/s] 69%|██████▉   | 277807/400000 [00:30<00:12, 9545.61it/s] 70%|██████▉   | 278768/400000 [00:30<00:12, 9564.55it/s] 70%|██████▉   | 279743/400000 [00:30<00:12, 9618.98it/s] 70%|███████   | 280706/400000 [00:30<00:12, 9603.06it/s] 70%|███████   | 281667/400000 [00:30<00:12, 9444.01it/s] 71%|███████   | 282613/400000 [00:31<00:12, 9257.71it/s] 71%|███████   | 283541/400000 [00:31<00:13, 8884.39it/s] 71%|███████   | 284496/400000 [00:31<00:12, 9073.80it/s] 71%|███████▏  | 285455/400000 [00:31<00:12, 9220.35it/s] 72%|███████▏  | 286381/400000 [00:31<00:12, 9088.95it/s] 72%|███████▏  | 287297/400000 [00:31<00:12, 9107.80it/s] 72%|███████▏  | 288252/400000 [00:31<00:12, 9234.15it/s] 72%|███████▏  | 289178/400000 [00:31<00:12, 9233.98it/s] 73%|███████▎  | 290121/400000 [00:31<00:11, 9290.71it/s] 73%|███████▎  | 291052/400000 [00:31<00:11, 9184.21it/s] 73%|███████▎  | 292007/400000 [00:32<00:11, 9290.56it/s] 73%|███████▎  | 292971/400000 [00:32<00:11, 9390.88it/s] 73%|███████▎  | 293945/400000 [00:32<00:11, 9490.41it/s] 74%|███████▎  | 294895/400000 [00:32<00:11, 9376.68it/s] 74%|███████▍  | 295834/400000 [00:32<00:11, 9289.88it/s] 74%|███████▍  | 296764/400000 [00:32<00:11, 9169.76it/s] 74%|███████▍  | 297682/400000 [00:32<00:11, 9058.61it/s] 75%|███████▍  | 298589/400000 [00:32<00:11, 8936.69it/s] 75%|███████▍  | 299536/400000 [00:32<00:11, 9088.54it/s] 75%|███████▌  | 300447/400000 [00:32<00:11, 8967.93it/s] 75%|███████▌  | 301407/400000 [00:33<00:10, 9148.41it/s] 76%|███████▌  | 302374/400000 [00:33<00:10, 9297.06it/s] 76%|███████▌  | 303336/400000 [00:33<00:10, 9388.82it/s] 76%|███████▌  | 304290/400000 [00:33<00:10, 9433.01it/s] 76%|███████▋  | 305235/400000 [00:33<00:10, 9152.91it/s] 77%|███████▋  | 306209/400000 [00:33<00:10, 9319.68it/s] 77%|███████▋  | 307164/400000 [00:33<00:09, 9387.01it/s] 77%|███████▋  | 308123/400000 [00:33<00:09, 9446.30it/s] 77%|███████▋  | 309070/400000 [00:33<00:09, 9444.36it/s] 78%|███████▊  | 310016/400000 [00:34<00:09, 9278.71it/s] 78%|███████▊  | 310984/400000 [00:34<00:09, 9395.53it/s] 78%|███████▊  | 311925/400000 [00:34<00:09, 9143.11it/s] 78%|███████▊  | 312842/400000 [00:34<00:09, 9096.40it/s] 78%|███████▊  | 313754/400000 [00:34<00:09, 9017.00it/s] 79%|███████▊  | 314658/400000 [00:34<00:09, 8956.58it/s] 79%|███████▉  | 315589/400000 [00:34<00:09, 9059.09it/s] 79%|███████▉  | 316563/400000 [00:34<00:09, 9250.72it/s] 79%|███████▉  | 317490/400000 [00:34<00:08, 9204.54it/s] 80%|███████▉  | 318462/400000 [00:34<00:08, 9352.01it/s] 80%|███████▉  | 319399/400000 [00:35<00:08, 9272.26it/s] 80%|████████  | 320363/400000 [00:35<00:08, 9378.38it/s] 80%|████████  | 321327/400000 [00:35<00:08, 9454.48it/s] 81%|████████  | 322274/400000 [00:35<00:08, 9415.71it/s] 81%|████████  | 323217/400000 [00:35<00:08, 9389.64it/s] 81%|████████  | 324157/400000 [00:35<00:08, 9297.32it/s] 81%|████████▏ | 325130/400000 [00:35<00:07, 9422.29it/s] 82%|████████▏ | 326074/400000 [00:35<00:07, 9392.94it/s] 82%|████████▏ | 327014/400000 [00:35<00:07, 9382.56it/s] 82%|████████▏ | 327976/400000 [00:35<00:07, 9451.75it/s] 82%|████████▏ | 328922/400000 [00:36<00:07, 9134.49it/s] 82%|████████▏ | 329839/400000 [00:36<00:07, 9024.83it/s] 83%|████████▎ | 330780/400000 [00:36<00:07, 9134.66it/s] 83%|████████▎ | 331755/400000 [00:36<00:07, 9309.31it/s] 83%|████████▎ | 332740/400000 [00:36<00:07, 9464.97it/s] 83%|████████▎ | 333689/400000 [00:36<00:07, 9315.02it/s] 84%|████████▎ | 334623/400000 [00:36<00:07, 9194.86it/s] 84%|████████▍ | 335561/400000 [00:36<00:06, 9248.23it/s] 84%|████████▍ | 336525/400000 [00:36<00:06, 9360.21it/s] 84%|████████▍ | 337463/400000 [00:36<00:06, 9115.73it/s] 85%|████████▍ | 338377/400000 [00:37<00:06, 9068.75it/s] 85%|████████▍ | 339302/400000 [00:37<00:06, 9120.83it/s] 85%|████████▌ | 340220/400000 [00:37<00:06, 9138.10it/s] 85%|████████▌ | 341163/400000 [00:37<00:06, 9221.51it/s] 86%|████████▌ | 342107/400000 [00:37<00:06, 9285.76it/s] 86%|████████▌ | 343037/400000 [00:37<00:06, 9064.16it/s] 86%|████████▌ | 343982/400000 [00:37<00:06, 9175.57it/s] 86%|████████▌ | 344902/400000 [00:37<00:06, 9055.14it/s] 86%|████████▋ | 345843/400000 [00:37<00:05, 9156.85it/s] 87%|████████▋ | 346765/400000 [00:37<00:05, 9175.07it/s] 87%|████████▋ | 347684/400000 [00:38<00:05, 9139.79it/s] 87%|████████▋ | 348659/400000 [00:38<00:05, 9312.91it/s] 87%|████████▋ | 349625/400000 [00:38<00:05, 9413.28it/s] 88%|████████▊ | 350588/400000 [00:38<00:05, 9477.17it/s] 88%|████████▊ | 351537/400000 [00:38<00:05, 9472.72it/s] 88%|████████▊ | 352485/400000 [00:38<00:05, 9277.02it/s] 88%|████████▊ | 353415/400000 [00:38<00:05, 9264.56it/s] 89%|████████▊ | 354361/400000 [00:38<00:04, 9321.98it/s] 89%|████████▉ | 355294/400000 [00:38<00:04, 9323.01it/s] 89%|████████▉ | 356262/400000 [00:38<00:04, 9424.94it/s] 89%|████████▉ | 357206/400000 [00:39<00:04, 9350.53it/s] 90%|████████▉ | 358179/400000 [00:39<00:04, 9459.41it/s] 90%|████████▉ | 359132/400000 [00:39<00:04, 9479.62it/s] 90%|█████████ | 360081/400000 [00:39<00:04, 9432.74it/s] 90%|█████████ | 361025/400000 [00:39<00:04, 9307.72it/s] 90%|█████████ | 361957/400000 [00:39<00:04, 9272.90it/s] 91%|█████████ | 362941/400000 [00:39<00:03, 9433.30it/s] 91%|█████████ | 363912/400000 [00:39<00:03, 9512.24it/s] 91%|█████████ | 364880/400000 [00:39<00:03, 9559.31it/s] 91%|█████████▏| 365837/400000 [00:40<00:03, 9370.64it/s] 92%|█████████▏| 366776/400000 [00:40<00:03, 9316.37it/s] 92%|█████████▏| 367763/400000 [00:40<00:03, 9474.14it/s] 92%|█████████▏| 368741/400000 [00:40<00:03, 9561.24it/s] 92%|█████████▏| 369722/400000 [00:40<00:03, 9632.62it/s] 93%|█████████▎| 370705/400000 [00:40<00:03, 9689.51it/s] 93%|█████████▎| 371675/400000 [00:40<00:02, 9577.29it/s] 93%|█████████▎| 372634/400000 [00:40<00:02, 9472.93it/s] 93%|█████████▎| 373583/400000 [00:40<00:02, 9455.53it/s] 94%|█████████▎| 374552/400000 [00:40<00:02, 9522.26it/s] 94%|█████████▍| 375505/400000 [00:41<00:02, 9441.04it/s] 94%|█████████▍| 376450/400000 [00:41<00:02, 9127.43it/s] 94%|█████████▍| 377366/400000 [00:41<00:02, 9127.43it/s] 95%|█████████▍| 378320/400000 [00:41<00:02, 9244.28it/s] 95%|█████████▍| 379282/400000 [00:41<00:02, 9351.64it/s] 95%|█████████▌| 380219/400000 [00:41<00:02, 9216.52it/s] 95%|█████████▌| 381143/400000 [00:41<00:02, 8866.47it/s] 96%|█████████▌| 382072/400000 [00:41<00:01, 8989.00it/s] 96%|█████████▌| 382989/400000 [00:41<00:01, 9040.06it/s] 96%|█████████▌| 383915/400000 [00:41<00:01, 9103.21it/s] 96%|█████████▌| 384856/400000 [00:42<00:01, 9192.76it/s] 96%|█████████▋| 385777/400000 [00:42<00:01, 8976.53it/s] 97%|█████████▋| 386735/400000 [00:42<00:01, 9148.84it/s] 97%|█████████▋| 387696/400000 [00:42<00:01, 9281.00it/s] 97%|█████████▋| 388652/400000 [00:42<00:01, 9361.95it/s] 97%|█████████▋| 389600/400000 [00:42<00:01, 9394.93it/s] 98%|█████████▊| 390541/400000 [00:42<00:01, 8981.65it/s] 98%|█████████▊| 391444/400000 [00:42<00:00, 8704.97it/s] 98%|█████████▊| 392344/400000 [00:42<00:00, 8790.02it/s] 98%|█████████▊| 393261/400000 [00:42<00:00, 8900.12it/s] 99%|█████████▊| 394161/400000 [00:43<00:00, 8929.23it/s] 99%|█████████▉| 395112/400000 [00:43<00:00, 9093.35it/s] 99%|█████████▉| 396062/400000 [00:43<00:00, 9209.16it/s] 99%|█████████▉| 397042/400000 [00:43<00:00, 9378.40it/s] 99%|█████████▉| 397996/400000 [00:43<00:00, 9426.07it/s]100%|█████████▉| 398942/400000 [00:43<00:00, 9433.23it/s]100%|█████████▉| 399887/400000 [00:43<00:00, 9330.59it/s]100%|█████████▉| 399999/400000 [00:43<00:00, 9150.64it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f4fd53113c8>, <torchtext.data.dataset.TabularDataset object at 0x7f4fd5311518>, <torchtext.vocab.Vocab object at 0x7f4fd5311438>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  6.41 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  6.41 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  6.41 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  7.72 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  7.72 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  3.88 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  3.88 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.70 file/s]2020-06-16 00:17:36.921684: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-16 00:17:36.924979: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-06-16 00:17:36.925187: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55f289936350 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-16 00:17:36.925203: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 39%|███▊      | 3825664/9912422 [00:00<00:00, 38247922.22it/s]9920512it [00:00, 36512594.51it/s]                             
0it [00:00, ?it/s]32768it [00:00, 525037.07it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:09, 163633.00it/s]1654784it [00:00, 11513301.48it/s]                         
0it [00:00, ?it/s]8192it [00:00, 196027.72it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
