
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7fb2508586a8> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7fb2508586a8> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7fb2bbe200d0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7fb2bbe200d0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7fb2d5b4cea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7fb2d5b4cea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fb26969b950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fb26969b950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fb26969b950> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 27%|██▋       | 2711552/9912422 [00:00<00:00, 26856159.78it/s]9920512it [00:00, 35200659.15it/s]                             
0it [00:00, ?it/s]32768it [00:00, 582444.95it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 158593.41it/s]1654784it [00:00, 11191212.92it/s]                         
0it [00:00, ?it/s]8192it [00:00, 194375.39it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fb269689be0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fb250726e48>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7fb26969b598> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7fb26969b598> 

  function with postional parmater data_info <function tf_dataset_download at 0x7fb26969b598> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:19,  2.02 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:19,  2.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:19,  2.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:18,  2.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:18,  2.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:17,  2.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:17,  2.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▍         | 7/162 [00:00<00:54,  2.84 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:54,  2.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:54,  2.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:53,  2.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:53,  2.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:53,  2.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:52,  2.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:52,  2.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:52,  2.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:51,  2.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  10%|▉         | 16/162 [00:00<00:36,  3.99 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:36,  3.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:36,  3.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:36,  3.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:35,  3.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:35,  3.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:35,  3.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:35,  3.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:34,  3.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  15%|█▍        | 24/162 [00:00<00:24,  5.58 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:24,  5.58 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:24,  5.58 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:24,  5.58 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:24,  5.58 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:24,  5.58 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:23,  5.58 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:23,  5.58 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:23,  5.58 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  20%|█▉        | 32/162 [00:00<00:16,  7.74 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:16,  7.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:16,  7.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:16,  7.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:16,  7.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:16,  7.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:16,  7.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:16,  7.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:15,  7.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  25%|██▍       | 40/162 [00:01<00:11, 10.61 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:11, 10.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:11, 10.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:11, 10.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:11, 10.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:11, 10.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:11, 10.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:10, 10.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:10, 10.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  30%|██▉       | 48/162 [00:01<00:07, 14.33 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:07, 14.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:07, 14.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:07, 14.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:07, 14.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:07, 14.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:07, 14.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:07, 14.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:07, 14.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  35%|███▍      | 56/162 [00:01<00:05, 18.99 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:05, 18.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:05, 18.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:05, 18.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:05, 18.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:05, 18.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:05, 18.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:05, 18.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:05, 18.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  40%|███▉      | 64/162 [00:01<00:03, 24.62 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:03, 24.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:03, 24.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:03, 24.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 24.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:03, 24.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:03, 24.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:03, 24.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:03, 24.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 30.75 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 30.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 30.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 30.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 30.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 30.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 30.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 30.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 30.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 30.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 37.77 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 37.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 37.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 37.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 37.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 37.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 37.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 37.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 37.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 37.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 44.72 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 44.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 44.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 44.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 44.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 44.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 44.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 44.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 44.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 51.23 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 51.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 51.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 51.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 51.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 51.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 51.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 51.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 51.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 51.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:00, 58.05 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:00, 58.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:00, 58.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:00, 58.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 58.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 58.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 58.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 58.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 58.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 62.60 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 62.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 62.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 62.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 62.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 62.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 62.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 62.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 62.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 66.59 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 66.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 66.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 66.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 66.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 66.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 66.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 66.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 66.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 68.98 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 68.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 68.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 68.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 68.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 68.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 68.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 68.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 68.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 71.15 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 71.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 71.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 71.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 71.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 71.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 71.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 71.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 71.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 73.03 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 73.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 73.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 73.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 73.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 73.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 73.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 73.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 73.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 74.33 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 74.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 74.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 74.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 74.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 74.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 74.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 74.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 74.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.58s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.58s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 74.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.58s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 74.33 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.82s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.58s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 74.33 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.82s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.82s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 33.59 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.82s/ url]
0 examples [00:00, ? examples/s]2020-06-15 00:09:00.045541: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-06-15 00:09:00.068102: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-06-15 00:09:00.068584: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5585cf389e10 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-15 00:09:00.068609: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
5 examples [00:00, 49.62 examples/s]110 examples [00:00, 69.47 examples/s]218 examples [00:00, 96.57 examples/s]328 examples [00:00, 132.94 examples/s]437 examples [00:00, 180.46 examples/s]547 examples [00:00, 240.75 examples/s]658 examples [00:00, 314.57 examples/s]767 examples [00:00, 399.85 examples/s]874 examples [00:00, 491.99 examples/s]984 examples [00:01, 589.38 examples/s]1089 examples [00:01, 671.94 examples/s]1192 examples [00:01, 656.26 examples/s]1285 examples [00:01, 719.50 examples/s]1377 examples [00:01, 744.43 examples/s]1466 examples [00:01, 780.14 examples/s]1555 examples [00:01, 782.92 examples/s]1641 examples [00:01, 779.72 examples/s]1736 examples [00:01, 823.98 examples/s]1836 examples [00:02, 869.27 examples/s]1927 examples [00:02, 874.53 examples/s]2023 examples [00:02, 897.83 examples/s]2124 examples [00:02, 928.12 examples/s]2229 examples [00:02, 961.54 examples/s]2327 examples [00:02, 928.05 examples/s]2430 examples [00:02, 955.50 examples/s]2529 examples [00:02, 964.63 examples/s]2628 examples [00:02, 970.73 examples/s]2735 examples [00:02, 996.30 examples/s]2842 examples [00:03, 1015.04 examples/s]2944 examples [00:03, 996.08 examples/s] 3052 examples [00:03, 1017.38 examples/s]3155 examples [00:03, 1001.56 examples/s]3265 examples [00:03, 1027.29 examples/s]3375 examples [00:03, 1045.56 examples/s]3486 examples [00:03, 1062.17 examples/s]3593 examples [00:03, 1062.27 examples/s]3702 examples [00:03, 1069.30 examples/s]3810 examples [00:03, 1069.77 examples/s]3921 examples [00:04, 1079.41 examples/s]4030 examples [00:04, 1064.59 examples/s]4137 examples [00:04, 1057.84 examples/s]4243 examples [00:04, 1045.85 examples/s]4352 examples [00:04, 1055.77 examples/s]4458 examples [00:04, 1056.44 examples/s]4564 examples [00:04, 1048.04 examples/s]4669 examples [00:04, 1013.16 examples/s]4772 examples [00:04, 1017.48 examples/s]4878 examples [00:04, 1027.69 examples/s]4983 examples [00:05, 1033.27 examples/s]5087 examples [00:05, 1028.66 examples/s]5196 examples [00:05, 1045.41 examples/s]5301 examples [00:05, 1046.71 examples/s]5408 examples [00:05, 1052.56 examples/s]5514 examples [00:05, 1009.36 examples/s]5619 examples [00:05, 1020.61 examples/s]5728 examples [00:05, 1038.10 examples/s]5833 examples [00:05, 1041.27 examples/s]5943 examples [00:06, 1057.98 examples/s]6051 examples [00:06, 1062.80 examples/s]6158 examples [00:06, 1062.65 examples/s]6268 examples [00:06, 1072.69 examples/s]6376 examples [00:06, 1066.78 examples/s]6483 examples [00:06, 1063.61 examples/s]6590 examples [00:06, 1061.97 examples/s]6697 examples [00:06, 1041.45 examples/s]6802 examples [00:06, 1007.25 examples/s]6906 examples [00:06, 1014.67 examples/s]7017 examples [00:07, 1039.02 examples/s]7122 examples [00:07, 1010.26 examples/s]7231 examples [00:07, 1032.60 examples/s]7335 examples [00:07, 1030.43 examples/s]7441 examples [00:07, 1038.31 examples/s]7546 examples [00:07, 1027.66 examples/s]7649 examples [00:07, 1000.61 examples/s]7750 examples [00:07, 983.38 examples/s] 7849 examples [00:07, 981.08 examples/s]7948 examples [00:07, 981.93 examples/s]8051 examples [00:08, 995.12 examples/s]8156 examples [00:08, 1009.72 examples/s]8264 examples [00:08, 1028.39 examples/s]8368 examples [00:08, 1025.92 examples/s]8471 examples [00:08, 1014.26 examples/s]8578 examples [00:08, 1029.68 examples/s]8686 examples [00:08, 1043.33 examples/s]8791 examples [00:08, 1035.59 examples/s]8901 examples [00:08, 1052.40 examples/s]9009 examples [00:08, 1060.45 examples/s]9117 examples [00:09, 1063.77 examples/s]9225 examples [00:09, 1068.22 examples/s]9337 examples [00:09, 1080.58 examples/s]9448 examples [00:09, 1086.50 examples/s]9557 examples [00:09, 1067.75 examples/s]9664 examples [00:09, 1040.62 examples/s]9769 examples [00:09, 1015.06 examples/s]9875 examples [00:09, 1026.85 examples/s]9978 examples [00:09, 1023.44 examples/s]10081 examples [00:10, 968.06 examples/s]10180 examples [00:10, 973.54 examples/s]10279 examples [00:10, 976.36 examples/s]10388 examples [00:10, 1006.66 examples/s]10496 examples [00:10, 1027.13 examples/s]10602 examples [00:10, 1034.81 examples/s]10706 examples [00:10, 1014.00 examples/s]10808 examples [00:10, 1000.67 examples/s]10909 examples [00:10, 989.07 examples/s] 11009 examples [00:10, 985.41 examples/s]11112 examples [00:11, 997.59 examples/s]11219 examples [00:11, 1010.73 examples/s]11323 examples [00:11, 1016.76 examples/s]11434 examples [00:11, 1040.44 examples/s]11540 examples [00:11, 1039.75 examples/s]11645 examples [00:11, 1025.58 examples/s]11748 examples [00:11, 1021.41 examples/s]11851 examples [00:11, 1002.00 examples/s]11952 examples [00:11, 984.11 examples/s] 12051 examples [00:11, 985.03 examples/s]12159 examples [00:12, 1009.60 examples/s]12263 examples [00:12, 1017.44 examples/s]12365 examples [00:12, 1013.94 examples/s]12473 examples [00:12, 1031.69 examples/s]12581 examples [00:12, 1043.90 examples/s]12689 examples [00:12, 1052.16 examples/s]12795 examples [00:12, 1033.42 examples/s]12899 examples [00:12, 1022.38 examples/s]13002 examples [00:12, 1023.88 examples/s]13110 examples [00:12, 1039.66 examples/s]13220 examples [00:13, 1055.87 examples/s]13326 examples [00:13, 1043.41 examples/s]13431 examples [00:13, 1025.93 examples/s]13534 examples [00:13, 1024.93 examples/s]13637 examples [00:13, 1024.25 examples/s]13746 examples [00:13, 1040.77 examples/s]13856 examples [00:13, 1055.58 examples/s]13963 examples [00:13, 1059.65 examples/s]14073 examples [00:13, 1071.29 examples/s]14181 examples [00:14, 1069.27 examples/s]14288 examples [00:14, 1039.03 examples/s]14394 examples [00:14, 1042.21 examples/s]14504 examples [00:14, 1056.78 examples/s]14613 examples [00:14, 1066.40 examples/s]14720 examples [00:14, 1057.65 examples/s]14828 examples [00:14, 1062.29 examples/s]14935 examples [00:14, 1046.34 examples/s]15040 examples [00:14, 1031.47 examples/s]15150 examples [00:14, 1048.83 examples/s]15256 examples [00:15, 1026.31 examples/s]15359 examples [00:15, 1021.27 examples/s]15462 examples [00:15, 1023.19 examples/s]15566 examples [00:15, 1026.90 examples/s]15676 examples [00:15, 1047.69 examples/s]15784 examples [00:15, 1056.17 examples/s]15896 examples [00:15, 1072.10 examples/s]16004 examples [00:15, 1032.55 examples/s]16109 examples [00:15, 1035.67 examples/s]16216 examples [00:15, 1044.01 examples/s]16323 examples [00:16, 1049.58 examples/s]16429 examples [00:16, 1021.87 examples/s]16532 examples [00:16, 1002.36 examples/s]16638 examples [00:16, 1017.97 examples/s]16741 examples [00:16, 1021.45 examples/s]16847 examples [00:16, 1030.87 examples/s]16951 examples [00:16, 1007.61 examples/s]17054 examples [00:16, 1013.18 examples/s]17158 examples [00:16, 1014.70 examples/s]17260 examples [00:16, 1008.21 examples/s]17364 examples [00:17, 1016.02 examples/s]17472 examples [00:17, 1033.92 examples/s]17578 examples [00:17, 1039.02 examples/s]17688 examples [00:17, 1055.80 examples/s]17797 examples [00:17, 1062.88 examples/s]17904 examples [00:17, 1057.99 examples/s]18012 examples [00:17, 1064.16 examples/s]18121 examples [00:17, 1070.61 examples/s]18229 examples [00:17, 1049.82 examples/s]18335 examples [00:18, 1051.62 examples/s]18446 examples [00:18, 1067.88 examples/s]18558 examples [00:18, 1082.66 examples/s]18669 examples [00:18, 1089.33 examples/s]18779 examples [00:18, 1065.71 examples/s]18887 examples [00:18, 1068.92 examples/s]18995 examples [00:18, 1066.95 examples/s]19102 examples [00:18, 1057.41 examples/s]19208 examples [00:18, 1033.63 examples/s]19312 examples [00:18, 1016.38 examples/s]19417 examples [00:19, 1024.80 examples/s]19520 examples [00:19, 998.93 examples/s] 19626 examples [00:19, 1013.82 examples/s]19729 examples [00:19, 1016.40 examples/s]19832 examples [00:19, 1019.95 examples/s]19941 examples [00:19, 1037.94 examples/s]20045 examples [00:19, 1004.44 examples/s]20155 examples [00:19, 1031.05 examples/s]20264 examples [00:19, 1046.91 examples/s]20377 examples [00:19, 1067.50 examples/s]20486 examples [00:20, 1073.77 examples/s]20597 examples [00:20, 1081.99 examples/s]20706 examples [00:20, 1084.27 examples/s]20817 examples [00:20, 1090.61 examples/s]20929 examples [00:20, 1097.60 examples/s]21039 examples [00:20, 1079.37 examples/s]21149 examples [00:20, 1084.29 examples/s]21258 examples [00:20, 1083.52 examples/s]21367 examples [00:20, 1069.05 examples/s]21476 examples [00:20, 1073.09 examples/s]21584 examples [00:21, 1069.46 examples/s]21691 examples [00:21, 1044.96 examples/s]21798 examples [00:21, 1052.27 examples/s]21904 examples [00:21, 1006.18 examples/s]22006 examples [00:21, 985.27 examples/s] 22105 examples [00:21, 974.52 examples/s]22208 examples [00:21, 989.84 examples/s]22308 examples [00:21, 990.95 examples/s]22420 examples [00:21, 1025.35 examples/s]22523 examples [00:22, 1014.68 examples/s]22628 examples [00:22, 1024.23 examples/s]22731 examples [00:22, 991.09 examples/s] 22838 examples [00:22, 1012.01 examples/s]22947 examples [00:22, 1034.06 examples/s]23058 examples [00:22, 1053.30 examples/s]23166 examples [00:22, 1059.98 examples/s]23276 examples [00:22, 1069.62 examples/s]23385 examples [00:22, 1066.13 examples/s]23493 examples [00:22, 1068.38 examples/s]23604 examples [00:23, 1077.97 examples/s]23712 examples [00:23, 1055.63 examples/s]23820 examples [00:23, 1060.91 examples/s]23927 examples [00:23, 1024.91 examples/s]24034 examples [00:23, 1033.57 examples/s]24139 examples [00:23, 1035.67 examples/s]24243 examples [00:23, 1013.10 examples/s]24348 examples [00:23, 1021.09 examples/s]24451 examples [00:23, 1006.69 examples/s]24561 examples [00:23, 1031.35 examples/s]24665 examples [00:24, 1033.00 examples/s]24773 examples [00:24, 1044.49 examples/s]24878 examples [00:24, 1043.17 examples/s]24985 examples [00:24, 1049.62 examples/s]25091 examples [00:24, 1003.10 examples/s]25199 examples [00:24, 1022.10 examples/s]25304 examples [00:24, 1029.84 examples/s]25415 examples [00:24, 1051.47 examples/s]25521 examples [00:24, 1018.19 examples/s]25629 examples [00:25, 1035.30 examples/s]25733 examples [00:25, 1034.27 examples/s]25837 examples [00:25, 995.56 examples/s] 25941 examples [00:25, 1008.14 examples/s]26049 examples [00:25, 1026.30 examples/s]26153 examples [00:25, 1029.17 examples/s]26257 examples [00:25, 1025.72 examples/s]26362 examples [00:25, 1031.72 examples/s]26472 examples [00:25, 1049.24 examples/s]26578 examples [00:25, 1044.14 examples/s]26685 examples [00:26, 1049.09 examples/s]26795 examples [00:26, 1063.47 examples/s]26904 examples [00:26, 1069.35 examples/s]27012 examples [00:26, 1062.98 examples/s]27121 examples [00:26, 1070.23 examples/s]27230 examples [00:26, 1073.23 examples/s]27340 examples [00:26, 1079.12 examples/s]27450 examples [00:26, 1084.66 examples/s]27560 examples [00:26, 1088.64 examples/s]27669 examples [00:26, 1082.71 examples/s]27778 examples [00:27, 1046.41 examples/s]27883 examples [00:27, 1019.98 examples/s]27989 examples [00:27, 1029.63 examples/s]28098 examples [00:27, 1046.48 examples/s]28208 examples [00:27, 1060.04 examples/s]28317 examples [00:27, 1066.59 examples/s]28424 examples [00:27, 1063.78 examples/s]28531 examples [00:27, 1039.00 examples/s]28636 examples [00:27, 1022.70 examples/s]28740 examples [00:27, 1027.28 examples/s]28849 examples [00:28, 1042.98 examples/s]28954 examples [00:28, 1030.18 examples/s]29066 examples [00:28, 1053.05 examples/s]29177 examples [00:28, 1069.00 examples/s]29290 examples [00:28, 1083.88 examples/s]29402 examples [00:28, 1092.50 examples/s]29514 examples [00:28, 1100.47 examples/s]29625 examples [00:28, 1085.38 examples/s]29734 examples [00:28, 1041.10 examples/s]29847 examples [00:29, 1065.11 examples/s]29954 examples [00:29, 1066.35 examples/s]30061 examples [00:29, 1028.24 examples/s]30172 examples [00:29, 1051.10 examples/s]30285 examples [00:29, 1071.74 examples/s]30397 examples [00:29, 1085.64 examples/s]30508 examples [00:29, 1090.36 examples/s]30618 examples [00:29, 1079.77 examples/s]30727 examples [00:29, 1070.68 examples/s]30835 examples [00:29, 1070.86 examples/s]30943 examples [00:30, 1069.57 examples/s]31051 examples [00:30, 1059.94 examples/s]31162 examples [00:30, 1073.31 examples/s]31271 examples [00:30, 1077.61 examples/s]31379 examples [00:30, 1069.95 examples/s]31487 examples [00:30, 1070.31 examples/s]31595 examples [00:30, 1071.34 examples/s]31703 examples [00:30, 1072.12 examples/s]31811 examples [00:30, 1066.53 examples/s]31918 examples [00:30, 1040.48 examples/s]32029 examples [00:31, 1057.57 examples/s]32142 examples [00:31, 1077.55 examples/s]32250 examples [00:31, 1014.79 examples/s]32356 examples [00:31, 1025.19 examples/s]32465 examples [00:31, 1041.46 examples/s]32570 examples [00:31, 1027.37 examples/s]32680 examples [00:31, 1045.81 examples/s]32786 examples [00:31, 1049.38 examples/s]32897 examples [00:31, 1065.86 examples/s]33008 examples [00:31, 1077.50 examples/s]33116 examples [00:32, 1050.64 examples/s]33222 examples [00:32, 1053.14 examples/s]33336 examples [00:32, 1075.33 examples/s]33444 examples [00:32, 1067.72 examples/s]33551 examples [00:32, 1062.22 examples/s]33660 examples [00:32, 1068.91 examples/s]33772 examples [00:32, 1083.51 examples/s]33881 examples [00:32, 1078.51 examples/s]33989 examples [00:32, 1065.16 examples/s]34101 examples [00:33, 1078.11 examples/s]34209 examples [00:33, 1070.44 examples/s]34317 examples [00:33, 1072.87 examples/s]34425 examples [00:33, 1060.29 examples/s]34532 examples [00:33, 1040.25 examples/s]34643 examples [00:33, 1058.75 examples/s]34753 examples [00:33, 1068.74 examples/s]34861 examples [00:33, 1068.17 examples/s]34968 examples [00:33, 1067.66 examples/s]35075 examples [00:33, 1047.75 examples/s]35184 examples [00:34, 1058.77 examples/s]35293 examples [00:34, 1066.75 examples/s]35400 examples [00:34, 1010.23 examples/s]35508 examples [00:34, 1029.72 examples/s]35617 examples [00:34, 1046.33 examples/s]35726 examples [00:34, 1058.26 examples/s]35835 examples [00:34, 1067.02 examples/s]35943 examples [00:34, 1070.16 examples/s]36051 examples [00:34, 1046.34 examples/s]36159 examples [00:34, 1054.91 examples/s]36265 examples [00:35, 1035.28 examples/s]36369 examples [00:35, 1031.59 examples/s]36476 examples [00:35, 1042.70 examples/s]36581 examples [00:35, 1043.67 examples/s]36687 examples [00:35, 1048.16 examples/s]36794 examples [00:35, 1054.37 examples/s]36904 examples [00:35, 1067.11 examples/s]37016 examples [00:35, 1082.38 examples/s]37125 examples [00:35, 1083.53 examples/s]37236 examples [00:35, 1089.79 examples/s]37346 examples [00:36, 1036.46 examples/s]37452 examples [00:36, 1041.00 examples/s]37562 examples [00:36, 1056.29 examples/s]37668 examples [00:36, 1047.59 examples/s]37780 examples [00:36, 1066.20 examples/s]37891 examples [00:36, 1077.39 examples/s]37999 examples [00:36, 1054.81 examples/s]38105 examples [00:36, 1055.70 examples/s]38211 examples [00:36, 1047.76 examples/s]38316 examples [00:37, 1036.34 examples/s]38423 examples [00:37, 1045.89 examples/s]38528 examples [00:37, 1027.88 examples/s]38631 examples [00:37, 1002.28 examples/s]38737 examples [00:37, 1017.07 examples/s]38845 examples [00:37, 1034.09 examples/s]38953 examples [00:37, 1046.21 examples/s]39058 examples [00:37, 1035.31 examples/s]39166 examples [00:37, 1047.62 examples/s]39271 examples [00:37, 1034.10 examples/s]39381 examples [00:38, 1052.23 examples/s]39487 examples [00:38, 1053.31 examples/s]39593 examples [00:38, 1029.78 examples/s]39697 examples [00:38, 1027.41 examples/s]39807 examples [00:38, 1048.07 examples/s]39915 examples [00:38, 1056.43 examples/s]40021 examples [00:38, 993.27 examples/s] 40129 examples [00:38, 1017.21 examples/s]40235 examples [00:38, 1023.99 examples/s]40338 examples [00:38, 999.86 examples/s] 40448 examples [00:39, 1025.88 examples/s]40556 examples [00:39, 1038.86 examples/s]40666 examples [00:39, 1055.15 examples/s]40775 examples [00:39, 1063.35 examples/s]40885 examples [00:39, 1071.55 examples/s]40993 examples [00:39, 1037.08 examples/s]41098 examples [00:39, 1036.96 examples/s]41205 examples [00:39, 1046.27 examples/s]41314 examples [00:39, 1057.36 examples/s]41420 examples [00:40, 1021.78 examples/s]41529 examples [00:40, 1040.66 examples/s]41634 examples [00:40, 1034.91 examples/s]41743 examples [00:40, 1050.01 examples/s]41849 examples [00:40, 997.51 examples/s] 41956 examples [00:40, 1016.07 examples/s]42059 examples [00:40, 1004.78 examples/s]42160 examples [00:40, 987.67 examples/s] 42262 examples [00:40, 994.84 examples/s]42369 examples [00:40, 1015.28 examples/s]42477 examples [00:41, 1031.06 examples/s]42586 examples [00:41, 1046.20 examples/s]42692 examples [00:41, 1049.64 examples/s]42801 examples [00:41, 1058.28 examples/s]42907 examples [00:41, 1057.49 examples/s]43013 examples [00:41, 1042.48 examples/s]43120 examples [00:41, 1047.40 examples/s]43227 examples [00:41, 1053.02 examples/s]43341 examples [00:41, 1075.25 examples/s]43449 examples [00:41, 1045.20 examples/s]43555 examples [00:42, 1047.74 examples/s]43660 examples [00:42, 1015.20 examples/s]43769 examples [00:42, 1034.61 examples/s]43876 examples [00:42, 1044.22 examples/s]43986 examples [00:42, 1059.75 examples/s]44098 examples [00:42, 1076.93 examples/s]44208 examples [00:42, 1081.34 examples/s]44317 examples [00:42, 1065.59 examples/s]44428 examples [00:42, 1078.34 examples/s]44536 examples [00:42, 1066.27 examples/s]44643 examples [00:43, 1061.11 examples/s]44750 examples [00:43, 1051.23 examples/s]44859 examples [00:43, 1061.34 examples/s]44966 examples [00:43, 1035.42 examples/s]45074 examples [00:43, 1045.52 examples/s]45179 examples [00:43, 1024.00 examples/s]45282 examples [00:43, 1019.83 examples/s]45385 examples [00:43, 1019.07 examples/s]45489 examples [00:43, 1024.62 examples/s]45592 examples [00:44, 1014.87 examples/s]45694 examples [00:44, 1012.19 examples/s]45805 examples [00:44, 1038.19 examples/s]45915 examples [00:44, 1055.18 examples/s]46023 examples [00:44, 1059.88 examples/s]46132 examples [00:44, 1066.76 examples/s]46243 examples [00:44, 1078.56 examples/s]46351 examples [00:44, 1065.15 examples/s]46458 examples [00:44, 1052.34 examples/s]46564 examples [00:44, 1052.89 examples/s]46670 examples [00:45, 1046.81 examples/s]46775 examples [00:45, 1042.93 examples/s]46882 examples [00:45, 1049.87 examples/s]46992 examples [00:45, 1061.47 examples/s]47099 examples [00:45, 1055.68 examples/s]47205 examples [00:45, 1052.54 examples/s]47311 examples [00:45, 1048.58 examples/s]47416 examples [00:45, 1045.15 examples/s]47523 examples [00:45, 1050.21 examples/s]47631 examples [00:45, 1058.94 examples/s]47740 examples [00:46, 1066.06 examples/s]47847 examples [00:46, 1062.63 examples/s]47954 examples [00:46, 1007.29 examples/s]48056 examples [00:46, 988.42 examples/s] 48156 examples [00:46, 966.27 examples/s]48263 examples [00:46, 994.11 examples/s]48363 examples [00:46, 985.43 examples/s]48466 examples [00:46, 998.24 examples/s]48577 examples [00:46, 1027.57 examples/s]48687 examples [00:46, 1047.63 examples/s]48793 examples [00:47, 1041.78 examples/s]48898 examples [00:47, 1034.06 examples/s]49009 examples [00:47, 1053.53 examples/s]49117 examples [00:47, 1060.35 examples/s]49224 examples [00:47, 1043.46 examples/s]49329 examples [00:47, 1018.32 examples/s]49432 examples [00:47, 1010.81 examples/s]49543 examples [00:47, 1038.32 examples/s]49649 examples [00:47, 1041.84 examples/s]49754 examples [00:48, 1033.52 examples/s]49862 examples [00:48, 1046.92 examples/s]49971 examples [00:48, 1059.44 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 15%|█▌        | 7505/50000 [00:00<00:00, 75046.31 examples/s] 42%|████▏     | 21121/50000 [00:00<00:00, 86723.16 examples/s] 69%|██████▊   | 34350/50000 [00:00<00:00, 96716.46 examples/s] 96%|█████████▌| 47755/50000 [00:00<00:00, 105533.81 examples/s]                                                                0 examples [00:00, ? examples/s]92 examples [00:00, 918.69 examples/s]202 examples [00:00, 966.12 examples/s]313 examples [00:00, 1003.13 examples/s]423 examples [00:00, 1030.09 examples/s]533 examples [00:00, 1047.87 examples/s]634 examples [00:00, 1034.25 examples/s]745 examples [00:00, 1055.31 examples/s]850 examples [00:00, 1053.31 examples/s]951 examples [00:00, 1039.70 examples/s]1053 examples [00:01, 1031.03 examples/s]1158 examples [00:01, 1035.43 examples/s]1260 examples [00:01, 1017.22 examples/s]1367 examples [00:01, 1030.62 examples/s]1470 examples [00:01, 810.94 examples/s] 1572 examples [00:01, 863.88 examples/s]1677 examples [00:01, 911.41 examples/s]1783 examples [00:01, 949.44 examples/s]1886 examples [00:01, 969.78 examples/s]1986 examples [00:02, 971.71 examples/s]2094 examples [00:02, 1001.72 examples/s]2204 examples [00:02, 1027.60 examples/s]2317 examples [00:02, 1055.62 examples/s]2427 examples [00:02, 1067.25 examples/s]2535 examples [00:02, 1054.56 examples/s]2646 examples [00:02, 1069.84 examples/s]2756 examples [00:02, 1078.14 examples/s]2867 examples [00:02, 1084.82 examples/s]2977 examples [00:02, 1087.29 examples/s]3086 examples [00:03, 1075.09 examples/s]3194 examples [00:03, 1032.96 examples/s]3298 examples [00:03, 1027.86 examples/s]3402 examples [00:03, 1027.52 examples/s]3508 examples [00:03, 1036.79 examples/s]3612 examples [00:03, 1015.65 examples/s]3719 examples [00:03, 1029.66 examples/s]3825 examples [00:03, 1036.14 examples/s]3929 examples [00:03, 1031.81 examples/s]4033 examples [00:03, 1028.80 examples/s]4146 examples [00:04, 1054.55 examples/s]4256 examples [00:04, 1067.51 examples/s]4365 examples [00:04, 1071.16 examples/s]4477 examples [00:04, 1083.24 examples/s]4587 examples [00:04, 1086.12 examples/s]4696 examples [00:04, 1085.84 examples/s]4807 examples [00:04, 1092.40 examples/s]4917 examples [00:04, 1077.43 examples/s]5031 examples [00:04, 1093.65 examples/s]5141 examples [00:04, 1087.37 examples/s]5254 examples [00:05, 1097.04 examples/s]5367 examples [00:05, 1103.89 examples/s]5478 examples [00:05, 1095.49 examples/s]5588 examples [00:05, 1091.10 examples/s]5698 examples [00:05, 1090.33 examples/s]5808 examples [00:05, 1084.57 examples/s]5919 examples [00:05, 1090.69 examples/s]6030 examples [00:05, 1094.75 examples/s]6140 examples [00:05, 1095.45 examples/s]6253 examples [00:05, 1103.32 examples/s]6364 examples [00:06, 1102.07 examples/s]6475 examples [00:06, 1079.01 examples/s]6584 examples [00:06, 1068.73 examples/s]6691 examples [00:06, 1068.43 examples/s]6798 examples [00:06, 1062.24 examples/s]6905 examples [00:06, 1007.50 examples/s]7016 examples [00:06, 1035.27 examples/s]7127 examples [00:06, 1053.94 examples/s]7234 examples [00:06, 1057.00 examples/s]7341 examples [00:07, 1023.51 examples/s]7444 examples [00:07, 1015.16 examples/s]7552 examples [00:07, 1031.93 examples/s]7656 examples [00:07, 994.61 examples/s] 7758 examples [00:07, 1000.78 examples/s]7861 examples [00:07, 1009.17 examples/s]7970 examples [00:07, 1029.57 examples/s]8078 examples [00:07, 1044.01 examples/s]8188 examples [00:07, 1058.16 examples/s]8299 examples [00:07, 1069.52 examples/s]8411 examples [00:08, 1083.76 examples/s]8520 examples [00:08, 1077.67 examples/s]8628 examples [00:08, 1077.44 examples/s]8737 examples [00:08, 1079.53 examples/s]8846 examples [00:08, 1070.78 examples/s]8956 examples [00:08, 1077.39 examples/s]9064 examples [00:08, 1072.80 examples/s]9176 examples [00:08, 1084.89 examples/s]9289 examples [00:08, 1095.84 examples/s]9399 examples [00:08, 1095.97 examples/s]9509 examples [00:09, 1085.04 examples/s]9621 examples [00:09, 1092.80 examples/s]9731 examples [00:09, 1078.25 examples/s]9843 examples [00:09, 1090.26 examples/s]9953 examples [00:09, 1060.57 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompletePO3Q83/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompletePO3Q83/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fb26969b950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fb26969b950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fb26969b950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fb1f2ae2f60>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fb1f2b54fd0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fb26969b950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fb26969b950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fb26969b950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fb24fa5b080>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fb24fa5b0f0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7fb1e10bf2f0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7fb1e10bf2f0> 

  function with postional parmater data_info <function split_train_valid at 0x7fb1e10bf2f0> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7fb1e10bf400> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7fb1e10bf400> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7fb1e10bf400> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.5)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.1)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.2)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.1)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=5c17e2c6d5e71e7ea48c08ffb588ec44789dec18c0576eb9f3bf4a0e8e80a881
  Stored in directory: /tmp/pip-ephem-wheel-cache-yqdoh5e_/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:01<47:05:00, 5.09kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:01<33:11:19, 7.22kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<23:16:58, 10.3kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:02<16:17:57, 14.7kB/s].vector_cache/glove.6B.zip:   0%|          | 3.60M/862M [00:02<11:22:36, 21.0kB/s].vector_cache/glove.6B.zip:   1%|          | 8.43M/862M [00:02<7:55:13, 29.9kB/s] .vector_cache/glove.6B.zip:   1%|▏         | 12.3M/862M [00:02<5:31:14, 42.8kB/s].vector_cache/glove.6B.zip:   2%|▏         | 17.4M/862M [00:02<3:50:34, 61.1kB/s].vector_cache/glove.6B.zip:   3%|▎         | 21.7M/862M [00:02<2:40:41, 87.2kB/s].vector_cache/glove.6B.zip:   3%|▎         | 25.8M/862M [00:02<1:52:01, 124kB/s] .vector_cache/glove.6B.zip:   4%|▎         | 30.7M/862M [00:02<1:18:04, 178kB/s].vector_cache/glove.6B.zip:   4%|▍         | 35.8M/862M [00:02<54:23, 253kB/s]  .vector_cache/glove.6B.zip:   5%|▍         | 39.1M/862M [00:02<38:02, 361kB/s].vector_cache/glove.6B.zip:   5%|▌         | 44.4M/862M [00:03<26:32, 514kB/s].vector_cache/glove.6B.zip:   6%|▌         | 47.9M/862M [00:03<18:36, 729kB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.7M/862M [00:03<13:17, 1.02MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.9M/862M [00:05<11:11, 1.20MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.1M/862M [00:05<11:27, 1.17MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.6M/862M [00:05<08:47, 1.53MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.7M/862M [00:05<06:20, 2.11MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.1M/862M [00:07<09:09, 1.46MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.4M/862M [00:07<07:55, 1.69MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.8M/862M [00:07<05:55, 2.25MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.2M/862M [00:09<06:57, 1.91MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.6M/862M [00:09<06:12, 2.14MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.2M/862M [00:09<04:40, 2.84MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.4M/862M [00:11<06:26, 2.06MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.6M/862M [00:11<07:11, 1.84MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.3M/862M [00:11<05:35, 2.36MB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.8M/862M [00:11<04:03, 3.24MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.5M/862M [00:13<12:34, 1.05MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.9M/862M [00:13<10:09, 1.30MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.4M/862M [00:13<07:23, 1.77MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.6M/862M [00:15<08:13, 1.59MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.0M/862M [00:15<07:05, 1.85MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.6M/862M [00:15<05:16, 2.47MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.7M/862M [00:17<06:46, 1.92MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.9M/862M [00:17<07:24, 1.76MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.7M/862M [00:17<05:43, 2.27MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.5M/862M [00:17<04:13, 3.07MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.8M/862M [00:19<07:32, 1.72MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.2M/862M [00:19<06:35, 1.96MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.8M/862M [00:19<04:56, 2.62MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.9M/862M [00:21<06:28, 1.99MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.1M/862M [00:21<07:02, 1.83MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.9M/862M [00:21<05:34, 2.31MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.9M/862M [00:21<04:01, 3.19MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.1M/862M [00:23<36:50, 348kB/s] .vector_cache/glove.6B.zip:  11%|█         | 93.4M/862M [00:23<27:08, 472kB/s].vector_cache/glove.6B.zip:  11%|█         | 95.0M/862M [00:23<19:14, 664kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.2M/862M [00:25<16:25, 776kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.4M/862M [00:25<14:05, 904kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.1M/862M [00:25<10:24, 1.22MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 100M/862M [00:25<07:26, 1.71MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:27<11:30, 1.10MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:27<09:20, 1.36MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:27<06:51, 1.85MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:29<07:44, 1.63MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:29<07:59, 1.58MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:29<06:08, 2.05MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:29<04:27, 2.81MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:31<09:09, 1.37MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:31<07:40, 1.63MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:31<05:38, 2.22MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<06:52, 1.82MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:33<06:06, 2.04MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:33<04:34, 2.72MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<06:06, 2.03MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:35<06:47, 1.83MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:35<05:16, 2.35MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:35<03:52, 3.19MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<07:51, 1.57MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:37<06:44, 1.83MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:37<05:01, 2.45MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<06:23, 1.92MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<06:58, 1.76MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:39<05:24, 2.26MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:39<03:59, 3.07MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<07:11, 1.70MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<06:07, 1.99MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:41<04:35, 2.65MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:42<06:01, 2.01MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<05:26, 2.23MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:43<04:03, 2.98MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:44<05:42, 2.12MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<06:25, 1.88MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:45<05:06, 2.36MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 142M/862M [00:45<03:42, 3.24MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 142M/862M [00:46<11:32:48, 17.3kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<8:05:57, 24.7kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<5:39:45, 35.2kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<3:59:52, 49.7kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<2:50:24, 70.0kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<1:59:41, 99.5kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:49<1:23:35, 142kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:50<1:12:12, 164kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<51:45, 229kB/s]  .vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<36:27, 324kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<28:12, 418kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<22:08, 533kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<16:04, 733kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:54<13:06, 894kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:54<10:09, 1.15MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<07:25, 1.58MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:54<05:20, 2.18MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:56<1:13:13, 159kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:56<52:22, 222kB/s]  .vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<36:52, 315kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:58<28:29, 407kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:58<20:54, 554kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<14:53, 776kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [01:00<13:08, 876kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<10:21, 1.11MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<07:29, 1.53MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:02<07:57, 1.44MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<06:43, 1.70MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<04:59, 2.29MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:04<06:10, 1.84MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<06:38, 1.71MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<05:13, 2.18MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:06<05:28, 2.07MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:06<06:08, 1.84MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<04:52, 2.32MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:06<03:32, 3.17MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:08<1:22:48, 136kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:08<59:06, 190kB/s]  .vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<41:34, 270kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:10<31:36, 353kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:10<24:23, 458kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<17:36, 634kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:12<14:04, 789kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:12<12:06, 917kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<08:56, 1.24MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:12<06:23, 1.73MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:14<09:55, 1.11MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:14<08:03, 1.37MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<05:51, 1.88MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:16<06:40, 1.64MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:16<05:46, 1.90MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<04:16, 2.56MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:18<05:35, 1.95MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:18<06:07, 1.78MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:18<04:45, 2.29MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:18<03:26, 3.15MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:20<11:33, 936kB/s] .vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:20<09:11, 1.18MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<06:41, 1.61MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:22<07:11, 1.49MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:22<07:08, 1.50MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:22<05:32, 1.94MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:22<03:58, 2.69MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<10:22:39, 17.2kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:24<7:16:42, 24.5kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<5:05:11, 34.9kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<3:35:24, 49.3kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:26<2:31:45, 70.0kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<1:46:14, 99.7kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<1:16:37, 138kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:28<55:46, 189kB/s]  .vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:28<39:26, 267kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:28<27:39, 380kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<24:25, 429kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<18:09, 577kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<12:54, 810kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:31<11:27, 909kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:31<10:09, 1.03MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:32<07:37, 1.36MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<05:25, 1.91MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:33<10:01:29, 17.2kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<7:01:42, 24.5kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:34<4:54:45, 35.0kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:35<3:27:56, 49.4kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 246M/862M [01:35<2:26:20, 70.2kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<1:42:27, 100kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:37<1:13:52, 138kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<52:42, 194kB/s]  .vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<37:01, 275kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:39<28:15, 359kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<20:46, 488kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<14:43, 687kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<12:40, 795kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<09:53, 1.02MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<07:09, 1.40MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:43<07:22, 1.36MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:43<06:10, 1.62MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<04:33, 2.19MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:45<05:32, 1.80MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:45<04:43, 2.10MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<03:33, 2.79MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:47<04:48, 2.05MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:47<05:22, 1.84MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<04:15, 2.32MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:49<04:33, 2.15MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:49<04:02, 2.43MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<03:04, 3.19MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:51<04:26, 2.19MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:51<05:05, 1.91MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<03:58, 2.44MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<02:55, 3.31MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:53<06:05, 1.59MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<05:15, 1.83MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<03:52, 2.48MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:55<04:58, 1.93MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:55<05:25, 1.77MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:55<04:12, 2.28MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:55<03:03, 3.12MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:57<07:24, 1.29MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:57<06:10, 1.54MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<04:33, 2.09MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:59<05:22, 1.76MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:59<05:41, 1.66MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<04:27, 2.11MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:01<04:37, 2.03MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:01<04:11, 2.24MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<03:07, 2.99MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:03<04:23, 2.13MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:03<04:53, 1.90MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:03<03:53, 2.39MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:05<04:12, 2.20MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:05<03:54, 2.37MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<02:55, 3.15MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:07<04:11, 2.19MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:07<04:53, 1.88MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:07<03:53, 2.35MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:07<02:49, 3.22MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:09<1:07:05, 136kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:09<47:52, 190kB/s]  .vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<33:39, 270kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:11<25:35, 354kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:11<19:44, 458kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:11<14:15, 633kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 323M/862M [02:13<11:23, 789kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:13<08:52, 1.01MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:13<06:23, 1.40MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<06:33, 1.36MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:15<06:24, 1.39MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:15<04:51, 1.83MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<03:30, 2.53MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:16<06:55, 1.28MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:17<05:46, 1.53MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:17<04:15, 2.07MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<05:00, 1.75MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:19<04:23, 2.00MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<03:15, 2.69MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<04:20, 2.00MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<04:49, 1.81MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:21<03:48, 2.28MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<04:03, 2.13MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<03:42, 2.32MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:23<02:48, 3.06MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:24<03:58, 2.16MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<03:37, 2.36MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:25<02:43, 3.12MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:26<03:55, 2.17MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:26<04:28, 1.90MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:27<03:33, 2.39MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:28<03:50, 2.19MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<04:24, 1.91MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<03:27, 2.44MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<02:30, 3.34MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:30<07:00, 1.19MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<05:46, 1.45MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<04:12, 1.98MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<04:51, 1.71MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<05:05, 1.63MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<03:55, 2.11MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<02:57, 2.79MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:33<02:24, 3.41MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:35<8:52:18, 15.4kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:35<6:13:55, 21.9kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:35<4:21:37, 31.3kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:37<3:03:25, 44.4kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:37<2:09:06, 63.0kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<1:30:23, 89.7kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:39<1:04:36, 125kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:39<46:54, 172kB/s]  .vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:39<33:08, 243kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<23:12, 345kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:41<20:03, 399kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:41<14:51, 538kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<10:32, 755kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:43<09:12, 862kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:43<08:02, 985kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:43<05:57, 1.33MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<04:15, 1.85MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:45<07:33, 1.04MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:45<06:05, 1.29MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<04:27, 1.76MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:47<04:56, 1.58MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:47<05:02, 1.54MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:47<03:51, 2.01MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<02:49, 2.75MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<04:50, 1.59MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:49<04:11, 1.84MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:49<03:07, 2.46MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<03:57, 1.93MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:51<03:32, 2.16MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<02:40, 2.86MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<03:38, 2.08MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:53<03:18, 2.29MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<02:30, 3.02MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:54<03:31, 2.13MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<03:13, 2.33MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:55<02:26, 3.07MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:56<03:27, 2.15MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:56<03:56, 1.89MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:57<03:07, 2.37MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:58<03:22, 2.18MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<03:06, 2.37MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:59<02:21, 3.12MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<03:21, 2.17MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<03:47, 1.93MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<02:57, 2.47MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<02:11, 3.31MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<03:51, 1.88MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<03:27, 2.09MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<02:34, 2.80MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:04<03:25, 2.10MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:04<03:54, 1.83MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<03:03, 2.34MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<02:14, 3.17MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:06<04:01, 1.76MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:06<03:27, 2.05MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<02:36, 2.72MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:08<03:23, 2.07MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:08<03:07, 2.25MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:08<02:21, 2.96MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:10<03:13, 2.16MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:10<03:43, 1.87MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<02:54, 2.38MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<02:07, 3.26MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:12<05:11, 1.33MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<04:21, 1.58MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<03:13, 2.13MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 453M/862M [03:14<03:47, 1.80MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:14<04:05, 1.66MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:14<03:09, 2.15MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<02:18, 2.94MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:16<04:14, 1.59MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:16<03:41, 1.83MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<02:43, 2.47MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:18<03:24, 1.96MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:18<03:47, 1.76MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<02:57, 2.26MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:18<02:08, 3.10MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:20<05:19, 1.24MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:20<04:25, 1.49MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<03:14, 2.04MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:22<03:45, 1.74MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:22<04:00, 1.63MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:22<03:09, 2.07MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:22<02:16, 2.86MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:24<17:49, 364kB/s] .vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:24<13:09, 492kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<09:19, 692kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:26<07:56, 808kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:26<06:54, 928kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:26<05:10, 1.24MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:26<03:40, 1.73MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:28<22:43, 279kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:28<16:32, 383kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<11:41, 540kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:30<09:32, 657kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:30<07:48, 803kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:30<05:47, 1.08MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<04:07, 1.51MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:32<05:31, 1.12MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:32<04:31, 1.37MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<03:17, 1.88MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:34<03:41, 1.66MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:34<03:13, 1.90MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<02:24, 2.53MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:36<03:02, 1.99MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:36<03:24, 1.78MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:36<02:39, 2.28MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<01:55, 3.12MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:38<04:17, 1.40MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:38<03:37, 1.65MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<02:39, 2.24MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:40<03:12, 1.85MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:40<03:30, 1.69MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:40<02:45, 2.14MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<01:59, 2.94MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:42<19:58, 293kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:42<14:35, 401kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<10:17, 566kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:44<08:28, 683kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:44<07:09, 809kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<05:15, 1.10MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<03:43, 1.54MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:46<06:27, 887kB/s] .vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:46<05:07, 1.12MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<03:42, 1.53MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:48<03:51, 1.46MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:48<03:53, 1.45MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:48<03:01, 1.87MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:48<02:09, 2.58MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:50<14:56, 374kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:50<11:02, 505kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<07:48, 711kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:52<06:41, 825kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:52<05:50, 943kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:52<04:22, 1.26MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<03:05, 1.76MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:54<15:49, 344kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:54<11:38, 467kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<08:13, 658kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:56<06:56, 774kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:56<05:59, 896kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:56<04:25, 1.21MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<03:09, 1.69MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:58<04:41, 1.13MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:58<03:50, 1.38MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<02:47, 1.89MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:00<03:08, 1.67MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:00<03:18, 1.58MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:00<02:32, 2.06MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<01:49, 2.83MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:02<04:06, 1.26MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:02<03:25, 1.51MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<02:30, 2.05MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:04<02:53, 1.77MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:04<03:06, 1.64MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:04<02:26, 2.09MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<01:45, 2.87MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<17:12, 292kB/s] .vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:06<12:33, 400kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<08:51, 564kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:07<07:17, 681kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:08<05:36, 884kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:08<04:02, 1.22MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<03:56, 1.24MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:10<03:47, 1.29MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:10<02:54, 1.67MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<02:04, 2.33MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:12<05:24, 892kB/s] .vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:12<04:45, 1.01MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:12<03:33, 1.35MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:14<03:14, 1.47MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:14<02:45, 1.71MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<02:02, 2.32MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:16<02:28, 1.90MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:16<02:12, 2.11MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<01:39, 2.80MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:18<02:13, 2.07MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:18<02:32, 1.82MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:18<01:58, 2.33MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<01:26, 3.18MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<03:19, 1.37MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:20<02:48, 1.62MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:20<02:03, 2.19MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<02:27, 1.82MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:22<02:40, 1.67MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:22<02:06, 2.12MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:22<01:30, 2.92MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<12:04, 365kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:24<08:54, 494kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<06:18, 694kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<05:21, 809kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:26<04:40, 928kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:26<03:26, 1.25MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<02:26, 1.75MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:27<03:57, 1.08MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:27<03:13, 1.32MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:28<02:20, 1.81MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:29<02:34, 1.63MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:29<02:39, 1.58MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:30<02:02, 2.04MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<01:28, 2.82MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 614M/862M [04:31<03:49, 1.08MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<03:06, 1.33MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:32<02:15, 1.82MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<02:29, 1.63MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<02:10, 1.87MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:34<01:37, 2.49MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<02:01, 1.97MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<02:15, 1.77MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:36<01:46, 2.23MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<01:16, 3.08MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:37<19:51, 198kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:37<14:18, 274kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<10:02, 388kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:39<07:49, 493kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:39<06:14, 616kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<04:33, 842kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<03:11, 1.19MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:41<22:10, 171kB/s] .vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:41<15:53, 238kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<11:08, 337kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:43<08:34, 433kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<06:22, 581kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<04:31, 816kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:45<03:59, 915kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<03:10, 1.15MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<02:17, 1.57MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:47<02:25, 1.48MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<02:27, 1.46MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<01:52, 1.90MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:47<01:20, 2.63MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:49<02:54, 1.20MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:49<02:20, 1.50MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<01:42, 2.05MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:49<01:13, 2.80MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:51<07:21, 468kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:51<05:50, 589kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<04:13, 811kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:51<02:57, 1.14MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:53<04:18, 782kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:53<03:18, 1.02MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<02:22, 1.41MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:53<01:42, 1.94MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:55<06:04, 543kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:55<04:35, 718kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<03:16, 1.00MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:57<03:01, 1.07MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [04:57<02:47, 1.15MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<02:05, 1.54MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<01:29, 2.13MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:59<02:46, 1.14MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:59<02:16, 1.39MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<01:39, 1.89MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:01<01:51, 1.66MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:01<01:55, 1.60MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:01<01:28, 2.08MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:01<01:04, 2.85MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:03<02:02, 1.48MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:03<01:44, 1.73MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<01:17, 2.32MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:05<01:33, 1.89MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:05<01:24, 2.10MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<01:03, 2.78MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:07<01:22, 2.10MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:07<01:35, 1.82MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [05:07<01:14, 2.31MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:07<00:53, 3.19MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:09<04:04, 692kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:09<03:08, 897kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<02:14, 1.24MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:11<02:11, 1.26MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:11<02:06, 1.30MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:11<01:37, 1.69MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:11<01:08, 2.34MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:13<08:52, 302kB/s] .vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:13<06:29, 412kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<04:34, 580kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:15<03:44, 698kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:15<03:10, 824kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:15<02:19, 1.12MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:15<01:38, 1.56MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:17<02:47, 911kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:17<02:13, 1.14MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 711M/862M [05:17<01:35, 1.57MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:19<01:40, 1.48MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:19<01:41, 1.46MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:19<01:17, 1.91MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<00:55, 2.62MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:21<01:33, 1.55MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:21<01:17, 1.85MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<00:57, 2.46MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:22<01:10, 1.98MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:23<01:19, 1.77MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:23<01:01, 2.27MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<00:44, 3.11MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<01:36, 1.41MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:25<01:21, 1.66MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<01:00, 2.24MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<01:11, 1.85MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:27<01:16, 1.72MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:27<01:00, 2.18MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<00:42, 3.01MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:28<11:41, 182kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:29<08:21, 255kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:29<05:49, 361kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<04:02, 512kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:30<05:50, 353kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:31<04:31, 456kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:31<03:14, 633kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<02:14, 894kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:32<02:47, 714kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<02:09, 921kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:33<01:32, 1.28MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<01:29, 1.29MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<01:12, 1.59MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:35<00:53, 2.14MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:35<00:37, 2.95MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<07:55, 234kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<05:54, 313kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:37<04:12, 438kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:37<02:53, 620kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<16:37, 108kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<11:47, 151kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<08:10, 215kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<05:59, 287kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<04:32, 377kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:41<03:14, 526kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<02:14, 744kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:42<02:32, 648kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<01:57, 843kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<01:22, 1.17MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:44<01:18, 1.21MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<01:04, 1.47MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<00:46, 1.99MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:46<00:53, 1.71MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<00:56, 1.61MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<00:43, 2.07MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<00:30, 2.84MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:48<00:57, 1.51MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:48<00:48, 1.76MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:35, 2.36MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:50<00:43, 1.90MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:50<00:47, 1.72MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:37, 2.18MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:50<00:26, 3.00MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:52<03:27, 378kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:52<02:32, 511kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<01:46, 718kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:54<01:29, 831kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:54<01:18, 949kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<00:57, 1.28MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:54<00:39, 1.79MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:56<01:24, 828kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:56<01:06, 1.05MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:47, 1.44MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:58<00:47, 1.40MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:58<00:46, 1.40MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:35, 1.84MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<00:25, 2.52MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:00<00:39, 1.56MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<00:34, 1.80MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:24, 2.41MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:02<00:29, 1.94MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<00:26, 2.14MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:02<00:19, 2.83MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:04<00:25, 2.11MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:04<00:28, 1.84MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:22, 2.36MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:16, 3.18MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:06<00:27, 1.78MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:06<00:29, 1.65MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:23, 2.10MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:06<00:15, 2.90MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:08<01:55, 393kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:08<01:24, 530kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<00:58, 743kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:10<00:48, 854kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:10<00:42, 969kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:30, 1.30MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:10<00:21, 1.82MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:12<00:30, 1.20MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:12<00:25, 1.45MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:17, 1.98MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:14<00:19, 1.72MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:14<00:16, 1.95MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:11, 2.60MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:16<00:14, 2.02MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:16<00:12, 2.21MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:09, 2.91MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:18<00:11, 2.14MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:18<00:13, 1.86MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:10, 2.32MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:18<00:06, 3.20MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:20<00:55, 367kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:20<00:40, 496kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:26, 698kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:22<00:20, 813kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:22<00:15, 1.04MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:09, 1.43MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:24<00:08, 1.38MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:24<00:08, 1.39MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:24<00:06, 1.81MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:24<00:03, 2.50MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:26<00:16, 494kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:26<00:11, 656kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:06, 915kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:28<00:03, 1.00MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:28<00:03, 1.10MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:28<00:02, 1.45MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<28:51:26,  3.85it/s]  0%|          | 875/400000 [00:00<20:09:35,  5.50it/s]  0%|          | 1752/400000 [00:00<14:05:04,  7.85it/s]  1%|          | 2626/400000 [00:00<9:50:29, 11.22it/s]   1%|          | 3490/400000 [00:00<6:52:40, 16.01it/s]  1%|          | 4377/400000 [00:00<4:48:26, 22.86it/s]  1%|▏         | 5263/400000 [00:00<3:21:41, 32.62it/s]  2%|▏         | 6153/400000 [00:00<2:21:04, 46.53it/s]  2%|▏         | 7041/400000 [00:01<1:38:45, 66.32it/s]  2%|▏         | 7914/400000 [00:01<1:09:12, 94.43it/s]  2%|▏         | 8815/400000 [00:01<48:32, 134.30it/s]   2%|▏         | 9679/400000 [00:01<34:07, 190.59it/s]  3%|▎         | 10557/400000 [00:01<24:03, 269.76it/s]  3%|▎         | 11424/400000 [00:01<17:01, 380.24it/s]  3%|▎         | 12289/400000 [00:01<12:07, 532.95it/s]  3%|▎         | 13173/400000 [00:01<08:41, 742.16it/s]  4%|▎         | 14044/400000 [00:01<06:17, 1022.87it/s]  4%|▎         | 14912/400000 [00:01<04:38, 1380.91it/s]  4%|▍         | 15779/400000 [00:02<03:28, 1846.61it/s]  4%|▍         | 16623/400000 [00:02<02:39, 2410.18it/s]  4%|▍         | 17465/400000 [00:02<02:05, 3059.78it/s]  5%|▍         | 18360/400000 [00:02<01:40, 3812.39it/s]  5%|▍         | 19242/400000 [00:02<01:22, 4594.36it/s]  5%|▌         | 20105/400000 [00:02<01:11, 5332.41it/s]  5%|▌         | 20973/400000 [00:02<01:02, 6030.02it/s]  5%|▌         | 21868/400000 [00:02<00:56, 6683.38it/s]  6%|▌         | 22758/400000 [00:02<00:52, 7222.23it/s]  6%|▌         | 23642/400000 [00:02<00:49, 7640.39it/s]  6%|▌         | 24521/400000 [00:03<00:47, 7948.99it/s]  6%|▋         | 25400/400000 [00:03<00:45, 8160.37it/s]  7%|▋         | 26300/400000 [00:03<00:44, 8395.22it/s]  7%|▋         | 27189/400000 [00:03<00:43, 8535.17it/s]  7%|▋         | 28081/400000 [00:03<00:43, 8646.25it/s]  7%|▋         | 28968/400000 [00:03<00:42, 8689.97it/s]  7%|▋         | 29853/400000 [00:03<00:42, 8723.35it/s]  8%|▊         | 30748/400000 [00:03<00:42, 8788.13it/s]  8%|▊         | 31638/400000 [00:03<00:41, 8819.26it/s]  8%|▊         | 32526/400000 [00:03<00:41, 8820.94it/s]  8%|▊         | 33412/400000 [00:04<00:41, 8830.89it/s]  9%|▊         | 34298/400000 [00:04<00:42, 8676.11it/s]  9%|▉         | 35169/400000 [00:04<00:42, 8639.78it/s]  9%|▉         | 36038/400000 [00:04<00:42, 8653.92it/s]  9%|▉         | 36923/400000 [00:04<00:41, 8711.11it/s]  9%|▉         | 37796/400000 [00:04<00:41, 8698.44it/s] 10%|▉         | 38667/400000 [00:04<00:41, 8691.75it/s] 10%|▉         | 39552/400000 [00:04<00:41, 8736.67it/s] 10%|█         | 40427/400000 [00:04<00:41, 8719.60it/s] 10%|█         | 41300/400000 [00:05<00:41, 8605.62it/s] 11%|█         | 42172/400000 [00:05<00:41, 8637.89it/s] 11%|█         | 43037/400000 [00:05<00:41, 8603.27it/s] 11%|█         | 43915/400000 [00:05<00:41, 8655.34it/s] 11%|█         | 44781/400000 [00:05<00:41, 8560.29it/s] 11%|█▏        | 45638/400000 [00:05<00:41, 8455.42it/s] 12%|█▏        | 46522/400000 [00:05<00:41, 8566.45it/s] 12%|█▏        | 47380/400000 [00:05<00:41, 8419.95it/s] 12%|█▏        | 48275/400000 [00:05<00:41, 8571.15it/s] 12%|█▏        | 49164/400000 [00:05<00:40, 8663.17it/s] 13%|█▎        | 50050/400000 [00:06<00:40, 8719.75it/s] 13%|█▎        | 50923/400000 [00:06<00:40, 8646.84it/s] 13%|█▎        | 51796/400000 [00:06<00:40, 8670.57it/s] 13%|█▎        | 52677/400000 [00:06<00:39, 8709.63it/s] 13%|█▎        | 53563/400000 [00:06<00:39, 8751.63it/s] 14%|█▎        | 54447/400000 [00:06<00:39, 8775.73it/s] 14%|█▍        | 55325/400000 [00:06<00:39, 8691.90it/s] 14%|█▍        | 56202/400000 [00:06<00:39, 8714.26it/s] 14%|█▍        | 57086/400000 [00:06<00:39, 8750.94it/s] 14%|█▍        | 57973/400000 [00:06<00:38, 8783.74it/s] 15%|█▍        | 58852/400000 [00:07<00:38, 8756.62it/s] 15%|█▍        | 59728/400000 [00:07<00:39, 8536.66it/s] 15%|█▌        | 60587/400000 [00:07<00:39, 8550.14it/s] 15%|█▌        | 61479/400000 [00:07<00:39, 8656.39it/s] 16%|█▌        | 62374/400000 [00:07<00:38, 8741.59it/s] 16%|█▌        | 63259/400000 [00:07<00:38, 8773.56it/s] 16%|█▌        | 64137/400000 [00:07<00:38, 8646.35it/s] 16%|█▋        | 65028/400000 [00:07<00:38, 8723.18it/s] 16%|█▋        | 65919/400000 [00:07<00:38, 8777.81it/s] 17%|█▋        | 66798/400000 [00:07<00:37, 8776.23it/s] 17%|█▋        | 67677/400000 [00:08<00:38, 8644.70it/s] 17%|█▋        | 68543/400000 [00:08<00:38, 8554.78it/s] 17%|█▋        | 69437/400000 [00:08<00:38, 8665.39it/s] 18%|█▊        | 70305/400000 [00:08<00:38, 8620.30it/s] 18%|█▊        | 71186/400000 [00:08<00:37, 8675.07it/s] 18%|█▊        | 72076/400000 [00:08<00:37, 8740.11it/s] 18%|█▊        | 72951/400000 [00:08<00:37, 8675.72it/s] 18%|█▊        | 73839/400000 [00:08<00:37, 8734.09it/s] 19%|█▊        | 74735/400000 [00:08<00:36, 8799.30it/s] 19%|█▉        | 75640/400000 [00:08<00:36, 8872.77it/s] 19%|█▉        | 76543/400000 [00:09<00:36, 8917.26it/s] 19%|█▉        | 77436/400000 [00:09<00:36, 8777.44it/s] 20%|█▉        | 78315/400000 [00:09<00:36, 8758.30it/s] 20%|█▉        | 79200/400000 [00:09<00:36, 8785.02it/s] 20%|██        | 80081/400000 [00:09<00:36, 8789.63it/s] 20%|██        | 80961/400000 [00:09<00:36, 8791.21it/s] 20%|██        | 81841/400000 [00:09<00:36, 8714.42it/s] 21%|██        | 82745/400000 [00:09<00:36, 8808.52it/s] 21%|██        | 83627/400000 [00:09<00:36, 8775.68it/s] 21%|██        | 84506/400000 [00:09<00:35, 8777.70it/s] 21%|██▏       | 85385/400000 [00:10<00:36, 8702.42it/s] 22%|██▏       | 86256/400000 [00:10<00:36, 8674.80it/s] 22%|██▏       | 87175/400000 [00:10<00:35, 8823.09it/s] 22%|██▏       | 88059/400000 [00:10<00:35, 8803.36it/s] 22%|██▏       | 88947/400000 [00:10<00:35, 8825.04it/s] 22%|██▏       | 89836/400000 [00:10<00:35, 8843.87it/s] 23%|██▎       | 90721/400000 [00:10<00:35, 8788.05it/s] 23%|██▎       | 91628/400000 [00:10<00:34, 8869.00it/s] 23%|██▎       | 92549/400000 [00:10<00:34, 8966.62it/s] 23%|██▎       | 93447/400000 [00:10<00:34, 8956.60it/s] 24%|██▎       | 94344/400000 [00:11<00:34, 8937.78it/s] 24%|██▍       | 95239/400000 [00:11<00:34, 8840.80it/s] 24%|██▍       | 96139/400000 [00:11<00:34, 8885.76it/s] 24%|██▍       | 97043/400000 [00:11<00:33, 8930.45it/s] 24%|██▍       | 97939/400000 [00:11<00:33, 8937.48it/s] 25%|██▍       | 98833/400000 [00:11<00:33, 8892.50it/s] 25%|██▍       | 99723/400000 [00:11<00:33, 8888.98it/s] 25%|██▌       | 100616/400000 [00:11<00:33, 8901.19it/s] 25%|██▌       | 101517/400000 [00:11<00:33, 8931.77it/s] 26%|██▌       | 102413/400000 [00:11<00:33, 8939.99it/s] 26%|██▌       | 103308/400000 [00:12<00:33, 8808.00it/s] 26%|██▌       | 104195/400000 [00:12<00:33, 8820.78it/s] 26%|██▋       | 105098/400000 [00:12<00:33, 8882.12it/s] 27%|██▋       | 106009/400000 [00:12<00:32, 8945.48it/s] 27%|██▋       | 106914/400000 [00:12<00:32, 8975.14it/s] 27%|██▋       | 107812/400000 [00:12<00:32, 8898.74it/s] 27%|██▋       | 108703/400000 [00:12<00:32, 8890.72it/s] 27%|██▋       | 109609/400000 [00:12<00:32, 8938.61it/s] 28%|██▊       | 110563/400000 [00:12<00:31, 9110.62it/s] 28%|██▊       | 111476/400000 [00:12<00:31, 9100.22it/s] 28%|██▊       | 112387/400000 [00:13<00:31, 8995.03it/s] 28%|██▊       | 113288/400000 [00:13<00:32, 8902.48it/s] 29%|██▊       | 114179/400000 [00:13<00:32, 8881.31it/s] 29%|██▉       | 115072/400000 [00:13<00:32, 8894.94it/s] 29%|██▉       | 115962/400000 [00:13<00:31, 8896.32it/s] 29%|██▉       | 116852/400000 [00:13<00:31, 8852.38it/s] 29%|██▉       | 117738/400000 [00:13<00:32, 8777.58it/s] 30%|██▉       | 118625/400000 [00:13<00:31, 8803.27it/s] 30%|██▉       | 119531/400000 [00:13<00:31, 8876.52it/s] 30%|███       | 120419/400000 [00:14<00:32, 8609.07it/s] 30%|███       | 121330/400000 [00:14<00:31, 8752.67it/s] 31%|███       | 122220/400000 [00:14<00:31, 8790.55it/s] 31%|███       | 123101/400000 [00:14<00:31, 8690.87it/s] 31%|███       | 124011/400000 [00:14<00:31, 8807.24it/s] 31%|███       | 124893/400000 [00:14<00:31, 8807.52it/s] 31%|███▏      | 125803/400000 [00:14<00:30, 8892.59it/s] 32%|███▏      | 126702/400000 [00:14<00:30, 8920.28it/s] 32%|███▏      | 127595/400000 [00:14<00:30, 8910.44it/s] 32%|███▏      | 128501/400000 [00:14<00:30, 8954.64it/s] 32%|███▏      | 129397/400000 [00:15<00:30, 8907.93it/s] 33%|███▎      | 130298/400000 [00:15<00:30, 8937.50it/s] 33%|███▎      | 131192/400000 [00:15<00:30, 8923.76it/s] 33%|███▎      | 132096/400000 [00:15<00:29, 8958.25it/s] 33%|███▎      | 133003/400000 [00:15<00:29, 8988.72it/s] 33%|███▎      | 133902/400000 [00:15<00:29, 8925.28it/s] 34%|███▎      | 134795/400000 [00:15<00:30, 8716.94it/s] 34%|███▍      | 135686/400000 [00:15<00:30, 8772.74it/s] 34%|███▍      | 136587/400000 [00:15<00:29, 8840.77it/s] 34%|███▍      | 137472/400000 [00:15<00:31, 8452.77it/s] 35%|███▍      | 138322/400000 [00:16<00:31, 8398.41it/s] 35%|███▍      | 139165/400000 [00:16<00:31, 8340.57it/s] 35%|███▌      | 140025/400000 [00:16<00:30, 8416.46it/s] 35%|███▌      | 140918/400000 [00:16<00:30, 8563.56it/s] 35%|███▌      | 141786/400000 [00:16<00:30, 8597.28it/s] 36%|███▌      | 142668/400000 [00:16<00:29, 8660.25it/s] 36%|███▌      | 143553/400000 [00:16<00:29, 8715.96it/s] 36%|███▌      | 144436/400000 [00:16<00:29, 8749.37it/s] 36%|███▋      | 145313/400000 [00:16<00:29, 8753.69it/s] 37%|███▋      | 146189/400000 [00:16<00:29, 8624.43it/s] 37%|███▋      | 147053/400000 [00:17<00:29, 8563.89it/s] 37%|███▋      | 147910/400000 [00:17<00:30, 8203.22it/s] 37%|███▋      | 148785/400000 [00:17<00:30, 8358.19it/s] 37%|███▋      | 149699/400000 [00:17<00:29, 8578.00it/s] 38%|███▊      | 150576/400000 [00:17<00:28, 8633.95it/s] 38%|███▊      | 151469/400000 [00:17<00:28, 8718.85it/s] 38%|███▊      | 152348/400000 [00:17<00:28, 8736.10it/s] 38%|███▊      | 153244/400000 [00:17<00:28, 8801.49it/s] 39%|███▊      | 154138/400000 [00:17<00:27, 8841.49it/s] 39%|███▉      | 155023/400000 [00:17<00:27, 8812.90it/s] 39%|███▉      | 155905/400000 [00:18<00:28, 8596.74it/s] 39%|███▉      | 156785/400000 [00:18<00:28, 8514.52it/s] 39%|███▉      | 157638/400000 [00:18<00:28, 8362.00it/s] 40%|███▉      | 158518/400000 [00:18<00:28, 8487.54it/s] 40%|███▉      | 159395/400000 [00:18<00:28, 8569.21it/s] 40%|████      | 160289/400000 [00:18<00:27, 8677.15it/s] 40%|████      | 161203/400000 [00:18<00:27, 8809.33it/s] 41%|████      | 162086/400000 [00:18<00:26, 8815.39it/s] 41%|████      | 162977/400000 [00:18<00:26, 8841.00it/s] 41%|████      | 163862/400000 [00:18<00:26, 8769.76it/s] 41%|████      | 164761/400000 [00:19<00:26, 8832.16it/s] 41%|████▏     | 165672/400000 [00:19<00:26, 8911.86it/s] 42%|████▏     | 166564/400000 [00:19<00:26, 8882.73it/s] 42%|████▏     | 167453/400000 [00:19<00:26, 8813.88it/s] 42%|████▏     | 168335/400000 [00:19<00:26, 8809.80it/s] 42%|████▏     | 169217/400000 [00:19<00:26, 8651.85it/s] 43%|████▎     | 170084/400000 [00:19<00:26, 8609.77it/s] 43%|████▎     | 170946/400000 [00:19<00:26, 8484.53it/s] 43%|████▎     | 171796/400000 [00:19<00:27, 8401.15it/s] 43%|████▎     | 172648/400000 [00:20<00:26, 8433.62it/s] 43%|████▎     | 173522/400000 [00:20<00:26, 8520.42it/s] 44%|████▎     | 174417/400000 [00:20<00:26, 8643.40it/s] 44%|████▍     | 175286/400000 [00:20<00:25, 8654.94it/s] 44%|████▍     | 176153/400000 [00:20<00:26, 8346.45it/s] 44%|████▍     | 177047/400000 [00:20<00:26, 8513.78it/s] 44%|████▍     | 177923/400000 [00:20<00:25, 8584.96it/s] 45%|████▍     | 178808/400000 [00:20<00:25, 8660.22it/s] 45%|████▍     | 179676/400000 [00:20<00:25, 8656.76it/s] 45%|████▌     | 180543/400000 [00:20<00:26, 8390.27it/s] 45%|████▌     | 181406/400000 [00:21<00:25, 8460.40it/s] 46%|████▌     | 182307/400000 [00:21<00:25, 8616.91it/s] 46%|████▌     | 183202/400000 [00:21<00:24, 8712.68it/s] 46%|████▌     | 184075/400000 [00:21<00:24, 8656.40it/s] 46%|████▌     | 184942/400000 [00:21<00:24, 8606.92it/s] 46%|████▋     | 185804/400000 [00:21<00:24, 8593.00it/s] 47%|████▋     | 186669/400000 [00:21<00:24, 8609.79it/s] 47%|████▋     | 187554/400000 [00:21<00:24, 8680.07it/s] 47%|████▋     | 188445/400000 [00:21<00:24, 8745.69it/s] 47%|████▋     | 189322/400000 [00:21<00:24, 8752.91it/s] 48%|████▊     | 190221/400000 [00:22<00:23, 8820.65it/s] 48%|████▊     | 191116/400000 [00:22<00:23, 8856.53it/s] 48%|████▊     | 192018/400000 [00:22<00:23, 8903.05it/s] 48%|████▊     | 192909/400000 [00:22<00:23, 8883.50it/s] 48%|████▊     | 193798/400000 [00:22<00:24, 8583.07it/s] 49%|████▊     | 194673/400000 [00:22<00:23, 8630.59it/s] 49%|████▉     | 195563/400000 [00:22<00:23, 8706.94it/s] 49%|████▉     | 196467/400000 [00:22<00:23, 8801.64it/s] 49%|████▉     | 197349/400000 [00:22<00:23, 8796.83it/s] 50%|████▉     | 198230/400000 [00:22<00:23, 8704.79it/s] 50%|████▉     | 199102/400000 [00:23<00:23, 8510.28it/s] 50%|████▉     | 199996/400000 [00:23<00:23, 8634.53it/s] 50%|█████     | 200862/400000 [00:23<00:23, 8640.31it/s] 50%|█████     | 201745/400000 [00:23<00:22, 8694.42it/s] 51%|█████     | 202631/400000 [00:23<00:22, 8743.41it/s] 51%|█████     | 203506/400000 [00:23<00:22, 8676.36it/s] 51%|█████     | 204375/400000 [00:23<00:22, 8564.10it/s] 51%|█████▏    | 205271/400000 [00:23<00:22, 8677.01it/s] 52%|█████▏    | 206140/400000 [00:23<00:22, 8615.95it/s] 52%|█████▏    | 207021/400000 [00:23<00:22, 8672.11it/s] 52%|█████▏    | 207889/400000 [00:24<00:22, 8460.81it/s] 52%|█████▏    | 208783/400000 [00:24<00:22, 8597.41it/s] 52%|█████▏    | 209675/400000 [00:24<00:21, 8691.25it/s] 53%|█████▎    | 210574/400000 [00:24<00:21, 8776.23it/s] 53%|█████▎    | 211469/400000 [00:24<00:21, 8827.48it/s] 53%|█████▎    | 212353/400000 [00:24<00:21, 8717.28it/s] 53%|█████▎    | 213226/400000 [00:24<00:21, 8544.62it/s] 54%|█████▎    | 214082/400000 [00:24<00:21, 8537.29it/s] 54%|█████▎    | 214937/400000 [00:24<00:21, 8495.08it/s] 54%|█████▍    | 215837/400000 [00:25<00:21, 8640.00it/s] 54%|█████▍    | 216731/400000 [00:25<00:21, 8725.75it/s] 54%|█████▍    | 217635/400000 [00:25<00:20, 8815.67it/s] 55%|█████▍    | 218537/400000 [00:25<00:20, 8873.64it/s] 55%|█████▍    | 219426/400000 [00:25<00:20, 8814.31it/s] 55%|█████▌    | 220331/400000 [00:25<00:20, 8883.10it/s] 55%|█████▌    | 221220/400000 [00:25<00:20, 8755.13it/s] 56%|█████▌    | 222097/400000 [00:25<00:20, 8558.55it/s] 56%|█████▌    | 222981/400000 [00:25<00:20, 8639.42it/s] 56%|█████▌    | 223847/400000 [00:25<00:20, 8635.10it/s] 56%|█████▌    | 224712/400000 [00:26<00:20, 8509.28it/s] 56%|█████▋    | 225607/400000 [00:26<00:20, 8634.34it/s] 57%|█████▋    | 226472/400000 [00:26<00:20, 8304.34it/s] 57%|█████▋    | 227306/400000 [00:26<00:21, 8215.81it/s] 57%|█████▋    | 228167/400000 [00:26<00:20, 8329.31it/s] 57%|█████▋    | 229003/400000 [00:26<00:20, 8297.00it/s] 57%|█████▋    | 229880/400000 [00:26<00:20, 8431.58it/s] 58%|█████▊    | 230725/400000 [00:26<00:20, 8230.94it/s] 58%|█████▊    | 231611/400000 [00:26<00:20, 8408.75it/s] 58%|█████▊    | 232455/400000 [00:26<00:19, 8395.69it/s] 58%|█████▊    | 233328/400000 [00:27<00:19, 8490.90it/s] 59%|█████▊    | 234186/400000 [00:27<00:19, 8514.99it/s] 59%|█████▉    | 235076/400000 [00:27<00:19, 8625.46it/s] 59%|█████▉    | 235940/400000 [00:27<00:19, 8627.64it/s] 59%|█████▉    | 236804/400000 [00:27<00:18, 8589.91it/s] 59%|█████▉    | 237683/400000 [00:27<00:18, 8647.21it/s] 60%|█████▉    | 238569/400000 [00:27<00:18, 8709.35it/s] 60%|█████▉    | 239461/400000 [00:27<00:18, 8770.20it/s] 60%|██████    | 240339/400000 [00:27<00:18, 8679.22it/s] 60%|██████    | 241228/400000 [00:27<00:18, 8739.81it/s] 61%|██████    | 242105/400000 [00:28<00:18, 8747.89it/s] 61%|██████    | 242981/400000 [00:28<00:17, 8746.24it/s] 61%|██████    | 243865/400000 [00:28<00:17, 8773.16it/s] 61%|██████    | 244743/400000 [00:28<00:17, 8633.30it/s] 61%|██████▏   | 245607/400000 [00:28<00:17, 8608.21it/s] 62%|██████▏   | 246475/400000 [00:28<00:17, 8628.26it/s] 62%|██████▏   | 247367/400000 [00:28<00:17, 8713.26it/s] 62%|██████▏   | 248259/400000 [00:28<00:17, 8774.01it/s] 62%|██████▏   | 249137/400000 [00:28<00:17, 8718.02it/s] 63%|██████▎   | 250030/400000 [00:28<00:17, 8780.10it/s] 63%|██████▎   | 250909/400000 [00:29<00:17, 8672.02it/s] 63%|██████▎   | 251813/400000 [00:29<00:16, 8776.54it/s] 63%|██████▎   | 252692/400000 [00:29<00:16, 8763.11it/s] 63%|██████▎   | 253569/400000 [00:29<00:16, 8709.98it/s] 64%|██████▎   | 254452/400000 [00:29<00:16, 8744.74it/s] 64%|██████▍   | 255333/400000 [00:29<00:16, 8763.86it/s] 64%|██████▍   | 256210/400000 [00:29<00:16, 8653.75it/s] 64%|██████▍   | 257099/400000 [00:29<00:16, 8722.68it/s] 64%|██████▍   | 257972/400000 [00:29<00:16, 8668.92it/s] 65%|██████▍   | 258861/400000 [00:29<00:16, 8733.13it/s] 65%|██████▍   | 259741/400000 [00:30<00:16, 8751.50it/s] 65%|██████▌   | 260628/400000 [00:30<00:15, 8784.94it/s] 65%|██████▌   | 261520/400000 [00:30<00:15, 8823.91it/s] 66%|██████▌   | 262403/400000 [00:30<00:15, 8810.21it/s] 66%|██████▌   | 263289/400000 [00:30<00:15, 8822.13it/s] 66%|██████▌   | 264172/400000 [00:30<00:15, 8787.67it/s] 66%|██████▋   | 265054/400000 [00:30<00:15, 8796.10it/s] 66%|██████▋   | 265934/400000 [00:30<00:15, 8669.48it/s] 67%|██████▋   | 266802/400000 [00:30<00:15, 8582.22it/s] 67%|██████▋   | 267689/400000 [00:30<00:15, 8666.37it/s] 67%|██████▋   | 268568/400000 [00:31<00:15, 8702.17it/s] 67%|██████▋   | 269471/400000 [00:31<00:14, 8796.89it/s] 68%|██████▊   | 270352/400000 [00:31<00:15, 8358.33it/s] 68%|██████▊   | 271241/400000 [00:31<00:15, 8510.62it/s] 68%|██████▊   | 272135/400000 [00:31<00:14, 8632.39it/s] 68%|██████▊   | 273002/400000 [00:31<00:14, 8601.04it/s] 68%|██████▊   | 273897/400000 [00:31<00:14, 8701.19it/s] 69%|██████▊   | 274770/400000 [00:31<00:14, 8658.67it/s] 69%|██████▉   | 275638/400000 [00:31<00:14, 8615.55it/s] 69%|██████▉   | 276515/400000 [00:32<00:14, 8660.14it/s] 69%|██████▉   | 277382/400000 [00:32<00:14, 8337.17it/s] 70%|██████▉   | 278219/400000 [00:32<00:14, 8302.61it/s] 70%|██████▉   | 279052/400000 [00:32<00:15, 7981.62it/s] 70%|██████▉   | 279913/400000 [00:32<00:14, 8158.66it/s] 70%|███████   | 280796/400000 [00:32<00:14, 8348.56it/s] 70%|███████   | 281642/400000 [00:32<00:14, 8379.47it/s] 71%|███████   | 282526/400000 [00:32<00:13, 8511.72it/s] 71%|███████   | 283396/400000 [00:32<00:13, 8567.14it/s] 71%|███████   | 284291/400000 [00:32<00:13, 8676.18it/s] 71%|███████▏  | 285161/400000 [00:33<00:13, 8525.41it/s] 72%|███████▏  | 286033/400000 [00:33<00:13, 8580.50it/s] 72%|███████▏  | 286893/400000 [00:33<00:14, 8012.50it/s] 72%|███████▏  | 287703/400000 [00:33<00:14, 7648.67it/s] 72%|███████▏  | 288506/400000 [00:33<00:14, 7757.09it/s] 72%|███████▏  | 289310/400000 [00:33<00:14, 7837.91it/s] 73%|███████▎  | 290176/400000 [00:33<00:13, 8067.32it/s] 73%|███████▎  | 291025/400000 [00:33<00:13, 8188.20it/s] 73%|███████▎  | 291892/400000 [00:33<00:12, 8324.57it/s] 73%|███████▎  | 292728/400000 [00:34<00:13, 8130.66it/s] 73%|███████▎  | 293607/400000 [00:34<00:12, 8317.59it/s] 74%|███████▎  | 294480/400000 [00:34<00:12, 8435.04it/s] 74%|███████▍  | 295337/400000 [00:34<00:12, 8474.04it/s] 74%|███████▍  | 296187/400000 [00:34<00:12, 8465.78it/s] 74%|███████▍  | 297072/400000 [00:34<00:12, 8576.09it/s] 74%|███████▍  | 297939/400000 [00:34<00:11, 8603.91it/s] 75%|███████▍  | 298801/400000 [00:34<00:11, 8471.20it/s] 75%|███████▍  | 299673/400000 [00:34<00:11, 8544.12it/s] 75%|███████▌  | 300529/400000 [00:34<00:11, 8333.62it/s] 75%|███████▌  | 301365/400000 [00:35<00:12, 8120.46it/s] 76%|███████▌  | 302180/400000 [00:35<00:12, 8061.81it/s] 76%|███████▌  | 303028/400000 [00:35<00:11, 8179.87it/s] 76%|███████▌  | 303893/400000 [00:35<00:11, 8315.19it/s] 76%|███████▌  | 304737/400000 [00:35<00:11, 8351.29it/s] 76%|███████▋  | 305616/400000 [00:35<00:11, 8476.73it/s] 77%|███████▋  | 306466/400000 [00:35<00:11, 8347.47it/s] 77%|███████▋  | 307303/400000 [00:35<00:11, 8318.37it/s] 77%|███████▋  | 308198/400000 [00:35<00:10, 8497.44it/s] 77%|███████▋  | 309070/400000 [00:35<00:10, 8560.46it/s] 77%|███████▋  | 309985/400000 [00:36<00:10, 8728.66it/s] 78%|███████▊  | 310893/400000 [00:36<00:10, 8830.27it/s] 78%|███████▊  | 311802/400000 [00:36<00:09, 8904.21it/s] 78%|███████▊  | 312721/400000 [00:36<00:09, 8987.00it/s] 78%|███████▊  | 313629/400000 [00:36<00:09, 9012.61it/s] 79%|███████▊  | 314554/400000 [00:36<00:09, 9080.96it/s] 79%|███████▉  | 315463/400000 [00:36<00:09, 8994.38it/s] 79%|███████▉  | 316364/400000 [00:36<00:09, 8944.41it/s] 79%|███████▉  | 317259/400000 [00:36<00:09, 8841.94it/s] 80%|███████▉  | 318158/400000 [00:36<00:09, 8883.79it/s] 80%|███████▉  | 319055/400000 [00:37<00:09, 8907.60it/s] 80%|███████▉  | 319947/400000 [00:37<00:09, 8849.93it/s] 80%|████████  | 320833/400000 [00:37<00:08, 8824.77it/s] 80%|████████  | 321727/400000 [00:37<00:08, 8856.49it/s] 81%|████████  | 322646/400000 [00:37<00:08, 8952.06it/s] 81%|████████  | 323572/400000 [00:37<00:08, 9040.62it/s] 81%|████████  | 324477/400000 [00:37<00:08, 9005.11it/s] 81%|████████▏ | 325378/400000 [00:37<00:08, 8981.62it/s] 82%|████████▏ | 326277/400000 [00:37<00:08, 8931.20it/s] 82%|████████▏ | 327211/400000 [00:37<00:08, 9047.81it/s] 82%|████████▏ | 328128/400000 [00:38<00:07, 9081.74it/s] 82%|████████▏ | 329037/400000 [00:38<00:07, 8925.42it/s] 82%|████████▏ | 329931/400000 [00:38<00:07, 8834.42it/s] 83%|████████▎ | 330832/400000 [00:38<00:07, 8883.46it/s] 83%|████████▎ | 331721/400000 [00:38<00:07, 8729.31it/s] 83%|████████▎ | 332595/400000 [00:38<00:07, 8632.43it/s] 83%|████████▎ | 333473/400000 [00:38<00:07, 8675.05it/s] 84%|████████▎ | 334347/400000 [00:38<00:07, 8691.74it/s] 84%|████████▍ | 335221/400000 [00:38<00:07, 8703.54it/s] 84%|████████▍ | 336121/400000 [00:38<00:07, 8790.17it/s] 84%|████████▍ | 337001/400000 [00:39<00:07, 8581.63it/s] 84%|████████▍ | 337873/400000 [00:39<00:07, 8621.65it/s] 85%|████████▍ | 338741/400000 [00:39<00:07, 8636.96it/s] 85%|████████▍ | 339629/400000 [00:39<00:06, 8706.26it/s] 85%|████████▌ | 340529/400000 [00:39<00:06, 8792.17it/s] 85%|████████▌ | 341425/400000 [00:39<00:06, 8841.32it/s] 86%|████████▌ | 342310/400000 [00:39<00:06, 8788.57it/s] 86%|████████▌ | 343190/400000 [00:39<00:06, 8722.84it/s] 86%|████████▌ | 344069/400000 [00:39<00:06, 8741.62it/s] 86%|████████▌ | 344952/400000 [00:39<00:06, 8767.64it/s] 86%|████████▋ | 345831/400000 [00:40<00:06, 8773.10it/s] 87%|████████▋ | 346709/400000 [00:40<00:06, 8342.50it/s] 87%|████████▋ | 347548/400000 [00:40<00:06, 8196.74it/s] 87%|████████▋ | 348450/400000 [00:40<00:06, 8425.41it/s] 87%|████████▋ | 349316/400000 [00:40<00:05, 8492.31it/s] 88%|████████▊ | 350209/400000 [00:40<00:05, 8618.62it/s] 88%|████████▊ | 351074/400000 [00:40<00:05, 8539.98it/s] 88%|████████▊ | 351931/400000 [00:40<00:05, 8546.27it/s] 88%|████████▊ | 352825/400000 [00:40<00:05, 8659.93it/s] 88%|████████▊ | 353717/400000 [00:41<00:05, 8734.65it/s] 89%|████████▊ | 354592/400000 [00:41<00:05, 8576.61it/s] 89%|████████▉ | 355452/400000 [00:41<00:05, 8516.55it/s] 89%|████████▉ | 356313/400000 [00:41<00:05, 8544.14it/s] 89%|████████▉ | 357204/400000 [00:41<00:04, 8648.61it/s] 90%|████████▉ | 358082/400000 [00:41<00:04, 8686.40it/s] 90%|████████▉ | 358979/400000 [00:41<00:04, 8769.01it/s] 90%|████████▉ | 359857/400000 [00:41<00:04, 8572.14it/s] 90%|█████████ | 360716/400000 [00:41<00:04, 8411.49it/s] 90%|█████████ | 361590/400000 [00:41<00:04, 8506.61it/s] 91%|█████████ | 362447/400000 [00:42<00:04, 8524.50it/s] 91%|█████████ | 363312/400000 [00:42<00:04, 8561.62it/s] 91%|█████████ | 364169/400000 [00:42<00:04, 8473.22it/s] 91%|█████████▏| 365051/400000 [00:42<00:04, 8573.32it/s] 91%|█████████▏| 365945/400000 [00:42<00:03, 8677.45it/s] 92%|█████████▏| 366835/400000 [00:42<00:03, 8740.20it/s] 92%|█████████▏| 367728/400000 [00:42<00:03, 8793.53it/s] 92%|█████████▏| 368608/400000 [00:42<00:03, 8751.03it/s] 92%|█████████▏| 369499/400000 [00:42<00:03, 8797.67it/s] 93%|█████████▎| 370380/400000 [00:42<00:03, 8691.50it/s] 93%|█████████▎| 371270/400000 [00:43<00:03, 8750.46it/s] 93%|█████████▎| 372169/400000 [00:43<00:03, 8818.79it/s] 93%|█████████▎| 373052/400000 [00:43<00:03, 8773.11it/s] 93%|█████████▎| 373955/400000 [00:43<00:02, 8845.78it/s] 94%|█████████▎| 374860/400000 [00:43<00:02, 8904.73it/s] 94%|█████████▍| 375759/400000 [00:43<00:02, 8929.11it/s] 94%|█████████▍| 376653/400000 [00:43<00:02, 8766.61it/s] 94%|█████████▍| 377531/400000 [00:43<00:02, 8649.28it/s] 95%|█████████▍| 378419/400000 [00:43<00:02, 8716.34it/s] 95%|█████████▍| 379301/400000 [00:43<00:02, 8744.41it/s] 95%|█████████▌| 380203/400000 [00:44<00:02, 8824.01it/s] 95%|█████████▌| 381086/400000 [00:44<00:02, 8504.11it/s] 95%|█████████▌| 381956/400000 [00:44<00:02, 8559.50it/s] 96%|█████████▌| 382855/400000 [00:44<00:01, 8683.48it/s] 96%|█████████▌| 383726/400000 [00:44<00:01, 8647.34it/s] 96%|█████████▌| 384604/400000 [00:44<00:01, 8684.60it/s] 96%|█████████▋| 385491/400000 [00:44<00:01, 8737.67it/s] 97%|█████████▋| 386366/400000 [00:44<00:01, 8507.98it/s] 97%|█████████▋| 387254/400000 [00:44<00:01, 8614.02it/s] 97%|█████████▋| 388135/400000 [00:44<00:01, 8670.09it/s] 97%|█████████▋| 389029/400000 [00:45<00:01, 8747.25it/s] 97%|█████████▋| 389905/400000 [00:45<00:01, 8745.38it/s] 98%|█████████▊| 390781/400000 [00:45<00:01, 8585.60it/s] 98%|█████████▊| 391671/400000 [00:45<00:00, 8677.18it/s] 98%|█████████▊| 392566/400000 [00:45<00:00, 8756.75it/s] 98%|█████████▊| 393465/400000 [00:45<00:00, 8825.43it/s] 99%|█████████▊| 394349/400000 [00:45<00:00, 8805.28it/s] 99%|█████████▉| 395231/400000 [00:45<00:00, 8744.60it/s] 99%|█████████▉| 396106/400000 [00:45<00:00, 8722.97it/s] 99%|█████████▉| 396979/400000 [00:45<00:00, 8721.71it/s] 99%|█████████▉| 397852/400000 [00:46<00:00, 8670.64it/s]100%|█████████▉| 398738/400000 [00:46<00:00, 8724.35it/s]100%|█████████▉| 399611/400000 [00:46<00:00, 8612.77it/s]100%|█████████▉| 399999/400000 [00:46<00:00, 8629.72it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7fb2d58cd128>, <torchtext.data.dataset.TabularDataset object at 0x7fb2d58cd278>, <torchtext.vocab.Vocab object at 0x7fb2d58cd198>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 30.52 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 45.20 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 24.50 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 24.50 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.78 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.78 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  7.69 file/s]2020-06-15 00:18:06.759591: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-06-15 00:18:06.763992: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-06-15 00:18:06.764294: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x555d87926b60 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-15 00:18:06.764425: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 37%|███▋      | 3686400/9912422 [00:00<00:00, 36513378.58it/s]9920512it [00:00, 35284686.66it/s]                             
0it [00:00, ?it/s]32768it [00:00, 414131.17it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 162450.86it/s]1654784it [00:00, 11544670.00it/s]                         
0it [00:00, ?it/s]8192it [00:00, 193526.93it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
