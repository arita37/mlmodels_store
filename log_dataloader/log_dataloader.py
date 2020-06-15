
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f851cb7e620> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f851cb7e620> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f85881450d0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f85881450d0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f85a1e71ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f85a1e71ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f85359be950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f85359be950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f85359be950> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 21%|██        | 2080768/9912422 [00:00<00:00, 19453995.28it/s]9920512it [00:00, 32886265.31it/s]                             
0it [00:00, ?it/s]32768it [00:00, 703465.93it/s]
0it [00:00, ?it/s]  6%|▋         | 106496/1648877 [00:00<00:01, 1011429.50it/s]1654784it [00:00, 12402198.50it/s]                           
0it [00:00, ?it/s]8192it [00:00, 247487.92it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f851ca55b38>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f851ca4bdd8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f85359be598> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f85359be598> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f85359be598> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:43,  1.56 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:43,  1.56 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:42,  1.56 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:41,  1.56 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:41,  1.56 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:40,  1.56 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:40,  1.56 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:39,  1.56 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<01:38,  1.56 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   6%|▌         | 9/162 [00:00<01:09,  2.21 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<01:09,  2.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<01:08,  2.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<01:08,  2.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<01:07,  2.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<01:07,  2.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<01:06,  2.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<01:06,  2.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<01:06,  2.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<01:05,  2.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<01:05,  2.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<01:04,  2.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  12%|█▏        | 20/162 [00:00<00:45,  3.13 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:45,  3.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:45,  3.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:44,  3.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:44,  3.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:44,  3.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:43,  3.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:43,  3.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:43,  3.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:42,  3.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:42,  3.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:42,  3.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  19%|█▉        | 31/162 [00:00<00:29,  4.41 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:29,  4.41 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:29,  4.41 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:29,  4.41 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:29,  4.41 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:28,  4.41 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:28,  4.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:28,  4.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:28,  4.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:27,  4.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:27,  4.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:27,  4.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  26%|██▌       | 42/162 [00:01<00:19,  6.19 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:19,  6.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:19,  6.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:19,  6.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:18,  6.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:18,  6.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:18,  6.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:18,  6.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:18,  6.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:18,  6.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:17,  6.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:17,  6.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  33%|███▎      | 53/162 [00:01<00:12,  8.63 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:12,  8.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:12,  8.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:12,  8.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:12,  8.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:12,  8.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:12,  8.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:11,  8.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:11,  8.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:11,  8.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:11,  8.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:11,  8.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  40%|███▉      | 64/162 [00:01<00:08, 11.90 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:08, 11.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:08, 11.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:08, 11.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:07, 11.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:07, 11.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:07, 11.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:07, 11.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:07, 11.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:07, 11.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:07, 11.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:07, 11.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  46%|████▋     | 75/162 [00:01<00:05, 16.20 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:05, 16.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:05, 16.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:05, 16.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:05, 16.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:05, 16.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:05, 16.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:05, 16.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:04, 16.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:04, 16.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:04, 16.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:04, 16.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:03, 21.72 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:03, 21.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:03, 21.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:03, 21.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:03, 21.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:03, 21.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:03, 21.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:03, 21.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:03, 21.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:03, 21.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:03, 21.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:03, 21.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:02, 28.45 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:02, 28.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:02, 28.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:02, 28.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:02, 28.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:02, 28.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:02, 28.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:02, 28.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:02, 28.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:02, 28.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 28.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 35.76 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 35.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 35.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 35.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:01, 35.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:01, 35.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:01, 35.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:01, 35.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:01, 35.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:01, 35.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:01, 35.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:01, 43.72 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:01, 43.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:01, 43.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 43.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 43.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 43.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 43.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 43.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 43.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 43.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 43.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 51.83 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 51.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 51.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 51.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:01<00:00, 51.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 51.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:01<00:00, 51.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:01<00:00, 51.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:01<00:00, 51.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 51.83 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 51.83 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 59.75 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 59.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 59.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 59.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 59.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 59.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 59.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 59.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 59.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 59.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 59.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 66.72 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 66.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 66.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 66.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 66.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 66.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 66.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 66.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 66.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 66.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 66.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 73.22 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 73.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 73.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 73.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 73.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 73.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 73.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.29s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.29s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 73.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.29s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 73.22 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.33s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.29s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 73.22 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.33s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.33s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 37.42 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.33s/ url]
0 examples [00:00, ? examples/s]2020-06-15 06:09:11.691570: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-06-15 06:09:11.708230: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-06-15 06:09:11.708411: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x557c3a5575b0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-15 06:09:11.708428: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  9.60 examples/s]111 examples [00:00, 13.66 examples/s]225 examples [00:00, 19.42 examples/s]337 examples [00:00, 27.53 examples/s]450 examples [00:00, 38.93 examples/s]549 examples [00:00, 54.68 examples/s]662 examples [00:00, 76.52 examples/s]775 examples [00:00, 106.21 examples/s]886 examples [00:00, 145.75 examples/s]1001 examples [00:01, 197.45 examples/s]1115 examples [00:01, 262.57 examples/s]1228 examples [00:01, 341.07 examples/s]1339 examples [00:01, 425.78 examples/s]1448 examples [00:01, 520.53 examples/s]1561 examples [00:01, 620.31 examples/s]1673 examples [00:01, 715.87 examples/s]1784 examples [00:01, 801.11 examples/s]1895 examples [00:01, 870.95 examples/s]2009 examples [00:01, 936.46 examples/s]2123 examples [00:02, 987.09 examples/s]2236 examples [00:02, 1025.01 examples/s]2348 examples [00:02, 1051.76 examples/s]2461 examples [00:02, 1071.94 examples/s]2574 examples [00:02, 1082.60 examples/s]2686 examples [00:02, 1083.68 examples/s]2800 examples [00:02, 1098.50 examples/s]2913 examples [00:02, 1106.72 examples/s]3025 examples [00:02, 1089.51 examples/s]3135 examples [00:02, 1089.03 examples/s]3247 examples [00:03, 1094.93 examples/s]3357 examples [00:03, 1092.50 examples/s]3470 examples [00:03, 1101.80 examples/s]3581 examples [00:03, 1099.98 examples/s]3692 examples [00:03, 1094.58 examples/s]3802 examples [00:03, 1091.17 examples/s]3913 examples [00:03, 1095.68 examples/s]4026 examples [00:03, 1104.89 examples/s]4137 examples [00:03, 1102.44 examples/s]4251 examples [00:03, 1113.06 examples/s]4363 examples [00:04, 1094.04 examples/s]4473 examples [00:04, 1093.95 examples/s]4583 examples [00:04, 1095.28 examples/s]4693 examples [00:04, 1068.09 examples/s]4801 examples [00:04, 1069.49 examples/s]4916 examples [00:04, 1090.72 examples/s]5026 examples [00:04, 1085.45 examples/s]5140 examples [00:04, 1099.91 examples/s]5251 examples [00:04, 1097.82 examples/s]5365 examples [00:04, 1109.93 examples/s]5477 examples [00:05, 1110.39 examples/s]5591 examples [00:05, 1116.57 examples/s]5703 examples [00:05, 1110.22 examples/s]5816 examples [00:05, 1115.95 examples/s]5928 examples [00:05, 1112.15 examples/s]6041 examples [00:05, 1116.96 examples/s]6153 examples [00:05, 1117.62 examples/s]6267 examples [00:05, 1121.70 examples/s]6383 examples [00:05, 1130.60 examples/s]6497 examples [00:05, 1094.06 examples/s]6607 examples [00:06, 1094.59 examples/s]6717 examples [00:06, 1089.09 examples/s]6828 examples [00:06, 1092.57 examples/s]6938 examples [00:06, 1084.89 examples/s]7049 examples [00:06, 1090.78 examples/s]7162 examples [00:06, 1099.84 examples/s]7273 examples [00:06, 1087.29 examples/s]7382 examples [00:06, 1085.74 examples/s]7496 examples [00:06, 1099.07 examples/s]7606 examples [00:07, 1094.75 examples/s]7716 examples [00:07, 1091.58 examples/s]7826 examples [00:07, 1087.64 examples/s]7935 examples [00:07, 1078.32 examples/s]8043 examples [00:07, 1060.96 examples/s]8150 examples [00:07, 1034.42 examples/s]8260 examples [00:07, 1053.11 examples/s]8370 examples [00:07, 1066.11 examples/s]8483 examples [00:07, 1082.75 examples/s]8595 examples [00:07, 1093.21 examples/s]8707 examples [00:08, 1098.68 examples/s]8818 examples [00:08, 1100.93 examples/s]8929 examples [00:08, 1088.85 examples/s]9040 examples [00:08, 1092.45 examples/s]9153 examples [00:08, 1101.17 examples/s]9264 examples [00:08, 1097.10 examples/s]9378 examples [00:08, 1107.31 examples/s]9491 examples [00:08, 1111.27 examples/s]9604 examples [00:08, 1115.45 examples/s]9717 examples [00:08, 1117.04 examples/s]9829 examples [00:09, 1098.90 examples/s]9942 examples [00:09, 1107.37 examples/s]10053 examples [00:09, 1008.42 examples/s]10161 examples [00:09, 1027.91 examples/s]10270 examples [00:09, 1045.43 examples/s]10381 examples [00:09, 1061.96 examples/s]10493 examples [00:09, 1078.60 examples/s]10602 examples [00:09, 1076.27 examples/s]10711 examples [00:09, 1075.88 examples/s]10819 examples [00:09, 1076.48 examples/s]10934 examples [00:10, 1094.96 examples/s]11049 examples [00:10, 1109.06 examples/s]11161 examples [00:10, 1102.71 examples/s]11272 examples [00:10, 1062.79 examples/s]11379 examples [00:10, 1010.19 examples/s]11484 examples [00:10, 1021.45 examples/s]11595 examples [00:10, 1046.31 examples/s]11705 examples [00:10, 1061.29 examples/s]11816 examples [00:10, 1074.42 examples/s]11929 examples [00:11, 1089.64 examples/s]12040 examples [00:11, 1094.67 examples/s]12154 examples [00:11, 1105.92 examples/s]12265 examples [00:11, 1098.32 examples/s]12375 examples [00:11, 1055.00 examples/s]12486 examples [00:11, 1069.00 examples/s]12596 examples [00:11, 1075.64 examples/s]12711 examples [00:11, 1094.19 examples/s]12821 examples [00:11, 1089.62 examples/s]12934 examples [00:11, 1099.15 examples/s]13045 examples [00:12, 1101.33 examples/s]13156 examples [00:12, 1090.32 examples/s]13269 examples [00:12, 1101.07 examples/s]13380 examples [00:12, 1086.35 examples/s]13489 examples [00:12, 1076.13 examples/s]13602 examples [00:12, 1089.14 examples/s]13713 examples [00:12, 1094.47 examples/s]13824 examples [00:12, 1096.41 examples/s]13934 examples [00:12, 1074.04 examples/s]14046 examples [00:12, 1085.74 examples/s]14158 examples [00:13, 1094.54 examples/s]14269 examples [00:13, 1098.93 examples/s]14381 examples [00:13, 1102.55 examples/s]14492 examples [00:13, 1090.98 examples/s]14602 examples [00:13, 1067.34 examples/s]14709 examples [00:13, 1053.07 examples/s]14817 examples [00:13, 1060.56 examples/s]14931 examples [00:13, 1081.83 examples/s]15040 examples [00:13, 1062.22 examples/s]15150 examples [00:13, 1072.21 examples/s]15259 examples [00:14, 1076.82 examples/s]15372 examples [00:14, 1089.60 examples/s]15486 examples [00:14, 1103.18 examples/s]15597 examples [00:14, 1095.47 examples/s]15707 examples [00:14, 1079.61 examples/s]15819 examples [00:14, 1089.32 examples/s]15929 examples [00:14, 1091.99 examples/s]16044 examples [00:14, 1108.64 examples/s]16155 examples [00:14, 1082.20 examples/s]16267 examples [00:15, 1093.24 examples/s]16377 examples [00:15, 1080.29 examples/s]16488 examples [00:15, 1087.85 examples/s]16601 examples [00:15, 1099.29 examples/s]16712 examples [00:15, 1092.77 examples/s]16825 examples [00:15, 1100.97 examples/s]16936 examples [00:15, 1090.39 examples/s]17049 examples [00:15, 1101.86 examples/s]17163 examples [00:15, 1111.57 examples/s]17275 examples [00:15, 1102.19 examples/s]17388 examples [00:16, 1109.82 examples/s]17502 examples [00:16, 1116.66 examples/s]17614 examples [00:16, 1106.08 examples/s]17725 examples [00:16, 1097.59 examples/s]17835 examples [00:16, 1090.88 examples/s]17945 examples [00:16, 1092.29 examples/s]18055 examples [00:16, 1077.63 examples/s]18163 examples [00:16, 1061.26 examples/s]18271 examples [00:16, 1066.13 examples/s]18382 examples [00:16, 1076.44 examples/s]18493 examples [00:17, 1085.34 examples/s]18606 examples [00:17, 1097.37 examples/s]18716 examples [00:17, 1078.02 examples/s]18826 examples [00:17, 1083.99 examples/s]18935 examples [00:17, 1082.76 examples/s]19046 examples [00:17, 1089.32 examples/s]19161 examples [00:17, 1104.74 examples/s]19272 examples [00:17, 1091.09 examples/s]19382 examples [00:17, 1089.01 examples/s]19491 examples [00:17, 1081.74 examples/s]19600 examples [00:18, 1053.60 examples/s]19706 examples [00:18, 1054.95 examples/s]19816 examples [00:18, 1067.19 examples/s]19925 examples [00:18, 1073.83 examples/s]20033 examples [00:18, 1005.62 examples/s]20148 examples [00:18, 1043.22 examples/s]20260 examples [00:18, 1065.00 examples/s]20373 examples [00:18, 1083.70 examples/s]20483 examples [00:18, 1085.45 examples/s]20596 examples [00:19, 1097.77 examples/s]20707 examples [00:19, 1073.56 examples/s]20815 examples [00:19, 1074.43 examples/s]20923 examples [00:19, 1050.82 examples/s]21029 examples [00:19, 1031.53 examples/s]21133 examples [00:19, 1031.23 examples/s]21238 examples [00:19, 1034.40 examples/s]21342 examples [00:19, 1025.90 examples/s]21451 examples [00:19, 1042.92 examples/s]21556 examples [00:19, 1031.54 examples/s]21668 examples [00:20, 1053.66 examples/s]21779 examples [00:20, 1069.82 examples/s]21889 examples [00:20, 1077.67 examples/s]22000 examples [00:20, 1084.62 examples/s]22109 examples [00:20, 1067.44 examples/s]22219 examples [00:20, 1074.45 examples/s]22328 examples [00:20, 1076.87 examples/s]22437 examples [00:20, 1077.82 examples/s]22548 examples [00:20, 1085.03 examples/s]22657 examples [00:20, 1078.63 examples/s]22767 examples [00:21, 1083.65 examples/s]22876 examples [00:21, 1082.80 examples/s]22986 examples [00:21, 1086.24 examples/s]23097 examples [00:21, 1091.91 examples/s]23207 examples [00:21, 1071.80 examples/s]23319 examples [00:21, 1085.45 examples/s]23428 examples [00:21, 1081.66 examples/s]23537 examples [00:21, 1075.83 examples/s]23646 examples [00:21, 1078.84 examples/s]23754 examples [00:21, 1071.72 examples/s]23862 examples [00:22, 1045.09 examples/s]23969 examples [00:22, 1051.20 examples/s]24076 examples [00:22, 1056.72 examples/s]24182 examples [00:22, 1043.35 examples/s]24292 examples [00:22, 1058.02 examples/s]24400 examples [00:22, 1063.29 examples/s]24508 examples [00:22, 1067.71 examples/s]24619 examples [00:22, 1077.81 examples/s]24727 examples [00:22, 1068.47 examples/s]24834 examples [00:22, 1059.65 examples/s]24941 examples [00:23, 1054.15 examples/s]25047 examples [00:23, 1050.86 examples/s]25153 examples [00:23, 1032.72 examples/s]25261 examples [00:23, 1043.78 examples/s]25369 examples [00:23, 1052.13 examples/s]25480 examples [00:23, 1068.44 examples/s]25587 examples [00:23, 1060.13 examples/s]25696 examples [00:23, 1067.70 examples/s]25805 examples [00:23, 1073.08 examples/s]25913 examples [00:24, 1064.32 examples/s]26020 examples [00:24, 1063.56 examples/s]26127 examples [00:24, 1065.11 examples/s]26234 examples [00:24, 1054.59 examples/s]26340 examples [00:24, 1052.83 examples/s]26451 examples [00:24, 1069.21 examples/s]26560 examples [00:24, 1075.27 examples/s]26673 examples [00:24, 1089.31 examples/s]26783 examples [00:24, 1091.00 examples/s]26893 examples [00:24, 1092.84 examples/s]27003 examples [00:25, 1093.17 examples/s]27116 examples [00:25, 1103.07 examples/s]27227 examples [00:25, 1104.85 examples/s]27338 examples [00:25, 1105.28 examples/s]27449 examples [00:25, 1094.05 examples/s]27562 examples [00:25, 1102.41 examples/s]27673 examples [00:25, 1046.04 examples/s]27783 examples [00:25, 1059.94 examples/s]27894 examples [00:25, 1072.90 examples/s]28002 examples [00:25, 1068.32 examples/s]28112 examples [00:26, 1076.72 examples/s]28220 examples [00:26, 1059.48 examples/s]28327 examples [00:26, 1054.91 examples/s]28435 examples [00:26, 1060.29 examples/s]28542 examples [00:26, 1054.27 examples/s]28652 examples [00:26, 1067.14 examples/s]28761 examples [00:26, 1072.40 examples/s]28869 examples [00:26, 1068.63 examples/s]28976 examples [00:26, 1068.20 examples/s]29083 examples [00:26, 1053.77 examples/s]29193 examples [00:27, 1066.39 examples/s]29303 examples [00:27, 1074.13 examples/s]29411 examples [00:27, 1066.53 examples/s]29518 examples [00:27, 1065.47 examples/s]29625 examples [00:27, 1055.11 examples/s]29731 examples [00:27, 1054.85 examples/s]29839 examples [00:27, 1060.69 examples/s]29947 examples [00:27, 1066.32 examples/s]30054 examples [00:27, 1012.23 examples/s]30160 examples [00:27, 1025.28 examples/s]30269 examples [00:28, 1042.41 examples/s]30379 examples [00:28, 1058.25 examples/s]30491 examples [00:28, 1074.34 examples/s]30599 examples [00:28, 1048.39 examples/s]30705 examples [00:28, 1030.90 examples/s]30811 examples [00:28, 1039.46 examples/s]30922 examples [00:28, 1058.08 examples/s]31034 examples [00:28, 1075.30 examples/s]31142 examples [00:28, 1046.95 examples/s]31252 examples [00:29, 1061.22 examples/s]31360 examples [00:29, 1065.26 examples/s]31467 examples [00:29, 1045.54 examples/s]31575 examples [00:29, 1055.58 examples/s]31681 examples [00:29, 1031.72 examples/s]31789 examples [00:29, 1044.11 examples/s]31897 examples [00:29, 1054.07 examples/s]32009 examples [00:29, 1071.05 examples/s]32120 examples [00:29, 1080.70 examples/s]32233 examples [00:29, 1091.09 examples/s]32343 examples [00:30, 1084.18 examples/s]32455 examples [00:30, 1093.89 examples/s]32565 examples [00:30, 1088.56 examples/s]32677 examples [00:30, 1095.77 examples/s]32787 examples [00:30, 1091.34 examples/s]32897 examples [00:30, 1071.75 examples/s]33005 examples [00:30, 1012.61 examples/s]33108 examples [00:30, 1009.76 examples/s]33221 examples [00:30, 1042.77 examples/s]33326 examples [00:30, 1036.81 examples/s]33436 examples [00:31, 1053.08 examples/s]33550 examples [00:31, 1075.13 examples/s]33661 examples [00:31, 1083.80 examples/s]33774 examples [00:31, 1095.52 examples/s]33884 examples [00:31, 1089.68 examples/s]33995 examples [00:31, 1093.02 examples/s]34105 examples [00:31, 1092.68 examples/s]34216 examples [00:31, 1097.50 examples/s]34329 examples [00:31, 1105.79 examples/s]34440 examples [00:31, 1097.12 examples/s]34550 examples [00:32, 1091.35 examples/s]34663 examples [00:32, 1100.14 examples/s]34774 examples [00:32, 1072.98 examples/s]34882 examples [00:32, 1070.57 examples/s]34993 examples [00:32, 1081.68 examples/s]35102 examples [00:32, 1081.14 examples/s]35211 examples [00:32, 1082.41 examples/s]35325 examples [00:32, 1097.37 examples/s]35435 examples [00:32, 1066.64 examples/s]35542 examples [00:33, 1047.24 examples/s]35650 examples [00:33, 1054.42 examples/s]35758 examples [00:33, 1060.93 examples/s]35868 examples [00:33, 1070.89 examples/s]35976 examples [00:33, 1058.17 examples/s]36088 examples [00:33, 1072.89 examples/s]36196 examples [00:33, 1068.90 examples/s]36304 examples [00:33, 1071.48 examples/s]36414 examples [00:33, 1077.20 examples/s]36527 examples [00:33, 1091.37 examples/s]36641 examples [00:34, 1102.05 examples/s]36752 examples [00:34, 1088.82 examples/s]36863 examples [00:34, 1093.30 examples/s]36977 examples [00:34, 1104.72 examples/s]37088 examples [00:34, 1103.85 examples/s]37199 examples [00:34, 1101.21 examples/s]37311 examples [00:34, 1105.96 examples/s]37422 examples [00:34, 1099.51 examples/s]37536 examples [00:34, 1109.55 examples/s]37648 examples [00:34, 1111.36 examples/s]37760 examples [00:35, 1101.32 examples/s]37871 examples [00:35, 1103.58 examples/s]37984 examples [00:35, 1110.47 examples/s]38096 examples [00:35, 1079.54 examples/s]38206 examples [00:35, 1085.47 examples/s]38315 examples [00:35, 1066.74 examples/s]38422 examples [00:35, 1055.02 examples/s]38532 examples [00:35, 1065.44 examples/s]38641 examples [00:35, 1072.45 examples/s]38751 examples [00:35, 1079.73 examples/s]38862 examples [00:36, 1087.19 examples/s]38971 examples [00:36, 1077.79 examples/s]39080 examples [00:36, 1079.51 examples/s]39191 examples [00:36, 1087.52 examples/s]39301 examples [00:36, 1088.83 examples/s]39412 examples [00:36, 1093.92 examples/s]39522 examples [00:36, 1085.28 examples/s]39631 examples [00:36, 1084.17 examples/s]39743 examples [00:36, 1092.50 examples/s]39853 examples [00:36, 1093.49 examples/s]39963 examples [00:37, 1086.80 examples/s]40072 examples [00:37, 1021.71 examples/s]40186 examples [00:37, 1054.48 examples/s]40293 examples [00:37, 1050.84 examples/s]40399 examples [00:37, 1029.19 examples/s]40505 examples [00:37, 1037.89 examples/s]40614 examples [00:37, 1052.67 examples/s]40720 examples [00:37, 1053.65 examples/s]40832 examples [00:37, 1072.30 examples/s]40943 examples [00:38, 1082.68 examples/s]41053 examples [00:38, 1087.24 examples/s]41162 examples [00:38, 1084.27 examples/s]41273 examples [00:38, 1089.35 examples/s]41383 examples [00:38, 1053.40 examples/s]41491 examples [00:38, 1058.99 examples/s]41600 examples [00:38, 1067.62 examples/s]41714 examples [00:38, 1086.41 examples/s]41827 examples [00:38, 1096.13 examples/s]41937 examples [00:38, 1084.17 examples/s]42046 examples [00:39, 1080.01 examples/s]42155 examples [00:39, 1042.31 examples/s]42262 examples [00:39, 1049.45 examples/s]42374 examples [00:39, 1068.12 examples/s]42484 examples [00:39, 1075.50 examples/s]42594 examples [00:39, 1082.30 examples/s]42704 examples [00:39, 1087.49 examples/s]42818 examples [00:39, 1101.24 examples/s]42930 examples [00:39, 1106.79 examples/s]43041 examples [00:39, 1107.59 examples/s]43152 examples [00:40, 1079.65 examples/s]43262 examples [00:40, 1082.71 examples/s]43374 examples [00:40, 1092.95 examples/s]43484 examples [00:40, 1091.64 examples/s]43594 examples [00:40, 1081.09 examples/s]43703 examples [00:40, 1081.54 examples/s]43812 examples [00:40, 1076.37 examples/s]43921 examples [00:40, 1079.95 examples/s]44031 examples [00:40, 1085.40 examples/s]44140 examples [00:40, 1086.39 examples/s]44250 examples [00:41, 1089.86 examples/s]44360 examples [00:41, 1086.90 examples/s]44469 examples [00:41, 1081.19 examples/s]44578 examples [00:41, 1064.82 examples/s]44685 examples [00:41, 1030.14 examples/s]44797 examples [00:41, 1055.49 examples/s]44903 examples [00:41, 1050.68 examples/s]45012 examples [00:41, 1060.79 examples/s]45122 examples [00:41, 1071.42 examples/s]45231 examples [00:41, 1074.16 examples/s]45339 examples [00:42, 1045.83 examples/s]45445 examples [00:42, 1049.55 examples/s]45553 examples [00:42, 1056.81 examples/s]45663 examples [00:42, 1067.33 examples/s]45776 examples [00:42, 1084.93 examples/s]45888 examples [00:42, 1094.64 examples/s]45998 examples [00:42, 1086.80 examples/s]46107 examples [00:42, 1083.13 examples/s]46217 examples [00:42, 1087.66 examples/s]46328 examples [00:43, 1092.39 examples/s]46440 examples [00:43, 1100.37 examples/s]46551 examples [00:43, 1088.48 examples/s]46660 examples [00:43, 1083.84 examples/s]46771 examples [00:43, 1090.30 examples/s]46881 examples [00:43, 1087.95 examples/s]46990 examples [00:43, 1079.10 examples/s]47098 examples [00:43, 1077.71 examples/s]47208 examples [00:43, 1082.54 examples/s]47317 examples [00:43, 1082.26 examples/s]47426 examples [00:44, 1069.84 examples/s]47534 examples [00:44, 1060.18 examples/s]47641 examples [00:44, 1052.97 examples/s]47752 examples [00:44, 1067.67 examples/s]47863 examples [00:44, 1079.10 examples/s]47971 examples [00:44, 1014.63 examples/s]48079 examples [00:44, 1032.44 examples/s]48185 examples [00:44, 1040.11 examples/s]48293 examples [00:44, 1051.74 examples/s]48403 examples [00:44, 1064.73 examples/s]48510 examples [00:45, 1058.31 examples/s]48618 examples [00:45, 1061.86 examples/s]48725 examples [00:45, 1060.51 examples/s]48834 examples [00:45, 1066.27 examples/s]48943 examples [00:45, 1070.46 examples/s]49052 examples [00:45, 1074.24 examples/s]49160 examples [00:45, 1045.44 examples/s]49265 examples [00:45, 1033.98 examples/s]49372 examples [00:45, 1044.36 examples/s]49485 examples [00:45, 1066.79 examples/s]49598 examples [00:46, 1082.81 examples/s]49709 examples [00:46, 1090.43 examples/s]49819 examples [00:46, 1068.41 examples/s]49928 examples [00:46, 1072.42 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 14%|█▍        | 7221/50000 [00:00<00:00, 72209.04 examples/s] 42%|████▏     | 21009/50000 [00:00<00:00, 84246.48 examples/s] 69%|██████▉   | 34510/50000 [00:00<00:00, 94955.95 examples/s] 96%|█████████▋| 48203/50000 [00:00<00:00, 104571.14 examples/s]                                                                0 examples [00:00, ? examples/s]93 examples [00:00, 927.60 examples/s]203 examples [00:00, 972.82 examples/s]317 examples [00:00, 1015.56 examples/s]430 examples [00:00, 1045.04 examples/s]546 examples [00:00, 1074.87 examples/s]660 examples [00:00, 1093.00 examples/s]760 examples [00:00, 1063.15 examples/s]870 examples [00:00, 1073.60 examples/s]982 examples [00:00, 1086.40 examples/s]1096 examples [00:01, 1100.80 examples/s]1212 examples [00:01, 1116.57 examples/s]1324 examples [00:01, 1115.77 examples/s]1435 examples [00:01, 857.29 examples/s] 1546 examples [00:01, 919.41 examples/s]1651 examples [00:01, 954.70 examples/s]1761 examples [00:01, 993.88 examples/s]1874 examples [00:01, 1028.68 examples/s]1985 examples [00:01, 1050.81 examples/s]2098 examples [00:02, 1070.83 examples/s]2211 examples [00:02, 1085.97 examples/s]2325 examples [00:02, 1100.19 examples/s]2436 examples [00:02, 1089.34 examples/s]2546 examples [00:02, 1049.17 examples/s]2652 examples [00:02, 1034.81 examples/s]2758 examples [00:02, 1042.09 examples/s]2867 examples [00:02, 1054.42 examples/s]2981 examples [00:02, 1076.37 examples/s]3094 examples [00:02, 1090.19 examples/s]3204 examples [00:03, 1085.12 examples/s]3320 examples [00:03, 1103.89 examples/s]3436 examples [00:03, 1118.44 examples/s]3551 examples [00:03, 1125.87 examples/s]3664 examples [00:03, 1124.38 examples/s]3777 examples [00:03, 1108.79 examples/s]3890 examples [00:03, 1114.48 examples/s]4002 examples [00:03, 1076.66 examples/s]4112 examples [00:03, 1082.86 examples/s]4222 examples [00:03, 1085.44 examples/s]4334 examples [00:04, 1094.75 examples/s]4449 examples [00:04, 1109.35 examples/s]4565 examples [00:04, 1122.51 examples/s]4679 examples [00:04, 1126.64 examples/s]4792 examples [00:04, 1117.45 examples/s]4904 examples [00:04, 1072.50 examples/s]5013 examples [00:04, 1075.72 examples/s]5121 examples [00:04, 1072.67 examples/s]5233 examples [00:04, 1084.41 examples/s]5342 examples [00:04, 1081.93 examples/s]5451 examples [00:05, 1070.98 examples/s]5564 examples [00:05, 1087.34 examples/s]5673 examples [00:05, 1087.93 examples/s]5786 examples [00:05, 1099.42 examples/s]5897 examples [00:05, 1093.03 examples/s]6008 examples [00:05, 1097.12 examples/s]6121 examples [00:05, 1105.97 examples/s]6235 examples [00:05, 1115.36 examples/s]6349 examples [00:05, 1120.82 examples/s]6462 examples [00:05, 1119.44 examples/s]6574 examples [00:06, 1115.89 examples/s]6689 examples [00:06, 1125.28 examples/s]6802 examples [00:06, 1118.12 examples/s]6917 examples [00:06, 1125.46 examples/s]7030 examples [00:06, 1117.97 examples/s]7142 examples [00:06, 1113.39 examples/s]7254 examples [00:06, 1107.57 examples/s]7365 examples [00:06, 1072.31 examples/s]7475 examples [00:06, 1079.32 examples/s]7586 examples [00:07, 1085.76 examples/s]7697 examples [00:07, 1091.94 examples/s]7811 examples [00:07, 1105.56 examples/s]7922 examples [00:07, 1092.61 examples/s]8032 examples [00:07, 1090.27 examples/s]8142 examples [00:07, 1084.64 examples/s]8251 examples [00:07, 1083.13 examples/s]8362 examples [00:07, 1088.54 examples/s]8475 examples [00:07, 1098.96 examples/s]8589 examples [00:07, 1109.60 examples/s]8701 examples [00:08, 1106.24 examples/s]8812 examples [00:08, 1103.69 examples/s]8926 examples [00:08, 1112.96 examples/s]9040 examples [00:08, 1119.65 examples/s]9153 examples [00:08, 1118.56 examples/s]9265 examples [00:08, 1110.48 examples/s]9377 examples [00:08, 1106.81 examples/s]9491 examples [00:08, 1114.25 examples/s]9603 examples [00:08, 1113.88 examples/s]9715 examples [00:08, 1112.47 examples/s]9828 examples [00:09, 1115.74 examples/s]9940 examples [00:09, 1110.12 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteX9KUYZ/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteX9KUYZ/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f85359be950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f85359be950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f85359be950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f84bee12eb8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f84bee00b00>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f85359be950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f85359be950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f85359be950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f84bed85fd0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f851bd80048>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f84ad3b3268> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f84ad3b3268> 

  function with postional parmater data_info <function split_train_valid at 0x7f84ad3b3268> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f84ad3b3378> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f84ad3b3378> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f84ad3b3378> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.1)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.5)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.1)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.2)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=9227d4b7c5b0b47eba287f8c8def6245b40c8373d0aa721b15475a8c198dfb5e
  Stored in directory: /tmp/pip-ephem-wheel-cache-ngr5m4z0/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<21:39:51, 11.1kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<15:24:07, 15.5kB/s].vector_cache/glove.6B.zip:   0%|          | 213k/862M [00:00<10:49:41, 22.1kB/s] .vector_cache/glove.6B.zip:   0%|          | 885k/862M [00:01<7:35:04, 31.5kB/s] .vector_cache/glove.6B.zip:   0%|          | 1.78M/862M [00:01<5:18:43, 45.0kB/s].vector_cache/glove.6B.zip:   1%|          | 4.57M/862M [00:01<3:42:32, 64.2kB/s].vector_cache/glove.6B.zip:   1%|          | 8.96M/862M [00:01<2:35:04, 91.7kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.2M/862M [00:01<1:48:15, 131kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 17.8M/862M [00:01<1:15:23, 187kB/s].vector_cache/glove.6B.zip:   3%|▎         | 23.4M/862M [00:01<52:30, 266kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 26.4M/862M [00:01<36:46, 379kB/s].vector_cache/glove.6B.zip:   4%|▎         | 32.1M/862M [00:01<25:39, 539kB/s].vector_cache/glove.6B.zip:   4%|▍         | 37.4M/862M [00:02<17:55, 767kB/s].vector_cache/glove.6B.zip:   5%|▍         | 40.9M/862M [00:02<12:37, 1.08MB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.5M/862M [00:02<08:52, 1.53MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.2M/862M [00:02<06:14, 2.16MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.4M/862M [00:02<18:34, 727kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 56.1M/862M [00:03<13:03, 1.03MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.5M/862M [00:04<29:02, 462kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 56.7M/862M [00:05<22:42, 591kB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.7M/862M [00:05<16:22, 818kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.3M/862M [00:05<11:34, 1.15MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.6M/862M [00:06<29:26, 454kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 61.0M/862M [00:07<21:58, 608kB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.6M/862M [00:07<15:42, 849kB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.7M/862M [00:08<14:03, 945kB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.1M/862M [00:08<11:13, 1.18MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.7M/862M [00:09<08:11, 1.62MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.8M/862M [00:10<08:50, 1.50MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.2M/862M [00:10<07:32, 1.75MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.8M/862M [00:11<05:36, 2.35MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.0M/862M [00:12<07:01, 1.87MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.2M/862M [00:12<07:37, 1.72MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.9M/862M [00:13<06:00, 2.19MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.1M/862M [00:14<06:18, 2.07MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.5M/862M [00:14<05:45, 2.27MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.0M/862M [00:15<04:22, 2.98MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.2M/862M [00:16<06:05, 2.14MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.6M/862M [00:16<05:36, 2.32MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.2M/862M [00:16<04:15, 3.05MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.3M/862M [00:18<06:01, 2.15MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.7M/862M [00:18<05:34, 2.32MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.3M/862M [00:18<04:13, 3.06MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.4M/862M [00:20<06:00, 2.14MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.8M/862M [00:20<05:31, 2.33MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.4M/862M [00:20<04:11, 3.07MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.5M/862M [00:22<05:56, 2.15MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.9M/862M [00:22<05:29, 2.33MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.5M/862M [00:22<04:07, 3.10MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.3M/862M [00:23<03:39, 3.48MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.3M/862M [00:24<9:14:35, 23.0kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.0M/862M [00:24<6:28:36, 32.8kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:24<4:31:13, 46.8kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<3:18:15, 63.9kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<2:21:43, 89.4kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<1:39:46, 127kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:26<1:10:00, 181kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<51:58, 243kB/s]  .vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<37:39, 335kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 108M/862M [00:28<26:35, 473kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<21:32, 582kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<17:39, 710kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<12:53, 971kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<09:14, 1.35MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<10:15, 1.22MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<08:27, 1.47MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<06:10, 2.01MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<07:13, 1.71MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<06:20, 1.95MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<04:45, 2.60MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<06:13, 1.98MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<05:37, 2.19MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<04:14, 2.89MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<05:52, 2.09MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<05:20, 2.29MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<04:03, 3.02MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<05:42, 2.13MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<05:15, 2.32MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<03:58, 3.05MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:42<05:38, 2.15MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<05:12, 2.33MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<03:56, 3.06MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<05:36, 2.15MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<05:10, 2.33MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<03:55, 3.06MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<05:34, 2.15MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<05:07, 2.33MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<03:53, 3.07MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<05:32, 2.15MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<05:06, 2.34MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<03:52, 3.07MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:49<05:30, 2.15MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<05:04, 2.34MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<03:50, 3.07MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<05:28, 2.15MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<05:02, 2.34MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<03:49, 3.07MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<05:26, 2.15MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<04:59, 2.34MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<03:44, 3.13MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<05:23, 2.16MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:55<04:46, 2.44MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<03:37, 3.21MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:56<02:40, 4.33MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<48:31, 239kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<35:08, 329kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<24:48, 466kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<20:02, 574kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<16:24, 701kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<12:04, 952kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<10:15, 1.11MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<08:22, 1.36MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:01<06:06, 1.87MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<06:56, 1.64MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<06:02, 1.88MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:03<04:30, 2.51MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<05:47, 1.95MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<06:23, 1.77MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:05<05:03, 2.23MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<05:20, 2.10MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<04:53, 2.30MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:07<03:42, 3.02MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<05:11, 2.15MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<04:47, 2.33MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:09<03:37, 3.07MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:10<03:15, 3.40MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<7:46:38, 23.8kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<5:26:57, 33.9kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:11<3:48:05, 48.4kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<2:47:27, 65.9kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<1:59:27, 92.4kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<1:24:06, 131kB/s] .vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:13<58:44, 187kB/s]  .vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<1:59:12, 92.0kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:15<1:24:31, 130kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<59:19, 184kB/s]  .vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<43:55, 248kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<31:53, 342kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<22:32, 482kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:19<18:15, 593kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<15:07, 716kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<11:08, 971kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:19<07:52, 1.37MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<1:14:36, 144kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<53:19, 202kB/s]  .vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<37:31, 286kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<28:40, 373kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<21:09, 505kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:23<15:03, 708kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<13:01, 816kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<10:12, 1.04MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<07:21, 1.44MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<07:39, 1.38MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<06:26, 1.64MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<04:46, 2.21MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<05:48, 1.81MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<05:08, 2.04MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<03:51, 2.71MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:31<05:09, 2.02MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:31<04:39, 2.23MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<03:29, 2.98MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:33<04:53, 2.11MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:33<05:33, 1.86MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<04:19, 2.39MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:33<03:10, 3.25MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<06:41, 1.54MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 246M/862M [01:35<05:44, 1.79MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<04:14, 2.42MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<05:21, 1.91MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<04:48, 2.12MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<03:36, 2.82MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:38<04:55, 2.06MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<04:28, 2.26MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<03:24, 2.97MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<04:45, 2.12MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<04:22, 2.30MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<03:16, 3.07MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<04:40, 2.14MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<04:16, 2.34MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<03:14, 3.08MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<04:36, 2.15MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<04:15, 2.34MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<03:13, 3.07MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<04:34, 2.15MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<05:14, 1.88MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<04:10, 2.36MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<04:29, 2.18MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:48<04:08, 2.36MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:48<03:09, 3.10MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<04:28, 2.18MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<04:07, 2.35MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:50<03:07, 3.10MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<04:27, 2.16MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<04:07, 2.34MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:52<03:05, 3.12MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<04:26, 2.16MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<04:06, 2.34MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:54<03:05, 3.10MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:55<02:43, 3.49MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<6:54:01, 23.0kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:56<4:50:01, 32.8kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:56<3:22:09, 46.9kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<2:28:41, 63.6kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<1:46:00, 89.2kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<1:14:35, 127kB/s] .vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<53:26, 176kB/s]  .vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<38:21, 245kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:00<27:01, 346kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<21:01, 443kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<16:38, 560kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<12:03, 772kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:02<08:33, 1.08MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<09:38, 960kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<07:43, 1.20MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:04<05:36, 1.65MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<06:01, 1.52MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<06:06, 1.50MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<04:39, 1.97MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:06<03:24, 2.69MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<05:41, 1.60MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<04:56, 1.85MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<03:41, 2.46MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<04:40, 1.93MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<05:08, 1.76MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<03:59, 2.26MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:10<02:53, 3.10MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:12<08:24, 1.07MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<06:48, 1.32MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<04:58, 1.80MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:14<05:33, 1.60MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<04:48, 1.85MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<03:35, 2.47MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:16<04:34, 1.93MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:16<04:06, 2.15MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<03:03, 2.88MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<04:13, 2.08MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<03:51, 2.27MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:18<02:55, 2.99MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<04:05, 2.13MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<03:46, 2.31MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<02:48, 3.08MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<04:01, 2.14MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<03:42, 2.33MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<02:46, 3.11MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<03:58, 2.16MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:24<03:39, 2.34MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<02:46, 3.08MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<03:56, 2.16MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<03:38, 2.34MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<02:45, 3.07MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<03:54, 2.15MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:27<03:37, 2.33MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<02:42, 3.10MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<03:52, 2.16MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<03:34, 2.34MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<02:42, 3.08MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<03:51, 2.15MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<03:32, 2.35MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<02:41, 3.07MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<03:48, 2.16MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<03:31, 2.33MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:33<02:40, 3.07MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<03:47, 2.15MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<03:29, 2.34MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:35<02:36, 3.11MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<03:44, 2.16MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<03:27, 2.34MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:37<02:37, 3.07MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<03:43, 2.16MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<03:25, 2.34MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:39<02:33, 3.12MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:40<02:18, 3.45MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<5:40:22, 23.4kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<3:58:23, 33.3kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:41<2:46:01, 47.6kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<2:01:21, 65.0kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<1:26:36, 91.1kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<1:00:53, 129kB/s] .vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:43<42:36, 184kB/s]  .vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<32:10, 243kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<23:20, 335kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:45<16:27, 473kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<13:15, 585kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<10:52, 712kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<07:56, 974kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:47<05:38, 1.36MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<07:18, 1.05MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<05:53, 1.30MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:49<04:18, 1.77MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<04:47, 1.59MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<03:59, 1.91MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<02:56, 2.57MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:51<02:09, 3.50MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<29:59, 252kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<21:45, 347kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<15:22, 489kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:55<12:28, 599kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<09:29, 787kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<06:48, 1.09MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<06:30, 1.14MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<05:19, 1.39MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:57<03:54, 1.89MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<04:27, 1.65MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<03:52, 1.89MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<02:53, 2.53MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<03:44, 1.95MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<03:21, 2.16MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<02:32, 2.86MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<03:28, 2.08MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<03:10, 2.26MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<02:24, 2.98MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<03:20, 2.13MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:05<03:05, 2.31MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<02:20, 3.04MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<03:17, 2.15MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:07<03:45, 1.88MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<02:59, 2.35MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:08<03:12, 2.18MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<02:59, 2.34MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:09<02:15, 3.08MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<03:11, 2.17MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<02:56, 2.35MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<02:13, 3.09MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<03:10, 2.16MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<02:55, 2.34MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<02:11, 3.12MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<03:08, 2.16MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<02:53, 2.34MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<02:09, 3.12MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<03:06, 2.16MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<02:52, 2.34MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:16<02:09, 3.11MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<03:04, 2.16MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<02:50, 2.34MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:18<02:09, 3.08MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<03:03, 2.16MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<02:48, 2.34MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:20<02:07, 3.07MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<03:01, 2.15MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<02:47, 2.33MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:22<02:06, 3.07MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<02:59, 2.15MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<02:45, 2.33MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:24<02:05, 3.07MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:25<01:53, 3.39MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<4:26:28, 24.0kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<3:06:34, 34.2kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:26<2:09:45, 48.8kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<1:35:06, 66.4kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<1:07:51, 93.0kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<47:41, 132kB/s]   .vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:28<33:14, 188kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:30<27:39, 226kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<19:59, 312kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<14:06, 441kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<11:15, 549kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<09:08, 675kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<06:42, 919kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [03:34<05:38, 1.08MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<04:35, 1.33MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<03:21, 1.81MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<03:45, 1.61MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<03:52, 1.56MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<03:00, 2.00MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<03:03, 1.95MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<02:46, 2.16MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<02:05, 2.85MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<02:49, 2.09MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<03:11, 1.85MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<02:31, 2.32MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<02:42, 2.16MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<02:30, 2.33MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<01:53, 3.06MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<02:39, 2.17MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<02:27, 2.35MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<01:51, 3.09MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:46<02:42, 2.10MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:46<03:04, 1.85MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<02:23, 2.37MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:46<01:45, 3.22MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:48<03:31, 1.60MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<03:02, 1.85MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<02:15, 2.47MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:50<02:52, 1.93MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<03:09, 1.76MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<02:29, 2.22MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:50<01:47, 3.06MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<40:51, 134kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<29:07, 188kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<20:26, 267kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:53<15:28, 351kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<11:23, 476kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:54<08:04, 668kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<06:52, 780kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<05:21, 999kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<03:52, 1.38MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<03:56, 1.34MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:58<03:17, 1.60MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<02:24, 2.18MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<02:55, 1.79MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:00<02:34, 2.02MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<01:55, 2.68MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<02:33, 2.01MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:01<02:19, 2.22MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<01:44, 2.93MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<02:25, 2.10MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:03<02:13, 2.29MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<01:40, 3.02MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<02:20, 2.13MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<02:09, 2.32MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<01:38, 3.05MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<02:18, 2.15MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<02:07, 2.33MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:07<01:36, 3.06MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<02:16, 2.15MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<02:05, 2.33MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:09<01:33, 3.10MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<02:13, 2.16MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<02:03, 2.33MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:11<01:32, 3.10MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<02:11, 2.16MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<02:01, 2.34MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:13<01:31, 3.08MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:14<01:18, 3.56MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:15<3:37:31, 21.5kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<2:32:06, 30.6kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:15<1:45:24, 43.7kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<1:20:01, 57.5kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<56:56, 80.8kB/s]  .vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<39:58, 115kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<28:24, 160kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<20:20, 223kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:19<14:16, 315kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<10:56, 408kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<08:07, 550kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:21<05:45, 769kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<05:02, 874kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<04:25, 993kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<03:17, 1.33MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:23<02:20, 1.86MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<04:22, 989kB/s] .vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<03:30, 1.23MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<02:33, 1.68MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:27<02:46, 1.54MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<02:22, 1.79MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:27<01:45, 2.40MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<02:12, 1.90MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<02:24, 1.74MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<01:51, 2.25MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:29<01:21, 3.04MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<02:19, 1.78MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<02:02, 2.01MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:31<01:31, 2.67MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<02:00, 2.02MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<01:48, 2.24MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:33<01:21, 2.95MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<01:53, 2.11MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<01:43, 2.30MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:35<01:18, 3.03MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:37<01:49, 2.14MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:37<01:41, 2.32MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<01:15, 3.10MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:39<01:47, 2.15MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<01:38, 2.33MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<01:14, 3.07MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<01:45, 2.15MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:41<01:36, 2.34MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<01:13, 3.07MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<01:43, 2.15MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<01:35, 2.33MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<01:11, 3.07MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<01:41, 2.15MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<01:29, 2.43MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<01:08, 3.18MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<01:37, 2.19MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<01:30, 2.37MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<01:07, 3.15MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<01:36, 2.18MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<01:29, 2.35MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<01:07, 3.09MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<01:35, 2.16MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<01:28, 2.34MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<01:06, 3.07MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<01:33, 2.15MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<01:47, 1.88MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<01:25, 2.36MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<01:30, 2.18MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<01:24, 2.35MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:54<01:03, 3.09MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<01:29, 2.17MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<01:22, 2.35MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:56<01:02, 3.08MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<01:27, 2.16MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<01:21, 2.33MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:58<01:01, 3.07MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<01:26, 2.15MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<01:19, 2.34MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:00<00:59, 3.07MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<01:24, 2.15MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<01:17, 2.34MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:02<00:57, 3.11MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:03<00:50, 3.49MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<2:15:00, 21.9kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<1:34:15, 31.3kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:04<1:04:54, 44.7kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<49:06, 58.9kB/s]  .vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<34:56, 82.7kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [05:06<24:28, 118kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:06<16:56, 168kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<13:17, 212kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<09:34, 294kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:08<06:43, 416kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<05:16, 522kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<03:58, 692kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:10<02:48, 968kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<02:35, 1.04MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<02:04, 1.29MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:12<01:29, 1.77MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<01:39, 1.58MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<01:25, 1.83MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:14<01:03, 2.45MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:16<01:20, 1.91MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:11, 2.13MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<00:52, 2.85MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:18<01:12, 2.07MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:05, 2.26MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<00:49, 2.98MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:20<01:07, 2.13MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:02, 2.31MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<00:47, 3.04MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:22<01:05, 2.15MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:22<01:00, 2.31MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:22<00:45, 3.07MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<01:03, 2.16MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<00:58, 2.34MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:24<00:43, 3.07MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<01:01, 2.15MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<00:56, 2.35MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<00:42, 3.09MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:28<00:59, 2.16MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:28<00:54, 2.34MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<00:40, 3.12MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:30<00:57, 2.16MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:30<00:52, 2.34MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<00:39, 3.08MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<00:55, 2.16MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<00:51, 2.33MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<00:38, 3.09MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:33<00:53, 2.16MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<00:49, 2.34MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<00:36, 3.11MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<00:51, 2.16MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<00:47, 2.34MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<00:35, 3.07MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 754M/862M [05:37<00:49, 2.17MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:37<00:45, 2.35MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<00:34, 3.09MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<00:47, 2.16MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<00:44, 2.34MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<00:32, 3.08MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:41<00:46, 2.15MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:41<00:42, 2.34MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<00:31, 3.07MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<00:44, 2.15MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<00:40, 2.34MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<00:30, 3.07MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<00:42, 2.15MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<00:38, 2.34MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:45<00:29, 3.07MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<00:40, 2.15MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<00:36, 2.35MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:47<00:27, 3.12MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<00:38, 2.16MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<00:35, 2.34MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:49<00:26, 3.07MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<00:36, 2.15MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<00:33, 2.34MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:51<00:24, 3.11MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:52<00:20, 3.57MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<58:51, 21.2kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<40:50, 30.2kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:53<27:25, 43.2kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<21:05, 55.9kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<14:57, 78.6kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<10:24, 112kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<07:08, 155kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<05:05, 217kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<03:30, 308kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<02:36, 399kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<01:55, 539kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [05:59<01:20, 754kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<01:08, 859kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<00:53, 1.09MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:01<00:37, 1.50MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<00:38, 1.42MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<00:32, 1.68MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:03<00:23, 2.26MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<00:27, 1.83MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<00:24, 2.06MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:05<00:17, 2.73MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:07<00:22, 2.03MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:07<00:20, 2.23MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<00:14, 2.95MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:09<00:19, 2.11MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:18, 2.30MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:13, 3.03MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:11<00:17, 2.14MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:16, 2.32MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:11, 3.06MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:13<00:15, 2.14MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:14, 2.34MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<00:10, 3.07MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:13, 2.16MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:12, 2.34MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:15<00:09, 3.07MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<00:11, 2.15MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<00:10, 2.34MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<00:07, 3.07MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:18<00:09, 2.15MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:19<00:08, 2.33MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:06, 3.07MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:20<00:08, 2.16MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:21<00:07, 2.34MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:04, 3.07MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:22<00:06, 2.16MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:23<00:06, 1.89MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:05, 2.36MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:23<00:02, 3.23MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<01:06, 136kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:45, 191kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:26, 271kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:13, 355kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:26<00:09, 482kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:04, 676kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:01, 788kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:28<00:00, 906kB/s].vector_cache/glove.6B.zip: 862MB [06:29, 2.22MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<15:24:06,  7.21it/s]  0%|          | 869/400000 [00:00<10:45:41, 10.30it/s]  0%|          | 1766/400000 [00:00<7:31:11, 14.71it/s]  1%|          | 2657/400000 [00:00<5:15:21, 21.00it/s]  1%|          | 3559/400000 [00:00<3:40:27, 29.97it/s]  1%|          | 4378/400000 [00:00<2:34:14, 42.75it/s]  1%|▏         | 5285/400000 [00:00<1:47:56, 60.94it/s]  2%|▏         | 6199/400000 [00:00<1:15:36, 86.81it/s]  2%|▏         | 7110/400000 [00:00<53:00, 123.52it/s]   2%|▏         | 8013/400000 [00:01<37:14, 175.42it/s]  2%|▏         | 8927/400000 [00:01<26:13, 248.56it/s]  2%|▏         | 9805/400000 [00:01<18:33, 350.39it/s]  3%|▎         | 10683/400000 [00:01<13:11, 492.14it/s]  3%|▎         | 11573/400000 [00:01<09:25, 686.78it/s]  3%|▎         | 12443/400000 [00:01<06:48, 948.14it/s]  3%|▎         | 13306/400000 [00:01<04:59, 1290.21it/s]  4%|▎         | 14155/400000 [00:01<03:43, 1729.21it/s]  4%|▍         | 15036/400000 [00:01<02:48, 2278.60it/s]  4%|▍         | 15937/400000 [00:01<02:10, 2936.72it/s]  4%|▍         | 16807/400000 [00:02<01:44, 3654.60it/s]  4%|▍         | 17671/400000 [00:02<01:26, 4415.00it/s]  5%|▍         | 18553/400000 [00:02<01:13, 5192.15it/s]  5%|▍         | 19435/400000 [00:02<01:04, 5922.50it/s]  5%|▌         | 20324/400000 [00:02<00:57, 6581.42it/s]  5%|▌         | 21203/400000 [00:02<00:53, 7117.14it/s]  6%|▌         | 22114/400000 [00:02<00:49, 7616.08it/s]  6%|▌         | 23002/400000 [00:02<00:47, 7855.35it/s]  6%|▌         | 23901/400000 [00:02<00:46, 8163.35it/s]  6%|▌         | 24811/400000 [00:02<00:44, 8422.91it/s]  6%|▋         | 25702/400000 [00:03<00:43, 8539.03it/s]  7%|▋         | 26620/400000 [00:03<00:42, 8721.58it/s]  7%|▋         | 27518/400000 [00:03<00:42, 8665.13it/s]  7%|▋         | 28409/400000 [00:03<00:42, 8736.63it/s]  7%|▋         | 29315/400000 [00:03<00:41, 8828.58it/s]  8%|▊         | 30219/400000 [00:03<00:41, 8888.48it/s]  8%|▊         | 31124/400000 [00:03<00:41, 8933.87it/s]  8%|▊         | 32022/400000 [00:03<00:41, 8860.74it/s]  8%|▊         | 32931/400000 [00:03<00:41, 8927.32it/s]  8%|▊         | 33827/400000 [00:03<00:41, 8924.91it/s]  9%|▊         | 34736/400000 [00:04<00:40, 8972.63it/s]  9%|▉         | 35651/400000 [00:04<00:40, 9024.72it/s]  9%|▉         | 36555/400000 [00:04<00:40, 8905.18it/s]  9%|▉         | 37469/400000 [00:04<00:40, 8972.43it/s] 10%|▉         | 38368/400000 [00:04<00:40, 8965.59it/s] 10%|▉         | 39266/400000 [00:04<00:40, 8903.39it/s] 10%|█         | 40157/400000 [00:04<00:40, 8897.92it/s] 10%|█         | 41048/400000 [00:04<00:40, 8889.93it/s] 10%|█         | 41962/400000 [00:04<00:39, 8963.24it/s] 11%|█         | 42859/400000 [00:04<00:40, 8871.22it/s] 11%|█         | 43784/400000 [00:05<00:39, 8980.18it/s] 11%|█         | 44705/400000 [00:05<00:39, 9046.05it/s] 11%|█▏        | 45611/400000 [00:05<00:39, 9033.16it/s] 12%|█▏        | 46515/400000 [00:05<00:40, 8703.18it/s] 12%|█▏        | 47389/400000 [00:05<00:41, 8594.53it/s] 12%|█▏        | 48251/400000 [00:05<00:41, 8502.40it/s] 12%|█▏        | 49164/400000 [00:05<00:40, 8679.17it/s] 13%|█▎        | 50068/400000 [00:05<00:39, 8783.38it/s] 13%|█▎        | 50996/400000 [00:05<00:39, 8925.98it/s] 13%|█▎        | 51891/400000 [00:06<00:39, 8910.93it/s] 13%|█▎        | 52813/400000 [00:06<00:38, 8998.89it/s] 13%|█▎        | 53715/400000 [00:06<00:38, 8999.84it/s] 14%|█▎        | 54616/400000 [00:06<00:38, 8956.95it/s] 14%|█▍        | 55513/400000 [00:06<00:38, 8931.37it/s] 14%|█▍        | 56411/400000 [00:06<00:38, 8943.62it/s] 14%|█▍        | 57341/400000 [00:06<00:37, 9045.27it/s] 15%|█▍        | 58259/400000 [00:06<00:37, 9083.35it/s] 15%|█▍        | 59168/400000 [00:06<00:37, 8997.87it/s] 15%|█▌        | 60070/400000 [00:06<00:37, 9003.66it/s] 15%|█▌        | 60971/400000 [00:07<00:37, 8922.73it/s] 15%|█▌        | 61879/400000 [00:07<00:37, 8966.70it/s] 16%|█▌        | 62804/400000 [00:07<00:37, 9049.16it/s] 16%|█▌        | 63710/400000 [00:07<00:37, 8939.19it/s] 16%|█▌        | 64605/400000 [00:07<00:37, 8872.11it/s] 16%|█▋        | 65516/400000 [00:07<00:37, 8939.92it/s] 17%|█▋        | 66411/400000 [00:07<00:37, 8922.22it/s] 17%|█▋        | 67304/400000 [00:07<00:37, 8898.76it/s] 17%|█▋        | 68195/400000 [00:07<00:37, 8750.61it/s] 17%|█▋        | 69071/400000 [00:07<00:38, 8693.54it/s] 17%|█▋        | 69941/400000 [00:08<00:38, 8685.49it/s] 18%|█▊        | 70838/400000 [00:08<00:37, 8767.87it/s] 18%|█▊        | 71740/400000 [00:08<00:37, 8841.20it/s] 18%|█▊        | 72625/400000 [00:08<00:38, 8596.11it/s] 18%|█▊        | 73494/400000 [00:08<00:37, 8621.39it/s] 19%|█▊        | 74421/400000 [00:08<00:36, 8804.30it/s] 19%|█▉        | 75313/400000 [00:08<00:36, 8838.34it/s] 19%|█▉        | 76205/400000 [00:08<00:36, 8860.85it/s] 19%|█▉        | 77093/400000 [00:08<00:36, 8857.03it/s] 20%|█▉        | 78002/400000 [00:08<00:36, 8923.99it/s] 20%|█▉        | 78925/400000 [00:09<00:35, 9011.26it/s] 20%|█▉        | 79842/400000 [00:09<00:35, 9055.33it/s] 20%|██        | 80762/400000 [00:09<00:35, 9096.95it/s] 20%|██        | 81673/400000 [00:09<00:35, 8888.03it/s] 21%|██        | 82564/400000 [00:09<00:36, 8770.18it/s] 21%|██        | 83443/400000 [00:09<00:36, 8578.54it/s] 21%|██        | 84350/400000 [00:09<00:36, 8719.84it/s] 21%|██▏       | 85251/400000 [00:09<00:35, 8804.31it/s] 22%|██▏       | 86133/400000 [00:09<00:36, 8656.57it/s] 22%|██▏       | 87001/400000 [00:09<00:37, 8409.61it/s] 22%|██▏       | 87889/400000 [00:10<00:36, 8543.19it/s] 22%|██▏       | 88754/400000 [00:10<00:36, 8574.88it/s] 22%|██▏       | 89671/400000 [00:10<00:35, 8742.50it/s] 23%|██▎       | 90548/400000 [00:10<00:35, 8687.18it/s] 23%|██▎       | 91468/400000 [00:10<00:34, 8834.87it/s] 23%|██▎       | 92388/400000 [00:10<00:34, 8941.22it/s] 23%|██▎       | 93284/400000 [00:10<00:34, 8933.14it/s] 24%|██▎       | 94179/400000 [00:10<00:34, 8791.06it/s] 24%|██▍       | 95060/400000 [00:10<00:34, 8788.27it/s] 24%|██▍       | 95987/400000 [00:10<00:34, 8925.70it/s] 24%|██▍       | 96881/400000 [00:11<00:34, 8897.29it/s] 24%|██▍       | 97801/400000 [00:11<00:33, 8985.95it/s] 25%|██▍       | 98701/400000 [00:11<00:34, 8747.92it/s] 25%|██▍       | 99578/400000 [00:11<00:34, 8741.80it/s] 25%|██▌       | 100476/400000 [00:11<00:33, 8810.48it/s] 25%|██▌       | 101389/400000 [00:11<00:33, 8859.82it/s] 26%|██▌       | 102276/400000 [00:11<00:33, 8845.69it/s] 26%|██▌       | 103178/400000 [00:11<00:33, 8894.79it/s] 26%|██▌       | 104068/400000 [00:11<00:33, 8804.62it/s] 26%|██▌       | 104972/400000 [00:12<00:33, 8872.16it/s] 26%|██▋       | 105883/400000 [00:12<00:32, 8940.03it/s] 27%|██▋       | 106793/400000 [00:12<00:32, 8985.61it/s] 27%|██▋       | 107700/400000 [00:12<00:32, 9010.68it/s] 27%|██▋       | 108602/400000 [00:12<00:32, 8888.16it/s] 27%|██▋       | 109510/400000 [00:12<00:32, 8944.52it/s] 28%|██▊       | 110412/400000 [00:12<00:32, 8966.15it/s] 28%|██▊       | 111309/400000 [00:12<00:33, 8589.04it/s] 28%|██▊       | 112172/400000 [00:12<00:33, 8575.52it/s] 28%|██▊       | 113079/400000 [00:12<00:32, 8716.06it/s] 28%|██▊       | 113978/400000 [00:13<00:32, 8795.00it/s] 29%|██▊       | 114903/400000 [00:13<00:31, 8926.23it/s] 29%|██▉       | 115817/400000 [00:13<00:31, 8987.89it/s] 29%|██▉       | 116718/400000 [00:13<00:32, 8807.85it/s] 29%|██▉       | 117601/400000 [00:13<00:32, 8726.55it/s] 30%|██▉       | 118522/400000 [00:13<00:31, 8865.43it/s] 30%|██▉       | 119411/400000 [00:13<00:31, 8793.20it/s] 30%|███       | 120322/400000 [00:13<00:31, 8884.71it/s] 30%|███       | 121212/400000 [00:13<00:31, 8837.24it/s] 31%|███       | 122104/400000 [00:13<00:31, 8859.59it/s] 31%|███       | 123025/400000 [00:14<00:30, 8960.82it/s] 31%|███       | 123953/400000 [00:14<00:30, 9053.25it/s] 31%|███       | 124866/400000 [00:14<00:30, 9074.00it/s] 31%|███▏      | 125774/400000 [00:14<00:30, 8879.48it/s] 32%|███▏      | 126664/400000 [00:14<00:30, 8850.58it/s] 32%|███▏      | 127570/400000 [00:14<00:30, 8911.41it/s] 32%|███▏      | 128462/400000 [00:14<00:30, 8886.87it/s] 32%|███▏      | 129352/400000 [00:14<00:30, 8879.79it/s] 33%|███▎      | 130249/400000 [00:14<00:30, 8904.82it/s] 33%|███▎      | 131168/400000 [00:14<00:29, 8985.54it/s] 33%|███▎      | 132095/400000 [00:15<00:29, 9067.80it/s] 33%|███▎      | 133003/400000 [00:15<00:29, 9059.99it/s] 33%|███▎      | 133910/400000 [00:15<00:29, 9034.42it/s] 34%|███▎      | 134814/400000 [00:15<00:29, 9012.22it/s] 34%|███▍      | 135716/400000 [00:15<00:29, 8967.29it/s] 34%|███▍      | 136613/400000 [00:15<00:29, 8852.27it/s] 34%|███▍      | 137513/400000 [00:15<00:29, 8894.65it/s] 35%|███▍      | 138403/400000 [00:15<00:29, 8884.53it/s] 35%|███▍      | 139292/400000 [00:15<00:29, 8880.57it/s] 35%|███▌      | 140181/400000 [00:15<00:29, 8870.20it/s] 35%|███▌      | 141094/400000 [00:16<00:28, 8945.91it/s] 36%|███▌      | 142005/400000 [00:16<00:28, 8992.40it/s] 36%|███▌      | 142911/400000 [00:16<00:28, 9010.56it/s] 36%|███▌      | 143813/400000 [00:16<00:28, 8879.61it/s] 36%|███▌      | 144720/400000 [00:16<00:28, 8935.47it/s] 36%|███▋      | 145646/400000 [00:16<00:28, 9028.91it/s] 37%|███▋      | 146550/400000 [00:16<00:28, 8989.29it/s] 37%|███▋      | 147459/400000 [00:16<00:28, 9018.73it/s] 37%|███▋      | 148365/400000 [00:16<00:27, 9029.68it/s] 37%|███▋      | 149269/400000 [00:16<00:27, 9014.80it/s] 38%|███▊      | 150171/400000 [00:17<00:27, 9002.55it/s] 38%|███▊      | 151072/400000 [00:17<00:27, 8968.40it/s] 38%|███▊      | 151996/400000 [00:17<00:27, 9045.34it/s] 38%|███▊      | 152901/400000 [00:17<00:27, 9029.09it/s] 38%|███▊      | 153805/400000 [00:17<00:27, 8988.80it/s] 39%|███▊      | 154705/400000 [00:17<00:27, 8979.28it/s] 39%|███▉      | 155606/400000 [00:17<00:27, 8987.54it/s] 39%|███▉      | 156526/400000 [00:17<00:26, 9049.56it/s] 39%|███▉      | 157432/400000 [00:17<00:27, 8754.87it/s] 40%|███▉      | 158339/400000 [00:17<00:27, 8844.19it/s] 40%|███▉      | 159238/400000 [00:18<00:27, 8885.87it/s] 40%|████      | 160136/400000 [00:18<00:26, 8912.02it/s] 40%|████      | 161050/400000 [00:18<00:26, 8978.94it/s] 40%|████      | 161949/400000 [00:18<00:26, 8885.35it/s] 41%|████      | 162841/400000 [00:18<00:26, 8894.77it/s] 41%|████      | 163748/400000 [00:18<00:26, 8944.13it/s] 41%|████      | 164657/400000 [00:18<00:26, 8986.43it/s] 41%|████▏     | 165556/400000 [00:18<00:26, 8818.16it/s] 42%|████▏     | 166440/400000 [00:18<00:26, 8824.50it/s] 42%|████▏     | 167324/400000 [00:19<00:26, 8633.97it/s] 42%|████▏     | 168224/400000 [00:19<00:26, 8737.97it/s] 42%|████▏     | 169100/400000 [00:19<00:26, 8599.32it/s] 42%|████▏     | 169994/400000 [00:19<00:26, 8697.84it/s] 43%|████▎     | 170891/400000 [00:19<00:26, 8777.18it/s] 43%|████▎     | 171770/400000 [00:19<00:26, 8721.71it/s] 43%|████▎     | 172643/400000 [00:19<00:26, 8660.56it/s] 43%|████▎     | 173510/400000 [00:19<00:26, 8659.63it/s] 44%|████▎     | 174413/400000 [00:19<00:25, 8766.50it/s] 44%|████▍     | 175307/400000 [00:19<00:25, 8817.29it/s] 44%|████▍     | 176190/400000 [00:20<00:25, 8724.75it/s] 44%|████▍     | 177069/400000 [00:20<00:25, 8743.43it/s] 44%|████▍     | 177972/400000 [00:20<00:25, 8824.46it/s] 45%|████▍     | 178885/400000 [00:20<00:24, 8911.36it/s] 45%|████▍     | 179777/400000 [00:20<00:24, 8904.42it/s] 45%|████▌     | 180685/400000 [00:20<00:24, 8955.84it/s] 45%|████▌     | 181581/400000 [00:20<00:24, 8892.07it/s] 46%|████▌     | 182471/400000 [00:20<00:24, 8839.97it/s] 46%|████▌     | 183378/400000 [00:20<00:24, 8905.96it/s] 46%|████▌     | 184289/400000 [00:20<00:24, 8964.53it/s] 46%|████▋     | 185191/400000 [00:21<00:23, 8978.42it/s] 47%|████▋     | 186114/400000 [00:21<00:23, 9049.68it/s] 47%|████▋     | 187037/400000 [00:21<00:23, 9102.23it/s] 47%|████▋     | 187960/400000 [00:21<00:23, 9137.84it/s] 47%|████▋     | 188875/400000 [00:21<00:23, 8970.14it/s] 47%|████▋     | 189773/400000 [00:21<00:23, 8886.14it/s] 48%|████▊     | 190692/400000 [00:21<00:23, 8973.23it/s] 48%|████▊     | 191615/400000 [00:21<00:23, 9046.02it/s] 48%|████▊     | 192521/400000 [00:21<00:22, 9024.57it/s] 48%|████▊     | 193424/400000 [00:21<00:23, 8873.19it/s] 49%|████▊     | 194313/400000 [00:22<00:23, 8817.01it/s] 49%|████▉     | 195222/400000 [00:22<00:23, 8895.95it/s] 49%|████▉     | 196140/400000 [00:22<00:22, 8977.98it/s] 49%|████▉     | 197041/400000 [00:22<00:22, 8986.39it/s] 49%|████▉     | 197941/400000 [00:22<00:22, 8965.05it/s] 50%|████▉     | 198838/400000 [00:22<00:22, 8939.75it/s] 50%|████▉     | 199769/400000 [00:22<00:22, 9047.19it/s] 50%|█████     | 200675/400000 [00:22<00:22, 9026.55it/s] 50%|█████     | 201588/400000 [00:22<00:21, 9055.92it/s] 51%|█████     | 202494/400000 [00:22<00:21, 9033.59it/s] 51%|█████     | 203398/400000 [00:23<00:22, 8872.45it/s] 51%|█████     | 204292/400000 [00:23<00:22, 8892.15it/s] 51%|█████▏    | 205182/400000 [00:23<00:22, 8840.97it/s] 52%|█████▏    | 206067/400000 [00:23<00:22, 8766.53it/s] 52%|█████▏    | 206945/400000 [00:23<00:22, 8508.25it/s] 52%|█████▏    | 207824/400000 [00:23<00:22, 8589.37it/s] 52%|█████▏    | 208698/400000 [00:23<00:22, 8632.33it/s] 52%|█████▏    | 209590/400000 [00:23<00:21, 8715.41it/s] 53%|█████▎    | 210463/400000 [00:23<00:21, 8712.81it/s] 53%|█████▎    | 211335/400000 [00:23<00:21, 8635.71it/s] 53%|█████▎    | 212235/400000 [00:24<00:21, 8741.37it/s] 53%|█████▎    | 213153/400000 [00:24<00:21, 8866.08it/s] 54%|█████▎    | 214056/400000 [00:24<00:20, 8912.05it/s] 54%|█████▎    | 214948/400000 [00:24<00:20, 8879.51it/s] 54%|█████▍    | 215837/400000 [00:24<00:20, 8877.45it/s] 54%|█████▍    | 216732/400000 [00:24<00:20, 8896.80it/s] 54%|█████▍    | 217641/400000 [00:24<00:20, 8952.68it/s] 55%|█████▍    | 218556/400000 [00:24<00:20, 9007.45it/s] 55%|█████▍    | 219471/400000 [00:24<00:19, 9047.24it/s] 55%|█████▌    | 220380/400000 [00:24<00:19, 9056.85it/s] 55%|█████▌    | 221286/400000 [00:25<00:19, 8946.94it/s] 56%|█████▌    | 222206/400000 [00:25<00:19, 9019.79it/s] 56%|█████▌    | 223118/400000 [00:25<00:19, 9046.97it/s] 56%|█████▌    | 224024/400000 [00:25<00:19, 8968.90it/s] 56%|█████▌    | 224939/400000 [00:25<00:19, 9021.00it/s] 56%|█████▋    | 225842/400000 [00:25<00:19, 8953.38it/s] 57%|█████▋    | 226766/400000 [00:25<00:19, 9036.45it/s] 57%|█████▋    | 227671/400000 [00:25<00:19, 8879.42it/s] 57%|█████▋    | 228571/400000 [00:25<00:19, 8914.88it/s] 57%|█████▋    | 229480/400000 [00:25<00:19, 8964.15it/s] 58%|█████▊    | 230389/400000 [00:26<00:18, 8999.36it/s] 58%|█████▊    | 231300/400000 [00:26<00:18, 9032.17it/s] 58%|█████▊    | 232204/400000 [00:26<00:18, 8926.22it/s] 58%|█████▊    | 233098/400000 [00:26<00:18, 8893.36it/s] 59%|█████▊    | 234007/400000 [00:26<00:18, 8951.19it/s] 59%|█████▊    | 234903/400000 [00:26<00:18, 8910.33it/s] 59%|█████▉    | 235817/400000 [00:26<00:18, 8975.26it/s] 59%|█████▉    | 236715/400000 [00:26<00:18, 8945.37it/s] 59%|█████▉    | 237610/400000 [00:26<00:18, 8796.43it/s] 60%|█████▉    | 238491/400000 [00:27<00:18, 8788.78it/s] 60%|█████▉    | 239402/400000 [00:27<00:18, 8881.22it/s] 60%|██████    | 240320/400000 [00:27<00:17, 8967.36it/s] 60%|██████    | 241218/400000 [00:27<00:17, 8955.99it/s] 61%|██████    | 242125/400000 [00:27<00:17, 8988.90it/s] 61%|██████    | 243043/400000 [00:27<00:17, 9043.95it/s] 61%|██████    | 243948/400000 [00:27<00:17, 8955.99it/s] 61%|██████    | 244870/400000 [00:27<00:17, 9032.70it/s] 61%|██████▏   | 245778/400000 [00:27<00:17, 9044.22it/s] 62%|██████▏   | 246683/400000 [00:27<00:17, 9000.79it/s] 62%|██████▏   | 247584/400000 [00:28<00:17, 8812.63it/s] 62%|██████▏   | 248476/400000 [00:28<00:17, 8841.97it/s] 62%|██████▏   | 249361/400000 [00:28<00:17, 8624.55it/s] 63%|██████▎   | 250248/400000 [00:28<00:17, 8694.10it/s] 63%|██████▎   | 251134/400000 [00:28<00:17, 8740.35it/s] 63%|██████▎   | 252025/400000 [00:28<00:16, 8789.40it/s] 63%|██████▎   | 252922/400000 [00:28<00:16, 8840.92it/s] 63%|██████▎   | 253819/400000 [00:28<00:16, 8876.67it/s] 64%|██████▎   | 254723/400000 [00:28<00:16, 8922.94it/s] 64%|██████▍   | 255616/400000 [00:28<00:16, 8882.89it/s] 64%|██████▍   | 256517/400000 [00:29<00:16, 8918.42it/s] 64%|██████▍   | 257420/400000 [00:29<00:15, 8949.14it/s] 65%|██████▍   | 258316/400000 [00:29<00:15, 8908.75it/s] 65%|██████▍   | 259217/400000 [00:29<00:15, 8936.09it/s] 65%|██████▌   | 260111/400000 [00:29<00:15, 8898.95it/s] 65%|██████▌   | 261002/400000 [00:29<00:15, 8886.36it/s] 65%|██████▌   | 261898/400000 [00:29<00:15, 8907.19it/s] 66%|██████▌   | 262794/400000 [00:29<00:15, 8922.32it/s] 66%|██████▌   | 263706/400000 [00:29<00:15, 8979.96it/s] 66%|██████▌   | 264607/400000 [00:29<00:15, 8987.20it/s] 66%|██████▋   | 265515/400000 [00:30<00:14, 9012.43it/s] 67%|██████▋   | 266417/400000 [00:30<00:15, 8895.52it/s] 67%|██████▋   | 267334/400000 [00:30<00:14, 8973.84it/s] 67%|██████▋   | 268236/400000 [00:30<00:14, 8986.17it/s] 67%|██████▋   | 269135/400000 [00:30<00:14, 8908.64it/s] 68%|██████▊   | 270027/400000 [00:30<00:14, 8791.49it/s] 68%|██████▊   | 270907/400000 [00:30<00:14, 8637.17it/s] 68%|██████▊   | 271772/400000 [00:30<00:14, 8631.27it/s] 68%|██████▊   | 272636/400000 [00:30<00:14, 8629.89it/s] 68%|██████▊   | 273529/400000 [00:30<00:14, 8715.24it/s] 69%|██████▊   | 274409/400000 [00:31<00:14, 8739.19it/s] 69%|██████▉   | 275284/400000 [00:31<00:14, 8723.85it/s] 69%|██████▉   | 276168/400000 [00:31<00:14, 8757.30it/s] 69%|██████▉   | 277044/400000 [00:31<00:14, 8724.73it/s] 69%|██████▉   | 277947/400000 [00:31<00:13, 8811.13it/s] 70%|██████▉   | 278829/400000 [00:31<00:13, 8740.63it/s] 70%|██████▉   | 279704/400000 [00:31<00:14, 8492.00it/s] 70%|███████   | 280582/400000 [00:31<00:13, 8573.53it/s] 70%|███████   | 281490/400000 [00:31<00:13, 8716.40it/s] 71%|███████   | 282384/400000 [00:31<00:13, 8780.44it/s] 71%|███████   | 283307/400000 [00:32<00:13, 8907.91it/s] 71%|███████   | 284200/400000 [00:32<00:13, 8853.45it/s] 71%|███████▏  | 285109/400000 [00:32<00:12, 8921.86it/s] 72%|███████▏  | 286033/400000 [00:32<00:12, 9014.36it/s] 72%|███████▏  | 286957/400000 [00:32<00:12, 9079.67it/s] 72%|███████▏  | 287866/400000 [00:32<00:12, 9080.05it/s] 72%|███████▏  | 288775/400000 [00:32<00:12, 9059.25it/s] 72%|███████▏  | 289695/400000 [00:32<00:12, 9100.01it/s] 73%|███████▎  | 290606/400000 [00:32<00:12, 9100.18it/s] 73%|███████▎  | 291526/400000 [00:32<00:11, 9128.25it/s] 73%|███████▎  | 292439/400000 [00:33<00:11, 9032.64it/s] 73%|███████▎  | 293343/400000 [00:33<00:11, 9031.00it/s] 74%|███████▎  | 294256/400000 [00:33<00:11, 9057.92it/s] 74%|███████▍  | 295162/400000 [00:33<00:11, 9009.76it/s] 74%|███████▍  | 296064/400000 [00:33<00:11, 8985.12it/s] 74%|███████▍  | 296963/400000 [00:33<00:11, 8941.03it/s] 74%|███████▍  | 297858/400000 [00:33<00:11, 8897.52it/s] 75%|███████▍  | 298759/400000 [00:33<00:11, 8928.27it/s] 75%|███████▍  | 299653/400000 [00:33<00:11, 8930.01it/s] 75%|███████▌  | 300548/400000 [00:33<00:11, 8935.13it/s] 75%|███████▌  | 301456/400000 [00:34<00:10, 8976.11it/s] 76%|███████▌  | 302354/400000 [00:34<00:10, 8958.70it/s] 76%|███████▌  | 303250/400000 [00:34<00:10, 8942.06it/s] 76%|███████▌  | 304145/400000 [00:34<00:10, 8885.57it/s] 76%|███████▋  | 305039/400000 [00:34<00:10, 8900.33it/s] 76%|███████▋  | 305935/400000 [00:34<00:10, 8917.70it/s] 77%|███████▋  | 306827/400000 [00:34<00:10, 8849.68it/s] 77%|███████▋  | 307713/400000 [00:34<00:10, 8718.52it/s] 77%|███████▋  | 308605/400000 [00:34<00:10, 8776.81it/s] 77%|███████▋  | 309484/400000 [00:35<00:10, 8708.11it/s] 78%|███████▊  | 310366/400000 [00:35<00:10, 8738.62it/s] 78%|███████▊  | 311268/400000 [00:35<00:10, 8818.72it/s] 78%|███████▊  | 312181/400000 [00:35<00:09, 8907.04it/s] 78%|███████▊  | 313073/400000 [00:35<00:09, 8717.62it/s] 78%|███████▊  | 313947/400000 [00:35<00:09, 8711.38it/s] 79%|███████▊  | 314820/400000 [00:35<00:09, 8579.25it/s] 79%|███████▉  | 315711/400000 [00:35<00:09, 8673.90it/s] 79%|███████▉  | 316580/400000 [00:35<00:09, 8594.57it/s] 79%|███████▉  | 317485/400000 [00:35<00:09, 8725.93it/s] 80%|███████▉  | 318389/400000 [00:36<00:09, 8815.97it/s] 80%|███████▉  | 319291/400000 [00:36<00:09, 8874.63it/s] 80%|████████  | 320192/400000 [00:36<00:08, 8912.92it/s] 80%|████████  | 321091/400000 [00:36<00:08, 8934.07it/s] 80%|████████  | 321985/400000 [00:36<00:09, 8641.98it/s] 81%|████████  | 322852/400000 [00:36<00:08, 8609.60it/s] 81%|████████  | 323715/400000 [00:36<00:08, 8610.76it/s] 81%|████████  | 324578/400000 [00:36<00:08, 8573.00it/s] 81%|████████▏ | 325465/400000 [00:36<00:08, 8658.75it/s] 82%|████████▏ | 326332/400000 [00:36<00:08, 8554.33it/s] 82%|████████▏ | 327241/400000 [00:37<00:08, 8706.75it/s] 82%|████████▏ | 328137/400000 [00:37<00:08, 8779.32it/s] 82%|████████▏ | 329016/400000 [00:37<00:08, 8707.17it/s] 82%|████████▏ | 329940/400000 [00:37<00:07, 8858.90it/s] 83%|████████▎ | 330839/400000 [00:37<00:07, 8896.34it/s] 83%|████████▎ | 331747/400000 [00:37<00:07, 8948.36it/s] 83%|████████▎ | 332671/400000 [00:37<00:07, 9032.82it/s] 83%|████████▎ | 333575/400000 [00:37<00:07, 9034.33it/s] 84%|████████▎ | 334480/400000 [00:37<00:07, 9037.69it/s] 84%|████████▍ | 335385/400000 [00:37<00:07, 9040.44it/s] 84%|████████▍ | 336297/400000 [00:38<00:07, 9063.34it/s] 84%|████████▍ | 337212/400000 [00:38<00:06, 9087.96it/s] 85%|████████▍ | 338121/400000 [00:38<00:06, 9071.08it/s] 85%|████████▍ | 339029/400000 [00:38<00:06, 8960.54it/s] 85%|████████▍ | 339926/400000 [00:38<00:06, 8936.79it/s] 85%|████████▌ | 340836/400000 [00:38<00:06, 8983.14it/s] 85%|████████▌ | 341743/400000 [00:38<00:06, 9007.60it/s] 86%|████████▌ | 342644/400000 [00:38<00:06, 8960.40it/s] 86%|████████▌ | 343541/400000 [00:38<00:06, 8861.65it/s] 86%|████████▌ | 344454/400000 [00:38<00:06, 8938.62it/s] 86%|████████▋ | 345368/400000 [00:39<00:06, 8996.62it/s] 87%|████████▋ | 346269/400000 [00:39<00:06, 8884.23it/s] 87%|████████▋ | 347159/400000 [00:39<00:05, 8824.16it/s] 87%|████████▋ | 348042/400000 [00:39<00:05, 8757.32it/s] 87%|████████▋ | 348940/400000 [00:39<00:05, 8821.51it/s] 87%|████████▋ | 349823/400000 [00:39<00:05, 8668.69it/s] 88%|████████▊ | 350719/400000 [00:39<00:05, 8753.38it/s] 88%|████████▊ | 351596/400000 [00:39<00:05, 8710.34it/s] 88%|████████▊ | 352487/400000 [00:39<00:05, 8767.74it/s] 88%|████████▊ | 353399/400000 [00:39<00:05, 8869.36it/s] 89%|████████▊ | 354287/400000 [00:40<00:05, 8614.42it/s] 89%|████████▉ | 355191/400000 [00:40<00:05, 8736.53it/s] 89%|████████▉ | 356067/400000 [00:40<00:05, 8673.77it/s] 89%|████████▉ | 356966/400000 [00:40<00:04, 8765.99it/s] 89%|████████▉ | 357846/400000 [00:40<00:04, 8774.53it/s] 90%|████████▉ | 358737/400000 [00:40<00:04, 8812.03it/s] 90%|████████▉ | 359619/400000 [00:40<00:04, 8793.21it/s] 90%|█████████ | 360499/400000 [00:40<00:04, 8582.14it/s] 90%|█████████ | 361388/400000 [00:40<00:04, 8670.55it/s] 91%|█████████ | 362300/400000 [00:40<00:04, 8798.66it/s] 91%|█████████ | 363213/400000 [00:41<00:04, 8894.02it/s] 91%|█████████ | 364113/400000 [00:41<00:04, 8924.67it/s] 91%|█████████▏| 365007/400000 [00:41<00:03, 8755.43it/s] 91%|█████████▏| 365922/400000 [00:41<00:03, 8868.32it/s] 92%|█████████▏| 366842/400000 [00:41<00:03, 8962.92it/s] 92%|█████████▏| 367750/400000 [00:41<00:03, 8995.11it/s] 92%|█████████▏| 368651/400000 [00:41<00:03, 8987.95it/s] 92%|█████████▏| 369551/400000 [00:41<00:03, 8870.52it/s] 93%|█████████▎| 370464/400000 [00:41<00:03, 8944.65it/s] 93%|█████████▎| 371378/400000 [00:42<00:03, 9001.34it/s] 93%|█████████▎| 372279/400000 [00:42<00:03, 8963.11it/s] 93%|█████████▎| 373176/400000 [00:42<00:03, 8854.93it/s] 94%|█████████▎| 374063/400000 [00:42<00:02, 8814.00it/s] 94%|█████████▎| 374945/400000 [00:42<00:02, 8793.67it/s] 94%|█████████▍| 375869/400000 [00:42<00:02, 8922.43it/s] 94%|█████████▍| 376796/400000 [00:42<00:02, 9022.39it/s] 94%|█████████▍| 377699/400000 [00:42<00:02, 8981.47it/s] 95%|█████████▍| 378599/400000 [00:42<00:02, 8984.01it/s] 95%|█████████▍| 379498/400000 [00:42<00:02, 8657.89it/s] 95%|█████████▌| 380417/400000 [00:43<00:02, 8808.32it/s] 95%|█████████▌| 381321/400000 [00:43<00:02, 8875.49it/s] 96%|█████████▌| 382215/400000 [00:43<00:01, 8894.55it/s] 96%|█████████▌| 383106/400000 [00:43<00:01, 8835.08it/s] 96%|█████████▌| 383995/400000 [00:43<00:01, 8850.93it/s] 96%|█████████▌| 384910/400000 [00:43<00:01, 8936.17it/s] 96%|█████████▋| 385822/400000 [00:43<00:01, 8990.08it/s] 97%|█████████▋| 386722/400000 [00:43<00:01, 8940.89it/s] 97%|█████████▋| 387617/400000 [00:43<00:01, 8905.84it/s] 97%|█████████▋| 388512/400000 [00:43<00:01, 8918.54it/s] 97%|█████████▋| 389405/400000 [00:44<00:01, 8846.46it/s] 98%|█████████▊| 390290/400000 [00:44<00:01, 8561.57it/s] 98%|█████████▊| 391184/400000 [00:44<00:01, 8671.46it/s] 98%|█████████▊| 392054/400000 [00:44<00:00, 8655.30it/s] 98%|█████████▊| 392954/400000 [00:44<00:00, 8755.48it/s] 98%|█████████▊| 393860/400000 [00:44<00:00, 8841.81it/s] 99%|█████████▊| 394746/400000 [00:44<00:00, 8753.14it/s] 99%|█████████▉| 395661/400000 [00:44<00:00, 8865.78it/s] 99%|█████████▉| 396549/400000 [00:44<00:00, 8784.01it/s] 99%|█████████▉| 397448/400000 [00:44<00:00, 8842.51it/s]100%|█████████▉| 398344/400000 [00:45<00:00, 8877.14it/s]100%|█████████▉| 399254/400000 [00:45<00:00, 8941.49it/s]100%|█████████▉| 399999/400000 [00:45<00:00, 8842.06it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f85a1bf2198>, <torchtext.data.dataset.TabularDataset object at 0x7f85a1bf22e8>, <torchtext.vocab.Vocab object at 0x7f85a1bf2208>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 13.64 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 25.49 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 13.51 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 13.51 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  7.02 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  7.02 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  7.60 file/s]2020-06-15 06:18:16.513905: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-06-15 06:18:16.518184: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-06-15 06:18:16.518903: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x563d76337070 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-15 06:18:16.518918: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 49152/9912422 [00:00<00:20, 481697.71it/s] 95%|█████████▍| 9412608/9912422 [00:00<00:00, 686602.97it/s]9920512it [00:00, 47961634.01it/s]                           
0it [00:00, ?it/s]32768it [00:00, 738093.71it/s]
0it [00:00, ?it/s]  6%|▋         | 106496/1648877 [00:00<00:01, 1015183.32it/s]1654784it [00:00, 12521589.82it/s]                           
0it [00:00, ?it/s]8192it [00:00, 217191.77it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
