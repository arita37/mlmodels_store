
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/561e81504931f72fe00de6bf5a2e274589f15d68', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '561e81504931f72fe00de6bf5a2e274589f15d68', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/561e81504931f72fe00de6bf5a2e274589f15d68

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/561e81504931f72fe00de6bf5a2e274589f15d68

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/561e81504931f72fe00de6bf5a2e274589f15d68

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7fb50a56b620> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7fb50a56b620> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7fb575b320d0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7fb575b320d0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7fb58f85fea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7fb58f85fea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fb5233aa950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fb5233aa950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fb5233aa950> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 32%|███▏      | 3153920/9912422 [00:00<00:00, 31207018.89it/s]9920512it [00:00, 34354773.81it/s]                             
0it [00:00, ?it/s]32768it [00:00, 423434.99it/s]
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]1654784it [00:00, 8027363.45it/s]          
0it [00:00, ?it/s]8192it [00:00, 113664.07it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fb50a445b00>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fb50a43bd68>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7fb5233aa598> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7fb5233aa598> 

  function with postional parmater data_info <function tf_dataset_download at 0x7fb5233aa598> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:20,  1.99 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:20,  1.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:20,  1.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   2%|▏         | 3/162 [00:00<01:00,  2.65 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:00,  2.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<00:59,  2.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   3%|▎         | 5/162 [00:00<00:45,  3.45 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<00:45,  3.45 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<00:45,  3.45 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▍         | 7/162 [00:01<00:35,  4.37 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:01<00:35,  4.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:01<00:35,  4.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   6%|▌         | 9/162 [00:01<00:27,  5.47 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:01<00:27,  5.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:01<00:27,  5.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   7%|▋         | 11/162 [00:01<00:23,  6.54 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:01<00:23,  6.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:01<00:22,  6.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   8%|▊         | 13/162 [00:01<00:19,  7.71 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:01<00:19,  7.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:01<00:19,  7.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   9%|▉         | 15/162 [00:01<00:16,  8.84 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:01<00:16,  8.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:01<00:16,  8.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  10%|█         | 17/162 [00:01<00:14,  9.91 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:01<00:14,  9.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:01<00:14,  9.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  12%|█▏        | 19/162 [00:01<00:13, 10.80 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:01<00:13, 10.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:02<00:13, 10.80 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  13%|█▎        | 21/162 [00:02<00:12, 11.45 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:02<00:12, 11.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:02<00:12, 11.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  14%|█▍        | 23/162 [00:02<00:11, 12.24 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:02<00:11, 12.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:02<00:11, 12.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  15%|█▌        | 25/162 [00:02<00:10, 12.69 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:02<00:10, 12.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:02<00:10, 12.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  17%|█▋        | 27/162 [00:02<00:10, 13.07 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:02<00:10, 13.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:02<00:10, 13.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  18%|█▊        | 29/162 [00:02<00:09, 13.36 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:02<00:09, 13.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:02<00:09, 13.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  19%|█▉        | 31/162 [00:02<00:09, 13.60 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:02<00:09, 13.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:02<00:09, 13.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  20%|██        | 33/162 [00:02<00:09, 14.26 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:02<00:09, 14.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:02<00:08, 14.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  22%|██▏       | 35/162 [00:03<00:08, 14.22 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:03<00:08, 14.22 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:03<00:08, 14.22 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  23%|██▎       | 37/162 [00:03<00:08, 14.30 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:03<00:08, 14.30 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:03<00:08, 14.30 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  24%|██▍       | 39/162 [00:03<00:08, 14.74 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:03<00:08, 14.74 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:03<00:08, 14.74 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  25%|██▌       | 41/162 [00:03<00:08, 14.72 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:03<00:08, 14.72 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:03<00:08, 14.72 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  27%|██▋       | 43/162 [00:03<00:07, 14.94 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:03<00:07, 14.94 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:03<00:07, 14.94 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  28%|██▊       | 45/162 [00:03<00:07, 14.91 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:03<00:07, 14.91 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:03<00:07, 14.91 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  29%|██▉       | 47/162 [00:03<00:07, 14.97 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:03<00:07, 14.97 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:03<00:07, 14.97 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  30%|███       | 49/162 [00:03<00:07, 14.90 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:03<00:07, 14.90 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:04<00:07, 14.90 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  31%|███▏      | 51/162 [00:04<00:07, 14.67 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:04<00:07, 14.67 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:04<00:07, 14.67 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  33%|███▎      | 53/162 [00:04<00:08, 13.30 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:04<00:08, 13.30 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:04<00:08, 13.30 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  34%|███▍      | 55/162 [00:04<00:08, 12.57 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:04<00:08, 12.57 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:04<00:08, 12.57 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  35%|███▌      | 57/162 [00:04<00:08, 12.21 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:04<00:08, 12.21 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:04<00:08, 12.21 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  36%|███▋      | 59/162 [00:04<00:08, 12.08 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:04<00:08, 12.08 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:04<00:08, 12.08 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  38%|███▊      | 61/162 [00:04<00:08, 12.10 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:04<00:08, 12.10 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:05<00:08, 12.10 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  39%|███▉      | 63/162 [00:05<00:08, 12.06 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:05<00:08, 12.06 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:05<00:08, 12.06 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  40%|████      | 65/162 [00:05<00:08, 12.07 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:05<00:08, 12.07 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:05<00:07, 12.07 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  41%|████▏     | 67/162 [00:05<00:07, 12.11 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:05<00:07, 12.11 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:05<00:07, 12.11 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  43%|████▎     | 69/162 [00:05<00:07, 12.24 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:05<00:07, 12.24 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:05<00:07, 12.24 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  44%|████▍     | 71/162 [00:05<00:07, 12.31 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:05<00:07, 12.31 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:05<00:07, 12.31 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  45%|████▌     | 73/162 [00:05<00:07, 12.35 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:05<00:07, 12.35 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:06<00:07, 12.35 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  46%|████▋     | 75/162 [00:06<00:06, 12.55 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:06<00:06, 12.55 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:06<00:06, 12.55 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  48%|████▊     | 77/162 [00:06<00:06, 12.26 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:06<00:06, 12.26 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:06<00:06, 12.26 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  49%|████▉     | 79/162 [00:06<00:06, 12.41 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:06<00:06, 12.41 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:06<00:06, 12.41 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  50%|█████     | 81/162 [00:06<00:06, 12.61 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:06<00:06, 12.61 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:06<00:06, 12.61 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  51%|█████     | 83/162 [00:06<00:06, 12.57 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:06<00:06, 12.57 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:06<00:06, 12.57 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  52%|█████▏    | 85/162 [00:06<00:06, 12.31 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:06<00:06, 12.31 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:07<00:06, 12.31 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[A
Dl Size...:  54%|█████▎    | 87/162 [00:07<00:06, 11.19 MiB/s][ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:07<00:06, 11.19 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:07<00:06, 11.19 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[A
Dl Size...:  55%|█████▍    | 89/162 [00:07<00:06, 10.58 MiB/s][ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:07<00:06, 10.58 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:07<00:06, 10.58 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[A
Dl Size...:  56%|█████▌    | 91/162 [00:07<00:06, 10.32 MiB/s][ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:07<00:06, 10.32 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:07<00:06, 10.32 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[A
Dl Size...:  57%|█████▋    | 93/162 [00:07<00:06, 10.37 MiB/s][ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:07<00:06, 10.37 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:07<00:06, 10.37 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[A
Dl Size...:  59%|█████▊    | 95/162 [00:07<00:06, 10.30 MiB/s][ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:07<00:06, 10.30 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:08<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:08<00:06, 10.30 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[A
Dl Size...:  60%|█████▉    | 97/162 [00:08<00:06, 10.25 MiB/s][ADl Completed...:   0%|          | 0/1 [00:08<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:08<00:06, 10.25 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:08<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:08<00:06, 10.25 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[A
Dl Size...:  61%|██████    | 99/162 [00:08<00:06, 10.24 MiB/s][ADl Completed...:   0%|          | 0/1 [00:08<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:08<00:06, 10.24 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:08<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:08<00:06, 10.24 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[A
Dl Size...:  62%|██████▏   | 101/162 [00:08<00:05, 10.49 MiB/s][ADl Completed...:   0%|          | 0/1 [00:08<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:08<00:05, 10.49 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:08<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:08<00:05, 10.49 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[A
Dl Size...:  64%|██████▎   | 103/162 [00:08<00:05, 10.38 MiB/s][ADl Completed...:   0%|          | 0/1 [00:08<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:08<00:05, 10.38 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:08<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:08<00:05, 10.38 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[A
Dl Size...:  65%|██████▍   | 105/162 [00:08<00:05, 10.53 MiB/s][ADl Completed...:   0%|          | 0/1 [00:08<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:08<00:05, 10.53 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:09<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:08<00:05, 10.53 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[A
Dl Size...:  66%|██████▌   | 107/162 [00:09<00:05, 10.69 MiB/s][ADl Completed...:   0%|          | 0/1 [00:09<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:09<00:05, 10.69 MiB/s][A

Extraction completed...: 0 file [00:09, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:09<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:09<00:05, 10.69 MiB/s][A

Extraction completed...: 0 file [00:09, ? file/s][A[A
Dl Size...:  67%|██████▋   | 109/162 [00:09<00:04, 10.69 MiB/s][ADl Completed...:   0%|          | 0/1 [00:09<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:09<00:04, 10.69 MiB/s][A

Extraction completed...: 0 file [00:09, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:09<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:09<00:04, 10.69 MiB/s][A

Extraction completed...: 0 file [00:09, ? file/s][A[A
Dl Size...:  69%|██████▊   | 111/162 [00:09<00:04, 10.89 MiB/s][ADl Completed...:   0%|          | 0/1 [00:09<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:09<00:04, 10.89 MiB/s][A

Extraction completed...: 0 file [00:09, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:09<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:09<00:04, 10.89 MiB/s][A

Extraction completed...: 0 file [00:09, ? file/s][A[A
Dl Size...:  70%|██████▉   | 113/162 [00:09<00:04, 11.00 MiB/s][ADl Completed...:   0%|          | 0/1 [00:09<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:09<00:04, 11.00 MiB/s][A

Extraction completed...: 0 file [00:09, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:09<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:09<00:04, 11.00 MiB/s][A

Extraction completed...: 0 file [00:09, ? file/s][A[A
Dl Size...:  71%|███████   | 115/162 [00:09<00:04, 11.14 MiB/s][ADl Completed...:   0%|          | 0/1 [00:09<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:09<00:04, 11.14 MiB/s][A

Extraction completed...: 0 file [00:09, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:09<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:09<00:04, 11.14 MiB/s][A

Extraction completed...: 0 file [00:09, ? file/s][A[A
Dl Size...:  72%|███████▏  | 117/162 [00:09<00:04, 11.25 MiB/s][ADl Completed...:   0%|          | 0/1 [00:09<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:09<00:04, 11.25 MiB/s][A

Extraction completed...: 0 file [00:09, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:10<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:10<00:03, 11.25 MiB/s][A

Extraction completed...: 0 file [00:10, ? file/s][A[A
Dl Size...:  73%|███████▎  | 119/162 [00:10<00:03, 11.46 MiB/s][ADl Completed...:   0%|          | 0/1 [00:10<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:10<00:03, 11.46 MiB/s][A

Extraction completed...: 0 file [00:10, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:10<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:10<00:03, 11.46 MiB/s][A

Extraction completed...: 0 file [00:10, ? file/s][A[A
Dl Size...:  75%|███████▍  | 121/162 [00:10<00:03, 11.54 MiB/s][ADl Completed...:   0%|          | 0/1 [00:10<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:10<00:03, 11.54 MiB/s][A

Extraction completed...: 0 file [00:10, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:10<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:10<00:03, 11.54 MiB/s][A

Extraction completed...: 0 file [00:10, ? file/s][A[A
Dl Size...:  76%|███████▌  | 123/162 [00:10<00:03, 11.77 MiB/s][ADl Completed...:   0%|          | 0/1 [00:10<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:10<00:03, 11.77 MiB/s][A

Extraction completed...: 0 file [00:10, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:10<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:10<00:03, 11.77 MiB/s][A

Extraction completed...: 0 file [00:10, ? file/s][A[A
Dl Size...:  77%|███████▋  | 125/162 [00:10<00:03, 11.81 MiB/s][ADl Completed...:   0%|          | 0/1 [00:10<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:10<00:03, 11.81 MiB/s][A

Extraction completed...: 0 file [00:10, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:10<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:10<00:03, 11.81 MiB/s][A

Extraction completed...: 0 file [00:10, ? file/s][A[A
Dl Size...:  78%|███████▊  | 127/162 [00:10<00:02, 11.84 MiB/s][ADl Completed...:   0%|          | 0/1 [00:10<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:10<00:02, 11.84 MiB/s][A

Extraction completed...: 0 file [00:10, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:10<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:10<00:02, 11.84 MiB/s][A

Extraction completed...: 0 file [00:10, ? file/s][A[A
Dl Size...:  80%|███████▉  | 129/162 [00:10<00:02, 12.13 MiB/s][ADl Completed...:   0%|          | 0/1 [00:10<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:10<00:02, 12.13 MiB/s][A

Extraction completed...: 0 file [00:10, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:11<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:11<00:02, 12.13 MiB/s][A

Extraction completed...: 0 file [00:11, ? file/s][A[A
Dl Size...:  81%|████████  | 131/162 [00:11<00:02, 12.11 MiB/s][ADl Completed...:   0%|          | 0/1 [00:11<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:11<00:02, 12.11 MiB/s][A

Extraction completed...: 0 file [00:11, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:11<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:11<00:02, 12.11 MiB/s][A

Extraction completed...: 0 file [00:11, ? file/s][A[A
Dl Size...:  82%|████████▏ | 133/162 [00:11<00:02, 12.46 MiB/s][ADl Completed...:   0%|          | 0/1 [00:11<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:11<00:02, 12.46 MiB/s][A

Extraction completed...: 0 file [00:11, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:11<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:11<00:02, 12.46 MiB/s][A

Extraction completed...: 0 file [00:11, ? file/s][A[A
Dl Size...:  83%|████████▎ | 135/162 [00:11<00:02, 12.46 MiB/s][ADl Completed...:   0%|          | 0/1 [00:11<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:11<00:02, 12.46 MiB/s][A

Extraction completed...: 0 file [00:11, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:11<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:11<00:02, 12.46 MiB/s][A

Extraction completed...: 0 file [00:11, ? file/s][A[A
Dl Size...:  85%|████████▍ | 137/162 [00:11<00:01, 12.54 MiB/s][ADl Completed...:   0%|          | 0/1 [00:11<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:11<00:01, 12.54 MiB/s][A

Extraction completed...: 0 file [00:11, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:11<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:11<00:01, 12.54 MiB/s][A

Extraction completed...: 0 file [00:11, ? file/s][A[A
Dl Size...:  86%|████████▌ | 139/162 [00:11<00:01, 12.80 MiB/s][ADl Completed...:   0%|          | 0/1 [00:11<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:11<00:01, 12.80 MiB/s][A

Extraction completed...: 0 file [00:11, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:11<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:11<00:01, 12.80 MiB/s][A

Extraction completed...: 0 file [00:11, ? file/s][A[A
Dl Size...:  87%|████████▋ | 141/162 [00:11<00:01, 13.00 MiB/s][ADl Completed...:   0%|          | 0/1 [00:11<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:11<00:01, 13.00 MiB/s][A

Extraction completed...: 0 file [00:11, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:11<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:11<00:01, 13.00 MiB/s][A

Extraction completed...: 0 file [00:11, ? file/s][A[A
Dl Size...:  88%|████████▊ | 143/162 [00:12<00:01, 13.07 MiB/s][ADl Completed...:   0%|          | 0/1 [00:12<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:12<00:01, 13.07 MiB/s][A

Extraction completed...: 0 file [00:12, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:12<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:12<00:01, 13.07 MiB/s][A

Extraction completed...: 0 file [00:12, ? file/s][A[A
Dl Size...:  90%|████████▉ | 145/162 [00:12<00:01, 13.02 MiB/s][ADl Completed...:   0%|          | 0/1 [00:12<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:12<00:01, 13.02 MiB/s][A

Extraction completed...: 0 file [00:12, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:12<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:12<00:01, 13.02 MiB/s][A

Extraction completed...: 0 file [00:12, ? file/s][A[A
Dl Size...:  91%|█████████ | 147/162 [00:12<00:01, 13.26 MiB/s][ADl Completed...:   0%|          | 0/1 [00:12<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:12<00:01, 13.26 MiB/s][A

Extraction completed...: 0 file [00:12, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:12<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:12<00:01, 13.26 MiB/s][A

Extraction completed...: 0 file [00:12, ? file/s][A[A
Dl Size...:  92%|█████████▏| 149/162 [00:12<00:00, 13.38 MiB/s][ADl Completed...:   0%|          | 0/1 [00:12<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:12<00:00, 13.38 MiB/s][A

Extraction completed...: 0 file [00:12, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:12<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:12<00:00, 13.38 MiB/s][A

Extraction completed...: 0 file [00:12, ? file/s][A[A
Dl Size...:  93%|█████████▎| 151/162 [00:12<00:00, 13.49 MiB/s][ADl Completed...:   0%|          | 0/1 [00:12<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:12<00:00, 13.49 MiB/s][A

Extraction completed...: 0 file [00:12, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:12<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:12<00:00, 13.49 MiB/s][A

Extraction completed...: 0 file [00:12, ? file/s][A[A
Dl Size...:  94%|█████████▍| 153/162 [00:12<00:00, 13.64 MiB/s][ADl Completed...:   0%|          | 0/1 [00:12<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:12<00:00, 13.64 MiB/s][A

Extraction completed...: 0 file [00:12, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:12<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:12<00:00, 13.64 MiB/s][A

Extraction completed...: 0 file [00:12, ? file/s][A[A
Dl Size...:  96%|█████████▌| 155/162 [00:12<00:00, 13.78 MiB/s][ADl Completed...:   0%|          | 0/1 [00:12<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:12<00:00, 13.78 MiB/s][A

Extraction completed...: 0 file [00:12, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:12<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:12<00:00, 13.78 MiB/s][A

Extraction completed...: 0 file [00:12, ? file/s][A[A
Dl Size...:  97%|█████████▋| 157/162 [00:13<00:00, 13.85 MiB/s][ADl Completed...:   0%|          | 0/1 [00:13<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:13<00:00, 13.85 MiB/s][A

Extraction completed...: 0 file [00:13, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:13<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:13<00:00, 13.85 MiB/s][A

Extraction completed...: 0 file [00:13, ? file/s][A[A
Dl Size...:  98%|█████████▊| 159/162 [00:13<00:00, 14.13 MiB/s][ADl Completed...:   0%|          | 0/1 [00:13<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:13<00:00, 14.13 MiB/s][A

Extraction completed...: 0 file [00:13, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:13<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:13<00:00, 14.13 MiB/s][A

Extraction completed...: 0 file [00:13, ? file/s][A[A
Dl Size...:  99%|█████████▉| 161/162 [00:13<00:00, 14.35 MiB/s][ADl Completed...:   0%|          | 0/1 [00:13<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:13<00:00, 14.35 MiB/s][A

Extraction completed...: 0 file [00:13, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:13<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:13<00:00, 14.35 MiB/s][A

Extraction completed...: 0 file [00:13, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:13<00:00, 13.43s/ url]Dl Completed...: 100%|██████████| 1/1 [00:13<00:00, 13.43s/ url]
Dl Size...: 100%|██████████| 162/162 [00:13<00:00, 14.35 MiB/s][A

Extraction completed...: 0 file [00:13, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:13<00:00, 13.43s/ url]
Dl Size...: 100%|██████████| 162/162 [00:13<00:00, 14.35 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:13<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:15<00:00, 15.87s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:15<00:00, 13.43s/ url]
Dl Size...: 100%|██████████| 162/162 [00:15<00:00, 14.35 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:15<00:00, 15.87s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:15<00:00, 15.87s/ file]
Dl Size...: 100%|██████████| 162/162 [00:15<00:00, 10.21 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:15<00:00, 15.87s/ url]
0 examples [00:00, ? examples/s]2020-06-07 12:10:11.110114: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-07 12:10:11.123239: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-06-07 12:10:11.123492: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x563365e0e660 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-07 12:10:11.123514: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
16 examples [00:00, 159.34 examples/s]103 examples [00:00, 211.06 examples/s]187 examples [00:00, 272.09 examples/s]267 examples [00:00, 338.98 examples/s]354 examples [00:00, 414.72 examples/s]440 examples [00:00, 490.85 examples/s]526 examples [00:00, 562.43 examples/s]608 examples [00:00, 619.32 examples/s]694 examples [00:00, 674.46 examples/s]777 examples [00:01, 713.99 examples/s]861 examples [00:01, 746.82 examples/s]948 examples [00:01, 779.92 examples/s]1035 examples [00:01, 802.87 examples/s]1122 examples [00:01, 820.18 examples/s]1207 examples [00:01, 801.59 examples/s]1290 examples [00:01, 807.42 examples/s]1373 examples [00:01, 804.19 examples/s]1460 examples [00:01, 820.49 examples/s]1549 examples [00:01, 838.67 examples/s]1635 examples [00:02, 843.06 examples/s]1720 examples [00:02, 836.65 examples/s]1809 examples [00:02, 851.66 examples/s]1895 examples [00:02, 828.74 examples/s]1985 examples [00:02, 846.42 examples/s]2070 examples [00:02, 832.33 examples/s]2155 examples [00:02, 835.05 examples/s]2240 examples [00:02, 839.04 examples/s]2325 examples [00:02, 830.46 examples/s]2412 examples [00:02, 841.50 examples/s]2497 examples [00:03, 840.13 examples/s]2583 examples [00:03, 844.26 examples/s]2671 examples [00:03, 852.67 examples/s]2760 examples [00:03, 862.10 examples/s]2847 examples [00:03, 847.92 examples/s]2941 examples [00:03, 871.14 examples/s]3035 examples [00:03, 888.04 examples/s]3125 examples [00:03, 878.61 examples/s]3217 examples [00:03, 890.37 examples/s]3307 examples [00:03, 843.76 examples/s]3393 examples [00:04, 847.35 examples/s]3479 examples [00:04, 847.17 examples/s]3565 examples [00:04, 814.28 examples/s]3652 examples [00:04, 828.65 examples/s]3738 examples [00:04, 833.91 examples/s]3822 examples [00:04, 805.86 examples/s]3910 examples [00:04, 826.33 examples/s]3994 examples [00:04, 825.83 examples/s]4079 examples [00:04, 831.79 examples/s]4163 examples [00:05, 800.56 examples/s]4248 examples [00:05, 812.62 examples/s]4330 examples [00:05, 800.49 examples/s]4419 examples [00:05, 823.65 examples/s]4510 examples [00:05, 846.16 examples/s]4601 examples [00:05, 862.33 examples/s]4693 examples [00:05, 877.77 examples/s]4782 examples [00:05, 879.46 examples/s]4874 examples [00:05, 890.03 examples/s]4964 examples [00:05, 866.32 examples/s]5051 examples [00:06, 866.97 examples/s]5141 examples [00:06, 875.37 examples/s]5231 examples [00:06, 882.48 examples/s]5321 examples [00:06, 885.06 examples/s]5410 examples [00:06, 881.42 examples/s]5501 examples [00:06, 889.47 examples/s]5594 examples [00:06, 898.69 examples/s]5688 examples [00:06, 909.06 examples/s]5781 examples [00:06, 913.14 examples/s]5873 examples [00:06, 912.15 examples/s]5965 examples [00:07, 904.20 examples/s]6057 examples [00:07, 908.58 examples/s]6150 examples [00:07, 913.00 examples/s]6242 examples [00:07, 911.25 examples/s]6338 examples [00:07, 924.55 examples/s]6434 examples [00:07, 934.51 examples/s]6528 examples [00:07, 899.57 examples/s]6619 examples [00:07, 897.03 examples/s]6709 examples [00:07, 874.66 examples/s]6797 examples [00:08, 872.36 examples/s]6886 examples [00:08, 877.09 examples/s]6976 examples [00:08, 883.32 examples/s]7066 examples [00:08, 885.80 examples/s]7156 examples [00:08, 887.54 examples/s]7250 examples [00:08, 902.54 examples/s]7343 examples [00:08, 909.81 examples/s]7438 examples [00:08, 919.70 examples/s]7531 examples [00:08, 922.18 examples/s]7624 examples [00:08, 923.29 examples/s]7717 examples [00:09, 916.69 examples/s]7810 examples [00:09, 918.56 examples/s]7903 examples [00:09, 920.75 examples/s]7996 examples [00:09, 919.13 examples/s]8088 examples [00:09, 912.42 examples/s]8180 examples [00:09, 878.28 examples/s]8269 examples [00:09, 867.59 examples/s]8357 examples [00:09, 869.08 examples/s]8445 examples [00:09, 860.23 examples/s]8532 examples [00:09, 858.93 examples/s]8618 examples [00:10, 848.00 examples/s]8704 examples [00:10, 851.16 examples/s]8790 examples [00:10, 849.71 examples/s]8876 examples [00:10, 851.06 examples/s]8965 examples [00:10, 861.18 examples/s]9052 examples [00:10, 853.18 examples/s]9139 examples [00:10, 858.02 examples/s]9225 examples [00:10, 855.45 examples/s]9315 examples [00:10, 865.96 examples/s]9402 examples [00:10, 864.07 examples/s]9491 examples [00:11, 870.81 examples/s]9584 examples [00:11, 885.68 examples/s]9678 examples [00:11, 900.82 examples/s]9772 examples [00:11, 911.70 examples/s]9865 examples [00:11, 916.10 examples/s]9957 examples [00:11, 911.06 examples/s]10049 examples [00:11, 849.83 examples/s]10142 examples [00:11, 870.77 examples/s]10233 examples [00:11, 882.00 examples/s]10327 examples [00:11, 897.00 examples/s]10419 examples [00:12, 901.57 examples/s]10512 examples [00:12, 909.51 examples/s]10606 examples [00:12, 915.86 examples/s]10699 examples [00:12, 918.28 examples/s]10791 examples [00:12, 916.47 examples/s]10883 examples [00:12, 904.43 examples/s]10974 examples [00:12, 889.92 examples/s]11065 examples [00:12, 894.51 examples/s]11159 examples [00:12, 906.74 examples/s]11250 examples [00:13, 891.91 examples/s]11340 examples [00:13, 849.18 examples/s]11432 examples [00:13, 867.47 examples/s]11522 examples [00:13, 876.42 examples/s]11610 examples [00:13, 871.00 examples/s]11698 examples [00:13, 870.53 examples/s]11789 examples [00:13, 881.12 examples/s]11878 examples [00:13, 881.86 examples/s]11967 examples [00:13, 855.44 examples/s]12055 examples [00:13, 862.20 examples/s]12142 examples [00:14, 863.48 examples/s]12229 examples [00:14, 825.98 examples/s]12313 examples [00:14, 778.78 examples/s]12392 examples [00:14, 742.96 examples/s]12481 examples [00:14, 781.12 examples/s]12568 examples [00:14, 805.39 examples/s]12664 examples [00:14, 844.96 examples/s]12758 examples [00:14, 869.21 examples/s]12852 examples [00:14, 889.25 examples/s]12943 examples [00:15, 893.61 examples/s]13035 examples [00:15, 898.97 examples/s]13126 examples [00:15, 901.21 examples/s]13220 examples [00:15, 911.40 examples/s]13314 examples [00:15, 918.23 examples/s]13407 examples [00:15, 921.34 examples/s]13500 examples [00:15, 920.28 examples/s]13593 examples [00:15, 921.28 examples/s]13686 examples [00:15, 911.52 examples/s]13778 examples [00:15, 903.87 examples/s]13869 examples [00:16, 873.66 examples/s]13957 examples [00:16, 856.85 examples/s]14043 examples [00:16, 807.80 examples/s]14131 examples [00:16, 825.83 examples/s]14220 examples [00:16, 842.17 examples/s]14306 examples [00:16, 846.52 examples/s]14392 examples [00:16, 847.99 examples/s]14478 examples [00:16, 844.65 examples/s]14563 examples [00:16, 828.46 examples/s]14647 examples [00:16, 819.09 examples/s]14733 examples [00:17, 827.61 examples/s]14816 examples [00:17, 825.69 examples/s]14901 examples [00:17, 829.81 examples/s]14988 examples [00:17, 840.21 examples/s]15076 examples [00:17, 850.93 examples/s]15166 examples [00:17, 863.86 examples/s]15253 examples [00:17, 860.72 examples/s]15340 examples [00:17, 829.33 examples/s]15426 examples [00:17, 836.94 examples/s]15510 examples [00:18, 814.45 examples/s]15597 examples [00:18, 829.50 examples/s]15683 examples [00:18, 836.01 examples/s]15767 examples [00:18, 821.06 examples/s]15850 examples [00:18, 812.73 examples/s]15941 examples [00:18, 837.67 examples/s]16035 examples [00:18, 865.36 examples/s]16128 examples [00:18, 882.84 examples/s]16219 examples [00:18, 888.40 examples/s]16309 examples [00:18, 890.16 examples/s]16399 examples [00:19, 873.79 examples/s]16487 examples [00:19, 862.43 examples/s]16574 examples [00:19, 849.22 examples/s]16660 examples [00:19, 838.33 examples/s]16748 examples [00:19, 848.53 examples/s]16833 examples [00:19, 846.96 examples/s]16918 examples [00:19, 838.05 examples/s]17009 examples [00:19, 857.30 examples/s]17100 examples [00:19, 870.61 examples/s]17188 examples [00:19, 852.78 examples/s]17275 examples [00:20, 856.18 examples/s]17363 examples [00:20, 860.11 examples/s]17452 examples [00:20, 868.64 examples/s]17542 examples [00:20, 876.73 examples/s]17630 examples [00:20, 868.53 examples/s]17717 examples [00:20, 861.45 examples/s]17809 examples [00:20, 875.98 examples/s]17897 examples [00:20, 873.24 examples/s]17985 examples [00:20, 863.28 examples/s]18073 examples [00:20, 868.07 examples/s]18161 examples [00:21, 869.57 examples/s]18249 examples [00:21, 870.48 examples/s]18337 examples [00:21, 869.59 examples/s]18424 examples [00:21, 857.73 examples/s]18512 examples [00:21, 861.20 examples/s]18601 examples [00:21, 867.48 examples/s]18689 examples [00:21, 869.57 examples/s]18776 examples [00:21, 862.99 examples/s]18863 examples [00:21, 864.11 examples/s]18950 examples [00:21, 852.00 examples/s]19036 examples [00:22, 852.48 examples/s]19122 examples [00:22, 851.96 examples/s]19208 examples [00:22, 842.04 examples/s]19295 examples [00:22, 847.58 examples/s]19385 examples [00:22, 860.29 examples/s]19472 examples [00:22, 860.81 examples/s]19561 examples [00:22, 867.80 examples/s]19648 examples [00:22, 851.20 examples/s]19735 examples [00:22, 856.10 examples/s]19821 examples [00:23, 854.24 examples/s]19908 examples [00:23, 856.62 examples/s]19998 examples [00:23, 868.86 examples/s]20085 examples [00:23, 827.18 examples/s]20179 examples [00:23, 851.28 examples/s]20273 examples [00:23, 875.63 examples/s]20362 examples [00:23, 876.76 examples/s]20456 examples [00:23, 893.04 examples/s]20548 examples [00:23, 899.85 examples/s]20642 examples [00:23, 909.83 examples/s]20735 examples [00:24, 914.44 examples/s]20827 examples [00:24, 915.68 examples/s]20921 examples [00:24, 921.79 examples/s]21014 examples [00:24, 922.18 examples/s]21107 examples [00:24, 909.52 examples/s]21201 examples [00:24, 915.50 examples/s]21293 examples [00:24, 911.18 examples/s]21387 examples [00:24, 916.88 examples/s]21479 examples [00:24, 911.58 examples/s]21571 examples [00:24, 901.40 examples/s]21663 examples [00:25, 905.01 examples/s]21755 examples [00:25, 907.14 examples/s]21846 examples [00:25, 905.52 examples/s]21938 examples [00:25, 907.47 examples/s]22030 examples [00:25, 908.63 examples/s]22122 examples [00:25, 909.37 examples/s]22214 examples [00:25, 910.46 examples/s]22306 examples [00:25, 912.93 examples/s]22398 examples [00:25, 896.54 examples/s]22488 examples [00:25, 891.04 examples/s]22578 examples [00:26, 875.34 examples/s]22666 examples [00:26, 858.89 examples/s]22757 examples [00:26, 871.43 examples/s]22848 examples [00:26, 880.52 examples/s]22938 examples [00:26, 885.59 examples/s]23030 examples [00:26, 892.76 examples/s]23120 examples [00:26, 894.36 examples/s]23213 examples [00:26, 902.82 examples/s]23305 examples [00:26, 906.12 examples/s]23398 examples [00:26, 912.98 examples/s]23493 examples [00:27, 922.63 examples/s]23586 examples [00:27, 922.30 examples/s]23679 examples [00:27, 921.95 examples/s]23772 examples [00:27, 915.95 examples/s]23864 examples [00:27, 910.34 examples/s]23956 examples [00:27, 906.70 examples/s]24047 examples [00:27, 897.28 examples/s]24137 examples [00:27, 897.21 examples/s]24227 examples [00:27, 896.80 examples/s]24320 examples [00:28, 906.20 examples/s]24413 examples [00:28, 911.08 examples/s]24505 examples [00:28, 908.99 examples/s]24596 examples [00:28, 889.70 examples/s]24687 examples [00:28, 895.41 examples/s]24777 examples [00:28, 890.12 examples/s]24869 examples [00:28, 898.51 examples/s]24959 examples [00:28, 891.72 examples/s]25050 examples [00:28, 896.25 examples/s]25140 examples [00:28, 895.04 examples/s]25233 examples [00:29, 903.28 examples/s]25326 examples [00:29, 908.66 examples/s]25417 examples [00:29, 889.63 examples/s]25507 examples [00:29, 888.87 examples/s]25601 examples [00:29, 903.42 examples/s]25692 examples [00:29, 900.94 examples/s]25783 examples [00:29, 901.56 examples/s]25874 examples [00:29, 879.52 examples/s]25963 examples [00:29, 878.57 examples/s]26053 examples [00:29, 882.37 examples/s]26142 examples [00:30, 866.81 examples/s]26236 examples [00:30, 885.25 examples/s]26327 examples [00:30, 892.49 examples/s]26418 examples [00:30, 896.70 examples/s]26508 examples [00:30, 895.04 examples/s]26598 examples [00:30, 896.28 examples/s]26688 examples [00:30, 894.95 examples/s]26778 examples [00:30, 891.59 examples/s]26872 examples [00:30, 903.17 examples/s]26963 examples [00:30, 896.85 examples/s]27053 examples [00:31, 897.27 examples/s]27143 examples [00:31, 885.34 examples/s]27232 examples [00:31, 872.39 examples/s]27325 examples [00:31, 888.19 examples/s]27419 examples [00:31, 901.30 examples/s]27512 examples [00:31, 908.00 examples/s]27604 examples [00:31, 909.22 examples/s]27696 examples [00:31, 912.14 examples/s]27788 examples [00:31, 914.37 examples/s]27880 examples [00:31, 886.16 examples/s]27972 examples [00:32, 894.34 examples/s]28064 examples [00:32, 900.10 examples/s]28155 examples [00:32, 875.49 examples/s]28248 examples [00:32, 890.68 examples/s]28339 examples [00:32, 894.59 examples/s]28429 examples [00:32, 887.79 examples/s]28519 examples [00:32, 889.08 examples/s]28610 examples [00:32, 894.57 examples/s]28700 examples [00:32, 892.13 examples/s]28790 examples [00:33, 887.75 examples/s]28880 examples [00:33, 889.59 examples/s]28972 examples [00:33, 897.70 examples/s]29065 examples [00:33, 906.54 examples/s]29156 examples [00:33, 889.70 examples/s]29249 examples [00:33, 899.11 examples/s]29344 examples [00:33, 913.22 examples/s]29436 examples [00:33, 906.80 examples/s]29532 examples [00:33, 919.86 examples/s]29625 examples [00:33, 921.23 examples/s]29718 examples [00:34, 919.59 examples/s]29811 examples [00:34, 919.78 examples/s]29904 examples [00:34, 919.93 examples/s]29997 examples [00:34, 913.97 examples/s]30089 examples [00:34, 859.52 examples/s]30176 examples [00:34, 857.68 examples/s]30270 examples [00:34, 878.84 examples/s]30362 examples [00:34, 888.60 examples/s]30454 examples [00:34, 896.29 examples/s]30547 examples [00:34, 903.93 examples/s]30642 examples [00:35, 914.36 examples/s]30734 examples [00:35, 884.83 examples/s]30825 examples [00:35, 891.65 examples/s]30915 examples [00:35, 867.21 examples/s]31004 examples [00:35, 872.20 examples/s]31092 examples [00:35, 865.59 examples/s]31179 examples [00:35, 856.09 examples/s]31267 examples [00:35, 860.64 examples/s]31356 examples [00:35, 868.34 examples/s]31445 examples [00:35, 874.42 examples/s]31533 examples [00:36, 872.76 examples/s]31621 examples [00:36, 860.46 examples/s]31708 examples [00:36, 851.22 examples/s]31800 examples [00:36, 870.70 examples/s]31890 examples [00:36, 879.25 examples/s]31982 examples [00:36, 889.41 examples/s]32074 examples [00:36, 895.77 examples/s]32164 examples [00:36, 893.11 examples/s]32254 examples [00:36, 894.72 examples/s]32344 examples [00:37, 885.77 examples/s]32433 examples [00:37, 885.90 examples/s]32522 examples [00:37, 886.32 examples/s]32612 examples [00:37, 887.74 examples/s]32705 examples [00:37, 899.09 examples/s]32795 examples [00:37, 895.84 examples/s]32885 examples [00:37, 879.84 examples/s]32974 examples [00:37, 877.95 examples/s]33062 examples [00:37, 863.58 examples/s]33149 examples [00:37, 863.66 examples/s]33236 examples [00:38, 861.41 examples/s]33323 examples [00:38, 863.47 examples/s]33414 examples [00:38, 876.00 examples/s]33502 examples [00:38, 863.67 examples/s]33589 examples [00:38, 847.48 examples/s]33679 examples [00:38, 862.38 examples/s]33772 examples [00:38, 879.25 examples/s]33867 examples [00:38, 896.96 examples/s]33961 examples [00:38, 907.22 examples/s]34056 examples [00:38, 917.49 examples/s]34149 examples [00:39, 919.08 examples/s]34242 examples [00:39, 859.89 examples/s]34330 examples [00:39, 864.19 examples/s]34419 examples [00:39, 869.74 examples/s]34507 examples [00:39, 872.46 examples/s]34595 examples [00:39, 873.49 examples/s]34683 examples [00:39, 866.12 examples/s]34771 examples [00:39, 869.39 examples/s]34859 examples [00:39, 866.63 examples/s]34948 examples [00:39, 872.51 examples/s]35036 examples [00:40, 869.14 examples/s]35125 examples [00:40, 874.45 examples/s]35213 examples [00:40, 874.83 examples/s]35301 examples [00:40, 873.10 examples/s]35389 examples [00:40, 858.63 examples/s]35475 examples [00:40, 854.18 examples/s]35561 examples [00:40, 854.76 examples/s]35648 examples [00:40, 857.38 examples/s]35734 examples [00:40, 856.42 examples/s]35820 examples [00:40, 845.25 examples/s]35910 examples [00:41, 860.96 examples/s]35997 examples [00:41, 851.01 examples/s]36086 examples [00:41, 859.89 examples/s]36173 examples [00:41, 850.87 examples/s]36259 examples [00:41, 850.56 examples/s]36348 examples [00:41, 859.76 examples/s]36439 examples [00:41, 871.80 examples/s]36532 examples [00:41, 887.09 examples/s]36622 examples [00:41, 888.32 examples/s]36711 examples [00:42, 884.62 examples/s]36801 examples [00:42, 886.40 examples/s]36890 examples [00:42, 884.94 examples/s]36979 examples [00:42, 886.07 examples/s]37069 examples [00:42, 887.46 examples/s]37158 examples [00:42, 861.81 examples/s]37246 examples [00:42, 866.35 examples/s]37338 examples [00:42, 879.75 examples/s]37427 examples [00:42, 881.79 examples/s]37516 examples [00:42, 860.75 examples/s]37603 examples [00:43, 843.53 examples/s]37688 examples [00:43, 834.52 examples/s]37773 examples [00:43, 836.96 examples/s]37861 examples [00:43, 847.08 examples/s]37949 examples [00:43, 855.25 examples/s]38038 examples [00:43, 863.74 examples/s]38128 examples [00:43, 873.25 examples/s]38221 examples [00:43, 888.11 examples/s]38314 examples [00:43, 899.62 examples/s]38407 examples [00:43, 907.18 examples/s]38500 examples [00:44, 910.04 examples/s]38592 examples [00:44, 905.55 examples/s]38683 examples [00:44, 888.82 examples/s]38772 examples [00:44, 884.52 examples/s]38861 examples [00:44, 885.08 examples/s]38950 examples [00:44, 839.54 examples/s]39035 examples [00:44, 830.74 examples/s]39122 examples [00:44, 841.53 examples/s]39211 examples [00:44, 855.18 examples/s]39299 examples [00:44, 860.63 examples/s]39387 examples [00:45, 865.29 examples/s]39474 examples [00:45, 863.26 examples/s]39561 examples [00:45, 852.14 examples/s]39647 examples [00:45, 849.02 examples/s]39732 examples [00:45, 848.11 examples/s]39817 examples [00:45, 842.88 examples/s]39902 examples [00:45, 842.39 examples/s]39987 examples [00:45, 838.22 examples/s]40071 examples [00:45, 791.90 examples/s]40162 examples [00:46, 821.92 examples/s]40252 examples [00:46, 843.25 examples/s]40339 examples [00:46, 849.49 examples/s]40429 examples [00:46, 862.90 examples/s]40519 examples [00:46, 872.05 examples/s]40607 examples [00:46, 871.14 examples/s]40697 examples [00:46, 877.30 examples/s]40785 examples [00:46, 877.57 examples/s]40875 examples [00:46, 884.14 examples/s]40966 examples [00:46, 891.33 examples/s]41057 examples [00:47, 894.71 examples/s]41148 examples [00:47, 898.36 examples/s]41238 examples [00:47, 890.49 examples/s]41328 examples [00:47, 890.22 examples/s]41418 examples [00:47, 891.02 examples/s]41508 examples [00:47, 869.05 examples/s]41596 examples [00:47, 871.97 examples/s]41684 examples [00:47, 862.09 examples/s]41772 examples [00:47, 867.12 examples/s]41861 examples [00:47, 871.43 examples/s]41949 examples [00:48, 859.25 examples/s]42036 examples [00:48, 846.52 examples/s]42123 examples [00:48, 853.40 examples/s]42214 examples [00:48, 866.44 examples/s]42301 examples [00:48, 862.19 examples/s]42391 examples [00:48, 870.65 examples/s]42482 examples [00:48, 881.63 examples/s]42571 examples [00:48, 883.47 examples/s]42660 examples [00:48, 876.18 examples/s]42748 examples [00:48, 873.88 examples/s]42836 examples [00:49, 873.14 examples/s]42926 examples [00:49, 878.60 examples/s]43014 examples [00:49, 864.19 examples/s]43102 examples [00:49, 866.21 examples/s]43190 examples [00:49, 868.24 examples/s]43279 examples [00:49, 874.64 examples/s]43368 examples [00:49, 876.15 examples/s]43456 examples [00:49, 870.08 examples/s]43544 examples [00:49, 864.91 examples/s]43631 examples [00:49, 859.52 examples/s]43717 examples [00:50, 858.95 examples/s]43806 examples [00:50, 865.58 examples/s]43893 examples [00:50, 861.69 examples/s]43981 examples [00:50, 865.69 examples/s]44068 examples [00:50, 858.74 examples/s]44154 examples [00:50, 837.11 examples/s]44240 examples [00:50, 842.42 examples/s]44325 examples [00:50, 831.40 examples/s]44412 examples [00:50, 841.20 examples/s]44497 examples [00:51, 827.58 examples/s]44584 examples [00:51, 839.29 examples/s]44669 examples [00:51, 842.43 examples/s]44754 examples [00:51, 817.31 examples/s]44843 examples [00:51, 837.07 examples/s]44934 examples [00:51, 855.93 examples/s]45025 examples [00:51, 869.56 examples/s]45118 examples [00:51, 884.77 examples/s]45207 examples [00:51, 884.15 examples/s]45298 examples [00:51, 889.63 examples/s]45388 examples [00:52, 887.82 examples/s]45480 examples [00:52, 894.33 examples/s]45575 examples [00:52, 907.67 examples/s]45666 examples [00:52, 906.48 examples/s]45757 examples [00:52, 907.26 examples/s]45850 examples [00:52, 911.48 examples/s]45943 examples [00:52, 914.27 examples/s]46036 examples [00:52, 916.65 examples/s]46128 examples [00:52, 911.77 examples/s]46220 examples [00:52, 910.10 examples/s]46312 examples [00:53, 909.53 examples/s]46403 examples [00:53, 886.44 examples/s]46492 examples [00:53, 874.16 examples/s]46580 examples [00:53, 859.87 examples/s]46667 examples [00:53, 860.81 examples/s]46754 examples [00:53, 845.64 examples/s]46839 examples [00:53, 801.71 examples/s]46920 examples [00:53, 797.99 examples/s]47007 examples [00:53, 818.04 examples/s]47095 examples [00:54, 834.14 examples/s]47184 examples [00:54, 848.99 examples/s]47273 examples [00:54, 859.81 examples/s]47364 examples [00:54, 872.57 examples/s]47452 examples [00:54, 873.94 examples/s]47543 examples [00:54, 883.98 examples/s]47632 examples [00:54, 870.17 examples/s]47720 examples [00:54, 865.59 examples/s]47807 examples [00:54, 841.47 examples/s]47897 examples [00:54, 857.83 examples/s]47986 examples [00:55, 865.30 examples/s]48073 examples [00:55, 866.41 examples/s]48160 examples [00:55, 859.01 examples/s]48247 examples [00:55, 855.24 examples/s]48333 examples [00:55, 837.06 examples/s]48417 examples [00:55, 836.13 examples/s]48502 examples [00:55, 838.93 examples/s]48588 examples [00:55, 842.30 examples/s]48674 examples [00:55, 844.62 examples/s]48759 examples [00:55, 831.56 examples/s]48850 examples [00:56, 853.18 examples/s]48941 examples [00:56, 869.04 examples/s]49033 examples [00:56, 883.67 examples/s]49125 examples [00:56, 893.45 examples/s]49215 examples [00:56, 866.78 examples/s]49303 examples [00:56, 869.33 examples/s]49392 examples [00:56, 872.53 examples/s]49480 examples [00:56, 854.68 examples/s]49571 examples [00:56, 868.56 examples/s]49659 examples [00:56, 866.70 examples/s]49746 examples [00:57, 857.04 examples/s]49836 examples [00:57, 867.21 examples/s]49925 examples [00:57, 872.61 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 10%|▉         | 4953/50000 [00:00<00:00, 49526.27 examples/s] 32%|███▏      | 15750/50000 [00:00<00:00, 59127.89 examples/s] 53%|█████▎    | 26289/50000 [00:00<00:00, 68095.06 examples/s] 74%|███████▍  | 37117/50000 [00:00<00:00, 76625.16 examples/s] 96%|█████████▌| 47982/50000 [00:00<00:00, 84054.07 examples/s]                                                               0 examples [00:00, ? examples/s]70 examples [00:00, 697.97 examples/s]160 examples [00:00, 747.27 examples/s]251 examples [00:00, 788.35 examples/s]343 examples [00:00, 821.17 examples/s]436 examples [00:00, 849.46 examples/s]525 examples [00:00, 858.00 examples/s]617 examples [00:00, 874.47 examples/s]709 examples [00:00, 886.42 examples/s]798 examples [00:00, 885.49 examples/s]885 examples [00:01, 876.02 examples/s]974 examples [00:01, 879.44 examples/s]1062 examples [00:01, 879.04 examples/s]1152 examples [00:01, 883.22 examples/s]1241 examples [00:01, 883.76 examples/s]1329 examples [00:01, 877.44 examples/s]1417 examples [00:01, 693.54 examples/s]1505 examples [00:01, 739.78 examples/s]1594 examples [00:01, 778.74 examples/s]1676 examples [00:02, 789.98 examples/s]1759 examples [00:02, 799.29 examples/s]1844 examples [00:02, 813.01 examples/s]1927 examples [00:02, 810.01 examples/s]2015 examples [00:02, 829.12 examples/s]2103 examples [00:02, 843.06 examples/s]2188 examples [00:02, 838.11 examples/s]2275 examples [00:02, 847.18 examples/s]2364 examples [00:02, 857.30 examples/s]2452 examples [00:02, 861.89 examples/s]2539 examples [00:03, 835.18 examples/s]2623 examples [00:03, 832.42 examples/s]2707 examples [00:03, 822.51 examples/s]2794 examples [00:03, 834.99 examples/s]2881 examples [00:03, 842.80 examples/s]2968 examples [00:03, 848.97 examples/s]3054 examples [00:03, 836.08 examples/s]3141 examples [00:03, 843.62 examples/s]3229 examples [00:03, 853.25 examples/s]3315 examples [00:03, 841.39 examples/s]3405 examples [00:04, 857.15 examples/s]3495 examples [00:04, 868.00 examples/s]3582 examples [00:04, 867.82 examples/s]3673 examples [00:04, 877.31 examples/s]3763 examples [00:04, 882.00 examples/s]3857 examples [00:04, 896.28 examples/s]3947 examples [00:04, 894.25 examples/s]4037 examples [00:04, 883.07 examples/s]4126 examples [00:04, 884.96 examples/s]4215 examples [00:04, 878.34 examples/s]4303 examples [00:05, 850.32 examples/s]4391 examples [00:05, 856.38 examples/s]4479 examples [00:05, 862.92 examples/s]4569 examples [00:05, 873.62 examples/s]4661 examples [00:05, 884.96 examples/s]4751 examples [00:05, 888.28 examples/s]4842 examples [00:05, 893.58 examples/s]4934 examples [00:05, 900.19 examples/s]5028 examples [00:05, 910.09 examples/s]5122 examples [00:05, 917.86 examples/s]5216 examples [00:06, 922.82 examples/s]5309 examples [00:06, 906.11 examples/s]5405 examples [00:06, 919.41 examples/s]5498 examples [00:06, 918.10 examples/s]5592 examples [00:06, 921.44 examples/s]5687 examples [00:06, 928.28 examples/s]5780 examples [00:06, 924.30 examples/s]5873 examples [00:06, 915.43 examples/s]5965 examples [00:06, 892.10 examples/s]6055 examples [00:07, 878.37 examples/s]6144 examples [00:07, 865.72 examples/s]6231 examples [00:07, 849.50 examples/s]6321 examples [00:07, 863.87 examples/s]6409 examples [00:07, 868.20 examples/s]6496 examples [00:07, 863.43 examples/s]6584 examples [00:07, 865.72 examples/s]6671 examples [00:07, 852.33 examples/s]6761 examples [00:07, 865.99 examples/s]6852 examples [00:07, 876.11 examples/s]6941 examples [00:08, 873.15 examples/s]7029 examples [00:08, 870.81 examples/s]7117 examples [00:08, 873.12 examples/s]7205 examples [00:08, 873.77 examples/s]7294 examples [00:08, 875.87 examples/s]7382 examples [00:08, 874.43 examples/s]7470 examples [00:08, 868.94 examples/s]7557 examples [00:08, 862.43 examples/s]7644 examples [00:08, 863.76 examples/s]7731 examples [00:08, 858.00 examples/s]7818 examples [00:09, 861.22 examples/s]7907 examples [00:09, 867.31 examples/s]7996 examples [00:09, 873.73 examples/s]8086 examples [00:09, 878.83 examples/s]8175 examples [00:09, 882.09 examples/s]8264 examples [00:09, 851.40 examples/s]8356 examples [00:09, 870.21 examples/s]8444 examples [00:09, 829.64 examples/s]8530 examples [00:09, 836.65 examples/s]8617 examples [00:09, 844.26 examples/s]8703 examples [00:10, 847.09 examples/s]8788 examples [00:10, 844.95 examples/s]8873 examples [00:10, 841.46 examples/s]8958 examples [00:10, 841.95 examples/s]9043 examples [00:10, 842.21 examples/s]9132 examples [00:10, 853.19 examples/s]9218 examples [00:10, 848.14 examples/s]9305 examples [00:10, 853.46 examples/s]9391 examples [00:10, 852.47 examples/s]9481 examples [00:10, 865.08 examples/s]9568 examples [00:11, 861.75 examples/s]9655 examples [00:11, 827.71 examples/s]9746 examples [00:11, 850.78 examples/s]9832 examples [00:11, 851.15 examples/s]9924 examples [00:11, 870.30 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteRV6PB5/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteRV6PB5/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fb5233aa950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fb5233aa950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fb5233aa950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fb4a98fa668>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fb50c2b18d0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fb5233aa950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fb5233aa950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fb5233aa950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fb52339bc50>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fb520b350f0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7fb4a41b3268> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7fb4a41b3268> 

  function with postional parmater data_info <function split_train_valid at 0x7fb4a41b3268> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7fb4a41b3378> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7fb4a41b3378> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7fb4a41b3378> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.5)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.1)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.1)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.2)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=4eef6975ff58d741e3245e00892b153c0cf995ed2deba18cd04b6c6bcc5acb5b
  Stored in directory: /tmp/pip-ephem-wheel-cache-dadimtaz/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
Successfully built en-core-web-sm
Installing collected packages: en-core-web-sm
Successfully installed en-core-web-sm-2.2.5
WARNING: You are using pip version 20.1; however, version 20.1.1 is available.
You should consider upgrading via the '/opt/hostedtoolcache/Python/3.6.10/x64/bin/python -m pip install --upgrade pip' command.
[38;5;2m✔ Download and installation successful[0m
You can now load the model via spacy.load('en_core_web_sm')
[38;5;2m✔ Linking successful[0m
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/en_core_web_sm
-->
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/data/en
You can now load the model via spacy.load('en')
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<24:37:55, 9.72kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<17:28:35, 13.7kB/s].vector_cache/glove.6B.zip:   0%|          | 197k/862M [00:01<12:17:48, 19.5kB/s] .vector_cache/glove.6B.zip:   0%|          | 745k/862M [00:01<8:37:11, 27.8kB/s] .vector_cache/glove.6B.zip:   0%|          | 2.93M/862M [00:01<6:01:22, 39.6kB/s].vector_cache/glove.6B.zip:   1%|          | 8.38M/862M [00:01<4:11:26, 56.6kB/s].vector_cache/glove.6B.zip:   1%|▏         | 11.6M/862M [00:01<2:55:29, 80.8kB/s].vector_cache/glove.6B.zip:   2%|▏         | 16.7M/862M [00:01<2:02:11, 115kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 20.0M/862M [00:01<1:25:18, 165kB/s].vector_cache/glove.6B.zip:   3%|▎         | 24.9M/862M [00:01<59:27, 235kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 28.6M/862M [00:01<41:33, 334kB/s].vector_cache/glove.6B.zip:   4%|▍         | 33.4M/862M [00:02<29:00, 476kB/s].vector_cache/glove.6B.zip:   4%|▍         | 37.1M/862M [00:02<20:19, 676kB/s].vector_cache/glove.6B.zip:   5%|▍         | 42.1M/862M [00:02<14:13, 961kB/s].vector_cache/glove.6B.zip:   5%|▌         | 45.6M/862M [00:02<10:02, 1.36MB/s].vector_cache/glove.6B.zip:   6%|▌         | 50.6M/862M [00:02<07:03, 1.91MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.7M/862M [00:03<06:17, 2.14MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.8M/862M [00:05<06:18, 2.13MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.0M/862M [00:05<06:40, 2.01MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.0M/862M [00:05<05:14, 2.56MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.9M/862M [00:07<05:57, 2.24MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.1M/862M [00:07<07:00, 1.90MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.9M/862M [00:07<05:36, 2.38MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.0M/862M [00:07<04:04, 3.26MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.1M/862M [00:09<1:27:28, 152kB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.4M/862M [00:09<1:02:33, 212kB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.0M/862M [00:09<44:02, 301kB/s]  .vector_cache/glove.6B.zip:   8%|▊         | 69.2M/862M [00:11<33:51, 390kB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.4M/862M [00:11<26:25, 500kB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.1M/862M [00:11<19:03, 693kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.2M/862M [00:11<13:29, 976kB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.3M/862M [00:13<15:22, 855kB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.7M/862M [00:13<11:54, 1.10MB/s].vector_cache/glove.6B.zip:   9%|▊         | 75.2M/862M [00:13<08:39, 1.52MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.4M/862M [00:14<09:08, 1.43MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.6M/862M [00:15<09:03, 1.44MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.4M/862M [00:15<06:54, 1.89MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.2M/862M [00:15<04:57, 2.63MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.5M/862M [00:16<27:26, 474kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 81.9M/862M [00:17<20:32, 633kB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.5M/862M [00:17<14:41, 884kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.6M/862M [00:18<13:17, 974kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.8M/862M [00:19<12:02, 1.07MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.6M/862M [00:19<09:01, 1.43MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.5M/862M [00:19<06:25, 2.00MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.7M/862M [00:20<28:57, 445kB/s] .vector_cache/glove.6B.zip:  10%|█         | 90.1M/862M [00:20<21:36, 596kB/s].vector_cache/glove.6B.zip:  11%|█         | 91.7M/862M [00:21<15:25, 833kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.8M/862M [00:22<13:46, 930kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.2M/862M [00:22<10:44, 1.19MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.8M/862M [00:23<07:49, 1.63MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.0M/862M [00:24<08:29, 1.50MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.2M/862M [00:24<08:31, 1.49MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.9M/862M [00:25<06:30, 1.95MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:25<04:42, 2.70MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<11:31, 1.10MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<09:20, 1.35MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:27<06:51, 1.84MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<07:44, 1.63MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<07:58, 1.58MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<06:12, 2.02MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<06:22, 1.97MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<05:45, 2.18MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<04:20, 2.88MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<05:56, 2.10MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<06:43, 1.85MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<05:12, 2.39MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:32<03:51, 3.21MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 119M/862M [00:34<06:51, 1.81MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<06:03, 2.05MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<04:32, 2.72MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<06:04, 2.03MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<06:39, 1.85MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<05:16, 2.33MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<05:40, 2.16MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<05:14, 2.34MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:38<03:55, 3.12MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<05:36, 2.17MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<06:23, 1.90MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<05:02, 2.42MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:40<03:39, 3.31MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<11:42:12, 17.3kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<8:12:30, 24.6kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<5:44:20, 35.1kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<4:03:08, 49.6kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<2:52:36, 69.8kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<2:01:19, 99.2kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:44<1:24:43, 141kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<2:24:56, 82.7kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<1:42:37, 117kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<1:11:56, 166kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<53:00, 225kB/s]  .vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<39:11, 304kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<28:01, 425kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:48<19:45, 600kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<18:06, 654kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<13:54, 851kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<09:58, 1.18MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<09:43, 1.21MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<09:12, 1.28MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<07:02, 1.67MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<06:48, 1.72MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<05:57, 1.96MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 162M/862M [00:54<04:27, 2.62MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<05:50, 1.99MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<06:27, 1.80MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<05:07, 2.27MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:56<03:43, 3.11MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<1:37:08, 119kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:58<1:09:09, 167kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<48:33, 238kB/s]  .vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<36:35, 314kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<26:44, 430kB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<18:58, 604kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<15:56, 717kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<13:29, 847kB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<09:55, 1.15MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:02<07:06, 1.60MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<09:22, 1.21MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<07:42, 1.47MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<05:37, 2.01MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<06:36, 1.71MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<05:46, 1.96MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<04:15, 2.64MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<05:39, 1.98MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<06:14, 1.80MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<04:52, 2.30MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:08<03:31, 3.17MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<16:11, 689kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<12:29, 893kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:10<09:00, 1.24MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<08:53, 1.25MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<08:29, 1.30MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<06:25, 1.72MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:12<04:38, 2.38MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<08:54, 1.24MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<07:22, 1.49MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<05:26, 2.02MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<06:20, 1.73MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<05:32, 1.97MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<04:06, 2.66MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<05:28, 1.99MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<06:02, 1.80MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<04:42, 2.31MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:18<03:24, 3.18MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<14:25, 750kB/s] .vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<11:13, 964kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:19<08:06, 1.33MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<08:11, 1.31MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<06:49, 1.58MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:21<05:02, 2.13MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<06:02, 1.77MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<06:23, 1.67MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<05:00, 2.13MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:24<03:36, 2.94MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<1:30:13, 118kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<1:04:11, 165kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<45:05, 235kB/s]  .vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<33:55, 311kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<25:58, 406kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<18:37, 565kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:27<13:10, 796kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<12:52, 814kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<10:05, 1.04MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:29<07:16, 1.44MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<07:32, 1.38MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<07:23, 1.41MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<05:37, 1.85MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:31<04:02, 2.56MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<10:06, 1.02MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<08:08, 1.27MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:33<05:57, 1.73MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<06:32, 1.57MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<07:14, 1.42MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<05:37, 1.82MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:35<04:06, 2.49MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<05:55, 1.72MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<06:46, 1.51MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<05:23, 1.89MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:37<03:54, 2.60MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 254M/862M [01:39<07:43, 1.31MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<07:52, 1.29MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<06:02, 1.68MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:39<04:20, 2.32MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 259M/862M [01:41<07:17, 1.38MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<07:27, 1.35MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<05:43, 1.75MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:41<04:10, 2.40MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:43<05:56, 1.68MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:43<06:29, 1.54MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<05:01, 1.98MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:43<03:39, 2.71MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<06:04, 1.63MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<06:42, 1.48MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<05:35, 1.78MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<04:50, 2.05MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<05:13, 1.90MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<04:38, 2.13MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<04:22, 2.26MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<04:11, 2.36MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<04:00, 2.46MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<03:52, 2.55MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<03:46, 2.61MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<03:41, 2.67MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:46<03:36, 2.73MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<09:17, 1.06MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<08:01, 1.23MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<06:38, 1.48MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<05:38, 1.74MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<04:57, 1.98MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<04:28, 2.20MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<04:07, 2.38MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<03:46, 2.60MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<03:43, 2.63MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<03:34, 2.73MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:48<03:23, 2.89MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<08:25, 1.16MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<08:32, 1.14MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<07:02, 1.39MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<05:54, 1.65MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<05:06, 1.91MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<04:33, 2.14MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<04:09, 2.34MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<03:52, 2.51MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<03:39, 2.66MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<03:21, 2.90MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<03:06, 3.13MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<03:26, 2.82MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<13:18, 730kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<11:49, 821kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<09:18, 1.04MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [01:51<07:19, 1.32MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<06:02, 1.61MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<05:12, 1.86MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<04:32, 2.13MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<04:04, 2.38MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<03:38, 2.66MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<03:24, 2.83MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<03:15, 2.96MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<17:29, 551kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<14:35, 661kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<11:08, 864kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<08:31, 1.13MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<06:48, 1.41MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<05:24, 1.78MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:53<04:57, 1.94MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<04:18, 2.23MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<03:50, 2.50MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<03:30, 2.73MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<03:15, 2.93MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<25:15, 379kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<19:48, 483kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<14:36, 655kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:55<10:55, 875kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<08:44, 1.09MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<06:46, 1.41MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:55<05:25, 1.76MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:55<04:54, 1.94MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:56<04:13, 2.26MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:56<03:35, 2.65MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:56<03:11, 2.98MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<14:39, 649kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<12:01, 790kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<09:26, 1.01MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<07:22, 1.29MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<05:47, 1.64MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:57<04:54, 1.93MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:57<04:08, 2.28MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<03:36, 2.62MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<03:06, 3.03MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<08:32, 1.10MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<09:50, 958kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<07:57, 1.19MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<06:13, 1.51MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<04:54, 1.92MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:59<04:02, 2.32MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [01:59<03:26, 2.74MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<02:59, 3.14MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<06:39, 1.41MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<07:31, 1.25MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<06:00, 1.56MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<04:48, 1.94MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:01<03:58, 2.35MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:01<03:12, 2.91MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:01<02:47, 3.33MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:03<06:19, 1.47MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:03<09:28, 982kB/s] .vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:03<07:55, 1.17MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<06:01, 1.54MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:03<04:39, 1.99MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:03<03:40, 2.52MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:03<02:57, 3.12MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:05<09:07, 1.01MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:05<10:31, 877kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<08:13, 1.12MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<06:14, 1.48MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:05<04:39, 1.97MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:05<03:35, 2.55MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<07:37, 1.20MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:07<09:06, 1.01MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<07:07, 1.29MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<05:15, 1.74MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<04:11, 2.18MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:07<03:10, 2.87MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<07:09, 1.27MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<07:58, 1.14MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<06:20, 1.43MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<04:41, 1.93MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:09<03:31, 2.56MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<06:33, 1.38MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<07:21, 1.23MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<05:42, 1.58MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<04:15, 2.11MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:11<03:11, 2.80MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<09:17, 963kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:13<11:21, 788kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:13<08:56, 1.00MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<06:33, 1.36MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:13<04:47, 1.86MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<07:40, 1.16MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:15<09:18, 954kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:15<07:34, 1.17MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<05:35, 1.59MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:15<04:05, 2.16MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<07:58, 1.11MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<09:10, 960kB/s] .vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:17<07:12, 1.22MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<05:18, 1.66MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:17<03:52, 2.26MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<14:07, 620kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<13:09, 665kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<09:54, 883kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<07:07, 1.22MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<05:10, 1.68MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<07:23, 1.17MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<07:59, 1.09MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<06:19, 1.37MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<04:37, 1.87MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<05:19, 1.62MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<06:21, 1.36MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:23<04:59, 1.72MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<03:37, 2.36MB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:23<02:42, 3.15MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<12:49, 666kB/s] .vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<11:18, 756kB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:25<08:25, 1.01MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<06:03, 1.40MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<06:50, 1.24MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<07:04, 1.20MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:27<05:27, 1.55MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<03:58, 2.12MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<04:54, 1.71MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<05:28, 1.53MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<04:15, 1.97MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<03:07, 2.68MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<05:45, 1.45MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<06:10, 1.35MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:30<04:45, 1.75MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<03:28, 2.39MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<04:49, 1.71MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 366M/862M [02:32<05:23, 1.53MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<04:12, 1.96MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<03:03, 2.69MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<05:13, 1.57MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<05:52, 1.39MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<04:35, 1.78MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<03:21, 2.43MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:36<05:21, 1.52MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:36<05:56, 1.37MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:36<04:42, 1.72MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:37<03:24, 2.36MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<05:18, 1.52MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<05:09, 1.56MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<03:58, 2.02MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:39<02:51, 2.80MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<1:00:08, 133kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<43:56, 182kB/s]  .vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:40<31:07, 256kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<21:47, 364kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<21:11, 374kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<16:30, 480kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:42<11:57, 660kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:43<08:26, 931kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<13:55, 564kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<11:33, 679kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<08:29, 923kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:44<06:00, 1.30MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<08:45, 888kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<07:51, 990kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<05:54, 1.31MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:46<04:13, 1.83MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<10:50, 711kB/s] .vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<08:42, 884kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:48<06:22, 1.21MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<05:50, 1.31MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<05:34, 1.37MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:50<04:16, 1.79MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<04:12, 1.80MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<06:00, 1.26MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<04:58, 1.52MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:52<03:38, 2.07MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<04:15, 1.76MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<04:29, 1.67MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<03:30, 2.13MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<03:39, 2.03MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<04:00, 1.86MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<03:10, 2.34MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<03:24, 2.16MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 420M/862M [02:58<03:52, 1.90MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<03:04, 2.39MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<03:19, 2.19MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<03:44, 1.95MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<02:58, 2.44MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<03:15, 2.22MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<03:37, 1.99MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<02:53, 2.50MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<03:11, 2.25MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<03:37, 1.98MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<02:50, 2.52MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:04<02:04, 3.43MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<05:22, 1.32MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<05:08, 1.38MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<03:55, 1.80MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<03:53, 1.81MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<04:08, 1.69MB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:08<03:14, 2.16MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<03:23, 2.05MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<05:11, 1.34MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<04:20, 1.60MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<03:10, 2.18MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<03:47, 1.82MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<03:59, 1.72MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<03:07, 2.19MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:14<03:17, 2.07MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:14<05:04, 1.34MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<04:14, 1.60MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<03:06, 2.18MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:14<02:16, 2.97MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:16<24:39, 274kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:16<19:10, 352kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<13:49, 487kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:16<09:45, 687kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 461M/862M [03:18<09:25, 708kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<08:30, 785kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<06:19, 1.05MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:18<04:32, 1.46MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<04:51, 1.36MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<05:05, 1.30MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<03:55, 1.68MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<02:49, 2.32MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:22<04:21, 1.50MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:22<04:49, 1.36MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<03:47, 1.72MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<02:51, 2.28MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:24<03:25, 1.89MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:24<05:00, 1.29MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:24<04:08, 1.56MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<03:01, 2.13MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:26<03:37, 1.77MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:26<04:55, 1.30MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<03:58, 1.61MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<02:53, 2.20MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:26<02:10, 2.91MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:28<04:52, 1.30MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:28<05:47, 1.09MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<04:32, 1.39MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<03:16, 1.92MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:28<02:28, 2.53MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<05:25, 1.16MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:30<06:08, 1.02MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:30<04:50, 1.29MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<03:31, 1.77MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<04:01, 1.54MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<04:57, 1.25MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<04:00, 1.55MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<02:54, 2.12MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<03:37, 1.69MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<04:40, 1.31MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<03:47, 1.61MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<02:46, 2.19MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<03:31, 1.72MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:36<04:33, 1.33MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:36<03:41, 1.64MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<02:42, 2.22MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<03:28, 1.73MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:38<04:30, 1.33MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<03:38, 1.64MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<02:40, 2.22MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<03:26, 1.72MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:40<04:27, 1.33MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<03:31, 1.67MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<02:35, 2.27MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<03:22, 1.73MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:42<04:25, 1.32MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<03:29, 1.67MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<02:34, 2.26MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<03:19, 1.74MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:43<04:05, 1.41MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<03:16, 1.76MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<02:24, 2.38MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<03:13, 1.77MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<04:07, 1.38MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:46<03:21, 1.70MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<02:26, 2.32MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<03:14, 1.74MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<04:05, 1.38MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:48<03:14, 1.74MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<02:22, 2.36MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:48<01:46, 3.13MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:49<06:23, 871kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:49<06:16, 886kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<04:45, 1.17MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<03:25, 1.62MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:50<02:29, 2.21MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<07:05, 776kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<06:45, 814kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<05:09, 1.06MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<03:42, 1.47MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<04:04, 1.34MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<04:36, 1.18MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<03:36, 1.50MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:54<02:38, 2.05MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<03:17, 1.63MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<04:02, 1.33MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<03:11, 1.68MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<02:18, 2.30MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:56<01:45, 3.03MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<06:27, 819kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<06:14, 848kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<04:46, 1.10MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<03:26, 1.52MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<03:49, 1.37MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<04:23, 1.19MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<03:25, 1.53MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<02:28, 2.09MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:00<01:56, 2.67MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<03:50, 1.34MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<04:58, 1.04MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<04:05, 1.26MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<02:59, 1.71MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:02<02:10, 2.34MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<06:38, 765kB/s] .vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<06:54, 737kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<05:17, 960kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<03:49, 1.32MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:04<02:45, 1.82MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<08:51, 567kB/s] .vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<08:13, 609kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<06:13, 805kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<04:28, 1.11MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:06<03:12, 1.54MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<10:48, 458kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<09:34, 517kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<07:12, 685kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<05:09, 953kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:08<03:40, 1.33MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<14:51, 328kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<12:22, 394kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<09:08, 533kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:09<06:28, 748kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 573M/862M [04:10<04:35, 1.05MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<23:05, 208kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<18:06, 265kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<13:09, 365kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:11<09:16, 516kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:12<06:30, 728kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<15:24, 308kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<12:42, 373kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<09:21, 506kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:13<06:37, 711kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:14<04:41, 996kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:15<26:04, 179kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:15<19:59, 234kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:15<14:22, 325kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:15<10:07, 458kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<08:04, 571kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<07:21, 625kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<05:31, 832kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:17<03:58, 1.15MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:17<02:49, 1.61MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<13:51, 327kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<11:32, 393kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<08:26, 536kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:19<06:00, 750kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:19<04:15, 1.05MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<29:33, 151kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<22:29, 198kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<16:05, 277kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:21<11:19, 391kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:21<07:56, 554kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<12:16, 358kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<10:13, 430kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<07:31, 583kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:23<05:21, 814kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<04:41, 923kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<04:53, 884kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<03:50, 1.12MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:25<02:46, 1.55MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<02:51, 1.49MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<03:43, 1.14MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<03:01, 1.41MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:27<02:13, 1.90MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<02:27, 1.70MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<03:25, 1.22MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<02:44, 1.53MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:29<02:01, 2.06MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:29<01:27, 2.82MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<10:52, 379kB/s] .vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<09:08, 450kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<06:47, 605kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:31<04:49, 847kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:31<03:24, 1.19MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<17:12, 235kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<13:34, 298kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<09:48, 412kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:33<06:54, 581kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:33<04:57, 808kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<05:06, 780kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<05:04, 785kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<03:52, 1.02MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:35<02:48, 1.41MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:35<02:06, 1.87MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:37<03:13, 1.21MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<04:21, 896kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<03:35, 1.09MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<02:38, 1.47MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:37<01:55, 2.01MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<02:59, 1.28MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<04:10, 921kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<03:24, 1.13MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<02:30, 1.53MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:39<01:49, 2.07MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:41<03:05, 1.22MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:41<04:11, 899kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<03:24, 1.10MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<02:30, 1.50MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:41<01:49, 2.05MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<03:08, 1.18MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<04:00, 925kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<03:14, 1.14MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<02:21, 1.55MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:43<01:43, 2.11MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<03:13, 1.12MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<04:01, 902kB/s] .vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<03:12, 1.13MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<02:20, 1.54MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:45<01:42, 2.09MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<03:12, 1.11MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<03:58, 897kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<03:13, 1.10MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<02:20, 1.51MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:47<01:42, 2.06MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:49<03:15, 1.08MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:49<03:58, 880kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<03:10, 1.10MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<02:19, 1.49MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:49<01:40, 2.05MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:51<03:23, 1.01MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<04:03, 844kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<03:10, 1.08MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<02:18, 1.47MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:51<01:40, 2.01MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<03:20, 1.00MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<03:58, 845kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<03:09, 1.06MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<02:18, 1.44MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:53<01:39, 1.98MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<03:26, 957kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<04:00, 821kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<03:10, 1.03MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<02:19, 1.41MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [04:55<01:40, 1.93MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<03:21, 961kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<03:46, 855kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<03:00, 1.07MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<02:11, 1.45MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:57<01:35, 1.99MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<03:14, 974kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:59<03:47, 831kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:59<02:57, 1.07MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<02:09, 1.45MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:59<01:33, 1.99MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<02:39, 1.16MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:01<03:13, 958kB/s] .vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<02:36, 1.18MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:01<01:54, 1.60MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:01<01:23, 2.17MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<03:04, 983kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:03<03:36, 834kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<02:49, 1.06MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<02:03, 1.45MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:03<01:29, 1.98MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<03:03, 961kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<03:26, 854kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<02:45, 1.07MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<01:59, 1.46MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:05<01:26, 2.00MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [05:06<02:57, 972kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [05:07<03:28, 829kB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<02:45, 1.04MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<02:00, 1.42MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:07<01:27, 1.94MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:08<02:53, 972kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:08<03:15, 859kB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:09<02:36, 1.07MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<01:54, 1.46MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:09<01:22, 2.00MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<02:53, 948kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<03:21, 814kB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:11<02:40, 1.02MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<01:55, 1.41MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:11<01:23, 1.93MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<02:51, 936kB/s] .vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<03:11, 837kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:13<02:32, 1.05MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<01:50, 1.43MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:13<01:20, 1.96MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<02:55, 890kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<03:11, 814kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<02:28, 1.05MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<01:47, 1.43MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:15<01:17, 1.96MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<02:49, 897kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<03:05, 818kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<02:26, 1.04MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<01:45, 1.42MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:17<01:16, 1.94MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<02:57, 832kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<03:09, 779kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<02:26, 1.01MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<01:45, 1.39MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<01:17, 1.88MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<01:38, 1.46MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<02:12, 1.08MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<01:47, 1.33MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:21<01:19, 1.79MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:21<00:57, 2.44MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<03:10, 731kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<03:15, 714kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:23<02:38, 878kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<01:54, 1.20MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<01:27, 1.57MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:23<01:08, 1.99MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:23<00:54, 2.47MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<03:15, 692kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<03:39, 618kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<02:53, 779kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<02:07, 1.05MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<01:36, 1.39MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<01:12, 1.83MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:25<00:57, 2.30MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<01:43, 1.26MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<01:48, 1.21MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<01:23, 1.56MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<01:04, 2.00MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:27<00:51, 2.51MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<00:41, 3.05MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:27<00:35, 3.60MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<04:25, 479kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<04:14, 499kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<03:11, 661kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<02:20, 893kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<01:43, 1.21MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<01:17, 1.60MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:29<00:59, 2.06MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<02:11, 938kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<02:32, 808kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<02:02, 1.00MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<01:31, 1.33MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<01:08, 1.76MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<00:54, 2.22MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:31<00:42, 2.80MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:57, 1.01MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<02:26, 811kB/s] .vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<01:57, 1.01MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:32<01:27, 1.34MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<01:06, 1.76MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:33<00:51, 2.25MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:33<00:40, 2.85MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<02:02, 933kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<02:27, 775kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<01:57, 970kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:34<01:27, 1.29MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<01:06, 1.70MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<00:51, 2.18MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:35<00:40, 2.76MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<01:58, 929kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<02:17, 806kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<01:50, 1.00MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<01:21, 1.34MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:36<01:01, 1.75MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 754M/862M [05:37<00:48, 2.24MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:37<00:38, 2.78MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<01:48, 985kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<02:13, 799kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<01:46, 996kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<01:19, 1.32MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:38<01:00, 1.74MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<00:46, 2.22MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:39<00:37, 2.77MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:51, 913kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<02:13, 764kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:46, 958kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<01:19, 1.28MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:40<00:59, 1.68MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:41<00:45, 2.16MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:41<00:36, 2.73MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<01:55, 849kB/s] .vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<02:08, 763kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<01:41, 962kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 765M/862M [05:42<01:14, 1.29MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:42<00:56, 1.69MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<00:43, 2.21MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:43<00:34, 2.77MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<02:17, 682kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<02:21, 662kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<01:50, 848kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<01:21, 1.14MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:44<00:59, 1.54MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<00:45, 1.99MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:45<00:35, 2.57MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<05:03, 296kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<04:16, 350kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<03:07, 477kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<02:14, 660kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:46<01:36, 908kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:46<01:10, 1.22MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<01:25, 1.00MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<01:34, 905kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<01:15, 1.13MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<00:55, 1.52MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:48<00:41, 2.01MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:48<00:31, 2.59MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<01:10, 1.15MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<01:22, 987kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<01:05, 1.24MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<00:48, 1.65MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:50<00:36, 2.17MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:50<00:27, 2.81MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<01:38, 786kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<01:37, 791kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<01:15, 1.02MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:54, 1.39MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:52<00:39, 1.87MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:52<00:29, 2.46MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:54<07:00, 174kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<05:17, 230kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<03:47, 320kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:54<02:38, 451kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:54<01:51, 632kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<01:45, 654kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<01:55, 596kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<01:31, 751kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<01:05, 1.03MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:56<00:47, 1.41MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:56<00:33, 1.93MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<02:33, 424kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<02:05, 516kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<01:31, 707kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<01:04, 977kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:58<00:45, 1.36MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<01:06, 922kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<01:16, 799kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<01:00, 993kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<00:43, 1.35MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:00<00:31, 1.85MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<00:46, 1.23MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<00:45, 1.25MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:02<00:34, 1.63MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:02<00:24, 2.21MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:32, 1.64MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:45, 1.15MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:37, 1.40MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:27, 1.88MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:04<00:19, 2.56MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<01:17, 627kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<01:13, 655kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:55, 858kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:06<00:39, 1.19MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:35, 1.23MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:41, 1.06MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:32, 1.36MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:08<00:22, 1.85MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:08<00:15, 2.53MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<01:18, 515kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<01:07, 594kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:49, 801kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:10<00:34, 1.11MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<00:31, 1.14MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<00:32, 1.09MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:25, 1.41MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:12<00:17, 1.92MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:12<00:12, 2.61MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:14<01:57, 270kB/s] .vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:14<01:31, 345kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<01:05, 476kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:14<00:43, 670kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:15<00:37, 743kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:33, 825kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:24, 1.11MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:16<00:16, 1.53MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<00:18, 1.26MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:20, 1.16MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:15, 1.50MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:18<00:10, 2.07MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:12, 1.60MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:13, 1.44MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:10, 1.82MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:20<00:06, 2.48MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:10, 1.42MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:10, 1.40MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:07, 1.80MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:22<00:04, 2.48MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:09, 1.19MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:24<00:09, 1.18MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:06, 1.54MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:24<00:03, 2.13MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:04, 1.46MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:26<00:04, 1.36MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:03, 1.74MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:26<00:01, 2.38MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:02, 1.31MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:02, 1.27MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:01, 1.61MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<22:16:01,  4.99it/s]  0%|          | 728/400000 [00:00<15:33:47,  7.13it/s]  0%|          | 1485/400000 [00:00<10:52:40, 10.18it/s]  1%|          | 2218/400000 [00:00<7:36:18, 14.53it/s]   1%|          | 2968/400000 [00:00<5:19:04, 20.74it/s]  1%|          | 3688/400000 [00:00<3:43:13, 29.59it/s]  1%|          | 4408/400000 [00:00<2:36:14, 42.20it/s]  1%|▏         | 5156/400000 [00:00<1:49:25, 60.14it/s]  1%|▏         | 5900/400000 [00:01<1:16:43, 85.61it/s]  2%|▏         | 6659/400000 [00:01<53:51, 121.71it/s]   2%|▏         | 7406/400000 [00:01<37:53, 172.67it/s]  2%|▏         | 8130/400000 [00:01<26:45, 244.07it/s]  2%|▏         | 8881/400000 [00:01<18:57, 343.88it/s]  2%|▏         | 9620/400000 [00:01<13:30, 481.64it/s]  3%|▎         | 10389/400000 [00:01<09:41, 670.06it/s]  3%|▎         | 11140/400000 [00:01<07:01, 921.96it/s]  3%|▎         | 11884/400000 [00:01<05:10, 1250.49it/s]  3%|▎         | 12628/400000 [00:01<03:53, 1660.25it/s]  3%|▎         | 13390/400000 [00:02<02:58, 2169.15it/s]  4%|▎         | 14140/400000 [00:02<02:19, 2756.91it/s]  4%|▎         | 14898/400000 [00:02<01:53, 3406.85it/s]  4%|▍         | 15646/400000 [00:02<01:34, 4064.62it/s]  4%|▍         | 16392/400000 [00:02<01:21, 4685.29it/s]  4%|▍         | 17133/400000 [00:02<01:13, 5239.20it/s]  4%|▍         | 17882/400000 [00:02<01:06, 5758.09it/s]  5%|▍         | 18639/400000 [00:02<01:01, 6201.81it/s]  5%|▍         | 19384/400000 [00:02<00:58, 6509.03it/s]  5%|▌         | 20141/400000 [00:02<00:55, 6793.23it/s]  5%|▌         | 20888/400000 [00:03<00:54, 6913.16it/s]  5%|▌         | 21642/400000 [00:03<00:53, 7089.59it/s]  6%|▌         | 22398/400000 [00:03<00:52, 7222.18it/s]  6%|▌         | 23151/400000 [00:03<00:51, 7309.59it/s]  6%|▌         | 23910/400000 [00:03<00:50, 7389.54it/s]  6%|▌         | 24662/400000 [00:03<00:50, 7405.96it/s]  6%|▋         | 25412/400000 [00:03<00:50, 7351.19it/s]  7%|▋         | 26168/400000 [00:03<00:50, 7410.07it/s]  7%|▋         | 26914/400000 [00:03<00:50, 7414.43it/s]  7%|▋         | 27666/400000 [00:03<00:50, 7445.16it/s]  7%|▋         | 28413/400000 [00:04<00:50, 7416.63it/s]  7%|▋         | 29157/400000 [00:04<00:50, 7396.70it/s]  7%|▋         | 29898/400000 [00:04<00:50, 7301.74it/s]  8%|▊         | 30630/400000 [00:04<00:50, 7292.87it/s]  8%|▊         | 31377/400000 [00:04<00:50, 7344.83it/s]  8%|▊         | 32121/400000 [00:04<00:49, 7371.07it/s]  8%|▊         | 32859/400000 [00:04<00:50, 7249.04it/s]  8%|▊         | 33600/400000 [00:04<00:50, 7296.10it/s]  9%|▊         | 34331/400000 [00:04<00:52, 7010.07it/s]  9%|▉         | 35035/400000 [00:04<00:53, 6856.78it/s]  9%|▉         | 35724/400000 [00:05<00:53, 6765.26it/s]  9%|▉         | 36404/400000 [00:05<00:53, 6773.01it/s]  9%|▉         | 37083/400000 [00:05<00:53, 6753.79it/s]  9%|▉         | 37760/400000 [00:05<00:53, 6725.93it/s] 10%|▉         | 38499/400000 [00:05<00:52, 6911.73it/s] 10%|▉         | 39259/400000 [00:05<00:50, 7102.78it/s] 10%|█         | 40011/400000 [00:05<00:49, 7220.48it/s] 10%|█         | 40748/400000 [00:05<00:49, 7263.96it/s] 10%|█         | 41488/400000 [00:05<00:49, 7303.70it/s] 11%|█         | 42239/400000 [00:05<00:48, 7361.75it/s] 11%|█         | 42977/400000 [00:06<00:48, 7323.22it/s] 11%|█         | 43736/400000 [00:06<00:48, 7399.71it/s] 11%|█         | 44490/400000 [00:06<00:47, 7440.26it/s] 11%|█▏        | 45235/400000 [00:06<00:47, 7405.92it/s] 11%|█▏        | 45989/400000 [00:06<00:47, 7444.82it/s] 12%|█▏        | 46741/400000 [00:06<00:47, 7466.64it/s] 12%|█▏        | 47488/400000 [00:06<00:47, 7403.69it/s] 12%|█▏        | 48245/400000 [00:06<00:47, 7451.31it/s] 12%|█▏        | 48991/400000 [00:06<00:47, 7451.88it/s] 12%|█▏        | 49738/400000 [00:06<00:46, 7456.94it/s] 13%|█▎        | 50487/400000 [00:07<00:46, 7465.03it/s] 13%|█▎        | 51234/400000 [00:07<00:47, 7372.95it/s] 13%|█▎        | 51972/400000 [00:07<00:47, 7352.93it/s] 13%|█▎        | 52717/400000 [00:07<00:47, 7380.19it/s] 13%|█▎        | 53466/400000 [00:07<00:46, 7412.53it/s] 14%|█▎        | 54236/400000 [00:07<00:46, 7495.02it/s] 14%|█▎        | 54986/400000 [00:07<00:47, 7281.50it/s] 14%|█▍        | 55716/400000 [00:07<00:49, 7021.98it/s] 14%|█▍        | 56422/400000 [00:07<00:50, 6818.03it/s] 14%|█▍        | 57108/400000 [00:08<00:52, 6587.35it/s] 14%|█▍        | 57786/400000 [00:08<00:51, 6642.31it/s] 15%|█▍        | 58471/400000 [00:08<00:50, 6702.89it/s] 15%|█▍        | 59160/400000 [00:08<00:50, 6756.86it/s] 15%|█▍        | 59899/400000 [00:08<00:49, 6932.31it/s] 15%|█▌        | 60650/400000 [00:08<00:47, 7094.69it/s] 15%|█▌        | 61411/400000 [00:08<00:46, 7239.75it/s] 16%|█▌        | 62159/400000 [00:08<00:46, 7308.59it/s] 16%|█▌        | 62896/400000 [00:08<00:46, 7324.96it/s] 16%|█▌        | 63630/400000 [00:08<00:46, 7283.81it/s] 16%|█▌        | 64360/400000 [00:09<00:46, 7260.55it/s] 16%|█▋        | 65106/400000 [00:09<00:45, 7317.88it/s] 16%|█▋        | 65848/400000 [00:09<00:45, 7346.49it/s] 17%|█▋        | 66590/400000 [00:09<00:45, 7367.99it/s] 17%|█▋        | 67328/400000 [00:09<00:45, 7363.01it/s] 17%|█▋        | 68065/400000 [00:09<00:45, 7318.18it/s] 17%|█▋        | 68798/400000 [00:09<00:46, 7136.00it/s] 17%|█▋        | 69526/400000 [00:09<00:46, 7177.91it/s] 18%|█▊        | 70273/400000 [00:09<00:45, 7260.61it/s] 18%|█▊        | 71023/400000 [00:09<00:44, 7328.99it/s] 18%|█▊        | 71757/400000 [00:10<00:44, 7305.74it/s] 18%|█▊        | 72489/400000 [00:10<00:45, 7203.67it/s] 18%|█▊        | 73211/400000 [00:10<00:45, 7110.13it/s] 18%|█▊        | 73960/400000 [00:10<00:45, 7218.35it/s] 19%|█▊        | 74711/400000 [00:10<00:44, 7301.78it/s] 19%|█▉        | 75468/400000 [00:10<00:43, 7379.13it/s] 19%|█▉        | 76226/400000 [00:10<00:43, 7437.79it/s] 19%|█▉        | 77003/400000 [00:10<00:42, 7531.95it/s] 19%|█▉        | 77757/400000 [00:10<00:43, 7439.84it/s] 20%|█▉        | 78511/400000 [00:10<00:43, 7468.59it/s] 20%|█▉        | 79270/400000 [00:11<00:42, 7503.37it/s] 20%|██        | 80031/400000 [00:11<00:42, 7532.25it/s] 20%|██        | 80785/400000 [00:11<00:42, 7504.11it/s] 20%|██        | 81536/400000 [00:11<00:42, 7437.94it/s] 21%|██        | 82281/400000 [00:11<00:42, 7416.33it/s] 21%|██        | 83028/400000 [00:11<00:42, 7430.52it/s] 21%|██        | 83786/400000 [00:11<00:42, 7472.58it/s] 21%|██        | 84534/400000 [00:11<00:42, 7454.69it/s] 21%|██▏       | 85286/400000 [00:11<00:42, 7472.60it/s] 22%|██▏       | 86034/400000 [00:11<00:42, 7329.90it/s] 22%|██▏       | 86805/400000 [00:12<00:42, 7438.42it/s] 22%|██▏       | 87560/400000 [00:12<00:41, 7470.79it/s] 22%|██▏       | 88308/400000 [00:12<00:42, 7416.57it/s] 22%|██▏       | 89059/400000 [00:12<00:41, 7443.44it/s] 22%|██▏       | 89804/400000 [00:12<00:41, 7430.06it/s] 23%|██▎       | 90548/400000 [00:12<00:43, 7172.77it/s] 23%|██▎       | 91268/400000 [00:12<00:43, 7056.06it/s] 23%|██▎       | 91976/400000 [00:12<00:44, 6983.65it/s] 23%|██▎       | 92676/400000 [00:12<00:44, 6900.06it/s] 23%|██▎       | 93368/400000 [00:13<00:45, 6713.59it/s] 24%|██▎       | 94119/400000 [00:13<00:44, 6931.98it/s] 24%|██▎       | 94846/400000 [00:13<00:43, 7029.17it/s] 24%|██▍       | 95592/400000 [00:13<00:42, 7150.73it/s] 24%|██▍       | 96317/400000 [00:13<00:42, 7178.02it/s] 24%|██▍       | 97073/400000 [00:13<00:41, 7285.51it/s] 24%|██▍       | 97830/400000 [00:13<00:41, 7367.39it/s] 25%|██▍       | 98577/400000 [00:13<00:40, 7396.09it/s] 25%|██▍       | 99318/400000 [00:13<00:41, 7306.23it/s] 25%|██▌       | 100061/400000 [00:13<00:40, 7340.50it/s] 25%|██▌       | 100813/400000 [00:14<00:40, 7392.40it/s] 25%|██▌       | 101560/400000 [00:14<00:40, 7414.25it/s] 26%|██▌       | 102303/400000 [00:14<00:40, 7418.71it/s] 26%|██▌       | 103046/400000 [00:14<00:40, 7359.09it/s] 26%|██▌       | 103783/400000 [00:14<00:40, 7282.12it/s] 26%|██▌       | 104535/400000 [00:14<00:40, 7350.11it/s] 26%|██▋       | 105271/400000 [00:14<00:40, 7273.10it/s] 26%|██▋       | 105999/400000 [00:14<00:40, 7180.58it/s] 27%|██▋       | 106754/400000 [00:14<00:40, 7287.40it/s] 27%|██▋       | 107498/400000 [00:14<00:39, 7329.21it/s] 27%|██▋       | 108232/400000 [00:15<00:39, 7330.59it/s] 27%|██▋       | 108991/400000 [00:15<00:39, 7405.69it/s] 27%|██▋       | 109754/400000 [00:15<00:38, 7469.94it/s] 28%|██▊       | 110512/400000 [00:15<00:38, 7501.52it/s] 28%|██▊       | 111269/400000 [00:15<00:38, 7520.76it/s] 28%|██▊       | 112022/400000 [00:15<00:38, 7407.59it/s] 28%|██▊       | 112764/400000 [00:15<00:39, 7223.42it/s] 28%|██▊       | 113488/400000 [00:15<00:39, 7204.39it/s] 29%|██▊       | 114244/400000 [00:15<00:39, 7306.98it/s] 29%|██▊       | 114993/400000 [00:15<00:38, 7359.48it/s] 29%|██▉       | 115756/400000 [00:16<00:38, 7436.22it/s] 29%|██▉       | 116501/400000 [00:16<00:38, 7403.20it/s] 29%|██▉       | 117251/400000 [00:16<00:38, 7429.69it/s] 30%|██▉       | 118002/400000 [00:16<00:37, 7452.98it/s] 30%|██▉       | 118748/400000 [00:16<00:37, 7449.66it/s] 30%|██▉       | 119507/400000 [00:16<00:37, 7488.67it/s] 30%|███       | 120269/400000 [00:16<00:37, 7526.92it/s] 30%|███       | 121022/400000 [00:16<00:38, 7332.28it/s] 30%|███       | 121772/400000 [00:16<00:37, 7381.48it/s] 31%|███       | 122523/400000 [00:16<00:37, 7417.93it/s] 31%|███       | 123277/400000 [00:17<00:37, 7453.57it/s] 31%|███       | 124037/400000 [00:17<00:36, 7496.60it/s] 31%|███       | 124788/400000 [00:17<00:37, 7425.55it/s] 31%|███▏      | 125532/400000 [00:17<00:37, 7410.50it/s] 32%|███▏      | 126274/400000 [00:17<00:36, 7411.39it/s] 32%|███▏      | 127031/400000 [00:17<00:36, 7457.73it/s] 32%|███▏      | 127791/400000 [00:17<00:36, 7498.54it/s] 32%|███▏      | 128555/400000 [00:17<00:36, 7539.19it/s] 32%|███▏      | 129310/400000 [00:17<00:36, 7477.06it/s] 33%|███▎      | 130068/400000 [00:17<00:35, 7507.00it/s] 33%|███▎      | 130819/400000 [00:18<00:35, 7497.54it/s] 33%|███▎      | 131577/400000 [00:18<00:35, 7519.71it/s] 33%|███▎      | 132341/400000 [00:18<00:35, 7554.89it/s] 33%|███▎      | 133097/400000 [00:18<00:35, 7541.72it/s] 33%|███▎      | 133852/400000 [00:18<00:35, 7399.28it/s] 34%|███▎      | 134600/400000 [00:18<00:35, 7422.21it/s] 34%|███▍      | 135353/400000 [00:18<00:35, 7452.83it/s] 34%|███▍      | 136099/400000 [00:18<00:35, 7449.15it/s] 34%|███▍      | 136847/400000 [00:18<00:35, 7456.51it/s] 34%|███▍      | 137593/400000 [00:18<00:35, 7443.36it/s] 35%|███▍      | 138338/400000 [00:19<00:35, 7333.28it/s] 35%|███▍      | 139097/400000 [00:19<00:35, 7406.31it/s] 35%|███▍      | 139848/400000 [00:19<00:34, 7432.98it/s] 35%|███▌      | 140592/400000 [00:19<00:35, 7397.35it/s] 35%|███▌      | 141336/400000 [00:19<00:34, 7407.04it/s] 36%|███▌      | 142077/400000 [00:19<00:35, 7260.67it/s] 36%|███▌      | 142806/400000 [00:19<00:35, 7268.06it/s] 36%|███▌      | 143540/400000 [00:19<00:35, 7287.56it/s] 36%|███▌      | 144271/400000 [00:19<00:35, 7293.27it/s] 36%|███▋      | 145009/400000 [00:19<00:34, 7317.71it/s] 36%|███▋      | 145741/400000 [00:20<00:34, 7312.75it/s] 37%|███▋      | 146473/400000 [00:20<00:34, 7254.89it/s] 37%|███▋      | 147234/400000 [00:20<00:34, 7357.28it/s] 37%|███▋      | 147995/400000 [00:20<00:33, 7429.77it/s] 37%|███▋      | 148753/400000 [00:20<00:33, 7473.53it/s] 37%|███▋      | 149502/400000 [00:20<00:33, 7475.83it/s] 38%|███▊      | 150250/400000 [00:20<00:33, 7464.92it/s] 38%|███▊      | 150997/400000 [00:20<00:33, 7404.28it/s] 38%|███▊      | 151762/400000 [00:20<00:33, 7476.15it/s] 38%|███▊      | 152511/400000 [00:20<00:33, 7478.56it/s] 38%|███▊      | 153260/400000 [00:21<00:34, 7233.04it/s] 38%|███▊      | 153986/400000 [00:21<00:34, 7084.66it/s] 39%|███▊      | 154697/400000 [00:21<00:34, 7064.87it/s] 39%|███▉      | 155409/400000 [00:21<00:34, 7079.08it/s] 39%|███▉      | 156164/400000 [00:21<00:33, 7213.61it/s] 39%|███▉      | 156919/400000 [00:21<00:33, 7310.51it/s] 39%|███▉      | 157665/400000 [00:21<00:32, 7352.86it/s] 40%|███▉      | 158425/400000 [00:21<00:32, 7422.73it/s] 40%|███▉      | 159169/400000 [00:21<00:32, 7376.42it/s] 40%|███▉      | 159923/400000 [00:22<00:32, 7423.58it/s] 40%|████      | 160666/400000 [00:22<00:32, 7394.49it/s] 40%|████      | 161433/400000 [00:22<00:31, 7471.56it/s] 41%|████      | 162181/400000 [00:22<00:31, 7460.99it/s] 41%|████      | 162941/400000 [00:22<00:31, 7501.27it/s] 41%|████      | 163692/400000 [00:22<00:31, 7431.13it/s] 41%|████      | 164436/400000 [00:22<00:31, 7396.78it/s] 41%|████▏     | 165194/400000 [00:22<00:31, 7448.54it/s] 41%|████▏     | 165953/400000 [00:22<00:31, 7487.72it/s] 42%|████▏     | 166716/400000 [00:22<00:30, 7529.72it/s] 42%|████▏     | 167471/400000 [00:23<00:30, 7533.06it/s] 42%|████▏     | 168225/400000 [00:23<00:31, 7379.97it/s] 42%|████▏     | 168979/400000 [00:23<00:31, 7424.76it/s] 42%|████▏     | 169729/400000 [00:23<00:30, 7445.81it/s] 43%|████▎     | 170475/400000 [00:23<00:30, 7437.36it/s] 43%|████▎     | 171239/400000 [00:23<00:30, 7496.86it/s] 43%|████▎     | 171990/400000 [00:23<00:30, 7389.74it/s] 43%|████▎     | 172734/400000 [00:23<00:30, 7403.31it/s] 43%|████▎     | 173487/400000 [00:23<00:30, 7438.41it/s] 44%|████▎     | 174239/400000 [00:23<00:30, 7460.57it/s] 44%|████▎     | 174992/400000 [00:24<00:30, 7480.63it/s] 44%|████▍     | 175741/400000 [00:24<00:30, 7455.93it/s] 44%|████▍     | 176487/400000 [00:24<00:30, 7398.92it/s] 44%|████▍     | 177228/400000 [00:24<00:30, 7373.25it/s] 44%|████▍     | 177974/400000 [00:24<00:30, 7396.98it/s] 45%|████▍     | 178720/400000 [00:24<00:29, 7413.16it/s] 45%|████▍     | 179462/400000 [00:24<00:30, 7333.39it/s] 45%|████▌     | 180209/400000 [00:24<00:29, 7371.63it/s] 45%|████▌     | 180947/400000 [00:24<00:30, 7272.07it/s] 45%|████▌     | 181675/400000 [00:24<00:30, 7242.31it/s] 46%|████▌     | 182409/400000 [00:25<00:29, 7269.56it/s] 46%|████▌     | 183156/400000 [00:25<00:29, 7326.61it/s] 46%|████▌     | 183889/400000 [00:25<00:29, 7325.05it/s] 46%|████▌     | 184640/400000 [00:25<00:29, 7376.92it/s] 46%|████▋     | 185378/400000 [00:25<00:29, 7318.90it/s] 47%|████▋     | 186142/400000 [00:25<00:28, 7411.88it/s] 47%|████▋     | 186884/400000 [00:25<00:28, 7405.74it/s] 47%|████▋     | 187642/400000 [00:25<00:28, 7455.93it/s] 47%|████▋     | 188402/400000 [00:25<00:28, 7495.44it/s] 47%|████▋     | 189152/400000 [00:25<00:28, 7423.28it/s] 47%|████▋     | 189895/400000 [00:26<00:28, 7353.92it/s] 48%|████▊     | 190642/400000 [00:26<00:28, 7388.25it/s] 48%|████▊     | 191404/400000 [00:26<00:27, 7455.77it/s] 48%|████▊     | 192168/400000 [00:26<00:27, 7507.97it/s] 48%|████▊     | 192920/400000 [00:26<00:27, 7418.48it/s] 48%|████▊     | 193663/400000 [00:26<00:28, 7366.81it/s] 49%|████▊     | 194408/400000 [00:26<00:27, 7390.78it/s] 49%|████▉     | 195169/400000 [00:26<00:27, 7452.73it/s] 49%|████▉     | 195927/400000 [00:26<00:27, 7489.30it/s] 49%|████▉     | 196683/400000 [00:26<00:27, 7509.42it/s] 49%|████▉     | 197440/400000 [00:27<00:26, 7527.21it/s] 50%|████▉     | 198193/400000 [00:27<00:27, 7453.38it/s] 50%|████▉     | 198942/400000 [00:27<00:26, 7463.64it/s] 50%|████▉     | 199698/400000 [00:27<00:26, 7491.08it/s] 50%|█████     | 200452/400000 [00:27<00:26, 7503.44it/s] 50%|█████     | 201209/400000 [00:27<00:26, 7521.68it/s] 50%|█████     | 201962/400000 [00:27<00:26, 7445.36it/s] 51%|█████     | 202707/400000 [00:27<00:26, 7411.57it/s] 51%|█████     | 203463/400000 [00:27<00:26, 7454.34it/s] 51%|█████     | 204209/400000 [00:27<00:26, 7364.85it/s] 51%|█████     | 204968/400000 [00:28<00:26, 7429.04it/s] 51%|█████▏    | 205712/400000 [00:28<00:26, 7288.31it/s] 52%|█████▏    | 206442/400000 [00:28<00:27, 7006.84it/s] 52%|█████▏    | 207146/400000 [00:28<00:27, 6968.78it/s] 52%|█████▏    | 207845/400000 [00:28<00:27, 6948.14it/s] 52%|█████▏    | 208570/400000 [00:28<00:27, 7034.39it/s] 52%|█████▏    | 209313/400000 [00:28<00:26, 7147.48it/s] 53%|█████▎    | 210056/400000 [00:28<00:26, 7228.52it/s] 53%|█████▎    | 210781/400000 [00:28<00:26, 7201.01it/s] 53%|█████▎    | 211521/400000 [00:29<00:25, 7257.67it/s] 53%|█████▎    | 212259/400000 [00:29<00:25, 7293.05it/s] 53%|█████▎    | 212994/400000 [00:29<00:25, 7308.33it/s] 53%|█████▎    | 213727/400000 [00:29<00:25, 7312.43it/s] 54%|█████▎    | 214459/400000 [00:29<00:25, 7313.42it/s] 54%|█████▍    | 215191/400000 [00:29<00:25, 7237.57it/s] 54%|█████▍    | 215931/400000 [00:29<00:25, 7282.91it/s] 54%|█████▍    | 216660/400000 [00:29<00:25, 7274.10it/s] 54%|█████▍    | 217388/400000 [00:29<00:25, 7254.43it/s] 55%|█████▍    | 218130/400000 [00:29<00:24, 7302.60it/s] 55%|█████▍    | 218877/400000 [00:30<00:24, 7350.06it/s] 55%|█████▍    | 219613/400000 [00:30<00:24, 7312.33it/s] 55%|█████▌    | 220357/400000 [00:30<00:24, 7348.85it/s] 55%|█████▌    | 221106/400000 [00:30<00:24, 7388.55it/s] 55%|█████▌    | 221846/400000 [00:30<00:24, 7279.02it/s] 56%|█████▌    | 222600/400000 [00:30<00:24, 7355.30it/s] 56%|█████▌    | 223350/400000 [00:30<00:23, 7395.01it/s] 56%|█████▌    | 224090/400000 [00:30<00:24, 7265.27it/s] 56%|█████▌    | 224846/400000 [00:30<00:23, 7348.69it/s] 56%|█████▋    | 225612/400000 [00:30<00:23, 7438.39it/s] 57%|█████▋    | 226365/400000 [00:31<00:23, 7462.76it/s] 57%|█████▋    | 227112/400000 [00:31<00:23, 7456.07it/s] 57%|█████▋    | 227859/400000 [00:31<00:23, 7336.79it/s] 57%|█████▋    | 228606/400000 [00:31<00:23, 7373.29it/s] 57%|█████▋    | 229355/400000 [00:31<00:23, 7407.64it/s] 58%|█████▊    | 230106/400000 [00:31<00:22, 7436.42it/s] 58%|█████▊    | 230861/400000 [00:31<00:22, 7468.09it/s] 58%|█████▊    | 231611/400000 [00:31<00:22, 7476.11it/s] 58%|█████▊    | 232359/400000 [00:31<00:23, 7200.10it/s] 58%|█████▊    | 233115/400000 [00:31<00:22, 7303.42it/s] 58%|█████▊    | 233853/400000 [00:32<00:22, 7324.57it/s] 59%|█████▊    | 234589/400000 [00:32<00:22, 7333.90it/s] 59%|█████▉    | 235328/400000 [00:32<00:22, 7349.73it/s] 59%|█████▉    | 236065/400000 [00:32<00:22, 7352.36it/s] 59%|█████▉    | 236801/400000 [00:32<00:22, 7311.49it/s] 59%|█████▉    | 237554/400000 [00:32<00:22, 7375.63it/s] 60%|█████▉    | 238304/400000 [00:32<00:21, 7410.23it/s] 60%|█████▉    | 239046/400000 [00:32<00:21, 7408.27it/s] 60%|█████▉    | 239788/400000 [00:32<00:21, 7399.01it/s] 60%|██████    | 240529/400000 [00:32<00:21, 7344.74it/s] 60%|██████    | 241278/400000 [00:33<00:21, 7386.96it/s] 61%|██████    | 242028/400000 [00:33<00:21, 7418.33it/s] 61%|██████    | 242773/400000 [00:33<00:21, 7427.34it/s] 61%|██████    | 243516/400000 [00:33<00:21, 7427.33it/s] 61%|██████    | 244280/400000 [00:33<00:20, 7487.16it/s] 61%|██████▏   | 245029/400000 [00:33<00:21, 7359.25it/s] 61%|██████▏   | 245780/400000 [00:33<00:20, 7403.24it/s] 62%|██████▏   | 246521/400000 [00:33<00:20, 7402.09it/s] 62%|██████▏   | 247262/400000 [00:33<00:20, 7395.82it/s] 62%|██████▏   | 248013/400000 [00:33<00:20, 7429.17it/s] 62%|██████▏   | 248757/400000 [00:34<00:20, 7430.21it/s] 62%|██████▏   | 249501/400000 [00:34<00:20, 7356.11it/s] 63%|██████▎   | 250242/400000 [00:34<00:20, 7372.11it/s] 63%|██████▎   | 250982/400000 [00:34<00:20, 7379.55it/s] 63%|██████▎   | 251726/400000 [00:34<00:20, 7394.93it/s] 63%|██████▎   | 252478/400000 [00:34<00:19, 7429.85it/s] 63%|██████▎   | 253222/400000 [00:34<00:19, 7403.81it/s] 63%|██████▎   | 253963/400000 [00:34<00:19, 7320.31it/s] 64%|██████▎   | 254696/400000 [00:34<00:19, 7299.00it/s] 64%|██████▍   | 255432/400000 [00:34<00:19, 7316.36it/s] 64%|██████▍   | 256182/400000 [00:35<00:19, 7368.22it/s] 64%|██████▍   | 256924/400000 [00:35<00:19, 7382.78it/s] 64%|██████▍   | 257663/400000 [00:35<00:19, 7266.41it/s] 65%|██████▍   | 258407/400000 [00:35<00:19, 7315.13it/s] 65%|██████▍   | 259175/400000 [00:35<00:18, 7419.13it/s] 65%|██████▍   | 259941/400000 [00:35<00:18, 7487.84it/s] 65%|██████▌   | 260698/400000 [00:35<00:18, 7511.28it/s] 65%|██████▌   | 261450/400000 [00:35<00:18, 7509.82it/s] 66%|██████▌   | 262202/400000 [00:35<00:18, 7396.88it/s] 66%|██████▌   | 262956/400000 [00:35<00:18, 7436.36it/s] 66%|██████▌   | 263712/400000 [00:36<00:18, 7470.99it/s] 66%|██████▌   | 264468/400000 [00:36<00:18, 7496.95it/s] 66%|██████▋   | 265225/400000 [00:36<00:17, 7516.19it/s] 66%|██████▋   | 265977/400000 [00:36<00:17, 7485.49it/s] 67%|██████▋   | 266726/400000 [00:36<00:18, 7365.94it/s] 67%|██████▋   | 267480/400000 [00:36<00:17, 7415.08it/s] 67%|██████▋   | 268238/400000 [00:36<00:17, 7461.66it/s] 67%|██████▋   | 268996/400000 [00:36<00:17, 7496.30it/s] 67%|██████▋   | 269746/400000 [00:36<00:17, 7493.63it/s] 68%|██████▊   | 270496/400000 [00:36<00:17, 7430.29it/s] 68%|██████▊   | 271253/400000 [00:37<00:17, 7471.49it/s] 68%|██████▊   | 272007/400000 [00:37<00:17, 7489.96it/s] 68%|██████▊   | 272766/400000 [00:37<00:16, 7517.77it/s] 68%|██████▊   | 273518/400000 [00:37<00:16, 7508.96it/s] 69%|██████▊   | 274270/400000 [00:37<00:16, 7486.00it/s] 69%|██████▉   | 275019/400000 [00:37<00:16, 7393.09it/s] 69%|██████▉   | 275777/400000 [00:37<00:16, 7448.14it/s] 69%|██████▉   | 276529/400000 [00:37<00:16, 7466.67it/s] 69%|██████▉   | 277278/400000 [00:37<00:16, 7472.54it/s] 70%|██████▉   | 278026/400000 [00:38<00:16, 7472.26it/s] 70%|██████▉   | 278783/400000 [00:38<00:16, 7500.33it/s] 70%|██████▉   | 279534/400000 [00:38<00:16, 7381.33it/s] 70%|███████   | 280288/400000 [00:38<00:16, 7425.65it/s] 70%|███████   | 281031/400000 [00:38<00:16, 7381.06it/s] 70%|███████   | 281787/400000 [00:38<00:15, 7431.57it/s] 71%|███████   | 282543/400000 [00:38<00:15, 7467.50it/s] 71%|███████   | 283291/400000 [00:38<00:15, 7410.27it/s] 71%|███████   | 284033/400000 [00:38<00:15, 7354.98it/s] 71%|███████   | 284774/400000 [00:38<00:15, 7369.85it/s] 71%|███████▏  | 285518/400000 [00:39<00:15, 7389.74it/s] 72%|███████▏  | 286275/400000 [00:39<00:15, 7442.03it/s] 72%|███████▏  | 287020/400000 [00:39<00:15, 7438.57it/s] 72%|███████▏  | 287764/400000 [00:39<00:15, 7341.25it/s] 72%|███████▏  | 288499/400000 [00:39<00:15, 7330.56it/s] 72%|███████▏  | 289236/400000 [00:39<00:15, 7340.32it/s] 72%|███████▏  | 289977/400000 [00:39<00:14, 7359.46it/s] 73%|███████▎  | 290714/400000 [00:39<00:14, 7329.53it/s] 73%|███████▎  | 291454/400000 [00:39<00:14, 7348.75it/s] 73%|███████▎  | 292189/400000 [00:39<00:15, 7176.47it/s] 73%|███████▎  | 292928/400000 [00:40<00:14, 7238.59it/s] 73%|███████▎  | 293672/400000 [00:40<00:14, 7296.19it/s] 74%|███████▎  | 294417/400000 [00:40<00:14, 7339.58it/s] 74%|███████▍  | 295167/400000 [00:40<00:14, 7386.24it/s] 74%|███████▍  | 295908/400000 [00:40<00:14, 7393.15it/s] 74%|███████▍  | 296648/400000 [00:40<00:14, 7369.12it/s] 74%|███████▍  | 297394/400000 [00:40<00:13, 7394.27it/s] 75%|███████▍  | 298144/400000 [00:40<00:13, 7422.36it/s] 75%|███████▍  | 298887/400000 [00:40<00:13, 7379.20it/s] 75%|███████▍  | 299635/400000 [00:40<00:13, 7408.23it/s] 75%|███████▌  | 300376/400000 [00:41<00:13, 7314.92it/s] 75%|███████▌  | 301137/400000 [00:41<00:13, 7400.71it/s] 75%|███████▌  | 301889/400000 [00:41<00:13, 7433.56it/s] 76%|███████▌  | 302648/400000 [00:41<00:13, 7477.46it/s] 76%|███████▌  | 303397/400000 [00:41<00:12, 7466.97it/s] 76%|███████▌  | 304144/400000 [00:41<00:12, 7460.02it/s] 76%|███████▌  | 304891/400000 [00:41<00:12, 7448.42it/s] 76%|███████▋  | 305636/400000 [00:41<00:12, 7377.38it/s] 77%|███████▋  | 306374/400000 [00:41<00:13, 7201.15it/s] 77%|███████▋  | 307096/400000 [00:41<00:13, 7057.04it/s] 77%|███████▋  | 307804/400000 [00:42<00:13, 6969.10it/s] 77%|███████▋  | 308503/400000 [00:42<00:13, 6933.99it/s] 77%|███████▋  | 309198/400000 [00:42<00:13, 6799.71it/s] 77%|███████▋  | 309880/400000 [00:42<00:13, 6761.02it/s] 78%|███████▊  | 310566/400000 [00:42<00:13, 6787.76it/s] 78%|███████▊  | 311246/400000 [00:42<00:13, 6745.33it/s] 78%|███████▊  | 311934/400000 [00:42<00:12, 6784.36it/s] 78%|███████▊  | 312623/400000 [00:42<00:12, 6813.35it/s] 78%|███████▊  | 313344/400000 [00:42<00:12, 6927.29it/s] 79%|███████▊  | 314092/400000 [00:42<00:12, 7084.16it/s] 79%|███████▊  | 314856/400000 [00:43<00:11, 7241.95it/s] 79%|███████▉  | 315611/400000 [00:43<00:11, 7329.51it/s] 79%|███████▉  | 316351/400000 [00:43<00:11, 7349.17it/s] 79%|███████▉  | 317087/400000 [00:43<00:11, 7275.72it/s] 79%|███████▉  | 317816/400000 [00:43<00:11, 7207.40it/s] 80%|███████▉  | 318568/400000 [00:43<00:11, 7296.47it/s] 80%|███████▉  | 319315/400000 [00:43<00:10, 7341.04it/s] 80%|████████  | 320057/400000 [00:43<00:10, 7362.47it/s] 80%|████████  | 320802/400000 [00:43<00:10, 7387.15it/s] 80%|████████  | 321542/400000 [00:43<00:10, 7350.81it/s] 81%|████████  | 322278/400000 [00:44<00:10, 7287.84it/s] 81%|████████  | 323008/400000 [00:44<00:10, 7246.83it/s] 81%|████████  | 323733/400000 [00:44<00:10, 7161.72it/s] 81%|████████  | 324463/400000 [00:44<00:10, 7201.65it/s] 81%|████████▏ | 325202/400000 [00:44<00:10, 7255.80it/s] 81%|████████▏ | 325942/400000 [00:44<00:10, 7296.14it/s] 82%|████████▏ | 326685/400000 [00:44<00:09, 7333.31it/s] 82%|████████▏ | 327434/400000 [00:44<00:09, 7378.39it/s] 82%|████████▏ | 328173/400000 [00:44<00:09, 7370.42it/s] 82%|████████▏ | 328911/400000 [00:44<00:09, 7361.23it/s] 82%|████████▏ | 329666/400000 [00:45<00:09, 7416.48it/s] 83%|████████▎ | 330408/400000 [00:45<00:09, 7403.52it/s] 83%|████████▎ | 331159/400000 [00:45<00:09, 7432.36it/s] 83%|████████▎ | 331905/400000 [00:45<00:09, 7440.21it/s] 83%|████████▎ | 332655/400000 [00:45<00:09, 7457.19it/s] 83%|████████▎ | 333416/400000 [00:45<00:08, 7501.09it/s] 84%|████████▎ | 334173/400000 [00:45<00:08, 7520.86it/s] 84%|████████▎ | 334926/400000 [00:45<00:08, 7465.80it/s] 84%|████████▍ | 335673/400000 [00:45<00:08, 7446.70it/s] 84%|████████▍ | 336418/400000 [00:46<00:08, 7430.51it/s] 84%|████████▍ | 337177/400000 [00:46<00:08, 7477.46it/s] 84%|████████▍ | 337934/400000 [00:46<00:08, 7504.10it/s] 85%|████████▍ | 338685/400000 [00:46<00:08, 7465.26it/s] 85%|████████▍ | 339432/400000 [00:46<00:08, 7443.02it/s] 85%|████████▌ | 340178/400000 [00:46<00:08, 7447.66it/s] 85%|████████▌ | 340928/400000 [00:46<00:07, 7459.27it/s] 85%|████████▌ | 341674/400000 [00:46<00:07, 7459.17it/s] 86%|████████▌ | 342420/400000 [00:46<00:07, 7244.84it/s] 86%|████████▌ | 343146/400000 [00:46<00:08, 7041.56it/s] 86%|████████▌ | 343892/400000 [00:47<00:07, 7159.53it/s] 86%|████████▌ | 344633/400000 [00:47<00:07, 7232.21it/s] 86%|████████▋ | 345374/400000 [00:47<00:07, 7281.32it/s] 87%|████████▋ | 346128/400000 [00:47<00:07, 7355.12it/s] 87%|████████▋ | 346867/400000 [00:47<00:07, 7361.44it/s] 87%|████████▋ | 347604/400000 [00:47<00:07, 7303.95it/s] 87%|████████▋ | 348360/400000 [00:47<00:07, 7376.49it/s] 87%|████████▋ | 349114/400000 [00:47<00:06, 7422.07it/s] 87%|████████▋ | 349870/400000 [00:47<00:06, 7460.11it/s] 88%|████████▊ | 350617/400000 [00:47<00:06, 7462.51it/s] 88%|████████▊ | 351364/400000 [00:48<00:06, 7454.00it/s] 88%|████████▊ | 352110/400000 [00:48<00:06, 7443.23it/s] 88%|████████▊ | 352855/400000 [00:48<00:06, 7373.12it/s] 88%|████████▊ | 353593/400000 [00:48<00:06, 7313.13it/s] 89%|████████▊ | 354333/400000 [00:48<00:06, 7337.07it/s] 89%|████████▉ | 355077/400000 [00:48<00:06, 7367.46it/s] 89%|████████▉ | 355814/400000 [00:48<00:06, 7360.77it/s] 89%|████████▉ | 356568/400000 [00:48<00:05, 7413.25it/s] 89%|████████▉ | 357318/400000 [00:48<00:05, 7437.45it/s] 90%|████████▉ | 358062/400000 [00:48<00:05, 7434.58it/s] 90%|████████▉ | 358806/400000 [00:49<00:05, 7431.86it/s] 90%|████████▉ | 359556/400000 [00:49<00:05, 7450.77it/s] 90%|█████████ | 360302/400000 [00:49<00:05, 7367.56it/s] 90%|█████████ | 361051/400000 [00:49<00:05, 7403.74it/s] 90%|█████████ | 361793/400000 [00:49<00:05, 7407.41it/s] 91%|█████████ | 362534/400000 [00:49<00:05, 7406.65it/s] 91%|█████████ | 363279/400000 [00:49<00:04, 7419.11it/s] 91%|█████████ | 364021/400000 [00:49<00:04, 7356.55it/s] 91%|█████████ | 364757/400000 [00:49<00:04, 7328.75it/s] 91%|█████████▏| 365491/400000 [00:49<00:04, 7279.32it/s] 92%|█████████▏| 366230/400000 [00:50<00:04, 7309.97it/s] 92%|█████████▏| 366979/400000 [00:50<00:04, 7361.93it/s] 92%|█████████▏| 367720/400000 [00:50<00:04, 7373.31it/s] 92%|█████████▏| 368458/400000 [00:50<00:04, 7337.13it/s] 92%|█████████▏| 369192/400000 [00:50<00:04, 7332.46it/s] 92%|█████████▏| 369936/400000 [00:50<00:04, 7362.07it/s] 93%|█████████▎| 370685/400000 [00:50<00:03, 7398.16it/s] 93%|█████████▎| 371437/400000 [00:50<00:03, 7434.19it/s] 93%|█████████▎| 372187/400000 [00:50<00:03, 7452.46it/s] 93%|█████████▎| 372933/400000 [00:50<00:03, 7374.44it/s] 93%|█████████▎| 373677/400000 [00:51<00:03, 7391.57it/s] 94%|█████████▎| 374425/400000 [00:51<00:03, 7414.68it/s] 94%|█████████▍| 375179/400000 [00:51<00:03, 7449.11it/s] 94%|█████████▍| 375925/400000 [00:51<00:03, 7372.09it/s] 94%|█████████▍| 376679/400000 [00:51<00:03, 7419.71it/s] 94%|█████████▍| 377422/400000 [00:51<00:03, 7361.71it/s] 95%|█████████▍| 378167/400000 [00:51<00:02, 7386.68it/s] 95%|█████████▍| 378922/400000 [00:51<00:02, 7434.89it/s] 95%|█████████▍| 379675/400000 [00:51<00:02, 7462.73it/s] 95%|█████████▌| 380433/400000 [00:51<00:02, 7495.01it/s] 95%|█████████▌| 381183/400000 [00:52<00:02, 7452.10it/s] 95%|█████████▌| 381929/400000 [00:52<00:02, 7411.86it/s] 96%|█████████▌| 382682/400000 [00:52<00:02, 7445.35it/s] 96%|█████████▌| 383427/400000 [00:52<00:02, 7406.08it/s] 96%|█████████▌| 384168/400000 [00:52<00:02, 7399.00it/s] 96%|█████████▌| 384909/400000 [00:52<00:02, 7372.71it/s] 96%|█████████▋| 385647/400000 [00:52<00:01, 7322.93it/s] 97%|█████████▋| 386391/400000 [00:52<00:01, 7356.85it/s] 97%|█████████▋| 387141/400000 [00:52<00:01, 7398.71it/s] 97%|█████████▋| 387896/400000 [00:52<00:01, 7443.16it/s] 97%|█████████▋| 388643/400000 [00:53<00:01, 7450.38it/s] 97%|█████████▋| 389392/400000 [00:53<00:01, 7461.21it/s] 98%|█████████▊| 390139/400000 [00:53<00:01, 7216.22it/s] 98%|█████████▊| 390867/400000 [00:53<00:01, 7235.05it/s] 98%|█████████▊| 391617/400000 [00:53<00:01, 7310.42it/s] 98%|█████████▊| 392368/400000 [00:53<00:01, 7367.12it/s] 98%|█████████▊| 393121/400000 [00:53<00:00, 7415.15it/s] 98%|█████████▊| 393864/400000 [00:53<00:00, 7314.24it/s] 99%|█████████▊| 394603/400000 [00:53<00:00, 7335.03it/s] 99%|█████████▉| 395360/400000 [00:53<00:00, 7401.62it/s] 99%|█████████▉| 396106/400000 [00:54<00:00, 7418.20it/s] 99%|█████████▉| 396852/400000 [00:54<00:00, 7428.23it/s] 99%|█████████▉| 397602/400000 [00:54<00:00, 7448.46it/s]100%|█████████▉| 398348/400000 [00:54<00:00, 7367.65it/s]100%|█████████▉| 399086/400000 [00:54<00:00, 7324.03it/s]100%|█████████▉| 399819/400000 [00:54<00:00, 7072.90it/s]100%|█████████▉| 399999/400000 [00:54<00:00, 7321.88it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7fb58f5e10f0>, <torchtext.data.dataset.TabularDataset object at 0x7fb58f5e1240>, <torchtext.vocab.Vocab object at 0x7fb58f5e1160>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 10.71 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 10.57 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 10.57 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  4.60 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  4.60 file/s]Dl Completed...: 100%|██████████| 4/4 [00:01<00:00,  3.69 file/s]Dl Completed...: 100%|██████████| 4/4 [00:01<00:00,  3.69 file/s]Dl Completed...: 100%|██████████| 4/4 [00:01<00:00,  3.67 file/s]2020-06-07 12:19:40.301398: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-07 12:19:40.305985: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-06-07 12:19:40.306169: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x560598ea6bc0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-07 12:19:40.306186: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
[1mDownloading and preparing dataset mnist/3.0.1 (download: 11.06 MiB, generated: 21.00 MiB, total: 32.06 MiB) to /home/runner/tensorflow_datasets/mnist/3.0.1...[0m

[1mDataset mnist downloaded and prepared to /home/runner/tensorflow_datasets/mnist/3.0.1. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/ ['train', 'test', 'mnist2', 'cifar10', 'mnist_dataset_small.npy', 'fashion-mnist_small.npy'] 

  


 #################### get_dataset_torch 

  get_dataset_torch mlmodels/preprocess/generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 36%|███▋      | 3612672/9912422 [00:00<00:00, 36054718.94it/s]9920512it [00:00, 34510604.71it/s]                             
0it [00:00, ?it/s]32768it [00:00, 599424.09it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:11, 146000.11it/s]1654784it [00:00, 10910133.90it/s]                         
0it [00:00, ?it/s]8192it [00:00, 180834.07it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
