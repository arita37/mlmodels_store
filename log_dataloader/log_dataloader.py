
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '2675f1e090030e6958e45c46c6313291532e6ed8', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/2675f1e090030e6958e45c46c6313291532e6ed8

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f52bc5c66a8> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f52bc5c66a8> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f5327b8e0d0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f5327b8e0d0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f53418bbea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f53418bbea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f52d5406950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f52d5406950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f52d5406950> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s]  7%|▋         | 655360/9912422 [00:00<00:01, 6428268.58it/s] 25%|██▍       | 2441216/9912422 [00:00<00:00, 7906636.53it/s] 66%|██████▋   | 6578176/9912422 [00:00<00:00, 10439959.20it/s]9920512it [00:00, 20523161.71it/s]                             
0it [00:00, ?it/s]32768it [00:00, 455765.94it/s]
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]1654784it [00:00, 7496478.56it/s]          
0it [00:00, ?it/s]8192it [00:00, 162166.03it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f52d2637240>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f52d294fdd8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f52d5406598> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f52d5406598> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f52d5406598> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<00:53,  2.99 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<00:53,  2.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<00:53,  2.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<00:53,  2.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<00:52,  2.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<00:52,  2.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<00:52,  2.99 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▍         | 7/162 [00:00<00:37,  4.18 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:37,  4.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:36,  4.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:36,  4.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:36,  4.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:36,  4.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:35,  4.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:35,  4.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:35,  4.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   9%|▉         | 15/162 [00:00<00:25,  5.82 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:25,  5.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:25,  5.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:24,  5.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:24,  5.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:24,  5.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:24,  5.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:24,  5.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:24,  5.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  14%|█▍        | 23/162 [00:00<00:17,  8.04 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:17,  8.04 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:17,  8.04 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:17,  8.04 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:16,  8.04 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:16,  8.04 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:16,  8.04 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:16,  8.04 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:16,  8.04 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  19%|█▉        | 31/162 [00:00<00:11, 10.98 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:11, 10.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:11, 10.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:11, 10.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:11, 10.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:11, 10.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:11, 10.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:11, 10.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:11, 10.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  24%|██▍       | 39/162 [00:00<00:08, 14.75 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:08, 14.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:08, 14.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:08, 14.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:08, 14.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:08, 14.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:00<00:07, 14.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:00<00:07, 14.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:00<00:07, 14.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  29%|██▉       | 47/162 [00:00<00:05, 19.39 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:00<00:05, 19.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:00<00:05, 19.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:05, 19.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:05, 19.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:05, 19.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:05, 19.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:05, 19.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  33%|███▎      | 54/162 [00:01<00:04, 24.64 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:04, 24.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:04, 24.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:04, 24.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:04, 24.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:04, 24.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:04, 24.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:04, 24.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:04, 24.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:04, 24.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  39%|███▉      | 63/162 [00:01<00:03, 31.15 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:03, 31.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:03, 31.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:03, 31.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:03, 31.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 31.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:03, 31.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:02, 31.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:02, 31.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 38.08 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 38.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 38.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 38.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 38.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 38.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 38.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 38.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 38.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  49%|████▉     | 79/162 [00:01<00:01, 44.86 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:01, 44.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:01, 44.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:01, 44.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:01, 44.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:01, 44.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:01, 44.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 44.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 44.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 51.36 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 51.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 51.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 51.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 51.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 51.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 51.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 51.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 51.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 55.92 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 55.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 55.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 55.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 55.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 55.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 55.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 55.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 55.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:00, 61.19 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:00, 61.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:00, 61.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:00, 61.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:00, 61.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:00, 61.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:00, 61.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:00, 61.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 61.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 65.44 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 65.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 65.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 65.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 65.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 65.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 65.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 65.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 65.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 68.46 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 68.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 68.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 68.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 68.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 68.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 68.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 68.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 68.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 70.23 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 70.23 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 70.23 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 70.23 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 70.23 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 70.23 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 70.23 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 70.23 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 70.23 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 71.77 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 71.77 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 71.77 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 71.77 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 71.77 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 71.77 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 71.77 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 71.77 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 71.77 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 73.67 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 73.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 73.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 73.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 73.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 73.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 73.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 73.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 73.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 73.75 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 73.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 73.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 73.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 73.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 73.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 73.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 73.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 73.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 74.04 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 74.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 74.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 74.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 74.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.50s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.50s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 74.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.50s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 74.04 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.73s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.50s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 74.04 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.73s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.73s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 34.21 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.74s/ url]
0 examples [00:00, ? examples/s]2020-06-07 18:10:57.685629: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-07 18:10:57.733118: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-06-07 18:10:57.733344: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55b92c620710 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-07 18:10:57.733379: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  1.00 examples/s]91 examples [00:01,  1.43 examples/s]190 examples [00:01,  2.04 examples/s]289 examples [00:01,  2.91 examples/s]391 examples [00:01,  4.16 examples/s]486 examples [00:01,  5.92 examples/s]572 examples [00:01,  8.44 examples/s]657 examples [00:01, 12.00 examples/s]744 examples [00:01, 17.05 examples/s]840 examples [00:01, 24.17 examples/s]939 examples [00:02, 34.17 examples/s]1035 examples [00:02, 48.08 examples/s]1129 examples [00:02, 67.20 examples/s]1226 examples [00:02, 93.22 examples/s]1325 examples [00:02, 127.97 examples/s]1422 examples [00:02, 172.96 examples/s]1522 examples [00:02, 229.93 examples/s]1622 examples [00:02, 298.94 examples/s]1721 examples [00:02, 377.91 examples/s]1819 examples [00:02, 462.37 examples/s]1917 examples [00:03, 545.15 examples/s]2014 examples [00:03, 625.59 examples/s]2116 examples [00:03, 706.84 examples/s]2214 examples [00:03, 762.80 examples/s]2311 examples [00:03, 804.93 examples/s]2407 examples [00:03, 831.19 examples/s]2501 examples [00:03, 850.79 examples/s]2601 examples [00:03, 889.51 examples/s]2696 examples [00:03, 903.86 examples/s]2792 examples [00:03, 918.03 examples/s]2887 examples [00:04, 912.30 examples/s]2981 examples [00:04, 896.23 examples/s]3075 examples [00:04, 906.87 examples/s]3171 examples [00:04, 919.75 examples/s]3270 examples [00:04, 938.93 examples/s]3365 examples [00:04, 932.90 examples/s]3459 examples [00:04, 933.97 examples/s]3554 examples [00:04, 935.49 examples/s]3653 examples [00:04, 949.87 examples/s]3749 examples [00:04, 936.67 examples/s]3843 examples [00:05, 923.14 examples/s]3936 examples [00:05, 922.30 examples/s]4033 examples [00:05, 934.31 examples/s]4131 examples [00:05, 946.60 examples/s]4227 examples [00:05, 949.31 examples/s]4326 examples [00:05, 960.94 examples/s]4423 examples [00:05, 956.85 examples/s]4520 examples [00:05, 960.63 examples/s]4620 examples [00:05, 970.77 examples/s]4721 examples [00:05, 979.76 examples/s]4820 examples [00:06, 964.51 examples/s]4921 examples [00:06, 976.51 examples/s]5024 examples [00:06, 991.83 examples/s]5124 examples [00:06, 994.17 examples/s]5225 examples [00:06, 998.09 examples/s]5325 examples [00:06, 993.46 examples/s]5425 examples [00:06, 994.23 examples/s]5525 examples [00:06, 994.26 examples/s]5626 examples [00:06, 998.49 examples/s]5728 examples [00:06, 1003.38 examples/s]5830 examples [00:07, 1007.10 examples/s]5931 examples [00:07, 979.85 examples/s] 6030 examples [00:07, 981.19 examples/s]6129 examples [00:07, 975.29 examples/s]6227 examples [00:07, 968.77 examples/s]6324 examples [00:07, 966.96 examples/s]6421 examples [00:07, 955.46 examples/s]6517 examples [00:07, 955.38 examples/s]6618 examples [00:07, 970.38 examples/s]6716 examples [00:08, 926.22 examples/s]6810 examples [00:08, 925.89 examples/s]6911 examples [00:08, 947.16 examples/s]7011 examples [00:08, 962.02 examples/s]7109 examples [00:08, 966.11 examples/s]7207 examples [00:08, 969.67 examples/s]7313 examples [00:08, 993.11 examples/s]7416 examples [00:08, 1003.30 examples/s]7518 examples [00:08, 1007.68 examples/s]7619 examples [00:08, 1003.41 examples/s]7720 examples [00:09, 934.74 examples/s] 7824 examples [00:09, 962.98 examples/s]7925 examples [00:09, 974.73 examples/s]8026 examples [00:09, 983.37 examples/s]8127 examples [00:09, 990.57 examples/s]8227 examples [00:09, 972.64 examples/s]8326 examples [00:09, 975.19 examples/s]8427 examples [00:09, 984.27 examples/s]8529 examples [00:09, 992.08 examples/s]8629 examples [00:09, 994.18 examples/s]8729 examples [00:10, 972.29 examples/s]8827 examples [00:10, 970.01 examples/s]8926 examples [00:10, 973.31 examples/s]9024 examples [00:10, 972.24 examples/s]9124 examples [00:10, 978.54 examples/s]9223 examples [00:10, 981.04 examples/s]9323 examples [00:10, 984.84 examples/s]9422 examples [00:10, 953.63 examples/s]9518 examples [00:10, 942.42 examples/s]9618 examples [00:11, 956.98 examples/s]9716 examples [00:11, 960.64 examples/s]9816 examples [00:11, 971.74 examples/s]9914 examples [00:11, 964.81 examples/s]10011 examples [00:11, 904.01 examples/s]10109 examples [00:11, 922.78 examples/s]10202 examples [00:11, 918.94 examples/s]10295 examples [00:11, 920.92 examples/s]10396 examples [00:11, 945.41 examples/s]10500 examples [00:11, 971.85 examples/s]10598 examples [00:12, 971.69 examples/s]10696 examples [00:12, 952.00 examples/s]10796 examples [00:12, 964.65 examples/s]10897 examples [00:12, 975.11 examples/s]11000 examples [00:12, 990.24 examples/s]11104 examples [00:12, 1004.27 examples/s]11205 examples [00:12, 993.46 examples/s] 11305 examples [00:12, 994.71 examples/s]11405 examples [00:12, 961.84 examples/s]11506 examples [00:12, 975.58 examples/s]11604 examples [00:13, 973.47 examples/s]11702 examples [00:13, 958.51 examples/s]11803 examples [00:13, 970.91 examples/s]11901 examples [00:13, 948.23 examples/s]11997 examples [00:13, 942.56 examples/s]12098 examples [00:13, 959.24 examples/s]12195 examples [00:13, 947.31 examples/s]12290 examples [00:13, 943.92 examples/s]12390 examples [00:13, 958.10 examples/s]12487 examples [00:13, 958.86 examples/s]12586 examples [00:14, 967.48 examples/s]12683 examples [00:14, 967.64 examples/s]12780 examples [00:14, 965.82 examples/s]12877 examples [00:14, 965.34 examples/s]12984 examples [00:14, 993.93 examples/s]13084 examples [00:14, 986.21 examples/s]13187 examples [00:14, 998.13 examples/s]13289 examples [00:14, 1003.19 examples/s]13390 examples [00:14, 1003.22 examples/s]13491 examples [00:14, 1000.24 examples/s]13592 examples [00:15, 987.45 examples/s] 13691 examples [00:15, 966.01 examples/s]13792 examples [00:15, 976.18 examples/s]13891 examples [00:15, 978.84 examples/s]13990 examples [00:15, 981.22 examples/s]14091 examples [00:15, 986.99 examples/s]14191 examples [00:15, 990.82 examples/s]14294 examples [00:15, 1000.60 examples/s]14398 examples [00:15, 1009.16 examples/s]14499 examples [00:16, 999.03 examples/s] 14600 examples [00:16, 1002.17 examples/s]14701 examples [00:16, 1003.14 examples/s]14802 examples [00:16, 1003.36 examples/s]14903 examples [00:16, 999.73 examples/s] 15006 examples [00:16, 1006.61 examples/s]15111 examples [00:16, 1017.88 examples/s]15213 examples [00:16, 995.87 examples/s] 15313 examples [00:16, 988.81 examples/s]15416 examples [00:16, 999.85 examples/s]15517 examples [00:17, 995.45 examples/s]15617 examples [00:17, 966.46 examples/s]15715 examples [00:17, 967.92 examples/s]15813 examples [00:17, 970.32 examples/s]15911 examples [00:17, 938.52 examples/s]16014 examples [00:17, 963.72 examples/s]16114 examples [00:17, 973.12 examples/s]16217 examples [00:17, 987.82 examples/s]16317 examples [00:17, 990.28 examples/s]16417 examples [00:17, 988.55 examples/s]16519 examples [00:18, 994.93 examples/s]16619 examples [00:18, 987.10 examples/s]16718 examples [00:18, 987.39 examples/s]16817 examples [00:18, 972.60 examples/s]16915 examples [00:18, 938.50 examples/s]17021 examples [00:18, 971.20 examples/s]17119 examples [00:18, 968.11 examples/s]17217 examples [00:18, 960.76 examples/s]17314 examples [00:18, 960.07 examples/s]17411 examples [00:18, 958.92 examples/s]17508 examples [00:19, 958.96 examples/s]17604 examples [00:19, 942.05 examples/s]17699 examples [00:19, 932.49 examples/s]17795 examples [00:19, 938.19 examples/s]17889 examples [00:19, 921.89 examples/s]17988 examples [00:19, 938.88 examples/s]18083 examples [00:19, 938.65 examples/s]18180 examples [00:19, 945.62 examples/s]18281 examples [00:19, 961.70 examples/s]18380 examples [00:20, 969.07 examples/s]18478 examples [00:20, 957.52 examples/s]18574 examples [00:20, 951.43 examples/s]18675 examples [00:20, 966.33 examples/s]18774 examples [00:20, 971.85 examples/s]18873 examples [00:20, 975.46 examples/s]18975 examples [00:20, 985.31 examples/s]19074 examples [00:20, 969.07 examples/s]19176 examples [00:20, 983.51 examples/s]19279 examples [00:20, 994.16 examples/s]19383 examples [00:21, 1004.85 examples/s]19484 examples [00:21, 1004.54 examples/s]19585 examples [00:21, 994.28 examples/s] 19685 examples [00:21, 994.66 examples/s]19789 examples [00:21, 1005.77 examples/s]19892 examples [00:21, 1012.83 examples/s]19996 examples [00:21, 1019.92 examples/s]20099 examples [00:21, 910.37 examples/s] 20202 examples [00:21, 941.37 examples/s]20308 examples [00:21, 972.64 examples/s]20408 examples [00:22, 979.71 examples/s]20508 examples [00:22, 981.96 examples/s]20607 examples [00:22, 970.94 examples/s]20711 examples [00:22, 988.18 examples/s]20818 examples [00:22, 1010.98 examples/s]20920 examples [00:22, 982.02 examples/s] 21022 examples [00:22, 990.45 examples/s]21122 examples [00:22, 971.24 examples/s]21226 examples [00:22, 989.31 examples/s]21330 examples [00:23, 1002.26 examples/s]21431 examples [00:23, 999.72 examples/s] 21532 examples [00:23, 964.73 examples/s]21630 examples [00:23, 966.15 examples/s]21727 examples [00:23, 963.43 examples/s]21826 examples [00:23, 970.08 examples/s]21929 examples [00:23, 985.57 examples/s]22030 examples [00:23, 992.70 examples/s]22130 examples [00:23, 990.72 examples/s]22236 examples [00:23, 1008.92 examples/s]22338 examples [00:24, 991.96 examples/s] 22438 examples [00:24, 957.26 examples/s]22535 examples [00:24, 950.33 examples/s]22631 examples [00:24, 932.99 examples/s]22734 examples [00:24, 959.90 examples/s]22840 examples [00:24, 986.65 examples/s]22940 examples [00:24, 983.63 examples/s]23039 examples [00:24, 983.49 examples/s]23138 examples [00:24, 980.11 examples/s]23245 examples [00:24, 1004.46 examples/s]23346 examples [00:25, 991.42 examples/s] 23446 examples [00:25, 978.38 examples/s]23546 examples [00:25, 983.56 examples/s]23651 examples [00:25, 1000.01 examples/s]23753 examples [00:25, 1004.91 examples/s]23854 examples [00:25, 992.56 examples/s] 23954 examples [00:25, 987.50 examples/s]24053 examples [00:25, 980.51 examples/s]24153 examples [00:25, 985.55 examples/s]24252 examples [00:25, 984.08 examples/s]24359 examples [00:26, 1006.74 examples/s]24460 examples [00:26, 991.14 examples/s] 24560 examples [00:26, 962.95 examples/s]24657 examples [00:26, 945.44 examples/s]24752 examples [00:26, 935.07 examples/s]24858 examples [00:26, 968.69 examples/s]24956 examples [00:26, 971.91 examples/s]25054 examples [00:26, 968.33 examples/s]25155 examples [00:26, 978.69 examples/s]25254 examples [00:27, 971.82 examples/s]25354 examples [00:27, 979.46 examples/s]25457 examples [00:27, 993.62 examples/s]25557 examples [00:27, 965.90 examples/s]25658 examples [00:27, 977.42 examples/s]25762 examples [00:27, 995.08 examples/s]25870 examples [00:27, 1018.43 examples/s]25973 examples [00:27, 1017.41 examples/s]26075 examples [00:27, 991.76 examples/s] 26179 examples [00:27, 1003.17 examples/s]26280 examples [00:28, 1003.68 examples/s]26382 examples [00:28, 1005.56 examples/s]26483 examples [00:28, 997.64 examples/s] 26583 examples [00:28, 972.74 examples/s]26681 examples [00:28, 969.84 examples/s]26783 examples [00:28, 983.65 examples/s]26882 examples [00:28, 963.94 examples/s]26981 examples [00:28, 969.71 examples/s]27079 examples [00:28, 937.10 examples/s]27177 examples [00:28, 947.12 examples/s]27278 examples [00:29, 963.63 examples/s]27379 examples [00:29, 974.76 examples/s]27477 examples [00:29, 961.51 examples/s]27574 examples [00:29, 963.77 examples/s]27676 examples [00:29, 979.69 examples/s]27775 examples [00:29, 979.95 examples/s]27879 examples [00:29, 995.36 examples/s]27979 examples [00:29, 964.41 examples/s]28076 examples [00:29, 952.59 examples/s]28177 examples [00:30, 969.04 examples/s]28276 examples [00:30, 975.15 examples/s]28374 examples [00:30, 944.33 examples/s]28469 examples [00:30, 876.80 examples/s]28559 examples [00:30, 882.61 examples/s]28660 examples [00:30, 916.35 examples/s]28759 examples [00:30, 935.63 examples/s]28854 examples [00:30, 931.03 examples/s]28948 examples [00:30, 915.86 examples/s]29044 examples [00:30, 925.89 examples/s]29145 examples [00:31, 948.01 examples/s]29249 examples [00:31, 972.45 examples/s]29353 examples [00:31, 989.54 examples/s]29453 examples [00:31, 990.19 examples/s]29553 examples [00:31, 982.56 examples/s]29658 examples [00:31, 1001.19 examples/s]29760 examples [00:31, 1006.55 examples/s]29861 examples [00:31, 971.89 examples/s] 29963 examples [00:31, 983.42 examples/s]30062 examples [00:32, 921.24 examples/s]30163 examples [00:32, 945.77 examples/s]30264 examples [00:32, 963.72 examples/s]30362 examples [00:32, 965.00 examples/s]30464 examples [00:32, 978.86 examples/s]30563 examples [00:32, 968.35 examples/s]30664 examples [00:32, 978.05 examples/s]30768 examples [00:32, 995.42 examples/s]30868 examples [00:32, 995.45 examples/s]30968 examples [00:32, 946.98 examples/s]31071 examples [00:33, 970.17 examples/s]31178 examples [00:33, 995.68 examples/s]31282 examples [00:33, 1008.25 examples/s]31387 examples [00:33, 1019.71 examples/s]31490 examples [00:33, 981.80 examples/s] 31597 examples [00:33, 1004.07 examples/s]31706 examples [00:33, 1026.11 examples/s]31816 examples [00:33, 1044.32 examples/s]31921 examples [00:33, 1031.04 examples/s]32025 examples [00:33, 1007.44 examples/s]32129 examples [00:34, 1015.01 examples/s]32232 examples [00:34, 1018.62 examples/s]32335 examples [00:34, 1006.86 examples/s]32437 examples [00:34, 1009.82 examples/s]32539 examples [00:34, 997.24 examples/s] 32645 examples [00:34, 1014.44 examples/s]32747 examples [00:34, 1011.14 examples/s]32849 examples [00:34, 973.35 examples/s] 32949 examples [00:34, 979.51 examples/s]33048 examples [00:35, 941.01 examples/s]33143 examples [00:35, 928.28 examples/s]33241 examples [00:35, 940.83 examples/s]33341 examples [00:35, 956.45 examples/s]33439 examples [00:35, 961.00 examples/s]33536 examples [00:35, 953.67 examples/s]33634 examples [00:35, 959.82 examples/s]33736 examples [00:35, 976.30 examples/s]33843 examples [00:35, 1002.14 examples/s]33944 examples [00:35, 1000.62 examples/s]34045 examples [00:36, 986.67 examples/s] 34148 examples [00:36, 998.52 examples/s]34257 examples [00:36, 1024.13 examples/s]34360 examples [00:36, 1014.13 examples/s]34462 examples [00:36, 1008.98 examples/s]34564 examples [00:36, 1008.49 examples/s]34669 examples [00:36, 1018.25 examples/s]34771 examples [00:36, 1008.89 examples/s]34872 examples [00:36, 999.03 examples/s] 34972 examples [00:36, 994.03 examples/s]35072 examples [00:37, 991.21 examples/s]35172 examples [00:37, 991.32 examples/s]35272 examples [00:37, 986.19 examples/s]35371 examples [00:37, 950.91 examples/s]35468 examples [00:37, 954.94 examples/s]35572 examples [00:37, 978.72 examples/s]35676 examples [00:37, 993.39 examples/s]35776 examples [00:37, 991.32 examples/s]35876 examples [00:37, 980.66 examples/s]35975 examples [00:37, 969.10 examples/s]36073 examples [00:38, 966.77 examples/s]36178 examples [00:38, 990.15 examples/s]36281 examples [00:38, 998.09 examples/s]36381 examples [00:38, 997.47 examples/s]36481 examples [00:38, 984.43 examples/s]36580 examples [00:38, 968.80 examples/s]36678 examples [00:38, 959.81 examples/s]36775 examples [00:38, 955.70 examples/s]36873 examples [00:38, 962.18 examples/s]36970 examples [00:39, 947.19 examples/s]37071 examples [00:39, 963.54 examples/s]37168 examples [00:39, 962.98 examples/s]37265 examples [00:39, 956.47 examples/s]37361 examples [00:39, 953.82 examples/s]37457 examples [00:39, 942.10 examples/s]37555 examples [00:39, 951.86 examples/s]37651 examples [00:39, 947.15 examples/s]37752 examples [00:39, 963.28 examples/s]37855 examples [00:39, 981.05 examples/s]37954 examples [00:40, 972.45 examples/s]38052 examples [00:40, 932.97 examples/s]38146 examples [00:40, 929.17 examples/s]38240 examples [00:40, 895.45 examples/s]38335 examples [00:40, 910.97 examples/s]38436 examples [00:40, 938.23 examples/s]38539 examples [00:40, 962.42 examples/s]38641 examples [00:40, 978.14 examples/s]38740 examples [00:40, 970.26 examples/s]38842 examples [00:40, 982.82 examples/s]38941 examples [00:41, 979.46 examples/s]39044 examples [00:41, 992.60 examples/s]39147 examples [00:41, 1001.49 examples/s]39248 examples [00:41, 1003.72 examples/s]39349 examples [00:41, 999.32 examples/s] 39450 examples [00:41, 1001.30 examples/s]39553 examples [00:41, 1007.34 examples/s]39654 examples [00:41, 995.92 examples/s] 39754 examples [00:41, 978.28 examples/s]39854 examples [00:41, 981.97 examples/s]39953 examples [00:42, 955.35 examples/s]40049 examples [00:42, 893.24 examples/s]40153 examples [00:42, 931.79 examples/s]40256 examples [00:42, 956.77 examples/s]40353 examples [00:42, 953.89 examples/s]40462 examples [00:42, 989.24 examples/s]40563 examples [00:42, 995.04 examples/s]40664 examples [00:42, 994.29 examples/s]40769 examples [00:42, 1008.83 examples/s]40871 examples [00:43, 1005.53 examples/s]40975 examples [00:43, 1014.08 examples/s]41080 examples [00:43, 1023.12 examples/s]41183 examples [00:43, 1023.06 examples/s]41286 examples [00:43, 1001.44 examples/s]41387 examples [00:43, 970.24 examples/s] 41489 examples [00:43, 984.36 examples/s]41591 examples [00:43, 992.83 examples/s]41691 examples [00:43, 988.03 examples/s]41790 examples [00:43, 980.01 examples/s]41889 examples [00:44, 963.76 examples/s]41989 examples [00:44, 973.16 examples/s]42087 examples [00:44, 966.13 examples/s]42184 examples [00:44, 967.15 examples/s]42281 examples [00:44, 956.47 examples/s]42381 examples [00:44, 967.68 examples/s]42478 examples [00:44, 968.24 examples/s]42575 examples [00:44, 934.34 examples/s]42671 examples [00:44, 940.25 examples/s]42766 examples [00:44, 936.70 examples/s]42860 examples [00:45, 931.41 examples/s]42954 examples [00:45, 907.14 examples/s]43047 examples [00:45, 911.39 examples/s]43143 examples [00:45, 922.93 examples/s]43240 examples [00:45, 933.75 examples/s]43334 examples [00:45, 935.26 examples/s]43437 examples [00:45, 960.52 examples/s]43538 examples [00:45, 971.85 examples/s]43638 examples [00:45, 977.78 examples/s]43739 examples [00:45, 985.17 examples/s]43838 examples [00:46, 969.09 examples/s]43939 examples [00:46, 978.97 examples/s]44038 examples [00:46, 979.90 examples/s]44137 examples [00:46, 974.49 examples/s]44235 examples [00:46, 976.13 examples/s]44333 examples [00:46, 952.76 examples/s]44441 examples [00:46, 987.36 examples/s]44541 examples [00:46, 962.06 examples/s]44643 examples [00:46, 978.69 examples/s]44742 examples [00:47, 966.32 examples/s]44843 examples [00:47, 976.40 examples/s]44946 examples [00:47, 991.26 examples/s]45046 examples [00:47, 966.26 examples/s]45143 examples [00:47, 934.37 examples/s]45243 examples [00:47, 951.07 examples/s]45342 examples [00:47, 960.53 examples/s]45444 examples [00:47, 976.16 examples/s]45544 examples [00:47, 982.19 examples/s]45646 examples [00:47, 991.64 examples/s]45748 examples [00:48, 998.98 examples/s]45849 examples [00:48, 987.74 examples/s]45948 examples [00:48, 971.40 examples/s]46046 examples [00:48, 963.45 examples/s]46143 examples [00:48, 952.03 examples/s]46239 examples [00:48, 954.01 examples/s]46338 examples [00:48, 961.91 examples/s]46436 examples [00:48, 964.56 examples/s]46534 examples [00:48, 967.33 examples/s]46631 examples [00:48, 957.97 examples/s]46727 examples [00:49, 955.61 examples/s]46823 examples [00:49, 952.30 examples/s]46919 examples [00:49, 954.54 examples/s]47020 examples [00:49, 968.30 examples/s]47117 examples [00:49, 966.23 examples/s]47222 examples [00:49, 988.97 examples/s]47324 examples [00:49, 997.02 examples/s]47426 examples [00:49, 1001.95 examples/s]47527 examples [00:49, 998.26 examples/s] 47627 examples [00:49, 992.98 examples/s]47727 examples [00:50, 962.69 examples/s]47824 examples [00:50, 942.49 examples/s]47919 examples [00:50, 939.95 examples/s]48018 examples [00:50, 954.13 examples/s]48116 examples [00:50, 959.36 examples/s]48213 examples [00:50, 947.44 examples/s]48311 examples [00:50, 954.99 examples/s]48410 examples [00:50, 962.55 examples/s]48507 examples [00:50, 962.52 examples/s]48604 examples [00:51, 963.56 examples/s]48701 examples [00:51, 937.57 examples/s]48795 examples [00:51, 934.64 examples/s]48889 examples [00:51, 935.63 examples/s]48986 examples [00:51, 944.61 examples/s]49085 examples [00:51, 954.92 examples/s]49181 examples [00:51, 919.97 examples/s]49274 examples [00:51, 906.17 examples/s]49365 examples [00:51, 904.02 examples/s]49462 examples [00:51, 922.33 examples/s]49562 examples [00:52, 943.41 examples/s]49658 examples [00:52, 946.18 examples/s]49758 examples [00:52, 959.53 examples/s]49857 examples [00:52, 968.43 examples/s]49955 examples [00:52, 969.34 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 16%|█▌        | 7992/50000 [00:00<00:00, 79916.65 examples/s] 40%|████      | 20185/50000 [00:00<00:00, 89129.65 examples/s] 67%|██████▋   | 33387/50000 [00:00<00:00, 98753.99 examples/s] 93%|█████████▎| 46400/50000 [00:00<00:00, 106453.14 examples/s]                                                                0 examples [00:00, ? examples/s]75 examples [00:00, 743.96 examples/s]169 examples [00:00, 793.42 examples/s]272 examples [00:00, 850.95 examples/s]372 examples [00:00, 890.39 examples/s]474 examples [00:00, 925.38 examples/s]574 examples [00:00, 945.45 examples/s]679 examples [00:00, 971.38 examples/s]778 examples [00:00, 976.52 examples/s]885 examples [00:00, 1001.33 examples/s]990 examples [00:01, 1012.56 examples/s]1090 examples [00:01, 1004.44 examples/s]1192 examples [00:01, 1005.85 examples/s]1292 examples [00:01, 983.00 examples/s] 1390 examples [00:01, 819.37 examples/s]1493 examples [00:01, 872.54 examples/s]1588 examples [00:01, 891.57 examples/s]1688 examples [00:01, 921.24 examples/s]1795 examples [00:01, 959.25 examples/s]1898 examples [00:01, 979.01 examples/s]2002 examples [00:02, 995.18 examples/s]2103 examples [00:02, 989.56 examples/s]2203 examples [00:02, 981.73 examples/s]2305 examples [00:02, 992.04 examples/s]2407 examples [00:02, 998.73 examples/s]2509 examples [00:02, 1004.12 examples/s]2613 examples [00:02, 1013.65 examples/s]2715 examples [00:02, 968.61 examples/s] 2815 examples [00:02, 977.04 examples/s]2914 examples [00:03, 976.18 examples/s]3016 examples [00:03, 986.24 examples/s]3115 examples [00:03, 973.81 examples/s]3213 examples [00:03, 973.39 examples/s]3317 examples [00:03, 991.15 examples/s]3422 examples [00:03, 1007.88 examples/s]3530 examples [00:03, 1026.23 examples/s]3634 examples [00:03, 1030.15 examples/s]3738 examples [00:03, 1016.12 examples/s]3840 examples [00:03, 1006.60 examples/s]3941 examples [00:04, 993.01 examples/s] 4043 examples [00:04, 998.64 examples/s]4143 examples [00:04, 997.71 examples/s]4243 examples [00:04, 982.42 examples/s]4342 examples [00:04, 979.96 examples/s]4443 examples [00:04, 988.62 examples/s]4547 examples [00:04, 1000.58 examples/s]4651 examples [00:04, 1010.25 examples/s]4753 examples [00:04, 1005.13 examples/s]4859 examples [00:04, 1019.73 examples/s]4963 examples [00:05, 1024.56 examples/s]5067 examples [00:05, 1027.00 examples/s]5171 examples [00:05, 1029.08 examples/s]5274 examples [00:05, 1013.61 examples/s]5376 examples [00:05, 1008.47 examples/s]5477 examples [00:05, 1002.50 examples/s]5579 examples [00:05, 1006.54 examples/s]5682 examples [00:05, 1013.39 examples/s]5784 examples [00:05, 989.58 examples/s] 5884 examples [00:05, 991.31 examples/s]5984 examples [00:06, 986.93 examples/s]6085 examples [00:06, 989.28 examples/s]6188 examples [00:06, 999.52 examples/s]6289 examples [00:06, 987.43 examples/s]6391 examples [00:06, 995.27 examples/s]6492 examples [00:06, 996.91 examples/s]6592 examples [00:06, 963.73 examples/s]6689 examples [00:06, 954.40 examples/s]6789 examples [00:06, 966.23 examples/s]6892 examples [00:06, 983.77 examples/s]6993 examples [00:07, 989.88 examples/s]7093 examples [00:07, 981.54 examples/s]7192 examples [00:07, 977.14 examples/s]7290 examples [00:07, 977.72 examples/s]7388 examples [00:07, 975.22 examples/s]7489 examples [00:07, 982.77 examples/s]7594 examples [00:07, 1000.30 examples/s]7695 examples [00:07, 993.40 examples/s] 7795 examples [00:07, 976.11 examples/s]7893 examples [00:08, 972.11 examples/s]7991 examples [00:08, 963.34 examples/s]8088 examples [00:08, 965.05 examples/s]8190 examples [00:08, 978.68 examples/s]8290 examples [00:08, 984.93 examples/s]8392 examples [00:08, 993.56 examples/s]8495 examples [00:08, 1003.90 examples/s]8601 examples [00:08, 1018.82 examples/s]8703 examples [00:08, 1010.91 examples/s]8805 examples [00:08, 973.60 examples/s] 8908 examples [00:09, 987.84 examples/s]9008 examples [00:09, 988.98 examples/s]9108 examples [00:09, 977.21 examples/s]9206 examples [00:09, 970.93 examples/s]9304 examples [00:09, 973.13 examples/s]9410 examples [00:09, 997.58 examples/s]9517 examples [00:09, 1015.84 examples/s]9623 examples [00:09, 1026.27 examples/s]9726 examples [00:09, 1005.21 examples/s]9830 examples [00:09, 1014.22 examples/s]9932 examples [00:10, 1014.56 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteI4F2AQ/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteI4F2AQ/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f52d5406950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f52d5406950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f52d5406950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f533b697b38>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5270099ba8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f52d5406950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f52d5406950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f52d5406950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f52bb7c80f0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f52bb7c80b8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f52680442f0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f52680442f0> 

  function with postional parmater data_info <function split_train_valid at 0x7f52680442f0> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f5268044400> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f5268044400> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f5268044400> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.5)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.1)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.2)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.1)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=dbd49548372a4bb423e20e24019a5dfe5875e7525bf901e7e93aefa4b6102c65
  Stored in directory: /tmp/pip-ephem-wheel-cache-1xhf7tkb/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<24:16:43, 9.86kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<17:13:45, 13.9kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<12:06:48, 19.8kB/s] .vector_cache/glove.6B.zip:   0%|          | 893k/862M [00:01<8:29:06, 28.2kB/s] .vector_cache/glove.6B.zip:   0%|          | 2.97M/862M [00:01<5:55:43, 40.3kB/s].vector_cache/glove.6B.zip:   1%|          | 6.66M/862M [00:01<4:08:03, 57.5kB/s].vector_cache/glove.6B.zip:   1%|          | 10.8M/862M [00:01<2:52:54, 82.1kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.9M/862M [00:01<2:00:23, 117kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 19.4M/862M [00:01<1:24:03, 167kB/s].vector_cache/glove.6B.zip:   3%|▎         | 24.7M/862M [00:01<58:32, 238kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 27.9M/862M [00:01<40:57, 339kB/s].vector_cache/glove.6B.zip:   4%|▍         | 33.1M/862M [00:02<28:34, 484kB/s].vector_cache/glove.6B.zip:   4%|▍         | 36.3M/862M [00:02<20:02, 687kB/s].vector_cache/glove.6B.zip:   5%|▍         | 41.1M/862M [00:02<14:02, 975kB/s].vector_cache/glove.6B.zip:   5%|▌         | 44.8M/862M [00:02<09:53, 1.38MB/s].vector_cache/glove.6B.zip:   6%|▌         | 49.4M/862M [00:02<07:00, 1.93MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.8M/862M [00:02<05:19, 2.54MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.9M/862M [00:04<05:37, 2.39MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.2M/862M [00:04<05:45, 2.33MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.4M/862M [00:04<04:28, 2.99MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.1M/862M [00:06<05:40, 2.35MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.2M/862M [00:06<07:13, 1.85MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.9M/862M [00:06<05:44, 2.33MB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.0M/862M [00:06<04:11, 3.17MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.2M/862M [00:08<08:11, 1.62MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.6M/862M [00:08<07:16, 1.83MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.9M/862M [00:08<05:25, 2.45MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.4M/862M [00:10<06:33, 2.02MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.6M/862M [00:10<07:24, 1.78MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.3M/862M [00:10<05:46, 2.29MB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.5M/862M [00:10<04:12, 3.13MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.6M/862M [00:12<08:44, 1.51MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.0M/862M [00:12<07:30, 1.75MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.4M/862M [00:12<05:32, 2.37MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.7M/862M [00:14<06:51, 1.91MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.9M/862M [00:14<07:28, 1.75MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.7M/862M [00:14<05:47, 2.26MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.4M/862M [00:14<04:16, 3.05MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.8M/862M [00:16<07:23, 1.76MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.2M/862M [00:16<06:30, 2.00MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.7M/862M [00:16<04:50, 2.68MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.9M/862M [00:18<06:25, 2.02MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.1M/862M [00:18<07:08, 1.81MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.9M/862M [00:18<05:38, 2.29MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.4M/862M [00:18<04:05, 3.15MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.0M/862M [00:20<13:28, 957kB/s] .vector_cache/glove.6B.zip:  10%|█         | 89.4M/862M [00:20<10:54, 1.18MB/s].vector_cache/glove.6B.zip:  11%|█         | 90.7M/862M [00:20<07:56, 1.62MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.2M/862M [00:22<08:15, 1.55MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.4M/862M [00:22<08:44, 1.47MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.0M/862M [00:22<06:44, 1.90MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.3M/862M [00:22<04:52, 2.61MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.4M/862M [00:24<09:11, 1.39MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.7M/862M [00:24<07:41, 1.66MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.8M/862M [00:24<05:44, 2.22MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:24<04:11, 3.03MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<11:39, 1.09MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<11:07, 1.14MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<08:31, 1.49MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:26<06:06, 2.07MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<13:46, 916kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<11:06, 1.13MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<08:07, 1.55MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<08:18, 1.51MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<08:48, 1.42MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<06:53, 1.82MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:30<04:59, 2.50MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<13:41, 911kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<11:04, 1.12MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<08:03, 1.54MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:32<05:46, 2.15MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<1:11:30, 173kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 119M/862M [00:34<51:31, 241kB/s]  .vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<36:19, 341kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<27:56, 441kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<22:25, 550kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<16:24, 750kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:36<11:37, 1.06MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<18:34, 660kB/s] .vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<14:25, 849kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<10:22, 1.18MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:38<07:23, 1.65MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<1:06:20, 184kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<49:17, 247kB/s]  .vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<35:10, 346kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:40<24:42, 491kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<27:20, 443kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<20:32, 590kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<14:39, 825kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:42<10:29, 1.15MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:43<07:49, 1.54MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:45<17:09:33, 11.7kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:45<12:12:26, 16.4kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:45<8:35:32, 23.3kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 142M/862M [00:45<6:01:08, 33.2kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:45<4:12:20, 47.4kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:47<3:00:08, 66.3kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:47<2:07:25, 93.7kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<1:29:22, 133kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:49<1:04:52, 183kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:49<48:08, 247kB/s]  .vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:49<34:17, 346kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:49<24:05, 491kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:51<26:13, 450kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:51<19:42, 599kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:51<14:07, 834kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:53<12:18, 953kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:53<11:19, 1.04MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<08:36, 1.36MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 162M/862M [00:53<06:10, 1.89MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:55<14:09, 824kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:55<11:15, 1.04MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:55<08:08, 1.43MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:57<08:08, 1.42MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:57<08:24, 1.38MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<06:28, 1.79MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:57<04:41, 2.46MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:59<07:19, 1.57MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<06:27, 1.78MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<04:47, 2.40MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<05:46, 1.98MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<06:49, 1.68MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<05:20, 2.14MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:01<03:55, 2.90MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<06:12, 1.83MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<05:41, 2.00MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<04:15, 2.66MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:05<05:22, 2.11MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:05<06:25, 1.76MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<05:03, 2.24MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:05<03:43, 3.03MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:07<06:11, 1.82MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:07<05:38, 1.99MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<04:13, 2.65MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:09<05:18, 2.10MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<05:01, 2.23MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<03:49, 2.91MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:11<05:01, 2.21MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<06:05, 1.82MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<04:54, 2.26MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:11<03:34, 3.09MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<11:41, 945kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<09:26, 1.17MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<06:54, 1.59MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<07:09, 1.53MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<08:05, 1.36MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:15<06:16, 1.74MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<04:46, 2.29MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<05:31, 1.97MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<05:00, 2.18MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<04:21, 2.50MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 211M/862M [01:17<03:13, 3.36MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:19<11:07, 973kB/s] .vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:19<09:18, 1.16MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<06:46, 1.59MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:19<04:55, 2.19MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:21<11:45, 915kB/s] .vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<09:59, 1.08MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<07:16, 1.48MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 220M/862M [01:21<05:17, 2.02MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:23<12:02, 888kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<09:02, 1.18MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<06:50, 1.56MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:25<06:54, 1.54MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<06:07, 1.73MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<04:32, 2.33MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<03:25, 3.10MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<06:20, 1.67MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<05:25, 1.95MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<04:41, 2.25MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<03:28, 3.03MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<05:35, 1.88MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<05:08, 2.04MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<03:53, 2.69MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:31<04:55, 2.12MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:31<04:40, 2.23MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<03:31, 2.94MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:33<04:38, 2.23MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:33<05:38, 1.83MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<04:33, 2.27MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:33<03:19, 3.10MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:35<10:52, 946kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 246M/862M [01:35<08:43, 1.18MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<06:19, 1.62MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:37<06:41, 1.53MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<05:47, 1.76MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<04:19, 2.36MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<05:16, 1.92MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<04:41, 2.16MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<03:31, 2.87MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:39<02:39, 3.80MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<09:33, 1.05MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<07:51, 1.28MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<05:44, 1.75MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:43<06:11, 1.62MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:43<06:38, 1.51MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:43<05:13, 1.91MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:43<03:47, 2.63MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:45<10:54, 911kB/s] .vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:45<08:47, 1.13MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<06:23, 1.55MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:47<06:31, 1.51MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:47<06:51, 1.44MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<05:21, 1.84MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:47<03:51, 2.54MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:49<10:48, 906kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<08:36, 1.14MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<06:16, 1.56MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<06:31, 1.49MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<06:55, 1.40MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<05:19, 1.82MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:51<03:50, 2.52MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<07:08, 1.35MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<06:08, 1.57MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<04:31, 2.13MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<05:11, 1.85MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:55<05:51, 1.64MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<04:34, 2.09MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:55<03:18, 2.88MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:56<07:22, 1.29MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:57<06:13, 1.53MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<04:33, 2.08MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<05:16, 1.79MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:59<05:52, 1.61MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<04:39, 2.02MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [01:59<03:22, 2.78MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<10:10, 922kB/s] .vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<08:11, 1.14MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<05:58, 1.57MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<06:07, 1.52MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:03<06:26, 1.45MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:03<05:02, 1.84MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:03<03:38, 2.54MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<10:14, 903kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:05<08:13, 1.12MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<06:01, 1.53MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<06:07, 1.50MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:07<06:24, 1.43MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<04:57, 1.85MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:07<03:33, 2.56MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<08:14, 1.10MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:09<06:49, 1.33MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<05:01, 1.80MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<05:24, 1.67MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:11<05:41, 1.59MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<04:27, 2.02MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:11<03:13, 2.79MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<14:33, 615kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:13<11:14, 797kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<08:04, 1.11MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<07:30, 1.18MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:15<07:13, 1.23MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:15<05:31, 1.61MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:15<03:57, 2.23MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<15:45, 560kB/s] .vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<12:02, 733kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:17<08:37, 1.02MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<07:51, 1.11MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<07:26, 1.18MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<05:36, 1.56MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<04:02, 2.15MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<06:15, 1.39MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<05:23, 1.61MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<03:58, 2.18MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<04:35, 1.87MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<04:12, 2.04MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<03:09, 2.71MB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:24<04:00, 2.13MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<03:34, 2.39MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<02:42, 3.15MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:25<02:00, 4.23MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<10:52, 780kB/s] .vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<09:41, 875kB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:27<07:13, 1.17MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:27<05:08, 1.63MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 358M/862M [02:28<10:15, 819kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<08:11, 1.03MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<05:57, 1.41MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<05:54, 1.41MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<06:06, 1.37MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<04:45, 1.75MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:31<03:25, 2.41MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<09:15, 893kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<07:27, 1.11MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:32<05:24, 1.52MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<05:30, 1.49MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<05:45, 1.42MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:35<04:30, 1.82MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:35<03:14, 2.51MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<08:54, 913kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:36<07:11, 1.13MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:36<05:13, 1.55MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<05:19, 1.51MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<05:35, 1.44MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<04:22, 1.84MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<03:09, 2.53MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<08:51, 902kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<07:07, 1.12MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:40<05:10, 1.54MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<05:17, 1.50MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<05:31, 1.43MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:42<04:19, 1.83MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<03:06, 2.52MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<08:40, 906kB/s] .vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<06:58, 1.12MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:44<05:06, 1.53MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<05:11, 1.50MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<05:26, 1.43MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<04:11, 1.85MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:46<03:03, 2.52MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<04:23, 1.76MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<03:54, 1.97MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:48<02:54, 2.64MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<03:43, 2.05MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<04:23, 1.74MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<03:26, 2.21MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:50<02:31, 3.01MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<04:13, 1.79MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<03:49, 1.98MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<02:53, 2.61MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<03:35, 2.09MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<03:16, 2.29MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:54<02:30, 2.99MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<03:18, 2.25MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<04:01, 1.84MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<03:11, 2.33MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:56<02:19, 3.18MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<05:11, 1.42MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<04:30, 1.63MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:58<03:21, 2.18MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<03:52, 1.88MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<03:33, 2.05MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:00<02:41, 2.69MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<03:24, 2.12MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<03:14, 2.23MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<02:26, 2.95MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<03:13, 2.22MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<03:50, 1.86MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<03:01, 2.37MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:04<02:11, 3.23MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<04:46, 1.49MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<04:09, 1.70MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:06<03:04, 2.29MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<03:38, 1.93MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<04:09, 1.68MB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:08<03:15, 2.15MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:08<02:22, 2.94MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<04:35, 1.51MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<03:57, 1.75MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<02:55, 2.36MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<03:34, 1.93MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<03:14, 2.12MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<02:26, 2.80MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<03:13, 2.11MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<03:45, 1.81MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<02:56, 2.31MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:14<02:08, 3.16MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<05:03, 1.33MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<04:16, 1.57MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:16<03:09, 2.12MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<03:40, 1.81MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<04:07, 1.62MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<03:13, 2.07MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:18<02:20, 2.82MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<03:52, 1.71MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<03:25, 1.93MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<02:32, 2.59MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:22<03:10, 2.06MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<03:35, 1.81MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<02:48, 2.32MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:22<02:01, 3.19MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<05:51, 1.10MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<04:43, 1.37MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<03:29, 1.85MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:24<02:32, 2.53MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<06:44, 949kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<06:12, 1.03MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<04:39, 1.37MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:26<03:20, 1.90MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<04:33, 1.39MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<03:52, 1.63MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:28<02:52, 2.19MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:30<03:23, 1.85MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:30<03:49, 1.64MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<02:58, 2.10MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:30<02:09, 2.87MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<04:05, 1.51MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<03:34, 1.73MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<02:38, 2.33MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<03:08, 1.95MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<02:55, 2.09MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<02:11, 2.78MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<02:48, 2.15MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<03:22, 1.79MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<02:42, 2.23MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:36<01:58, 3.04MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<06:19, 943kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<05:07, 1.17MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<03:43, 1.60MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<03:50, 1.54MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<03:22, 1.75MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<02:31, 2.33MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<02:59, 1.95MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<02:45, 2.11MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<02:05, 2.77MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<02:40, 2.16MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<03:12, 1.80MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<02:34, 2.24MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:44<01:52, 3.05MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:46<06:02, 943kB/s] .vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<04:53, 1.16MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<03:33, 1.60MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<03:39, 1.54MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<03:48, 1.48MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<02:57, 1.90MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:48<02:07, 2.61MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<10:15, 542kB/s] .vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<07:49, 710kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 530M/862M [03:50<05:36, 986kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<05:03, 1.09MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<04:10, 1.31MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<03:03, 1.79MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<03:16, 1.66MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<03:32, 1.53MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<02:45, 1.95MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:54<01:59, 2.68MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<05:22, 995kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<04:24, 1.21MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<03:12, 1.66MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:56<02:18, 2.29MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:58<05:38, 937kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<05:09, 1.02MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<03:54, 1.35MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:58<02:47, 1.87MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 550M/862M [04:00<06:16, 830kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<04:51, 1.07MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<03:30, 1.48MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:00<02:31, 2.04MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<08:24, 612kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<06:57, 738kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<05:06, 1.00MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:02<03:36, 1.41MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<05:37, 901kB/s] .vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<04:32, 1.12MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<03:17, 1.54MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<03:20, 1.50MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<03:29, 1.43MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<02:41, 1.85MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:06<01:56, 2.55MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:08<03:49, 1.29MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<03:08, 1.57MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<02:18, 2.12MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:08<01:42, 2.86MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:10<04:51, 1.00MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<03:58, 1.22MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<02:53, 1.67MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:12<03:01, 1.59MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<02:34, 1.86MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<01:55, 2.47MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<02:20, 2.02MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<02:11, 2.15MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<01:38, 2.85MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<02:07, 2.19MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<02:34, 1.81MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<02:01, 2.30MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<01:28, 3.11MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<02:33, 1.79MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<02:19, 1.98MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<01:44, 2.61MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:20<02:09, 2.09MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:20<02:33, 1.76MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<02:03, 2.20MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:20<01:28, 3.02MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:22<04:40, 952kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<03:46, 1.18MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<02:44, 1.61MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:24<02:49, 1.55MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<02:59, 1.46MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<02:17, 1.90MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<01:39, 2.61MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<02:49, 1.52MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<02:28, 1.74MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<01:51, 2.31MB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:28<02:10, 1.95MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:28<02:30, 1.69MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<01:59, 2.12MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:28<01:26, 2.90MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<04:28, 933kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:30<03:36, 1.15MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<02:37, 1.57MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<02:41, 1.53MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:32<02:49, 1.45MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<02:12, 1.85MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:32<01:35, 2.54MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<04:30, 893kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<03:37, 1.11MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<02:37, 1.53MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<02:39, 1.49MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:36<02:17, 1.73MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<01:41, 2.32MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<02:02, 1.91MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:38<01:50, 2.11MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<01:23, 2.79MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<01:49, 2.10MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:40<01:43, 2.22MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<01:17, 2.95MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<01:41, 2.22MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:42<01:58, 1.90MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<01:34, 2.36MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:42<01:08, 3.24MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<06:12, 593kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:44<04:46, 770kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<03:25, 1.07MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<03:07, 1.15MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:46<02:59, 1.21MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:46<02:14, 1.60MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<01:36, 2.21MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<02:33, 1.39MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<02:07, 1.66MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:48<01:35, 2.20MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<01:50, 1.89MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<02:07, 1.63MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:50<01:41, 2.05MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:50<01:12, 2.83MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<03:35, 949kB/s] .vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<02:54, 1.17MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<02:07, 1.59MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<02:10, 1.54MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<01:53, 1.77MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<01:23, 2.39MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<01:40, 1.94MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<01:56, 1.68MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:56<01:30, 2.15MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<01:06, 2.93MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<01:48, 1.77MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<01:37, 1.96MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<01:12, 2.61MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<01:30, 2.08MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<01:46, 1.76MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:00<01:23, 2.23MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:00<01:00, 3.06MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:01<02:37, 1.16MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:01<02:08, 1.43MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<01:34, 1.92MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<01:43, 1.74MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<01:32, 1.93MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<01:09, 2.55MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<01:24, 2.06MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<01:39, 1.75MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:06<01:18, 2.22MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:06<00:56, 3.06MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<02:41, 1.06MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<02:12, 1.29MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<01:36, 1.75MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<01:42, 1.64MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<01:50, 1.51MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<01:25, 1.95MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<01:00, 2.69MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<01:54, 1.42MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<01:39, 1.64MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:11<01:12, 2.21MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<01:23, 1.90MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<01:36, 1.64MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<01:15, 2.09MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:14<00:54, 2.87MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<01:43, 1.48MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<01:30, 1.69MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:15<01:06, 2.29MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<01:17, 1.93MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<01:28, 1.70MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<01:08, 2.19MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<00:50, 2.95MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<01:16, 1.92MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<01:09, 2.08MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:19<00:52, 2.77MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:21<01:05, 2.15MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:21<01:19, 1.79MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<01:03, 2.23MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:22<00:45, 3.06MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<02:27, 936kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<01:58, 1.16MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:23<01:26, 1.58MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<01:27, 1.53MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<01:31, 1.45MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<01:10, 1.88MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:25<00:50, 2.58MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:27<01:19, 1.62MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:27<01:09, 1.86MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:27<00:51, 2.47MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<01:03, 1.98MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 737M/862M [05:29<00:59, 2.11MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:29<00:43, 2.81MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:29<00:31, 3.81MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<04:25, 457kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<03:33, 567kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<02:35, 773kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:31<01:47, 1.09MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<02:42, 718kB/s] .vector_cache/glove.6B.zip:  86%|████████▋ | 746M/862M [05:33<02:06, 924kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:33<01:29, 1.28MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<01:26, 1.30MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<01:26, 1.30MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<01:06, 1.67MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:35<00:47, 2.32MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<02:02, 885kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<01:38, 1.10MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:37<01:10, 1.51MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<01:10, 1.48MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<01:13, 1.42MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<00:56, 1.82MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:39<00:40, 2.50MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<01:51, 900kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<01:28, 1.13MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:41<01:03, 1.55MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<01:04, 1.48MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<00:55, 1.72MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:43<00:40, 2.32MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<00:48, 1.90MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<00:54, 1.70MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<00:41, 2.18MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:45<00:30, 2.96MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<00:50, 1.75MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<00:45, 1.94MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:47<00:33, 2.56MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<00:40, 2.07MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<00:47, 1.75MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<00:37, 2.18MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:49<00:26, 2.98MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<01:25, 929kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<01:07, 1.17MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<00:50, 1.56MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:51<00:35, 2.16MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<01:22, 911kB/s] .vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<01:15, 999kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:56, 1.32MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:53<00:39, 1.83MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<01:27, 813kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<01:09, 1.02MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:55<00:49, 1.40MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:57<00:47, 1.41MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:57<00:48, 1.39MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<00:36, 1.79MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:57<00:25, 2.47MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:59<01:56, 538kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<01:28, 705kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<01:02, 981kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<00:54, 1.08MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<00:44, 1.31MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:31, 1.79MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<00:32, 1.66MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<00:34, 1.58MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:26, 2.04MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:03<00:17, 2.82MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<01:00, 832kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<00:46, 1.07MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:05<00:32, 1.48MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:05<00:22, 2.03MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:07<01:10, 653kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:07<00:58, 779kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:42, 1.06MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:07<00:28, 1.48MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:09<00:46, 899kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:37, 1.11MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:26, 1.53MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:11<00:25, 1.50MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:26, 1.42MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:20, 1.82MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:11<00:13, 2.51MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:37, 905kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:29, 1.12MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<00:20, 1.54MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:19, 1.50MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:16, 1.72MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:11, 2.31MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<00:12, 1.95MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<00:11, 2.11MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<00:08, 2.80MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:19<00:09, 2.17MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:19<00:11, 1.77MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:09, 2.24MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:19<00:06, 3.05MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:21<00:09, 1.73MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:08, 1.91MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:05, 2.56MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 849M/862M [06:23<00:06, 2.06MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:07, 1.75MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:05, 2.18MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:23<00:03, 2.98MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:09, 939kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:07, 1.17MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:04, 1.61MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:27<00:02, 1.52MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:27<00:02, 1.46MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:01, 1.91MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:27<00:00, 2.63MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:29<00:00, 1.30MB/s].vector_cache/glove.6B.zip: 862MB [06:29, 2.21MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 819/400000 [00:00<00:48, 8188.72it/s]  0%|          | 1623/400000 [00:00<00:48, 8141.12it/s]  1%|          | 2481/400000 [00:00<00:48, 8267.43it/s]  1%|          | 3366/400000 [00:00<00:47, 8431.45it/s]  1%|          | 4255/400000 [00:00<00:46, 8562.15it/s]  1%|▏         | 5153/400000 [00:00<00:45, 8681.53it/s]  1%|▏         | 5955/400000 [00:00<00:46, 8470.57it/s]  2%|▏         | 6836/400000 [00:00<00:45, 8568.11it/s]  2%|▏         | 7692/400000 [00:00<00:45, 8563.38it/s]  2%|▏         | 8521/400000 [00:01<00:46, 8478.93it/s]  2%|▏         | 9347/400000 [00:01<00:46, 8348.92it/s]  3%|▎         | 10167/400000 [00:01<00:47, 8270.31it/s]  3%|▎         | 10992/400000 [00:01<00:47, 8261.77it/s]  3%|▎         | 11811/400000 [00:01<00:48, 7937.97it/s]  3%|▎         | 12603/400000 [00:01<00:50, 7734.43it/s]  3%|▎         | 13387/400000 [00:01<00:49, 7763.22it/s]  4%|▎         | 14163/400000 [00:01<00:49, 7723.86it/s]  4%|▍         | 15010/400000 [00:01<00:48, 7932.30it/s]  4%|▍         | 15870/400000 [00:01<00:47, 8121.10it/s]  4%|▍         | 16698/400000 [00:02<00:46, 8165.34it/s]  4%|▍         | 17516/400000 [00:02<00:46, 8158.55it/s]  5%|▍         | 18333/400000 [00:02<00:46, 8151.39it/s]  5%|▍         | 19175/400000 [00:02<00:46, 8228.75it/s]  5%|▌         | 20017/400000 [00:02<00:45, 8283.04it/s]  5%|▌         | 20846/400000 [00:02<00:46, 8238.72it/s]  5%|▌         | 21671/400000 [00:02<00:47, 8025.26it/s]  6%|▌         | 22476/400000 [00:02<00:47, 7883.67it/s]  6%|▌         | 23281/400000 [00:02<00:47, 7932.68it/s]  6%|▌         | 24076/400000 [00:02<00:48, 7742.69it/s]  6%|▌         | 24860/400000 [00:03<00:48, 7770.30it/s]  6%|▋         | 25690/400000 [00:03<00:47, 7920.71it/s]  7%|▋         | 26487/400000 [00:03<00:47, 7933.13it/s]  7%|▋         | 27286/400000 [00:03<00:46, 7949.63it/s]  7%|▋         | 28117/400000 [00:03<00:46, 8052.35it/s]  7%|▋         | 28973/400000 [00:03<00:45, 8197.20it/s]  7%|▋         | 29801/400000 [00:03<00:45, 8220.64it/s]  8%|▊         | 30624/400000 [00:03<00:45, 8117.29it/s]  8%|▊         | 31437/400000 [00:03<00:45, 8079.13it/s]  8%|▊         | 32302/400000 [00:03<00:44, 8241.39it/s]  8%|▊         | 33197/400000 [00:04<00:43, 8441.86it/s]  9%|▊         | 34081/400000 [00:04<00:42, 8554.27it/s]  9%|▊         | 34939/400000 [00:04<00:44, 8291.48it/s]  9%|▉         | 35816/400000 [00:04<00:43, 8429.17it/s]  9%|▉         | 36705/400000 [00:04<00:42, 8561.50it/s]  9%|▉         | 37603/400000 [00:04<00:41, 8680.08it/s] 10%|▉         | 38474/400000 [00:04<00:42, 8434.26it/s] 10%|▉         | 39321/400000 [00:04<00:43, 8272.74it/s] 10%|█         | 40194/400000 [00:04<00:42, 8403.13it/s] 10%|█         | 41070/400000 [00:04<00:42, 8505.87it/s] 10%|█         | 41950/400000 [00:05<00:41, 8591.31it/s] 11%|█         | 42811/400000 [00:05<00:41, 8573.69it/s] 11%|█         | 43672/400000 [00:05<00:41, 8582.15it/s] 11%|█         | 44584/400000 [00:05<00:40, 8735.83it/s] 11%|█▏        | 45503/400000 [00:05<00:39, 8864.93it/s] 12%|█▏        | 46417/400000 [00:05<00:39, 8943.79it/s] 12%|█▏        | 47313/400000 [00:05<00:40, 8682.99it/s] 12%|█▏        | 48184/400000 [00:05<00:41, 8559.33it/s] 12%|█▏        | 49066/400000 [00:05<00:40, 8635.14it/s] 12%|█▏        | 49951/400000 [00:06<00:40, 8695.94it/s] 13%|█▎        | 50822/400000 [00:06<00:40, 8550.85it/s] 13%|█▎        | 51690/400000 [00:06<00:40, 8587.29it/s] 13%|█▎        | 52550/400000 [00:06<00:41, 8337.64it/s] 13%|█▎        | 53388/400000 [00:06<00:41, 8350.11it/s] 14%|█▎        | 54255/400000 [00:06<00:40, 8441.29it/s] 14%|█▍        | 55143/400000 [00:06<00:40, 8565.92it/s] 14%|█▍        | 56028/400000 [00:06<00:39, 8647.38it/s] 14%|█▍        | 56894/400000 [00:06<00:41, 8296.87it/s] 14%|█▍        | 57730/400000 [00:06<00:41, 8313.88it/s] 15%|█▍        | 58565/400000 [00:07<00:41, 8238.37it/s] 15%|█▍        | 59411/400000 [00:07<00:41, 8302.90it/s] 15%|█▌        | 60243/400000 [00:07<00:41, 8211.02it/s] 15%|█▌        | 61070/400000 [00:07<00:41, 8227.23it/s] 15%|█▌        | 61942/400000 [00:07<00:40, 8367.78it/s] 16%|█▌        | 62782/400000 [00:07<00:40, 8375.08it/s] 16%|█▌        | 63642/400000 [00:07<00:39, 8440.55it/s] 16%|█▌        | 64487/400000 [00:07<00:40, 8345.23it/s] 16%|█▋        | 65335/400000 [00:07<00:39, 8381.41it/s] 17%|█▋        | 66174/400000 [00:07<00:39, 8365.25it/s] 17%|█▋        | 67011/400000 [00:08<00:39, 8359.78it/s] 17%|█▋        | 67848/400000 [00:08<00:40, 8277.55it/s] 17%|█▋        | 68677/400000 [00:08<00:40, 8226.15it/s] 17%|█▋        | 69526/400000 [00:08<00:39, 8301.19it/s] 18%|█▊        | 70357/400000 [00:08<00:40, 8195.03it/s] 18%|█▊        | 71214/400000 [00:08<00:39, 8301.90it/s] 18%|█▊        | 72045/400000 [00:08<00:39, 8304.18it/s] 18%|█▊        | 72911/400000 [00:08<00:38, 8406.38it/s] 18%|█▊        | 73753/400000 [00:08<00:39, 8279.62it/s] 19%|█▊        | 74648/400000 [00:08<00:38, 8468.87it/s] 19%|█▉        | 75536/400000 [00:09<00:37, 8586.12it/s] 19%|█▉        | 76429/400000 [00:09<00:37, 8685.47it/s] 19%|█▉        | 77309/400000 [00:09<00:37, 8718.43it/s] 20%|█▉        | 78182/400000 [00:09<00:38, 8465.44it/s] 20%|█▉        | 79075/400000 [00:09<00:37, 8597.90it/s] 20%|█▉        | 79988/400000 [00:09<00:36, 8748.96it/s] 20%|██        | 80866/400000 [00:09<00:36, 8747.62it/s] 20%|██        | 81749/400000 [00:09<00:36, 8770.41it/s] 21%|██        | 82628/400000 [00:09<00:36, 8616.72it/s] 21%|██        | 83527/400000 [00:09<00:36, 8723.37it/s] 21%|██        | 84401/400000 [00:10<00:36, 8695.18it/s] 21%|██▏       | 85330/400000 [00:10<00:35, 8864.40it/s] 22%|██▏       | 86218/400000 [00:10<00:35, 8777.49it/s] 22%|██▏       | 87097/400000 [00:10<00:36, 8608.50it/s] 22%|██▏       | 88003/400000 [00:10<00:35, 8736.86it/s] 22%|██▏       | 88934/400000 [00:10<00:34, 8898.73it/s] 22%|██▏       | 89882/400000 [00:10<00:34, 9062.82it/s] 23%|██▎       | 90791/400000 [00:10<00:36, 8514.65it/s] 23%|██▎       | 91654/400000 [00:10<00:36, 8545.99it/s] 23%|██▎       | 92515/400000 [00:11<00:35, 8549.03it/s] 23%|██▎       | 93374/400000 [00:11<00:36, 8452.76it/s] 24%|██▎       | 94250/400000 [00:11<00:35, 8542.46it/s] 24%|██▍       | 95107/400000 [00:11<00:38, 8004.54it/s] 24%|██▍       | 95959/400000 [00:11<00:37, 8150.50it/s] 24%|██▍       | 96836/400000 [00:11<00:36, 8325.29it/s] 24%|██▍       | 97752/400000 [00:11<00:35, 8557.35it/s] 25%|██▍       | 98650/400000 [00:11<00:34, 8678.48it/s] 25%|██▍       | 99523/400000 [00:11<00:35, 8482.95it/s] 25%|██▌       | 100429/400000 [00:11<00:34, 8646.37it/s] 25%|██▌       | 101338/400000 [00:12<00:34, 8774.17it/s] 26%|██▌       | 102219/400000 [00:12<00:34, 8622.85it/s] 26%|██▌       | 103149/400000 [00:12<00:33, 8814.04it/s] 26%|██▌       | 104034/400000 [00:12<00:34, 8542.46it/s] 26%|██▌       | 104926/400000 [00:12<00:34, 8652.11it/s] 26%|██▋       | 105846/400000 [00:12<00:33, 8807.75it/s] 27%|██▋       | 106798/400000 [00:12<00:32, 9008.02it/s] 27%|██▋       | 107723/400000 [00:12<00:32, 9078.07it/s] 27%|██▋       | 108634/400000 [00:12<00:32, 8875.14it/s] 27%|██▋       | 109525/400000 [00:12<00:32, 8870.77it/s] 28%|██▊       | 110414/400000 [00:13<00:32, 8787.65it/s] 28%|██▊       | 111357/400000 [00:13<00:32, 8970.11it/s] 28%|██▊       | 112256/400000 [00:13<00:32, 8885.09it/s] 28%|██▊       | 113147/400000 [00:13<00:32, 8722.05it/s] 29%|██▊       | 114021/400000 [00:13<00:32, 8714.80it/s] 29%|██▊       | 114894/400000 [00:13<00:33, 8581.53it/s] 29%|██▉       | 115800/400000 [00:13<00:32, 8718.51it/s] 29%|██▉       | 116696/400000 [00:13<00:32, 8789.44it/s] 29%|██▉       | 117577/400000 [00:13<00:32, 8721.81it/s] 30%|██▉       | 118451/400000 [00:14<00:32, 8726.40it/s] 30%|██▉       | 119370/400000 [00:14<00:31, 8859.91it/s] 30%|███       | 120257/400000 [00:14<00:32, 8734.52it/s] 30%|███       | 121132/400000 [00:14<00:32, 8698.23it/s] 31%|███       | 122003/400000 [00:14<00:32, 8526.03it/s] 31%|███       | 122894/400000 [00:14<00:32, 8635.39it/s] 31%|███       | 123799/400000 [00:14<00:31, 8754.12it/s] 31%|███       | 124676/400000 [00:14<00:31, 8668.76it/s] 31%|███▏      | 125544/400000 [00:14<00:31, 8668.85it/s] 32%|███▏      | 126412/400000 [00:14<00:32, 8449.16it/s] 32%|███▏      | 127318/400000 [00:15<00:31, 8621.81it/s] 32%|███▏      | 128183/400000 [00:15<00:31, 8607.23it/s] 32%|███▏      | 129090/400000 [00:15<00:30, 8739.46it/s] 32%|███▏      | 129997/400000 [00:15<00:30, 8835.40it/s] 33%|███▎      | 130882/400000 [00:15<00:30, 8795.98it/s] 33%|███▎      | 131794/400000 [00:15<00:30, 8886.30it/s] 33%|███▎      | 132705/400000 [00:15<00:29, 8949.28it/s] 33%|███▎      | 133601/400000 [00:15<00:29, 8949.75it/s] 34%|███▎      | 134497/400000 [00:15<00:30, 8840.13it/s] 34%|███▍      | 135382/400000 [00:15<00:30, 8639.32it/s] 34%|███▍      | 136280/400000 [00:16<00:30, 8737.70it/s] 34%|███▍      | 137203/400000 [00:16<00:29, 8879.62it/s] 35%|███▍      | 138093/400000 [00:16<00:29, 8762.48it/s] 35%|███▍      | 138971/400000 [00:16<00:30, 8591.67it/s] 35%|███▍      | 139832/400000 [00:16<00:30, 8417.71it/s] 35%|███▌      | 140746/400000 [00:16<00:30, 8622.05it/s] 35%|███▌      | 141705/400000 [00:16<00:29, 8889.08it/s] 36%|███▌      | 142607/400000 [00:16<00:28, 8927.68it/s] 36%|███▌      | 143536/400000 [00:16<00:28, 9031.39it/s] 36%|███▌      | 144442/400000 [00:16<00:28, 8908.59it/s] 36%|███▋      | 145335/400000 [00:17<00:28, 8802.27it/s] 37%|███▋      | 146217/400000 [00:17<00:28, 8785.55it/s] 37%|███▋      | 147097/400000 [00:17<00:28, 8741.61it/s] 37%|███▋      | 147973/400000 [00:17<00:29, 8597.06it/s] 37%|███▋      | 148834/400000 [00:17<00:29, 8415.71it/s] 37%|███▋      | 149719/400000 [00:17<00:29, 8540.15it/s] 38%|███▊      | 150671/400000 [00:17<00:28, 8809.97it/s] 38%|███▊      | 151556/400000 [00:17<00:28, 8811.56it/s] 38%|███▊      | 152492/400000 [00:17<00:27, 8967.98it/s] 38%|███▊      | 153392/400000 [00:17<00:27, 8829.76it/s] 39%|███▊      | 154278/400000 [00:18<00:28, 8666.43it/s] 39%|███▉      | 155155/400000 [00:18<00:28, 8696.84it/s] 39%|███▉      | 156073/400000 [00:18<00:27, 8835.28it/s] 39%|███▉      | 156988/400000 [00:18<00:27, 8926.76it/s] 39%|███▉      | 157883/400000 [00:18<00:27, 8817.96it/s] 40%|███▉      | 158782/400000 [00:18<00:27, 8867.66it/s] 40%|███▉      | 159670/400000 [00:18<00:27, 8821.97it/s] 40%|████      | 160553/400000 [00:18<00:27, 8744.98it/s] 40%|████      | 161435/400000 [00:18<00:27, 8766.92it/s] 41%|████      | 162313/400000 [00:19<00:27, 8574.86it/s] 41%|████      | 163240/400000 [00:19<00:26, 8770.80it/s] 41%|████      | 164120/400000 [00:19<00:27, 8727.77it/s] 41%|████      | 164998/400000 [00:19<00:26, 8732.33it/s] 41%|████▏     | 165873/400000 [00:19<00:26, 8731.82it/s] 42%|████▏     | 166747/400000 [00:19<00:27, 8500.16it/s] 42%|████▏     | 167625/400000 [00:19<00:27, 8579.80it/s] 42%|████▏     | 168485/400000 [00:19<00:27, 8464.05it/s] 42%|████▏     | 169358/400000 [00:19<00:27, 8539.37it/s] 43%|████▎     | 170216/400000 [00:19<00:26, 8550.36it/s] 43%|████▎     | 171083/400000 [00:20<00:26, 8585.45it/s] 43%|████▎     | 171979/400000 [00:20<00:26, 8694.06it/s] 43%|████▎     | 172850/400000 [00:20<00:26, 8671.08it/s] 43%|████▎     | 173718/400000 [00:20<00:26, 8621.20it/s] 44%|████▎     | 174591/400000 [00:20<00:26, 8651.16it/s] 44%|████▍     | 175458/400000 [00:20<00:25, 8654.08it/s] 44%|████▍     | 176387/400000 [00:20<00:25, 8834.45it/s] 44%|████▍     | 177313/400000 [00:20<00:24, 8955.59it/s] 45%|████▍     | 178210/400000 [00:20<00:25, 8860.36it/s] 45%|████▍     | 179098/400000 [00:20<00:26, 8457.28it/s] 45%|████▍     | 179949/400000 [00:21<00:26, 8261.34it/s] 45%|████▌     | 180780/400000 [00:21<00:26, 8261.48it/s] 45%|████▌     | 181610/400000 [00:21<00:26, 8226.90it/s] 46%|████▌     | 182435/400000 [00:21<00:26, 8091.56it/s] 46%|████▌     | 183286/400000 [00:21<00:26, 8209.93it/s] 46%|████▌     | 184167/400000 [00:21<00:25, 8380.47it/s] 46%|████▋     | 185064/400000 [00:21<00:25, 8545.61it/s] 46%|████▋     | 185986/400000 [00:21<00:24, 8733.92it/s] 47%|████▋     | 186903/400000 [00:21<00:24, 8858.30it/s] 47%|████▋     | 187804/400000 [00:21<00:23, 8901.75it/s] 47%|████▋     | 188696/400000 [00:22<00:24, 8611.32it/s] 47%|████▋     | 189561/400000 [00:22<00:24, 8435.13it/s] 48%|████▊     | 190484/400000 [00:22<00:24, 8658.63it/s] 48%|████▊     | 191421/400000 [00:22<00:23, 8858.54it/s] 48%|████▊     | 192311/400000 [00:22<00:23, 8658.34it/s] 48%|████▊     | 193181/400000 [00:22<00:23, 8635.77it/s] 49%|████▊     | 194064/400000 [00:22<00:23, 8691.75it/s] 49%|████▊     | 194988/400000 [00:22<00:23, 8847.25it/s] 49%|████▉     | 195897/400000 [00:22<00:22, 8916.71it/s] 49%|████▉     | 196791/400000 [00:23<00:22, 8836.10it/s] 49%|████▉     | 197676/400000 [00:23<00:23, 8643.23it/s] 50%|████▉     | 198546/400000 [00:23<00:23, 8656.81it/s] 50%|████▉     | 199415/400000 [00:23<00:23, 8665.62it/s] 50%|█████     | 200283/400000 [00:23<00:23, 8371.84it/s] 50%|█████     | 201129/400000 [00:23<00:23, 8395.80it/s] 51%|█████     | 202009/400000 [00:23<00:23, 8512.80it/s] 51%|█████     | 202863/400000 [00:23<00:23, 8508.39it/s] 51%|█████     | 203773/400000 [00:23<00:22, 8675.28it/s] 51%|█████     | 204643/400000 [00:23<00:22, 8546.81it/s] 51%|█████▏    | 205500/400000 [00:24<00:24, 8038.33it/s] 52%|█████▏    | 206334/400000 [00:24<00:23, 8124.95it/s] 52%|█████▏    | 207214/400000 [00:24<00:23, 8314.30it/s] 52%|█████▏    | 208071/400000 [00:24<00:22, 8389.36it/s] 52%|█████▏    | 208979/400000 [00:24<00:22, 8582.34it/s] 52%|█████▏    | 209841/400000 [00:24<00:22, 8571.53it/s] 53%|█████▎    | 210701/400000 [00:24<00:22, 8544.90it/s] 53%|█████▎    | 211558/400000 [00:24<00:22, 8506.74it/s] 53%|█████▎    | 212483/400000 [00:24<00:21, 8715.78it/s] 53%|█████▎    | 213379/400000 [00:24<00:21, 8778.40it/s] 54%|█████▎    | 214260/400000 [00:25<00:21, 8786.06it/s] 54%|█████▍    | 215140/400000 [00:25<00:21, 8600.34it/s] 54%|█████▍    | 216026/400000 [00:25<00:21, 8675.71it/s] 54%|█████▍    | 216912/400000 [00:25<00:20, 8729.74it/s] 54%|█████▍    | 217815/400000 [00:25<00:20, 8815.14it/s] 55%|█████▍    | 218703/400000 [00:25<00:20, 8831.82it/s] 55%|█████▍    | 219587/400000 [00:25<00:21, 8444.82it/s] 55%|█████▌    | 220463/400000 [00:25<00:21, 8534.86it/s] 55%|█████▌    | 221346/400000 [00:25<00:20, 8618.51it/s] 56%|█████▌    | 222211/400000 [00:25<00:21, 8405.25it/s] 56%|█████▌    | 223055/400000 [00:26<00:21, 8292.88it/s] 56%|█████▌    | 223903/400000 [00:26<00:21, 8347.56it/s] 56%|█████▌    | 224763/400000 [00:26<00:20, 8420.21it/s] 56%|█████▋    | 225607/400000 [00:26<00:21, 8230.84it/s] 57%|█████▋    | 226432/400000 [00:26<00:21, 8116.94it/s] 57%|█████▋    | 227260/400000 [00:26<00:21, 8165.03it/s] 57%|█████▋    | 228078/400000 [00:26<00:21, 8111.72it/s] 57%|█████▋    | 228992/400000 [00:26<00:20, 8393.51it/s] 57%|█████▋    | 229903/400000 [00:26<00:19, 8596.02it/s] 58%|█████▊    | 230767/400000 [00:27<00:19, 8534.74it/s] 58%|█████▊    | 231638/400000 [00:27<00:19, 8586.03it/s] 58%|█████▊    | 232499/400000 [00:27<00:19, 8539.09it/s] 58%|█████▊    | 233355/400000 [00:27<00:19, 8490.18it/s] 59%|█████▊    | 234265/400000 [00:27<00:19, 8661.77it/s] 59%|█████▉    | 235133/400000 [00:27<00:19, 8626.79it/s] 59%|█████▉    | 235997/400000 [00:27<00:19, 8567.55it/s] 59%|█████▉    | 236855/400000 [00:27<00:19, 8559.98it/s] 59%|█████▉    | 237751/400000 [00:27<00:18, 8675.00it/s] 60%|█████▉    | 238641/400000 [00:27<00:18, 8738.88it/s] 60%|█████▉    | 239516/400000 [00:28<00:18, 8687.17it/s] 60%|██████    | 240386/400000 [00:28<00:19, 8372.66it/s] 60%|██████    | 241227/400000 [00:28<00:19, 8287.71it/s] 61%|██████    | 242124/400000 [00:28<00:18, 8479.10it/s] 61%|██████    | 243006/400000 [00:28<00:18, 8577.33it/s] 61%|██████    | 243866/400000 [00:28<00:18, 8523.50it/s] 61%|██████    | 244720/400000 [00:28<00:18, 8478.48it/s] 61%|██████▏   | 245625/400000 [00:28<00:17, 8641.85it/s] 62%|██████▏   | 246576/400000 [00:28<00:17, 8883.93it/s] 62%|██████▏   | 247489/400000 [00:28<00:17, 8956.31it/s] 62%|██████▏   | 248402/400000 [00:29<00:16, 9005.08it/s] 62%|██████▏   | 249305/400000 [00:29<00:17, 8568.85it/s] 63%|██████▎   | 250168/400000 [00:29<00:17, 8465.60it/s] 63%|██████▎   | 251019/400000 [00:29<00:17, 8448.28it/s] 63%|██████▎   | 251867/400000 [00:29<00:17, 8449.75it/s] 63%|██████▎   | 252715/400000 [00:29<00:17, 8399.20it/s] 63%|██████▎   | 253557/400000 [00:29<00:17, 8336.00it/s] 64%|██████▎   | 254461/400000 [00:29<00:17, 8534.91it/s] 64%|██████▍   | 255382/400000 [00:29<00:16, 8724.49it/s] 64%|██████▍   | 256286/400000 [00:29<00:16, 8816.60it/s] 64%|██████▍   | 257180/400000 [00:30<00:16, 8850.08it/s] 65%|██████▍   | 258067/400000 [00:30<00:16, 8649.02it/s] 65%|██████▍   | 258949/400000 [00:30<00:16, 8698.21it/s] 65%|██████▍   | 259842/400000 [00:30<00:15, 8765.66it/s] 65%|██████▌   | 260743/400000 [00:30<00:15, 8837.22it/s] 65%|██████▌   | 261628/400000 [00:30<00:15, 8775.66it/s] 66%|██████▌   | 262507/400000 [00:30<00:16, 8545.28it/s] 66%|██████▌   | 263408/400000 [00:30<00:15, 8677.82it/s] 66%|██████▌   | 264308/400000 [00:30<00:15, 8770.09it/s] 66%|██████▋   | 265214/400000 [00:30<00:15, 8853.90it/s] 67%|██████▋   | 266101/400000 [00:31<00:15, 8796.35it/s] 67%|██████▋   | 266982/400000 [00:31<00:15, 8501.91it/s] 67%|██████▋   | 267835/400000 [00:31<00:15, 8385.63it/s] 67%|██████▋   | 268676/400000 [00:31<00:15, 8381.96it/s] 67%|██████▋   | 269526/400000 [00:31<00:15, 8415.29it/s] 68%|██████▊   | 270382/400000 [00:31<00:15, 8458.01it/s] 68%|██████▊   | 271229/400000 [00:31<00:15, 8208.03it/s] 68%|██████▊   | 272099/400000 [00:31<00:15, 8349.52it/s] 68%|██████▊   | 272996/400000 [00:31<00:14, 8524.19it/s] 68%|██████▊   | 273889/400000 [00:32<00:14, 8640.69it/s] 69%|██████▊   | 274776/400000 [00:32<00:14, 8705.76it/s] 69%|██████▉   | 275649/400000 [00:32<00:14, 8510.88it/s] 69%|██████▉   | 276534/400000 [00:32<00:14, 8607.45it/s] 69%|██████▉   | 277444/400000 [00:32<00:14, 8749.30it/s] 70%|██████▉   | 278321/400000 [00:32<00:14, 8671.61it/s] 70%|██████▉   | 279190/400000 [00:32<00:13, 8634.52it/s] 70%|███████   | 280055/400000 [00:32<00:13, 8616.35it/s] 70%|███████   | 280970/400000 [00:32<00:13, 8769.52it/s] 70%|███████   | 281849/400000 [00:32<00:13, 8554.38it/s] 71%|███████   | 282707/400000 [00:33<00:13, 8485.43it/s] 71%|███████   | 283558/400000 [00:33<00:13, 8476.23it/s] 71%|███████   | 284407/400000 [00:33<00:13, 8463.91it/s] 71%|███████▏  | 285319/400000 [00:33<00:13, 8650.42it/s] 72%|███████▏  | 286223/400000 [00:33<00:12, 8761.09it/s] 72%|███████▏  | 287163/400000 [00:33<00:12, 8943.37it/s] 72%|███████▏  | 288060/400000 [00:33<00:12, 8936.60it/s] 72%|███████▏  | 288956/400000 [00:33<00:12, 8861.31it/s] 72%|███████▏  | 289884/400000 [00:33<00:12, 8982.12it/s] 73%|███████▎  | 290789/400000 [00:33<00:12, 9002.27it/s] 73%|███████▎  | 291713/400000 [00:34<00:11, 9071.45it/s] 73%|███████▎  | 292621/400000 [00:34<00:11, 9034.44it/s] 73%|███████▎  | 293525/400000 [00:34<00:12, 8636.70it/s] 74%|███████▎  | 294393/400000 [00:34<00:12, 8412.10it/s] 74%|███████▍  | 295239/400000 [00:34<00:12, 8148.00it/s] 74%|███████▍  | 296059/400000 [00:34<00:12, 8041.88it/s] 74%|███████▍  | 296867/400000 [00:34<00:12, 8015.81it/s] 74%|███████▍  | 297721/400000 [00:34<00:12, 8164.71it/s] 75%|███████▍  | 298633/400000 [00:34<00:12, 8427.86it/s] 75%|███████▍  | 299545/400000 [00:34<00:11, 8621.90it/s] 75%|███████▌  | 300477/400000 [00:35<00:11, 8819.72it/s] 75%|███████▌  | 301368/400000 [00:35<00:11, 8844.94it/s] 76%|███████▌  | 302256/400000 [00:35<00:11, 8663.41it/s] 76%|███████▌  | 303126/400000 [00:35<00:11, 8613.44it/s] 76%|███████▌  | 304008/400000 [00:35<00:11, 8674.35it/s] 76%|███████▌  | 304877/400000 [00:35<00:11, 8435.20it/s] 76%|███████▋  | 305723/400000 [00:35<00:11, 8037.08it/s] 77%|███████▋  | 306585/400000 [00:35<00:11, 8202.42it/s] 77%|███████▋  | 307452/400000 [00:35<00:11, 8336.68it/s] 77%|███████▋  | 308330/400000 [00:36<00:10, 8462.87it/s] 77%|███████▋  | 309180/400000 [00:36<00:10, 8397.75it/s] 78%|███████▊  | 310023/400000 [00:36<00:10, 8237.06it/s] 78%|███████▊  | 310850/400000 [00:36<00:11, 8097.56it/s] 78%|███████▊  | 311752/400000 [00:36<00:10, 8353.32it/s] 78%|███████▊  | 312679/400000 [00:36<00:10, 8607.66it/s] 78%|███████▊  | 313567/400000 [00:36<00:09, 8687.48it/s] 79%|███████▊  | 314455/400000 [00:36<00:09, 8742.89it/s] 79%|███████▉  | 315350/400000 [00:36<00:09, 8803.69it/s] 79%|███████▉  | 316233/400000 [00:36<00:09, 8810.26it/s] 79%|███████▉  | 317135/400000 [00:37<00:09, 8870.05it/s] 80%|███████▉  | 318023/400000 [00:37<00:09, 8857.80it/s] 80%|███████▉  | 318939/400000 [00:37<00:09, 8945.51it/s] 80%|███████▉  | 319835/400000 [00:37<00:09, 8813.40it/s] 80%|████████  | 320718/400000 [00:37<00:09, 8787.26it/s] 80%|████████  | 321607/400000 [00:37<00:08, 8815.96it/s] 81%|████████  | 322490/400000 [00:37<00:08, 8788.47it/s] 81%|████████  | 323370/400000 [00:37<00:08, 8675.43it/s] 81%|████████  | 324239/400000 [00:37<00:08, 8609.01it/s] 81%|████████▏ | 325101/400000 [00:37<00:08, 8496.46it/s] 81%|████████▏ | 325952/400000 [00:38<00:08, 8489.02it/s] 82%|████████▏ | 326817/400000 [00:38<00:08, 8535.59it/s] 82%|████████▏ | 327714/400000 [00:38<00:08, 8660.18it/s] 82%|████████▏ | 328581/400000 [00:38<00:08, 8560.21it/s] 82%|████████▏ | 329479/400000 [00:38<00:08, 8680.82it/s] 83%|████████▎ | 330349/400000 [00:38<00:08, 8681.09it/s] 83%|████████▎ | 331226/400000 [00:38<00:07, 8706.86it/s] 83%|████████▎ | 332098/400000 [00:38<00:07, 8536.71it/s] 83%|████████▎ | 332953/400000 [00:38<00:07, 8518.00it/s] 83%|████████▎ | 333866/400000 [00:38<00:07, 8690.82it/s] 84%|████████▎ | 334769/400000 [00:39<00:07, 8789.73it/s] 84%|████████▍ | 335650/400000 [00:39<00:07, 8663.34it/s] 84%|████████▍ | 336518/400000 [00:39<00:07, 8550.85it/s] 84%|████████▍ | 337375/400000 [00:39<00:07, 8509.76it/s] 85%|████████▍ | 338239/400000 [00:39<00:07, 8548.13it/s] 85%|████████▍ | 339095/400000 [00:39<00:07, 8232.94it/s] 85%|████████▍ | 339922/400000 [00:39<00:07, 8130.72it/s] 85%|████████▌ | 340744/400000 [00:39<00:07, 8154.71it/s] 85%|████████▌ | 341591/400000 [00:39<00:07, 8245.32it/s] 86%|████████▌ | 342425/400000 [00:40<00:06, 8271.81it/s] 86%|████████▌ | 343271/400000 [00:40<00:06, 8326.75it/s] 86%|████████▌ | 344145/400000 [00:40<00:06, 8445.85it/s] 86%|████████▌ | 344991/400000 [00:40<00:06, 8435.45it/s] 86%|████████▋ | 345836/400000 [00:40<00:06, 8421.66it/s] 87%|████████▋ | 346750/400000 [00:40<00:06, 8624.10it/s] 87%|████████▋ | 347653/400000 [00:40<00:05, 8740.49it/s] 87%|████████▋ | 348598/400000 [00:40<00:05, 8939.73it/s] 87%|████████▋ | 349495/400000 [00:40<00:05, 8856.30it/s] 88%|████████▊ | 350383/400000 [00:40<00:05, 8855.70it/s] 88%|████████▊ | 351281/400000 [00:41<00:05, 8889.53it/s] 88%|████████▊ | 352171/400000 [00:41<00:05, 8655.34it/s] 88%|████████▊ | 353039/400000 [00:41<00:05, 8622.04it/s] 88%|████████▊ | 353903/400000 [00:41<00:05, 8244.44it/s] 89%|████████▊ | 354804/400000 [00:41<00:05, 8458.21it/s] 89%|████████▉ | 355731/400000 [00:41<00:05, 8686.00it/s] 89%|████████▉ | 356622/400000 [00:41<00:04, 8751.61it/s] 89%|████████▉ | 357517/400000 [00:41<00:04, 8808.17it/s] 90%|████████▉ | 358401/400000 [00:41<00:04, 8636.72it/s] 90%|████████▉ | 359271/400000 [00:41<00:04, 8652.84it/s] 90%|█████████ | 360139/400000 [00:42<00:04, 8638.72it/s] 90%|█████████ | 361034/400000 [00:42<00:04, 8728.59it/s] 90%|█████████ | 361933/400000 [00:42<00:04, 8804.73it/s] 91%|█████████ | 362815/400000 [00:42<00:04, 8499.55it/s] 91%|█████████ | 363689/400000 [00:42<00:04, 8567.70it/s] 91%|█████████ | 364548/400000 [00:42<00:04, 8463.07it/s] 91%|█████████▏| 365455/400000 [00:42<00:04, 8634.43it/s] 92%|█████████▏| 366367/400000 [00:42<00:03, 8774.19it/s] 92%|█████████▏| 367247/400000 [00:42<00:03, 8697.28it/s] 92%|█████████▏| 368119/400000 [00:42<00:03, 8397.69it/s] 92%|█████████▏| 368963/400000 [00:43<00:03, 8212.90it/s] 92%|█████████▏| 369851/400000 [00:43<00:03, 8400.68it/s] 93%|█████████▎| 370704/400000 [00:43<00:03, 8438.54it/s] 93%|█████████▎| 371551/400000 [00:43<00:03, 8381.65it/s] 93%|█████████▎| 372431/400000 [00:43<00:03, 8501.53it/s] 93%|█████████▎| 373283/400000 [00:43<00:03, 8414.70it/s] 94%|█████████▎| 374189/400000 [00:43<00:03, 8596.53it/s] 94%|█████████▍| 375092/400000 [00:43<00:02, 8721.54it/s] 94%|█████████▍| 375966/400000 [00:43<00:02, 8672.63it/s] 94%|█████████▍| 376841/400000 [00:44<00:02, 8695.23it/s] 94%|█████████▍| 377712/400000 [00:44<00:02, 8675.85it/s] 95%|█████████▍| 378625/400000 [00:44<00:02, 8805.41it/s] 95%|█████████▍| 379542/400000 [00:44<00:02, 8910.13it/s] 95%|█████████▌| 380434/400000 [00:44<00:02, 8837.55it/s] 95%|█████████▌| 381319/400000 [00:44<00:02, 8799.96it/s] 96%|█████████▌| 382200/400000 [00:44<00:02, 8416.62it/s] 96%|█████████▌| 383046/400000 [00:44<00:02, 8416.38it/s] 96%|█████████▌| 383891/400000 [00:44<00:01, 8296.98it/s] 96%|█████████▌| 384724/400000 [00:44<00:01, 8249.97it/s] 96%|█████████▋| 385551/400000 [00:45<00:01, 8195.77it/s] 97%|█████████▋| 386372/400000 [00:45<00:01, 8017.52it/s] 97%|█████████▋| 387187/400000 [00:45<00:01, 8055.49it/s] 97%|█████████▋| 388037/400000 [00:45<00:01, 8183.32it/s] 97%|█████████▋| 388895/400000 [00:45<00:01, 8296.10it/s] 97%|█████████▋| 389796/400000 [00:45<00:01, 8496.90it/s] 98%|█████████▊| 390655/400000 [00:45<00:01, 8523.43it/s] 98%|█████████▊| 391583/400000 [00:45<00:00, 8735.73it/s] 98%|█████████▊| 392530/400000 [00:45<00:00, 8943.00it/s] 98%|█████████▊| 393428/400000 [00:45<00:00, 8908.87it/s] 99%|█████████▊| 394321/400000 [00:46<00:00, 8668.47it/s] 99%|█████████▉| 395253/400000 [00:46<00:00, 8852.90it/s] 99%|█████████▉| 396143/400000 [00:46<00:00, 8864.63it/s] 99%|█████████▉| 397032/400000 [00:46<00:00, 8750.02it/s] 99%|█████████▉| 397918/400000 [00:46<00:00, 8780.12it/s]100%|█████████▉| 398798/400000 [00:46<00:00, 8632.90it/s]100%|█████████▉| 399728/400000 [00:46<00:00, 8822.36it/s]100%|█████████▉| 399999/400000 [00:46<00:00, 8566.33it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f534163d0f0>, <torchtext.data.dataset.TabularDataset object at 0x7f534163d240>, <torchtext.vocab.Vocab object at 0x7f534163d160>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  9.42 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  9.42 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  9.42 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  9.99 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  9.99 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  3.75 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  3.75 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.28 file/s]2020-06-07 18:20:10.661791: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-07 18:20:10.666575: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-06-07 18:20:10.666751: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55fd60eb71f0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-07 18:20:10.666767: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 38%|███▊      | 3760128/9912422 [00:00<00:00, 37582767.75it/s]9920512it [00:00, 36811910.11it/s]                             
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 218507.72it/s]           
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 151798.48it/s]1654784it [00:00, 10909893.80it/s]                         
0it [00:00, ?it/s]8192it [00:00, 136161.22it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
