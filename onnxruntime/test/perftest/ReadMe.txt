onnxruntime_perf_test [options...] <model_path> <result_file>

Options:
         -A: Disable memory arena.
         -M: Disable memory pattern.
         -P: Use parallel executor instead of sequential executor.
         -c: [parallel runs]: Specifies the (max) number of runs to invoke simultaneously. Default:1.
         -e: [cpu|cuda|dnnl|tensorrt|ngraph|openvino|nuphar|acl|armnn|vsi_npu]: Specifies the execution provider 'cpu','cuda','dnnl','tensorrt', 'ngraph', 'openvino', 'nuphar', 'acl', 'armnn' or 'vsi_npu'. Default is 'cpu'.
         -m: [test_mode]: Specifies the test mode. Value could be 'duration' or 'times'. Provide 'duration' to run the test for a fix duration, and 'times' to repeated for a certain times. Default:'duration'.
         -o: [optimization level]: Default is 1. Valid values are 0 (disable), 1 (basic), 2 (extended), 99 (all). Please see __onnxruntime_c_api.h__ (enum GraphOptimizationLevel) for the full list of all optimization levels.
         -u: [path to save optimized model]: Default is empty so no optimized model would be saved.
         -p: [profile_file]: Specifies the profile name to enable profiling and dump the profile data to the file.
         -r: [repeated_times]: Specifies the repeated times if running in 'times' test mode.Default:1000.
         -s: Show statistics result, like P75, P90.
         -t: [seconds_to_run]: Specifies the seconds to run for 'duration' mode. Default:600.
         -v: Show verbose information.
         -x: [intra_op_num_threads]: Sets the number of threads used to parallelize the execution within nodes. A value of 0 means the test will auto-select a default. Must >=0.
         -y: [inter_op_num_threads]: Sets the number of threads used to parallelize the execution of the graph (across nodes), A value of 0 means the test will auto-select a default. Must >=0.
         -h: help.

Model path and input data dependency:
    Performance test uses the same input structure as onnx_test_runner. It requrires the direcotry trees as below: 
    
    --ModelName
        --test_data_set_0
            --input0.pb
        --test_data_set_2
            --input0.pb
        --model.onnx
    The path of model.onnx needs to be provided as <model_path> argument.

How to download sample test data from VSTS drop:
   1) Download drop app from https://aiinfra.artifacts.visualstudio.com/_apis/drop/client/exe
      Unzip the downloaded file and add lib/net45 dir to your PATH
   2) Download the test data by using this command:
      drop get -a -s https://aiinfra.artifacts.visualstudio.com/DefaultCollection -n Lotus/testdata/onnx/model/16 -d C:\testdata
	  You may change C:\testdata to any directory in your disk.
   Full document: https://www.1eswiki.com/wiki/VSTS_Drop


How to run performance tests for batch of models:
   1) Download the driver by using this command:
      drop get -a -s https://aiinfra.artifacts.visualstudio.com/DefaultCollection -n Lotus/test/perfdriver/$(perfTestDriverVersion) -d C:\perfdriver
      You may change C:\perfdriver to any directory in your disk.
      Currently, the $(perfTestDriverVersion) is 6
   2) Run the PerfTestDriver.py under python environment with proper arguments.