# Copyright 2024 NXP
# All rights reserved.

# Utility to get parents from a given path
#   E.g.:
#   get_parent_directory: /path/to/really/deep/dir
#        num_levels: 0 -> /path/to/really/deep/
#        num_levels: 1 -> /path/to/really/
#        num_levels: 2 -> /path/to/
function(get_parent_directory target_path num_levels out_var)
    set(current_path ${target_path})
    foreach(i RANGE ${num_levels})
        get_filename_component(current_path ${current_path} DIRECTORY)
    endforeach()
    set(${out_var} ${current_path} PARENT_SCOPE)
endfunction()
