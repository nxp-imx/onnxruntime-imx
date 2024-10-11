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


# Utility to filter out paths from a given flag string
#   E.g.: "-a -b -c=/path/to/c" -> "-a -b"
function(filter_paths_from_flags in_var out_var)
    set(flags_str "${${in_var}}")
    # Split each flag
    string(REGEX MATCHALL "[^ ]+" flags_list "${flags_str}")
    # Find '/', asume path, skip when merging into list.
    set(filtered_flags)
    foreach(flag IN LISTS flags_list)
        if ("${flag}" MATCHES "/")
            continue()
        endif()
        list(APPEND filtered_flags "${flag}")
    endforeach()
    # Stringify the list
    string(REPLACE ";" " " filtered_flags_str "${filtered_flags}")
    set(${out_var} "${filtered_flags_str}" PARENT_SCOPE)
endfunction()
