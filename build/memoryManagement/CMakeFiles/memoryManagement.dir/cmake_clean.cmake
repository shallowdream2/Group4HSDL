file(REMOVE_RECURSE
  "libmemoryManagement.a"
  "libmemoryManagement.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/memoryManagement.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
