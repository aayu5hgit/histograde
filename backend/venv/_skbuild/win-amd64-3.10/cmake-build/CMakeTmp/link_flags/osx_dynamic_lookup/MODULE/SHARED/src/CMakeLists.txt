
      cmake_minimum_required(VERSION 3.27.2)
      project(link_flags C)

      include_directories(D:/a/HistomicsTK/HistomicsTK/_skbuild/win-amd64-3.10/cmake-build/CMakeTmp/link_flags/osx_dynamic_lookup/MODULE/SHARED/src)

      add_library(number SHARED number.c)
      add_library(counter MODULE counter.c)
    
        set_target_properties(counter PROPERTIES PREFIX "")
      
        set_target_properties(counter PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
      
      add_executable(main main.c)
    
        target_link_libraries(main  number)
      
        target_link_libraries(main  "")
      