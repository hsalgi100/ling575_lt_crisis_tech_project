# Load all necessary packages
library(tidyr)
library(dplyr)
library(ggplot2)
library(jsonlite)
library(pastecs)
library(stringr)


# Function to load all the metrics into a mega df using the MT EVAL PATH
generate_evals_df <- function(MT_EVAL_PATH){
  evals <- data.frame(matrix(ncol = 7,nrow = 0))
  
  progress_bar = 1
  for(dirname in list.dirs(MT_EVAL_PATH)){
    if(dirname == MT_EVAL_PATH){ next }
    
    for(filename in list.files(dirname)){
      file_content <- readLines(str_c(dirname,"/",filename))
      
      for(line in file_content){
        print(str_c(filename," - Line #: ",progress_bar))
        JSONL_as_df <- data.frame(fromJSON(line))
        JSONL_as_df_pivoted <- JSONL_as_df %>% pivot_longer(cols = -c(id,scenario,technology,source_language,target_language),names_to="metric",values_to = "score" )
        
        evals <- evals %>% rbind(JSONL_as_df_pivoted)
        progress_bar <- progress_bar + 1
      }
    }
  }
  
  return(evals)
}

