# SCRIPT ARGUMENTS:
# 1. MT_EVAL_PATH:character - This is a path to the relational .jsonl files that have the ID, source, target, and a 
# variety of eval metrics. It should be down to the location level and should only contain directories with depth=1 (AKA
# it has directories within it that only have files.) 

MT_EVAL_PATH = "../machine_translations/kc"

# Script Description: 
# This file obtains descriptive statistics for the machine translations.
# Set the working directory to "./Analysis an Visuals" and run this script

# Load All necessary Resources:
source("0-LOAD_ALL.R")

# STEP 1: load all the metrics into a mega df
df <- data.frame()

for(dirname in list.dirs(MT_EVAL_PATH)){
  if(dirname == MT_EVAL_PATH){ next }
  
  for(filename in list.files(dirname)){
    
  }
}