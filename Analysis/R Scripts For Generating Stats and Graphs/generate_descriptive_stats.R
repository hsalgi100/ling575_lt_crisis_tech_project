# SCRIPT ARGUMENTS:
# 1. MT_EVAL_PATH:character - This is a path to the relational .jsonl files that have the ID, source, target, and a 
# variety of eval metrics. It should be down to the location level and should only contain directories with depth=1 (AKA
# it has directories within it that only have files.) 

MT_EVAL_PATH = "../../machine_translations/kc"

# Script Description: 
# This file obtains descriptive statistics for the machine translations.
# Set the working directory to "./Analysis an Visuals" and run this script
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################

# Load All necessary Resources in the correct order:
source("RESOURCES/LOAD_ALL_RESOURCES.R")

# STEP 1: load all the metrics into a mega df
evals <- generate_evals_df(MT_EVAL_PATH) # helper function

# STEP 2: Generate descriptive statistics
langs <- unique(evals$target_language)
technologies <- unique(evals$technology)

for(tech in technologies){
  for(lang in langs){
    result <- capture.output(
      evals %>% 
        filter(target_language == lang, technology == tech) %>% 
        select(c(metric,score)) %>% 
        pivot_wider(names_from = metric, values_from = score) %>% 
        stat.desc()
    )
    
    if(!dir.exists(str_c("../Statistics/",tech))){
      print("YOYOYOYOYOOYOY")
      dir.create(str_c("../Statistics/",tech,"/"),showWarnings = F)
    }
    
    write(
      c(str_c("# ",tech," Performance: English into ", lang),
      result),
      file = str_c("../Statistics/",tech,"/",tech," - [en] into [",lang,"].md")
    )
    
  }
}
