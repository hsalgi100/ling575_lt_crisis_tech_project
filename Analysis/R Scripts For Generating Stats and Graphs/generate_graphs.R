# SCRIPT ARGUMENTS:
# 1. MT_EVAL_PATH:character - This is a path to the relational .jsonl files that have the ID, source, target, and a 
# variety of eval metrics. It should be down to the location level and should only contain directories with depth=1 (AKA
# it has directories within it that only have files.) 

MT_EVAL_PATH = "../../machine_translations/kc"

# Script Description: 
# This file creates graphs for descriptive statistics for the machine translations.
# Set the working directory to "./Analysis/R Scripts For Generating Stats and Graphs" and run this script
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################

# Load All necessary Resources in the correct order:
source("RESOURCES/LOAD_ALL_RESOURCES.R")

# STEP 1: load all the metrics into a mega df
# evals <- generate_evals_df(MT_EVAL_PATH) # helper function


# STEP 2: Generate descriptive statistics graphs
langs <- unique(evals$target_language)
technologies <- unique(evals$technology)
metrics <- unique(evals$metric)


for(m in metrics){
  print(m)
  
  plot <- evals %>% 
    filter(metric == m) %>% 
    # mutate(score = case_when(
    #   technology == "Gemma4" ~ score / 100,
    #   technology == "Qwen35" ~ score / 100,
    #   TRUE ~ score
    # )) %>% 
    group_by(technology,target_language,metric) %>% 
    summarize(score = mean(score)) %>% 
      ggplot(mapping = aes(x = target_language, y = score,fill = technology)) +
        geom_col(position = "dodge") +
        labs(title = str_c(m," Scores")) +
        xlab("Target Language") +
        ylab("")
 
 ggsave(str_c("../Graphs/Bar Plots/", m," bar plot.png"), plot,create.dir = TRUE)
 
}