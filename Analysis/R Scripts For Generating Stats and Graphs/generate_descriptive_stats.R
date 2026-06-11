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
metrics <- unique(evals$metric)
lang_lookup <- list() # since open source LLMs didn't use simplified/traditional Chinese split.
for(lang in langs){
  if(str_detect(lang,"zh")){
    lang_lookup[[lang]] <- "zh"
  } else {
    lang_lookup[[lang]] <- lang
  }
}
print(lang_lookup)


# Descriptive Statistics
for(tech in technologies){
  # print(tech)
  for(lang in langs){
    # print(lang)
    language <- ifelse(tech == "Gemma4" || tech == "Qwen35",lang_lookup[[lang]],lang)
    # print(language)
    
    if(language %in% unique((evals %>% filter(technology == tech))[["target_language"]])){
      result <- capture.output(
        evals %>% 
          filter(target_language == language, technology == tech) %>% 
          select(c(metric,score)) %>% 
          pivot_wider(names_from = metric, values_from = score) %>% 
          stat.desc()
      )
      
      if(!dir.exists(str_c("../Statistics/",tech))){
        # print("YOYOYOYOYOOYOY")
        dir.create(str_c("../Statistics/",tech,"/"),showWarnings = F)
      }
      
      write(
        c(str_c("# ",tech," Performance: English into ", language),
        result),
        file = str_c("../Statistics/",tech,"/",tech," - [en] into [",language,"].md")
      )
    }
  }
}


# Metric by Language CSVs
evals %>% 
  group_by(target_language,metric) %>% 
  summarize("score" = mean(score)) %>% 
  write.csv(file=str_c("../Statistics/metrics_by_language.csv"),row.names = F)


# Metric by Technology CSVs
evals %>% 
  group_by(technology,metric) %>% 
  summarize("score" = mean(score)) %>% 
  write.csv(file=str_c("../Statistics/metrics_by_technology.csv"),row.names = F)



# Normalized Overall Metric (NOM) Scores
evals %>% mutate(score = case_when(
  metric == "chrf" ~ score / 100,
  metric == "chrf.." ~ score / 100,
  TRUE ~ score
)) %>% group_by(technology,target_language) %>% 
  summarize(normalized_overall_metric_score = mean(score)) %>%
  pivot_wider(names_from = technology,values_from = normalized_overall_metric_score) %>%
  write.csv(file = str_c("../Statistics/Normalized-Overall-Metric Scores.csv"),row.names = F)







