# SCRIPT ARGUMENTS:
# 1. MT_EVAL_PATH:character - This is a path to the relational .jsonl files that have the ID, source, target, and a 
# variety of eval metrics. It should be down to the location level and should only contain directories with depth=1 (AKA
# it has directories within it that only have files.) 
# 2. GOLD_PATH: character - This is the path to the metadata about the gold data

MT_EVAL_PATH = "../../machine_translations/kc_metadata/"
GOLD_PATH = "../../data/kc_metadata/"

# Script Description: 
# This file looks at the metadata for the machine translations.
# Set the working directory to "./Analysis/R Scripts For Generating Stats and Graphs" and run this script
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################

# Load All necessary Resources in the correct order:
source("RESOURCES/LOAD_ALL_RESOURCES.R")

metadata <- data.frame(matrix(ncol = 4,nrow = 0))
for(filename in list.files(MT_EVAL_PATH)){
  file_content <- readLines(str_c(MT_EVAL_PATH,"/",filename))
  
  progress_bar <- 1
  for(line in file_content){
    print(str_c(filename," - Line #: ",progress_bar))
    JSONL_as_df <- data.frame(fromJSON(line))
    JSONL_as_df_pivoted <- JSONL_as_df %>% pivot_longer(cols = -c(id,technology), names_to="lang", values_to = "length" )
    
    metadata <- metadata %>% rbind(JSONL_as_df_pivoted)
    progress_bar <- progress_bar + 1
  }
}


# gold
for(filename in list.files(GOLD_PATH)){
  file_content <- readLines(str_c(GOLD_PATH,"/",filename))
  
  progress_bar <- 1
  for(line in file_content){
    print(str_c(filename," - Line #: ",progress_bar))
    JSONL_as_df <- data.frame(fromJSON(line))
    JSONL_as_df_pivoted <- JSONL_as_df %>% pivot_longer(cols = -c(id,technology), names_to="lang", values_to = "length" )
    
    metadata <- metadata %>% rbind(JSONL_as_df_pivoted)
    progress_bar <- progress_bar + 1
  }
}

metadata %>% group_by(technology) %>% summarize(ave_length = mean(length)) %>% write.csv("../Statistics/metadata.csv",row.names = F)


# 
technologies <- unique(metadata$technology)
languages <- unique(metadata$lang)

for(tech in technologies){
  print(tech)
  for(l in languages){
    plt <- metadata %>% 
      filter(technology == tech, lang == l) %>% 
        ggplot(aes(x = length)) + 
          geom_histogram(bins = 50) + 
          labs(title = str_c(tech,": character length distribution in language: ", l)) +
          ylab("Count")
    
    ggsave(str_c("../Graphs/Histograms/",tech,"/",tech,":length distribution of - [", l,"].png"),create.dir = T)
  }
  plt2 <- metadata %>% 
    filter(technology == tech) %>% 
    ggplot(aes(x = length)) + 
    geom_histogram(bins = 50) + 
    labs(title = str_c(tech,": character length distribution across languages")) +
    ylab("Count")
  
  ggsave(str_c("../Graphs/Histograms/",tech," output length distribution.png"),create.dir = T)
}



