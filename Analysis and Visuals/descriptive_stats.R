# SCRIPT ARGUMENTS:
# 1. mt_data:character - This is a string path to directory with all the machine-generated translations
# 2. mt_evals:character - This is a path to the relational .jsonl file that has the ID, source, target, and a variety of eval metrics
#
# Script Description: This file obtains descriptive statistics for the machine translations.

# Load All necessary Resources:
source("0-LOAD_ALL.R")

