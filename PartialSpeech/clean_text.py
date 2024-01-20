import os
import re

def clean_file(inputTXT):
    with open(inputTXT, 'r', encoding='utf-8') as file_in:
        lines = file_in.readlines()

    with open(inputTXT, 'w', encoding='utf-8') as file_out:
        for line in lines:
            clean_line = ' '.join(line.split())
            clean_line = re.sub(r'^\s*\*\s+', '* ', clean_line)
            file_out.write(clean_line + '\n') 

def remove_sym(inputTXT):
    with open(inputTXT, 'r', encoding='utf-8') as file_in:
        lines = file_in.readlines()

    with open(inputTXT, 'w', encoding='utf-8') as file_out:
        for line in lines:
            updated_line = line.lstrip('"') if line.startswith('"') else line

            parts = updated_line.split('|', 1)
            if len(parts) > 1:
                parts[1] = parts[1].replace('|', '')
            updated_line = '|'.join(parts)
            file_out.write(updated_line) 

inputTXT = 'D:\CPE124-GRP4\PartialSpeech\metadata.txt'
clean_file(inputTXT)
remove_sym(inputTXT)