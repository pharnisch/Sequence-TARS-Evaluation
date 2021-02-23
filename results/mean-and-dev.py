file1 = open('summary of f1 scores.txt', 'r')
file2 = open('summary of f1 scores with mean and dev.txt', 'w')
Lines = file1.readlines()
 

new_lines = []
# Strips the newline character
for line in Lines:
    #line = line.strip()
    idx = line.find("0.")
    if idx == -1: 
        new_lines.append(line + "\n")
        continue 
    numbers = line[idx:]
    numbers_spl = numbers.split("\t")
    numbers_spl = [float(i) for i in numbers_spl]
    total = sum(numbers_spl)
    mean = total/3.
    var = sum([(i - mean)**2 for i in numbers_spl])
    dev = var**(1/2)

    mean_str = f"{mean:.4f}"
    dev_str = f"{dev:.4f}"

    new_line = line[:idx] + "\t\t\t" + mean_str + "\t" + dev_str + "\n"

    new_lines.append(new_line)

file2.writelines(new_lines)