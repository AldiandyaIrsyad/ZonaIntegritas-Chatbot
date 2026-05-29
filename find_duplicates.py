import os

def find_duplicate_lines(directory):
    duplicate_issues = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if not file.endswith('.py'):
                continue
                
            filepath = os.path.join(root, file)
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            for i in range(1, len(lines)):
                current_line = lines[i].strip()
                prev_line = lines[i-1].strip()
                
                # Ignore empty lines or lines with just brackets/parentheses
                if not current_line or current_line in ('}', ']', ')', '{', '[', '('):
                    continue
                    
                if current_line == prev_line:
                    duplicate_issues.append({
                        'file': filepath,
                        'line_num': i + 1,
                        'content': lines[i].rstrip()
                    })
                    
    return duplicate_issues

if __name__ == '__main__':
    print("Scanning for consecutive duplicate lines in src/...")
    issues = find_duplicate_lines('src/')
    
    if issues:
        print(f"Found {len(issues)} consecutive duplicate lines:")
        for issue in issues:
            print(f"{issue['file']}:{issue['line_num']} - {issue['content']}")
    else:
        print("No consecutive duplicate lines found!")
