def clean_line(line):
    """Clean a line for comparison by removing comments and extra whitespace."""
    # Remove comments
    line = line.split('#')[0].strip()
    if not line:
        return None
        
    # Split into words and rejoin with single tab
    words = [word for word in line.split() if word]
    if not words:
        return None
        
    return '\t'.join(words)

def compare_line_lists(file1, file2):
    """Compare two line list files and identify differences."""
    # Read and clean lines from both files
    lines1 = []
    lines2 = []
    
    with open(file1, 'r') as f:
        for line in f:
            cleaned = clean_line(line)
            if cleaned:
                lines1.append(cleaned)
                
    with open(file2, 'r') as f:
        for line in f:
            cleaned = clean_line(line)
            if cleaned:
                lines2.append(cleaned)
                
    # Convert to sets for comparison
    set1 = set(lines1)
    set2 = set(lines2)
    
    # Find differences
    only_in_1 = sorted(set1 - set2)
    only_in_2 = sorted(set2 - set1)
    
    # Print results
    print(f"\nComparison between {file1} and {file2}:")
    print("-" * 80)
    
    print(f"\nLines only in {file1}:")
    print("-" * 40)
    for line in only_in_1:
        print(f"- {line}")
        
    print(f"\nLines only in {file2}:")
    print("-" * 40)
    for line in only_in_2:
        print(f"+ {line}")
        
    print("\nSummary:")
    print("-" * 40)
    print(f"Total lines in {file1}: {len(lines1)}")
    print(f"Total lines in {file2}: {len(lines2)}")
    print(f"Lines only in {file1}: {len(only_in_1)}")
    print(f"Lines only in {file2}: {len(only_in_2)}")
    print(f"Lines in both files: {len(set1 & set2)}")

# Run comparison
compare_line_lists('cloudy_lines_TODDLERS_v1.dat', 'cloudy_lines_TODDLERS_v2.dat')