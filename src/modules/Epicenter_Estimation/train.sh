# declare test_path
test_path="./data/tests/"

for file in "$test_path"*; do
    if [[ $(basename "$file") == *ba* ]]; then
        echo Running training for backazimuth --test_file="$test_path$(basename "$file")"
        pdm train-backazimuth --test_file="$test_path$(basename "$file")" --extract_features
    elif [[ $(basename "$file") == *dist* ]]; then
        echo Running training for distance --test_file="$test_path$(basename "$file")"
        pdm train-distance --test_file="$test_path$(basename "$file")" --extract_features
    fi
done
